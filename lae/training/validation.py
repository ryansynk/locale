"""In-training validation: run the benchmark on a just-saved checkpoint.

After every checkpoint save, each node's local rank 0 runs
benchmark/scripts/run_benchmark.py on the sampled sra50 bundle, one call per
mutation rate, with benchmark/configs/val/sra50_sampled.yaml. The benchmark
brings its own parallelism (one embedding worker per GPU, file-based
coordination between the node shards), so nothing here touches
torch.distributed; the training ranks wait on a gloo barrier meanwhile.
Shard 0 then scores the results with scripts/score_recall.py and returns
accession-level recall at the top-10 vectors per rate for wandb, plus the
participation ratio of the index embeddings the benchmark just built.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download

from lae.config import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
# torchrun's variables must not leak into the benchmark's spawned workers.
TORCHRUN_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def participation_ratio(
    fbin: Path, device: str | torch.device | None = None, block_rows: int = 65_536
) -> dict[str, float]:
    """Participation ratio of the centered covariance of the rows of ``fbin``
    (benchmark/src/fbin.py's format: uint32 [n, d] header, float32 rows).

    PR = (sum lam)^2 / sum lam^2 over the covariance eigenvalues: d for an
    isotropic cloud, 1 when every row lies on one line through the mean. The
    file is streamed in ``block_rows`` blocks (sequential reads, no memory
    map) into float64 Gram and sum accumulators on ``device`` (the current
    CUDA device if any), so only one d x d matrix and one block are ever
    resident. Returns, keyed for wandb, the ratio, the ratio over d (so runs
    with different embedding widths compare), and the covariance trace: for
    the index's unit-norm rows that is 1 - |mean|^2, which shrinks as the
    embeddings collapse into a cone."""
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    with open(fbin, "rb") as f:
        n, d = (int(v) for v in np.frombuffer(f.read(8), dtype=np.uint32))
        if n == 0:
            raise ValueError(f"{fbin} holds no rows")
        S = torch.zeros(d, d, dtype=torch.float64, device=device)
        mu = torch.zeros(d, dtype=torch.float64, device=device)
        buf = np.empty((block_rows, d), dtype=np.float32)
        for start in range(0, n, block_rows):
            rows = buf[: min(block_rows, n - start)]
            got = f.readinto(memoryview(rows).cast("B"))
            if got != rows.nbytes:
                raise OSError(f"{fbin}: short read at row {start}")
            xb = torch.from_numpy(rows).to(device, torch.float64)
            S += xb.T @ xb
            mu += xb.sum(0)
    mu /= n
    cov = S / n - torch.outer(mu, mu)
    lam = torch.linalg.eigvalsh(cov).clamp_min(0)
    trace = lam.sum()
    pr = trace**2 / (lam**2).sum()
    return {
        "val/participation_ratio": pr.item(),
        "val/participation_ratio_frac": (pr / d).item(),
        "val/embed_cov_trace": trace.item(),
    }


def fetch_bundle(repo: str, dest: Path) -> Path:
    """Download the benchmark bundle (contigs included); a no-op once present."""
    return Path(snapshot_download(repo, repo_type="dataset", local_dir=dest))


def run_benchmark_val(
    ckpt: Path,
    cfg: TrainConfig,
    bundle: Path,
    work_dir: Path,
    shard: int,
    num_shards: int,
) -> dict[str, float] | None:
    """Benchmark ``ckpt`` on ``bundle``. Every shard runs run_benchmark once per
    rate (the first call builds the index, the rest search it); shard 0 scores
    the results, computes the participation ratio of the index embeddings,
    deletes the index, and returns {"val/recall10_mut0.05": ...,
    "val/participation_ratio": ...}. Other shards return None."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in TORCHRUN_VARS and not k.startswith("TORCHELASTIC_")
    }
    config = Path(cfg.val_benchmark_config)
    config = config if config.is_absolute() else ROOT / config
    base = [
        sys.executable,
        str(BENCHMARK / "scripts" / "run_benchmark.py"),
        "--config",
        str(config),
        "--model.encoder.checkpoint_path",
        str(ckpt),
        "--model.encoder.pooling",
        cfg.pooling,
        "--model.encoder.project",
        str(cfg.use_projection_head).lower(),
        "--dataset_dir",
        str(bundle),
        "--index_dir",
        str(work_dir / "index"),
        "--results_dir",
        str(work_dir / "results"),
        "--shard",
        str(shard),
        "--num_shards",
        str(num_shards),
    ]
    for rate in cfg.val_rates:
        subprocess.run(base + ["--mutation_rate", str(rate)], check=True, env=env)
    if shard != 0:
        return None

    labels = yaml.safe_load(config.read_text())["model"]
    results_dir = (
        work_dir
        / "results"
        / labels["encoder_label"]
        / labels["index_label"]
        / labels["search_label"]
    )
    score = [
        sys.executable,
        str(BENCHMARK / "scripts" / "score_recall.py"),
        "--results_dir",
        str(results_dir),
        "--queries",
        str(bundle / "queries.parquet"),
    ]
    out = subprocess.run(score, check=True, env=env, capture_output=True, text=True)
    recall = json.loads(out.stdout.strip().splitlines()[-1])
    # The merged fbin exists only on shard 0 and only until the rmtree below.
    index_path = work_dir / "index" / labels["encoder_label"]
    spectrum = participation_ratio(index_path / "embeddings.fbin")
    shutil.rmtree(work_dir / "index")
    return {f"val/recall10_mut{rate}": v for rate, v in recall.items()} | spectrum
