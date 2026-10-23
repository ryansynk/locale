"""In-training validation: run the benchmark on a just-saved checkpoint.

After every checkpoint save, each node's local rank 0 runs
benchmark/scripts/run_benchmark.py on the sampled sra50 bundle, one call per
mutation rate, with benchmark/configs/val/sra50_sampled.yaml. The benchmark
brings its own parallelism (one embedding worker per GPU, file-based
coordination between the node shards), so nothing here touches
torch.distributed; the training ranks wait on a gloo barrier meanwhile.
Shard 0 then scores the results with scripts/score_recall.py and returns
accession-level recall at the top-10 vectors per rate for wandb.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    the results, deletes the index, and returns {"val/recall10_mut0.05": ...}.
    Other shards return None."""
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
    shutil.rmtree(work_dir / "index")
    return {f"val/recall10_mut{rate}": v for rate, v in recall.items()}
