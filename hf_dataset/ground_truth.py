"""Exact top-k ground truth for the query bundles over the *published* base.

For each mutation rate directory produced by generate_queries.py, computes the
exact inner-product top-k of query.fbin over the base vectors and writes
gt.bin beside it.

gt.bin uses the big-ann-benchmarks ground-truth layout -- two blocks, two
dtypes, which is why it carries no .fbin/.u8bin style suffix:

    uint32   nq
    uint32   k
    uint32   ids[nq * k]
    float32  dists[nq * k]

A submitted result file uses this same layout, so one reader serves both.

Rows are query *chunks* (and strands), not queries: query.fbin has one row
per chunk and query_meta.parquet maps rows back to query ids. The ids in
gt.bin index rows of the base, which meta.parquet maps to accessions.

The scan is benchmark/src/exact.py's ExactEngine, the same block-streamed,
all-GPU, TF32-off scan the benchmark's exact and PCA rows were scored with.
It reads the merged base file (embeddings.fbin, or the 2-byte
embeddings.{fp16,bf16}bin of a quantized / PCA index, upcast to fp32 per
block on the device), not the published shards: one memmap is simpler than
a few hundred files and the merged file still exists at this point in the
pipeline. The base dimension must equal query.fbin's, so for a PCA base
generate_queries.py must have been run with --projector.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from jsonargparse import auto_cli
from jsonargparse.typing import Path_drw, Path_fr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.src.config import ExactIndex  # noqa: E402
from benchmark.src.exact import ExactEngine  # noqa: E402
from benchmark.src.fbin import _load_fbin_mmap  # noqa: E402

MUTATION_RATES = [0.00, 0.05, 0.10]
BASE_DTYPES = {
    "embeddings.fbin": torch.float32,
    "embeddings.fp16bin": torch.float16,
    "embeddings.bf16bin": torch.bfloat16,
}


def _write_gt_bin(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    nq, k = ids.shape
    with open(path, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dists.astype(np.float32).tofile(f)


def ground_truth(
    base: Path_fr,
    queries_dir: Path_drw,
    k: int = 100,
    block_rows: int = 200_000,
):
    """Write gt.bin into every mutation-rate directory under queries_dir.

    Args:
        base: the merged base file (embeddings.fbin / embeddings.bf16bin ...).
        queries_dir: holds mut<rate>/query.fbin from generate_queries.py.
        k: neighbours per query row.
        block_rows: base rows per streamed block. The logits tile is
            n_query_rows x block_rows fp32 per GPU: 20k rows x 200k = 16 GB,
            which fits a 40 GB A100 beside the block itself.
    """
    base = Path(base)
    queries_dir = Path(queries_dir)
    if base.name not in BASE_DTYPES:
        raise ValueError(f"{base.name} is not one of {list(BASE_DTYPES)}")
    dtype = BASE_DTYPES[base.name]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available for ground truth.")
    devices = [f"cuda:{i}" for i in range(n_gpus)]

    # This is published as exact. TF32 is on by default from A100 up and
    # carries ~1e-3 relative error, which flips near-ties -- and at k=100
    # there are many.
    torch.backends.cuda.matmul.allow_tf32 = False

    engine = ExactEngine(ExactIndex(top_k=k))
    engine.load_file(base, devices, dtype)
    n, d = engine.all_embeddings.shape
    print(
        f"base: {n:,} vectors, dim={d}, {dtype}, {n_gpus} GPUs, block={block_rows:,} rows"
    )

    for mutation_rate in MUTATION_RATES:
        mut_dir = queries_dir / f"mut{mutation_rate:.2f}"
        query_path = mut_dir / "query.fbin"
        if not query_path.exists():
            raise FileNotFoundError(f"No query.fbin in {mut_dir}")

        queries = np.array(_load_fbin_mmap(query_path), dtype=np.float32)
        if queries.shape[1] != d:
            raise ValueError(
                f"dim mismatch: base d={d}, {query_path} d={queries.shape[1]}"
            )

        print(f"mut{mutation_rate:.2f}: {queries.shape[0]} query rows", flush=True)
        started = time.time()
        dists, ids = engine.topk_hits(queries, k, block_rows=block_rows)
        if (ids < 0).any():
            raise RuntimeError("padding in the top-k: base smaller than k?")
        _write_gt_bin(mut_dir / "gt.bin", ids, dists)
        print(
            f"mut{mutation_rate:.2f}: wrote gt.bin {ids.shape} in "
            f"{time.time() - started:.0f}s -> {mut_dir}",
            flush=True,
        )


if __name__ == "__main__":
    auto_cli(ground_truth, as_positional=False)
