"""Exact top-k ground truth for the query bundles, sharded across all GPUs.

For each mutation rate directory produced by generate_queries.py, computes the
exact inner-product top-k over the full base and writes gt.bin beside the
query.fbin it was computed from.

gt.bin uses the big-ann-benchmarks ground-truth layout -- two blocks, two
dtypes, which is why it carries no .fbin/.u8bin style suffix:

    uint32   nq
    uint32   k
    uint32   ids[nq * k]
    float32  dists[nq * k]

A submitted result file uses this same layout, so one reader serves both.

Rows are query *chunks*, not queries: query.fbin has one row per chunk and
query_meta.parquet maps chunks back to query ids. The ids in gt.bin index rows
of the base embeddings.fbin, which meta.parquet maps to accessions.

The base is streamed in tiles rather than loaded. The logits matrix alone is
nq x n (329 GB for sra500), so it can never be materialised. Each GPU owns a
disjoint stride of tiles and keeps its own running top-k; the per-GPU results
are merged once at the end.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from jsonargparse import auto_cli
from jsonargparse.typing import Path_drw
from tqdm import tqdm

MUTATION_RATES = [0.00, 0.05, 0.10]


def _load_fbin_mmap(path: Path) -> np.memmap:
    """Memory-map an fbin file and return a read-only float32 array."""
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(int(n), int(d)))


def _merge_topk(
    vals_a: torch.Tensor,
    ids_a: torch.Tensor,
    vals_b: torch.Tensor,
    ids_b: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold two (nq, *) score/id pairs into a single global top-k."""
    vals = torch.cat([vals_a, vals_b], dim=1)
    ids = torch.cat([ids_a, ids_b], dim=1)
    top_vals, top_pos = torch.topk(vals, k, dim=-1)
    return top_vals, torch.gather(ids, 1, top_pos)


def _write_gt_bin(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    nq, k = ids.shape
    with open(path, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dists.astype(np.float32).tofile(f)


def _exact_topk(
    base: np.memmap, queries: np.ndarray, k: int, tile_rows: int, n_gpus: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k of queries against the whole base.

    GPU g processes tiles g, g+n_gpus, ... so no two devices touch the same
    rows and there is no per-tile synchronisation. Each keeps its own running
    top-k; the n_gpus partials are merged once at the end.
    """
    n, d = base.shape
    nq = queries.shape[0]
    tile_starts = list(range(0, n, tile_rows))
    row_bytes = d * np.dtype(np.float32).itemsize

    # Counted in bytes, not tiles: the run is bound by reading the base off
    # NFS, so a GB/s rate is the number worth watching.
    progress = tqdm(
        total=n * row_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"k={k}, {len(tile_starts)} tiles",
    )
    progress_lock = threading.Lock()

    def _run_device(gpu_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(f"cuda:{gpu_id}")
        q = torch.from_numpy(queries).to(device)
        run_vals = torch.full((nq, k), float("-inf"), device=device)
        run_ids = torch.zeros((nq, k), dtype=torch.int64, device=device)

        for start in tile_starts[gpu_id::n_gpus]:
            end = min(start + tile_rows, n)
            # np.array() forces the NFS read into a real buffer; handing torch
            # a memmap slice would defer the paging into the H2D copy.
            tile = torch.from_numpy(np.array(base[start:end], dtype=np.float32))
            tile = tile.to(device)

            logits = q @ tile.T  # (nq, end - start)
            take = min(k, logits.shape[1])
            vals, ids = torch.topk(logits, take, dim=-1)
            run_vals, run_ids = _merge_topk(run_vals, run_ids, vals, ids + start, k)

            del logits, tile, vals, ids
            with progress_lock:
                progress.update((end - start) * row_bytes)

        return run_vals.cpu(), run_ids.cpu()

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        partials = list(pool.map(_run_device, range(n_gpus)))
    progress.close()

    vals, ids = partials[0]
    for other_vals, other_ids in partials[1:]:
        vals, ids = _merge_topk(vals, ids, other_vals, other_ids, k)

    return ids.numpy(), vals.numpy()


def ground_truth(
    base_dir: Path_drw,
    queries_dir: Path_drw,
    k: int = 100,
    tile_rows: int = 500_000,
):
    """Write gt.bin into every mutation-rate directory under queries_dir.

    base_dir holds embeddings.fbin -- the merged build output, not the shards.
    Tiling one memmap is far simpler than opening several hundred files, and
    the merged file still exists at this point in the pipeline.
    """
    base_dir = Path(base_dir)
    queries_dir = Path(queries_dir)

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available for ground truth.")

    # This is published as exact. TF32 is on by default from A100 up and
    # carries ~1e-3 relative error, which flips near-ties -- and at k=100
    # there are many.
    torch.backends.cuda.matmul.allow_tf32 = False

    base = _load_fbin_mmap(base_dir / "embeddings.fbin")
    n, d = base.shape
    print(f"base: {n:,} vectors, dim={d}, {n_gpus} GPUs, tile={tile_rows:,} rows")

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

        gib = n * d * np.dtype(np.float32).itemsize / 1024**3
        print(
            f"mut{mutation_rate:.2f}: {queries.shape[0]} query chunks, "
            f"reading {gib:.1f} GiB of base"
        )
        started = time.time()
        ids, dists = _exact_topk(base, queries, k, tile_rows, n_gpus)
        _write_gt_bin(mut_dir / "gt.bin", ids, dists)
        print(
            f"mut{mutation_rate:.2f}: wrote gt.bin {ids.shape} in "
            f"{time.time() - started:.0f}s -> {mut_dir}"
        )


if __name__ == "__main__":
    auto_cli(ground_truth, as_positional=False)
