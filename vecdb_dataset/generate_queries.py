"""Embed the benchmark queries at each mutation rate into fbin bundles.

One output directory per mutation rate, each holding:

    query.fbin        (n_chunks, dim) float32, big-ann fbin layout
    query_meta.parquet  query_id -> [start_row, start_row + num_rows)

Queries longer than max_seq_len are split into non-overlapping chunks, so
query.fbin has one row per *chunk*, not per query. query_meta.parquet is what
maps a chunk row back to the query that produced it -- the same run-length
shape meta.parquet uses for base vectors.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import auto_cli
from jsonargparse.typing import Path_dc, Path_drw

from benchmark.src.config import ExperimentConfig
from benchmark.src.dense_index import DenseIndex
from lae.training.batcher import Augmenter

MUTATION_RATES = [0.00, 0.05, 0.10]


def _create_fbin_memmap(path: Path, n: int, d: int) -> np.memmap:
    """Create an fbin file with a uint32 [n, d] header and return a writable float32 memmap."""
    with open(path, "wb") as f:
        np.array([n, d], dtype=np.uint32).tofile(f)
        f.seek(n * d * np.dtype(np.float32).itemsize - 1, 1)
        f.write(b"\x00")
    return np.memmap(path, dtype=np.float32, mode="r+", offset=8, shape=(n, d))


def _apply_mutations(queries: pl.DataFrame, mutation_rate: float) -> pl.DataFrame:
    # Inlined from run_benchmark.apply_mutations: that module does `from
    # src.config import ...`, which only resolves with cwd=benchmark/.
    return queries.with_columns(
        pl.col("query_sequence").map_elements(
            lambda query_seq: Augmenter.augment(query_seq, identity=1 - mutation_rate),
            return_dtype=pl.String,
        )
    )


def generate_queries(
    dataset_path: Path_drw, cfg: ExperimentConfig, out: Path_dc, seed: int = 0
):
    dataset_path = Path(dataset_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    queries = pl.read_parquet(dataset_path / "queries.parquet")
    index = DenseIndex(cfg)

    for mutation_rate in MUTATION_RATES:
        mut_out = out / f"mut{mutation_rate:.2f}"
        mut_out.mkdir(exist_ok=True)

        # Reseed per rate rather than once up front, so any single rate can be
        # regenerated on its own. Augmenter draws from the global torch RNG.
        torch.manual_seed(seed)
        mut_queries = _apply_mutations(queries, mutation_rate)

        embeds, query_indices = index._embed_queries(mut_queries)
        embeds = embeds.cpu().numpy()

        n, d = embeds.shape
        mmap = _create_fbin_memmap(mut_out / "query.fbin", n, d)
        mmap[:] = embeds
        mmap.flush()
        del mmap

        query_meta = pl.DataFrame(
            {
                "query_id": queries["query_id"],
                "start_row": [start for start, _ in query_indices],
                "num_rows": [end - start for start, end in query_indices],
            }
        )
        assert query_meta["num_rows"].sum() == n
        query_meta.write_parquet(mut_out / "query_meta.parquet")

        print(f"mut{mutation_rate:.2f}: {len(queries)} queries -> {n} chunks, dim={d}")


if __name__ == "__main__":
    auto_cli(generate_queries, as_positional=False)
