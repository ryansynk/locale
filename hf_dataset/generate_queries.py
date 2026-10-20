"""Embed the benchmark queries at each mutation rate into fbin bundles.

One output directory per mutation rate, each holding:

    query.fbin          (n_rows, dim) float32, big-ann fbin layout
    query_meta.parquet  query_id -> [start_row, start_row + num_rows), and per
                        query how many of those rows are the forward strand

Queries longer than max_seq_len are split into non-overlapping chunks, so
query.fbin has one row per *chunk*, not per query. With both_strands (the
benchmark default) every query also contributes its reverse complement,
chunked the same way: inside a query's row range the forward chunks come
first, then the reverse-complement chunks, and ``num_forward`` in
query_meta.parquet says where the split is. A scorer reduces over a query's
whole range (max score per base vector), as the benchmark does.

``--projector`` makes the rows match a PCA-projected base: the encoder
vectors are multiplied by the (encoder_dim, pca_dim) projector.pt in fp32,
exactly as PCAEngine.topk_hits does, so query.fbin has the base's dimension
and gt.bin can be computed against the published vectors directly.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import auto_cli
from jsonargparse.typing import Path_dc, Path_drw, Path_fr

from benchmark.src.config import ExperimentConfig
from benchmark.src.dense_index import DenseIndex

MUTATION_RATES = [0.00, 0.05, 0.10]


def _create_fbin_memmap(path: Path, n: int, d: int) -> np.memmap:
    """Create an fbin file with a uint32 [n, d] header and return a writable float32 memmap."""
    with open(path, "wb") as f:
        np.array([n, d], dtype=np.uint32).tofile(f)
        f.seek(n * d * np.dtype(np.float32).itemsize - 1, 1)
        f.write(b"\x00")
    return np.memmap(path, dtype=np.float32, mode="r+", offset=8, shape=(n, d))


def generate_queries(
    dataset_path: Path_drw,
    cfg: ExperimentConfig,
    out: Path_dc,
    projector: Path_fr | None = None,
):
    """
    Args:
        dataset_path: bundle dir holding queries_mut<rate>.parquet.
        cfg: benchmark config (the encoder and both_strands come from it).
        out: bundle dir to write mut<rate>/ into.
        projector: projector.pt of a PCA index; when given, rows are the
            projected vectors (fp32) instead of the encoder's.
    """
    dataset_path = Path(dataset_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    index = DenseIndex(cfg)
    proj = None
    if projector is not None:
        proj = torch.load(Path(projector), map_location="cpu").to(torch.float32)
        print(f"projector {tuple(proj.shape)} from {projector}")

    for mutation_rate in MUTATION_RATES:
        mut_out = out / f"mut{mutation_rate:.2f}"
        mut_out.mkdir(exist_ok=True)

        # The pre-mutated bundle files (locale-data/benchmark/mutate_queries.py),
        # so these vectors embed exactly the sequences the benchmark searches.
        queries = pl.read_parquet(
            dataset_path / f"queries_mut{mutation_rate:.2f}.parquet"
        )

        embeds, query_indices, strands = index._embed_queries(queries)
        embeds = embeds.to(torch.float32).cpu()
        if proj is not None:
            if embeds.shape[1] != proj.shape[0]:
                raise ValueError(
                    f"encoder dim {embeds.shape[1]} != projector rows {proj.shape[0]}"
                )
            embeds = embeds @ proj  # same fp32 CPU matmul as PCAEngine.topk_hits
        embeds = embeds.numpy()

        n, d = embeds.shape
        mmap = _create_fbin_memmap(mut_out / "query.fbin", n, d)
        mmap[:] = embeds
        mmap.flush()
        del mmap

        strands = strands.numpy()
        num_forward = [int((strands[s:e] == 0).sum()) for s, e in query_indices]
        query_meta = pl.DataFrame(
            {
                "query_id": queries["query_id"],
                "start_row": [start for start, _ in query_indices],
                "num_rows": [end - start for start, end in query_indices],
                "num_forward": num_forward,
            }
        )
        assert query_meta["num_rows"].sum() == n
        query_meta.write_parquet(mut_out / "query_meta.parquet")

        print(
            f"mut{mutation_rate:.2f}: {len(queries)} queries -> {n} rows "
            f"({int((strands == 0).sum())} forward), dim={d}"
        )


if __name__ == "__main__":
    auto_cli(generate_queries, as_positional=False)
