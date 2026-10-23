"""Accession-level recall at the search's top-k vectors, per mutation rate.

For each mut<rate>.parquet in a results directory: the accessions a query's
top-k vectors reached are the ones scored above MISS_SCORE; recall is the
share of the query's relevant accessions (queries.parquet contig_accession)
among them, averaged over the queries in the file. Prints one JSON line,
e.g. {"0.00": 0.93, "0.05": 0.81, "0.10": 0.52}.

  uv run python scripts/score_recall.py \\
      --results_dir results/sra50/locale@8vqiabk9/exact/top100 \\
      --queries /pscratch/sd/r/rsynk/locale-data/constructed/sra50/bundle/queries.parquet
"""

import json
import sys
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/, for `src`

from src.topk_regroup import MISS_SCORE  # noqa: E402


def recall(results: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Mean over queries of |reached relevant accessions| / |relevant accessions|."""
    reached = (
        results.select("query_id", "results")
        .explode("results")
        .unnest("results")
        .filter(pl.col("score") > MISS_SCORE)
        .select("query_id", "accession", pl.lit(True).alias("hit"))
    )
    per_query = (
        truth.filter(pl.col("query_id").is_in(results["query_id"].implode()))
        .join(reached, on=["query_id", "accession"], how="left")
        .group_by("query_id")
        .agg(pl.col("hit").fill_null(False).mean())
    )
    return float(per_query["hit"].mean())


def main(results_dir: Path, queries: Path):
    """
    Args:
        results_dir: directory of mut<rate>.parquet result files from run_benchmark.
        queries: the bundle's queries.parquet (ground truth).
    """
    truth = (
        pl.read_parquet(queries)
        .select("query_id", pl.col("contig_accession").alias("accession"))
        .explode("accession")
        .unique()
    )
    out = {}
    for f in sorted(Path(results_dir).glob("mut?.??.parquet")):
        out[f.stem[len("mut") :]] = recall(pl.read_parquet(f), truth)
    print(json.dumps(out))


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
