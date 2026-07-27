"""Print benchmark result tables without plotting.

plot_results.py's main() interleaves tables and figures, and two of its figure
calls are hardcoded to mutation_rate=10 (plot_recall_at_k_vs_k_line's default at
line 447, plot_r_precision_vs_time's at line 595). On a single-rate run those
filters empty the frame and pl.from_dicts([]) raises, which kills the process
before the AUPRC and systems tables ever print.

This reuses the same functions -- no metric logic is duplicated -- but calls
only the four that emit tables, so it works with any set of mutation rates.
"""

import tempfile
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr
from matplotlib import pyplot as plt

import plot_results as pr

# The two table-emitting functions also savefig. We throw the PDFs away, so skip
# the LaTeX round-trip that makes that slow.
plt.rcParams["text.usetex"] = False

# Kept in sync with plot_results.main() by hand; polars validates it on read, so
# a drift shows up as a loud schema error rather than silently wrong numbers.
SCHEMA = pl.Schema(
    {
        "query_id": pl.String,
        "results": pl.List(pl.Struct({"accession": pl.String, "score": pl.Float64})),
        "model": pl.String,
        "mutation_rate": pl.Float64,
        "query_type": pl.String,
        "checkpoint": pl.String,
        "max_len": pl.Int64,
        "checkpoint_step_num": pl.Int64,
        "chunk_type": pl.String,
        "avg_time": pl.Float64,
        "index_size_gb": pl.Float64,
    }
)


def main(
    results_dir: str,
    raw_read_queries_path: Path_fr,
    accessions: Path_fr,
    k: int = 7,
    bootstrap_samples: int = 10000,
):
    results_dir = Path(results_dir)
    with open(accessions) as f:
        accs = f.read().splitlines()

    files = sorted(results_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files under {results_dir}")
    data = pl.concat([pl.read_parquet(f, schema=SCHEMA) for f in files])

    oracle = pr.raw_read_oracle_results(pl.read_parquet(Path(raw_read_queries_path)))
    oracle = oracle.join(data.select("mutation_rate").unique(), how="cross")

    rates = sorted(data["mutation_rate"].unique().to_list())
    print(f"Loaded {len(files)} parquet(s), {len(data)} rows")
    print(f"models        : {sorted(data['model'].unique().to_list())}")
    print(f"mutation rates: {rates}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pr.plot_r_precision_vs_noise_line(
            data, oracle, accs, tmp, bootstrap_samples, print_data=True
        )
        pr.plot_recall_at_k_vs_noise_line(
            data, oracle, accs, k, tmp, bootstrap_samples, print_data=True
        )
    pr.print_auprc(data, oracle, accs, bootstrap_samples)
    print("=========== SYSTEMS DATA =============")
    pr.print_systems_data(data)


if __name__ == "__main__":
    auto_cli(main)
