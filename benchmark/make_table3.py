"""Render one backbone's augmentation ladder in the Table 3 layout.

Recall@Rq is R-precision: rank accessions by score, take the top Rq where Rq is
the number of relevant accessions for that query, and report the fraction of
relevant ones recovered. ``r_precision_per_query`` is imported from
plot_results rather than reimplemented so the two can never drift.

The reported quantity is the WITHIN-backbone trend: how much 10%-mutation
recall improves from none-aug to heavy-aug on this backbone. It is not a
head-to-head against DNABERT-2, and the output is deliberately not formatted to
invite one.

Usage:
    uv run python make_table3.py --results_dir results/ \
        --queries <dataset_dir>/queries.parquet --runs runs_nt50m.yaml

where runs_nt50m.yaml maps each rung to the wandb run id that trained it:

    backbone: nt50m
    step: 11718
    runs:
      none: abcd1234
      light: efgh5678
      medium: ijkl9012
      heavy: mnop3456
"""

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from jsonargparse import auto_cli

from plot_results import r_precision_per_query

RUNGS = ["none", "light", "medium", "heavy"]
MUTATION_RATES = [0.0, 0.05, 0.1]
N_BOOTSTRAP = 1000


def _recall_at_rq(results: pl.DataFrame, relevant: dict[str, set[str]]) -> np.ndarray:
    """Per-query Recall@Rq for one results file."""
    per_query = []
    for row in results.iter_rows(named=True):
        gold = relevant.get(row["query_id"], set())
        if not gold:
            continue
        accs = [r["accession"] for r in row["results"]]
        scores = np.array([r["score"] for r in row["results"]], dtype=np.float64)
        y_true = np.fromiter((a in gold for a in accs), dtype=bool, count=len(accs))
        value = r_precision_per_query(y_true.astype(np.int64), scores)
        if not np.isnan(value):
            per_query.append(value)
    return np.array(per_query, dtype=np.float64)


def _bootstrap_ci(values: np.ndarray, seed: int = 1337) -> float:
    """Half-width of the 95% CI of the mean."""
    if len(values) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ]
    return 1.96 * float(np.std(means))


def main(
    results_dir: Path,
    queries: Path,
    runs: Path,
    max_len: int = 256,
    pooling: str = "mean",
):
    spec = yaml.safe_load(Path(runs).read_text())
    backbone = spec["backbone"]
    step = spec["step"]
    config_tag = f"maxlen{max_len}_pool{pooling}_chunkstride"

    queries_df = pl.read_parquet(queries)
    relevant = {
        row["query_id"]: set(row["contig_accession"])
        for row in queries_df.select("query_id", "contig_accession").iter_rows(
            named=True
        )
    }

    table: dict[str, dict[float, tuple[float, float]]] = {}
    # An empty `runs:` block parses to None, not {} — treat it as no runs yet
    # rather than crashing.
    recorded = spec.get("runs") or {}
    for rung in RUNGS:
        run_id = recorded.get(rung)
        if run_id is None:
            print(f"[skip] no run id for rung {rung!r}")
            continue
        experiment_id = f"locale_{run_id}_{step}_{config_tag}"
        table[rung] = {}
        for rate in MUTATION_RATES:
            path = Path(results_dir) / experiment_id / f"raw_read_mut_{rate}.parquet"
            if not path.exists():
                print(f"[missing] {path}")
                table[rung][rate] = (float("nan"), float("nan"))
                continue
            values = _recall_at_rq(pl.read_parquet(path), relevant)
            table[rung][rate] = (100 * values.mean(), 100 * _bootstrap_ci(values))

    header = " | ".join(f"{int(r * 100)}%" for r in MUTATION_RATES)
    print(f"\nBackbone: {backbone} (step {step}), 47-accession benchmark")
    print("Recall@Rq (%) at eval mutation rate\n")
    print(f"| Training mutation | {header} |")
    print("|---|" + "---|" * len(MUTATION_RATES))
    for rung in RUNGS:
        if rung not in table:
            continue
        cells = " | ".join(
            f"{table[rung][r][0]:.1f} ± {table[rung][r][1]:.1f}" for r in MUTATION_RATES
        )
        print(f"| {rung} | {cells} |")

    if "none" in table and "heavy" in table:
        worst = MUTATION_RATES[-1]
        delta = table["heavy"][worst][0] - table["none"][worst][0]
        clean = table["heavy"][0.0][0] - table["none"][0.0][0]
        print(
            f"\nWithin-backbone delta ({backbone}), none -> heavy:"
            f"\n  at {int(worst * 100)}% eval mutation: {delta:+.1f} points"
            f"\n  at 0% eval mutation:  {clean:+.1f} points"
        )
        print(
            "\nThe claim is that augmentation produces this same trend regardless "
            "of backbone —\nlarge gain in the noisy regime, little cost in the "
            "clean regime. Absolute recall is\nnot comparable to DNABERT-2 and "
            "should not be presented as such."
        )


if __name__ == "__main__":
    # as_positional=False to match train.py and run_benchmark.py, which take
    # named flags. (plot_results.py uses the positional default — the repo is
    # inconsistent, and the flag form is the one documented above.)
    auto_cli(main, as_positional=False)
