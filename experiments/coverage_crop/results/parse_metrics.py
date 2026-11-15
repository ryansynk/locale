"""Score every coverage/crop condition and print the method x condition grids.

Runs benchmark/scripts/print_results.py per condition (over the per-query
result parquets under results/<cond>/, the real data; everything else here is
derived from them), parses its tables with summarize_binned.parse, and writes
metrics_all.csv next to this file. The per-condition tables go to a temporary
directory unless --tables_dir is given.

Run from anywhere; needs the locale benchmark venv:
  cd /pscratch/sd/r/rsynk/locale/benchmark
  uv run --no-sync python ../experiments/coverage_crop/results/parse_metrics.py
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jsonargparse import auto_cli

BENCH = Path("/pscratch/sd/r/rsynk/locale/benchmark")
sys.path.insert(0, str(BENCH / "scripts"))
from summarize_binned import parse  # noqa: E402

HERE = Path(__file__).resolve().parent
BUNDLES = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500_coverage_crop/bundles")
CONDS = ["crop30", "crop50", "cov0.40", "cov0.60", "crop100", "cov0.80", "cov1.00", "orig"]
NAMES = {  # result-dir label prefix -> display name, in display order
    "locale@l6e9itq9": "LOCALE w128 l6e9itq9",
    "locale@8vqiabk9": "LOCALE w256 paper",
    "mmseqs": "MMseqs2",
    "metagraph": "MetaGraph k31",
    "llmed": "LLM-ED",
    "esa": "ESA",
}


def print_results(cond: str, tables_dir: Path) -> Path:
    out = tables_dir / f"print_results_{cond}.txt"
    with open(out, "w") as f:
        subprocess.run(
            ["uv", "run", "--no-sync", "python", "scripts/print_results.py",
             str(HERE / cond), str(BUNDLES / cond / "queries.parquet"), str(BUNDLES / cond / "accs.txt")],
            cwd=BENCH, stdout=f, stderr=subprocess.DEVNULL, check=True,
            env={**os.environ, "POLARS_FMT_STR_LEN": "200"},
        )
    return out


def main(
    results_dir: Path = HERE,
    tables_dir: Path | None = None,
    metrics: list[str] = ["AUPRC", "RECALL AT 7", "R PRECISION"],
):
    """
    Args:
        results_dir: holds <cond>/<method>/... per-query result parquets.
        tables_dir: where to keep the per-condition print_results tables;
            default a temporary directory, discarded afterwards.
        metrics: table sections to print as grids.
    """
    results_dir = Path(results_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tables_dir) if tables_dir else Path(tmp)
        tdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for cond in CONDS:
            if not (results_dir / cond).is_dir():
                print(f"{cond}: no results, skipped", file=sys.stderr)
                continue
            for r in parse(print_results(cond, tdir)):
                key = next((k for k in NAMES if r["model"].startswith(k)), r["model"])
                rows.append({**r, "condition": cond, "method": NAMES.get(key, key)})
    if not rows:
        raise SystemExit("no results parsed")
    with open(results_dir / "metrics_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    methods = [m for m in NAMES.values() if any(r["method"] == m for r in rows)]
    methods += sorted({r["method"] for r in rows} - set(methods))
    for metric in metrics:
        for rate in (0.0, 0.05, 0.1):
            print(f"\n{metric} @ rate {rate}")
            print(f"{'method':22s} " + " ".join(f"{c:>7s}" for c in CONDS))
            for m in methods:
                vals = {r["condition"]: r["mean"] for r in rows
                        if r["metric"] == metric and r["method"] == m and abs(r["mutation_rate"] - rate) < 1e-9}
                print(f"{m:22s} " + " ".join(f"{vals[c]:7.3f}" if c in vals else "      -" for c in CONDS))


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
