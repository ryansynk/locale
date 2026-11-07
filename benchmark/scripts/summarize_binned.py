"""Collect print_results.py outputs for the identity bins into one table.

Reads out_dir/print_results_<bin>.txt (written by run_sra500_binned.sh with
POLARS_FMT_STR_LEN raised so model ids are not truncated), parses the
R PRECISION / RECALL AT k / AUPRC tables, and writes out_dir/binned_metrics.csv
(metric, model, mutation_rate, bin, mean, margin, n_queries). Prints one pivot
per metric and rate: models x bins.

  uv run python scripts/summarize_binned.py --out_dir results/sra500_binned \\
      --models 'locale@8vqiabk9/exact|w128|metagraph|mmseqs|esa'
"""

import re
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli

ROW = re.compile(r"^│\s*(?P<model>\S.*?)\s*┆(?P<rest>.*)│\s*$")


def parse(path: Path) -> list[dict]:
    rows, section = [], None
    for line in open(path):
        m = re.match(r"=+ (.*?) DATA =+", line)
        if m:
            section = m.group(1)
            continue
        if section in (None, "SYSTEMS") or "+-" not in line:
            continue
        m = ROW.match(line.rstrip("\n"))
        if not m:
            continue
        cells = [c.strip() for c in m.group("rest").split("┆")]
        # columns differ per table; the rate is the one numeric cell that is
        # not the recall k, and percent in the R-precision/recall tables.
        nums = [c for c in cells[:-1] if re.fullmatch(r"[\d.]+", c)]
        rate = float(nums[0])
        rate = rate / 100 if rate > 1 else rate
        mean, margin = cells[-1].split("+-")
        rows.append({
            "metric": section, "model": m.group("model"), "mutation_rate": rate,
            "mean": float(mean), "margin": float(margin),
        })
    return rows


def main(out_dir: Path, models: str | None = None):
    """
    Args:
        out_dir: directory with print_results_<bin>.txt and query_bins.parquet.
        models: regex; only models matching it are printed (all are written to csv).
    """
    out_dir = Path(out_dir)
    counts = (
        pl.read_parquet(out_dir / "query_bins.parquet").group_by("bin").len()
        .rename({"len": "n_queries"})
    )
    counts = pl.concat([
        counts,
        pl.DataFrame({"bin": ["all"], "n_queries": [int(counts["n_queries"].sum())]},
                     schema=counts.schema),
    ])
    n_of = dict(zip(counts["bin"], counts["n_queries"]))
    frames = []
    for f in sorted(out_dir.glob("print_results_*.txt")):
        b = f.stem[len("print_results_"):]
        rows = parse(f)
        if rows:
            frames.append(pl.DataFrame(rows).with_columns(pl.lit(b).alias("bin")))
    if not frames:
        raise SystemExit(f"no print_results_*.txt under {out_dir}")
    df = pl.concat(frames)
    # A text file for a bin the bin table does not know about means the two
    # are out of sync; labelling it n=0 would quietly mislabel real numbers.
    unknown = sorted(set(df["bin"]) - set(n_of))
    if unknown:
        raise SystemExit(
            f"print_results_*.txt present for bins absent from query_bins.parquet: "
            f"{unknown}. Re-run bin_queries_by_reference.py, or delete the stale files."
        )
    df = df.join(counts, on="bin", how="left")
    df.write_csv(out_dir / "binned_metrics.csv")

    present = set(df["bin"])
    bins = [b for b in ["all", "unaligned"] if b in present]
    bins += sorted(b for b in set(df["bin"]) if b not in bins)
    show = df.filter(pl.col("model").str.contains(models)) if models else df
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=80, tbl_width_chars=250):
        for (metric, rate), g in show.group_by(["metric", "mutation_rate"], maintain_order=True):
            header = {b: f"{b} (n={n_of[b]})" for b in bins}
            print(f"\n=== {metric} @ mutation {rate:.2f}  (mean +- bootstrap margin) ===")
            piv = (
                g.with_columns((pl.col("mean").round(3).cast(pl.String) + "±" + pl.col("margin").round(3).cast(pl.String)).alias("v"))
                .pivot(on="bin", index="model", values="v")
            )
            cols = [b for b in bins if b in piv.columns]
            print(piv.select("model", *cols).rename({b: header[b] for b in cols}).sort("model"))
    print(f"\nwrote {out_dir / 'binned_metrics.csv'}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
