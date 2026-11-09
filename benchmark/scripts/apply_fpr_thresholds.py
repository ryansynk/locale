"""Rewrite positive-query results with scores below a calibrated cutoff removed.

Reads the cutoffs calibrate_fpr.py wrote (thresholds.json) and, for every
model in it, copies that model's mut<rate>.parquet results from the positive
results tree into a new tree, with every accession score below the cutoff set
to MISS_SCORE ("never returned"). Nothing else changes, so print_results.py
runs on the new tree unchanged and its AUPRC / recall@k / R-precision are the
metrics at that false-positive rate. One output tree per FPR level:

    <output_root>/fpr<level>/<model path>/mut<rate>.parquet (+ config.json)

The cutoff applied at every mutation rate is the one calibrated on the
negatives at ``threshold_rate`` (clean reads by default). For k-mer-count
models the cutoff is in units of found fraction of the read's k-mers, so the
raw count is compared against cutoff * (read length - k + 1), with the read
lengths of the matching queries_mut<rate>.parquet.

  uv run python scripts/apply_fpr_thresholds.py \\
      --thresholds results/sra500_negatives/calibration/thresholds.json \\
      --positive_results_dir results/sra500 \\
      --positive_bundle /pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle \\
      --index_dir indexes/sra500 \\
      --output_root results/sra500_fpr
  uv run python scripts/print_results.py results/sra500_fpr/fpr0.01 \\
      <bundle>/queries_mut0.00.parquet <bundle>/accs.txt
"""

import json
import shutil
import sys
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/, for `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_fpr import RATE_RE, is_kmer_count_model, metagraph_k, query_lengths  # noqa: E402
from src.topk_regroup import MISS_SCORE, RESULTS_DTYPE  # noqa: E402


def apply_cutoff(results: pl.DataFrame, cutoff: pl.Expr) -> pl.DataFrame:
    """results with every accession score below ``cutoff`` (an expression over
    the row, e.g. a literal or a per-query column) set to MISS_SCORE."""
    return (
        results.with_row_index("_row")
        .explode("results")
        .unnest("results")
        .with_columns(
            pl.when(pl.col("score") >= cutoff)
            .then(pl.col("score"))
            .otherwise(pl.lit(MISS_SCORE))
            .alias("score")
        )
        .group_by("_row", maintain_order=True)
        .agg(
            pl.exclude("accession", "score").first(),
            pl.struct("accession", "score").alias("results"),
        )
        .with_columns(
            # A query whose list was empty explodes to one null accession
            pl.col("results")
            .list.eval(
                pl.element().filter(
                    pl.element().struct.field("accession").is_not_null()
                )
            )
            .cast(RESULTS_DTYPE)
        )
        .drop("_row")
        .select(results.columns)
    )


def main(
    thresholds: Path,
    positive_results_dir: Path,
    output_root: Path,
    positive_bundle: Path | None = None,
    index_dir: Path | None = None,
    threshold_rate: float = 0.0,
    fpr_levels: list[float] | None = None,
):
    """
    Args:
        thresholds: thresholds.json written by calibrate_fpr.py.
        positive_results_dir: results tree of the positive runs (e.g. results/sra500).
        output_root: thresholded trees are written to <output_root>/fpr<level>/.
        positive_bundle: the positives' dataset directory; needed for k-mer-count
            models (read lengths).
        index_dir: index tree, for k of k-mer-count models.
        threshold_rate: negatives' mutation rate whose cutoffs are applied.
        fpr_levels: subset of the calibrated levels to apply (default: all).
    """
    cutoffs = json.loads(Path(thresholds).read_text())
    positive_results_dir = Path(positive_results_dir)
    output_root = Path(output_root)
    rate_key = f"{threshold_rate:.2f}"
    written = []
    for model, by_rate in cutoffs.items():
        if rate_key not in by_rate:
            print(f"[skip] {model}: no cutoff from negatives at rate {rate_key}")
            continue
        model_dir = positive_results_dir / model
        files = sorted(f for f in model_dir.glob("mut?.??.parquet"))
        if not files:
            print(f"[skip] {model}: no results under {model_dir}")
            continue
        kmer = is_kmer_count_model(model)
        if kmer and positive_bundle is None:
            raise SystemExit(f"{model} needs --positive_bundle for read lengths")
        k = metagraph_k(model, Path(index_dir) if index_dir else None) if kmer else None
        for fpr, entry in by_rate[rate_key].items():
            if fpr_levels is not None and float(fpr) not in fpr_levels:
                continue
            t = entry["threshold"]
            out_dir = output_root / f"fpr{fpr}" / model
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                rate = float(RATE_RE.search(f.name).group(1))
                results = pl.read_parquet(f)
                if kmer:
                    lengths = query_lengths(Path(positive_bundle), rate)
                    results = results.join(lengths, on="query_id", how="left")
                    if results["read_length"].null_count():
                        raise SystemExit(
                            f"{f}: queries missing from the bundle at rate {rate}"
                        )
                    cutoff = pl.lit(t) * (pl.col("read_length") - k + 1)
                    out = apply_cutoff(results, cutoff).drop("read_length")
                else:
                    out = apply_cutoff(results, pl.lit(t))
                out.write_parquet(out_dir / f.name)
            if (model_dir / "config.json").exists():
                shutil.copy(model_dir / "config.json", out_dir / "config.json")
            (out_dir / "threshold.json").write_text(
                json.dumps(
                    {"fpr": float(fpr), "negatives_rate": threshold_rate, **entry},
                    indent=2,
                )
                + "\n"
            )
            written.append((model, fpr, t, len(files)))
    for model, fpr, t, n in written:
        print(
            f"{model}: FPR {fpr} cutoff {t:.4f} applied to {n} rate files -> {output_root / f'fpr{fpr}' / model}"
        )
    if not written:
        raise SystemExit("nothing written")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
