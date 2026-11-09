"""Score cutoffs at fixed false-positive rates, calibrated on negative queries.

A negative query has no relevant accession in the index, so whatever it
retrieves is a false positive. Per method, the max accession score of each
negative query gives the null distribution; the cutoff at false-positive rate
``a`` is the score above which a fraction ``a`` of negative queries still
return at least one accession. A method that returns nothing for a query
(empty hit list) counts as scoring MISS_SCORE, i.e. it never crosses a cutoff.

Scores are compared across queries, so they must be comparable across queries:
LOCALE's cosine already is; metagraph's score is the raw count of query
k-mers found in an accession, which grows with read length, and is divided by
the number of k-mers in the read (len - k + 1) before anything else. k is read
from <index_dir>/<index label>/config.json when index_dir is given, else from
a ``-k<k>`` suffix of the label, else METAGRAPH_DEFAULT_K.

With a positive results directory and its bundle, each positive query's best
score among its relevant accessions is laid next to the null distribution,
and the true-positive rate at each cutoff (share of positive queries whose
best relevant accession clears it) is reported alongside: recall at fixed
false-positive rate. Cutoffs are taken from the negatives at
``threshold_rate`` (clean reads by default) and applied to the positives at
every mutation rate; a rate-matched cutoff is reported as well.

Outputs, under output_dir (default <negative_results_dir>/calibration):
  thresholds.json   {model: {neg_rate: {fpr: {threshold, ci95, realized_fpr,
                     n_negatives, no_hit_fraction}}}} - the input of the
                     thresholded rescoring step
  thresholds.csv    the same, flat
  tpr.csv           per model / positive rate / fpr: TPR under the
                     threshold_rate cutoff and under the matched-rate cutoff
  scores_mut<rate>.png  per model: negative max scores vs positive best
                     relevant scores, cutoffs marked

  uv run python scripts/calibrate_fpr.py \\
      --negative_results_dir results/sra500_negatives \\
      --negative_bundle /pscratch/sd/r/rsynk/locale-data/constructed/sra500_negatives/bundle \\
      --positive_results_dir results/sra500 \\
      --positive_bundle /pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle \\
      --index_dir indexes/sra500
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/, for `src`

from src.topk_regroup import MISS_SCORE  # noqa: E402

METAGRAPH_DEFAULT_K = 31
# dataviz reference palette, categorical slots 1 and 2 (light surface)
COLOR_NEGATIVE = "#2a78d6"
COLOR_POSITIVE = "#eb6834"
RATE_RE = re.compile(r"mut(\d\.\d\d)\.parquet$")


def model_key() -> pl.Expr:
    """ "<encoder>/<index>/<search>", or "<index>/<search>" for non-dense
    methods: plot_results.model_key, repeated here so this script does not
    import plot_results' matplotlib/LaTeX setup."""
    return (
        pl.when(pl.col("encoder").is_null())
        .then(pl.concat_str([pl.col("index"), pl.col("search")], separator="/"))
        .otherwise(
            pl.concat_str(
                [pl.col("encoder"), pl.col("index"), pl.col("search")], separator="/"
            )
        )
        .alias("model")
    )


def load_results(results_dir: Path, models: list[str] | None) -> pl.DataFrame:
    """Every mut<rate>.parquet under results_dir (hits/ and partials skipped),
    with ``model`` added; ``models`` keeps the keys containing any pattern."""
    frames = []
    for f in sorted(results_dir.rglob("mut?.??.parquet")):
        if any(
            p == "hits" or p.startswith("partials_")
            for p in f.relative_to(results_dir).parts
        ):
            continue
        frames.append(
            pl.read_parquet(
                f,
                columns=[
                    "query_id",
                    "results",
                    "encoder",
                    "index",
                    "search",
                    "mutation_rate",
                ],
            )
        )
    if not frames:
        raise SystemExit(f"No mut<rate>.parquet results under {results_dir}")
    data = pl.concat(frames).with_columns(model_key())
    if models:
        data = data.filter(
            pl.any_horizontal(
                [pl.col("model").str.contains(m, literal=True) for m in models]
            )
        )
        if data.is_empty():
            raise SystemExit(f"No model under {results_dir} matches {models}")
    return data


def metagraph_k(model: str, index_dir: Path | None) -> int:
    index_label = model.split("/")[0]
    if index_dir is not None:
        cfg = index_dir / index_label / "config.json"
        if cfg.exists():
            k = json.loads(cfg.read_text()).get("k")
            if k is not None:
                return int(k)
    if m := re.search(r"-k(\d+)$", index_label):
        return int(m.group(1))
    print(f"[warn] no k found for {model}; assuming k={METAGRAPH_DEFAULT_K}")
    return METAGRAPH_DEFAULT_K


def is_kmer_count_model(model: str) -> bool:
    return model.startswith("metagraph")


def query_lengths(bundle: Path, rate: float) -> pl.DataFrame:
    """query_id -> read length at the given mutation rate (mutated reads differ
    in length from the clean ones by their indels)."""
    return pl.read_parquet(
        bundle / f"queries_mut{rate:.2f}.parquet",
        columns=["query_id", "query_sequence"],
    ).select("query_id", pl.col("query_sequence").str.len_chars().alias("read_length"))


def per_query_scores(
    data: pl.DataFrame, truth: pl.DataFrame | None, bundle: Path, index_dir: Path | None
) -> pl.DataFrame:
    """One row per (model, mutation_rate, query_id): ``max_score`` over every
    returned accession and, with ``truth`` (query_id, accession), ``best_relevant``
    over the query's relevant accessions. Unreturned accessions and empty hit
    lists score MISS_SCORE. k-mer-count models are normalized per query."""
    exploded = (
        data.select("model", "mutation_rate", "query_id", "results")
        .explode("results")
        .unnest("results")
        .filter(pl.col("score") > MISS_SCORE)
    )
    if truth is not None:
        exploded = exploded.join(
            truth.with_columns(pl.lit(True).alias("relevant")),
            on=["query_id", "accession"],
            how="left",
        ).with_columns(pl.col("relevant").fill_null(False))
    else:
        exploded = exploded.with_columns(pl.lit(False).alias("relevant"))
    agg = exploded.group_by("model", "mutation_rate", "query_id").agg(
        pl.col("score").max().alias("max_score"),
        pl.col("score").filter(pl.col("relevant")).max().alias("best_relevant"),
        pl.len().alias("n_returned"),
    )
    frame = (
        data.select("model", "mutation_rate", "query_id")
        .unique()
        .join(agg, on=["model", "mutation_rate", "query_id"], how="left")
        .with_columns(
            pl.col("max_score").fill_null(MISS_SCORE),
            pl.col("best_relevant").fill_null(MISS_SCORE),
            pl.col("n_returned").fill_null(0),
        )
    )
    # Normalize k-mer counts by the number of k-mers in the read
    kmer_models = [m for m in frame["model"].unique() if is_kmer_count_model(m)]
    if kmer_models:
        lengths = pl.concat(
            [
                query_lengths(bundle, rate).with_columns(
                    pl.lit(rate).alias("mutation_rate")
                )
                for rate in frame["mutation_rate"].unique()
            ]
        )
        ks = pl.DataFrame(
            {
                "model": kmer_models,
                "k": [metagraph_k(m, index_dir) for m in kmer_models],
            }
        )
        frame = (
            frame.join(lengths, on=["query_id", "mutation_rate"], how="left")
            .join(ks, on="model", how="left")
            .with_columns(
                [
                    pl.when(pl.col("k").is_not_null() & (pl.col(c) > MISS_SCORE))
                    .then(pl.col(c) / (pl.col("read_length") - pl.col("k") + 1))
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ("max_score", "best_relevant")
                ]
            )
            .drop("read_length", "k")
        )
        if frame.filter(
            pl.col("model").is_in(kmer_models) & pl.col("max_score").is_nan()
        ).height:
            raise ValueError("k-mer normalization produced NaN: query lengths missing?")
    return frame


def threshold_at_fpr(neg_max: np.ndarray, fpr: float) -> float:
    """Smallest cutoff such that at most fpr of the negatives score >= it.

    With n negatives, m = floor(fpr * n) are allowed through: the cutoff is the
    m-th largest negative score (exactly m clear it when there are no ties).
    When m is 0 the cutoff sits just above the top negative."""
    s = np.sort(neg_max)[::-1]
    m = int(np.floor(fpr * len(s)))
    if m == 0:
        return float(np.nextafter(s[0], np.inf))
    if s[m - 1] <= MISS_SCORE:
        # Fewer than m negatives retrieved anything: "any hit" is the cutoff
        # (realized FPR is the hit fraction, below the nominal rate)
        return float(np.nextafter(MISS_SCORE, np.inf))
    return float(s[m - 1])


def bootstrap_ci(
    neg_max: np.ndarray, fpr: float, samples: int, rng
) -> tuple[float, float]:
    draws = [
        threshold_at_fpr(rng.choice(neg_max, size=len(neg_max), replace=True), fpr)
        for _ in range(samples)
    ]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def plot_scores(
    neg: pl.DataFrame,
    pos: pl.DataFrame | None,
    thresholds: dict,
    threshold_rate: float,
    rate: float,
    out_png: Path,
):
    """Per model, the null distribution and (if given) the positives' best
    relevant score at this rate, with the cutoffs from the threshold_rate
    negatives marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.usetex": False, "font.size": 9})
    models = sorted(neg["model"].unique())
    fig, axes = plt.subplots(
        len(models), 1, figsize=(7, 2.6 * len(models)), squeeze=False
    )
    for ax, model in zip(axes[:, 0], models):
        n = neg.filter((pl.col("model") == model) & (pl.col("mutation_rate") == rate))
        if n.is_empty():
            continue
        n_scores = n["max_score"].to_numpy()
        series = [
            (f"negatives, max score (n={len(n_scores)})", n_scores, COLOR_NEGATIVE)
        ]
        if pos is not None:
            p = pos.filter(
                (pl.col("model") == model) & (pl.col("mutation_rate") == rate)
            )
            if not p.is_empty():
                p_scores = p["best_relevant"].to_numpy()
                series.append(
                    (
                        f"positives, best relevant (n={len(p_scores)})",
                        p_scores,
                        COLOR_POSITIVE,
                    )
                )
        scored = np.concatenate([s[s > MISS_SCORE] for _, s, _ in series])
        lo = float(scored.min()) if len(scored) else 0.0
        hi = float(scored.max()) if len(scored) else 1.0
        bins = np.linspace(lo, hi, 50)
        for label, scores, color in series:
            no_hit = float((scores <= MISS_SCORE).mean())
            ax.hist(
                scores[scores > MISS_SCORE],
                bins=bins,
                histtype="step",
                linewidth=2,
                color=color,
                label=f"{label}, no hit {no_hit:.0%}",
            )
        cutoffs = thresholds.get(model, {}).get(f"{threshold_rate:.2f}", {})
        for i, (fpr, t) in enumerate(cutoffs.items()):
            # Labels staggered down the line so coinciding cutoffs stay legible
            ax.axvline(t["threshold"], color="#52514e", linestyle="--", linewidth=1)
            ax.text(
                t["threshold"],
                ax.get_ylim()[1] * (0.95 - 0.3 * i),
                f" FPR {float(fpr):.0%}",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="#52514e",
            )
        ax.set_title(f"{model}  (mutation rate {rate:.2f})", loc="left", fontsize=9)
        ax.set_ylabel("queries")
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel(
        "per-query score (k-mer models: found fraction of read k-mers)"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(
    negative_results_dir: Path,
    negative_bundle: Path,
    positive_results_dir: Path | None = None,
    positive_bundle: Path | None = None,
    index_dir: Path | None = None,
    fpr_levels: list[float] = [0.01, 0.05],
    threshold_rate: float = 0.0,
    models: list[str] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    output_dir: Path | None = None,
):
    """
    Args:
        negative_results_dir: results tree of the negative-query runs
            (run_benchmark output, e.g. results/sra500_negatives).
        negative_bundle: the negatives' dataset directory (queries_mut<rate>.parquet,
            for read lengths).
        positive_results_dir: results tree of the positive runs against the same
            index (e.g. results/sra500); only models present among the
            negatives are used.
        positive_bundle: the positives' dataset directory (queries.parquet for the
            relevant sets, queries_mut<rate>.parquet for read lengths).
        index_dir: index tree of the shared index (metagraph k is read from
            <index_dir>/<label>/config.json).
        fpr_levels: false-positive rates to calibrate.
        threshold_rate: negatives' mutation rate whose cutoffs are applied to
            the positives at every rate.
        models: keep only model keys containing one of these substrings
            (e.g. ["locale@8vqiabk9/exact", "metagraph-k19"]).
        bootstrap_samples: resamples for the cutoff confidence intervals.
        seed: bootstrap seed.
        output_dir: where to write; default <negative_results_dir>/calibration.
    """
    negative_results_dir = Path(negative_results_dir)
    output_dir = (
        Path(output_dir) if output_dir else negative_results_dir / "calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    neg_data = load_results(negative_results_dir, models)
    neg = per_query_scores(neg_data, None, Path(negative_bundle), index_dir)
    neg_models = sorted(neg["model"].unique())

    pos = None
    if positive_results_dir is not None:
        if positive_bundle is None:
            raise SystemExit("positive_bundle is required with positive_results_dir")
        truth = (
            pl.read_parquet(
                Path(positive_bundle) / "queries.parquet",
                columns=["query_id", "contig_accession"],
            )
            .select("query_id", pl.col("contig_accession").alias("accession"))
            .explode("accession")
            .unique()
        )
        pos_data = load_results(Path(positive_results_dir), None).filter(
            pl.col("model").is_in(neg_models)
        )
        if pos_data.is_empty():
            print(
                f"[warn] no model of the negatives has results under {positive_results_dir}"
            )
        else:
            pos = per_query_scores(pos_data, truth, Path(positive_bundle), index_dir)

    # Cutoffs per model and negatives' rate
    thresholds: dict[str, dict[str, dict[str, dict]]] = {}
    rows = []
    for (model, rate), group in neg.group_by(
        "model", "mutation_rate", maintain_order=True
    ):
        neg_max = group["max_score"].to_numpy()
        for fpr in fpr_levels:
            t = threshold_at_fpr(neg_max, fpr)
            lo, hi = bootstrap_ci(neg_max, fpr, bootstrap_samples, rng)
            entry = {
                "threshold": t,
                "ci95": [lo, hi],
                "realized_fpr": float((neg_max >= t).mean()),
                "n_negatives": int(len(neg_max)),
                "no_hit_fraction": float((neg_max <= MISS_SCORE).mean()),
            }
            thresholds.setdefault(model, {}).setdefault(f"{rate:.2f}", {})[f"{fpr}"] = (
                entry
            )
            rows.append(
                {
                    "model": model,
                    "negatives_rate": rate,
                    "fpr": fpr,
                    **{k: v for k, v in entry.items() if k != "ci95"},
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    thresholds_df = pl.from_dicts(rows).sort("model", "negatives_rate", "fpr")
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    thresholds_df.write_csv(output_dir / "thresholds.csv")
    pl.Config.set_tbl_rows(len(thresholds_df))
    pl.Config.set_fmt_str_lengths(60)
    print("=== cutoffs from negatives ===")
    print(thresholds_df)

    # Recall at fixed FPR on the positives
    if pos is not None:
        tpr_rows = []
        for (model, rate), group in pos.group_by(
            "model", "mutation_rate", maintain_order=True
        ):
            best = group["best_relevant"].to_numpy()
            for fpr in fpr_levels:
                fixed = (
                    thresholds.get(model, {})
                    .get(f"{threshold_rate:.2f}", {})
                    .get(f"{fpr}")
                )
                matched = thresholds.get(model, {}).get(f"{rate:.2f}", {}).get(f"{fpr}")
                tpr_rows.append(
                    {
                        "model": model,
                        "positives_rate": rate,
                        "fpr": fpr,
                        "n_positives": int(len(best)),
                        "threshold_fixed": fixed["threshold"] if fixed else None,
                        "tpr_fixed": float((best >= fixed["threshold"]).mean())
                        if fixed
                        else None,
                        "threshold_matched": matched["threshold"] if matched else None,
                        "tpr_matched": float((best >= matched["threshold"]).mean())
                        if matched
                        else None,
                    }
                )
        tpr_df = pl.from_dicts(tpr_rows).sort("model", "positives_rate", "fpr")
        tpr_df.write_csv(output_dir / "tpr.csv")
        print(
            f"=== recall at fixed FPR (fixed = cutoff from negatives at rate {threshold_rate:.2f}) ==="
        )
        print(tpr_df)

    rates = sorted(
        set(neg["mutation_rate"].unique())
        | (set(pos["mutation_rate"].unique()) if pos is not None else set())
    )
    for rate in rates:
        if neg.filter(pl.col("mutation_rate") == rate).is_empty():
            continue
        out_png = output_dir / f"scores_mut{rate:.2f}.png"
        plot_scores(neg, pos, thresholds, threshold_rate, rate, out_png)
    print(
        f"Wrote thresholds.json, thresholds.csv{', tpr.csv' if pos is not None else ''} and plots to {output_dir}"
    )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
