"""Score the contig-length cutoff sweep and plot AUPRC vs compression ratio in
the style of benchmark/plot_compression_sweep.py.

Compression ratio = total vectors / vectors from contigs >= cutoff (from the
sweep's sweep.json), the same "total / kept vectors" axis as the epsilon-net
plots. Each point is labelled with its cutoff in bp. Reference lines: the
unfiltered exact top-k (cutoff 0, which reproduces results/sra500/.../exact/
top100) and metagraph. The dashed "alignment oracle" curve is the ceiling the
benchmark allows at each cutoff: the oracle scores every (query, accession)
by the best minimap2 identity over contigs at least cutoff bp long, so a
relevant accession whose supporting contigs are all shorter than the cutoff
becomes unretrievable and the ceiling drops.

AUPRC is the per-query average precision of plot_results.compute_auprc, with
a 95% bootstrap band; the table also reports the paired per-query delta vs
cutoff 0 with a paired bootstrap CI, which is the right comparison for nested
subsets of one index on identical queries.

    cd benchmark
    uv run python ../experiments/seqlen_cutoff/score_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

BENCH = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCH))
import plot_results as pr  # noqa: E402

plt.rcParams["text.usetex"] = False
BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")

SERIES_COLOR, SERIES_MARKER = "#2a78d6", "o"  # categorical slot 1, as epsnet
ORACLE_COLOR = "#8c8a85"
REFERENCES = {
    "exact": ("Exact top-k, no cutoff", "#52514e"),
    "metagraph": ("Metagraph", "#eda100"),
}
GRID_INK = "#e4e3df"


def per_query_ap(
    results: pl.DataFrame, truth: pl.DataFrame, accessions: list[str]
) -> pl.DataFrame:
    """(query_id, ap) exactly as plot_results.compute_auprc scores a run."""
    order = sorted(accessions)
    acc_to_idx = {a: i for i, a in enumerate(order)}
    n_acc = len(order)
    joined = truth.select("query_id", pl.col("results").alias("true")).join(
        results.select("query_id", pl.col("results").alias("pred")), on="query_id"
    )
    rows = []
    for row in joined.iter_rows(named=True):
        y_score = np.full(n_acc, pr.DEFAULT_SCORE)
        for p in row["pred"]:
            if p["accession"] in acc_to_idx:
                y_score[acc_to_idx[p["accession"]]] = p["score"]
        y_true = np.zeros(n_acc, dtype=int)
        for t in row["true"]:
            if t["accession"] in acc_to_idx:
                y_true[acc_to_idx[t["accession"]]] = 1
        if y_true.sum() > 0:
            rows.append(
                {
                    "query_id": row["query_id"],
                    "ap": average_precision_score(y_true, y_score),
                }
            )
    return pl.DataFrame(rows)


def mean_ci(
    x: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(1.96 * boots.std())


def oracle_at_cutoff(queries: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """plot_results.raw_read_oracle_results restricted to contigs >= cutoff."""
    return (
        queries.select("query_id", "contig_accession", "identity", "contig_len")
        .explode("contig_accession", "identity", "contig_len")
        .filter(pl.col("contig_len") >= cutoff)
        .group_by("query_id", "contig_accession")
        .agg(pl.col("identity").max().alias("score"))
        .rename({"contig_accession": "accession"})
        .with_columns(pl.struct("accession", "score").alias("results"))
        .group_by("query_id")
        .agg(pl.col("results"))
    )


def main(
    sweep_dir: Path = BENCH / "results/sra500_minlen",
    encoder_label: str = "locale@8vqiabk9-endwin50",
    metagraph_dir: Path | None = BENCH / "results/sra500/metagraph",
    queries_path: Path = BUNDLE / "queries.parquet",
    accessions: Path = BUNDLE / "accs.txt",
    search_label: str = "top100",
    bootstrap_samples: int = 1000,
    seed: int = 0,
    out: Path = BENCH / "figs/sra500_compression_minlen",
    table_out: Path | None = None,
):
    """
    Args:
        sweep_dir: sweep_search.py's results tree (holds sweep.json).
        table_out: CSV of the per-cutoff table (default <sweep_dir>/scores.csv).
    """
    sweep = json.loads((Path(sweep_dir) / "sweep.json").read_text())
    cutoffs = sorted(int(c) for c in sweep["cutoffs"])
    n_total = sweep["n_vectors"]
    survivors = {int(k): v for k, v in sweep["surviving_vectors"].items()}
    accs = Path(accessions).read_text().splitlines()
    queries = pl.read_parquet(queries_path)
    truth = pr.raw_read_oracle_results(queries).select("query_id", "results")
    rng = np.random.default_rng(seed)

    runs = pr.load_results([Path(sweep_dir)])
    runs = runs.filter(
        (pl.col("encoder") == encoder_label) & (pl.col("search") == search_label)
    ).with_columns(
        pl.col("index")
        .str.extract(r"^exact-minlen(\d+)$")
        .cast(pl.Int64)
        .alias("cutoff")
    )
    rates = sorted(runs["mutation_rate"].unique().to_list())
    meta_runs = pr.load_results([Path(metagraph_dir)]) if metagraph_dir else None

    # Ground truth is independent of the cutoff; the oracle ceiling is not.
    truth_full = truth
    rows = []
    aps: dict[tuple[int, float], pl.DataFrame] = {}
    for rate in rates:
        for c in cutoffs:
            run = runs.filter(
                (pl.col("cutoff") == c) & (pl.col("mutation_rate") == rate)
            )
            if run.is_empty():
                continue
            aps[(c, rate)] = per_query_ap(run, truth_full, accs)
    oracle_ap = {
        c: per_query_ap(oracle_at_cutoff(queries, c), truth_full, accs) for c in cutoffs
    }
    # The oracle keeps every query (its own relevant set is its truth), so
    # restrict its ceiling to the queries the runs scored.
    scored_qids = aps[(0, rates[0])].select("query_id")

    for rate in rates:
        base = aps[(0, rate)].rename({"ap": "ap0"})
        for c in cutoffs:
            if (c, rate) not in aps:
                continue
            ap = aps[(c, rate)]
            mean, margin = mean_ci(ap["ap"].to_numpy(), bootstrap_samples, rng)
            paired = ap.join(base, on="query_id")
            delta = (paired["ap"] - paired["ap0"]).to_numpy()
            dmean, dmargin = mean_ci(delta, bootstrap_samples, rng)
            omean, omargin = mean_ci(
                oracle_ap[c].join(scored_qids, on="query_id")["ap"].to_numpy(),
                bootstrap_samples,
                rng,
            )
            rows.append(
                {
                    "mutation_rate": rate,
                    "cutoff_bp": c,
                    "vectors_kept": survivors[c],
                    "kept_fraction": survivors[c] / n_total,
                    "compression": n_total / survivors[c],
                    "auprc": mean,
                    "margin": margin,
                    "delta_vs_cutoff0": dmean,
                    "delta_margin": dmargin,
                    "oracle_auprc": omean,
                    "oracle_margin": omargin,
                    "n_queries": len(ap),
                }
            )
    table = pl.DataFrame(rows).sort("mutation_rate", "cutoff_bp")

    meta_auprc = {}
    if meta_runs is not None:
        for rate in rates:
            m = meta_runs.filter(pl.col("mutation_rate") == rate)
            if not m.is_empty():
                meta_auprc[rate] = mean_ci(
                    per_query_ap(m, truth_full, accs)["ap"].to_numpy(),
                    bootstrap_samples,
                    rng,
                )[0]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        t = table.filter(pl.col("mutation_rate") == rate).sort("compression")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        x = t["compression"].to_numpy()
        y, m = t["auprc"].to_numpy(), t["margin"].to_numpy()
        oy, om = t["oracle_auprc"].to_numpy(), t["oracle_margin"].to_numpy()
        ax.fill_between(x, oy - om, oy + om, color=ORACLE_COLOR, alpha=0.12, lw=0)
        ax.plot(
            x,
            oy,
            color=ORACLE_COLOR,
            lw=1.5,
            ls="--",
            marker="x",
            ms=5,
            label="Alignment oracle (ceiling)",
        )
        ax.fill_between(x, y - m, y + m, color=SERIES_COLOR, alpha=0.15, lw=0)
        ax.plot(
            x, y, color=SERIES_COLOR, lw=2, marker=SERIES_MARKER, ms=6,
            mec="white", mew=1.5, label="Min contig length", solid_capstyle="round",
        )  # fmt: skip
        for i, (xi, yi, c) in enumerate(zip(x, y, t["cutoff_bp"])):
            if c == 0:
                continue
            # Ratios cluster between 1.8 and 2; alternate above / below.
            dy = -13 if i % 2 else 8
            ax.annotate(
                f"{c}", (xi, yi), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=7, color=SERIES_COLOR,
            )  # fmt: skip
        exact = t.filter(pl.col("cutoff_bp") == 0)["auprc"][0]
        name, color = REFERENCES["exact"]
        ax.axhline(exact, color=color, lw=1.5, ls=":", label=f"{name} ({exact:.3f})")
        if rate in meta_auprc:
            name, color = REFERENCES["metagraph"]
            ax.axhline(
                meta_auprc[rate],
                color=color,
                lw=1.5,
                ls=":",
                label=f"{name} ({meta_auprc[rate]:.3f})",
            )
        ax.set_title(f"Mutation rate {rate:g}", fontsize=11)
        ax.set_xscale("log")
        ax.set_xticks([1, 1.5, 2, 3, 4])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Compression ratio (total / kept vectors)")
        ax.set_ylabel("AUPRC")
        ax.grid(True, color=GRID_INK, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.08)
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
        )
        fig.tight_layout()
        stem = f"{out.name}_mut{rate:.2f}"
        for ext in ("pdf", "png"):
            fig.savefig(out.parent / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.parent / stem}.pdf / .png")

    table_out = Path(table_out) if table_out else Path(sweep_dir) / "scores.csv"
    table.write_csv(table_out)
    pl.Config.set_tbl_rows(len(table))
    pl.Config.set_tbl_cols(20)
    print(
        table.with_columns(
            pl.format(
                "{} +- {}", pl.col("auprc").round(3), pl.col("margin").round(3)
            ).alias("AUPRC"),
            pl.format(
                "{} +- {}",
                pl.col("delta_vs_cutoff0").round(3),
                pl.col("delta_margin").round(3),
            ).alias("delta vs 0"),
            pl.format(
                "{} +- {}",
                pl.col("oracle_auprc").round(3),
                pl.col("oracle_margin").round(3),
            ).alias("oracle"),
            pl.col("compression").round(2),
            pl.col("kept_fraction").round(3),
        ).select(
            "mutation_rate",
            "cutoff_bp",
            "kept_fraction",
            "compression",
            "AUPRC",
            "delta vs 0",
            "oracle",
        )
    )
    if meta_auprc:
        print("metagraph AUPRC:", {r: round(v, 3) for r, v in meta_auprc.items()})
    print(f"table -> {table_out}")


if __name__ == "__main__":
    auto_cli(main)
