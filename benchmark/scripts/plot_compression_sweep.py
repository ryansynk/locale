"""AUPRC vs compression ratio for the epsilon-net, merged epsilon-net,
random-subsample and PCA sweeps, one figure per mutation rate, with exact
top-k and metagraph as dotted reference lines.

Compression ratio = total vectors / kept vectors, counted from each index's
net_ids.pt (all three engines write one; the merged net's holds one row per
merged center), so the sweeps sit on the same axis
even though epsilon only fixes the ratio after the build. PCA keeps every
vector and shrinks each one instead, so its ratio is encoder dim / pca_dim
(both read from the fbin headers): fp32 bytes of the full vectors over fp32
bytes of the index's, the same thing the net ratios measure. AUPRC and its 95%
bootstrap band come from plot_results.compute_auprc, the numbers
print_results.py prints.

    uv run python scripts/plot_compression_sweep.py            # sra50 defaults
    uv run python scripts/plot_compression_sweep.py --out figs/sra50_compression
    # -> figs/sra50_compression_mut0.00.pdf/.png, _mut0.05, _mut0.10
    uv run python scripts/plot_compression_sweep.py \
        --families '[epsnet, randsample, pca]' --out figs/sra50_compression_pca
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/, for `src`

import numpy as np
import plot_results as pr
import polars as pl
import torch
from jsonargparse import auto_cli
from matplotlib import pyplot as plt
from src.config import read_config
from src.fbin import _load_fbin_mmap

plt.rcParams["text.usetex"] = False  # plot_results turns it on; not needed here

BENCH = Path(__file__).resolve().parents[1]
BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra50/bundle")

# Categorical slots 1-3 of the reference palette, in fixed order.
SERIES = {
    "epsnet": ("Epsilon net", "#2a78d6", "o"),
    "randsample": ("Random subsample", "#eb6834", "s"),
    "mergednet": ("Merged epsilon net", "#1baf7a", "D"),
    # Slot 5: slot 4 is metagraph's reference line below.
    "pca": ("PCA", "#e87ba4", "^"),
}
# Reference lines: (label, color). Exact is neutral ink; metagraph takes the
# fourth categorical slot (slots 1-4 validate all-pairs).
REFERENCES = {
    "exact": ("Exact top-k", "#52514e"),
    "metagraph": ("Metagraph", "#eda100"),
}
GRID_INK = "#e4e3df"


def kept_fraction(index_dir: Path, index_label: str, total: int) -> float:
    if index_label.startswith("pca-"):
        # Stored bytes per vector over the encoder's fp32 bytes: pca_dim x
        # (4 for fp32, 2 for fp16 / bf16), from the index's config.json.
        ident = read_config(index_dir / index_label)
        full_dim = _load_fbin_mmap(index_dir / "embeddings.fbin").shape[1]
        itemsize = 4 if ident.get("dtype", "fp32") == "fp32" else 2
        return ident["pca_dim"] * itemsize / (full_dim * 4)
    return len(torch.load(index_dir / index_label / "net_ids.pt")) / total


def main(
    results_dir: Path = BENCH / "results/sra50/locale@8vqiabk9",
    metagraph_dir: Path | None = BENCH / "results/sra50/metagraph",
    index_dir: Path = BENCH / "indexes/sra50/locale@8vqiabk9",
    raw_read_queries_path: Path = BUNDLE / "queries.parquet",
    accessions: Path = BUNDLE / "accs.txt",
    exact_label: str = "exact",
    families: list[str] = ["epsnet", "randsample", "mergednet"],
    search_label: str = "top100",
    bootstrap_samples: int = 1000,
    out: Path = BENCH / "figs/sra50_compression_sweep",
):
    """
    Args:
        results_dir: the encoder's results directory (holds exact/, epsnet-*/,
            mergednet-*/, randsample-*/).
        metagraph_dir: metagraph's results directory (None: no metagraph line).
        index_dir: the encoder's index directory (holds each net_ids.pt and
            meta.parquet).
        families: sweeps to plot, from SERIES (epsnet, randsample, mergednet,
            pca). Including pca puts the x axis on a log scale: its ratios
            (3-24 for 256-32 dims) run far past the nets' (1.2-3).
        search_label: the dense runs' search label to plot (sra500 also has
            top300 runs of exact and some nets); metagraph keeps its own.
        out: output path prefix; writes <out>_mut<rate>.pdf and .png per rate.
    """
    dirs = [results_dir] + ([metagraph_dir] if metagraph_dir is not None else [])
    data = pr.load_results(dirs)
    if unknown := set(families) - set(SERIES):
        raise ValueError(
            f"unknown families {sorted(unknown)}; pick from {list(SERIES)}"
        )
    family_re = rf"^({'|'.join(families)})-"
    data = data.filter(
        (pl.col("index").str.contains(family_re) | (pl.col("index") == exact_label))
        & (pl.col("search") == search_label)
        | (pl.col("index") == "metagraph")
    )
    accs = Path(accessions).read_text().splitlines()
    oracle = pr.raw_read_oracle_results(pl.read_parquet(raw_read_queries_path))
    oracle = oracle.join(data.select("mutation_rate").unique(), how="cross")
    auprc = pr.compute_auprc(data, oracle, accs, bootstrap_samples)
    auprc = auprc.join(data.select("model", "index").unique(), on="model")

    total = int(pl.read_parquet(index_dir / "meta.parquet")["num_rows"].sum())
    ratios = {
        label: 1 / kept_fraction(index_dir, label, total)
        for label in auprc["index"].unique()
        if label not in (exact_label, "metagraph")
    }
    auprc = auprc.with_columns(
        pl.col("index").replace_strict(ratios, default=None).alias("compression"),
        pl.col("index").str.extract(family_re).alias("family"),
        pl.col("index")
        .replace_strict({exact_label: "exact", "metagraph": "metagraph"}, default=None)
        .alias("reference"),
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for rate in sorted(auprc["mutation_rate"].unique()):
        at_rate = auprc.filter(pl.col("mutation_rate") == rate)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for family in families:
            name, color, marker = SERIES[family]
            s = at_rate.filter(pl.col("family") == family).sort("compression")
            if s.is_empty():
                continue
            x, y, m = s["compression"], s["auprc"], s["margin"]
            ax.fill_between(x, y - m, y + m, color=color, alpha=0.15, lw=0)
            ax.plot(
                x, y, color=color, lw=2, marker=marker, ms=6,
                mec="white", mew=1.5, label=name, solid_capstyle="round",
            )  # fmt: skip
        for ref, (name, color) in REFERENCES.items():
            r = at_rate.filter(pl.col("reference") == ref)
            if not r.is_empty():
                ax.axhline(
                    r["auprc"][0], color=color, lw=1.5, ls=":",
                    label=f"{name} ({r['auprc'][0]:.3f})",
                )  # fmt: skip
        ax.set_title(f"Mutation rate {rate:g}", fontsize=11)
        if "pca" in families:
            ax.set_xscale("log")
            # Log decades leave 1-24 with one labelled tick; label the
            # PCA ratios (and 1-2 for the nets) instead.
            ax.set_xticks([1, 1.5, 2, 3, 6, 12, 24])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlabel("Compression ratio (full / index vector bytes)")
        else:
            ax.set_xlabel("Compression ratio (total / kept vectors)")
        ax.set_ylabel("AUPRC")
        ax.grid(True, color=GRID_INK, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.08)  # keep reference lines off the frame
        # Below the axes: a reference line can sit anywhere in the plot area
        # (metagraph ranges from above exact to far below it across rates).
        ax.legend(
            frameon=False, fontsize=8, ncol=2,
            loc="upper center", bbox_to_anchor=(0.5, -0.2),
        )  # fmt: skip
        fig.tight_layout()
        stem = f"{out.name}_mut{rate:.2f}"  # not with_suffix: "0.05" has a dot
        for ext in ("pdf", "png"):
            fig.savefig(out.parent / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.parent / stem}.pdf / .png")
    print(
        auprc.sort("family", "compression", "mutation_rate").select(
            "index", "compression", "mutation_rate", "range"
        )
    )


if __name__ == "__main__":
    auto_cli(main)
