"""AUPRC vs aligned bases, paper style (benchmark/scripts/plot_results.py).

Two figures, each 1x3 (0% / 5% / 10% mutation): crop_auprc_vs_bases.pdf and
coverage_auprc_vs_bases.pdf under results/plots/, plus PNG previews. Reads
results/metrics_all.csv written by results/parse_metrics.py.

    cd benchmark && uv run --no-sync python ../experiments/coverage_crop/plot_auprc.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same rcParams as benchmark/scripts/plot_results.py.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"""
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{amsfonts}
        \usepackage{amsmath}
        \renewcommand{\rmdefault}{ptm}
        \renewcommand{\sfdefault}{phv}
    """,
    }
)

E = Path(__file__).resolve().parent
OUT = E / "results" / "plots"
# condition -> aligned bases (coverage fills the rest of 128 bp with random bases)
CROP = {"crop30": 30, "crop50": 50, "crop100": 100, "cov1.00": 128}
COVER = {"cov0.40": 51, "cov0.60": 77, "cov0.80": 102, "cov1.00": 128}
RATES = [0.0, 0.05, 0.1]

# metrics_all.csv method -> display name, in plot order (paper MODEL_ORDER with
# the two LOCALE windows first). Colors come from viridis over this order, as
# in plot_results.py; line styles follow its cycle except MMseqs2, kept dotted.
MODELS = [
    ("LOCALE w128 l6e9itq9", "LOCALE (128 bp window)"),
    ("LOCALE w256 paper", "LOCALE (256 bp window)"),
    ("MMseqs2", "MMseqs2"),
    ("LLM-ED", "LLM-ED"),
    ("ESA", "ESA"),
    ("MetaGraph k31", "MetaGraph"),
]
LINESTYLES = {
    "LOCALE w128 l6e9itq9": "-",
    "LOCALE w256 paper": "--",
    "MMseqs2": ":",
    "LLM-ED": "-.",
    "ESA": (0, (3, 1, 1, 1)),
    "MetaGraph k31": (0, (5, 2)),
}
viridis = plt.colormaps["viridis"]
COLORS = {m: viridis(i / (len(MODELS) - 1)) for i, (m, _) in enumerate(MODELS)}


def load():
    rows = list(csv.DictReader(open(E / "results" / "metrics_all.csv")))
    return {
        (r["method"], r["condition"], float(r["mutation_rate"])): float(r["mean"])
        for r in rows if r["metric"] == "AUPRC"
    }


def figure(auprc, conds: dict, xlabel: str, stem: str):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), sharey=True)
    for ax, rate in zip(axes, RATES):
        for method, label in MODELS:
            xs = [bp for c, bp in conds.items() if (method, c, rate) in auprc]
            ys = [auprc[(method, c, rate)] for c in conds if (method, c, rate) in auprc]
            ax.plot(xs, ys, marker="o", label=label, color=COLORS[method],
                    linestyle=LINESTYLES[method], linewidth=3.5, markersize=12)
        ax.set_title(rf"{rate * 100:.0f}\% Mutation Rate", fontsize=30)
        ax.set_xticks(list(conds.values()))
        ax.set_xlim(min(conds.values()) - 5, max(conds.values()) + 5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlabel, fontsize=30)
        ax.tick_params(labelsize=22)
        ax.grid(True)
    axes[0].set_ylabel("Average AUPRC", fontsize=30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=20, loc="lower center", ncol=len(MODELS),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"{stem}.png")


def main():
    auprc = load()
    figure(auprc, CROP, "Query Length (bp)", "crop_auprc_vs_bases")
    figure(auprc, COVER, "Aligned Bases of 128 bp Query", "coverage_auprc_vs_bases")


if __name__ == "__main__":
    main()
