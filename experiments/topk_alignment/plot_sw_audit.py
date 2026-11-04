"""Histograms of SW identity over the top-k hits (sw_audit_<label>.parquet).

One figure per label: a 2 x 3 grid (rows: locus / window target; columns:
mutation rate). Each panel is the identity histogram of hits whose local
alignment covers >= min_qcov of the query, split into hits from a
ground-truth accession (what the benchmark credits) and hits from any other
accession (what it counts as false positives); hits failing the coverage
filter are the leftmost bar so the junk fraction is visible on the same axis.
A comparison figure overlays the other-accession histograms of all labels at
each rate.

  uv run --no-sync python experiments/topk_alignment/plot_sw_audit.py --labels <label> [<label> ...]
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jsonargparse import auto_cli

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"relevant": "#2a78d6", "other": "#eb6834"}   # categorical slots 1, 2
EDGES = np.round(np.arange(0.50, 1.0001, 0.025), 3)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _panel(ax, df: pl.DataFrame, tgt: str, min_qcov: float, title: str):
    idc, cov = f"{tgt}_identity", f"{tgt}_qcov"
    w = (EDGES[1] - EDGES[0]) * 0.42
    xs = (EDGES[:-1] + EDGES[1:]) / 2
    fail_x = EDGES[0] - 2.5 * (EDGES[1] - EDGES[0])
    for j, (name, rel) in enumerate((("relevant", True), ("other", False))):
        sub = df.filter(pl.col("acc_relevant") == rel)
        n = sub.height
        ok = sub.filter(pl.col(cov) >= min_qcov)[idc].to_numpy()
        ok = np.clip(ok, EDGES[0], EDGES[-1] - 1e-9)
        h, _ = np.histogram(ok, bins=EDGES)
        off = (j - 0.5) * w
        ax.bar(xs + off, h, width=w, color=SERIES[name], linewidth=0)
        ax.bar(fail_x + off, n - len(ok), width=w, color=SERIES[name], linewidth=0, hatch="////", edgecolor=SURFACE)
        lab = f"{'ground-truth' if rel else 'other'} accession: {n:,} hits, {len(ok) / max(n, 1):.0%} pass"
        ax.plot([], [], color=SERIES[name], linewidth=6, label=lab)
    ax.axvline(EDGES[0] - 1.2 * (EDGES[1] - EDGES[0]), color=GRID, linewidth=0.8)
    ticks = [fail_x] + list(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks(ticks, ["cov<50%"] + [f"{t:.0%}" for t in ticks[1:]])
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=INK2)
    _style(ax)


def plot(labels: list[str], in_dir: Path = HERE, out_dir: Path = HERE, min_qcov: float = 0.5,
         rates: list[float] = [0.0, 0.05, 0.1]):
    """Write sw_audit_<label>.png per label and sw_audit_compare.png.

    Args:
        labels: index labels whose sw_audit_<label>.parquet exist in in_dir.
        in_dir: where the parquets are.
        out_dir: where the PNGs go.
        min_qcov: query-coverage threshold.
        rates: mutation rates (columns).
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    frames = {lab: pl.read_parquet(in_dir / f"sw_audit_{lab}.parquet") for lab in labels}
    for lab, df in frames.items():
        fig, axes = plt.subplots(2, len(rates), figsize=(5.2 * len(rates), 7.4), facecolor=SURFACE)
        for c, rate in enumerate(rates):
            for r, tgt in enumerate(("locus", "win")):
                sub = df.filter(pl.col("rate") == rate)
                t = f"{rate:.0%} mutation, target = {'window + flank' if tgt == 'locus' else 'index window only'}"
                _panel(axes[r, c], sub, tgt, min_qcov, t)
                if c == 0:
                    axes[r, c].set_ylabel("hits", color=INK2, fontsize=8)
                if r == 1:
                    axes[r, c].set_xlabel("identity over aligned region (gaps count)", color=INK2, fontsize=8)
        fig.suptitle(f"{lab}: Smith-Waterman identity of the exact top-100 vector hits, sra500 (hits with query coverage >= {min_qcov:.0%})",
                     fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"sw_audit_{lab}.png", dpi=150)
        plt.close(fig)

    # comparison: other-accession hits, locus target, one line per label
    if len(labels) > 1:
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        fig, axes = plt.subplots(1, len(rates), figsize=(5.2 * len(rates), 3.8), facecolor=SURFACE, sharey=True)
        xs = (EDGES[:-1] + EDGES[1:]) / 2
        for c, rate in enumerate(rates):
            ax = axes[c]
            for k, lab in enumerate(labels):
                sub = frames[lab].filter((pl.col("rate") == rate) & ~pl.col("acc_relevant"))
                ok = sub.filter(pl.col("locus_qcov") >= min_qcov)["locus_identity"].to_numpy()
                h, _ = np.histogram(np.clip(ok, EDGES[0], EDGES[-1] - 1e-9), bins=EDGES)
                ax.plot(xs, h, color=palette[k], linewidth=2, marker="o", markersize=3,
                        label=f"{lab.replace('locale@', '')}: {len(ok):,}/{sub.height:,} pass")
            ax.set_title(f"{rate:.0%} mutation: other-accession hits passing coverage, window + flank", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("identity over aligned region", color=INK2, fontsize=8)
            ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            _style(ax)
        axes[0].set_ylabel("hits", color=INK2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "sw_audit_compare.png", dpi=150)
        plt.close(fig)
    print("wrote", [str(p) for p in sorted(out_dir.glob("sw_audit_*.png"))])


if __name__ == "__main__":
    auto_cli(plot, as_positional=False)
