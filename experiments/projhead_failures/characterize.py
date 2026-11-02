"""Per-query failure characterization for the projection-head checkpoints on
sra500v2 (2026-10-27). For every run x rate: per-query average precision
(same truth as plot_results.compute_auprc: every accession in
contig_accession is relevant), R-precision, best relevant / irrelevant
scores, and query attributes from the bundle. Writes per_query.parquet next
to this file and prints the breakdowns.

  cd benchmark && uv run --no-sync python ../experiments/projhead_failures/characterize.py
"""
import math
import sys
from collections import Counter
from pathlib import Path

import edlib
import numpy as np
import pandas
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import plot_results as pr  # noqa: E402

OUT = Path(__file__).resolve().parent
B = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
RUNS = {
    "w128": "results/sra500/locale@pc6t1x8j-proj-w128s64-endwin30-minlen100/exact/top100",
    "pc6t1x8j": "results/sra500/locale@pc6t1x8j-proj-endwin50-minlen100/exact/top100",
    "qpms6u7j": "results/sra500/locale@qpms6u7j-proj-endwin50-minlen100/exact/top100",
    "8vqiabk9": "results/sra500_minlen/locale@8vqiabk9-endwin50/exact-minlen100/top100",
}
RATES = ["0.00", "0.05", "0.10"]
FAIL_AP = 0.5
# FOCUS is the run whose failures are characterized, BASE the run it is
# compared against query by query (defaults: the 128/64 index vs the 256/106
# index of the same checkpoint; `characterize.py pc6t1x8j 8vqiabk9` for the
# 2026-10-27 projection-head vs paper-checkpoint analysis).
FOCUS = sys.argv[1] if len(sys.argv) > 1 else "w128"
BASE = sys.argv[2] if len(sys.argv) > 2 else "pc6t1x8j"

accs = sorted(B.joinpath("accs.txt").read_text().splitlines())
acc_idx = {a: i for i, a in enumerate(accs)}
q0 = pl.read_parquet(B / "queries_mut0.00.parquet")
qm = {r: pl.read_parquet(B / f"queries_mut{r}.parquet") for r in RATES}


def entropy_kmer(s, k=3):
    c = Counter(s[i : i + k] for i in range(len(s) - k + 1))
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values())


def longest_run(s):
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


# ---- query attributes ------------------------------------------------------
feat = []
for row in q0.iter_rows(named=True):
    s = row["query_sequence"]
    per_acc = {}
    for acc, ident, clen, iv in zip(
        row["contig_accession"], row["identity"], row["contig_len"], row["aln_interval_contig"]
    ):
        aln = iv["aln_end_index_contig"] - iv["aln_start_index_contig"]
        d = per_acc.setdefault(acc, {"ident": 0, "clen": 0, "aln": 0})
        d["ident"] = max(d["ident"], ident)
        d["clen"] = max(d["clen"], clen)
        d["aln"] = max(d["aln"], aln)
    ed = {
        r: edlib.align(s, qm[r].filter(pl.col("query_id") == row["query_id"])["query_sequence"][0])["editDistance"]
        for r in RATES[1:]
    }
    feat.append(
        {
            "query_id": row["query_id"],
            "strand": row["strand"],
            "qlen": len(s),
            "nrel": len(per_acc),
            "ident_max": max(d["ident"] for d in per_acc.values()),
            "ident_min": min(d["ident"] for d in per_acc.values()),
            "aln_max": max(d["aln"] for d in per_acc.values()),
            "aln_cov": max(d["aln"] for d in per_acc.values()) / len(s),
            "rel_contig_lt100": sum(d["clen"] < 100 for d in per_acc.values()),
            "rel_contig_lt256": sum(d["clen"] < 256 for d in per_acc.values()),
            "contig_len_max": max(d["clen"] for d in per_acc.values()),
            "gc": (s.count("G") + s.count("C")) / len(s),
            "entropy3": entropy_kmer(s),
            "max_base_frac": max(Counter(s).values()) / len(s),
            "homopolymer": longest_run(s),
            "edit_0.05": ed["0.05"],
            "edit_0.10": ed["0.10"],
        }
    )
feat = pl.from_dicts(feat)

# ---- per-query scores ------------------------------------------------------
truth = {}
for row in q0.iter_rows(named=True):
    y = np.zeros(len(accs), dtype=int)
    for a in row["contig_accession"]:
        y[acc_idx[a]] = 1
    truth[row["query_id"]] = y

rows = []
for run, d in RUNS.items():
    for r in RATES:
        res = pl.read_parquet(Path(d) / f"mut{r}.parquet")
        for row in res.iter_rows(named=True):
            y = truth[row["query_id"]]
            sc = np.full(len(accs), pr.DEFAULT_SCORE)
            for p in row["results"]:
                sc[acc_idx[p["accession"]]] = p["score"]
            scored = sc > pr.DEFAULT_SCORE
            rel_sc = sc[(y == 1) & scored]
            irr_sc = sc[(y == 0) & scored]
            order = np.argsort(-sc)
            order = order[scored[order]]
            first_rel = next((i + 1 for i, j in enumerate(order) if y[j]), None)
            rows.append(
                {
                    "run": run,
                    "rate": float(r),
                    "query_id": row["query_id"],
                    "ap": average_precision_score(y, sc),
                    "rprec": pr.r_precision_per_query(y, sc),
                    "n_scored": int(scored.sum()),
                    "n_rel_scored": int(len(rel_sc)),
                    "best_rel": float(rel_sc.max()) if len(rel_sc) else None,
                    "best_irr": float(irr_sc.max()) if len(irr_sc) else None,
                    "first_rel_rank": first_rel,
                }
            )
pq = pl.from_dicts(rows).join(feat, on="query_id")
pq.write_parquet(OUT / "per_query.parquet")
pd = pq.to_pandas()
pd["fail"] = pd["ap"] < FAIL_AP
pd["miss"] = pd["n_rel_scored"] == 0

print(f"\n=== mean AP (check vs print_results) and failure counts (AP < {FAIL_AP}), n=500")
print(pd.groupby(["run", "rate"]).agg(ap=("ap", "mean"), fail=("fail", "sum"), miss=("miss", "sum")).round(3))

# structural vs noise-induced at 10%
print("\n=== 10% failures split by the same query's 0% outcome")
for run in RUNS:
    a0 = pd[(pd.run == run) & (pd.rate == 0.0)].set_index("query_id")["fail"]
    a10 = pd[(pd.run == run) & (pd.rate == 0.1)].set_index("query_id")["fail"]
    print(f"{run}: fail@10%={a10.sum():3d}  of which already failing @0%={int((a10 & a0).sum()):3d}, "
          f"noise-induced={int((a10 & ~a0).sum()):3d}; fail@0%={a0.sum()}")

print(f"\n=== overlap with the {BASE} run at each rate")
for r in [0.0, 0.05, 0.1]:
    b = pd[(pd.run == BASE) & (pd.rate == r)].set_index("query_id")["fail"]
    for run in [k for k in RUNS if k != BASE]:
        n = pd[(pd.run == run) & (pd.rate == r)].set_index("query_id")["fail"]
        print(f"rate {r}: {run}: both fail={int((n & b).sum()):3d}  rescued (base fail, new ok)={int((~n & b).sum()):3d}  "
              f"newly failing={int((n & ~b).sum()):3d}  both ok={int((~n & ~b).sum()):3d}")

FEATS = ["qlen", "nrel", "ident_max", "ident_min", "aln_max", "aln_cov", "rel_contig_lt100",
         "rel_contig_lt256", "contig_len_max", "gc", "entropy3", "max_base_frac", "homopolymer", "edit_0.10"]
print(f"\n=== {FOCUS} @10%: medians, failed vs succeeded queries")
x = pd[(pd.run == FOCUS) & (pd.rate == 0.1)]
med = x.groupby("fail")[FEATS].median().T
med.columns = ["ok", "fail"]
print(med.round(3))
print("\nstrand failure rate:", x.groupby("strand")["fail"].mean().round(3).to_dict())

print(f"\n=== {FOCUS} @10%: Spearman(AP, feature)")
for f in FEATS:
    rho, p = spearmanr(x[f], x["ap"])
    print(f"  {f:18s} rho={rho:+.3f}  p={p:.1e}")

print(f"\n=== {FOCUS} @10%: failure rate by binned feature")
for f, bins in [("qlen", [0, 100, 120, 140, 151]), ("ident_max", [0, 0.9, 0.95, 0.98, 1.01]),
                ("nrel", [0, 1, 3, 10, 1000]), ("aln_cov", [0, 0.6, 0.8, 0.95, 1.01]),
                ("entropy3", [0, 4.5, 5.0, 5.5, 7]), ("edit_0.10", [0, 10, 14, 18, 100])]:
    g = x.groupby(np.digitize(x[f], bins[1:-1]))
    for i, grp in g:
        print(f"  {f:10s} bin {bins[i]}-{bins[i+1]}: n={len(grp):3d} fail={grp['fail'].mean():.2f} mean AP={grp['ap'].mean():.2f}")

print("\n=== still-failing-in-every-run @10% (query, features)")
allfail = pd[pd.rate == 0.1].groupby("query_id")["fail"].all()
ids = allfail[allfail].index
cols = ["query_id", "strand", "qlen", "nrel", "ident_max", "aln_cov", "rel_contig_lt100", "entropy3", "homopolymer", "edit_0.10"]
print(x[x.query_id.isin(ids)][cols + ["ap", "n_rel_scored", "best_rel", "best_irr"]].sort_values("ap").to_string(index=False))

print(f"\n=== {FOCUS} @10%: noise-induced failures (ok at 0%) sorted by AP")
a0 = pd[(pd.run == FOCUS) & (pd.rate == 0.0)].set_index("query_id")["ap"]
xx = x.copy(); xx["ap_0"] = xx.query_id.map(a0)
ni = xx[(xx.fail) & (xx.ap_0 >= FAIL_AP)]
print(ni[cols + ["ap_0", "ap", "n_rel_scored", "best_rel", "best_irr"]].sort_values("ap").head(25).to_string(index=False))
print("noise-induced:", len(ni), "median best_rel", ni.best_rel.median(), "median best_irr", ni.best_irr.median())

# ---- mechanism: headroom by read length, FOCUS vs BASE ------------------------
print(f"\n=== headroom by read length: clean true score, noisy true score, background (best irrelevant), {FOCUS} vs {BASE}")
pd["lb"] = pandas.cut(pd.qlen, [0, 100, 120, 140, 400], labels=["<100", "100-120", "120-140", "140+"])
for run in [BASE, FOCUS]:
    a0 = pd[(pd.run == run) & (pd.rate == 0.0)].set_index("query_id")
    for r in [0.05, 0.1]:
        a = pd[(pd.run == run) & (pd.rate == r)].set_index("query_id")
        d = pandas.DataFrame({"rel0": a0.best_rel, "rel": a.best_rel, "irr": a.best_irr, "irr0": a0.best_irr,
                              "fail": a.fail, "miss": a.miss, "lb": a.lb, "ap": a.ap})
        d["headroom"] = d.rel - d.irr
        print(f"--- {run} @ {r:.0%} noise")
        print(d.groupby("lb", observed=True)[["rel0", "rel", "irr", "headroom", "ap", "fail", "miss"]].mean().round(3).to_string())
print(f"\n=== {FOCUS} vs {BASE}: per-query AP delta at 10% by read length (positive = {FOCUS} better)")
f = pd[(pd.run == FOCUS) & (pd.rate == 0.1)].set_index("query_id")
b = pd[(pd.run == BASE) & (pd.rate == 0.1)].set_index("query_id")
dd = pandas.DataFrame({"delta": f.ap - b.ap, "lb": f.lb, "qlen": f.qlen, "nrel": f.nrel, "ident_min": f.ident_min,
                       "ap_f": f.ap, "ap_b": b.ap, "rel_f": f.best_rel, "rel_b": b.best_rel, "irr_f": f.best_irr, "irr_b": b.best_irr})
print(dd.groupby("lb", observed=True).agg(n=("delta", "size"), mean_delta=("delta", "mean"), improved=("delta", lambda s: (s > 0.1).mean()),
                                        worsened=("delta", lambda s: (s < -0.1).mean())).round(3).to_string())
print(f"\n=== the {int((dd.delta < -0.1).sum())} queries that got worse by > 0.1 AP (sorted)")
print(dd[dd.delta < -0.1].sort_values("delta")[["qlen", "nrel", "ident_min", "ap_b", "ap_f", "rel_b", "rel_f", "irr_b", "irr_f"]].head(25).round(3).to_string())
