"""How stable is a query's junk (non-relevant) top-100 across mutation rates?

Reads the sw_audit parquet (one row per rate x query x rank) and reports, per
query, |junk(rate a) & junk(rate b)| / min(|junk a|, |junk b|). Run on login.
"""
import sys
import numpy as np
import polars as pl

path = sys.argv[1] if len(sys.argv) > 1 else (
    "experiments/topk_alignment/sw_audit_locale@pc6t1x8j-proj-w128s64-endwin30-minlen100.parquet")
df = pl.read_parquet(path)
junk = df.filter(~pl.col("acc_relevant"))
qs = sorted(junk["query_id"].unique().to_list())

def sets(frame, key="vector_id"):
    return {(r, q): set(g[key].to_list()) for (r, q), g in frame.group_by(["rate", "query_id"])}

def overlap(s, ra, rb):
    ov = []
    for q in qs:
        a, b = s.get((ra, q)), s.get((rb, q))
        if a and b:
            ov.append(len(a & b) / min(len(a), len(b)))
    ov = np.array(ov)
    return ov.mean(), np.median(ov), (ov > 0.5).mean(), (ov == 0).mean(), len(ov)

s100 = sets(junk)
s20 = sets(junk.filter(pl.col("rank") < 20))
for ra, rb in [(0.0, 0.05), (0.0, 0.1), (0.05, 0.1)]:
    m, med, gt, z, n = overlap(s100, ra, rb)
    print(f"junk top-100 {ra} vs {rb}: mean {m:.3f} median {med:.3f} frac>0.5 {gt:.2f} frac=0 {z:.2f} n={n}")
m, med, gt, z, n = overlap(s20, 0.0, 0.1)
print(f"junk top-20 0.0 vs 0.1: mean {m:.3f} frac>0.5 {gt:.2f}")
r0 = junk.filter((pl.col("rate") == 0.0) & (pl.col("rank") < 10)).select("query_id", "vector_id")
r10 = junk.filter(pl.col("rate") == 0.1).select("query_id", "vector_id").with_columns(pl.lit(True).alias("hit"))
print("0%% top-10 junk still in 10%% top-100: %.3f" % r0.join(r10, on=["query_id", "vector_id"], how="left")["hit"].fill_null(False).mean())
