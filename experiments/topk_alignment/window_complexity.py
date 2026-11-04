"""Sequence complexity of junk top-k windows vs random index windows (2026-11-02).
Fetches 128 bp windows for a sample of junk hits (qcov<50%), alignable hits,
and random index rows (via the sidecar), and reports 3-mer entropy, longest
homopolymer run, DUST-like triplet score, GC, and window length. Compute node."""
import sys, math, json
from collections import Counter
from pathlib import Path
import numpy as np, polars as pl
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from sw_audit import _fetch_contigs, BUNDLE, SIDECAR_DIR
lab="locale@pc6t1x8j-proj-w128s64-endwin30-minlen100"; W=128; N=3000
df=pl.read_parquet(HERE/f"sw_audit_{lab}.parquet")
side=np.load(SIDECAR_DIR/f"{lab}.contig.u32.npy",mmap_mode="r"); st=np.load(SIDECAR_DIR/f"{lab}.start.u32.npy",mmap_mode="r")
meta=pl.read_parquet(f"benchmark/indexes/sra500/{lab}/meta.parquet").sort("start_row"); acc_start=meta["start_row"].to_numpy(); accs=meta["srr_id"].to_list()
rng=np.random.default_rng(0)
groups={}
for name,sub in [("junk",df.filter(pl.col("locus_qcov")<0.5)),("alignable",df.filter(pl.col("locus_qcov")>=0.5))]:
    s=sub.unique("vector_id").sample(N,seed=0); groups[name]=[(r["accession"],r["ordinal"],r["win_start"]) for r in s.iter_rows(named=True)]
ridx=np.sort(rng.choice(len(side),N,replace=False)); ai=np.searchsorted(acc_start,ridx,side="right")-1
groups["random"]=[(accs[a],int(side[i]),int(st[i])) for i,a in zip(ridx,ai)]
need={}
for g in groups.values():
    for a,o,_ in g: need.setdefault(a,set()).add(o)
jobs=[(a,str(BUNDLE/"logan_accessions"/a/f"{a}.contigs.fa"),sorted(o)) for a,o in need.items()]
seqs={}
with ProcessPoolExecutor(120) as ex:
    for a,d in ex.map(_fetch_contigs,jobs,chunksize=1):
        for o,s in d.items(): seqs[(a,o)]=s
def ent(s,k=3):
    c=Counter(s[i:i+k] for i in range(len(s)-k+1)); n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
def homopol(s):
    best=run=1
    for i in range(1,len(s)):
        run=run+1 if s[i]==s[i-1] else 1; best=max(best,run)
    return best
def dust(s):
    c=Counter(s[i:i+3] for i in range(len(s)-2)); l=len(s)-2; return sum(v*(v-1)/2 for v in c.values())/max(l-1,1)
rows=[]
for name,g in groups.items():
    for a,o,ws in g:
        w=seqs[(a,o)][ws:ws+W]; rows.append({"group":name,"len":len(w),"entropy":ent(w),"homopolymer":homopol(w),"dust":dust(w),"gc":(w.count("G")+w.count("C"))/len(w),"seq":w})
out=pl.from_dicts(rows); out.write_parquet(HERE/"window_complexity.parquet")
pl.Config.set_float_precision(3); pl.Config.set_tbl_width_chars(200)
print(out.group_by("group").agg(pl.len().alias("n"),pl.col("len").mean().alias("win_len"),pl.col("entropy").mean(),pl.col("entropy").quantile(0.1).alias("ent_p10"),(pl.col("entropy")<4.0).mean().alias("frac_ent<4"),pl.col("homopolymer").mean(),(pl.col("homopolymer")>=10).mean().alias("frac_homopol>=10"),pl.col("dust").mean(),(pl.col("dust")>2.0).mean().alias("frac_dust>2"),pl.col("gc").mean(),pl.col("gc").std().alias("gc_sd")).sort("group"))
print("\nlowest-entropy junk windows:")
for r in out.filter(pl.col("group")=="junk").sort("entropy").head(8).iter_rows(named=True): print(f"  ent {r['entropy']:.2f} homopol {r['homopolymer']:2d} dust {r['dust']:.2f} {r['seq'][:90]}")
