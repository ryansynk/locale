"""Is the noisy true-hit cosine (~0.62 at 10%) mutation loss or window dilution?
(2026-11-02) For every ground-truth (query, contig) pair of sra500 and every
mutation rate, embed with pc6t1x8j (project=True, 128 bp index settings):
  q_mut    the mutated read that is searched (both strands, max taken)
  q_clean  the clean read
  span     the exact aligned contig region (no flanks)
  span128  a 128 bp window centred on the span (clipped to the contig)
  windows  the index's stride-64 128 bp windows overlapping the span
           (+ end window when the tail > 30 bp; whole contig if <= 128 bp)
and record cos(q_mut, q_clean) [mutation only, identical span],
cos(q_clean, span) [natural divergence only], cos(q_clean, best window)
[dilution only], cos(q_mut, span) and cos(q_mut, best window) [both].
GPU node:
  srun ... python experiments/topk_alignment/dilution_vs_mutation.py
"""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, polars as pl, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; BENCH = ROOT / "benchmark"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(BENCH))
from sw_audit import _fetch_contigs, revcomp, BUNDLE  # noqa: E402
from src.encoders import DenseEncoder  # noqa: E402
from src.config import EncoderConfig  # noqa: E402

CKPT = ROOT / "checkpoints/pc6t1x8j/checkpoint21380.pth.tar"
W, STEP, END_TAIL = 128, 64, 30
RATES = [0.0, 0.05, 0.1]

def windows_over(contig_len, s0, e0):
    """start offsets of index windows overlapping [s0, e0)."""
    if contig_len <= W:
        return [0]
    starts = list(range(0, contig_len - W + 1, STEP))
    if (contig_len - W) % STEP > END_TAIL:
        starts.append(contig_len - W)
    return [s for s in starts if min(e0, s + W) > max(s0, s)]

def main():
    t0 = time.time()
    clean = pl.read_parquet(BUNDLE / "queries_mut0.00.parquet").select("query_id", pl.col("query_sequence").alias("q_clean"))
    truth = (pl.read_parquet(BUNDLE / "queries.parquet")
             .select("query_id", "strand", "contig_id", "contig_accession", "identity", "contig_len", "aln_interval_contig")
             .explode("contig_id", "contig_accession", "identity", "contig_len", "aln_interval_contig").unnest("aln_interval_contig")
             .with_columns(pl.col("contig_id").str.split("_").list.last().cast(pl.Int64).alias("ordinal")))
    need = {}
    for r in truth.iter_rows(named=True):
        need.setdefault(r["contig_accession"], set()).add(r["ordinal"])
    jobs = [(a, str(BUNDLE / "logan_accessions" / a / f"{a}.contigs.fa"), sorted(o)) for a, o in need.items()]
    seqs = {}
    with ProcessPoolExecutor(32) as ex:
        for a, d in ex.map(_fetch_contigs, jobs, chunksize=1):
            for o, s in d.items():
                seqs[(a, o)] = s
    print(f"fetched {len(seqs)} contigs ({time.time()-t0:.0f}s)", flush=True)

    enc = DenseEncoder(EncoderConfig(name="locale", checkpoint_path=str(CKPT), pooling="mean", max_seq_len=W,
                                     chunk_overlap=W - STEP, end_window_min_tail_bp=END_TAIL, min_seq_len=100,
                                     project=True, batch_size=512, device="cuda"))
    cache = {}
    def embed_all(strings):
        todo = [s for s in set(strings) if s not in cache]
        for i in range(0, len(todo), 4096):
            b = todo[i:i + 4096]
            E = enc.encode(b).float().cpu().numpy()
            cache.update(zip(b, E))
    rows = []
    for rate in RATES:
        qm = pl.read_parquet(BUNDLE / f"queries_mut{rate:.2f}.parquet").select("query_id", pl.col("query_sequence").alias("q_mut")).join(clean, on="query_id")
        pairs = truth.join(qm, on="query_id")
        strings = []
        for r in pairs.iter_rows(named=True):
            c = seqs[(r["contig_accession"], r["ordinal"])]
            s0, e0 = r["aln_start_index_contig"], r["aln_end_index_contig"]
            mid = (s0 + e0) // 2; a = max(0, min(mid - W // 2, len(c) - W))
            strings += [r["q_mut"], revcomp(r["q_mut"]), r["q_clean"], revcomp(r["q_clean"]), c[s0:e0], c[a:a + W]] + [c[s:s + W] for s in windows_over(len(c), s0, e0)]
        embed_all(strings)
        for r in pairs.iter_rows(named=True):
            c = seqs[(r["contig_accession"], r["ordinal"])]
            s0, e0 = r["aln_start_index_contig"], r["aln_end_index_contig"]
            mid = (s0 + e0) // 2; a = max(0, min(mid - W // 2, len(c) - W))
            Qm = np.stack([cache[r["q_mut"]], cache[revcomp(r["q_mut"])]]); Qc = np.stack([cache[r["q_clean"]], cache[revcomp(r["q_clean"])]])
            span, s128 = cache[c[s0:e0]], cache[c[a:a + W]]
            ws = windows_over(len(c), s0, e0); Wn = np.stack([cache[c[s:s + W]] for s in ws])
            ov = [min(e0, s + W) - max(s0, s) for s in ws]
            best = int(np.argmax(Qc @ Wn.T))  # best window for the clean query, by cosine
            rows.append({"rate": rate, "query_id": r["query_id"], "contig_id": r["contig_id"], "qlen": len(r["q_mut"]), "span_len": e0 - s0,
                         "truth_identity": r["identity"], "contig_len": len(c), "n_windows": len(ws), "best_overlap_frac": max(ov) / (e0 - s0),
                         "cos_mut_clean": float(np.max(Qm @ Qc.T)),
                         "cos_clean_span": float(np.max(Qc @ span)), "cos_clean_span128": float(np.max(Qc @ s128)), "cos_clean_win": float(np.max(Qc @ Wn.T)),
                         "cos_mut_span": float(np.max(Qm @ span)), "cos_mut_span128": float(np.max(Qm @ s128)), "cos_mut_win": float(np.max(Qm @ Wn.T))})
        print(f"rate {rate}: {pairs.height} pairs ({time.time()-t0:.0f}s)", flush=True)
    df = pl.from_dicts(rows); df.write_parquet(HERE / "dilution_vs_mutation.parquet")
    summarize(df)

def summarize(df):
    pl.Config.set_float_precision(3); pl.Config.set_tbl_width_chars(240); pl.Config.set_tbl_rows(40)
    cols = ["cos_mut_clean", "cos_clean_span", "cos_clean_span128", "cos_clean_win", "cos_mut_span", "cos_mut_span128", "cos_mut_win"]
    print("\nmean cosine per rate (mut_clean = mutation only; clean_win = dilution only; mut_win = what the index scores):")
    print(df.group_by("rate").agg(pl.len().alias("pairs"), *[pl.col(c).mean() for c in cols]).sort("rate"))
    print("\nmedians:"); print(df.group_by("rate").agg(*[pl.col(c).median() for c in cols]).sort("rate"))
    print("\nby read length, 10% noise:")
    print(df.filter(pl.col("rate") == 0.1).with_columns(pl.col("qlen").cut([100, 128, 150, 200], labels=["<100", "100-128", "128-150", "150-200", "200+"]).alias("L"))
          .group_by("L").agg(pl.len().alias("pairs"), pl.col("best_overlap_frac").mean(), *[pl.col(c).mean() for c in cols]).sort("L"))
    print("\nexact-identity pairs only (truth identity == 1), means:")
    print(df.filter(pl.col("truth_identity") >= 0.999).group_by("rate").agg(pl.len().alias("pairs"), *[pl.col(c).mean() for c in cols]).sort("rate"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(pl.read_parquet(HERE / "dilution_vs_mutation.parquet"))
    else:
        main()
