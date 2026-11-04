"""Smith-Waterman audit of the saved top-k vector hits on sra500 (2026-11-02).

For every (query, hit) pair in <results_dir>/hits/mut<rate>.parquet the hit
vector is mapped to its contig window through the row->contig sidecar
(experiments/projhead_failures/window_recall.py; built here if missing), the
window sequence is fetched from the bundle's Logan contigs, and the mutated
query (the sequence that was actually searched) is locally aligned to it on
both strands with Biopython's PairwiseAligner (match 2, mismatch -3, gap open
-5, gap extend -2: blastn-like). Two targets per hit:
  win    = the 128/256 bp window the vector was computed from
  locus  = the window extended by <flank> bp of contig on each side (flank =
           query length by default), so a read that straddles two windows can
           still be fully aligned to the locus the vector points at.
  null   = the locus of a random hit of a DIFFERENT query (same rate), the
           calibration for what the coverage filter lets through by chance.
Per hit we record identity (identities / alignment columns, gaps included),
query coverage ((qend - qstart) / qlen), and whether the hit accession /
contig is in the query's ground truth. Output: one parquet per results dir
and a printed summary; plotting lives in plot_sw_audit.py.

  srun -A m5408_g -C cpu -q interactive -N 1 -c 128 -t 60 \
      uv run --no-sync python experiments/topk_alignment/sw_audit.py \
      --index_dir benchmark/indexes/sra500/<label> \
      --results_dir benchmark/results/sra500/<label>/exact/top100
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from Bio.Align import PairwiseAligner
from jsonargparse import auto_cli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIDECAR_DIR = ROOT / "experiments" / "projhead_failures"
sys.path.insert(0, str(SIDECAR_DIR))
from window_recall import BUNDLE, sidecar  # noqa: E402

_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _fetch_contigs(args):
    """(acc, fasta, ordinals) -> (acc, {ordinal: seq}); scans FASTA records in order."""
    acc, fasta, ordinals = args
    want = set(ordinals)
    stop = max(want)
    out = {}
    ordinal = -1
    grab = False
    with open(fasta) as f:
        for line in f:
            if line[0] == ">":
                ordinal += 1
                if ordinal > stop:
                    break
                grab = ordinal in want
                if grab:
                    hid = line[1:].split()[0]
                    assert hid == f"{acc}_{ordinal}", (hid, acc, ordinal)
                    out[ordinal] = []
            elif grab:
                out[ordinal].append(line.strip())
    missing = want - out.keys()
    if missing:
        raise ValueError(f"{acc}: contigs {sorted(missing)[:5]}... not found in {fasta}")
    return acc, {k: "".join(v) for k, v in out.items()}


def _make_aligner():
    return PairwiseAligner(mode="local", match_score=2, mismatch_score=-3,
                           open_gap_score=-5, extend_gap_score=-2)


def _best_local(aligner, q: str, qrc: str, t: str):
    """Best local alignment of q or revcomp(q) against t -> dict of stats (None if no alignment)."""
    best = None
    for strand, qq in (("+", q), ("-", qrc)):
        alns = aligner.align(qq, t)
        if len(alns) == 0 or alns.score <= 0:
            continue
        a = alns[0]
        if best is not None and a.score <= best["score"]:
            continue
        c = a.counts()
        cols = c.aligned + c.gaps
        qs, qe = int(a.coordinates[0, 0]), int(a.coordinates[0, -1])
        ts, te = int(a.coordinates[1, 0]), int(a.coordinates[1, -1])
        best = {"score": float(a.score), "strand": strand, "identities": int(c.identities),
                "mismatches": int(c.mismatches), "gaps": int(c.gaps), "aln_cols": int(cols),
                "qstart": qs, "qend": qe, "tstart": ts, "tend": te}
    return best


def _align_chunk(tasks):
    """tasks: list of (key, qseq, win_seq, locus_seq, null_seq) -> list of (key, win, locus, null stats)."""
    aligner = _make_aligner()
    out = []
    for key, q, win, locus, null in tasks:
        qrc = revcomp(q)
        out.append((key, _best_local(aligner, q, qrc, win), _best_local(aligner, q, qrc, locus),
                    _best_local(aligner, q, qrc, null)))
    return out


def _flatten(prefix: str, st: dict | None, qlen: int) -> dict:
    if st is None:
        return {f"{prefix}_score": 0.0, f"{prefix}_strand": None, f"{prefix}_identity": None,
                f"{prefix}_qcov": 0.0, f"{prefix}_identities": 0, f"{prefix}_aln_cols": 0,
                f"{prefix}_gaps": 0, f"{prefix}_qstart": None, f"{prefix}_qend": None,
                f"{prefix}_tstart": None, f"{prefix}_tend": None}
    return {f"{prefix}_score": st["score"], f"{prefix}_strand": st["strand"],
            f"{prefix}_identity": st["identities"] / st["aln_cols"],
            f"{prefix}_qcov": (st["qend"] - st["qstart"]) / qlen,
            f"{prefix}_identities": st["identities"], f"{prefix}_aln_cols": st["aln_cols"],
            f"{prefix}_gaps": st["gaps"], f"{prefix}_qstart": st["qstart"], f"{prefix}_qend": st["qend"],
            f"{prefix}_tstart": st["tstart"], f"{prefix}_tend": st["tend"]}


def run(index_dir: Path, results_dir: Path, out_dir: Path = HERE, rates: list[float] = [0.0, 0.05, 0.1],
        flank: int | None = None, workers: int = 120, chunk: int = 200, min_qcov: float = 0.5, limit_queries: int | None = None):
    """Align every saved top-k hit to its window and locus; write <label>.parquet.

    Args:
        index_dir: dense index dir (config.json + meta.parquet); its name is the label.
        results_dir: results dir holding hits/mut<rate>.parquet.
        out_dir: where sw_audit_<label>.parquet goes.
        rates: mutation rates to audit.
        flank: contig bp added on each side of the window for the locus target (None = query length).
        workers: process pool size for FASTA scans and alignments.
        chunk: hits per alignment task.
        min_qcov: query-coverage threshold used in the printed summary.
        limit_queries: audit only the first N queries per rate (smoke test).
    """
    t0 = time.time()
    index_dir, results_dir, out_dir = Path(index_dir), Path(results_dir), Path(out_dir)
    label = index_dir.name
    side_json = SIDECAR_DIR / f"{label}.sidecar.json"
    if not side_json.exists() or not json.load(open(side_json))["complete"]:
        print(f"building sidecar for {label} ...", flush=True)
        sidecar(index_dir, out_dir=SIDECAR_DIR, workers=workers)
    side = json.load(open(side_json))
    W = side["identity"]["max_seq_len"]
    contig = np.load(SIDECAR_DIR / f"{label}.contig.u32.npy", mmap_mode="r")
    start = np.load(SIDECAR_DIR / f"{label}.start.u32.npy", mmap_mode="r")

    # 1. hit table: one row per (rate, query, rank)
    rows = []
    need: dict[str, set[int]] = {}
    queries = {}
    for rate in rates:
        q = pl.read_parquet(BUNDLE / f"queries_mut{rate:.2f}.parquet")
        truth = {r["query_id"]: (set(r["contig_accession"]), set(r["contig_id"]), r["query_sequence"])
                 for r in q.iter_rows(named=True)}
        hits = pl.read_parquet(results_dir / "hits" / f"mut{rate:.2f}.parquet").select("query_id", "hits")
        if limit_queries:
            hits = hits.head(limit_queries)
        for r in hits.iter_rows(named=True):
            qid = r["query_id"]
            accs, cids, qseq = truth[qid]
            queries[(rate, qid)] = qseq
            vids = np.array([h["vector_id"] for h in r["hits"]], dtype=np.int64)
            ords, sts = contig[vids], start[vids]
            for rank, (h, o, s) in enumerate(zip(r["hits"], ords, sts)):
                assert o != 0xFFFFFFFF, (qid, h)
                acc = h["accession"]
                need.setdefault(acc, set()).add(int(o))
                rows.append({"rate": rate, "query_id": qid, "rank": rank, "cosine": h["score"],
                             "vector_id": int(h["vector_id"]), "accession": acc, "ordinal": int(o),
                             "win_start": int(s), "qlen": len(qseq),
                             "acc_relevant": acc in accs, "contig_relevant": f"{acc}_{o}" in cids})
    print(f"{len(rows):,} hits over {len(queries)} (rate, query) pairs; {sum(map(len, need.values())):,} contigs "
          f"from {len(need)} accessions to fetch ({time.time() - t0:.0f}s)", flush=True)

    # 2. contig sequences
    jobs = [(acc, str(BUNDLE / "logan_accessions" / acc / f"{acc}.contigs.fa"), sorted(o)) for acc, o in need.items()]
    seqs: dict[tuple[str, int], str] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for acc, d in ex.map(_fetch_contigs, jobs, chunksize=1):
            for o, s in d.items():
                seqs[(acc, o)] = s
    print(f"fetched {len(seqs):,} contigs ({time.time() - t0:.0f}s)", flush=True)

    # 3. alignments (null decoy: locus of a random hit of another query at the same rate)
    rng = np.random.default_rng(0)
    loci = []
    tasks = []
    for i, r in enumerate(rows):
        c = seqs[(r["accession"], r["ordinal"])]
        ws = r["win_start"]
        qseq = queries[(r["rate"], r["query_id"])]
        fl = r["qlen"] if flank is None else flank
        win = c[ws: ws + W]
        locus = c[max(0, ws - fl): ws + W + fl]
        r["contig_len"] = len(c)
        r["locus_start"] = max(0, ws - fl)
        loci.append(locus)
        tasks.append([i, qseq, win, locus])
    by_rate = {}
    for i, r in enumerate(rows):
        by_rate.setdefault(r["rate"], []).append(i)
    for rate, idx in by_rate.items():
        idx = np.array(idx)
        perm = rng.permutation(idx)
        for i, j in zip(idx, perm):
            while rows[j]["query_id"] == rows[i]["query_id"]:
                j = rng.choice(idx)
            tasks[i].append(loci[j])
    chunks = [tasks[i: i + chunk] for i in range(0, len(tasks), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_align_chunk, chunks):
            for i, st_win, st_loc, st_null in res:
                rows[i].update(_flatten("win", st_win, rows[i]["qlen"]))
                rows[i].update(_flatten("locus", st_loc, rows[i]["qlen"]))
                rows[i].update(_flatten("null", st_null, rows[i]["qlen"]))
    df = pl.from_dicts(rows)
    out = out_dir / f"sw_audit_{label}.parquet"
    df.write_parquet(out)
    print(f"wrote {out} ({df.height:,} rows, {time.time() - t0:.0f}s)", flush=True)
    summarize(df, min_qcov=min_qcov, label=label)


def summarize(df: pl.DataFrame, min_qcov: float = 0.5, label: str = ""):
    with pl.Config(tbl_cols=30, tbl_width_chars=240, float_precision=3, tbl_rows=60):
        for tgt in ("locus", "win", "null"):
            idc, cov = f"{tgt}_identity", f"{tgt}_qcov"
            print(f"\n=== {label} target={tgt}: hits with query coverage >= {min_qcov:.0%} (per rate x accession relevance)")
            g = (df.with_columns((pl.col(cov) >= min_qcov).alias("pass"))
                   .group_by("rate", "acc_relevant").agg(
                       pl.len().alias("hits"), pl.col("pass").mean().alias("frac_pass"),
                       pl.col(idc).filter(pl.col("pass")).median().alias("id_median"),
                       pl.col(idc).filter(pl.col("pass")).quantile(0.1).alias("id_p10"),
                       pl.col(idc).filter(pl.col("pass")).quantile(0.9).alias("id_p90"),
                       (pl.col(idc).filter(pl.col("pass")) >= 0.75).mean().alias("frac_pass_id>=75"),
                       (pl.col(idc).filter(pl.col("pass")) >= 0.95).mean().alias("frac_pass_id>=95"),
                   ).sort("rate", "acc_relevant"))
            print(g)
        print(f"\n=== {label}: identity histogram (locus target, qcov >= {min_qcov:.0%}), counts per rate; 'fail' = below coverage")
        breaks = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        labels = ["<0.60", "0.60-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
        h = (df.with_columns(pl.when(pl.col("locus_qcov") < min_qcov).then(pl.lit("fail"))
                               .otherwise(pl.col("locus_identity").cut(breaks, labels=labels).cast(pl.Utf8)).alias("bin"))
               .group_by("rate", "acc_relevant", "bin").agg(pl.len().alias("n"))
               .pivot(on="bin", index=["rate", "acc_relevant"], values="n").fill_null(0).sort("rate", "acc_relevant"))
        h = h.select(["rate", "acc_relevant"] + [c for c in ["fail"] + labels if c in h.columns])
        print(h)


def summarize_file(path: Path, min_qcov: float = 0.5):
    """Re-print the summary for a saved sw_audit parquet."""
    summarize(pl.read_parquet(path), min_qcov=min_qcov, label=Path(path).stem)


if __name__ == "__main__":
    auto_cli([run, summarize_file], as_positional=False)
