"""Window-size / stride grid probe on sra500 (recall@10 ceilings, no build).

For every relevant (query, accession) alignment in the sra500 bundle, cut the
window that each candidate index grid (W, S) would store around that
alignment -- the same rules as benchmark/src/dense_index.py's iter_chunks:
contigs shorter than MIN_LEN get no vector, contigs of at most W bp are one
whole window, longer ones are cut at stride S with the end-aligned window
added when the uncovered tail exceeds END_TAIL bp -- and keep the window that
covers most of the alignment. Embed it with the paper checkpoint, score it
against the mutated read (max over both strands, like the search protocol),
and substitute that score for the accession's score in the saved exact
top-100 run at cutoff 100. Irrelevant accessions keep their saved scores, so
every number here is a ceiling that ignores the floor rise a denser grid
causes; the full builds measure that.

Reported per grid and noise level: recall@1/10/50 over accessions, AUPRC
(plot_results' per-query AP), and the vector count relative to the current
grid, computed exactly from the per-vector contig-length map of the endwin50
index (experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32).

Rows: "saved" (the baseline run as is), "exact" (the aligned span itself:
perfect windows), the current grid (256, 106) as a validation -- its scores
must never exceed the saved ones -- and the candidate grids in GRIDS.

Run on a GPU node from the benchmark directory (about a minute):
  srun -A m5408_g -C gpu -q interactive -N 1 --gpus=1 -t 30 \
    bash -c 'cd benchmark && uv run python ../experiments/window_grid/probe.py'

--count_only prints the vector multipliers and exits (CPU only).

Outputs, next to this file: summary.txt (the tables), results.csv (one row
per grid x rate), pairs.parquet (every alignment's score under every grid).
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE.parent / "seqlen_cutoff"))
import plot_results as pr  # noqa: E402
from score_sweep import per_query_ap  # noqa: E402
from src.config import EncoderConfig  # noqa: E402
from src.dense_index import _chunks_for_length, _end_tail, reverse_complement  # noqa: E402

BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
CKPT = "/pscratch/sd/r/rsynk/locale/checkpoints/8vqiabk9/checkpoint5859.pth.tar"
RESULTS = BENCH / "results/sra500_minlen"
ENCODER_LABEL = "locale@8vqiabk9-endwin50"
BASE_INDEX = "exact-minlen100"  # saved run the scores are substituted into
LENGTHS = HERE.parent / "seqlen_cutoff" / "sra500_endwin50_contig_len.u32"
RATES = [0.0, 0.05, 0.1]
RECALL_KS = [1, 10, 50]

CURRENT = (256, 106)  # (W, stride) of the saved run; chunk_overlap 150
GRIDS = [(192, 96), (128, 64), (128, 96), (128, 128), (192, 64)]
END_TAIL = 50  # end_window_min_tail_bp, as in the endwin50 index
MIN_LEN = 100  # min_seq_len of the deployed configs and of BASE_INDEX


def grid_name(w, s):
    return f"w{w}s{s}"


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def window_starts(n, w, s):
    """Start offsets iter_chunks would store for a contig of n > w bp."""
    starts = list(range(0, n - w + 1, s))
    if _end_tail(n, w, s) > END_TAIL:
        starts.append(n - w)
    return starts


def best_cover(starts, w, s, e):
    """Window start among `starts` (each of width w) covering most of [s, e)."""
    best, best_cov = None, -1
    for st in starts:
        cov = max(0, min(e, st + w) - max(s, st))
        if cov > best_cov:
            best, best_cov = st, cov
    return best, best_cov


def grid_window(contig, s, e, w, step):
    """(window sequence, covered bp) the (w, step) grid stores for alignment
    [s, e) of `contig`, or (None, 0) when the contig is below MIN_LEN."""
    n = len(contig)
    if n < MIN_LEN:
        return None, 0
    if n <= w:
        return contig, e - s
    st, cov = best_cover(window_starts(n, w, step), w, s, e)
    return contig[st : st + w], cov


def extract_contigs(args):
    """(accession, set of contig ids) -> {contig_id: sequence}; Logan FASTA
    files are one sequence line per record, id is the first header token."""
    acc, wanted = args
    out = {}
    path = BUNDLE / "logan_accessions" / acc / f"{acc}.contigs.fa"
    with open(path) as f:
        grab = None
        for line in f:
            if line[0] == ">":
                cid = line[1:].split(maxsplit=1)[0]
                grab = cid if cid in wanted else None
            elif grab is not None:
                out[grab] = out.get(grab, "") + line.strip().upper()
    missing = wanted - out.keys()
    if missing:
        raise RuntimeError(
            f"{acc}: {len(missing)} contigs not found, e.g. {sorted(missing)[:3]}"
        )
    return out


# --------------------------------------------------------------------------
# vector counts
# --------------------------------------------------------------------------
def contig_length_histogram():
    """{contig length: number of contigs} for sra500, recovered from the
    per-vector contig-length map of the endwin50 index: a contig of length n
    contributed _chunks_for_length(n, 256, 106, END_TAIL) rows to it."""
    lengths = np.fromfile(LENGTHS, dtype="<u4")
    uniq, rows = np.unique(lengths, return_counts=True)
    hist = {}
    w, s = CURRENT
    for n, r in zip(uniq.tolist(), rows.tolist()):
        per = _chunks_for_length(n, w, s, END_TAIL, None)
        assert r % per == 0, (n, r, per)
        hist[n] = r // per
    return hist


def vector_counts(hist, grids):
    """Total vectors each (W, S) grid stores at MIN_LEN, from the histogram."""
    out = {}
    for w, s in grids:
        out[(w, s)] = sum(
            cnt * _chunks_for_length(n, w, s, END_TAIL, MIN_LEN)
            for n, cnt in hist.items()
        )
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def recall_at_k(run, truth_sets, subst=None):
    """Mean recall@k over queries after replacing relevant accessions' scores
    with `subst[(query_id, accession)]` where present."""
    rec = {k: [] for k in RECALL_KS}
    for row in run.select("query_id", "results").iter_rows(named=True):
        qid = row["query_id"]
        T = truth_sets.get(qid)
        if not T:
            continue
        res = [(x["accession"], x["score"]) for x in row["results"]]
        if subst:
            res = [(a, subst.get((qid, a), sc)) for a, sc in res]
        ranked = [a for a, _ in sorted(res, key=lambda x: -x[1])]
        for k in rec:
            rec[k].append(len(T & set(ranked[:k])) / len(T))
    return {k: float(np.mean(v)) for k, v in rec.items()}


def substituted_run(run, subst):
    """The saved run with relevant accessions' scores replaced (for AUPRC)."""
    if not subst:
        return run.select("query_id", "results")
    rep = pl.DataFrame(
        {
            "query_id": [k[0] for k in subst],
            "accession": [k[1] for k in subst],
            "new": list(subst.values()),
        }
    )
    return (
        run.select("query_id", "results")
        .explode("results")
        .unnest("results")
        .join(rep, on=["query_id", "accession"], how="left")
        .with_columns(pl.coalesce("new", "score").alias("score"))
        .select("query_id", pl.struct("accession", "score").alias("results"))
        .group_by("query_id")
        .agg(pl.col("results"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count_only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # --- vector multipliers -------------------------------------------
    hist = contig_length_histogram()
    all_grids = [CURRENT] + GRIDS
    counts = vector_counts(hist, all_grids)
    base_count = counts[CURRENT]
    mult = {g: counts[g] / base_count for g in all_grids}
    lines = [
        f"vectors at min_seq_len {MIN_LEN} (current grid {grid_name(*CURRENT)} = "
        f"{base_count:,} = 1.00):"
    ]
    for g in all_grids:
        lines.append(f"  {grid_name(*g):9s} {counts[g]:>13,}  x{mult[g]:.3f}")
    print("\n".join(lines))
    if args.count_only:
        return

    # --- alignments ------------------------------------------------------
    q = pl.read_parquet(BUNDLE / "queries.parquet")
    aln = (
        q.select(
            "query_id",
            "contig_id",
            "contig_accession",
            "identity",
            "contig_len",
            "aln_interval_contig",
        )
        .explode(
            "contig_id",
            "contig_accession",
            "identity",
            "contig_len",
            "aln_interval_contig",
        )
        .unnest("aln_interval_contig")
        .rename({"aln_start_index_contig": "s", "aln_end_index_contig": "e"})
        .sort("identity", descending=True)
        .unique(["query_id", "contig_accession"], keep="first")
    )
    print(
        f"{len(aln)} relevant alignments over {aln['contig_accession'].n_unique()} accessions"
    )

    groups = [
        (acc[0] if isinstance(acc, tuple) else acc, set(g["contig_id"].to_list()))
        for acc, g in aln.group_by("contig_accession")
    ]
    contigs = {}
    with ProcessPoolExecutor(max_workers=min(64, len(groups))) as ex:
        for d in ex.map(extract_contigs, groups, chunksize=4):
            contigs.update(d)
    print(f"extracted {len(contigs)} contigs in {time.time() - t0:.0f}s")

    kinds = ["exact"] + [grid_name(*g) for g in all_grids]
    rows = []
    for r in aln.iter_rows(named=True):
        c = contigs[r["contig_id"]]
        assert len(c) == r["contig_len"], (r["contig_id"], len(c), r["contig_len"])
        s, e = r["s"], r["e"]
        assert 0 <= s < e <= len(c), (r["contig_id"], s, e, len(c))
        row = {**r, "aln_len": e - s, "exact": c[s:e] if len(c) >= MIN_LEN else None}
        for w, st in all_grids:
            win, cov = grid_window(c, s, e, w, st)
            row[grid_name(w, st)] = win
            row[f"cov_{grid_name(w, st)}"] = cov
        rows.append(row)
    pairs = pl.DataFrame(rows)

    # --- embed every unique window once ------------------------------------
    from src.encoders import DenseEncoder  # noqa: E402  (CUDA init here)

    enc = DenseEncoder(
        EncoderConfig(name="locale", checkpoint_path=CKPT, batch_size=256)
    )
    uniq = sorted({x for k in kinds for x in pairs[k].to_list() if x is not None})
    print(f"embedding {len(uniq)} unique windows")
    win_emb = enc.encode(uniq).float().cpu()
    win_idx = {x: i for i, x in enumerate(uniq)}

    truth = pr.raw_read_oracle_results(q).select("query_id", "results")
    truth_sets = {
        r["query_id"]: {t["accession"] for t in r["results"]}
        for r in truth.iter_rows(named=True)
    }
    accs = (BUNDLE / "accs.txt").read_text().splitlines()
    runs = pr.load_results([RESULTS]).filter(
        (pl.col("encoder") == ENCODER_LABEL) & (pl.col("index") == BASE_INDEX)
    )
    if runs.is_empty():
        raise RuntimeError(f"no saved run {ENCODER_LABEL}/{BASE_INDEX} under {RESULTS}")

    out_frames, summary, table = [], lines, []
    for rate in RATES:
        qf = pl.read_parquet(BUNDLE / f"queries_mut{rate:.2f}.parquet").select(
            "query_id", "query_sequence"
        )
        qids = qf["query_id"].to_list()
        fwd = qf["query_sequence"].str.to_uppercase().to_list()
        rc = [reverse_complement(x) for x in fwd]
        qe = enc.encode(fwd + rc).float().cpu()
        qf_emb, qr_emb = qe[: len(fwd)], qe[len(fwd) :]
        qpos = {qid: i for i, qid in enumerate(qids)}

        run = runs.filter(pl.col("mutation_rate") == rate)
        p = pairs.with_columns(pl.lit(rate).alias("mutation_rate"))
        qi = torch.tensor([qpos[x] for x in p["query_id"].to_list()])
        for k in kinds:
            seqs = p[k].to_list()
            have = torch.tensor([x is not None for x in seqs])
            wi = torch.tensor([win_idx[x] if x is not None else 0 for x in seqs])
            sf = (qf_emb[qi] * win_emb[wi]).sum(1)
            sr = (qr_emb[qi] * win_emb[wi]).sum(1)
            sc = torch.maximum(sf, sr).numpy().astype(np.float64)
            sc[~have.numpy()] = np.nan  # no vector for this contig under the grid
            p = p.with_columns(pl.Series(f"score_{k}", sc))

        # saved per-accession score of every relevant pair (validation)
        saved = (
            run.select("query_id", "results")
            .explode("results")
            .unnest("results")
            .rename({"accession": "contig_accession"})
            .select(
                "query_id", "contig_accession", pl.col("score").alias("saved_score")
            )
        )
        p = p.join(saved, on=["query_id", "contig_accession"], how="left")
        out_frames.append(p.drop(kinds))

        cur = grid_name(*CURRENT)
        d = (p["saved_score"] - p[f"score_{cur}"]).drop_nulls().drop_nans().to_numpy()
        summary.append(
            f"\n##### mutation rate {rate} #####\n"
            f"validation {cur} vs saved accession score: saved - probe  min {d.min():+.4f}  "
            f"median {np.median(d):+.4f}  frac |diff|<1e-3: {(np.abs(d) < 1e-3).mean():.3f}  "
            f"frac saved < probe - 1e-3: {(d < -1e-3).mean():.4f}"
        )

        base_rec = recall_at_k(run, truth_sets)
        base_ap = per_query_ap(run.select("query_id", "results"), truth, accs)[
            "ap"
        ].mean()
        table.append(
            {
                "rate": rate,
                "kind": "saved",
                "vectors_x": 1.0,
                **{f"R@{k}": v for k, v in base_rec.items()},
                "auprc": base_ap,
            }
        )
        for k in kinds:
            sub = {
                (r[0], r[1]): r[2]
                for r in p.select(
                    "query_id", "contig_accession", f"score_{k}"
                ).iter_rows()
                if r[2] is not None and not np.isnan(r[2])
            }
            rec = recall_at_k(run, truth_sets, sub)
            apv = per_query_ap(substituted_run(run, sub), truth, accs)["ap"].mean()
            g = (
                None
                if k == "exact"
                else next(gg for gg in all_grids if grid_name(*gg) == k)
            )
            table.append(
                {
                    "rate": rate,
                    "kind": k,
                    "vectors_x": float("nan") if g is None else mult[g],
                    **{f"R@{kk}": v for kk, v in rec.items()},
                    "auprc": apv,
                }
            )

    res = pl.DataFrame(table)
    res.write_csv(HERE / "results.csv")
    pl.concat(out_frames).write_parquet(HERE / "pairs.parquet")

    # --- the table that matters: one row per grid, R@10 and AUPRC per rate ---
    wide = res.pivot(
        index=["kind", "vectors_x"], on="rate", values=["R@10", "auprc"]
    ).sort("vectors_x", nulls_last=False)
    order = ["saved", "exact"] + [grid_name(*g) for g in all_grids]
    wide = (
        wide.with_columns(
            pl.col("kind")
            .map_elements(lambda k: order.index(k), return_dtype=pl.Int64)
            .alias("_o")
        )
        .sort("_o")
        .drop("_o")
    )
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=3, tbl_rows=50):
        summary.append(
            "\nrecall@10 and AUPRC ceilings by grid (irrelevant scores held fixed):\n"
            + str(wide)
        )
        summary.append("\nall recall levels:\n" + str(res))
    text = "\n".join(summary)
    (HERE / "summary.txt").write_text(text)
    print(text)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
