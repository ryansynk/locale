"""Dilution ceiling probe on sra500 (2026-10-14).

For every relevant (query, accession) alignment in queries.parquet, embed four
index-side windows cut from the aligned contig and score each against the
query (max over both strands, like the both_strands search protocol):

  exact  the aligned span contig[s:e]              (window/read ratio ~1)
  w128c  a 128 bp window centred on the alignment  (ideal sub-span placement)
  w128s  best-covering 128 bp window at stride 64  (a realistic W=128 index)
  w256   the window the endwin50 index actually holds (W=256, stride 106,
         end window when the tail > 50 bp; whole contig when <= 256 bp)

w256 is the validation: the saved exact top-100 search's per-accession score
must be >= it, and equal whenever this window is the accession's best vector.

Outputs, under this directory:
  pairs.parquet     one row per (alignment, mutation rate) with all scores
  summary.txt       score / margin tables by ratio bin and the AUPRC ceilings

Run on a GPU node from the benchmark directory:
  srun -A m5408_g -C gpu -q interactive -N 1 --gpus=1 -t 30 \
    bash -c 'cd benchmark && uv run python ../experiments/dilution_probe/probe.py'
"""

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
from src.dense_index import _end_tail, reverse_complement  # noqa: E402
from src.encoders import DenseEncoder  # noqa: E402

BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
CKPT = "/pscratch/sd/r/rsynk/locale/checkpoints/8vqiabk9/checkpoint5859.pth.tar"
RESULTS = BENCH / "results/sra500_minlen"
ENCODER_LABEL = "locale@8vqiabk9-endwin50"
RATES = [0.0, 0.05, 0.1]
W, STEP, END_TAIL = 256, 106, 50
RATIO_EDGES = [1.05, 1.3, 1.7, 2.0, 2.5, 3.0]


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


def best_cover(starts, w, s, e):
    """Window start among `starts` (each of width w) covering most of [s, e)."""
    best, best_cov = None, -1
    for st in starts:
        cov = max(0, min(e, st + w) - max(s, st))
        if cov > best_cov:
            best, best_cov = st, cov
    return best, best_cov


def windows_for(contig, s, e):
    n = len(contig)
    L = e - s
    out = {"exact": contig[s:e]}
    # w128c: centred 128 bp window (the alignment itself when it is longer)
    if L >= 128:
        out["w128c"] = contig[s:e]
    elif n <= 128:
        out["w128c"] = contig
    else:
        st = min(max((s + e) // 2 - 64, 0), n - 128)
        out["w128c"] = contig[st : st + 128]
    # w128s: stride-64 windows plus an end window
    if n <= 128:
        out["w128s"], cov = contig, L
    else:
        starts = list(range(0, n - 128 + 1, 64))
        if (n - 128) % 64:
            starts.append(n - 128)
        st, cov = best_cover(starts, 128, s, e)
        out["w128s"] = contig[st : st + 128]
    out["cov128s"] = cov
    # w256: what the endwin50 index holds
    if n <= W:
        out["w256"], cov = contig, L
    else:
        starts = list(range(0, n - W + 1, STEP))
        if _end_tail(n, W, STEP) > END_TAIL:
            starts.append(n - W)
        st, cov = best_cover(starts, W, s, e)
        out["w256"] = contig[st : st + W]
    out["cov256"] = cov
    return out


def main():
    t0 = time.time()
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
    )
    # One alignment per (query, accession) in this bundle; keep the best anyway.
    aln = aln.sort("identity", descending=True).unique(
        ["query_id", "contig_accession"], keep="first"
    )
    print(
        f"{len(aln)} relevant alignments over {aln['contig_accession'].n_unique()} accessions"
    )

    # --- pull the contigs -------------------------------------------------
    groups = [
        (acc, set(g["contig_id"].to_list()))
        for acc, g in aln.group_by("contig_accession")
    ]
    groups = [(a[0] if isinstance(a, tuple) else a, w) for a, w in groups]
    contigs = {}
    with ProcessPoolExecutor(max_workers=min(64, len(groups))) as ex:
        for d in ex.map(extract_contigs, groups, chunksize=4):
            contigs.update(d)
    print(f"extracted {len(contigs)} contigs in {time.time() - t0:.0f}s")

    rows = []
    for r in aln.iter_rows(named=True):
        c = contigs[r["contig_id"]]
        assert len(c) == r["contig_len"], (r["contig_id"], len(c), r["contig_len"])
        s, e = r["s"], r["e"]
        assert 0 <= s < e <= len(c), (r["contig_id"], s, e, len(c))
        w = windows_for(c, s, e)
        rows.append({**r, **w, "aln_len": e - s})
    pairs = pl.DataFrame(rows)

    # --- embed ------------------------------------------------------------
    enc = DenseEncoder(
        EncoderConfig(name="locale", checkpoint_path=CKPT, batch_size=256)
    )
    kinds = ["exact", "w128c", "w128s", "w256"]
    uniq = sorted({seq for k in kinds for seq in pairs[k].to_list()})
    print(f"embedding {len(uniq)} unique windows")
    win_emb = enc.encode(uniq).float().cpu()
    win_idx = {s: i for i, s in enumerate(uniq)}

    truth = pr.raw_read_oracle_results(q).select("query_id", "results")
    truth_pairs = (
        truth.explode("results")
        .unnest("results")
        .select(
            "query_id",
            pl.col("accession").alias("contig_accession"),
            pl.lit(1).alias("rel"),
        )
    )
    accs = (BUNDLE / "accs.txt").read_text().splitlines()
    runs = pr.load_results([RESULTS]).filter(
        (pl.col("encoder") == ENCODER_LABEL) & (pl.col("index") == "exact-minlen0")
    )

    out_frames, summary = [], []
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
        rlen = dict(zip(qids, [len(x) for x in fwd]))

        run = runs.filter(pl.col("mutation_rate") == rate)
        hits = (
            run.select("query_id", "results")
            .explode("results")
            .unnest("results")
            .rename({"accession": "contig_accession"})
            .join(truth_pairs, on=["query_id", "contig_accession"], how="left")
            .with_columns(pl.col("rel").fill_null(0))
        )
        irrel = (
            hits.filter(pl.col("rel") == 0)
            .group_by("query_id")
            .agg(pl.col("score").max().alias("best_irrel"))
        )
        saved = hits.filter(pl.col("rel") == 1).select(
            "query_id", "contig_accession", pl.col("score").alias("saved_score")
        )

        p = pairs.with_columns(pl.lit(rate).alias("mutation_rate"))
        qi = torch.tensor([qpos[x] for x in p["query_id"].to_list()])
        for k in kinds:
            wi = torch.tensor([win_idx[x] for x in p[k].to_list()])
            sf = (qf_emb[qi] * win_emb[wi]).sum(1)
            sr = (qr_emb[qi] * win_emb[wi]).sum(1)
            p = p.with_columns(pl.Series(f"score_{k}", torch.maximum(sf, sr).numpy()))
        p = (
            p.with_columns(
                pl.Series("rlen", [rlen[x] for x in p["query_id"].to_list()])
            )
            .with_columns(
                (pl.min_horizontal(pl.col("contig_len"), W) / pl.col("rlen")).alias(
                    "ratio"
                )
            )
            .join(irrel, on="query_id", how="left")
            .join(saved, on=["query_id", "contig_accession"], how="left")
        )
        out_frames.append(p.drop(kinds))

        # validation: saved per-accession score >= our w256 score, mostly equal
        d = (p["saved_score"] - p["score_w256"]).to_numpy()
        summary.append(
            f"\n##### mutation rate {rate} #####\n"
            f"validation w256 vs saved accession score: saved - probe  min {d.min():+.4f}  "
            f"median {np.median(d):+.4f}  frac |diff|<1e-3: {(np.abs(d) < 1e-3).mean():.3f}  "
            f"frac saved < probe - 1e-3: {(d < -1e-3).mean():.4f}"
        )
        pb = p.with_columns(pl.col("ratio").cut(RATIO_EDGES).alias("rbin"))
        tab = (
            pb.group_by("rbin")
            .agg(
                pl.len().alias("n"),
                pl.col("rlen").median().alias("rlen_med"),
                *[pl.col(f"score_{k}").mean().alias(f"mean_{k}") for k in kinds],
                *[
                    (pl.col(f"score_{k}") > pl.col("best_irrel"))
                    .mean()
                    .alias(f"above_{k}")
                    for k in kinds
                ],
            )
            .sort("rbin")
        )
        with pl.Config(tbl_cols=20, tbl_width_chars=200, float_precision=3):
            summary.append(
                "score by ratio bin (mean cosine, then fraction above the query's best irrelevant):\n"
                + str(tab)
            )

        # AUPRC ceiling: replace each relevant accession's score by the probe score
        base = per_query_ap(run, truth, accs)["ap"].mean()
        line = f"AUPRC: saved run {base:.3f}"
        for k in kinds:
            rep = p.select(
                "query_id", "contig_accession", pl.col(f"score_{k}").alias("new")
            )
            new_hits = (
                hits.join(rep, on=["query_id", "contig_accession"], how="left")
                .with_columns(pl.coalesce("new", "score").alias("score"))
                .select(
                    "query_id", pl.col("contig_accession").alias("accession"), "score"
                )
                .with_columns(pl.struct("accession", "score").alias("results"))
                .group_by("query_id")
                .agg(pl.col("results"))
            )
            line += f" | {k} {per_query_ap(new_hits, truth, accs)['ap'].mean():.3f}"
        summary.append(
            line + "   (irrelevant scores held fixed: ignores any floor rise)"
        )

    allp = pl.concat(out_frames)
    allp.write_parquet(HERE / "pairs.parquet")
    text = "\n".join(summary)
    (HERE / "summary.txt").write_text(text)
    print(text)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
