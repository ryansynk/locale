"""Can IVF find the exact top-k hits of noised queries? (sra4571 feasibility probe)

For each mutation rate, takes the exact top-1000 hits already saved under
results/<ds>_topk_hits/<...>_exact_bothstrands/, trains spherical k-means
centroids on a row sample, and asks for every hit which probe rank its cell
gets in the query's centroid ordering (min over the two strands). From that:

* coverage of the exact top-1/10/100 hits at each nprobe, and
* "IVF-oracle" accuracy: the exact hit list with every hit that nprobe would
  miss removed, regrouped and scored (AUPRC / R-prec / Recall@7). That is the
  ceiling of IVF + exact rerank at that nprobe, so it picks nlist/nprobe
  before any 2 B-row build.
* the scanned fraction of the index per nprobe, from the sample's cell sizes.

    python ivf_probe.py --config configs/sra4571/perlmutter_locale_sra4571_exact_topk1000.yaml \
        --nlists 16384,65536 --train_rows 6000000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import CLI
from sklearn.metrics import average_precision_score

from run_benchmark import query_file_name
from src.config import ExperimentConfig
from src.dense_index import DenseIndex
from src.ivf_rabitq import assign_rows, probe_ranks, read_rows_by_id, sample_rows, spherical_kmeans
from src.topk_regroup import regroup_topk_hits

NPROBES = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
EVAL_TOP = 100  # hits per query kept for the oracle (the default top_k)


def accuracy(hits: pl.DataFrame, gt: dict[str, set[str]], accs: list[str]) -> dict:
    """AUPRC / R-precision / Recall@7 of a hits frame, as print_results scores it."""
    res = regroup_topk_hits(hits, accs)
    acc_idx = {a: i for i, a in enumerate(accs)}
    ap, rp, r7 = [], [], []
    for qid, results in res.iter_rows():
        if qid not in gt:
            continue
        y_true = np.zeros(len(accs), dtype=int)
        for a in gt[qid]:
            if a in acc_idx:
                y_true[acc_idx[a]] = 1
        if not y_true.sum():
            continue
        y = np.array([r["score"] for r in results])
        ap.append(average_precision_score(y_true, y))
        order = (-y).argsort(kind="stable")
        nr = int(y_true.sum())
        top = order[:nr]
        top = top[y[top] > -2.0]
        rp.append(y_true[top].sum() / nr)
        top7 = order[:7]
        top7 = top7[y[top7] > -2.0]
        r7.append(y_true[top7].sum() / nr)
    return {"auprc": float(np.mean(ap)), "rprec": float(np.mean(rp)), "recall7": float(np.mean(r7))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nlists", default="16384,65536")
    ap.add_argument("--train_rows", type=int, default=6_000_000)
    ap.add_argument("--rates", default="0.0,0.05,0.1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = CLI(ExperimentConfig, as_positional=False, args=["--config", args.config])
    index_path = cfg.index_dir / cfg.model.index_suffix
    fbin = index_path / "embeddings.fbin"
    ivf_dir = index_path / "ivf"
    ivf_dir.mkdir(exist_ok=True)
    out_path = Path(args.out or ivf_dir / "probe_results.json")

    sample_p = ivf_dir / "train_sample.npy"
    if sample_p.exists() and len(np.load(sample_p, mmap_mode="r")) >= args.train_rows:
        train = np.load(sample_p, mmap_mode="r")
        train = np.ascontiguousarray(train[: args.train_rows])
    else:
        t0 = time.time()
        train = sample_rows(fbin, args.train_rows)
        print(f"sampled {len(train):,} rows in {time.time() - t0:.0f}s")
        np.save(sample_p, train)

    index = DenseIndex(cfg)
    index.both_strands = True
    meta = pl.read_parquet(index_path / "meta.parquet")
    accs = meta["srr_id"].to_list()
    starts = meta["start_row"].to_numpy()
    acc_offsets = np.append(starts, starts[-1] + meta["num_rows"][-1])

    hits_dir = cfg.topk_hits_dir / (cfg.model.hits_id)
    raw = pl.read_parquet(Path(cfg.dataset_dir) / "queries.parquet")
    gt = {
        r["query_id"]: set(r["contig_accession"])
        for r in raw.select("query_id", "contig_accession").iter_rows(named=True)
    }

    per_rate = {}
    for rate in [float(r) for r in args.rates.split(",")]:
        hits = pl.read_parquet(hits_dir / f"raw_read_mut_{rate}_topk1000.parquet", columns=["query_id", "hits"])
        queries = pl.read_parquet(Path(cfg.dataset_dir) / query_file_name(rate))
        queries = queries.sample(min(cfg.num_queries, len(queries)), seed=cfg.random_seed)
        queries = queries.join(hits.select("query_id"), on="query_id", how="semi")
        hits = queries.select("query_id").join(hits, on="query_id", how="left")
        feats, ranges, strand = index._embed_queries(queries)
        feats = feats.float().cpu().numpy()
        chunk_q = np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(ranges)])
        chunk_starts = np.array([s for s, _ in ranges])
        top = hits.with_columns(pl.col("hits").list.head(EVAL_TOP))
        vid = np.stack([np.array([h["vector_id"] for h in row]) for row in top["hits"].to_list()])
        t0 = time.time()
        vecs = read_rows_by_id(fbin, vid.ravel())
        dt = time.time() - t0
        print(f"rate {rate}: read {vid.size:,} hit rows in {dt:.1f}s ({vid.size / dt:,.0f} rows/s)")
        per_rate[rate] = dict(queries=queries, hits=top, feats=feats, vid=vid, vecs=vecs, chunk_q=chunk_q, chunk_starts=chunk_starts)

    results = {"nprobes": NPROBES, "baseline": {}, "nlist": {}}
    for rate, pr in per_rate.items():
        results["baseline"][str(rate)] = accuracy(pr["hits"], gt, accs)
        print(f"rate {rate} exact top-{EVAL_TOP}: {results['baseline'][str(rate)]}")

    for nlist in [int(x) for x in args.nlists.split(",")]:
        cent_p = ivf_dir / f"centroids_{nlist}.npy"
        if cent_p.exists():
            cent = np.load(cent_p)
        else:
            t0 = time.time()
            cent = spherical_kmeans(train, nlist, n_iter=20)
            print(f"kmeans nlist={nlist}: {time.time() - t0:.0f}s")
            np.save(cent_p, cent)
        sizes = np.bincount(assign_rows(train, cent), minlength=nlist).astype(np.float64)
        sizes /= sizes.sum()
        res_n = {}
        for rate, pr in per_rate.items():
            n_q, m = pr["vid"].shape
            cells = assign_rows(pr["vecs"], cent).reshape(n_q, m)
            # rank per query chunk (every strand/window), then the best chunk
            rk_chunk = probe_ranks(pr["feats"], cent, np.ascontiguousarray(cells[pr["chunk_q"]]))
            rk = np.minimum.reduceat(rk_chunk, pr["chunk_starts"], axis=0)
            # fraction of the index scanned at each nprobe (both strands, avg)
            order_sizes = []
            with torch.no_grad():
                qt = torch.from_numpy(pr["feats"]).cuda()
                ct = torch.from_numpy(cent).cuda()
                st = torch.from_numpy(sizes).cuda()
                srt = torch.argsort(qt @ ct.T, dim=1, descending=True)
                cum = torch.cumsum(st[srt], dim=1).mean(dim=0).cpu().numpy()
            rows = {}
            for p in NPROBES:
                if p > nlist:
                    continue
                keep = rk < p
                cov = {f"top{t}": float(keep[:, :t].mean()) for t in (1, 10, 100)}
                # oracle hits: drop the missed ones
                kept = [
                    [h for h, k in zip(row, kr) if k]
                    for row, kr in zip(pr["hits"]["hits"].to_list(), keep)
                ]
                ohits = pl.DataFrame(
                    {"query_id": pr["hits"]["query_id"], "hits": kept},
                    schema={"query_id": pl.String, "hits": pr["hits"].schema["hits"]},
                )
                acc = accuracy(ohits, gt, accs)
                rows[p] = {**cov, **acc, "scan_frac": float(cum[p - 1])}
                print(f"nlist {nlist} rate {rate} nprobe {p:5d}: {rows[p]}", flush=True)
            res_n[str(rate)] = rows
        results["nlist"][str(nlist)] = res_n
        out_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
