"""Sweep IVF search settings on one node with the index loaded once.

Works for both engines: faiss IVF-RaBitQ (use_ivf; --nprobes/--reranks/--qbs
map to ivf_nprobe/ivf_rerank/ivf_qb) and cuVS GPU IVF-PQ (use_ivfpq;
--nprobes/--reranks map to ivfpq_nprobe/ivfpq_rerank, per shard).

run_benchmark reloads the ~230 GB index per invocation, so a grid over
(nprobe, rerank) would spend most of its time in read_index. This loads the
index once (merging the build shards on first use), embeds each rate's
queries once, and for every setting runs the search ``--repeats`` times,
keeping the last as the timed one (the first also warms the page cache for
the rerank preads). Per setting and rate it writes, exactly as run_benchmark
would:

    <results_dir>/<experiment_id>/raw_read_mut_<rate>.parquet   (avg_time =
        embed + scan + rerank + regroup wall time of the timed repeat)
    <topk_hits_dir>/<hits_id>/raw_read_mut_<rate>_topk<k>.parquet

and prints AUPRC / R-precision / Recall@7 (same definitions as
print_results) with the stage timings.

    python ivf_sweep.py --config configs/sra4571/perlmutter_locale_sra4571_ivf.yaml \
        --nprobes 128,256,512,1024 --reranks 300 --rates 0.0,0.05,0.1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import CLI

from ivf_probe import accuracy
from run_benchmark import _annotate_results, query_file_name
from src.config import ExperimentConfig
from src.dense_index import DenseIndex
from src.topk_regroup import regroup_topk_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nprobes", default="256,512")
    ap.add_argument("--reranks", default="300")
    ap.add_argument("--qbs", default="8")
    ap.add_argument("--rates", default="0.0,0.05,0.1")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--threads", type=int, default=0, help="torch/faiss threads (0 = all)")
    args, rest = ap.parse_known_args()

    base_args = ["--config", args.config] + rest
    if args.results_dir:
        base_args += ["--results_dir", args.results_dir]
    cfg = CLI(ExperimentConfig, as_positional=False, args=base_args)
    if args.threads:
        import faiss

        torch.set_num_threads(args.threads)
        faiss.omp_set_num_threads(args.threads)
    index_path = cfg.index_dir / cfg.model.index_suffix
    index = DenseIndex(cfg)
    t0 = time.time()
    index.load(index_path)
    print(f"index ready in {time.time() - t0:.0f}s", flush=True)
    accs = index.indexed_accessions()
    gpu = cfg.model.use_ivfpq  # cuVS IVF-PQ engine; else faiss IVF-RaBitQ
    if gpu:
        args.qbs = "0"  # no qb knob
    raw = pl.read_parquet(Path(cfg.dataset_dir) / "queries.parquet")
    gt = {
        r["query_id"]: set(r["contig_accession"])
        for r in raw.select("query_id", "contig_accession").iter_rows(named=True)
    }

    summary = []
    for rate in [float(r) for r in args.rates.split(",")]:
        queries = pl.read_parquet(Path(cfg.dataset_dir) / query_file_name(rate))
        queries = queries.sample(min(cfg.num_queries, len(queries)), seed=cfg.random_seed)
        for _ in range(2):  # the second embed is the warm (timed) one
            t0 = time.time()
            feats, ranges, _ = index._embed_queries(queries)
            if feats.is_cuda:
                torch.cuda.synchronize()
            embed_s = time.time() - t0
        q = feats.float().cpu().numpy()
        print(f"rate {rate}: embedded {len(q)} chunks in {embed_s:.1f}s", flush=True)
        for qb in [int(x) for x in args.qbs.split(",")]:
            for rerank in [int(x) for x in args.reranks.split(",")]:
                for nprobe in [int(x) for x in args.nprobes.split(",")]:
                    for _ in range(args.repeats):
                        tm = {}
                        t0 = time.time()
                        if gpu:
                            scores, ids = index.ivfpq.search(
                                q, index.top_k, n_probes=nprobe, rerank=rerank, timings=tm,
                                lut_dtype=cfg.model.ivfpq_lut,
                            )
                        else:
                            scores, ids = index.ivf.search(
                                q, index.top_k, nprobe=nprobe, rerank=rerank, qb=qb, timings=tm
                            )
                        hits = index._hits_from_chunk_topk(
                            queries, ranges, torch.from_numpy(scores), torch.from_numpy(ids)
                        )
                        res = regroup_topk_hits(hits, accs)
                        search_s = time.time() - t0
                    total_s = embed_s + search_s
                    run_cfg = CLI(
                        ExperimentConfig,
                        as_positional=False,
                        args=base_args
                        + ["--mutation_rate", str(rate)]
                        + (
                            ["--model.ivfpq_nprobe", str(nprobe), "--model.ivfpq_rerank", str(rerank)]
                            if gpu
                            else [
                                "--model.ivf_nprobe", str(nprobe),
                                "--model.ivf_rerank", str(rerank),
                                "--model.ivf_qb", str(qb),
                            ]
                        ),
                    )
                    index.cfg = run_cfg
                    out = run_cfg.results_dir / run_cfg.model.experiment_id / f"raw_read_mut_{rate}.parquet"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _annotate_results(res, run_cfg, index, index_path, total_s).write_parquet(out)
                    hp = run_cfg.topk_hits_dir / run_cfg.model.hits_id / f"raw_read_mut_{rate}_topk{index.top_k}.parquet"
                    hp.parent.mkdir(parents=True, exist_ok=True)
                    _annotate_results(hits, run_cfg, index, index_path, total_s).write_parquet(hp)
                    acc = accuracy(hits, gt, accs)
                    row = {
                        "rate": rate, "nprobe": nprobe, "rerank": rerank, "qb": qb,
                        "total_s": round(total_s, 2), "embed_s": round(embed_s, 2),
                        "scan_s": round(tm.get("scan_s", 0), 2),
                        "rerank_s": round(tm.get("rerank_s", 0), 2),
                        **{k: round(v, 4) for k, v in acc.items()},
                    }
                    summary.append(row)
                    print(json.dumps(row), flush=True)
    print(pl.DataFrame(summary))


if __name__ == "__main__":
    main()
