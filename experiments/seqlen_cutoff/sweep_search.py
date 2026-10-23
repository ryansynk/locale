"""Exact top-k search of a dense index at several contig-length cutoffs, in one
scan per noise level.

For every cutoff c the result is the exact top-k over the vectors whose source
contig is at least c bp long (the map vector_contig_lengths.py writes), i.e.
what the exact scan would return on an index built with that length filter.
One pass over the fbin serves every cutoff: per block the logits tile is
computed once, then the cutoffs are applied in ascending order by masking the
newly excluded columns to -inf in place, and each cutoff keeps its own running
top-k. Cutoff 0 is the unfiltered scan and must reproduce the config's
existing exact top-k results (check it).

Multi-node exactly as run_benchmark._multi_node_topk_hits: every rank scans a
contiguous range of vector rows for all cutoffs and noise levels and writes
per-cutoff partial hits; rank 0 merges with topk_regroup.merge_topk_hits,
regroups to accessions and writes standard results parquets under

    <sweep_results_dir>/<encoder_label>/exact-minlen<c>/top<k>/mut<rate>.parquet
                                                            (+ config.json, hits/)

so print_results.py scores the whole sweep from <sweep_results_dir>. The
index_size_gb column is the fbin size scaled by the surviving row fraction,
the size of the index a build-time filter at c would have written.

    cd benchmark
    uv run python ../experiments/seqlen_cutoff/sweep_search.py \
        --bench_config configs/sra500/perlmutter_locale_sra500v2_endwin50.yaml \
        --lengths ../experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32 \
        --shard $SLURM_PROCID --num_shards $SLURM_NTASKS

Resumable: a rank whose partials for a rate exist skips that scan; a rate
whose results all exist is skipped by every rank.
"""

import copy
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import CLI, auto_cli
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

import run_benchmark as rb  # noqa: E402
from src.config import (  # noqa: E402
    DONE_FILE,
    DenseMethod,
    ExactIndex,
    ExperimentConfig,
    results_file_name,
    write_config,
)
from src.dense_index import DenseIndex  # noqa: E402
from src.exact import BLOCK_ROWS, ExactEngine, block_loader  # noqa: E402
from src.topk_regroup import merge_topk_hits, regroup_topk_hits  # noqa: E402

DEFAULT_CUTOFFS = [0, 45, 64, 77, 100, 128, 150, 200, 256]
DEFAULT_RATES = [0.0, 0.05, 0.10]


@torch.no_grad()
def scan_topk_by_cutoff(
    engine: ExactEngine,
    query_vecs: np.ndarray,
    lengths: np.ndarray,
    cutoffs: list[int],
    top_k: int,
    vec_range: tuple[int, int] | None = None,
    block_rows: int = BLOCK_ROWS,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """ExactEngine.topk_hits over rows [vec_range), once per cutoff.

    Returns {cutoff: (scores, ids)}, each (n_queries, top_k) score-descending
    with -1 ids / -inf scores as padding, where cutoff c's top-k ranges only
    over rows with lengths[row] >= c. Same block loop and device split as
    ExactEngine.topk_hits; the only addition is the in-place column mask,
    applied cumulatively as the (ascending) cutoffs grow.
    """
    cutoffs = sorted(cutoffs)
    assert engine.all_embeddings is not None
    n_vecs = engine.all_embeddings.shape[0]
    start, end = (0, n_vecs) if vec_range is None else vec_range
    if not (0 <= start <= end <= n_vecs):
        raise ValueError(f"vec_range {vec_range} outside [0, {n_vecs}]")
    qcf_cpu = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float32))
    n_chunks = len(qcf_cpu)
    devices = engine.devices
    n_dev = len(devices)
    per_dev = -(-(end - start) // n_dev) if end > start else 0

    def _scan(dev_idx: int):
        dev = torch.device(devices[dev_idx])
        s = start + dev_idx * per_dev
        e = min(s + per_dev, end)
        q = qcf_cpu.to(dev)
        best_vals = [
            torch.full((n_chunks, top_k), float("-inf"), device=dev, dtype=q.dtype)
            for _ in cutoffs
        ]
        best_ids = [
            torch.full((n_chunks, top_k), -1, device=dev, dtype=torch.int64)
            for _ in cutoffs
        ]
        blocks = range(s, e, block_rows)
        if dev_idx == 0:
            blocks = tqdm(
                blocks,
                desc=f"Exact top-{top_k} x {len(cutoffs)} cutoffs ({n_dev} device(s), "
                f"{end - start:,} vectors)",
            )
        with block_loader(
            engine.fbin_path, engine.all_embeddings, block_rows, dev
        ) as load_block:
            for bs in blocks:
                be = min(bs + block_rows, e)
                logits = q @ load_block(bs, be).T  # (n_chunks, be-bs)
                lens = torch.from_numpy(np.asarray(lengths[bs:be], dtype=np.int64)).to(
                    dev
                )
                k = min(top_k, be - bs)
                prev = None
                for ci, c in enumerate(cutoffs):
                    # Columns excluded by this cutoff and not by the previous.
                    newly = (lens < c) if prev is None else (lens >= prev) & (lens < c)
                    cols = newly.nonzero().squeeze(1)
                    if cols.numel():
                        logits.index_fill_(1, cols, float("-inf"))
                    prev = c
                    vals, idx = torch.topk(logits, k, dim=-1)
                    cand_vals = torch.cat([best_vals[ci], vals], dim=1)
                    cand_ids = torch.cat([best_ids[ci], idx + bs], dim=1)
                    best_vals[ci], pos = torch.topk(cand_vals, top_k, dim=-1)
                    best_ids[ci] = torch.gather(cand_ids, 1, pos)
                    del vals, idx, cand_vals, cand_ids, pos
                del logits, lens, cols
        out = []
        for bv, bi in zip(best_vals, best_ids):
            # A masked column can win a slot in a block with < top_k survivors;
            # its score is -inf, so make it padding like the initial buffer.
            bi = torch.where(torch.isinf(bv), torch.full_like(bi, -1), bi)
            out.append((bv.cpu(), bi.cpu()))
        return out

    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        shard_results = list(pool.map(_scan, range(n_dev)))

    result = {}
    for ci, c in enumerate(cutoffs):
        all_vals = torch.cat([r[ci][0] for r in shard_results], dim=1)
        all_ids = torch.cat([r[ci][1] for r in shard_results], dim=1)
        k = min(top_k, all_vals.shape[1])
        top_vals, top_pos = torch.topk(all_vals, k, dim=-1)
        top_ids = torch.gather(all_ids, 1, top_pos)
        result[c] = (top_vals.numpy(), top_ids.numpy())
    return result


def _wait_for_files(paths, timeout=8 * 3600, poll_interval=15):
    rb._wait_for_files(paths, timeout=timeout, poll_interval=poll_interval)


def main(
    bench_config: str,
    lengths: str,
    cutoffs: list[int] = DEFAULT_CUTOFFS,
    rates: list[float] = DEFAULT_RATES,
    top_k: int = 100,
    sweep_results_dir: str | None = None,
    shard: int | None = None,
    num_shards: int | None = None,
):
    """
    Args:
        bench_config: run_benchmark yaml of the index to search (dense, exact).
        lengths: uint32 contig length per fbin row (vector_contig_lengths.py).
        cutoffs: keep vectors whose contig is at least this long; 0 = all.
        rates: mutation rates (each needs queries_mut<rate>.parquet).
        top_k: vectors per query (exact top-k regroup protocol).
        sweep_results_dir: where the per-cutoff results trees go
            (default: <results_dir>_minlen next to the config's results_dir).
        shard, num_shards: this rank and the node count (default from
            SLURM_PROCID / SLURM_NTASKS, else single node).
    """
    if shard is None:
        shard = int(os.environ.get("SLURM_PROCID", 0))
    if num_shards is None:
        num_shards = int(os.environ.get("SLURM_NTASKS", 1))
    cutoffs = sorted(set(int(c) for c in cutoffs))
    cfg: ExperimentConfig = CLI(
        ExperimentConfig,
        as_positional=False,
        args=[
            "--config",
            bench_config,
            "--shard",
            str(shard),
            "--num_shards",
            str(num_shards),
        ],
    )
    if not (
        isinstance(cfg.model, DenseMethod) and isinstance(cfg.model.index, ExactIndex)
    ):
        raise ValueError("config must be a dense method with the exact engine")
    cfg.model.index.top_k = top_k
    index_path = cfg.model.index_path(cfg.index_dir)
    if not (index_path / DONE_FILE).exists():
        raise FileNotFoundError(f"{index_path} is not .done")
    sweep_dir = (
        Path(sweep_results_dir)
        if sweep_results_dir is not None
        else cfg.results_dir.parent / f"{cfg.results_dir.name}_minlen"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(cfg.dataset_dir).resolve()
    rank_tag = f"[rank {shard}/{num_shards}]"

    index = DenseIndex(cfg)
    index.load(index_path)
    n_vecs = index.num_vectors()
    lengths_arr = np.memmap(lengths, dtype="<u4", mode="r")
    if lengths_arr.shape[0] != n_vecs:
        raise ValueError(
            f"{lengths} has {lengths_arr.shape[0]:,} entries, index has {n_vecs:,} rows"
        )
    survivors = {c: int((np.asarray(lengths_arr) >= c).sum()) for c in cutoffs}
    fbin_gb = (index_path / "embeddings.fbin").stat().st_size / (1024**3)
    meta_gb = (index_path / "meta.parquet").stat().st_size / (1024**3)
    if shard == 0:
        with open(sweep_dir / "sweep.json", "w") as f:
            json.dump(
                {
                    "bench_config": str(Path(bench_config).resolve()),
                    "index_path": str(index_path),
                    "lengths": str(Path(lengths).resolve()),
                    "top_k": top_k,
                    "cutoffs": cutoffs,
                    "rates": rates,
                    "n_vectors": n_vecs,
                    "surviving_vectors": survivors,
                    "surviving_fraction": {c: survivors[c] / n_vecs for c in cutoffs},
                },
                f,
                indent=1,
            )
        for c in cutoffs:
            print(
                f"cutoff {c:>4}: {survivors[c]:,} vectors ({survivors[c] / n_vecs:.1%})"
            )

    def results_dir(c: int) -> Path:
        return sweep_dir / cfg.model.encoder_label / f"exact-minlen{c}" / f"top{top_k}"

    def search_identity(c: int) -> dict:
        return {**rb.run_search_identity(cfg), "min_seq_len": c}

    bounds = [n_vecs * r // num_shards for r in range(num_shards + 1)]
    vec_range = (bounds[shard], bounds[shard + 1])

    for rate in rates:
        out_files = {c: results_dir(c) / results_file_name(rate) for c in cutoffs}
        if all(p.exists() for p in out_files.values()):
            print(f"{rank_tag} rate {rate}: all cutoffs done, skipping")
            continue
        queries = pl.read_parquet(bundle / rb.query_file_name(rate))
        queries = queries.sample(
            min(cfg.num_queries, len(queries)), seed=cfg.random_seed
        )
        partials_dir = sweep_dir / (
            f"partials_minlen_mut{rate:.2f}_n{len(queries)}_seed{cfg.random_seed}"
        )
        partials_dir.mkdir(parents=True, exist_ok=True)

        def partial_path(r: int, c: int) -> Path:
            return partials_dir / f"rank_{r}_of_{num_shards}_minlen{c}.parquet"

        mine = {c: partial_path(shard, c) for c in cutoffs}
        if all(p.exists() for p in mine.values()):
            print(f"{rank_tag} rate {rate}: partials exist, reusing")
        else:
            t0 = time.time()
            print(
                f"{rank_tag} rate {rate}: scanning rows {vec_range[0]:,}-{vec_range[1]:,}"
            )
            q_feats, query_indices, _ = index._embed_queries(queries)
            q = q_feats.float().cpu().numpy()
            per_cutoff = scan_topk_by_cutoff(
                index.engine, q, lengths_arr, cutoffs, top_k, vec_range
            )
            for c, (scores, ids) in per_cutoff.items():
                hits = index._hits_from_chunk_topk(queries, query_indices, scores, ids)
                tmp = mine[c].with_suffix(".parquet.tmp")
                hits.write_parquet(tmp)
                tmp.rename(mine[c])
            print(f"{rank_tag} rate {rate}: scan done in {time.time() - t0:.0f}s")

        if shard != 0:
            continue
        needed = [partial_path(r, c) for r in range(num_shards) for c in cutoffs]
        print(f"{rank_tag} waiting for {num_shards - 1} other rank(s)...")
        _wait_for_files(needed)
        for c in cutoffs:
            hits = merge_topk_hits(
                [pl.read_parquet(partial_path(r, c)) for r in range(num_shards)], top_k
            )
            assert len(hits) == len(queries)
            results = regroup_topk_hits(hits, index.indexed_accessions())
            cfg_c = copy.deepcopy(cfg)
            cfg_c.model.index_label = f"exact-minlen{c}"
            cfg_c.model.search_label = f"top{top_k}"
            cfg_c.mutation_rate = rate  # _annotate_results stamps this column
            size_gb = fbin_gb * survivors[c] / n_vecs + meta_gb
            results = rb._annotate_results(results, cfg_c, index, index_path, -1.0)
            results = results.with_columns(pl.lit(size_gb).alias("index_size_gb"))
            out_dir = results_dir(c)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_config(out_dir, search_identity(c))
            results.write_parquet(out_files[c])
            hits_path = out_dir / "hits" / results_file_name(rate)
            hits_path.parent.mkdir(exist_ok=True)
            rb._annotate_results(hits, cfg_c, index, index_path, -1.0).with_columns(
                pl.lit(size_gb).alias("index_size_gb")
            ).write_parquet(hits_path)
            print(f"{rank_tag} rate {rate} cutoff {c}: wrote {out_files[c]}")
        shutil.rmtree(partials_dir)

    print(
        f"{rank_tag} done; score with:\n  uv run python scripts/print_results.py "
        f"--results_dir {sweep_dir} --raw_read_queries_path {bundle / 'queries.parquet'} "
        f"--accessions {bundle / 'accs.txt'}"
    )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
