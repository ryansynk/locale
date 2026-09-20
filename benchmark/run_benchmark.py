import os
import shutil
import sys
import time
from pathlib import Path

import polars as pl
import torch
from huggingface_hub import snapshot_download
from jsonargparse import CLI
from src.config import (
    DenseConfig,
    ExperimentConfig,
    MetagraphConfig,
    MMseqs2Config,
    CentroidConfig,
)
from src.dense_index import DenseIndex
from src.rabitq import build_rabitq_index
from src.centroid_index import CentroidIndex
from src.download_accessions import download_accessions
from src.metagraph_index import MetagraphIndex
from src.mmseqs2_index import MMseqs2Index
from src.topk_regroup import merge_topk_hits, regroup_topk_hits

from lae.training.batcher import Augmenter

DATASETS = {
    "sra50": "rsynk/locale-benchmark-sra50",
    "sra500": "rsynk/locale-benchmark-sra500",
    # Cross-genotype HBV retrieval: query with genotype-D reads, retrieve
    # genotype-B accessions (~11% divergent). Unlike sra50/sra500, ground truth
    # comes from SRA genotype metadata rather than from aligning reads to
    # contigs - the point being that no aligner decides the correct answer.
    # 52 accessions = 47 sra50 distractors + 5 genotype-B targets.
    "sra52viral": "rsynk/locale-benchmark-sra52-viral",
}


def verify_download(accession_ids: list[str], accession_paths: list[Path]):
    # Check all accessions in manifest were found
    manifest_ids = set(accession_ids)
    downloaded_ids = {p.name.removesuffix(".contigs.fa") for p in accession_paths}
    if missing := manifest_ids - downloaded_ids:
        raise RuntimeError(
            f"{len(missing)} accessions missing after download: {missing}"
        )


def apply_mutations(queries: pl.DataFrame, mutation_rate: float) -> pl.DataFrame:
    return queries.with_columns(
        pl.col("query_sequence").map_elements(
            lambda query_seq: Augmenter.augment(query_seq, identity=1 - mutation_rate)
        )
    )


def _wait_for_shards(
    index_path: Path, num_nodes: int, timeout: int = 7200, poll_interval: int = 30
):
    elapsed = 0
    while elapsed < timeout:
        if all(
            (index_path / f"shard_{r}" / ".done").exists() for r in range(num_nodes)
        ):
            return
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Timed out after {timeout}s waiting for all {num_nodes} shards")


def _wait_for_files(paths: list[Path], timeout: int = 7200, poll_interval: int = 15):
    elapsed = 0
    while elapsed < timeout:
        if all(p.exists() for p in paths):
            return
        time.sleep(poll_interval)
        elapsed += poll_interval
    missing = [str(p) for p in paths if not p.exists()]
    raise TimeoutError(f"Timed out after {timeout}s waiting for: {missing}")


def _multi_node_search(
    cfg: ExperimentConfig,
    index: DenseIndex,
    queries: pl.DataFrame,
    index_path: Path,
    node_rank: int,
    num_nodes: int,
) -> pl.DataFrame | None:
    """Shard the streaming dense search across nodes; node 0 merges partials.

    Every node scores a stride of the accessions and atomically writes a
    partial result; node 0 waits for all partials and reassembles each query's
    results in index order, identical to a single-node search. Returns the
    merged results on node 0, None on every other rank. Partials are keyed by
    the search parameters and reused when present, so an interrupted run
    resumes instead of rescoring.
    """
    acc_names = index.indexed_accessions()
    acc_indices = list(range(node_rank, len(acc_names), num_nodes))
    partials_dir = index_path / (
        f"search_partials_mut{cfg.mutation_rate}"
        f"_n{len(queries)}_seed{cfg.random_seed}"
    )
    partials_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partials_dir / f"rank_{node_rank}_of_{num_nodes}.parquet"

    if partial_path.exists():
        print(f"[Node {node_rank}] Partial search result exists, reusing.")
    else:
        partial = index.search(queries, acc_indices=acc_indices)
        tmp_path = partial_path.with_suffix(".parquet.tmp")
        partial.write_parquet(tmp_path)
        tmp_path.rename(partial_path)

    if node_rank != 0:
        print(f"[Node {node_rank}] Partial search saved. Exiting.")
        return None

    partial_paths = [
        partials_dir / f"rank_{r}_of_{num_nodes}.parquet" for r in range(num_nodes)
    ]
    print(f"[Node 0] Waiting for {num_nodes - 1} other partial result(s)...")
    _wait_for_files(partial_paths)

    # Reassemble each query's results in index order, matching what a
    # single-node search returns
    acc_order = pl.DataFrame({"accession": acc_names}).with_row_index("acc_pos")
    merged = (
        pl.concat([pl.read_parquet(p) for p in partial_paths])
        .explode("results")
        .unnest("results")
        .join(acc_order, on="accession")
        .sort("acc_pos")
        .group_by("query_id")
        .agg(pl.struct("accession", "score").alias("results"))
    )
    results = queries.select("query_id").join(merged, on="query_id", how="left")
    assert len(results) == len(queries)
    assert results["results"].null_count() == 0
    shutil.rmtree(partials_dir)
    return results


def _topk_engine_tag(cfg: ExperimentConfig) -> str:
    assert isinstance(cfg.model, DenseConfig)
    return "exact" if cfg.model.exact_search else "rabitq1bit"


def _multi_node_topk_hits(
    cfg: ExperimentConfig,
    index: DenseIndex,
    queries: pl.DataFrame,
    index_path: Path,
    node_rank: int,
    num_nodes: int,
) -> pl.DataFrame | None:
    """Shard a vector-level top-k scan across nodes; node 0 merges the hits.

    Unlike the streaming search, which deals out accessions, this splits the
    index's *vector rows* into num_nodes contiguous, equal ranges (the scan is
    a flat matmul over rows, so this balances the read exactly). Every node
    writes its per-query top-k hits over its range; node 0 unions the partials
    and re-takes each query's global top-k, which equals a single-node scan
    because each partial is complete over a disjoint range. Works for the
    exact fp32 scan and the 1-bit RaBitQ scan alike (index.topk_hits picks
    the engine; with RaBitQ each node loads only its range's codes). Returns
    the merged hits frame on node 0, None elsewhere. Partials are keyed by
    engine, top_k and the search parameters and reused when present, so a
    timed-out run resumes.
    """
    n_vecs = index.num_vectors()
    bounds = [n_vecs * r // num_nodes for r in range(num_nodes + 1)]
    vec_range = (bounds[node_rank], bounds[node_rank + 1])
    partials_dir = index_path / (
        f"{_topk_engine_tag(cfg)}_top{index.top_k}_partials_mut{cfg.mutation_rate}"
        f"_n{len(queries)}_seed{cfg.random_seed}"
    )
    partials_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partials_dir / f"rank_{node_rank}_of_{num_nodes}.parquet"

    if partial_path.exists():
        print(f"[Node {node_rank}] Partial top-k hits exist, reusing.")
    else:
        print(f"[Node {node_rank}] Scanning vectors {vec_range[0]:,}-{vec_range[1]:,}")
        partial = index.topk_hits(queries, vec_range=vec_range)
        tmp_path = partial_path.with_suffix(".parquet.tmp")
        partial.write_parquet(tmp_path)
        tmp_path.rename(partial_path)

    if node_rank != 0:
        print(f"[Node {node_rank}] Partial top-k hits saved. Exiting.")
        return None

    partial_paths = [
        partials_dir / f"rank_{r}_of_{num_nodes}.parquet" for r in range(num_nodes)
    ]
    print(f"[Node 0] Waiting for {num_nodes - 1} other partial result(s)...")
    _wait_for_files(partial_paths)
    hits = merge_topk_hits([pl.read_parquet(p) for p in partial_paths], index.top_k)
    assert len(hits) == len(queries)
    shutil.rmtree(partials_dir)
    return hits


def _annotate_results(
    df: pl.DataFrame, cfg: ExperimentConfig, index, index_path: Path, avg_time: float
) -> pl.DataFrame:
    """Attach the run metadata columns print_results.py's schema expects."""
    df = df.with_columns(pl.lit(index.index_size_gb(index_path)).alias("index_size_gb"))
    df = df.with_columns(pl.lit(avg_time).alias("avg_time"))
    df = df.with_columns(pl.lit(str(cfg.model)).alias("model"))
    df = df.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    df = df.with_columns(pl.lit("raw_read").alias("query_type"))  # legacy
    df = df.with_columns(
        pl.lit(cfg.model.checkpoint, dtype=pl.String).alias("checkpoint")
    )
    df = df.with_columns(pl.lit(cfg.model.max_len, dtype=pl.Int64).alias("max_len"))
    df = df.with_columns(
        pl.lit(cfg.model.checkpoint_step_num, dtype=pl.Int64).alias(
            "checkpoint_step_num"
        )
    )
    df = df.with_columns(pl.lit("stride").alias("chunk_type"))  # legacy
    return df


def main(cfg: ExperimentConfig):
    index_path: Path = cfg.index_dir / cfg.model.index_suffix

    # Download datasets and queries. A dataset_name not in DATASETS is a local
    # dataset: dataset_dir must already hold accs.txt and queries.parquet (the
    # bundle layout finalize_query_dataset.py writes).
    if cfg.dataset_name in DATASETS:
        local_path = snapshot_download(
            DATASETS[cfg.dataset_name],
            repo_type="dataset",
            local_dir=cfg.dataset_dir,
            allow_patterns=["accs.txt", "queries.parquet", "*.json"],
        )
    else:
        local_path = cfg.dataset_dir
        for required in ("accs.txt", "queries.parquet"):
            if not (Path(local_path) / required).exists():
                raise FileNotFoundError(
                    f"Local dataset '{cfg.dataset_name}': {required} not found in "
                    f"{local_path} (local datasets are not downloaded)"
                )
    accession_ids_path: Path = Path(local_path).resolve() / "accs.txt"
    with open(accession_ids_path) as f:
        accession_ids = f.read().splitlines()
    accessions_dir = Path(local_path).resolve() / "logan_accessions"
    queries_path: Path = Path(local_path).resolve() / "queries.parquet"
    accession_paths: list[Path] = sorted(list(accessions_dir.rglob("*.contigs.fa")))
    if not accession_paths:
        download_accessions(accession_ids, accessions_dir)
        accession_paths = sorted(accessions_dir.rglob("*.contigs.fa"))
    verify_download(accession_ids, accession_paths)

    # Truncate after verifying the full manifest downloaded, so a smoke run
    # still catches a broken/incomplete dataset.
    if cfg.max_accessions is not None:
        accession_paths = accession_paths[: cfg.max_accessions]
        print(f"[max_accessions] Indexing only {len(accession_paths)} accessions.")

    queries: pl.DataFrame = pl.read_parquet(queries_path)
    # subsample queries if needed
    queries = queries.sample(min(cfg.num_queries, len(queries)), seed=cfg.random_seed)

    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    elif isinstance(cfg.model, MMseqs2Config):
        index = MMseqs2Index(cfg)
    elif isinstance(cfg.model, CentroidConfig):
        index = CentroidIndex(cfg)
    else:
        raise ValueError("Unknown model config")

    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    # Build index if not already built
    if not (index_path / ".done").exists():
        if num_nodes > 1:
            node_accessions = accession_paths[node_rank::num_nodes]
            shard_path = index_path / f"shard_{node_rank}"
            # A shard marked .done was fully built by a previous run at the
            # same node count; skipping it makes timed-out runs resumable.
            if (shard_path / ".done").exists():
                print(f"[Node {node_rank}/{num_nodes}] Shard already built, skipping.")
            else:
                print(
                    f"[Node {node_rank}/{num_nodes}] Building shard from {len(node_accessions)} accessions..."
                )
                index.build(node_accessions, shard_path)
                index.save(shard_path)
                (shard_path / ".done").touch()

            if node_rank != 0:
                print(f"[Node {node_rank}] Shard saved. Exiting.")
                sys.exit(0)

            print(f"[Node 0] Waiting for {num_nodes - 1} other node(s) to finish...")
            _wait_for_shards(index_path, num_nodes)
            DenseIndex.merge_shards(index_path, num_nodes)
            (index_path / ".done").touch()
        else:
            index.build(accession_paths, index_path)
            index.save(index_path)
            (index_path / ".done").touch()

    if cfg.no_search:
        print("[no_search]: Index built. Exiting.")
        sys.exit(0)

    # The streaming dense search shards accessions across nodes and the exact
    # top-k scan shards vector rows; every other engine (ANN/RaBitQ dense
    # variants, metagraph, mmseqs, centroid) still searches on node 0 alone.
    dense_streaming = isinstance(cfg.model, DenseConfig) and not (
        cfg.model.use_ann or cfg.model.use_rabitq or cfg.model.exact_search
    )
    topk_engine = isinstance(cfg.model, DenseConfig) and (
        cfg.model.exact_search or cfg.model.use_rabitq
    )
    multi_node_search = num_nodes > 1 and (dense_streaming or topk_engine)
    if num_nodes > 1 and not multi_node_search and node_rank != 0:
        print(
            f"[Node {node_rank}] Index built. Skipping search (only node 0 searches)."
        )
        sys.exit(0)

    if isinstance(cfg.model, DenseConfig) and cfg.model.use_rabitq and num_nodes > 1:
        # Quantize the fbin into one shard per node (resumable; a no-op once
        # meta.json exists). Every rank returns with the index complete, so
        # load() below finds it and each node then searches its own row range.
        build_rabitq_index(
            index_path / "embeddings.fbin",
            index_path / "rabitq",
            rank=node_rank,
            num_ranks=num_nodes,
            centroid_sample_rows=cfg.model.rabitq_sample_rows,
        )

    index.load(index_path)
    if cfg.mutation_rate > 0.0:
        # Mutations draw from torch's RNG; seeding makes every node mutate the
        # queries identically, which multi-node search requires for a query's
        # scores to be comparable across accession shards.
        torch.manual_seed(cfg.random_seed)
        queries = apply_mutations(queries, cfg.mutation_rate)

    hits: pl.DataFrame | None = None
    if topk_engine and not cfg.do_timing:
        # Compute-once, re-score-many: the scan yields each query's raw top-k
        # vector hits, which are persisted (rescore_topk.py derives every
        # smaller k from them) and regrouped into the standard results here.
        if multi_node_search:
            hits = _multi_node_topk_hits(
                cfg, index, queries, index_path, node_rank, num_nodes
            )
            if hits is None:
                sys.exit(0)
        else:
            hits = index.topk_hits(queries)
        results = regroup_topk_hits(hits, index.indexed_accessions())
        avg_time = -1.0
    elif multi_node_search:
        if cfg.do_timing:
            raise ValueError("do_timing is not supported with multi-node search")
        results = _multi_node_search(
            cfg, index, queries, index_path, node_rank, num_nodes
        )
        if results is None:
            sys.exit(0)
        avg_time = -1.0
    elif cfg.do_timing:
        times = []
        for i in range(cfg.timing_runs):
            start = time.time()
            results: pl.DataFrame = index.search(queries)
            elapsed = time.time() - start
            times.append(elapsed)
        avg_time = sum(times[1:]) / (cfg.timing_runs - 1)
    else:
        results: pl.DataFrame = index.search(queries)
        avg_time = -1.0

    results = _annotate_results(results, cfg, index, index_path, avg_time)
    output_path: Path = (
        cfg.results_dir
        / cfg.model.experiment_id
        / f"raw_read_mut_{cfg.mutation_rate}.parquet"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)

    if hits is not None:
        # Same metadata columns as the results, so rescore_topk.py can copy
        # them through; lives outside results_dir (see topk_hits_dir).
        assert cfg.topk_hits_dir is not None
        hits_path: Path = (
            cfg.topk_hits_dir
            / cfg.model.experiment_id
            / f"raw_read_mut_{cfg.mutation_rate}_topk{cfg.model.top_k}.parquet"
        )
        hits_path.parent.mkdir(exist_ok=True, parents=True)
        _annotate_results(hits, cfg, index, index_path, avg_time).write_parquet(
            hits_path
        )
        print(f"Wrote top-{cfg.model.top_k} vector hits -> {hits_path}")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
