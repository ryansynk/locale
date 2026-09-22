import os
import re
import shutil
import sys
import time
from pathlib import Path

import polars as pl
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

DATASETS = {
    "sra50": "rsynk/locale-benchmark-sra50",
    "sra500": "rsynk/locale-benchmark-sra500",
    # Cross-genotype HBV retrieval: query with genotype-D reads, retrieve
    # genotype-B accessions (~11% divergent). Unlike sra50/sra500, ground truth
    # comes from SRA genotype metadata rather than from aligning reads to
    # contigs - the point being that no aligner decides the correct answer.
    # 52 accessions = 47 sra50 distractors + 5 genotype-B targets.
    "sra52viral": "rsynk/locale-benchmark-sra52-viral",
    # 2026-09 rebuild (locale-data list-first pipeline): fresh seeded draws
    # disjoint from the training/validation runs, source run required in the
    # relevant set, both strands aligned (`strand` column). sra4571 stays
    # local (see configs/perlmutter_locale_sra4571.yaml).
    "sra50v2": "rsynk/locale-benchmark-sra50-v2",
    "sra500v2": "rsynk/locale-benchmark-sra500-v2",
    "sra55viral": "rsynk/locale-benchmark-sra55viral",
}


def verify_download(accession_ids: list[str], accession_paths: list[Path]):
    # Check all accessions in manifest were found
    manifest_ids = set(accession_ids)
    downloaded_ids = {p.name.removesuffix(".contigs.fa") for p in accession_paths}
    if missing := manifest_ids - downloaded_ids:
        raise RuntimeError(
            f"{len(missing)} accessions missing after download: {missing}"
        )


def query_file_name(mutation_rate: float) -> str:
    """Per-rate query file in the bundle, e.g. queries_mut0.05.parquet.

    Queries are mutated ahead of time (locale-data/benchmark/mutate_queries.py)
    so every run at a rate sees identical sequences; mutation_rate only
    selects the file. Each file has queries.parquet's rows in the same order,
    so the seeded subsample below picks the same query_ids at every rate.
    """
    return f"queries_mut{mutation_rate:.2f}.parquet"


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
        f"search_partials_mut{cfg.mutation_rate}_n{len(queries)}_seed{cfg.random_seed}"
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
    if cfg.model.use_rabitq:
        return "rabitq1bit"
    if cfg.model.use_ann:
        return "cagra"
    return "exact"


def topk_hits_path(cfg: ExperimentConfig, k: int | None = None) -> Path:
    """Where a run persists its raw top-k hits (see topk_hits_dir).

    The directory is keyed by hits_id (encoder, engine, strands -- not k), the
    file by mutation rate and k, so runs at different k share a directory and
    find_cached_hits can serve a smaller k from a larger file.
    """
    assert isinstance(cfg.model, DenseConfig) and cfg.model.hits_id is not None
    assert cfg.topk_hits_dir is not None
    k = cfg.model.top_k if k is None else k
    return (
        cfg.topk_hits_dir
        / cfg.model.hits_id
        / f"raw_read_mut_{cfg.mutation_rate}_topk{k}.parquet"
    )


def find_cached_hits(
    cfg: ExperimentConfig, queries: pl.DataFrame
) -> pl.DataFrame | None:
    """Persisted hits that answer this run without a scan, or None.

    A hits file saved at K serves any run of the same hits_id at k <= K whose
    queries it covers: the top-k regroup ranking depends only on the first k
    hits, so the file is truncated per query and the rest is identical to a
    fresh scan. The largest qualifying K is used. Returns the hits frame
    (query_id, hits) in ``queries`` order.
    """
    assert isinstance(cfg.model, DenseConfig)
    k = cfg.model.top_k
    hits_dir = topk_hits_path(cfg).parent
    candidates = []
    for path in hits_dir.glob(f"raw_read_mut_{cfg.mutation_rate}_topk*.parquet"):
        m = re.fullmatch(r".*_topk(\d+)\.parquet", path.name)
        if m and int(m.group(1)) >= k:
            candidates.append((int(m.group(1)), path))
    wanted = queries.select("query_id")
    for saved_k, path in sorted(candidates, reverse=True):
        saved = pl.read_parquet(path, columns=["query_id", "hits"])
        if wanted.join(saved, on="query_id", how="anti").height:
            continue  # does not cover every requested query
        print(f"Reusing top-{saved_k} hits for k={k}: {path}")
        return wanted.join(saved, on="query_id", how="left").with_columns(
            pl.col("hits").list.head(k)
        )
    return None


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
    engine, top_k, strands and the search parameters and reused when present,
    so a timed-out run resumes.
    """
    n_vecs = index.num_vectors()
    bounds = [n_vecs * r // num_nodes for r in range(num_nodes + 1)]
    vec_range = (bounds[node_rank], bounds[node_rank + 1])
    strands = "_bothstrands" if index.both_strands else ""
    partials_dir = index_path / (
        f"{_topk_engine_tag(cfg)}_top{index.top_k}{strands}"
        f"_partials_mut{cfg.mutation_rate}_n{len(queries)}_seed{cfg.random_seed}"
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
    # dataset: dataset_dir must already hold accs.txt and the per-rate query
    # files (the bundle layout finalize_query_dataset.py + mutate_queries.py
    # write).
    queries_file = query_file_name(cfg.mutation_rate)
    if cfg.dataset_name in DATASETS:
        local_path = snapshot_download(
            DATASETS[cfg.dataset_name],
            repo_type="dataset",
            local_dir=cfg.dataset_dir,
            allow_patterns=["accs.txt", queries_file, "*.json"],
        )
    else:
        local_path = cfg.dataset_dir
        for required in ("accs.txt", queries_file):
            if not (Path(local_path) / required).exists():
                raise FileNotFoundError(
                    f"Local dataset '{cfg.dataset_name}': {required} not found in "
                    f"{local_path} (local datasets are not downloaded)"
                )
    accession_ids_path: Path = Path(local_path).resolve() / "accs.txt"
    with open(accession_ids_path) as f:
        accession_ids = f.read().splitlines()
    accessions_dir = Path(local_path).resolve() / "logan_accessions"
    queries_path: Path = Path(local_path).resolve() / queries_file
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

            if cfg.build_only:
                print(
                    "[build_only]: Shard 0 saved. Run finish_merge.py to merge. Exiting."
                )
                sys.exit(0)

            print(f"[Node 0] Waiting for {num_nodes - 1} other node(s) to finish...")
            _wait_for_shards(index_path, num_nodes)
            DenseIndex.merge_shards(index_path, num_nodes)
            (index_path / ".done").touch()
        else:
            index.build(accession_paths, index_path)
            index.save(index_path)
            (index_path / ".done").touch()

    if cfg.no_search or cfg.build_only:
        print("[no_search]: Index built. Exiting.")
        sys.exit(0)

    # Dense scoring protocols (DenseConfig; ESA/dna2vec and LLM-ED included):
    # the default top-k regroup retrieves vector hits with the exact fp32 scan
    # (or RaBitQ / CAGRA), the exhaustive reference scores every accession.
    # The exhaustive scan shards accessions across nodes and the two vector
    # scans shard vector rows; CAGRA and every non-dense method (metagraph,
    # mmseqs, centroid) search on node 0 alone.
    dense = isinstance(cfg.model, DenseConfig)
    dense_exhaustive = dense and cfg.model.exhaustive
    topk_engine = dense and not cfg.model.exhaustive
    shardable_topk = topk_engine and not cfg.model.use_ann
    multi_node_search = num_nodes > 1 and (dense_exhaustive or shardable_topk)
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

    hits: pl.DataFrame | None = None
    if topk_engine and not cfg.do_timing:
        # Compute-once, re-score-many: a scan yields each query's raw top_k
        # vector hits, which are persisted and regrouped into the standard
        # results. A later run at a smaller k (same encoder/engine/strands)
        # finds them and skips the scan entirely.
        cached = find_cached_hits(cfg, queries)
        if cached is not None:
            if node_rank != 0:
                sys.exit(0)
            hits = cached
        elif multi_node_search:
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
        # index.search embeds the queries inside the timed region (both
        # strands when both_strands), then retrieves and regroups.
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

    if hits is not None and cached is None:
        # Lives outside results_dir (see topk_hits_dir); a cached run leaves
        # the larger file it read from in place.
        hits_path = topk_hits_path(cfg)
        hits_path.parent.mkdir(exist_ok=True, parents=True)
        _annotate_results(hits, cfg, index, index_path, avg_time).write_parquet(
            hits_path
        )
        print(f"Wrote top-{cfg.model.top_k} vector hits -> {hits_path}")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
