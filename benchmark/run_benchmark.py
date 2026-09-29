import shutil
import sys
import time
from pathlib import Path

import polars as pl
from jsonargparse import CLI
from src.config import (
    DONE_FILE,
    DenseMethod,
    ExperimentConfig,
    MetagraphConfig,
    MMseqs2Config,
    ensure_config,
    read_config,
    results_file_name,
    run_search_identity,
)
from src.dense_index import DenseIndex
from src.engines import make_engine
from src.download_accessions import download_accessions
from src.metagraph_index import MetagraphIndex
from src.mmseqs2_index import MMseqs2Index
from src.topk_regroup import merge_topk_hits, regroup_topk_hits

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
    index_path: Path, num_nodes: int, timeout: int = 8 * 3600, poll_interval: int = 30
):
    # Node 0 may find its own shard .done from an earlier run while the others
    # build from scratch, so the wait spans a whole shard build, not a merge.
    elapsed = 0
    while elapsed < timeout:
        if all(
            (index_path / f"shard_{r}" / DONE_FILE).exists() for r in range(num_nodes)
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


def _partials_dir(cfg: ExperimentConfig, queries: pl.DataFrame) -> Path:
    """Where a multi-node search keeps per-rank partials until node 0 merges
    them: under the run's results directory, keyed by the query draw, so an
    interrupted run resumes instead of rescoring."""
    return results_path(cfg) / (
        f"partials_mut{cfg.mutation_rate:.2f}_n{len(queries)}_seed{cfg.random_seed}"
    )


def _multi_node_search(
    cfg: ExperimentConfig,
    index: DenseIndex,
    queries: pl.DataFrame,
    node_rank: int,
    num_nodes: int,
) -> pl.DataFrame | None:
    """Shard the exhaustive dense search across nodes; node 0 merges partials.

    Every node scores a stride of the accessions and atomically writes a
    partial result; node 0 waits for all partials and reassembles each query's
    results in index order, identical to a single-node search. Returns the
    merged results on node 0, None on every other rank.
    """
    acc_names = index.indexed_accessions()
    acc_indices = list(range(node_rank, len(acc_names), num_nodes))
    partials_dir = _partials_dir(cfg, queries)
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


def results_path(cfg: ExperimentConfig) -> Path:
    return cfg.model.results_path(cfg.results_dir)


def hits_path(cfg: ExperimentConfig) -> Path:
    """Where a run persists its raw top-k hits: <results_path>/hits/mut<rate>.parquet
    (K is in the search config.json)."""
    return results_path(cfg) / "hits" / results_file_name(cfg.mutation_rate)


def _hits_key(search_identity: dict) -> dict:
    """The part of a search identity a hits file must share to be reusable:
    everything but k and the query draw (coverage is checked by query_id)."""
    return {k: v for k, v in search_identity.items()
            if k not in ("top_k", "num_queries", "random_seed")}


def find_cached_hits(
    cfg: ExperimentConfig, queries: pl.DataFrame
) -> pl.DataFrame | None:
    """Persisted hits that answer this run without a scan, or None.

    A hits file saved at K under any search label of the same encoder and
    index (sibling directories of this run's) serves a run at k <= K whose
    queries it covers: the top-k regroup ranking depends only on the first k
    hits, so the file is truncated per query and the rest is identical to a
    fresh scan. The largest qualifying K is used. Returns the hits frame
    (query_id, hits) in ``queries`` order.
    """
    assert isinstance(cfg.model, DenseMethod)
    k = cfg.model.index.top_k
    want = run_search_identity(cfg)
    own = results_path(cfg)
    candidates = []
    for d in sorted(own.parent.iterdir()) if own.parent.is_dir() else []:
        saved = read_config(d) if d.is_dir() else None
        if saved is None or "top_k" not in saved:
            continue
        if _hits_key(saved) != _hits_key(want) or saved["top_k"] < k:
            continue
        path = d / "hits" / results_file_name(cfg.mutation_rate)
        if path.exists():
            candidates.append((saved["top_k"], d == own, path))
    wanted = queries.select("query_id")
    for saved_k, _, path in sorted(candidates, reverse=True):
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
    node_rank: int,
    num_nodes: int,
) -> pl.DataFrame | None:
    """Shard a vector-level top-k scan across nodes; node 0 merges the hits.

    Unlike the exhaustive search, which deals out accessions, this splits the
    index's *vector rows* into num_nodes contiguous, equal ranges (the scan is
    a flat matmul over rows, so this balances the read exactly). Every node
    writes its per-query top-k hits over its range; node 0 unions the partials
    and re-takes each query's global top-k, which equals a single-node scan
    because each partial is complete over a disjoint range. Works for every
    SHARDABLE engine (exact scan, RaBitQ codes: each node loads only its
    range). Returns the merged hits frame on node 0, None elsewhere.
    """
    n_vecs = index.num_vectors()
    bounds = [n_vecs * r // num_nodes for r in range(num_nodes + 1)]
    vec_range = (bounds[node_rank], bounds[node_rank + 1])
    partials_dir = _partials_dir(cfg, queries)
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
    labels = cfg.model.labels()
    encoder = getattr(cfg.model, "encoder", None)
    ckpt, step = encoder.checkpoint() if encoder is not None else (None, None)
    max_len = encoder.max_seq_len if encoder is not None else None
    df = df.with_columns(pl.lit(index.index_size_gb(index_path)).alias("index_size_gb"))
    df = df.with_columns(pl.lit(avg_time).alias("avg_time"))
    df = df.with_columns(pl.lit(cfg.dataset_name).alias("dataset"))
    df = df.with_columns(pl.lit(labels["encoder"], dtype=pl.String).alias("encoder"))
    df = df.with_columns(pl.lit(labels["index"], dtype=pl.String).alias("index"))
    df = df.with_columns(pl.lit(labels["search"], dtype=pl.String).alias("search"))
    df = df.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    df = df.with_columns(pl.lit("raw_read").alias("query_type"))  # legacy
    df = df.with_columns(pl.lit(ckpt, dtype=pl.String).alias("checkpoint"))
    df = df.with_columns(pl.lit(max_len, dtype=pl.Int64).alias("max_len"))
    df = df.with_columns(pl.lit(step, dtype=pl.Int64).alias("checkpoint_step_num"))
    df = df.with_columns(pl.lit("stride").alias("chunk_type"))  # legacy
    return df


def main(cfg: ExperimentConfig):
    method = cfg.model
    index_path: Path = method.index_path(cfg.index_dir)
    node_rank = cfg.shard
    num_nodes = cfg.num_shards
    dense = isinstance(method, DenseMethod)

    if isinstance(method, DenseMethod):
        index_cls = DenseIndex
    elif isinstance(method, MetagraphConfig):
        index_cls = MetagraphIndex
    elif isinstance(method, MMseqs2Config):
        index_cls = MMseqs2Index
    else:
        raise ValueError("Unknown model config")

    # The label's directory must have been built under this identity (or be
    # new). Written before the build so every rank of a multi-node job, and
    # every later run, can check it.
    ensure_config(index_path, method.index_identity(), (index_path / DONE_FILE).exists())

    if cfg.stage == "merge":
        # ExperimentConfig refused already unless every shard is .done. A
        # completed dense merge deletes its progress file, so merging again
        # would restart the copy from scratch -- skip when already done.
        if (index_path / DONE_FILE).exists():
            print(f"Index already merged and marked .done: {index_path}")
            return
        index_cls.merge_shards(index_path, num_nodes)
        (index_path / DONE_FILE).touch()
        print(f"Index marked .done: {index_path}")
        return

    if cfg.stage == "engine":
        # One shard of the engine artifact; the process that finds every
        # shard present marks the directory .done.
        if not dense:
            raise ValueError("stage engine needs a dense method (model.index)")
        if not (index_path / DONE_FILE).exists():
            raise FileNotFoundError(f"{index_path} is not .done: run stage embed (and merge) first")
        engine_path = method.engine_path(cfg.index_dir)
        ensure_config(engine_path, method.engine_identity(), (engine_path / DONE_FILE).exists())
        if (engine_path / DONE_FILE).exists():
            print(f"Engine already built and marked .done: {engine_path}")
            return
        engine = make_engine(method.index)
        t0 = time.time()
        engine.build(index_path, engine_path, node_rank, num_nodes)
        print(f"[shard {node_rank}/{num_nodes}] built in {time.time() - t0:.0f}s")
        complete = engine.complete(engine_path) if hasattr(engine, "complete") else node_rank == 0
        if complete:
            (engine_path / DONE_FILE).touch()
            print(f"all {num_nodes} shards present: {engine_path} marked {DONE_FILE}")
        return

    if cfg.stage == "search":
        needed = [index_path] + ([method.engine_path(cfg.index_dir)] if dense else [])
        if missing := [str(p) for p in needed if not (p / DONE_FILE).exists()]:
            raise FileNotFoundError(f"stage search needs built indexes; not .done: {missing}")

    # dataset_dir must already hold accs.txt and the per-rate query files (the
    # bundle layout finalize_query_dataset.py + mutate_queries.py write, or a
    # published bundle fetched with fetch_dataset.py).
    queries_file = query_file_name(cfg.mutation_rate)
    local_path = cfg.dataset_dir
    for required in ("accs.txt", queries_file):
        if not (Path(local_path) / required).exists():
            raise FileNotFoundError(
                f"Dataset '{cfg.dataset_name}': {required} not found in "
                f"{local_path} (fetch a published bundle with fetch_dataset.py)"
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

    index = index_cls(cfg)

    # Build index if not already built
    if not (index_path / DONE_FILE).exists():
        if num_nodes > 1:
            node_accessions = accession_paths[node_rank::num_nodes]
            shard_path = index_path / f"shard_{node_rank}"
            # A shard marked .done was fully built by a previous run at the
            # same node count; skipping it makes timed-out runs resumable.
            if (shard_path / DONE_FILE).exists():
                print(f"[Node {node_rank}/{num_nodes}] Shard already built, skipping.")
            else:
                print(
                    f"[Node {node_rank}/{num_nodes}] Building shard from {len(node_accessions)} accessions..."
                )
                index.build(node_accessions, shard_path)
                index.save(shard_path)
                (shard_path / DONE_FILE).touch()

            if cfg.stage == "embed":
                print(f"[Node {node_rank}] Shard saved. Run stage merge to merge. Exiting.")
                sys.exit(0)

            if node_rank != 0:
                # Stay for the search: the dense scans below shard across
                # nodes and node 0 waits for this rank's partial. (Exiting
                # here left node 0 waiting out its timeout whenever a build
                # and a sharded search ran in one job, backbone sweep
                # 2026-09-23.) The merged index appears when node 0 marks it.
                print(f"[Node {node_rank}] Shard saved. Waiting for node 0 to merge...")
                _wait_for_files([index_path / DONE_FILE], timeout=8 * 3600)
            else:
                print(
                    f"[Node 0] Waiting for {num_nodes - 1} other node(s) to finish..."
                )
                _wait_for_shards(index_path, num_nodes)
                type(index).merge_shards(index_path, num_nodes)
                (index_path / DONE_FILE).touch()
        else:
            index.build(accession_paths, index_path)
            index.save(index_path)
            (index_path / DONE_FILE).touch()

    if cfg.stage == "embed":
        print("[stage embed]: Index built. Exiting.")
        sys.exit(0)

    # Dense engine artifact (codes / shards; a no-op for the exact scans).
    # Every rank calls build: engines that shard their build across ranks
    # coordinate inside it and return once the index is complete.
    if dense:
        engine_path = method.engine_path(cfg.index_dir)
        ensure_config(engine_path, method.engine_identity(), (engine_path / DONE_FILE).exists())
        if not (engine_path / DONE_FILE).exists():
            index.engine.build(index_path, engine_path, node_rank, num_nodes)
            if node_rank == 0:
                (engine_path / DONE_FILE).touch()

    if cfg.no_search:
        print("[no_search]: Index built. Exiting.")
        sys.exit(0)

    # Dense scoring protocols (ESA/dna2vec and LLM-ED included): the default
    # top-k regroup retrieves vector hits with the engine, the exhaustive
    # reference scores every accession. The exhaustive scan shards
    # accessions across nodes and SHARDABLE engines shard vector rows; the
    # other engines and every non-dense method (metagraph, mmseqs) search on
    # node 0 alone.
    dense_exhaustive = dense and index.exhaustive
    topk_engine = dense and not index.exhaustive
    shardable_topk = topk_engine and index.engine.SHARDABLE
    multi_node_search = num_nodes > 1 and (dense_exhaustive or shardable_topk)
    if num_nodes > 1 and not multi_node_search and node_rank != 0:
        print(
            f"[Node {node_rank}] Index built. Skipping search (only node 0 searches)."
        )
        sys.exit(0)

    out_dir = results_path(cfg)
    ensure_config(
        out_dir, run_search_identity(cfg), any(out_dir.glob("mut*.parquet")) if out_dir.is_dir() else False
    )

    index.load(index_path)

    hits: pl.DataFrame | None = None
    cached = None
    if topk_engine and not cfg.do_timing:
        # Compute-once, re-score-many: a scan yields each query's raw top_k
        # vector hits, which are persisted and regrouped into the standard
        # results. A later run at a smaller k (same encoder/index/strands)
        # finds them and skips the scan entirely.
        cached = find_cached_hits(cfg, queries)
        if cached is not None:
            if node_rank != 0:
                sys.exit(0)
            hits = cached
        elif multi_node_search:
            hits = _multi_node_topk_hits(cfg, index, queries, node_rank, num_nodes)
            if hits is None:
                sys.exit(0)
        else:
            hits = index.topk_hits(queries)
        results = regroup_topk_hits(hits, index.indexed_accessions())
        avg_time = -1.0
    elif multi_node_search:
        if cfg.do_timing:
            raise ValueError("do_timing is not supported with multi-node search")
        results = _multi_node_search(cfg, index, queries, node_rank, num_nodes)
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
            print(f"[timing] run {i + 1}/{cfg.timing_runs}: {elapsed:.3f}s", flush=True)
        avg_time = sum(times[1:]) / (cfg.timing_runs - 1)
        print(f"[timing] avg_time (runs 2-{cfg.timing_runs}) {avg_time:.3f}s", flush=True)
    else:
        results: pl.DataFrame = index.search(queries)
        avg_time = -1.0

    results = _annotate_results(results, cfg, index, index_path, avg_time)
    # A timed run sits beside the accuracy run it times (timing is not identity).
    if cfg.do_timing:
        output_path: Path = out_dir / f"mut{cfg.mutation_rate:.2f}.timing.parquet"
    else:
        output_path: Path = out_dir / results_file_name(cfg.mutation_rate)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)

    if hits is not None and cached is None:
        # A cached run leaves the larger file it read from in place.
        hp = hits_path(cfg)
        hp.parent.mkdir(exist_ok=True, parents=True)
        _annotate_results(hits, cfg, index, index_path, avg_time).write_parquet(hp)
        print(f"Wrote top-{index.top_k} vector hits -> {hp}")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
