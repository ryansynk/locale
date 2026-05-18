import os
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
)
from src.dense_index import DenseIndex
from src.download_accessions import download_accessions
from src.metagraph_index import MetagraphIndex
from src.mmseqs2_index import MMseqs2Index

from lae.training.batcher import Augmenter

DATASETS = {
    "sra50": "rsynk/locale-benchmark-sra50",
    "sra500": "rsynk/locale-benchmark-sra500",
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


def main(cfg: ExperimentConfig):
    index_path: Path = cfg.index_dir / cfg.model.index_suffix

    # Download datasets and queries
    local_path = snapshot_download(
        DATASETS[cfg.dataset_name], local_dir=cfg.dataset_dir
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

    queries: pl.DataFrame = pl.read_parquet(queries_path)
    # subsample queries if needed
    queries = queries.sample(min(cfg.num_queries, len(queries)), seed=cfg.random_seed)

    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    elif isinstance(cfg.model, MMseqs2Config):
        index = MMseqs2Index(cfg)
    else:
        raise ValueError("Unknown model config")

    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    # Build index if not already built
    if not (index_path / ".done").exists():
        if num_nodes > 1:
            node_accessions = accession_paths[node_rank::num_nodes]
            shard_path = index_path / f"shard_{node_rank}"
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

    if num_nodes > 1 and node_rank != 0:
        print(
            f"[Node {node_rank}] Index built. Skipping search (only node 0 searches)."
        )
        sys.exit(0)

    index.load(index_path)
    if cfg.mutation_rate > 0.0:
        queries = apply_mutations(queries, cfg.mutation_rate)

    if cfg.do_timing:
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

    results = results.with_columns(
        pl.lit(index.index_size_gb(index_path)).alias("index_size_gb")
    )
    results = results.with_columns(pl.lit(avg_time).alias("avg_time"))
    results = results.with_columns(pl.lit(str(cfg.model)).alias("model"))
    results = results.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    results = results.with_columns(pl.lit("raw_read").alias("query_type"))  # legacy
    results = results.with_columns(
        pl.lit(cfg.model.checkpoint, dtype=pl.String).alias("checkpoint")
    )
    results = results.with_columns(
        pl.lit(cfg.model.max_len, dtype=pl.Int64).alias("max_len")
    )
    results = results.with_columns(
        pl.lit(cfg.model.checkpoint_step_num, dtype=pl.Int64).alias(
            "checkpoint_step_num"
        )
    )
    results = results.with_columns(pl.lit("stride").alias("chunk_type"))  # legacy
    output_path: Path = (
        cfg.results_dir
        / cfg.model.experiment_id
        / "raw_read_mut_{cfg.mutation_rate}.parquet"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
