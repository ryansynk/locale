import os
import sys
import time
from pathlib import Path

import polars as pl
from jsonargparse import CLI
from src.config import (
    DenseConfig,
    ExperimentConfig,
    MetagraphConfig,
    MantisConfig,
    MMseqs2Config,
)
from src.dense_index import DenseIndex
from src.metagraph_index import MetagraphIndex
from src.mantis_index import MantisIndex
from src.mmseqs2_index import MMseqs2Index

from rawbert.training.unsupervised_batcher import Augmenter


def apply_mutations(queries: pl.DataFrame, mutation_rate: float) -> pl.DataFrame:
    return queries.with_columns(
        pl.col("query_sequence").map_elements(
            lambda query_seq: Augmenter.augment(query_seq, identity=1 - mutation_rate)
        )
    )


def get_matching_regions_of_contigs(queries_df: pl.DataFrame):
    contig_interval_pairs = (
        queries_df.explode("contig_id", "aln_interval_contig")
        .group_by("contig_id")
        .agg(pl.col("aln_interval_contig"))
        .to_dicts()
    )
    intervals_dict = {}
    for contig_interval_pair in contig_interval_pairs:
        contig_id = contig_interval_pair["contig_id"]
        intervals_dict[contig_id] = [
            (interval["aln_start_index_contig"], interval["aln_end_index_contig"])
            for interval in contig_interval_pair["aln_interval_contig"]
        ]
    return intervals_dict


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
    accession_paths: list[Path] = sorted(list(cfg.accessions_dir.rglob("*.contigs.fa")))

    if cfg.query_type == "raw_read":
        queries: pl.DataFrame = pl.read_parquet(cfg.raw_read_queries_path)
    elif cfg.query_type == "logan_contig":
        queries: pl.DataFrame = pl.read_parquet(cfg.logan_contig_queries_path)
    elif cfg.query_type == "gencode":
        queries: pl.DataFrame = pl.read_parquet(cfg.gencode_queries_path)
    else:
        raise ValueError(
            f"Expected query_type to be 'raw_read', 'logan_contig', or 'gencode'. Got: {cfg.query_type}"
        )

    # subsample queries
    queries = queries.sample(min(cfg.num_queries, len(queries)), seed=cfg.random_seed)

    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
        if cfg.model.chunk_type == "exact_chunk":
            index.contig_align_intervals = get_matching_regions_of_contigs(queries)
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    elif isinstance(cfg.model, MantisConfig):
        index = MantisIndex(cfg)
    elif isinstance(cfg.model, MMseqs2Config):
        index = MMseqs2Index(cfg)
    else:
        raise ValueError("Unknown model config")

    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    if (index_path / ".done").exists():
        index.load(index_path)
    elif num_nodes > 1:
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
        index.load(index_path)
    else:
        index.build(accession_paths, index_path)
        index.save(index_path)
        (index_path / ".done").touch()

    if num_nodes > 1 and node_rank != 0:
        print(
            f"[Node {node_rank}] Index built. Skipping search (only node 0 searches)."
        )
        sys.exit(0)

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

    results = results.with_columns(pl.lit(avg_time).alias("avg_time"))
    results = results.with_columns(pl.lit(str(cfg.model)).alias("model"))
    results = results.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    results = results.with_columns(pl.lit(cfg.query_type).alias("query_type"))
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
    results = results.with_columns(pl.lit(cfg.model.chunk_type).alias("chunk_type"))
    output_path: Path = (
        cfg.results_dir
        / cfg.model.experiment_id
        / f"{cfg.query_type}_mut{cfg.mutation_rate}.parquet"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
