from pathlib import Path

import polars as pl
from jsonargparse import CLI
from src.config import (
    DenseConfig,
    Evo2Config,
    ExperimentConfig,
    MetagraphConfig,
)
from src.dense_index import DenseIndex
from src.evo2_index import Evo2Index
from src.metagraph_index import MetagraphIndex

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

    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
        if cfg.model.chunk_type == "exact_chunk":
            index.contig_align_intervals = get_matching_regions_of_contigs(queries)
    elif isinstance(cfg.model, Evo2Config):
        index = Evo2Index(cfg)
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    else:
        raise ValueError("Unknown model config")

    if index_path.exists():
        index.load(index_path)
        # indexed_accs = index.indexed_accessions()
        # assert Counter(indexed_accs) == Counter(accession_paths)
    else:
        index.build(accession_paths, index_path)
        index.save(index_path)

    if cfg.mutation_rate > 0.0:
        queries = apply_mutations(queries, cfg.mutation_rate)

    results: pl.DataFrame = index.search(queries)  # dataframe
    results = results.with_columns(pl.lit(str(cfg.model)).alias("model"))
    results = results.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    results = results.with_columns(pl.lit(cfg.query_type).alias("query_type"))
    results = results.with_columns(
        pl.lit(cfg.model.checkpoint, dtype=pl.String).alias("checkpoint")
    )
    results = results.with_columns(
        pl.lit(cfg.model.max_len, dtype=pl.Int64).alias("max_len")
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
