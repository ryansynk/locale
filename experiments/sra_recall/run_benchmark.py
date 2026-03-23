from pathlib import Path

import polars as pl
from jsonargparse import CLI
from src.config import (
    DenseConfig,
    MetagraphConfig,
    ExperimentConfig,
)
from src.dense_index import DenseIndex
from src.metagraph_index import MetagraphIndex
from rawbert.training.unsupervised_batcher import Augmenter


def apply_mutations(queries: pl.DataFrame, mutation_rate: float) -> pl.DataFrame:
    return queries.with_columns(
        pl.col("query_sequence").map_elements(
            lambda query_seq: Augmenter.augment(query_seq, identity=1 - mutation_rate)
        )
    )


def main(cfg: ExperimentConfig):
    index_path: Path = cfg.index_dir / cfg.model.name
    accession_paths: list[Path] = sorted(list(cfg.accessions_dir.rglob("*.contigs.fa")))
    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    else:
        raise ValueError("Unknown model config")

    index_path: Path = cfg.index_dir / cfg.model.name
    if index_path.exists():
        index.load(index_path)
        # indexed_accs = index.indexed_accessions()
        # assert Counter(indexed_accs) == Counter(accession_paths)
    else:
        index.build(accession_paths, index_path)
        index.save(index_path)

    if cfg.query_type == "raw_read":
        queries: pl.DataFrame = pl.read_parquet(cfg.raw_read_queries_path)
    elif cfg.query_type == "logan_contig":
        queries: pl.DataFrame = pl.read_parquet(cfg.logan_contig_queries_path)
    else:
        raise ValueError(
            f"Expected query_type to be 'raw_read' or 'logan_contig', got: {cfg.query_type}"
        )

    if cfg.filter_query_lens:
        queries = (
            queries.with_columns(
                pl.col("query_sequence").str.len_chars().alias("sequence_len")
            )
            .filter((pl.col("sequence_len") <= 1024) & (pl.col("sequence_len") >= 150))
            .drop("sequence_len")
        )
    if cfg.mutation_rate > 0.0:
        queries = apply_mutations(queries, cfg.mutation_rate)

    results: pl.DataFrame = index.search(queries)  # dataframe
    results = results.with_columns(pl.lit(str(cfg.model)).alias("model"))
    results = results.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    results = results.with_columns(pl.lit(cfg.query_type).alias("query_type"))
    output_path: Path = (
        cfg.results_dir
        / f"{cfg.model.name}.mutation_{cfg.mutation_rate}.query_type_{cfg.query_type}.parquet"
    )
    results.write_parquet(output_path)


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
