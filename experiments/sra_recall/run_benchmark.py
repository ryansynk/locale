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
from collections import Counter


# def apply_mutations(queries: list[str], mutation_rate: float) -> list[str]:
#    return ["a"]


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

    queries: pl.DataFrame = pl.read_parquet(cfg.queries_path)
    # queries = apply_mutations(queries, cfg.mutation_rate)

    results: pl.DataFrame = index.search(queries)  # dataframe
    results = results.with_columns(pl.lit(str(cfg.model)).alias("model"))
    results = results.with_columns(pl.lit(cfg.mutation_rate).alias("mutation_rate"))
    output_path: Path = (
        cfg.results_dir / f"{cfg.model.name}.mutation_{cfg.mutation_rate}.parquet"
    )
    results.write_parquet(output_path)


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
