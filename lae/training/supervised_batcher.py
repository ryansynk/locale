import random
from pathlib import Path

import polars as pl
from torch.utils.data import Dataset


class SupervisedBatcher(Dataset):
    def __init__(self, dataset_path, augment_config, num_examples=None):
        dataset_path = Path(dataset_path).resolve()
        self.cfg = augment_config

        self.df = pl.read_parquet(dataset_path)
        self.df = (
            self.df.with_columns(
                pl.col("query_seq").str.len_chars().alias("query_len"),
                pl.col("reference_seq").str.len_chars().alias("reference_len"),
            )
            .with_columns(
                pl.max_horizontal("query_len", "reference_len").alias("max_len")
            )
            .filter(
                (pl.col("max_len") < self.cfg.max_len)
                & (pl.col("coverage") > self.cfg.min_coverage)
            )
        )
        if num_examples is not None:
            self.df = self.df.head(num_examples)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        if random.random() < 0.5:
            query = row["query_seq"]
            ref = row["reference_seq"]
        else:
            ref = row["query_seq"]
            query = row["reference_seq"]

        return query, ref
