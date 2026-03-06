import subprocess
import time
from pathlib import Path

import polars as pl
from .config import ExperimentConfig, MetagraphConfig
from metagraph.client import GraphClient

from .base_index import BaseIndex


class MetagraphIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MetagraphConfig)
        assert cfg.model.graph_path is not None
        assert cfg.model.annotation_path is not None
        self.port: int = cfg.model.port
        self.graph_path: Path = cfg.model.graph_path
        self.annotation_path: Path = cfg.model.annotation_path
        self.k: int = cfg.model.k

    def load(self, index_path: Path):
        self.query_server_proc = subprocess.Popen(
            [
                "shifter",
                "metagraph",
                "server_query",
                "-i",
                str(self.graph_path),
                "-a",
                str(self.annotation_path),
                "--port",
                str(self.port),
            ]
        )
        # Good ol reliable sleep
        # Kludge -- I need to wait for server to start but don't want to figure out how
        # to communicate with metagraph process to see if its ready
        time.sleep(5)
        self.graph_client = GraphClient("127.0.0.1", self.port, api_path="")

    def build(self, accessions: list[Path]):
        raise NotImplementedError

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        queries = queries.with_columns(
            pl.col("query_sequence").str.len_chars().alias("query_length")
        )
        results = self.graph_client.search(
            queries["query_sequence"].to_list(), query_coords=True
        )
        df = pl.from_pandas(results)
        breakpoint()
        result_df = (
            df.with_row_index("row_id")
            .explode("kmer_coords")
            .with_columns(pl.col("kmer_coords").str.split("-").alias("parts"))
            .with_columns(
                # Extract query start (index 0)
                pl.col("parts").list.get(0).cast(pl.Int64).alias("q_start"),
                # Extract graph start and end to compute the number of k-mers in this block
                pl.col("parts").list.get(1).cast(pl.Int64).alias("g_start"),
                pl.col("parts")
                .list.get(2, null_on_oob=True)
                .cast(pl.Int64)
                .alias("g_end"),
            )
            .with_columns(pl.col("g_end").fill_null(pl.col("g_start")))
            .with_columns(
                # Calculate how many consecutive k-mers are matched in this specific block
                (pl.col("g_start") - pl.col("g_end")).abs().add(1).alias("num_kmers")
            )
            .with_columns(
                # A block starting at q_start with N k-mers covers bases: q_start to (q_start + N + k - 2)
                # int_ranges is exclusive of the end point, so we use + k - 1
                pl.int_ranges(
                    pl.col("q_start"),
                    pl.col("q_start") + pl.col("num_kmers") + self.k - 1,
                ).alias("q_bases")
            )
            .explode("q_bases")
            .group_by("row_id", maintain_order=True)
            .agg(
                # Count the unique query base coordinates covered by all matched k-mers
                pl.when(pl.col("q_bases").drop_nulls().len() == 0)
                .then(0)
                .otherwise(pl.col("q_bases").drop_nulls().n_unique())
                .alias("read_exact_matching_bp")
            )
            .join(df.with_row_index("row_id"), on="row_id", how="right")
            .drop("row_id")
        )
        result_df = (
            result_df.with_columns(pl.col("seq_description").cast(pl.Int64))
            .join(
                queries.select(["index", "accession", "read_id", "query_length"]),
                left_on="seq_description",
                right_on="index",
            )
            .with_columns(
                (pl.col("read_exact_matching_bp") / pl.col("query_length")).alias(
                    "identity"
                )
            )
        )

        result_df = result_df.select(["read_id", "accession", "sample", "identity"])

        # Post-process 'sample' string to extract accession
        result_df = result_df.with_columns(
            pl.col("sample").str.split("/").list.get(-2).alias("retrieved_accession")
        )
        result_df = result_df.rename(
            {"read_id": "query_read", "accession": "query_accession"}
        )
        self.query_server_proc.terminate()
        return result_df.select(
            ["query_read", "query_accession", "retrieved_accession", "identity"]
        )

    def indexed_accessions(self) -> list[Path]:
        return []

    def save(self, output_path: Path):
        pass
