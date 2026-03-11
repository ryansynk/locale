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

    def merge(self, listy):
        b = []
        for begin in sorted(listy):
            end = begin + self.k
            if b and b[-1][1] >= begin - 1:
                b[-1][1] = max(b[-1][1], end)
            else:
                b.append([begin, end])
        return b

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        queries = queries.with_columns(
            pl.col("query_sequence").str.len_chars().alias("query_length")
        )
        results = self.graph_client.search(
            queries["query_sequence"].to_list(), query_coords=True
        )
        df = pl.from_pandas(results)
        # Merge query k-mer intervals and sum lengths to get base pair exact matches
        splits = pl.element().str.split("-")
        query_kmer_idx = splits.list.get(0).cast(pl.Int64)
        df = (
            df.with_columns(
                pl.col("kmer_coords")
                .list.eval(query_kmer_idx)
                .list.eval(
                    pl.element().diff().fill_null(self.k).clip(upper_bound=self.k)
                )
                .list.sum()
                .alias("exact_matches"),
                pl.col("sample")
                .str.split("/")
                .list.get(-2)
                .alias("retrieved_accession"),
            )
            .cast({"seq_description": pl.Int64})
            .select(["retrieved_accession", "exact_matches", "seq_description"])
        )
        results_w_identity = (
            df.join(queries, left_on="seq_description", right_on="index", how="full")
            .select(
                [
                    "read_id",
                    "accession",
                    "retrieved_accession",
                    "exact_matches",
                    "query_length",
                ]
            )
            .with_columns(
                (pl.col("exact_matches") / pl.col("query_length")).alias("identity")
            )
            .rename({"read_id": "query_read", "accession": "query_accession"})
            .select(
                ["query_read", "query_accession", "retrieved_accession", "identity"]
            )
            .with_columns(
                pl.col("identity").fill_null(0.0),
                pl.col("retrieved_accession").fill_null(""),
            )
        )

        results_w_identity = (
            results_w_identity.with_columns(
                pl.struct("retrieved_accession", "identity").alias("retrievals")
            )
            .group_by("query_read")
            .agg(pl.col("query_accession").first(), pl.col("retrievals"))
        )
        assert len(results_w_identity) == len(queries)
        return results_w_identity

    def indexed_accessions(self) -> list[Path]:
        return []

    def save(self, output_path: Path):
        pass
