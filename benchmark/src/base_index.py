from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class BaseIndex(ABC):
    @abstractmethod
    def load(self, index_path: Path):
        pass

    @abstractmethod
    def build(self, accessions: list[Path], index_path: Path):
        pass

    @abstractmethod
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def indexed_accessions(self) -> list[str]:
        pass

    @abstractmethod
    def save(self, output_path: Path):
        pass

    @abstractmethod
    def index_size_gb(self, index_path: Path):
        pass

    @staticmethod
    def merge_shards(index_path: Path, num_nodes: int):
        """Combine index_path/shard_<r>/ for r < num_nodes into what load() reads.

        run_benchmark's multi-node build has node r build accessions[r::num_nodes]
        into shard_<r>/ and node 0 call this once every shard is .done. Index
        types that cannot be built across nodes leave this unimplemented.
        """
        raise NotImplementedError(
            "multi-node build is not supported for this index type; "
            "launch it with SLURM_NNODES=1"
        )
