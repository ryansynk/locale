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
    def indexed_accessions(self) -> list[Path]:
        pass

    @abstractmethod
    def save(self, output_path: Path):
        pass
