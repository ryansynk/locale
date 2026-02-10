from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class AlgorithmConfig:
    """Base configuration for any algorithm."""

    pass


@dataclass
class SourMashConfig(AlgorithmConfig):
    name: str = "sourmash"
    k: int = 31
    scaled: int = 1
    threshold: float = 0.0


@dataclass
class DenseConfig(AlgorithmConfig):
    name: Literal["rawbert", "dnabert"] = "rawbert"  # "dnabert", "rawbert"
    checkpoint_path: Optional[str] = None
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"

    def __str__(self):
        return f"{self.name}"


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, SourMashConfig]
    dataset_path: str = (
        "/fs/nexus-scratch/ryansynk/rawbert_data/data/test_contigs.parquet"
    )
    num_keys: int = 10000
    num_queries: int = 100
    max_seq_len: int = 1024
    min_coverage: float = 0.1
    similarity_threshold: float = 0.6
    topks: List[int] = field(default_factory=lambda: [1, 5])
