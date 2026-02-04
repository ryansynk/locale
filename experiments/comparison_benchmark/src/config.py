from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass
class AlgorithmConfig:
    """Base configuration for any algorithm."""

    pass


@dataclass
class SourMashConfig(AlgorithmConfig):
    pass


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
    dataset_path: str = (
        "/fs/nexus-scratch/ryansynk/rawbert_data/data/test_contigs.parquet"
    )
    num_keys: int = 10000
    num_queries: int = 100
    max_seq_len: int = 1024
    min_coverage: float = 0.1
    topk: int = 5
    model: Union[DenseConfig, SourMashConfig] = field(default_factory=DenseConfig)
