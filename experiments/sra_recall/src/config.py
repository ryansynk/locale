from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class AlgorithmConfig:
    """Base configuration for any algorithm."""

    pass


# @dataclass
# class SourMashConfig(AlgorithmConfig):
#     name: str = "sourmash"
#     k: int = 31
#     scaled: int = 1
#     threshold: float = 0.0


@dataclass
class DenseConfig(AlgorithmConfig):
    name: Literal["rawbert", "dnabert"] = "rawbert"  # "dnabert", "rawbert"
    checkpoint_path: Optional[str] = None
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"
    k: int = 100
    max_seq_len: int = 1024
    min_seq_len: int = 150
    min_overlap_percent: float = 0.6

    def __str__(self):
        return f"{self.name}"


@dataclass
class MetagraphConfig(AlgorithmConfig):
    name: str = "metagraph"
    executable: str = "metagraph"
    k: int = 31

    def __str__(self):
        return f"{self.name}"


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, MetagraphConfig]
    accessions_dir: Path
    queries_path: Path
    index_dir: Path
    results_dir: Path
    mutation_rate: float = 0.0
