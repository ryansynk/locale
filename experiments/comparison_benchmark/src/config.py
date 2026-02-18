from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union


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
class MMSeqs2Config(AlgorithmConfig):
    name: str = "mmseqs2"
    sensitivity: float = 7.5  # -s parameter (1.0=fast, 7.5=sensitive)
    search_type: int = 3  # 3 = nucleotide-nucleotide
    threads: int = 4
    mmseqs_binary: str = "mmseqs"  # path to mmseqs binary

    def __str__(self):
        return f"{self.name}"


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, SourMashConfig, MMSeqs2Config]
    reference_path: str | None = None
    dataset_path: str | None = None
    num_distractors: int = 10000
    results_dir: Path | str | None = None

    max_seq_len: int = 6000
    topks: List[int] = field(default_factory=lambda: [1, 5])
    identity_bins: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.7, 0.75),
            (0.75, 0.8),
            (0.8, 0.85),
            (0.9, 0.95),
        ]
    )

    def __post_init__(self):
        if self.results_dir is not None:
            self.results_dir = Path(self.results_dir)
        else:
            raise ValueError("No results dir provided")
