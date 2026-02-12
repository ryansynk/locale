from dataclasses import dataclass, field
from pathlib import Path
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
    # dataset_path: str = (
    #    "/fs/nexus-scratch/ryansynk/rawbert_data/data/test_contigs.parquet"
    # )
    alignments_path: str | None = None
    distractors_path: str | None = None
    num_distractors: int = 10000
    results_dir_str: str | None = None
    results_dir: Path | None = None

    # num_keys: int = 10000
    # num_queries: int = 100
    max_seq_len: int = 1024
    min_coverage: float = 0.65
    similarity_threshold: float = 0.6
    topks: List[int] = field(default_factory=lambda: [1, 5])

    def __post_init__(self):
        if self.results_dir_str is not None:
            self.results_dir = Path(self.results_dir_str)
        else:
            raise ValueError("No results dir provided")
