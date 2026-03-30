from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


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
class Evo2Config(AlgorithmConfig):
    name: str = "evo2"
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"

    def __post_init__(self):
        config_tag = f"pool{self.pooling}"
        self.index_suffix = Path("evo2") / config_tag
        self.experiment_id = f"evo2_{config_tag}"
        self.checkpoint: str | None = None
        self.max_len: int | None = None
        self.chunk_type: int | None = None

    def __str__(self):
        return self.experiment_id


@dataclass
class DenseConfig(AlgorithmConfig):
    name: Literal["rawbert", "dnabert", "generator"] = "rawbert"  # "dnabert", "rawbert"
    checkpoint_path: Optional[str] = None
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"
    k: int = 100
    max_seq_len: int = 1024
    chunk_type: Literal["stride", "exact_chunk"] = "stride"
    chunk_overlap: int = 150

    def __post_init__(self):
        config_tag = (
            f"maxlen{self.max_seq_len}_pool{self.pooling}_chunk{self.chunk_type}"
        )
        if self.name == "dnabert":
            self.index_suffix: Path = Path("dnabert") / config_tag
            self.experiment_id: str = f"dnabert_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "generator":
            self.index_suffix: Path = Path("generator") / config_tag
            self.experiment_id: str = f"generator_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "rawbert":
            assert self.checkpoint_path is not None
            ckpt_id = Path(self.checkpoint_path).resolve().parent.name
            self.index_suffix: Path = Path("rawbert") / ckpt_id / config_tag
            self.experiment_id: str = f"rawbert_{ckpt_id}_{config_tag}"
            self.checkpoint: str | None = ckpt_id
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        else:
            raise ValueError(f"name expected: rawbert or dnabert, got = {self.name}")

    def __str__(self):
        return self.experiment_id


@dataclass
class MetagraphConfig(AlgorithmConfig):
    name: str = "metagraph"
    executable: str = "metagraph"
    k: int = 31

    def __post_init__(self):
        self.index_suffix = Path("metagraph") / f"k{self.k}"
        self.experiment_id = f"metagraph_k{self.k}"
        self.checkpoint: str | None = None
        self.max_len: int | None = None
        self.chunk_type: int | None = None

    def __str__(self):
        return self.experiment_id


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, MetagraphConfig, Evo2Config]
    accessions_dir: Path
    raw_read_queries_path: Path
    logan_contig_queries_path: Path
    index_dir: Path
    results_dir: Path
    query_type: Literal["raw_read", "logan_contig"]
    mutation_rate: float = 0.0

    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)
