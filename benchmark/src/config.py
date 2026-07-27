from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass
class AlgorithmConfig:
    """Base configuration for any algorithm."""

    pass


@dataclass
class DenseConfig(AlgorithmConfig):
    name: Literal["locale", "dnabert", "generator", "neuroseed", "dna2vec", "llmed"] = (
        "locale"  # "dnabert", "locale"
    )
    checkpoint_path: Optional[str] = None
    checkpoint_step_num: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"
    k: int = 100
    max_seq_len: int = 1024
    chunk_overlap: int = 150
    neuroseed_path: Optional[str] = "/pscratch/sd/r/rsynk/NeuroSEED"
    use_ann: bool = False
    exact_search: bool = False
    use_rabitq: bool = False

    def __post_init__(self):
        config_tag = f"maxlen{self.max_seq_len}_pool{self.pooling}_chunkstride"
        if self.name == "dnabert":
            self.index_suffix: Path = Path("dnabert") / config_tag
            self.experiment_id: str = f"dnabert_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
        elif self.name == "generator":
            self.index_suffix: Path = Path("generator") / config_tag
            self.experiment_id: str = f"generator_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
        elif self.name == "locale":
            assert self.checkpoint_path is not None
            ckpt_id = Path(self.checkpoint_path).resolve().parent.name
            self.checkpoint_step_num = int(
                Path(self.checkpoint_path).name.split(".")[0][10:]
            )
            self.index_suffix: Path = (
                Path("locale") / ckpt_id / str(self.checkpoint_step_num) / config_tag
            )
            self.experiment_id: str = (
                f"locale_{ckpt_id}_{str(self.checkpoint_step_num)}_{config_tag}"
            )
            self.checkpoint: str | None = ckpt_id
            self.max_len: int = self.max_seq_len
        elif self.name == "neuroseed":
            assert self.checkpoint_path is not None
            ckpt_id = Path(self.checkpoint_path).resolve().stem
            self.index_suffix: Path = Path("neuroseed") / ckpt_id / config_tag
            self.experiment_id: str = f"neuroseed_{ckpt_id}_{config_tag}"
            self.checkpoint: str | None = ckpt_id
            self.max_len: int = self.max_seq_len
        elif self.name == "dna2vec":
            self.index_suffix: Path = Path("dna2vec") / config_tag
            self.experiment_id: str = f"dna2vec_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
        elif self.name == "llmed":
            # Abusing checkpoint path to distinguish between different kinds of llmed model
            self.index_suffix: Path = Path("llmed") / config_tag
            self.experiment_id: str = f"llmed_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
        else:
            raise ValueError(
                f"name expected: locale, dnabert, generator, neuroseed, dna2vec, or llmed. Got = {self.name}"
            )

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
        self.checkpoint_step_num: int | None = None

    def __str__(self):
        return self.experiment_id


@dataclass
class MMseqs2Config(AlgorithmConfig):
    name: str = "mmseqs"
    executable: str = "mmseqs"
    max_seqs: int = 300

    def __post_init__(self):
        self.index_suffix = Path("mmseqs")
        self.experiment_id = "mmseqs"
        self.checkpoint: str | None = None
        self.max_len: int | None = None
        self.chunk_type: int | None = None
        self.checkpoint_step_num: int | None = None

    def __str__(self):
        return self.experiment_id


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, MetagraphConfig, MMseqs2Config]
    dataset_name: str
    dataset_dir: str | None
    index_dir: Path
    results_dir: Path
    mutation_rate: float = 0.0
    do_timing: bool = False
    timing_runs: int = 5
    num_queries: int = 1000
    random_seed: int = 1337
    no_search: bool = False
    # Smoke-test knob: cap how many accessions enter the index so an end-to-end
    # run finishes in minutes. Leave unset for real runs — a truncated index is
    # still marked .done, so always pair this with a throwaway index_dir.
    max_accessions: int | None = None

    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)
