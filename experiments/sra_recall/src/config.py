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
class DenseConfig(AlgorithmConfig):
    name: Literal[
        "rawbert", "dnabert", "generator", "neuroseed", "dna2vec", "llmed", "evo2"
    ] = "rawbert"  # "dnabert", "rawbert"
    checkpoint_path: Optional[str] = None
    checkpoint_step_num: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"
    pooling: str = "max"
    k: int = 100
    max_seq_len: int = 1024
    chunk_type: Literal["stride", "exact_chunk"] = "stride"
    chunk_overlap: int = 150
    neuroseed_path: Optional[str] = "/pscratch/sd/r/rsynk/NeuroSEED"
    use_ann: bool = False

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
            self.checkpoint_step_num = int(
                Path(self.checkpoint_path).name.split(".")[0][10:]
            )
            self.index_suffix: Path = (
                Path("rawbert") / ckpt_id / str(self.checkpoint_step_num) / config_tag
            )
            self.experiment_id: str = (
                f"rawbert_{ckpt_id}_{str(self.checkpoint_step_num)}_{config_tag}"
            )
            self.checkpoint: str | None = ckpt_id
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "neuroseed":
            assert self.checkpoint_path is not None
            ckpt_id = Path(self.checkpoint_path).resolve().stem
            self.index_suffix: Path = Path("neuroseed") / ckpt_id / config_tag
            self.experiment_id: str = f"neuroseed_{ckpt_id}_{config_tag}"
            self.checkpoint: str | None = ckpt_id
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "dna2vec":
            self.index_suffix: Path = Path("dna2vec") / config_tag
            self.experiment_id: str = f"dna2vec_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "llmed":
            # Abusing checkpoint path to distinguish between different kinds of llmed model
            self.index_suffix: Path = Path("llmed") / config_tag
            self.experiment_id: str = f"llmed_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        elif self.name == "evo2":
            self.index_suffix: Path = Path("evo2") / config_tag
            self.experiment_id: str = f"evo2_{config_tag}"
            self.checkpoint: str | None = None
            self.max_len: int = self.max_seq_len
            self.chunk_type: str = self.chunk_type
        else:
            raise ValueError(
                f"name expected: rawbert, dnabert, generator, neuroseed, dna2vec, llmed, or evo2. Got = {self.name}"
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
class MantisConfig(AlgorithmConfig):
    name: str = "mantis"
    executable: str = "mantis"
    seqtk_executable: str = "seqtk"
    squeakr_executable: str = "squeakr"
    k: int = 31
    log_slots: int = 30
    num_threads: int = 32

    def __post_init__(self):
        self.index_suffix = Path("mantis") / f"k{self.k}"
        self.experiment_id = f"mantis_k{self.k}"
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
    model: Union[DenseConfig, MetagraphConfig, MantisConfig, MMseqs2Config]
    accessions_dir: Path
    raw_read_queries_path: Path
    logan_contig_queries_path: Path
    gencode_queries_path: Path
    index_dir: Path
    results_dir: Path
    query_type: Literal["raw_read", "logan_contig", "gencode"]
    mutation_rate: float = 0.0
    do_timing: bool = False
    timing_runs: int = 5
    num_queries: int = 1000
    random_seed: int = 1337
    no_search: bool = False

    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)
