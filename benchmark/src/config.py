"""Benchmark configuration: a method is an encoder plus an index, or a
non-dense tool (metagraph, mmseqs).

Layout on disk (labels are directory names chosen by the caller; every
``config.json`` is the identity the directory was built or searched under and
is checked before anything is reused)::

    <index_dir>/<encoder_label>/                config.json (EncoderConfig identity),
                                                embeddings.fbin, meta.parquet, .done
    <index_dir>/<encoder_label>/<index_label>/  config.json (engine + BUILD fields), codes, .done
    <index_dir>/<label>/                        metagraph / mmseqs: config.json, ..., .done
    <results_dir>/<encoder_label>/<index_label>/<search_label>/
                                                config.json (engine + SEARCH fields + strands,
                                                num_queries, seed), mut<rate>.parquet,
                                                hits/mut<rate>.parquet
    <results_dir>/<label>/<search_label>/       metagraph / mmseqs results

Each index dataclass declares which of its fields change the artifact on disk
(``BUILD``) and which change only the hits (``SEARCH``). That is the only
naming logic in the codebase; engines read their own dataclass and nothing
else from here.
"""

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Literal, Union

# The LOCALE checkpoint published with the paper, downloaded when a locale
# config leaves checkpoint_path unset.
#
# ckpt_id and step are pinned here rather than parsed from the file. For a local
# checkpoint both are read off the path -- the parent directory is the wandb run
# id and the digits after "checkpoint" are the step -- but a Hub download lands
# in a content-addressed cache under a blob hash, which encodes neither. Pinning
# them is what makes a downloaded checkpoint carry the same identity as the
# original run, so its config.json matches an index built from the local copy.
#
# revision is a commit sha, not a branch: a downloaded checkpoint that silently
# changes underneath a published benchmark is exactly the failure this pin
# exists to prevent.
PAPER_CHECKPOINT = {
    "repo_id": "rsynk/locale",
    "filename": "checkpoint5859.pth.tar",
    "revision": "2ab2b18f0b93bd0051e5a7d9e4e8696123cfab5a",
    "ckpt_id": "8vqiabk9",
    "step": 5859,
}

CONFIG_FILE = "config.json"
DONE_FILE = ".done"


def results_file_name(mutation_rate: float) -> str:
    """Results (and hits) file for one mutation rate: mut0.10.parquet."""
    return f"mut{mutation_rate:.2f}.parquet"


def _check_label(label: str, what: str) -> None:
    if not label or "/" in label or label in (".", "..") or label.startswith("."):
        raise ValueError(f"{what} must be a plain directory name, got {label!r}")


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #


@dataclass
class EncoderConfig:
    """How sequences become vectors. ``identity()`` is what an index directory
    is stamped with; batch_size / device / query_embed_gpus / both_strands do
    not change the vectors (both_strands is a query-time choice and lives in
    the search identity instead)."""

    name: Literal["locale", "dna2vec", "llmed"]
    checkpoint_path: str | None = None  # locale only; None = PAPER_CHECKPOINT
    pooling: str = "mean"
    max_seq_len: int = 256
    chunk_overlap: int = 150
    batch_size: int = 128
    # Also embed each query's reverse complement and union the two hit lists
    # before the top_k cut (dense methods only; metagraph/mmseqs already see
    # both strands).
    both_strands: bool = True
    # Embed query chunks with one encoder replica per GPU (the first
    # query_embed_gpus visible devices) instead of only on `device`. Search
    # time only; the embeddings are the same up to GPU float noise.
    query_embed_gpus: int = 1
    device: str = "cuda"

    def __post_init__(self):
        if self.name not in ("locale", "dna2vec", "llmed"):
            raise ValueError(
                f"name expected: locale, dna2vec, or llmed. Got = {self.name}"
            )
        if self.chunk_overlap >= self.max_seq_len:
            raise ValueError("chunk_overlap must be strictly less than max_seq_len")

    def checkpoint(self) -> tuple[str | None, int | None]:
        """(run id, step) of the checkpoint; (None, None) for pretrained encoders."""
        if self.name != "locale":
            return None, None
        if self.checkpoint_path is None:
            return PAPER_CHECKPOINT["ckpt_id"], PAPER_CHECKPOINT["step"]
        p = Path(self.checkpoint_path).resolve()
        return p.parent.name, int(p.name.split(".")[0][len("checkpoint"):])

    def identity(self) -> dict:
        ckpt, step = self.checkpoint()
        return {
            "name": self.name,
            "checkpoint": ckpt,
            "step": step,
            "pooling": self.pooling,
            "max_seq_len": self.max_seq_len,
            "chunk_overlap": self.chunk_overlap,
        }


# --------------------------------------------------------------------------- #
# index engines
# --------------------------------------------------------------------------- #


@dataclass
class ExactIndex:
    """Exact fp32 top-k scan over embeddings.fbin (no artifact of its own)."""

    BUILD: ClassVar[tuple[str, ...]] = ()
    SEARCH: ClassVar[tuple[str, ...]] = ("top_k",)
    engine: Literal["exact"] = "exact"
    top_k: int = 100


@dataclass
class ExhaustiveIndex:
    """Reference protocol: every accession scored by the max over ALL its
    vectors (streaming per-accession scan). No hits are persisted."""

    BUILD: ClassVar[tuple[str, ...]] = ()
    SEARCH: ClassVar[tuple[str, ...]] = ()
    engine: Literal["exhaustive"] = "exhaustive"


@dataclass
class RaBitQIndex:
    """1-bit RaBitQ codes (src/rabitq.py); sample_rows rows estimate the centroid."""

    BUILD: ClassVar[tuple[str, ...]] = ("sample_rows",)
    SEARCH: ClassVar[tuple[str, ...]] = ("top_k",)
    engine: Literal["rabitq"] = "rabitq"
    top_k: int = 100
    sample_rows: int = 2_000_000


@dataclass
class IVFPQIndex:
    """GPU IVF-PQ (cuVS, src/ivfpq_gpu.py): num_shards independent IVF-PQ
    indexes over row ranges of the fbin, lists_per_shard lists each, pq_dim x
    pq_bits codes, searched with nprobe lists per shard; the best rerank
    candidates per query chunk are re-scored exactly from the fbin (0 = PQ
    estimates only)."""

    BUILD: ClassVar[tuple[str, ...]] = ("pq_dim", "pq_bits", "lists_per_shard", "num_shards")
    SEARCH: ClassVar[tuple[str, ...]] = ("top_k", "nprobe", "rerank", "lut")
    engine: Literal["ivfpq"] = "ivfpq"
    top_k: int = 100
    pq_dim: int = 128
    pq_bits: int = 8
    lists_per_shard: int = 4096
    num_shards: int = 16
    nprobe: int = 34
    rerank: int = 0
    lut: Literal["float16", "float32"] = "float16"


@dataclass
class IVFRaBitQIndex:
    """IVF + RaBitQ (faiss, src/ivf_rabitq.py): nlist spherical k-means cells
    with nb_bits RaBitQ residual codes, nprobe cells scanned per query chunk,
    the best rerank candidates re-scored from the fbin. quantizer "hnsw" puts
    an HNSW graph over the centroids; fastscan converts the codes to faiss'
    SIMD 4-bit LUT layout at load time. Built multi-node by build_ivf.py."""

    BUILD: ClassVar[tuple[str, ...]] = ("nlist", "nb_bits", "train_rows")
    SEARCH: ClassVar[tuple[str, ...]] = ("top_k", "nprobe", "rerank", "qb", "quantizer", "fastscan")
    engine: Literal["ivfrabitq"] = "ivfrabitq"
    top_k: int = 100
    nlist: int = 16384
    nb_bits: int = 1
    train_rows: int = 6_000_000
    nprobe: int = 512
    rerank: int = 300
    qb: int = 8
    quantizer: Literal["flat", "hnsw"] = "flat"
    fastscan: bool = False


IndexConfig = Union[ExactIndex, ExhaustiveIndex, RaBitQIndex, IVFPQIndex, IVFRaBitQIndex]
INDEX_TYPES: tuple[type, ...] = (ExactIndex, ExhaustiveIndex, RaBitQIndex, IVFPQIndex, IVFRaBitQIndex)


def build_identity(index) -> dict:
    """Engine name + BUILD fields: what an index directory is stamped with."""
    return {"engine": index.engine, **{f: getattr(index, f) for f in index.BUILD}}


def search_fields(index) -> dict:
    return {f: getattr(index, f) for f in index.SEARCH}


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #


@dataclass
class DenseMethod:
    """An encoder plus an index. The three labels name the directories the
    artifacts live in (see the module docstring); they carry identity, not
    parameters -- the parameters are in config.json and checked on load."""

    encoder: EncoderConfig
    index: IndexConfig = field(default_factory=ExactIndex)
    encoder_label: str = "locale"
    index_label: str = "exact"
    search_label: str = "default"

    def __post_init__(self):
        _check_label(self.encoder_label, "encoder_label")
        _check_label(self.index_label, "index_label")
        _check_label(self.search_label, "search_label")

    def labels(self) -> dict[str, str | None]:
        return {
            "encoder": self.encoder_label,
            "index": self.index_label,
            "search": self.search_label,
        }

    def index_path(self, index_dir: Path) -> Path:
        """The encoder's directory: embeddings.fbin + meta.parquet."""
        return Path(index_dir) / self.encoder_label

    def engine_path(self, index_dir: Path) -> Path:
        return self.index_path(index_dir) / self.index_label

    def results_path(self, results_dir: Path) -> Path:
        return Path(results_dir) / self.encoder_label / self.index_label / self.search_label

    def index_identity(self) -> dict:
        return self.encoder.identity()

    def engine_identity(self) -> dict:
        return build_identity(self.index)

    def search_identity(self) -> dict:
        return {
            "engine": self.index.engine,
            **search_fields(self.index),
            "both_strands": self.encoder.both_strands,
        }

    def __str__(self):
        return "/".join(v for v in self.labels().values() if v)


@dataclass
class MetagraphConfig:
    name: str = "metagraph"
    executable: str = "metagraph"
    k: int = 31
    # server_query accepts this many connections at once (its -p) and the
    # client splits a query batch into as many concurrent requests; 1 is the
    # original single-request protocol, where one server thread answers the
    # whole batch. Rankings are per query, so results do not change -- only
    # wall time, which is why >1 belongs under its own search_label.
    server_parallel: int = 1
    label: str = "metagraph"
    search_label: str = "default"

    def __post_init__(self):
        _check_label(self.label, "label")
        _check_label(self.search_label, "search_label")

    def labels(self) -> dict[str, str | None]:
        return {"encoder": None, "index": self.label, "search": self.search_label}

    def index_path(self, index_dir: Path) -> Path:
        return Path(index_dir) / self.label

    def results_path(self, results_dir: Path) -> Path:
        return Path(results_dir) / self.label / self.search_label

    def index_identity(self) -> dict:
        return {"engine": "metagraph", "k": self.k}

    def search_identity(self) -> dict:
        return {"engine": "metagraph", "server_parallel": self.server_parallel}

    def __str__(self):
        return f"{self.label}/{self.search_label}"


@dataclass
class MMseqs2Config:
    name: str = "mmseqs"
    executable: str = "mmseqs"
    max_seqs: int = 300
    label: str = "mmseqs"
    search_label: str = "default"

    def __post_init__(self):
        _check_label(self.label, "label")
        _check_label(self.search_label, "search_label")

    def labels(self) -> dict[str, str | None]:
        return {"encoder": None, "index": self.label, "search": self.search_label}

    def index_path(self, index_dir: Path) -> Path:
        return Path(index_dir) / self.label

    def results_path(self, results_dir: Path) -> Path:
        return Path(results_dir) / self.label / self.search_label

    def index_identity(self) -> dict:
        return {"engine": "mmseqs"}

    def search_identity(self) -> dict:
        return {"engine": "mmseqs", "max_seqs": self.max_seqs}

    def __str__(self):
        return f"{self.label}/{self.search_label}"


MethodConfig = Union[DenseMethod, MetagraphConfig, MMseqs2Config]


@dataclass
class ExperimentConfig:
    model: MethodConfig
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
    # Multi-node build: every node (node 0 included) exits as soon as its shard
    # is .done, skipping the serial merge so GPU nodes are not held for it.
    # Run finish_merge.py (CPU) afterwards, then search with the index .done.
    build_only: bool = False
    # Smoke-test knob: cap how many accessions enter the index so an end-to-end
    # run finishes in minutes. Leave unset for real runs — a truncated index is
    # still marked .done, so always pair this with a throwaway index_dir.
    max_accessions: int | None = None

    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)


def run_search_identity(cfg: ExperimentConfig) -> dict:
    """The method's search identity plus the query draw: what a results
    directory is stamped with. (A function, not a method: jsonargparse's CLI
    turns public methods of ExperimentConfig into subcommands.)"""
    return {
        **cfg.model.search_identity(),
        "num_queries": cfg.num_queries,
        "random_seed": cfg.random_seed,
    }


# --------------------------------------------------------------------------- #
# config.json: written on build/search, checked on load
# --------------------------------------------------------------------------- #


class ConfigMismatch(RuntimeError):
    pass


def _canonical(identity: dict) -> dict:
    return json.loads(json.dumps(identity, sort_keys=True))


def read_config(directory: Path) -> dict | None:
    p = Path(directory) / CONFIG_FILE
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def write_config(directory: Path, identity: dict) -> None:
    """Atomic write of directory/config.json (concurrent identical writers are safe)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{CONFIG_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(_canonical(identity), f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, directory / CONFIG_FILE)


def check_config(directory: Path, identity: dict) -> None:
    """Raise ConfigMismatch if directory/config.json exists and differs."""
    on_disk = read_config(directory)
    if on_disk is None:
        return
    want = _canonical(identity)
    if on_disk != want:
        diff = {
            k: (on_disk.get(k), want.get(k))
            for k in sorted(set(on_disk) | set(want))
            if on_disk.get(k) != want.get(k)
        }
        raise ConfigMismatch(
            f"{Path(directory) / CONFIG_FILE} was written for a different "
            f"configuration (on disk, requested): {diff}. Use another label or "
            "delete the directory."
        )


def ensure_config(directory: Path, identity: dict, artifact_present: bool) -> None:
    """Check config.json against ``identity`` or write it.

    ``artifact_present`` says whether the directory already holds the thing
    the config describes (a .done index, a results parquet); such a directory
    without config.json was not built by this code and is refused rather
    than trusted.
    """
    on_disk = read_config(directory)
    if on_disk is not None:
        check_config(directory, identity)
        return
    if artifact_present:
        raise ConfigMismatch(
            f"{directory} holds an artifact but no {CONFIG_FILE}; it was not "
            "built under this layout. Stamp it with the identity it was built "
            "under, or use another label."
        )
    write_config(directory, identity)


def dataclass_to_dict(obj) -> dict:
    """Plain dict of a config dataclass, recursing into nested ones."""
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if hasattr(v, "__dataclass_fields__"):
            v = dataclass_to_dict(v)
        elif isinstance(v, Path):
            v = str(v)
        out[f.name] = v
    return out
