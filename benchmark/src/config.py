from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

# The LOCALE checkpoint published with the paper, downloaded when a locale
# config leaves checkpoint_path unset.
#
# ckpt_id and step are pinned here rather than parsed from the file. For a local
# checkpoint both are read off the path -- the parent directory is the wandb run
# id and the digits after "checkpoint" are the step -- but a Hub download lands
# in a content-addressed cache under a blob hash, which encodes neither. Pinning
# them is what makes a downloaded checkpoint produce the same index_suffix and
# experiment_id as the original run, so results land beside the paper's instead
# of in a directory named after a hash.
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
    # --- Scoring protocol (search-time only; the built index is identical) ---
    # Default: retrieve each query's top_k nearest index VECTORS, group the
    # hits by accession, score each accession by its max hit and rank;
    # accessions with no hit sit at MISS_SCORE, below every scored one. The
    # raw hits are persisted under <topk_hits_dir>/<hits_id>/ and reused by
    # any later run of the same encoder/dataset/strands whose top_k is no
    # larger (the ranking at k depends only on the first k hits), so a k-sweep
    # is one scan at the largest k followed by cheap reruns.
    top_k: int = 100
    # Reference protocol: score every accession by the max over ALL its
    # vectors (the streaming per-accession scan). No hits are persisted.
    exhaustive: bool = False
    # Vector engine for the top-k protocol. Default is the exact fp32 scan;
    # use_ann searches a CAGRA graph, use_rabitq scans 1-bit RaBitQ codes.
    use_ann: bool = False
    use_rabitq: bool = False
    # Also embed each query's reverse complement and union the two hit lists
    # before the top_k cut (dense methods only; metagraph/mmseqs already see
    # both strands). Suffixes experiment_id with _bothstrands.
    both_strands: bool = False
    # Deprecated no-op kept so older configs still parse: the exact top-k
    # regroup it used to switch on is now the default protocol.
    exact_search: bool = False
    # Rows sampled (in contiguous blocks) to estimate the RaBitQ centroid,
    # instead of a full pass over the fbin. Ignored unless use_rabitq.
    rabitq_sample_rows: int = 2_000_000

    def __post_init__(self):
        if self.use_ann and self.use_rabitq:
            raise ValueError("use_ann and use_rabitq are mutually exclusive")
        if self.exhaustive and (self.use_ann or self.use_rabitq or self.exact_search):
            raise ValueError(
                "exhaustive scores every accession directly; it has no vector "
                "engine (use_ann/use_rabitq/exact_search)"
            )
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
            if self.checkpoint_path is None:
                # Unset means the published checkpoint; dense_index fetches it.
                ckpt_id = PAPER_CHECKPOINT["ckpt_id"]
                self.checkpoint_step_num = PAPER_CHECKPOINT["step"]
            else:
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
        # Every scoring protocol reads the same built index (index_suffix is
        # unchanged) but ranks differently, so each gets its own experiment_id.
        # The bare id stays with the exhaustive per-accession max: that is what
        # the existing full-dense baseline directories were written under, and
        # what the top-k protocols are compared against. hits_id is the
        # experiment_id without k: the persisted hits directory, shared by
        # every k of the same encoder/engine/strands.
        strands = "_bothstrands" if self.both_strands else ""
        if self.exhaustive:
            self.hits_id: str | None = None
        else:
            if self.use_rabitq:
                engine, sep = "rabitq1bit", "_top"
            elif self.use_ann:
                engine, sep = "cagra", "_top"
            else:
                engine, sep = "exact", "top"
            self.hits_id = f"{self.experiment_id}_{engine}{strands}"
            self.experiment_id = f"{self.experiment_id}_{engine}{sep}{self.top_k}"
        self.experiment_id = f"{self.experiment_id}{strands}"

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
class CentroidConfig(AlgorithmConfig):
    """Bag-of-centroids index over embeddings produced by ``encoder``.

    This *composes* a DenseConfig rather than restating its fields, so any of the
    encoders works with this index and the two axes stay independent.

    ``source_index_dir`` points at a prebuilt dense index directory
    (``embeddings.fbin`` + ``meta.parquet``); the build clusters those vectors
    instead of re-embedding the FASTA.
    """

    encoder: DenseConfig = field(default_factory=DenseConfig)
    name: str = "centroid"
    source_index_dir: Optional[str] = None
    num_centroids: int = 4096
    nprobe: int = 32
    sample_size: int = 1_000_000
    kmeans_iters: int = 25
    device: str = "cuda"
    # Rows per streaming tile in the assignment pass. 500k x 768 float32 ~ 1.5 GB,
    # matching the tile size vecdb_dataset/ground_truth.py settled on for the same
    # NFS-read-bound scan.
    tile_rows: int = 500_000
    probe_weight: Literal["sim", "softmax", "uniform"] = "sim"
    softmax_temperature: float = 0.05
    random_seed: int = 0

    def __post_init__(self):
        # num_centroids changes the index on disk; nprobe and probe_weight are
        # query-time only, so one built index serves every setting of them. They
        # belong in experiment_id (results differ) but not in index_suffix.
        config_tag = f"K{self.num_centroids}"
        self.index_suffix: Path = (
            Path("centroid") / self.encoder.index_suffix / config_tag
        )
        self.experiment_id: str = (
            f"centroid_{self.encoder.experiment_id}_{config_tag}"
            f"_p{self.nprobe}_w{self.probe_weight}"
        )
        self.checkpoint: str | None = self.encoder.checkpoint
        self.max_len: int | None = self.encoder.max_len
        self.chunk_type: int | None = None
        self.checkpoint_step_num: int | None = self.encoder.checkpoint_step_num

    def __str__(self):
        return self.experiment_id


@dataclass
class ExperimentConfig:
    model: Union[DenseConfig, MetagraphConfig, MMseqs2Config, CentroidConfig]
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
    # Where a top-k run persists each query's raw vector hits, and where later
    # runs at a smaller top_k find them instead of scanning again. Kept OUT of
    # results_dir: print_results.py rglobs every parquet
    # under results_dir against a strict schema, and the hits artifact does not
    # match it. Unset means a sibling of results_dir named
    # "<results_dir>_topk_hits".
    topk_hits_dir: Path | None = None

    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)
        if self.topk_hits_dir is None:
            self.topk_hits_dir = self.results_dir.with_name(
                self.results_dir.name + "_topk_hits"
            )
