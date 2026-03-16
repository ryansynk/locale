from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np
import yaml


@dataclass
class AugmentConfig:
    # Crop settings
    max_len: int = 1024
    max_seq_len: int = 1024
    min_seq_len: int = 150
    min_overlap_percent: float = 0.2
    containment_prob: float = 0.4
    overlap_prob: float = 0.6
    min_alignment_len: int = 149
    max_alignment_ratio: float = 1.0
    min_coverage: float = 0.1
    alignment_threshold: float = 0.6

    identity_mean: float = 95
    identity_max: float = 99
    identity_stdev: float = 2.5

    # Mutation rates
    insertion_rate: float = 0.005
    deletion_rate: float = 0.005
    substitution_rate: float = 0.02

    # Mutation lengths (geometric distribution param)
    average_insertion_length: int = 10
    average_deletion_length: int = 10

    # Substitution matrix (A, C, G, T -> probs)
    # Using a dict for easy YAML mapping, but converting to something faster usually happens in __post_init__
    substitution_matrix: Dict[str, list[float]] = field(
        default_factory=lambda: {
            "A": [0.0, 0.15, 0.77, 0.08],
            "C": [0.13, 0.00, 0.17, 0.70],
            "G": [0.70, 0.17, 0.00, 0.13],
            "T": [0.08, 0.78, 0.14, 0.00],
        }
    )

    def __post_init__(self):
        # Validate logic immediately upon loading
        if not np.isclose(self.containment_prob + self.overlap_prob, 1.0):
            raise ValueError("Probabilities must sum to 1.0")

        for _, probs in self.substitution_matrix.items():
            if not np.isclose(sum(probs), 1.0):
                raise ValueError("Probabilities must sum to 1.0")

        if self.min_seq_len >= self.max_seq_len:
            raise ValueError("min_seq_len must be strictly less than max_seq_len")

        # Validate overlap percent
        if not (0.0 < self.min_overlap_percent < 1.0):
            raise ValueError("min_overlap_percent must be between 0.0 and 1.0")

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path):
        with open(path, "w") as f:
            # vars() converts the dataclass to a dict
            yaml.dump(vars(self), f, default_flow_style=False)


@dataclass
class TrainConfig:
    unsupervised: bool = False
    num_epochs: int = 1
    dataset_path: str | None = None
    val_dataset_path: str | None = None
    batch_size: int = 256
    val_batch_size: int = 512
    schedule: Literal["cosine", "hold"] = "cosine"
    lr: float = 1e-3
    backbone_lr: float = 4e-6
    total_steps: int = 1_000_000
    warmup_fraction: float | None = 0.05
    warmup_steps: int | None = None
    moco_queue_size: int = 65536
    moco_momentum: float = 0.999
    moco_softmax_temp: float = 0.07
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 1000
    num_val_queries: int = 100
    num_val_keys: int = 100_000
    dim: int = 128
    pooling: str = "mean"
    num_workers: int = 4
    sanity_test: bool = False
    augment_config: AugmentConfig = field(default_factory=AugmentConfig)
