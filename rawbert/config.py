from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import yaml


@dataclass
class AugmentConfig:
    # Crop settings
    max_len: int = 5000
    min_seq_len: int = 150
    containment_prob: float = 0.66
    overlap_prob: float = 0.34
    min_alignment_len: int = 150

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
    dataset_path: str | None = None
    test_dataset_path: str | None = None
    batch_size: int = 128
    lr: float = 4e-6
    total_steps: int = 1_000_000
    moco_queue_size: int = 65536
    moco_momentum: float = 0.999
    moco_softmax_temp: float = 0.05
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 1000
    num_val_queries: int = 100
    num_val_keys: int = 100_000
    dim: int = 128
    sanity_test: bool = False
    augment_config: AugmentConfig = field(default_factory=AugmentConfig)
