from dataclasses import dataclass
from typing import Dict

import numpy as np
import yaml


@dataclass
class AugmentConfig:
    # Crop settings
    max_len: int
    min_seq_len: int
    containment_prob: float
    overlap_prob: float
    min_alignment_len: int

    # Mutation rates
    insertion_rate: float
    deletion_rate: float
    substitution_rate: float

    # Mutation lengths (geometric distribution param)
    average_insertion_length: float
    average_deletion_length: float

    # Substitution matrix (A, C, G, T -> probs)
    # Using a dict for easy YAML mapping, but converting to something faster usually happens in __post_init__
    substitution_matrix: Dict[str, list[float]]

    def __post_init__(self):
        # Validate logic immediately upon loading
        if not np.isclose(self.containment_prob + self.overlap_prob, 1.0):
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
