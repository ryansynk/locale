import math
from enum import Enum
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset

from ..config import AugmentConfig


class ContainmentBatcher(Dataset):
    def __init__(
        self,
        dataset_path: str,
        augment_config: AugmentConfig,
        num_examples: int | None = None,
    ):
        dataset_path: Path = Path(dataset_path).resolve()
        self.cfg = augment_config
        self.augmenter = Augmenter(augment_config)
        self.df = pl.read_parquet(dataset_path)
        self.df = self.df.filter(
            pl.col("sequence_len")
            >= math.ceil(self.cfg.max_read_len * self.cfg.max_containment_ratio)
        )
        self.disable_mutations = self.cfg.disable_mutations
        if num_examples is not None:
            self.df = self.df.head(num_examples)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        seq = row["sequence"].upper()
        query, ref = self.augmenter.get_pairs(seq)
        return query, ref


class CropType(Enum):
    OVERLAP = 1
    CONTAINMENT = 2


class Augmenter:
    def __init__(self, config):
        self.cfg: AugmentConfig = config
        # self.rng = np.random.default_rng()
        self.beta_a, self.beta_b = beta_parameters(
            self.cfg.identity_mean, self.cfg.identity_stdev, self.cfg.identity_max
        )
        self.beta_distribution = torch.distributions.beta.Beta(self.beta_a, self.beta_b)

    def __call__(self, seq):
        return self.get_pairs(seq)

    def _get_crop_lens(self, seq):
        short_crop_len = torch.randint(
            self.cfg.min_read_len, self.cfg.max_read_len + 1, size=(1,)
        ).item()
        assert isinstance(short_crop_len, int)
        long_crop_len_ratio = (self.cfg.max_containment_ratio - 1.0) * torch.rand(
            size=(1,)
        ).item() + 1.0
        upper_long_crop_len: int = math.ceil(short_crop_len * long_crop_len_ratio)
        long_crop_len = torch.randint(
            short_crop_len, min(upper_long_crop_len, len(seq)), size=(1,)
        ).item()
        return short_crop_len, long_crop_len

    def _containment_crop(self, seq, crop_len_1, crop_len_2):
        short_crop_len = min(crop_len_1, crop_len_2)
        long_crop_len = max(crop_len_1, crop_len_2)
        start_long: int = int(
            torch.randint(0, len(seq) - long_crop_len + 1, size=(1,)).item()
        )
        start_short = torch.randint(
            start_long,
            high=(start_long + long_crop_len - short_crop_len + 1),
            size=(1,),
        ).item()

        long_crop = seq[start_long : (start_long + long_crop_len)]
        short_crop = seq[start_short : (start_short + short_crop_len)]
        if torch.rand(size=(1,)).item() < 0.5:
            query = long_crop
            query_range = (
                start_short - start_long,
                start_short - start_long + len(short_crop),
            )
            ref = short_crop
            ref_range = (0, len(short_crop))
        else:
            query = short_crop
            query_range = (0, len(short_crop))
            ref = long_crop
            ref_range = (
                start_short - start_long,
                start_short - start_long + len(short_crop),
            )

        return query, query_range, ref, ref_range

    def _sample_identity(self):
        return self.beta_distribution.sample().item()

    @staticmethod
    def _uniform_random_mutation(
        sequence: str, target_identity: float
    ) -> tuple[str, float]:
        """
        Applies uniform random substitutions, insertions, and deletions
        in a single pass to reach an approximate target identity.
        """
        BASES = ["A", "C", "G", "T"]
        seq_len = len(sequence)
        # 1. Mathematically define the exact edit distance needed
        target_errors = int(round(seq_len * (1.0 - target_identity)))

        if target_errors <= 0:
            return sequence, 1.0

        # Cap errors at sequence length to prevent sampling errors
        target_errors = min(target_errors, seq_len)

        # 2. Select mutation indices
        # Using random.sample ensures we don't pick the same index twice.
        # Converting to a set makes the lookup O(1) in the loop below.
        mutation_indices = set(torch.randperm(seq_len)[:target_errors].tolist())

        mut_probs = torch.tensor([0.4, 0.3, 0.3])
        mut_types = torch.multinomial(
            mut_probs, target_errors, replacement=True
        ).tolist()
        ins_bases_idx = torch.randint(0, 4, (target_errors,)).tolist()
        sub_bases_idx = torch.randint(0, 3, (target_errors,)).tolist()

        new_seq_parts = []
        actual_errors = 0

        muts = ["sub", "ins", "del"]
        # 3. Construct the sequence in a single O(N) pass
        for i, base in enumerate(sequence):
            if i in mutation_indices:
                # Randomly select the mutation type
                # (Weights can be adjusted, e.g., Nanopore has more indels than subs)
                # mut_type_idx = int(torch.multinomial(mut_probs, 1).item())
                mut_type = muts[mut_types.pop()]

                if mut_type == "sub":
                    # Substitute with a DIFFERENT base
                    bases_choices = [b for b in BASES if b != base]
                    idx = sub_bases_idx.pop() % len(bases_choices)
                    new_seq_parts.append(bases_choices[idx])
                    actual_errors += 1
                elif mut_type == "ins":
                    # Insert a random base BEFORE the current base
                    new_seq_parts.append(BASES[ins_bases_idx.pop()])
                    new_seq_parts.append(base)
                    actual_errors += 1
                elif mut_type == "del":
                    # Simply don't append the current base to the new sequence
                    actual_errors += 1
            else:
                # No mutation assigned to this index, keep the original base
                new_seq_parts.append(base)

        final_seq = "".join(new_seq_parts)

        # 4. Calculate actual identity mathematically
        # We use the max length between original and final to mirror alignment denominators
        actual_identity = 1.0 - (actual_errors / max(seq_len, len(final_seq)))

        return final_seq, actual_identity

    # Example execution:
    # original = "ATGCGTACGTAGCTAGCTAG" * 25  # 500 bp
    # augmented, actual_id = uniform_random_mutation(original, 0.85)

    @staticmethod
    def augment(seq, identity):
        augmented_seq, final_identity = Augmenter._uniform_random_mutation(
            seq, identity
        )
        return augmented_seq

    def get_pairs(self, seq):
        crop_len_1, crop_len_2 = self._get_crop_lens(seq)

        seq1, seq1_overlap_range, seq2, seq2_overlap_range = self._containment_crop(
            seq, crop_len_1, crop_len_2
        )

        if self.cfg.disable_mutations is False:
            seq1_overlap_start, seq1_overlap_end = seq1_overlap_range
            seq2_overlap_start, seq2_overlap_end = seq2_overlap_range

            identity = self._sample_identity()
            augmented_overlap = Augmenter.augment(
                seq1[seq1_overlap_start:seq1_overlap_end], identity
            )

            if torch.rand(size=(1,)).item() < 0.5:
                seq1 = (
                    seq1[:seq1_overlap_start]
                    + augmented_overlap
                    + seq1[seq1_overlap_end:]
                )
            else:
                seq2 = (
                    seq2[:seq2_overlap_start]
                    + augmented_overlap
                    + seq2[seq2_overlap_end:]
                )

        if torch.rand(size=(1,)).item() < 0.5:
            query = seq1
            ref = seq2
        else:
            query = seq2
            ref = seq1

        return query, ref


def beta_parameters(beta_mean, beta_stdev, beta_max):
    u, s, m = beta_mean, beta_stdev, beta_max
    beta_a = (((1 - (u / m)) / ((s / m) ** 2)) - (m / u)) * ((u / m) ** 2)
    beta_b = beta_a * ((m / u) - 1)
    if beta_a < 0.0 or beta_b < 0.0:
        raise ValueError(
            "Error: invalid beta parameters for identity distribution - trying increasing "
            "the maximum identity or reducing the standard deviation"
        )
    return beta_a, beta_b
