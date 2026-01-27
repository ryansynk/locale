import random
from enum import Enum

import numpy as np

from ..config import AugmentConfig


class CropType(Enum):
    CONTAINMENT = 1
    OVERLAP = 2


class SequenceAugmenter:
    def __init__(self, config):
        self.cfg: AugmentConfig = config
        self.rng = np.random.default_rng()

    def __call__(self, seq):
        return self.augment(seq)

    def _get_crop_windows(self, seq_len):
        effective_len = min(seq_len, self.cfg.max_len)
        global_max = seq_len - effective_len + 1
        global_start = self.rng.integers(0, global_max) if global_max > 0 else 0
        # Infer cfg min_len from data
        overlap_len = self.rng.integers(self.cfg.min_alignment_len, effective_len + 1)

        # Select Strategy
        crop_type: CropType = self.rng.choice(
            a=np.array([CropType.CONTAINMENT, CropType.OVERLAP]),
            p=[self.cfg.containment_prob, self.cfg.overlap_prob],
        )

        if crop_type == CropType.CONTAINMENT:
            q_start = global_start
            q_end = global_start + effective_len

            t_max_start = q_end - overlap_len + 1
            t_start = self.rng.integers(q_start, t_max_start)
            t_end = t_start + overlap_len
        elif crop_type == CropType.OVERLAP:
            q_start = global_start
            # q_end = self.rng.integers(q_start, q_start + effective_len)

            # Query MUST be at least as long as the overlap
            min_q_end = q_start + overlap_len
            max_q_end = q_start + effective_len

            if min_q_end >= max_q_end:
                q_end = max_q_end
            else:
                q_end = self.rng.integers(min_q_end, max_q_end + 1)

            t_start = q_end - overlap_len

            # FIX 2: Prevent Index Out of Bounds
            # Target cannot go past seq_len OR past its own max length
            max_t_end = min(seq_len, t_start + effective_len)

            # FIX 3: Prevent RNG Crash (low >= high)
            if t_start + overlap_len >= max_t_end:
                t_end = max_t_end
            else:
                t_end = self.rng.integers(t_start + overlap_len, max_t_end + 1)
        else:
            raise ValueError("Invalid CropType")

        assert q_start >= 0, f"q_start {q_start} < 0"
        assert q_end > q_start, f"q_end {q_end} <= q_start {q_start}"
        assert t_start >= 0, f"t_start {t_start} < 0"
        assert t_end > t_start, f"t_end {t_end} <= t_start {t_start}"

        assert q_end <= seq_len
        assert t_end <= seq_len

        q_range = slice(q_start, q_end)
        t_range = slice(t_start, t_end)

        return q_range, t_range, overlap_len

    def _apply_deletions(self, seq, rate, overlap_len, deletion_len):
        """
        Step 2: Noise.
        Randomly remove segments from a sequence.
        Does NOT care about coordinates.
        """
        assert len(seq) > 0, "Input sequence for deletion is empty"
        num_ops = self.rng.poisson(overlap_len * rate)

        # Convert to list for mutable operations or rebuild string
        # Rebuilding string is often easier for simple augmentations
        result = seq
        for _ in range(num_ops):
            if len(result) == 0:
                break

            # Geometric length of deletion
            length = self.rng.geometric(1 / deletion_len)

            # Random position
            if len(result) - length <= 0:
                continue  # Skip if deletion is too big for remaining seq

            start = self.rng.integers(0, len(result) - length + 1)
            result = result[:start] + result[start + length :]

        return result

    def _apply_substitutions(self, seq, rate, overlap_len):
        assert len(seq) > 0, "Input sequence for substitution is empty"
        num_ops = self.rng.poisson(overlap_len * rate)
        # Implementation of substitutions...
        result = list(seq)
        seq_len = len(result)
        for _ in range(num_ops):
            bp = self.rng.integers(0, seq_len)
            probs = self.cfg.substitution_matrix[result[bp]]
            new_bp = self.rng.choice(a=np.array(["a", "c", "g", "t"]), p=probs)
            result[bp] = new_bp
        return "".join(result)

    def augment(self, seq):
        # 1. GEOMETRY: Calculate where the sequences come from
        q_loc, t_loc, overlap_len = self._get_crop_windows(len(seq))

        # 2. EXTRACTION: Get the clean substrings
        query = seq[q_loc]
        target = seq[t_loc]

        # Verify crops are not empty before processing
        assert len(query) > 0, f"Query crop is empty. Slice: {q_loc}"
        assert len(target) > 0, f"Target crop is empty. Slice: {t_loc}"

        # 3. MUTATION: Apply noise independently
        # "Insertions" in alignment are simulated by deleting from Query
        # (Query becomes shorter than Target relative to reference)
        query = self._apply_deletions(
            query,
            self.cfg.insertion_rate,
            overlap_len,
            self.cfg.average_insertion_length,
        )

        # "Deletions" in alignment are simulated by deleting from Target
        target = self._apply_deletions(
            target,
            self.cfg.deletion_rate,
            overlap_len,
            self.cfg.average_deletion_length,
        )

        # Substitutions
        query = self._apply_substitutions(
            query, self.cfg.substitution_rate, overlap_len
        )
        target = self._apply_substitutions(
            target, self.cfg.substitution_rate, overlap_len
        )
        assert len(query) <= self.cfg.max_len
        assert len(target) <= self.cfg.max_len

        # 4. RANDOMIZE ORDER
        if self.rng.random() < 0.5:
            return query, target
        else:
            return target, query


class UnitigAugmenter:
    def __init__(self, min_len=30, fragment_prob=0.8, tip_noise_prob=0.1):
        """
        :param min_len: Minimum length of a synthetic unitig.
        :param fragment_prob: Probability that the sequence is fragmented (cropped).
        :param tip_noise_prob: Probability of adding assembly artifacts to the ends.
        """
        self.min_len = min_len
        self.fragment_prob = fragment_prob
        self.tip_noise_prob = tip_noise_prob
        self.complement_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")

    def reverse_complement(self, seq):
        return seq.reverse_complement()

    def simulate_fragmentation(self, seq):
        """
        Simulates the graph breaking due to low coverage or repeats.
        Returns a random substring of the sequence.
        """
        seq_len = len(seq)
        if seq_len <= self.min_len:
            return seq

        # TODO: crop length follow unitig N50 distribution
        # We ensure the crop is at least min_len
        crop_len = random.randint(self.min_len, seq_len)
        start_pos = random.randint(0, seq_len - crop_len)
        return seq[start_pos : start_pos + crop_len]

    def augment(self, seq):
        """
        Runs the full pipeline.
        """
        # 1. Fragmentation (The most important step)
        # This turns a "Transcript" into a "Contig"
        aug_seq = self.simulate_fragmentation(seq)

        # 3. Orientation (Assemblers don't know strand)
        if random.random() < 0.5:
            aug_seq = self.reverse_complement(aug_seq)

        return aug_seq
