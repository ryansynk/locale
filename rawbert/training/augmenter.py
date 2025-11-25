import random


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
