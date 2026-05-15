import random
from pathlib import Path

from Bio import SeqIO
from torch.utils.data import Dataset

from ..config import AugmentConfig


class ReferenceBatcher(Dataset):
    def __init__(
        self,
        dataset_path: str,
        augment_config: AugmentConfig,
        num_examples: int | None = None,
    ):
        dataset_path: Path = Path(dataset_path).resolve()
        self.cfg = augment_config
        # self.augmenter = Augmenter(augment_config)
        # self.df = pl.read_parquet(dataset_path)
        # self.disable_mutations = self.cfg.disable_mutations
        # if num_examples is not None:
        #    self.df = self.df.head(num_examples)

        ## The absolute maximum length a long_seq could ever be
        self.max_window_size = int(
            self.cfg.max_read_len * self.cfg.max_containment_ratio
        )

        self.sequences = {}
        self.chunks = []

        # 1. Load sequences using Biopython and create non-overlapping chunks
        fasta_files = [
            f for ext in ("*.fna", "*.fa", "*.fasta") for f in dataset_path.rglob(ext)
        ]
        for file_path in fasta_files:
            # SeqIO.parse returns an iterator of SeqRecord objects
            for record in SeqIO.parse(file_path, "fasta"):
                seq_id = record.id
                # Convert Biopython Seq object to a standard uppercase Python string
                seq_str = str(record.seq).upper()

                self.sequences[seq_id] = seq_str
                seq_len = len(seq_str)

                # Create non-overlapping windows to ensure sampling without replacement
                for start_idx in range(
                    0, seq_len - self.max_window_size + 1, self.max_window_size
                ):
                    self.chunks.append((seq_id, start_idx))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        # Fetch the specific non-overlapping window
        seq_id, window_start = self.chunks[index]
        full_seq = self.sequences[seq_id]

        # 1. Determine lengths
        l_short = random.randint(self.cfg.min_read_len, self.cfg.max_read_len)
        l_long_max = min(
            int(l_short * self.cfg.max_containment_ratio), self.max_window_size
        )
        l_long = random.randint(l_short, l_long_max)

        # 2. Position the long_seq randomly within our max window
        long_start_rel = random.randint(0, self.max_window_size - l_long)
        long_start_abs = window_start + long_start_rel

        # 3. Position the short_seq randomly entirely within the long_seq
        short_start_rel = random.randint(0, l_long - l_short)
        short_start_abs = long_start_abs + short_start_rel

        # 4. Extract strings
        long_seq = full_seq[long_start_abs : long_start_abs + l_long]
        short_seq = full_seq[short_start_abs : short_start_abs + l_short]

        # 5. Randomly assign to query and key
        if random.random() < 0.5:
            query, key = short_seq, long_seq
        else:
            query, key = long_seq, short_seq

        return query, key
