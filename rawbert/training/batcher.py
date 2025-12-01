import torch
from Bio import SeqIO
from transformers import AutoTokenizer

from .augmenter import UnitigAugmenter


class Batcher(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, seq_list=None):
        """Initializes the dataset by storing the file path."""
        if seq_list is None:
            assert jsonl_path is not None
            self.file_path = jsonl_path
            # For large files, it's better to get line offsets first
            self.lines = list(SeqIO.parse(self.file_path, "fasta"))
        else:
            # Can take in an already instantiated sequence list as well
            self.file_path = None
            self.lines = seq_list
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.augmenter = UnitigAugmenter(min_len=100)

    def __len__(self):
        """Returns the total number of samples (lines) in the file."""
        return len(self.lines)

    def __getitem__(self, idx):
        """Fetches one sample from the file by its index."""
        transcript = self.lines[idx]
        query = str(transcript.seq)
        key = self.augmenter.augment(transcript).seq
        return query, str(key)
