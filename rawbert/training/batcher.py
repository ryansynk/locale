import os
import torch
import json
from typing import List

from transformers import AutoTokenizer

from Bio import Seq


class Batcher(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        """Initializes the dataset by storing the file path."""
        self.file_path = jsonl_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        # For large files, it's better to get line offsets first
        with open(self.file_path, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        """Returns the total number of samples (lines) in the file."""
        return len(self.lines)

    def __getitem__(self, idx):
        """Fetches one sample from the file by its index."""
        line = self.lines[idx]
        data = json.loads(line)
        query = data["query"]
        reads = data["reads"]
        return self.collate(query, reads)

    def collate(self, query: str, reads: List[str]):
        query_tokens = self.tokenizer(query, return_tensors="pt")  # 1, query_length
        read_tokens = self.tokenizer(
            reads, return_tensors="pt", padding=True
        )  # num_reads, max_read_length
        return query_tokens, read_tokens
