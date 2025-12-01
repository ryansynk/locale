import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from Bio import SeqIO
from jsonargparse import auto_cli
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Get the path to the current script, go up one level to the root, and add to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import directly from the package
from rawbert.modeling.rawbert import RawBERT
from rawbert.training.batcher import Batcher


def collate(batch, tokenizer):
    # batch is list of (query, key) pairs
    queries, keys = zip(*batch)
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, key_tokens


def main(gencode_fasta: str, batch_size: int, num_sequences: int = 200):
    gencode_fasta = Path(gencode_fasta).resolve()
    transcripts = list(SeqIO.parse(gencode_fasta, "fasta"))
    transcripts = sorted(transcripts, key=lambda seq: len(str(seq.seq)), reverse=True)
    # only take top 100 length sequences
    transcripts = transcripts[:num_sequences]
    reader = Batcher(None, transcripts)
    collater = partial(collate, tokenizer=reader.tokenizer)
    dataloader = DataLoader(reader, batch_size, shuffle=False, collate_fn=collater)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rawbert = RawBERT(dim=128, K=4096, m=0.999)
    rawbert = rawbert.to(device)
    rawbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, rawbert.parameters()), lr=0.9)
    successful_lengths = []
    unsuccessful_lengths = []
    for batch in tqdm(dataloader):
        q, k = batch
        seqlen = q.input_ids.shape[1]
        bsize = q.input_ids.shape[0]
        print(
            f"Attempting forward with sequence length = {seqlen}, batch size = {bsize}"
        )
        try:
            logits, labels = rawbert(q, k)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if bsize == batch_size:
                successful_lengths.append(seqlen)
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory, batch size {bsize}, sequence length = {seqlen}")
            unsuccessful_lengths.append(seqlen)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    if successful_lengths:
        print(f"Largest sequence length fitting in memory: {max(successful_lengths)}")
    else:
        print(f"No sequences fit in memory with batch size {batch_size}!")
    print(f"Total sequences that OOM'd = {len(unsuccessful_lengths)}")


if __name__ == "__main__":
    auto_cli(main)
