from Bio import SeqIO
from jsonargparse import auto_cli
from transformers import AutoTokenizer
from pathlib import Path
import torch.nn.functional as F
import torch
from torch.optim import AdamW

import sys
import os

# Get the path to the current script, go up one level to the root, and add to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import directly from the package
from rawbert.training.augmenter import UnitigAugmenter
from rawbert.modeling.rawbert import RawBERT

def main(gencode_fasta: str):
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    augmenter = UnitigAugmenter(min_len=100)

    gencode_fasta = Path(gencode_fasta).resolve()
    transcripts = list(SeqIO.parse(gencode_fasta, "fasta"))
    transcripts = sorted(transcripts, key= lambda seq: len(str(seq.seq)))
    # only take top 100 length sequences
    transcripts = transcripts[-100:]
    print("Preparing query, key pairs")
    data = [(str(transcript.seq), str(augmenter.augment(transcript).seq)) for transcript in transcripts]
    print("Tokenizing")
    data = [(tokenizer(query, return_tensors="pt"), tokenizer(key, return_tensors="pt")) for (query, key) in data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rawbert = RawBERT(dim=128, K=4096, m=0.999)
    rawbert = rawbert.to(device)
    rawbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, rawbert.parameters()), lr=0.9)

    successful_lengths = []
    unsuccessful_lengths = []
    for q, k in data:
        seqlen = q.input_ids.shape[1]
        print(f"Attempting forward with sequence length = {seqlen}")
        try:
            logits, labels = rawbert(q, k)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            successful_lengths.append(seqlen)
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory, batch size 1, sequence length = {seqlen}")
            unsuccessful_lengths.append(seqlen)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print(f"Largest sequence length fitting in memory: {max(successful_lengths)}")
    print(f"Total sequences that OOM'd = {len(unsuccessful_lengths)}")
        

if __name__=="__main__":
    auto_cli(main)