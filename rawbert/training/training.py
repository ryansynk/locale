import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import BatchEncoding
from rawbert.training.batcher import Batcher
from rawbert.modeling.rawbert import RawBERT


def collate_fn(samples: list[tuple[BatchEncoding, BatchEncoding]], pad_token: int):
    # def collate_fn(batch_list, pad_token_id: int):
    # Extract tensors from all BatchEncodings
    max_query_len = max([sample[0].input_ids.shape[-1] for sample in samples])
    max_num_reads = max([sample[1].input_ids.shape[0] for sample in samples])
    max_seq_len = max([sample[1].input_ids.shape[1] for sample in samples])

    def pad_tensor(t, target_n, target_l, pad_val):
        n, l = t.shape
        out = torch.full((target_n, target_l), pad_val, dtype=t.dtype, device=t.device)
        out[:n, :l] = t
        return out

    padded_query_ids = torch.stack(
        [
            pad_tensor(sample[0].input_ids, 1, max_query_len, pad_token)
            for sample in samples
        ],
        dim=0,
    )
    padded_query_mask = torch.stack(
        [
            pad_tensor(sample[0].attention_mask, 1, max_query_len, 0)
            for sample in samples
        ],
        dim=0,
    )
    padded_queries = {
        "input_ids": padded_query_ids,
        "attention_mask": padded_query_mask,
    }
    padded_reads_ids = torch.stack(
        [
            pad_tensor(sample[1].input_ids, max_num_reads, max_seq_len, pad_token)
            for sample in samples
        ],
        dim=0,
    )
    padded_reads_mask = torch.stack(
        [
            pad_tensor(sample[1].attention_mask, max_num_reads, max_seq_len, 0)
            for sample in samples
        ],
        dim=0,
    )
    padded_reads = {"input_ids": padded_reads_ids, "attention_mask": padded_reads_mask}
    return padded_queries, padded_reads


def train(dataset_path, device, batch_size, lr, epochs, dim, single_batch, run):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    reader = Batcher(dataset_path)
    collater = partial(collate_fn, pad_token=reader.tokenizer.pad_token_id)

    if single_batch:
        reader = Subset(reader, range(batch_size))

    dataloader = DataLoader(reader, batch_size, shuffle=True, collate_fn=collater)

    rawbert = RawBERT(dim=dim)
    rawbert = rawbert.to(device)
    rawbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, rawbert.parameters()), lr=lr)

    start_time = time.time()

    for _ in range(epochs):
        for batch in dataloader:
            queries, reads = batch
            scores = rawbert(queries, reads)
            labels = torch.arange(scores.size(0), device=scores.device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed = float(time.time() - start_time)

            run.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/batch_size": batch_size,
                    "train/time_elapsed": elapsed,
                }
            )
