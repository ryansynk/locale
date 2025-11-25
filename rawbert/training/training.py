import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from rawbert.training.batcher import Batcher
from rawbert.modeling.rawbert import RawBERT


def collate(batch, tokenizer):
    # batch is list of (query, key) pairs
    queries, keys = zip(*batch)
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, key_tokens


def train(
    dataset_path,
    device,
    batch_size,
    lr,
    epochs,
    dim,
    single_batch,
    run,
):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    reader = Batcher(dataset_path)
    collater = partial(collate, tokenizer=reader.tokenizer)

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
            q, k = batch
            logits, labels = rawbert(q, k)
            loss = F.cross_entropy(logits, labels)
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
