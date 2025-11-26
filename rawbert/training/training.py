import random
import time
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from rawbert.modeling.rawbert import RawBERT
from rawbert.training.batcher import Batcher


def save_checkpoint(state, checkpoint_dir):
    filename = (checkpoint_dir / "checkpoint.pth.tar").resolve()
    torch.save(state, filename)


def evaluate(model, dataloader, num_batches):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in islice(dataloader, num_batches):
            q, k = batch
            logits, labels = model(q, k)
            loss += F.cross_entropy(logits, labels)
    return loss / num_batches


def collate(batch, tokenizer):
    # batch is list of (query, key) pairs
    queries, keys = zip(*batch)
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, key_tokens


def train(
    dataset_path,
    test_dataset_path,
    device,
    batch_size,
    lr,
    epochs,
    dim,
    moco_queue_size,
    moco_momentum,
    num_test_batches,
    checkpoint_dir,
    run,
):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    reader = Batcher(dataset_path)
    test_reader = Batcher(test_dataset_path)
    collater = partial(collate, tokenizer=reader.tokenizer)

    dataloader = DataLoader(reader, batch_size, shuffle=True, collate_fn=collater)
    test_dataloader = DataLoader(
        test_reader, batch_size, shuffle=True, collate_fn=collater
    )

    rawbert = RawBERT(dim=dim, K=moco_queue_size, m=moco_momentum)
    rawbert = rawbert.to(device)
    rawbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, rawbert.parameters()), lr=lr)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    global_step = 0
    log_interval = 1000

    for epoch in range(epochs):
        rawbert.train()
        for batch in tqdm(dataloader):
            q, k = batch
            N = q.input_ids.shape[1]
            # TODO: Hack, doesn't train on long examples to prevent OOM
            if N <= 10000:
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
                        "global_step": global_step,
                    }
                )

                if global_step % log_interval == 0 and global_step > 0:
                    test_loss = evaluate(rawbert, test_dataloader, num_test_batches)
                    run.log(
                        {
                            "test_loss": test_loss,
                            "global_step": global_step,
                        }
                    )
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "model": rawbert.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        checkpoint_dir,
                    )
                    rawbert.train()

                global_step += 1
