import time
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

from rawbert.modeling.rawbert import RawBERT
from rawbert.training.batcher import Batcher


def par_print(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        # Fallback for single GPU runs so it still works
        print(*args, **kwargs)


def save_checkpoint(state, checkpoint_dir):
    filename = (checkpoint_dir / "checkpoint.pth.tar").resolve()
    par_print(f"Saving checkpoint to {str(filename)}")
    torch.save(state, filename)


def collate(batch, tokenizer):
    # batch is list of (query, key) pairs
    queries, keys = zip(*batch)
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, key_tokens


def train(
    dataset_path,
    test_dataset_path,
    batch_size,
    per_device_batch_size,
    lr,
    epochs,
    dim,
    moco_queue_size,
    moco_momentum,
    moco_softmax_temp,
    checkpoint_dir,
    run,
    local_rank,
    global_rank,
    world_size,
    is_distributed,
    checkpoint_interval,
):
    warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")
    warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
    transformers_logging.set_verbosity_error()
    # Set the device for this process
    torch.cuda.set_device(local_rank)

    # Instantiate model and move to the correct GPU
    rawbert = RawBERT(dim=dim, K=moco_queue_size, m=moco_momentum, T=moco_softmax_temp)
    rawbert = rawbert.to(local_rank)
    rawbert.train()

    # Wrap model with DDP only if distributed
    if is_distributed:
        ddp_rawbert = DDP(rawbert, device_ids=[local_rank])
    else:
        ddp_rawbert = rawbert

    reader = Batcher(dataset_path)
    test_reader = Batcher(test_dataset_path)
    collater = partial(collate, tokenizer=reader.tokenizer)

    if is_distributed:
        sampler = DistributedSampler(reader)
        test_sampler = DistributedSampler(test_reader)
        dataloader = DataLoader(
            reader,
            batch_size=per_device_batch_size,
            sampler=sampler,
            collate_fn=collater,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test_reader,
            batch_size=per_device_batch_size,
            sampler=test_sampler,
            collate_fn=collater,
        )
    else:
        sampler = None
        # In single GPU mode, we just shuffle normally
        dataloader = DataLoader(
            reader, per_device_batch_size, shuffle=True, collate_fn=collater
        )
        test_dataloader = DataLoader(
            test_reader, per_device_batch_size, shuffle=True, collate_fn=collater
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, ddp_rawbert.parameters()), lr=lr
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    global_step = 0

    for epoch in range(epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        ddp_rawbert.train()

        for batch in tqdm(dataloader):
            q, k = batch
            q = q.to(local_rank)
            k = k.to(local_rank)
            optimizer.zero_grad()
            logits, labels = ddp_rawbert(q, k, is_distributed)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            elapsed = float(time.time() - start_time)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            if global_rank == 0:
                run.log(
                    {
                        "train/loss": loss.item(),
                        "train/acc1": acc1[0],
                        "train/acc5": acc5[0],
                        "train/lr": lr,
                        "train/global_batch_size": batch_size,
                        "train/per_device_batch_size": per_device_batch_size,
                        "train/time_elapsed": elapsed,
                        "train/step": global_step,
                    }
                )

            if global_step % checkpoint_interval == 0 and global_step > 0:
                if global_rank == 0:
                    test_acc1, test_acc5 = get_test_accuracy(
                        test_dataloader,
                        ddp_rawbert.module if is_distributed else ddp_rawbert,
                        local_rank,
                        temperature=moco_softmax_temp,
                    )
                    raw_model = ddp_rawbert.module if is_distributed else ddp_rawbert

                    run.log(
                        {
                            "test/acc1": test_acc1[0],
                            "test/acc5": test_acc5[0],
                            "test/step": global_step,
                        }
                    )
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        checkpoint_dir,
                    )
                torch.distributed.barrier()
                ddp_rawbert.train()

            global_step += 1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_test_accuracy(test_loader, model, local_rank, temperature=0.07):
    par_print("Evaluating test accuracy")
    model.eval()

    # store all queries and keys
    all_q = []
    all_k = []

    with torch.no_grad():
        for batch in test_loader:
            # Assuming inputs is a pair or your loader splits them
            # x_q = query sequences, x_k = key sequences
            q, k = batch
            q = q.to(local_rank)
            k = k.to(local_rank)
            # seq is (B, N_max, D)
            q = model._embed_q(q)
            q = F.normalize(q, dim=1)  # (B, D)
            all_q.append(q)

            # update key encoder
            k = model._embed_k(k)
            k = F.normalize(k, dim=1)  # (B, D)
            all_k.append(k)

    # Concatenate all features
    all_q = torch.cat(all_q, dim=0)
    all_k = torch.cat(all_k, dim=0)

    # 3. Compute logits: (N_test, N_test)
    # Every row i is the query i compared against ALL keys
    logits = torch.matmul(all_q, all_k.T) / temperature

    # 4. Create labels
    # The positive for query i is at index i (the diagonal)
    labels = torch.arange(all_q.shape[0]).to(local_rank)

    # 5. Calculate Accuracy
    acc1 = accuracy(logits, labels, topk=(1,))
    acc5 = accuracy(logits, labels, topk=(5,))

    return acc1, acc5
