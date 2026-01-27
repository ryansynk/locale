import time
import warnings
from functools import partial
from itertools import batched, islice  # ty: ignore unresolved-import
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from rawbert.modeling.model import RawBERT
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
    augment_config,
    num_val_queries,
    num_val_keys,
    batch_size,
    per_device_batch_size,
    lr,
    total_steps,
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
    sanity_test,
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

    reader = Batcher(dataset_path, augment_config)
    # test_reader = Batcher(test_dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    collater = partial(collate, tokenizer=tokenizer)

    if is_distributed:
        sampler = DistributedSampler(reader)
        dataloader = DataLoader(
            reader,
            batch_size=per_device_batch_size,
            sampler=sampler,
            collate_fn=collater,
            drop_last=True,
        )
    else:
        sampler = None
        # In single GPU mode, we just shuffle normally
        dataloader = DataLoader(reader, per_device_batch_size, collate_fn=collater)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, ddp_rawbert.parameters()), lr=lr
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    global_step = 0

    if is_distributed:
        sampler.set_epoch(0)
    ddp_rawbert.train()

    if sanity_test:
        par_print("Running in sanity test mode. Overfitting on a single batch.")
        # Grab a single batch from the dataloader
        single_batch = next(iter(dataloader))
        # Create an iterator that yields the same batch indefinitely
        data_iterator = (single_batch for _ in range(total_steps))
    else:
        data_iterator = islice(dataloader, total_steps)

    for batch in tqdm(data_iterator, total=total_steps):
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
                    "train/time_elapsed": elapsed,
                    "train/step": global_step,
                }
            )

        if global_step % checkpoint_interval == 0 and global_step > 0:
            if global_rank == 0:
                val_acc1, val_acc5 = get_test_accuracy(
                    test_dataset_path,
                    num_val_queries,
                    num_val_keys,
                    batch_size,
                    collate,
                    ddp_rawbert.module if is_distributed else ddp_rawbert,
                    local_rank,
                )
                raw_model = ddp_rawbert.module if is_distributed else ddp_rawbert

                run.log(
                    {
                        "val/acc1": val_acc1[0],
                        "val/acc5": val_acc5[0],
                        "val/step": global_step,
                    }
                )
                save_checkpoint(
                    {
                        "step": global_step,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": {
                            "dim": dim,
                            "K": moco_queue_size,
                            "m": moco_momentum,
                            "T": moco_softmax_temp,
                        },
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


def get_test_accuracy(
    test_dataset_path,
    num_queries,
    num_keys,
    batch_size,
    collate_fn,
    model,
    local_rank,
    temperature=0.07,
):
    par_print("Evaluating test accuracy")
    model.eval()

    # store all queries and keys
    all_q = []
    all_k = []

    with torch.no_grad():
        df = pl.read_parquet(test_dataset_path)
        queries = df.select("query_seq").limit(num_queries).to_list()
        keys = df.select("query_seq").limit(num_keys).to_list()

        q = collate_fn(queries)
        q = model._embed_q(q)
        q = F.normalize(q, dim=1)  # (B, D)
        all_q.append(q)

        for batch in batched(keys, batch_size):
            k = collate_fn(batch)
            k = k.to(local_rank)
            # update key encoder
            k = model._embed_q(k)
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
