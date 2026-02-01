import os
import warnings
from dataclasses import asdict
from functools import partial
from itertools import batched  # ty: ignore unresolved-import
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from rawbert.config import TrainConfig
from rawbert.modeling.model import RawBERT
from rawbert.training.batcher import Batcher
from rawbert.training.supervised_batcher import SupervisedBatcher
from wandb import Run


def par_print(*args, **kwargs):
    if dist.is_initialized():  # ty: ignore possibly-missing-attribute
        if dist.get_rank() == 0:  # ty: ignore possibly-missing-attribute
            print(*args, **kwargs)
    else:
        # Fallback for single GPU runs so it still works
        print(*args, **kwargs)


def par_tqdm_write(*args, **kwargs):
    if dist.is_initialized():  # ty: ignore possibly-missing-attribute
        if dist.get_rank() == 0:  # ty: ignore possibly-missing-attribute
            tqdm.write(*args, **kwargs)
    else:
        # Fallback for single GPU runs so it still works
        tqdm.write(*args, **kwargs)


def save_checkpoint(state, checkpoint_dir, cfg, run_id):
    this_ckpt_dir: Path = Path(checkpoint_dir / run_id).resolve()
    this_ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(this_ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f)

    filename = (this_ckpt_dir / "checkpoint.pth.tar").resolve()
    par_tqdm_write(f"Saving checkpoint to {str(filename)}")
    torch.save(state, filename)


def collate(batch, tokenizer):
    # batch is list of (query, key) pairs
    queries, keys = zip(*batch)
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, key_tokens


def train(
    cfg: TrainConfig,
    per_device_batch_size: int,
    run: Run | None,
    local_rank: int,
    global_rank: int,
    world_size: int,
    is_distributed: bool,
):
    warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")
    warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
    transformers_logging.set_verbosity_error()
    # Set the device for this process
    torch.cuda.set_device(local_rank)

    # Instantiate model and move to the correct GPU
    rawbert = RawBERT(
        dim=cfg.dim, K=cfg.moco_queue_size, m=cfg.moco_momentum, T=cfg.moco_softmax_temp
    )
    rawbert = rawbert.to(local_rank)
    rawbert.train()
    backbone_params = list(
        filter(lambda p: p.requires_grad, rawbert.bert_q.parameters())
    )
    head_params = list(
        filter(lambda p: p.requires_grad, rawbert.projector_q.parameters())
    )
    optimizer = AdamW(
        [
            # Backbones usually need a much lower learning rate (e.g., 1e-5)
            {"params": backbone_params, "lr": cfg.backbone_lr},
            # Heads need a higher learning rate to learn quickly (e.g., 1e-3 or cfg.lr)
            {"params": head_params, "lr": cfg.lr},
        ]
    )

    # Wrap model with DDP only if distributed
    if is_distributed:
        ddp_rawbert = DDP(rawbert, device_ids=[local_rank])
    else:
        ddp_rawbert = rawbert

    if cfg.unsupervised:
        par_print("Unsupervised Training Mode")
        reader = Batcher(cfg.dataset_path, cfg.augment_config)
        sampler = None
    else:
        par_print("Supervised Training Mode")
        reader = SupervisedBatcher(cfg.dataset_path, cfg.augment_config)
        sampler = (
            DistributedSampler(
                reader, num_replicas=world_size, rank=global_rank, shuffle=True
            )
            if is_distributed
            else None
        )
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    collater = partial(collate, tokenizer=tokenizer)

    dataloader = DataLoader(
        reader,
        batch_size=per_device_batch_size,
        collate_fn=collater,
        sampler=sampler,
        drop_last=True,
        shuffle=False,
        num_workers=int(os.environ["OMP_NUM_THREADS"]),
    )

    if getattr(cfg, "total_steps", None):
        total_steps = cfg.total_steps
        # Calculate required epochs to reach max_steps (ceiling division)
        num_epochs = (total_steps + len(dataloader) - 1) // len(dataloader)
    else:
        num_epochs = cfg.num_epochs
        total_steps = len(dataloader) * num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    if cfg.checkpoint_dir:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("No checkpoint_dir provided!")

    global_step = 0
    ddp_rawbert.train()

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for epoch in range(num_epochs):
            par_tqdm_write(f"Training epoch = {epoch + 1}/{num_epochs}")
            for batch in dataloader:
                q, k = batch
                q = q.to(local_rank)
                k = k.to(local_rank)
                optimizer.zero_grad()
                logits, labels = ddp_rawbert(q, k, is_distributed)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

                if global_rank == 0:
                    assert run
                    lrs = scheduler.get_last_lr()
                    run.log(
                        {
                            "train/loss": loss.item(),
                            "train/acc1": acc1[0],
                            "train/acc5": acc5[0],
                            "train/lr": lrs[1],
                            "train/backbone_lr": lrs[0],
                            "train/step": global_step,
                        }
                    )

                if global_step % cfg.checkpoint_interval == 0 and global_step > 0:
                    if global_rank == 0:
                        val_acc1, val_acc5 = get_val_accuracy(
                            cfg.val_dataset_path,
                            cfg.num_val_queries,
                            cfg.num_val_keys,
                            cfg.val_batch_size,
                            ddp_rawbert.module if is_distributed else ddp_rawbert,
                            local_rank,
                            tokenizer,
                            cfg.augment_config,
                        )
                        raw_model = (
                            ddp_rawbert.module if is_distributed else ddp_rawbert
                        )

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
                                    "dim": cfg.dim,
                                    "K": cfg.moco_queue_size,
                                    "m": cfg.moco_momentum,
                                    "T": cfg.moco_softmax_temp,
                                },
                            },
                            checkpoint_dir,
                            cfg,
                            run.id,
                        )
                    torch.distributed.barrier()
                    ddp_rawbert.train()

                global_step += 1
                pbar.update(1)
                if getattr(cfg, "total_steps", None) and global_step >= total_steps:
                    break

            # Check for step-based termination (Outer Loop)
            if getattr(cfg, "total_steps", None) and global_step >= total_steps:
                break


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


def get_val_accuracy(
    val_dataset_path,
    num_queries,
    num_keys,
    batch_size,
    model,
    local_rank,
    tokenizer,
    augment_config,
):
    par_tqdm_write("Evaluating val accuracy")
    model.eval()

    df = pl.read_parquet(val_dataset_path).sort("query_name")
    df = df.with_columns(
        pl.col("query_seq").str.len_chars().alias("query_seq_len"),
        pl.col("reference_seq").str.len_chars().alias("reference_seq_len"),
    ).filter(
        (pl.col("reference_seq_len") < augment_config.max_len)
        & (pl.col("query_seq_len") < augment_config.max_len)
    )
    queries = df.head(num_queries)["query_seq"].to_list()
    keys = df.head(num_keys)["reference_seq"].to_list()

    with torch.no_grad():
        all_q = []
        all_k = []

        q = tokenizer(queries, return_tensors="pt", padding=True).to(local_rank)
        q = model.encode(q)
        all_q.append(q)

        for batch in batched(keys, batch_size):
            k = tokenizer(batch, return_tensors="pt", padding=True).to(local_rank)
            k = model.encode(k)
            all_k.append(k)

        all_q = torch.cat(all_q, dim=0)
        all_k = torch.cat(all_k, dim=0)
        logits = torch.matmul(all_q, all_k.T)
        labels = torch.arange(all_q.shape[0]).to(local_rank)

    acc1 = accuracy(logits, labels, topk=(1,))
    acc5 = accuracy(logits, labels, topk=(5,))

    torch.cuda.empty_cache()
    return acc1, acc5
