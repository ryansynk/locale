import os
import warnings
from dataclasses import asdict
from functools import partial
from pathlib import Path

import edlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
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
    return query_tokens, queries, key_tokens, keys


def get_linear_warmup_with_hold_schedule(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


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
        pooling=cfg.pooling,
        dim=cfg.dim,
        K=cfg.moco_queue_size,
        m=cfg.moco_momentum,
        T=cfg.moco_softmax_temp,
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
            {"params": backbone_params, "lr": cfg.backbone_lr, "name": "backbone"},
            # Heads need a higher learning rate to learn quickly (e.g., 1e-3 or cfg.lr)
            {"params": head_params, "lr": cfg.lr, "name": "head"},
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
        val_reader = SupervisedBatcher(
            cfg.val_dataset_path, cfg.augment_config, num_examples=cfg.num_val_keys
        )
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
    val_dataloader = DataLoader(
        val_reader,
        batch_size=cfg.val_batch_size,
        collate_fn=collater,
        sampler=None,
        drop_last=False,
        shuffle=True,
        num_workers=1,
    )

    if getattr(cfg, "total_steps", None):
        total_steps = cfg.total_steps
        # Calculate required epochs to reach max_steps (ceiling division)
        num_epochs = (total_steps + len(dataloader) - 1) // len(dataloader)
    else:
        num_epochs = cfg.num_epochs
        total_steps = len(dataloader) * num_epochs

    assert not (cfg.warmup_steps is not None and cfg.warmup_fraction is not None), (
        "warmup_steps and warmup_fraction cannot be set simultaneously"
    )
    if cfg.warmup_steps:
        num_warmup_steps = cfg.warmup_steps
    elif cfg.warmup_fraction:
        num_warmup_steps = int(cfg.warmup_fraction * total_steps)

    if cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    elif cfg.schedule == "hold":
        scheduler = get_linear_warmup_with_hold_schedule(optimizer, num_warmup_steps)
    else:
        raise ValueError(
            f"Expected schedule to be one of 'cosine', 'hold', got: {cfg.schedule}"
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
                q, query_seqs, k, key_seqs = batch
                q = q.to(local_rank)
                k = k.to(local_rank)
                optimizer.zero_grad()
                logits, labels = ddp_rawbert(
                    q,
                    k,
                    is_distributed,
                    query_seqs=query_seqs,
                    key_seqs=key_seqs,
                    alignment_threshold=cfg.augment_config.min_coverage,
                    filter_aligned=True,
                )
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

                if global_rank == 0:
                    assert run
                    lrs = scheduler.get_last_lr()
                    metrics = {
                        "train/loss": loss.item(),
                        "train/acc1": acc1[0],
                        "train/acc5": acc5[0],
                        "train/lr": lrs[1],
                        "train/backbone_lr": lrs[0],
                        "train/step": global_step,
                    }
                    for i, group in enumerate(optimizer.param_groups):
                        norm = torch.nn.utils.get_total_norm(
                            [p.grad for p in group["params"]]
                        )
                        metrics[f"metrics/grad_norm_{group['name']}"] = norm
                    run.log(metrics)

                if global_step % cfg.checkpoint_interval == 0 and global_step > 0:
                    if global_rank == 0:
                        val_acc1, val_acc5 = get_val_accuracy(
                            val_dataloader,
                            cfg.num_val_queries,
                            cfg.num_val_keys,
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
                                    "pooling": cfg.pooling,
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
                    if is_distributed:
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
    val_dataloader,
    num_queries,
    num_keys,
    model,
    local_rank,
    tokenizer,
    augment_config,
):
    par_tqdm_write("Evaluating val accuracy")
    model.eval()

    with torch.no_grad():
        all_q = []
        all_k = []
        all_query_seqs = []
        all_key_seqs = []

        for batch in val_dataloader:
            q_tokens, query_seqs, k_tokens, key_seqs = batch
            len_q = sum([embedded_q.shape[0] for embedded_q in all_q])
            if len_q < num_queries:
                q_tokens = q_tokens.to(local_rank)
                q = model.encode(q_tokens)
                all_q.append(q)
                all_query_seqs.extend(query_seqs)

            k_tokens = k_tokens.to(local_rank)
            k = model.encode(k_tokens)
            all_k.append(k)
            all_key_seqs.extend(key_seqs)

        all_q = torch.cat(all_q, dim=0)
        all_k = torch.cat(all_k, dim=0)
        logits = torch.matmul(all_q, all_k.T)

        # Filter out aligned sequences from consideration
        alignment_threshold = augment_config.min_coverage
        logits = _filter_aligned_sequences(
            logits, all_query_seqs, all_key_seqs, alignment_threshold
        )

        labels = torch.arange(all_q.shape[0]).to(local_rank)

    acc1 = accuracy(logits, labels, topk=(1,))
    acc5 = accuracy(logits, labels, topk=(5,))

    torch.cuda.empty_cache()
    return acc1, acc5


def _filter_aligned_sequences(logits, query_seqs, key_seqs, alignment_threshold):
    """
    Filter out aligned sequences from validation logits.

    Args:
        logits: Tensor of shape (num_queries, num_keys) with similarity scores
        query_seqs: List of query DNA sequences
        key_seqs: List of key DNA sequences
        alignment_threshold: Minimum similarity to consider sequences aligned

    Returns:
        Filtered logits with aligned non-diagonal entries set to -1e9
    """
    num_queries = len(query_seqs)
    num_keys = len(key_seqs)
    total_filtered = []
    for i in range(num_queries):
        query_seq = query_seqs[i]

        num_filtered = 0
        for j in range(num_keys):
            # Skip the diagonal (true positive pair)
            if i == j:
                continue

            key_seq = key_seqs[j]

            # Check if sequences are aligned using edlib
            result = edlib.align(query=query_seq, target=key_seq, task="distance")
            edit_distance = result["editDistance"]

            # Calculate similarity
            max_length = max(len(query_seq), len(key_seq))
            similarity = 1.0 - (edit_distance / max_length)

            # If aligned, mask out this logit
            if similarity >= alignment_threshold:
                num_filtered += 1
                logits[i, j] = -1e9

        total_filtered.append(num_filtered)

    avg_filtered = sum(total_filtered) / num_queries
    max_filtered = max(total_filtered)
    min_filtered = min(total_filtered)
    par_tqdm_write(
        f"Val filtering stats: mean = {avg_filtered:.2f}, max = {max_filtered}, min = {min_filtered}"
    )
    return logits
