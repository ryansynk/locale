import warnings
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import cast

import edlib  # ty: ignore unresolved-import
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from lae.config import TrainConfig, AugmentConfig
from lae.modeling.model import LOCALE
from lae.training.containment_batcher import ContainmentBatcher
from lae.training.reference_batcher import ReferenceBatcher
from lae.training.supervised_batcher import SupervisedBatcher
from lae.training.unsupervised_batcher import UnsupervisedBatcher, Augmenter
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


def save_checkpoint(state, checkpoint_dir, cfg, run_id, kl: bool = False):
    this_ckpt_dir: Path = Path(checkpoint_dir / run_id).resolve()
    if kl:
        this_ckpt_dir = this_ckpt_dir / "kl"
    this_ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(this_ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f)

    step = state["step"]
    filename = (this_ckpt_dir / f"checkpoint{step}.pth.tar").resolve()
    par_tqdm_write(f"Saving checkpoint to {str(filename)}")
    torch.save(state, filename)


def collate(batch, tokenizer):
    # batch is list of dict[str, str]
    queries = [b["query"] for b in batch]
    keys = [b["ref"] for b in batch]
    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    return query_tokens, queries, key_tokens, keys


def collate_w_hard_negatives(batch, tokenizer):
    # batch is list of dict[str, str]
    queries = [b["query"] for b in batch]
    keys = [b["ref"] for b in batch]
    negatives = [b["negative"] for b in batch]
    neg_none_indices: list[int] = []
    for i, neg in enumerate(negatives):
        if neg is None:
            neg_none_indices.append(i)
    if neg_none_indices:
        none_indices = set(neg_none_indices)
        queries = [q for i, q in enumerate(queries) if i not in none_indices]
        keys = [k for i, k in enumerate(keys) if i not in none_indices]
        negatives = [n for i, n in enumerate(negatives) if i not in none_indices]

    query_tokens = tokenizer(queries, return_tensors="pt", padding=True)
    key_tokens = tokenizer(keys, return_tensors="pt", padding=True)
    negative_tokens = tokenizer(negatives, return_tensors="pt", padding=True)
    return query_tokens, queries, key_tokens, keys, negative_tokens, negatives


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
    locale = LOCALE(
        pooling=cfg.pooling,
        dim=cfg.dim,
        K=cfg.moco_queue_size,
        m=cfg.moco_momentum,
        T=cfg.moco_softmax_temp,
        use_projection_head=cfg.use_projection_head,
    )
    if cfg.moco_filter_queue:
        locale.set_kmer_k(cfg.moco_filter_queue_identity_cutoff)
    locale = locale.to(local_rank)
    locale.train()

    global_batch_size = per_device_batch_size * world_size
    ratio = global_batch_size / cfg.reference_global_batch_size
    scaled_backbone_lr = cfg.backbone_lr * math.sqrt(ratio)
    scaled_proj_lr = cfg.lr * math.sqrt(ratio)
    backbone_params = list(
        filter(lambda p: p.requires_grad, locale.bert_q.parameters())
    )
    if cfg.use_projection_head:
        assert locale.projector_q
        head_params = list(
            filter(lambda p: p.requires_grad, locale.projector_q.parameters())
        )
        optimizer = AdamW(
            [
                # Backbones usually need a much lower learning rate (e.g., 1e-5)
                {
                    "params": backbone_params,
                    "lr": scaled_backbone_lr,
                    "name": "backbone",
                },
                # Heads need a higher learning rate to learn quickly (e.g., 1e-3 or cfg.lr)
                {"params": head_params, "lr": scaled_proj_lr, "name": "head"},
            ]
        )
    else:
        optimizer = AdamW(
            [
                # Backbones usually need a much lower learning rate (e.g., 1e-5)
                {
                    "params": backbone_params,
                    "lr": scaled_backbone_lr,
                    "name": "backbone",
                },
            ]
        )

    # Wrap model with DDP only if distributed
    if is_distributed:
        ddp_locale = DDP(locale, device_ids=[local_rank])
    else:
        ddp_locale = locale

    val_config: AugmentConfig = AugmentConfig(
        disable_mutations=True,
        min_seq_len=100,
        max_seq_len=256,
        containment_prob=1.0,
        overlap_prob=0.0,
    )

    if cfg.unsupervised:
        par_print("Unsupervised Training Mode")
        if cfg.data_type == "contig":
            reader = UnsupervisedBatcher(
                cfg.dataset_path, cfg.use_hard_negatives, cfg.augment_config
            )
            val_reader = UnsupervisedBatcher(
                cfg.val_dataset_path,
                False,
                val_config,
                num_examples=cfg.num_val_keys,
            )
            sampler = (
                DistributedSampler(
                    reader, num_replicas=world_size, rank=global_rank, shuffle=True
                )
                if is_distributed
                else None
            )
        elif cfg.data_type == "reference":
            full_reader = ReferenceBatcher(cfg.dataset_path, cfg.augment_config)
            total_size = len(full_reader)
            train_size = total_size - cfg.num_val_keys

            # torch.manual_seed(42) # Optional: uncomment for reproducible splits
            reader, val_reader = random_split(
                full_reader, [train_size, cfg.num_val_keys]
            )
            sampler = (
                DistributedSampler(
                    reader, num_replicas=world_size, rank=global_rank, shuffle=True
                )
                if is_distributed
                else None
            )
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
    hn_collater = partial(collate_w_hard_negatives, tokenizer=tokenizer)
    collater = partial(collate, tokenizer=tokenizer)

    dataloader = DataLoader(
        reader,
        batch_size=per_device_batch_size,
        collate_fn=hn_collater if cfg.use_hard_negatives else collater,
        sampler=sampler,
        drop_last=True,
        shuffle=False,
        num_workers=cfg.num_workers,
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

    if cfg.total_samples is not None:
        total_steps = cfg.total_samples // global_batch_size
        # Calculate required epochs to reach total_steps (ceiling division)
        num_epochs = (total_steps + len(dataloader) - 1) // len(dataloader)
    else:
        num_epochs = cfg.num_epochs
        total_steps = len(dataloader) * num_epochs

    checkpoint_interval_steps = cfg.checkpoint_interval_samples // global_batch_size

    assert not (cfg.warmup_samples is not None and cfg.warmup_fraction is not None), (
        "warmup_samples and warmup_fraction cannot be set simultaneously"
    )
    if cfg.warmup_samples:
        num_warmup_steps = cfg.warmup_samples // global_batch_size
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
    ddp_locale.train()

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for epoch in range(num_epochs):
            par_tqdm_write(f"Training epoch = {epoch + 1}/{num_epochs}")
            for batch in dataloader:
                if cfg.use_hard_negatives:
                    q, query_seqs, k, key_seqs, neg_tokens, neg_seqs = batch
                else:
                    q, query_seqs, k, key_seqs = batch
                    neg_tokens = None
                    neg_seqs = None
                q = q.to(local_rank)
                k = k.to(local_rank)
                optimizer.zero_grad()
                logits, labels = ddp_locale(
                    q,
                    k,
                    cfg.moco_filter_queue_identity_cutoff,
                    is_distributed=is_distributed,
                    query_seqs=query_seqs,
                    key_seqs=key_seqs,
                    filter_aligned=cfg.moco_filter_queue,
                    neg_tokens=neg_tokens,
                    neg_seqs=neg_seqs,
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
                        "train/lr": lrs[0],
                        "train/samples_seen": global_step * global_batch_size,
                    }
                    if cfg.use_projection_head:
                        metrics["train/head_lr"] = lrs[1]
                    for i, group in enumerate(optimizer.param_groups):
                        norm = torch.nn.utils.get_total_norm(
                            [p.grad for p in group["params"]]
                        )
                        metrics[f"metrics/grad_norm_{group['name']}"] = norm
                    run.log(metrics)

                if global_step % checkpoint_interval_steps == 0 and global_step > 0:
                    if global_rank == 0:
                        assert run
                        if is_distributed:
                            module = ddp_locale.module
                        else:
                            module = ddp_locale

                        assert isinstance(module, torch.nn.Module)
                        model_state_dict = module.state_dict()

                        (
                            val_acc1,
                            val_acc5,
                            val_acc1_mut_95,
                            val_acc5_mut_95,
                            val_acc1_mut_90,
                            val_acc5_mut_90,
                        ) = get_val_accuracy(
                            val_dataloader,
                            cfg.num_val_queries,
                            cfg.num_val_keys,
                            module,
                            local_rank,
                            tokenizer,
                            cfg.moco_filter_queue_identity_cutoff,
                            val_config,
                        )

                        run.log(
                            {
                                "val/acc1_100_identity": val_acc1[0],
                                "val/acc5_100_identity": val_acc5[0],
                                "val/acc1_95_identity": val_acc1_mut_95[0],
                                "val/acc5_95_identity": val_acc5_mut_95[0],
                                "val/acc1_90_identity": val_acc1_mut_90[0],
                                "val/acc5_90_identity": val_acc5_mut_90[0],
                                "val/samples_seen": global_step * global_batch_size,
                            }
                        )
                        save_checkpoint(
                            {
                                "step": global_step,
                                "model": model_state_dict,
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
                    ddp_locale.train()

                global_step += 1
                pbar.update(1)
                if cfg.total_samples is not None and global_step >= total_steps:
                    break

            # Check for step-based termination (Outer Loop)
            if cfg.total_samples is not None and global_step >= total_steps:
                break

    if global_rank == 0:
        assert run
        if is_distributed:
            module = ddp_locale.module
        else:
            module = ddp_locale

        assert isinstance(module, torch.nn.Module)
        model_state_dict = module.state_dict()

        (
            val_acc1,
            val_acc5,
            val_acc1_mut_95,
            val_acc5_mut_95,
            val_acc1_mut_90,
            val_acc5_mut_90,
        ) = get_val_accuracy(
            val_dataloader,
            cfg.num_val_queries,
            cfg.num_val_keys,
            module,
            local_rank,
            tokenizer,
            cfg.moco_filter_queue_identity_cutoff,
            val_config,
        )

        run.log(
            {
                "val/acc1_100_identity": val_acc1[0],
                "val/acc5_100_identity": val_acc5[0],
                "val/acc1_95_identity": val_acc1_mut_95[0],
                "val/acc5_95_identity": val_acc5_mut_95[0],
                "val/acc1_90_identity": val_acc1_mut_90[0],
                "val/acc5_90_identity": val_acc5_mut_90[0],
                "val/samples_seen": global_step * global_batch_size,
            }
        )
        save_checkpoint(
            {
                "step": global_step,
                "model": model_state_dict,
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


def train_kl(
    cfg: TrainConfig,
    per_device_batch_size: int,
    run: Run | None,
    local_rank: int,
    global_rank: int,
    world_size: int,
    is_distributed: bool,
):
    # TODO FIX A BUNCH OF SHIT AND SCALE LEARNING RATE
    par_print("Executing KL Divergence Training")
    warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")
    warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
    transformers_logging.set_verbosity_error()
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    assert cfg.starting_checkpoint_path, "No checkpoint given"
    checkpoint = torch.load(cfg.starting_checkpoint_path)
    locale = LOCALE(
        pooling="max",
        dim=checkpoint["model_args"]["dim"],
        K=checkpoint["model_args"]["K"],
        m=checkpoint["model_args"]["m"],
        T=checkpoint["model_args"]["T"],
    )
    locale.load_state_dict(checkpoint["model"])
    locale = locale.to(local_rank)
    locale.train()
    backbone_params = list(
        filter(lambda p: p.requires_grad, locale.bert_q.parameters())
    )

    optimizer = AdamW(
        [
            # Backbones usually need a much lower learning rate (e.g., 1e-5)
            {"params": backbone_params, "lr": cfg.backbone_lr, "name": "backbone"},
        ]
    )

    # Wrap model with DDP only if distributed
    if is_distributed:
        ddp_locale = DDP(locale, device_ids=[local_rank])
    else:
        ddp_locale = locale

    par_print("Unsupervised Training Mode")
    reader = UnsupervisedBatcher(cfg.dataset_path, cfg.augment_config)
    val_reader = UnsupervisedBatcher(
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
        num_workers=cfg.num_workers,
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

    global_batch_size = per_device_batch_size * world_size

    if cfg.total_samples is not None:
        total_steps = cfg.total_samples // global_batch_size
        # Calculate required epochs to reach total_steps (ceiling division)
        num_epochs = (total_steps + len(dataloader) - 1) // len(dataloader)
    else:
        num_epochs = cfg.num_epochs
        total_steps = len(dataloader) * num_epochs

    checkpoint_interval_steps = cfg.checkpoint_interval_samples // global_batch_size

    assert not (cfg.warmup_samples is not None and cfg.warmup_fraction is not None), (
        "warmup_samples and warmup_fraction cannot be set simultaneously"
    )
    if cfg.warmup_samples:
        num_warmup_steps = cfg.warmup_samples // global_batch_size
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
    ddp_locale.train()
    model = cast(LOCALE, ddp_locale.module if is_distributed else ddp_locale)

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for epoch in range(num_epochs):
            par_tqdm_write(f"Training epoch = {epoch + 1}/{num_epochs}")
            for batch in dataloader:
                q, query_seqs, k, key_seqs = batch
                q = q.to(local_rank)
                k = k.to(local_rank)
                optimizer.zero_grad()

                q_embeds = model._embed(
                    model.bert_q,
                    model.projector_q,
                    q.to(model.device),
                    pooling=model.pooling,
                )
                q_embeds = F.normalize(q_embeds, dim=1)
                k_embeds = model._embed(
                    model.bert_q,
                    model.projector_q,
                    k.to(model.device),
                    pooling=model.pooling,
                )
                k_embeds = F.normalize(k_embeds, dim=1)
                B = q_embeds.shape[0]
                cosine_sims = torch.matmul(q_embeds, k_embeds.T)  # (B, B)
                sw_scores = get_smith_waterman_scores(query_seqs, key_seqs).to(
                    model.device
                )  # (B, B)

                # Split in-batch sims into positive (diagonal) and in-batch negatives
                # (off-diagonal). These form the base of the unified distribution.
                off_diag = ~torch.eye(B, dtype=torch.bool, device=model.device)
                pos_model_sims = cosine_sims.diagonal().unsqueeze(1)  # (B, 1)
                pos_sw = sw_scores.diagonal().unsqueeze(1)  # (B, 1)
                in_batch_sims = cosine_sims[off_diag].view(B, B - 1)  # (B, B-1)
                in_batch_sw = sw_scores[off_diag].view(B, B - 1)  # (B, B-1)

                if cfg.hnm_num_negatives > 0 and model.is_moco:
                    k_hn = cfg.hnm_num_negatives

                    with torch.no_grad():
                        # Find top-k hard negatives by Smith-Waterman score using
                        # the k-mer index for candidate pre-filtering (zero false negatives).
                        top_k_idx, _, hn_sw = model._get_top_k_sw_negatives(
                            query_seqs, k_hn
                        )
                        hn_sw = hn_sw.to(model.device)  # (B, k_hn)

                    # valid_hn_mask tracks slots where a candidate was found (-1 = none)
                    valid_hn_mask = top_k_idx >= 0  # (B, k_hn)
                    safe_idx = top_k_idx.clamp(min=0).flatten()  # (B * k_hn,)
                    hn_vecs = model.queue[:, safe_idx].T.view(
                        B, k_hn, -1
                    )  # (B, k_hn, D)
                    hn_model_sims = (q_embeds.unsqueeze(1) * hn_vecs).sum(
                        dim=-1
                    )  # (B, k_hn)
                    # Mask out slots with no candidate so they don't influence the loss
                    hn_model_sims = hn_model_sims.masked_fill(~valid_hn_mask, -1e9)
                    hn_sw = hn_sw.masked_fill(~valid_hn_mask, -1e9)

                    # Unified: [positive | hard negatives | in-batch negatives]
                    all_model_sims = torch.cat(
                        [pos_model_sims, hn_model_sims, in_batch_sims], dim=1
                    )  # (B, 1+k_hn+B-1)
                    all_sw = torch.cat(
                        [pos_sw, hn_sw, in_batch_sw], dim=1
                    )  # (B, 1+k_hn+B-1)
                else:
                    # Without HNM: unified distribution over positive + in-batch negatives
                    all_model_sims = torch.cat([pos_model_sims, in_batch_sims], dim=1)
                    all_sw = torch.cat([pos_sw, in_batch_sw], dim=1)

                pred_scores = F.log_softmax(all_model_sims / model.T, dim=-1)
                target_scores = F.log_softmax(
                    all_sw / cfg.smith_waterman_temperature, dim=-1
                )
                loss = F.kl_div(
                    pred_scores, target_scores, reduction="batchmean", log_target=True
                )
                loss.backward()

                # Enqueue current key embeddings so the hard negative pool
                # evolves as the model improves (uses bert_q, not momentum encoder)
                with torch.no_grad():
                    if model.is_moco:
                        model._dequeue_and_enqueue(
                            k_embeds.detach(), is_distributed, key_seqs
                        )
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    sw_probs = F.softmax(
                        sw_scores / cfg.smith_waterman_temperature, dim=-1
                    )
                    pred_probs = F.softmax(cosine_sims / model.T, dim=-1)
                    sw_entropy = -(sw_probs * sw_probs.log()).sum(dim=-1).mean()
                    pred_entropy = -(pred_probs * pred_probs.log()).sum(dim=-1).mean()

                if global_rank == 0:
                    frac_hn_high_score = (
                        (hn_sw >= pos_sw) & valid_hn_mask
                    ).sum() / valid_hn_mask.sum().clamp(min=1)
                    assert run
                    lrs = scheduler.get_last_lr()
                    metrics = {
                        "train/loss": loss.item(),
                        "train/lr": lrs[0],
                        "train/samples_seen": global_step * global_batch_size,
                        "train/model_entropy": pred_entropy.item(),
                        "train/sw_entropy": sw_entropy.item(),
                        "train/hn_high_sw_score_frac": frac_hn_high_score.item(),
                    }
                    for i, group in enumerate(optimizer.param_groups):
                        norm = torch.nn.utils.get_total_norm(
                            [p.grad for p in group["params"]]
                        )
                        metrics[f"metrics/grad_norm_{group['name']}"] = norm
                    run.log(metrics)

                if global_step % checkpoint_interval_steps == 0 and global_step > 0:
                    if global_rank == 0:
                        assert run
                        if is_distributed:
                            module = ddp_locale.module
                        else:
                            module = ddp_locale

                        assert isinstance(module, torch.nn.Module)
                        model_state_dict = module.state_dict()

                        val_acc1, val_acc5 = get_val_accuracy(
                            val_dataloader,
                            cfg.num_val_queries,
                            cfg.num_val_keys,
                            module,
                            local_rank,
                            tokenizer,
                            cfg.moco_filter_queue_identity_cutoff,
                        )

                        run.log(
                            {
                                "val/acc1": val_acc1[0],
                                "val/acc5": val_acc5[0],
                                "val/samples_seen": global_step * global_batch_size,
                            }
                        )
                        save_checkpoint(
                            {
                                "step": global_step,
                                "model": model_state_dict,
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
                            kl=True,
                        )
                    if is_distributed:
                        torch.distributed.barrier()
                    ddp_locale.train()

                global_step += 1
                pbar.update(1)
                if cfg.total_samples is not None and global_step >= total_steps:
                    break

            # Check for step-based termination (Outer Loop)
            if cfg.total_samples is not None and global_step >= total_steps:
                break

    if global_rank == 0:
        assert run
        if is_distributed:
            module = ddp_locale.module
        else:
            module = ddp_locale

        assert isinstance(module, torch.nn.Module)
        model_state_dict = module.state_dict()

        val_acc1, val_acc5 = get_val_accuracy(
            val_dataloader,
            cfg.num_val_queries,
            cfg.num_val_keys,
            module,
            local_rank,
            tokenizer,
            cfg.moco_filter_queue_identity_cutoff,
        )

        run.log(
            {
                "val/acc1": val_acc1[0],
                "val/acc5": val_acc5[0],
                "val/samples_seen": global_step * global_batch_size,
            }
        )
        save_checkpoint(
            {
                "step": global_step,
                "model": model_state_dict,
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
            kl=True,
        )


def _sw_score(seq_a: str, seq_b: str) -> float:
    """Smith-Waterman identity score (HW mode) for a single pair."""
    q, t = (seq_a, seq_b) if len(seq_a) <= len(seq_b) else (seq_b, seq_a)
    result = edlib.align(query=q, target=t, mode="HW", task="distance")
    return 1.0 - result["editDistance"] / len(q)


@torch.no_grad()
def get_hard_negative_sw_scores(
    query_seqs: list[str],
    hn_seqs_per_query: list[list[str | None]],
) -> torch.Tensor:
    """
    Compute SW identity scores between each query and its per-query hard negatives.

    Args:
        query_seqs: B query sequences
        hn_seqs_per_query: B lists of k hard negative sequences (None = missing slot)

    Returns:
        (B, k) tensor of identity scores; missing slots get 0.0
    """
    B = len(query_seqs)
    k = len(hn_seqs_per_query[0]) if B > 0 else 0
    scores = torch.zeros(B, k, dtype=torch.float)
    for i, query_seq in enumerate(query_seqs):
        for j, hn_seq in enumerate(hn_seqs_per_query[i]):
            if hn_seq is not None:
                scores[i, j] = _sw_score(query_seq, hn_seq)
    return scores


@torch.no_grad()
def get_smith_waterman_scores(
    query_seqs: list[str], key_seqs: list[str]
) -> torch.Tensor:
    assert len(query_seqs) == len(key_seqs)
    B = len(query_seqs)
    output_identities = torch.zeros((B, B), dtype=torch.float)

    for i, query_seq in enumerate(query_seqs):
        for j, key_seq in enumerate(key_seqs):
            # Use edlib for fast alignment
            if len(query_seq) <= len(key_seq):
                q = query_seq
                t = key_seq
            else:
                q = key_seq
                t = query_seq
            result = edlib.align(query=q, target=t, mode="HW", task="distance")
            edit_distance = result["editDistance"]
            output_identities[i, j] = 1.0 - (edit_distance / len(q))
    return output_identities


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
    similarity_cutoff,
    augment_config,
):
    par_tqdm_write("Evaluating val accuracy")
    model.eval()
    augmenter = Augmenter(augment_config)
    with torch.no_grad():
        all_q = []
        all_q_mut_95 = []
        all_q_mut_90 = []
        all_k = []
        all_query_seqs = []
        all_key_seqs = []

        for batch in val_dataloader:
            q_tokens, query_seqs, k_tokens, key_seqs = batch
            len_q = sum([embedded_q.shape[0] for embedded_q in all_q])
            if len_q < num_queries:
                q_tokens_mut_95 = tokenizer(
                    [augmenter.augment(query, identity=0.95) for query in query_seqs],
                    return_tensors="pt",
                    padding=True,
                ).to(local_rank)
                q_tokens_mut_90 = tokenizer(
                    [augmenter.augment(query, identity=0.90) for query in query_seqs],
                    return_tensors="pt",
                    padding=True,
                ).to(local_rank)
                q_mut_95 = model.encode(q_tokens_mut_95)
                all_q_mut_95.append(q_mut_95)
                q_mut_90 = model.encode(q_tokens_mut_90)
                all_q_mut_90.append(q_mut_90)
                q_tokens = q_tokens.to(local_rank)
                q = model.encode(q_tokens)
                all_q.append(q)
                all_query_seqs.extend(query_seqs)

            k_tokens = k_tokens.to(local_rank)
            k = model.encode(k_tokens)
            all_k.append(k)
            all_key_seqs.extend(key_seqs)

        all_q = torch.cat(all_q, dim=0)
        all_q_mut_95 = torch.cat(all_q_mut_95, dim=0)
        all_q_mut_90 = torch.cat(all_q_mut_90, dim=0)
        all_k = torch.cat(all_k, dim=0)
        logits = torch.matmul(all_q, all_k.T)
        logits_mut_95 = torch.matmul(all_q_mut_95, all_k.T)
        logits_mut_90 = torch.matmul(all_q_mut_90, all_k.T)

        # Filter out aligned sequences from consideration
        logits = _filter_aligned_sequences(
            logits, all_query_seqs, all_key_seqs, similarity_cutoff
        )
        logits_mut_95 = _filter_aligned_sequences(
            logits_mut_95, all_query_seqs, all_key_seqs, similarity_cutoff
        )
        logits_mut_90 = _filter_aligned_sequences(
            logits_mut_90, all_query_seqs, all_key_seqs, similarity_cutoff
        )

        labels = torch.arange(all_q.shape[0]).to(local_rank)
        labels_mut_95 = torch.arange(all_q_mut_95.shape[0]).to(local_rank)
        labels_mut_90 = torch.arange(all_q_mut_90.shape[0]).to(local_rank)

    acc1 = accuracy(logits, labels, topk=(1,))
    acc5 = accuracy(logits, labels, topk=(5,))
    acc1_mut_95 = accuracy(logits_mut_95, labels_mut_95, topk=(1,))
    acc5_mut_95 = accuracy(logits_mut_95, labels_mut_95, topk=(5,))
    acc1_mut_90 = accuracy(logits_mut_90, labels_mut_90, topk=(1,))
    acc5_mut_90 = accuracy(logits_mut_90, labels_mut_90, topk=(5,))

    torch.cuda.empty_cache()
    return acc1, acc5, acc1_mut_95, acc5_mut_95, acc1_mut_90, acc5_mut_90


def _filter_aligned_sequences(logits, query_seqs, key_seqs, similarity_cutoff):
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

            if len(query_seq) <= len(key_seq):
                q = query_seq
                t = key_seq
            else:
                q = key_seq
                t = query_seq
            result = edlib.align(query=q, target=t, mode="HW", task="distance")
            edit_distance = result["editDistance"]
            similarity = 1.0 - (edit_distance / len(q))

            # If aligned, mask out this logit
            if similarity >= similarity_cutoff:
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
