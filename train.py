import os
from dataclasses import asdict

import torch.distributed as dist
from jsonargparse import CLI

import wandb
from rawbert.config import TrainConfig
from rawbert.training.training import train


# 1. Basic setup function
def setup():
    """
    Initialize the distributed process group.
    torchrun sets environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK.
    """
    dist.init_process_group("nccl")


def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()


def par_print(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        # Fallback for single GPU runs so it still works
        print(*args, **kwargs)


def main(cfg: TrainConfig):
    # Check if we are running via torchrun (distributed) or standard python (single GPU)
    is_distributed = "RANK" in os.environ

    if is_distributed:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback for single GPU execution
        local_rank = 0
        global_rank = 0
        world_size = 1
        par_print("Running in Single-GPU mode (No DDP detected).")

    # Calculate per-device batch size
    per_device_batch_size = cfg.batch_size // world_size

    par_print(f"Global Batch Size: {cfg.batch_size}")
    par_print(f"World Size: {world_size}")
    par_print(f"Per-Device Batch Size: {per_device_batch_size}")
    if cfg.batch_size % world_size != 0:
        par_print(
            "Warning: Global batch size is not divisible by world size. This results in an uneven split."
        )

    par_print(f"Rawbert dim = {cfg.dim}")
    par_print(f"Rawbert queue size = {cfg.moco_queue_size}")

    if global_rank == 0:
        run = wandb.init(
            entity="tomg-group-umd",
            project="rawbert",
            config=asdict(cfg),
        )
    else:
        run = None

    train(
        cfg.dataset_path,
        cfg.test_dataset_path,
        cfg.augment_config,
        cfg.num_val_queries,
        cfg.num_val_keys,
        cfg.batch_size,
        per_device_batch_size,
        cfg.lr,
        cfg.total_steps,
        cfg.dim,
        cfg.moco_queue_size,
        cfg.moco_momentum,
        cfg.moco_softmax_temp,
        cfg.checkpoint_dir,
        run,
        local_rank,
        global_rank,
        world_size,
        is_distributed,
        cfg.checkpoint_interval,
        cfg.sanity_test,
    )

    if global_rank == 0:
        run.finish()


if __name__ == "__main__":
    cfg = CLI(TrainConfig, as_positional=False)
    main(cfg)
