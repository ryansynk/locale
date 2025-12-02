import os

import torch.distributed as dist
from jsonargparse import auto_cli

import wandb
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


def main(
    dataset_path: str,
    test_dataset_path: str,
    batch_size: int,
    lr: float,
    epochs: int,
    dim: int = 64,
    moco_queue_size: int = 4096,
    moco_momentum: float = 0.999,
    moco_softmax_temp: float = 0.07,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 1000,
):
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
    per_device_batch_size = batch_size // world_size

    par_print(f"Global Batch Size: {batch_size}")
    par_print(f"World Size: {world_size}")
    par_print(f"Per-Device Batch Size: {per_device_batch_size}")
    if batch_size % world_size != 0:
        par_print(
            "Warning: Global batch size is not divisible by world size. This results in an uneven split."
        )

    par_print(f"Rawbert dim = {dim}")
    par_print(f"Rawbert queue size = {moco_queue_size}")

    if global_rank == 0:
        run = wandb.init(
            entity="tomg-group-umd",
            project="rawbert",
            config={
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )
    else:
        run = None

    train(
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
    )

    if global_rank == 0:
        run.finish()


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
