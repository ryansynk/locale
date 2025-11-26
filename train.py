import torch
from jsonargparse import auto_cli

import wandb
from rawbert.training.training import train


def main(
    dataset_path: str,
    test_dataset_path: str,
    batch_size: int,
    lr: float,
    epochs: int,
    dim: int = 64,
    record_memory_snapshot: bool = False,
    moco_queue_size: int = 4096,
    moco_momentum: float = 0.999,
    num_test_batches: int = 100,
    checkpoint_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        entity="tomg-group-umd",
        project="rawbert",
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
        name="test-name",
    )

    if record_memory_snapshot:
        torch.cuda.memory._record_memory_history()
        try:
            train(
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
            )
        except torch.cuda.OutOfMemoryError:
            if record_memory_snapshot:
                torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        train(
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
        )

    run.finish()


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
