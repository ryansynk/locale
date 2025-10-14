import torch
import wandb
from jsonargparse import auto_cli
from rawbert.training.training import train


def main(
    dataset_path: str,
    batch_size: int,
    lr: float,
    epochs: int,
    dim: int = 64,
    single_batch: bool = False,
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
    train(dataset_path, device, batch_size, lr, epochs, dim, single_batch, run)
    run.finish()


if __name__ == "__main__":
    auto_cli(main, as_positional=False)