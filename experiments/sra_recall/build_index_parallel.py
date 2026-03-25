import os

# 1. Blindfold Vortex BEFORE importing PyTorch or Evo2
# Grab the local task ID assigned by Slurm (0, 1, 2, or 3)
local_rank = os.environ.get("SLURM_LOCALID", "0")

# Force this specific Python process to ONLY see its assigned GPU
os.environ["CUDA_VISIBLE_DEVICES"] = local_rank

# Now it is safe to do your heavy imports
import torch
import polars as pl
from pathlib import Path
from jsonargparse import CLI

# Your local imports
from src.config import DenseConfig, Evo2Config, MetagraphConfig, ExperimentConfig
from src.dense_index import DenseIndex
from src.evo2_index import Evo2Index
from src.metagraph_index import MetagraphIndex
from rawbert.training.unsupervised_batcher import Augmenter


def apply_mutations(queries: pl.DataFrame, mutation_rate: float) -> pl.DataFrame:
    return queries.with_columns(
        pl.col("query_sequence").map_elements(
            lambda query_seq: Augmenter.augment(query_seq, identity=1 - mutation_rate)
        )
    )


def main(cfg: ExperimentConfig):
    # 2. Grab Slurm environment variables for dataset sharding
    rank = int(os.environ.get("SLURM_PROCID", 0))  # Global task ID (0 to 31)
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of tasks (32)

    all_accessions: list[Path] = sorted(list(cfg.accessions_dir.rglob("*.contigs.fa")))
    accession_paths = [
        acc for i, acc in enumerate(all_accessions) if i % world_size == rank
    ]

    if not accession_paths:
        print(f"Rank {rank} has no accessions to process. Exiting.")
        return

    if isinstance(cfg.model, DenseConfig):
        index = DenseIndex(cfg)
        cfg.model.device = "cuda"
    elif isinstance(cfg.model, Evo2Config):
        index = Evo2Index(cfg)
        cfg.model.device = "cuda"
    elif isinstance(cfg.model, MetagraphConfig):
        index = MetagraphIndex(cfg)
    else:
        raise ValueError("Unknown model config")

    index_path: Path = cfg.index_dir / cfg.model.name / f"shard_{rank}"
    assert not index_path.exists()
    index.build(accession_paths, index_path)
    index.save(index_path)


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
