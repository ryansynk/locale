"""Build the cuVS IVF-PQ shards of a dense config's fbin, one shard per GPU.

Run with one task per GPU; SLURM_PROCID is the shard and SLURM_NTASKS must
equal model.ivfpq_num_shards (the shard count is in the file names, so keep it
fixed across resubmits -- finished shards are skipped):

    srun -N4 --ntasks-per-node=4 --gpus-per-task=1 \\
        uv run python build_ivfpq.py --config <dense config with use_ivfpq>
"""

import os
import time

from jsonargparse import CLI

from src.config import DenseConfig, ExperimentConfig
from src.ivfpq_gpu import build_ivfpq_shard, ivfpq_dir


def main(cfg: ExperimentConfig):
    m = cfg.model
    assert isinstance(m, DenseConfig) and m.use_ivfpq, "config must set model.use_ivfpq"
    shard = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ntasks != m.ivfpq_num_shards:
        raise ValueError(f"{ntasks} tasks for {m.ivfpq_num_shards} shards: launch one task per shard")
    index_path = cfg.index_dir / m.index_suffix
    t0 = time.time()
    out = build_ivfpq_shard(
        index_path / "embeddings.fbin",
        ivfpq_dir(index_path, m.ivfpq_pq_dim, m.ivfpq_pq_bits, m.ivfpq_lists_per_shard),
        shard,
        m.ivfpq_num_shards,
        pq_dim=m.ivfpq_pq_dim,
        pq_bits=m.ivfpq_pq_bits,
        lists_per_shard=m.ivfpq_lists_per_shard,
    )
    print(f"[shard {shard}/{ntasks}] {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(CLI(ExperimentConfig, as_positional=False))
