"""Build the cuVS IVF-PQ shards of a dense config's fbin, one shard per GPU.

Run with one task per GPU; SLURM_PROCID is the shard and SLURM_NTASKS must
equal model.index.num_shards (the shard count is in the file names, so keep it
fixed across resubmits -- finished shards are skipped). The shards land in
<index_dir>/<encoder_label>/<index_label>/ with the engine's config.json; the
last task to finish marks the directory .done.

    srun -N4 --ntasks-per-node=4 --gpus-per-task=1 \\
        uv run python build_ivfpq.py --config <dense config with index.engine: ivfpq>
"""

import os
import time

from jsonargparse import CLI

from src.config import DONE_FILE, DenseMethod, ExperimentConfig, IVFPQIndex, ensure_config
from src.engines import make_engine


def main(cfg: ExperimentConfig):
    m = cfg.model
    assert isinstance(m, DenseMethod) and isinstance(m.index, IVFPQIndex), (
        "config must set model.index.engine: ivfpq"
    )
    shard = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ntasks != m.index.num_shards:
        raise ValueError(f"{ntasks} tasks for {m.index.num_shards} shards: launch one task per shard")
    fbin_dir = m.index_path(cfg.index_dir)
    engine_dir = m.engine_path(cfg.index_dir)
    ensure_config(engine_dir, m.engine_identity(), (engine_dir / DONE_FILE).exists())
    engine = make_engine(m.index)
    t0 = time.time()
    engine.build(fbin_dir, engine_dir, shard, ntasks)
    print(f"[shard {shard}/{ntasks}] built in {time.time() - t0:.0f}s")
    if engine.complete(engine_dir):
        (engine_dir / DONE_FILE).touch()
        print(f"all {ntasks} shards present: {engine_dir} marked {DONE_FILE}")


if __name__ == "__main__":
    main(CLI(ExperimentConfig, as_positional=False))
