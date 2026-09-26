"""Build the IVF-RaBitQ index of a dense config's fbin, one shard per node.

Every node encodes an equal contiguous row range into
<index>/ivf/ivf<nlist>_rabitq<b>_shard_<r>_of_<R>.faiss (rank 0 first trains
the centroids from a row sample; the others wait for them). Shards are
resumable (an existing file is kept) and merged into ivf<nlist>_rabitq<b>.faiss
by the first non-FastScan search that loads the index (DenseIndex.load, needs
RAM for the whole index -- a 512 GB CPU node for sra4571), or here with --merge.
FastScan search (ivf_fastscan) loads the shards side by side and never merges.

    srun -N8 uv run python build_ivf.py --config <dense config with use_ivf>
    python build_ivf.py --config ... --merge true      # CPU node, after the build
"""

import os
import time

from jsonargparse import CLI

from src.config import DenseConfig, ExperimentConfig
from src.ivf_rabitq import build_ivf_shard, merge_ivf_shards


def main(cfg: ExperimentConfig, merge: bool = False):
    m = cfg.model
    assert isinstance(m, DenseConfig) and m.use_ivf, "config must set model.use_ivf"
    index_path = cfg.index_dir / m.index_suffix
    ivf_dir = index_path / "ivf"
    rank = int(os.environ.get("SLURM_NODEID", "0"))
    num_ranks = int(os.environ.get("SLURM_NNODES", "1"))
    if merge:
        shards = sorted(ivf_dir.glob(f"ivf{m.ivf_nlist}_rabitq{m.ivf_nb_bits}_shard_*_of_*.faiss"))
        num_ranks = int(shards[0].stem.rsplit("_of_", 1)[1])
        print(merge_ivf_shards(ivf_dir, m.ivf_nlist, m.ivf_nb_bits, num_ranks))
        return
    t0 = time.time()
    out = build_ivf_shard(
        index_path / "embeddings.fbin",
        ivf_dir,
        m.ivf_nlist,
        rank=rank,
        num_ranks=num_ranks,
        nb_bits=m.ivf_nb_bits,
        train_rows=m.ivf_train_rows,
    )
    print(f"[rank {rank}/{num_ranks}] {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    merge = False
    if "--merge" in args:
        i = args.index("--merge")
        merge = args[i + 1].lower() == "true"
        del args[i : i + 2]
    cfg = CLI(ExperimentConfig, as_positional=False, args=args)
    main(cfg, merge)
