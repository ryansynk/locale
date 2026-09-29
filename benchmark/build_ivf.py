"""Build the IVF-RaBitQ index of a dense config's fbin, one shard per node.

Every node encodes an equal contiguous row range into
<index_dir>/<encoder_label>/<index_label>/ivf<nlist>_rabitq<b>_shard_<r>_of_<R>.faiss
(rank 0 first trains the centroids from a row sample; the others wait for
them). Shards are resumable (an existing file is kept) and merged into
ivf<nlist>_rabitq<b>.faiss by the first non-FastScan search that loads the
index (needs RAM for the whole index -- a 512 GB CPU node for sra4571), or
here with --merge. FastScan search (index.fastscan) loads the shards side by
side and never merges. The last rank to finish marks the directory .done.

    # one process per node, r = its rank, R = the node count:
    uv run python build_ivf.py --config <dense config with index.engine: ivfrabitq> --shard r --num_shards R
    python build_ivf.py --config ... --merge true      # CPU node, after the build
"""

import time

from jsonargparse import CLI

from src.config import DONE_FILE, DenseMethod, ExperimentConfig, IVFRaBitQIndex, ensure_config
from src.engines import make_engine
from src.ivf_rabitq import merge_ivf_shards


def main(cfg: ExperimentConfig, merge: bool = False):
    m = cfg.model
    assert isinstance(m, DenseMethod) and isinstance(m.index, IVFRaBitQIndex), (
        "config must set model.index.engine: ivfrabitq"
    )
    fbin_dir = m.index_path(cfg.index_dir)
    engine_dir = m.engine_path(cfg.index_dir)
    rank = cfg.shard
    num_ranks = cfg.num_shards
    if merge:
        shards = sorted(engine_dir.glob(f"ivf{m.index.nlist}_rabitq{m.index.nb_bits}_shard_*_of_*.faiss"))
        num_ranks = int(shards[0].stem.rsplit("_of_", 1)[1])
        print(merge_ivf_shards(engine_dir, m.index.nlist, m.index.nb_bits, num_ranks))
        return
    ensure_config(engine_dir, m.engine_identity(), (engine_dir / DONE_FILE).exists())
    engine = make_engine(m.index)
    t0 = time.time()
    engine.build(fbin_dir, engine_dir, rank, num_ranks)
    print(f"[rank {rank}/{num_ranks}] built in {time.time() - t0:.0f}s")
    if engine.complete(engine_dir):
        (engine_dir / DONE_FILE).touch()
        print(f"all {num_ranks} shards present: {engine_dir} marked {DONE_FILE}")


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
