"""Index dataclass -> engine. The only place that knows every engine's name.

An engine is one module with a class exposing::

    SHARDABLE: bool         topk_hits accepts vec_range (multi-node row sharding)
    WORKER_EMBED: bool      load() takes the EncoderConfig and embed(chunks)
                            runs the query encoder on the engine's own devices
    build(fbin_dir, index_dir, shard, num_shards)
    load(fbin_dir, index_dir, devices, encoder_cfg=None)
    topk_hits(query_vecs, top_k, vec_range=None) -> (scores, ids)
    size_gb(fbin_dir, index_dir)

Engine modules are imported lazily: cuvs / faiss pull in CUDA at import time
and run_benchmark is imported on nodes that have neither.
"""

from .config import (
    ExactIndex,
    ExhaustiveIndex,
    IVFPQIndex,
    IVFRaBitQIndex,
    RaBitQIndex,
    EpsilonNetIndex,
)


def make_engine(index_cfg):
    if isinstance(index_cfg, ExactIndex):
        from .exact import ExactEngine

        return ExactEngine(index_cfg)
    if isinstance(index_cfg, ExhaustiveIndex):
        from .exhaustive import ExhaustiveEngine

        return ExhaustiveEngine(index_cfg)
    if isinstance(index_cfg, RaBitQIndex):
        from .rabitq import RaBitQEngine

        return RaBitQEngine(index_cfg)
    if isinstance(index_cfg, IVFPQIndex):
        from .ivfpq_gpu import IVFPQEngine

        return IVFPQEngine(index_cfg)
    if isinstance(index_cfg, IVFRaBitQIndex):
        from .ivf_rabitq import IVFRaBitQEngine

        return IVFRaBitQEngine(index_cfg)
    if isinstance(index_cfg, EpsilonNetIndex):
        from .epsilonnet import EpsilonNetEngine

        return EpsilonNetEngine(index_cfg)
    raise TypeError(f"no engine for index config {type(index_cfg).__name__}")
