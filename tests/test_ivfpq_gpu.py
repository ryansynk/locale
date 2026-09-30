"""cuVS GPU IVF-PQ shard build + multi-process search (skipped without a GPU).

Checks what the sra4571 GPU search relies on: shards cover disjoint row
ranges with global fbin row ids, the per-GPU worker processes return every
shard's candidates, probing every list recovers the self-match of an indexed
vector, and the exact rerank returns exact inner products.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from src.config import IVFPQIndex, build_identity, search_fields
from src.fbin import _create_fbin_memmap

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

N, D, LISTS = 20_000, 64, 16


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, np.ndarray, list[Path]]:
    from src.ivfpq_gpu import build_ivfpq_shard, list_shard_files

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((50, D)).astype(np.float32)
    x = centers[rng.integers(0, 50, N)] + 0.3 * rng.standard_normal((N, D)).astype(
        np.float32
    )
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    root = tmp_path_factory.mktemp("ivfpq")
    fbin = root / "embeddings.fbin"
    mm = _create_fbin_memmap(fbin, N, D)
    mm[:] = x
    mm.flush()
    del mm
    for s in range(2):
        build_ivfpq_shard(
            fbin,
            root / "pq",
            s,
            2,
            pq_dim=32,
            pq_bits=8,
            lists_per_shard=LISTS,
            block_rows=3000,
        )
    return fbin, x, list_shard_files(root / "pq", 2)


def test_search_full_probe_finds_self_and_rerank_is_exact(built):
    from src.ivfpq_gpu import IVFPQGPUSearcher

    fbin, x, shards = built
    s = IVFPQGPUSearcher(shards, fbin)
    try:
        # rows from both shards' ranges
        rows = np.r_[0:20, N - 20 : N]
        q = x[rows]
        _, ids = s.search(q, 5, n_probes=LISTS)
        assert ((ids >= 0) & (ids < N)).all()
        assert (ids[:, 0] == rows).mean() >= 0.95
        scores, ids = s.search(q, 5, n_probes=LISTS, rerank=5, io_threads=4)
        np.testing.assert_allclose(
            scores, np.einsum("qd,qkd->qk", q, x[ids]), rtol=1e-5, atol=1e-6
        )
        assert (np.diff(scores, axis=1) <= 1e-6).all()
    finally:
        s.close()


def test_config_ivfpq_identities():
    idx = IVFPQIndex(nprobe=272, lists_per_shard=16384, rerank=10)
    assert build_identity(idx) == {
        "engine": "ivfpq",
        "pq_dim": 128,
        "pq_bits": 8,
        "lists_per_shard": 16384,
        "num_shards": 16,
    }
    assert search_fields(idx) == {
        "top_k": 100,
        "nprobe": 272,
        "rerank": 10,
        "lut": "float16",
    }


def test_engine_builds_and_searches_shards(built):
    from src.ivfpq_gpu import IVFPQEngine

    fbin, x, shards = built
    engine = IVFPQEngine(
        IVFPQIndex(
            pq_dim=32,
            lists_per_shard=LISTS,
            num_shards=2,
            nprobe=LISTS,
            rerank=5,
            top_k=5,
        )
    )
    assert engine.complete(shards[0].parent)
    engine.load(fbin.parent, shards[0].parent, devices=["cuda:0"])
    try:
        rows = np.r_[0:20, N - 20 : N]
        scores, ids = engine.topk_hits(x[rows], 5)
        assert scores.shape == (40, 5) and (ids[:, 0] == rows).mean() >= 0.95
        assert engine.size_gb(fbin.parent, shards[0].parent) > 0
        with pytest.raises(ValueError):
            engine.build(fbin.parent, shards[0].parent, 0, 3)  # num_shards mismatch
    finally:
        engine.searcher.close()
