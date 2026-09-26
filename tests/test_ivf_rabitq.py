"""IVF + RaBitQ (faiss) build, shard merge and reranked search, CPU only.

A synthetic clustered fbin stands in for the index. The properties the
8-node build and single-node search rely on: every fbin row lands in the
merged index exactly once under its own row number (the build adds in many
blocks -- the path that corrupted ids when add_preassigned was handed a
temporary array), and probing every list with a full rerank reproduces the
exact top-k, for the flat and HNSW coarse quantizers and FastScan alike.
"""

from pathlib import Path

import faiss
import numpy as np
import pytest

from src.config import DenseConfig
from src.fbin import _create_fbin_memmap
from src.ivf_rabitq import (
    IVFRaBitQSearcher,
    build_ivf_shard,
    list_shards,
    merge_ivf_shards,
    merged_path,
    read_rows_by_id,
    spherical_kmeans,
)

N, D, NLIST = 6000, 64, 16


def _clustered_rows(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((40, d)).astype(np.float32)
    x = centers[rng.integers(0, 40, n)] + 0.5 * rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> tuple[Path, np.ndarray]:
    path = tmp_path_factory.mktemp("idx") / "embeddings.fbin"
    x = _clustered_rows(N, D, seed=0)
    mm = _create_fbin_memmap(path, N, D)
    mm[:] = x
    mm.flush()
    del mm
    return path, x


@pytest.fixture(scope="module")
def merged(data) -> Path:
    fbin, _ = data
    ivf_dir = fbin.parent / "ivf"
    for rank in range(2):
        build_ivf_shard(fbin, ivf_dir, NLIST, rank=rank, num_ranks=2, train_rows=N, block_rows=500)
    return merge_ivf_shards(ivf_dir, NLIST, 1, 2)


def test_kmeans_centroids_are_unit_and_used(data):
    _, x = data
    cent = spherical_kmeans(x, NLIST, n_iter=5, devices=["cpu"], verbose=False)
    assert cent.shape == (NLIST, D)
    np.testing.assert_allclose(np.linalg.norm(cent, axis=1), 1.0, rtol=1e-5)
    assert len(np.unique(np.argmax(x @ cent.T, axis=1))) == NLIST


def test_merged_index_holds_every_row_once(merged):
    index = faiss.read_index(str(merged))
    assert index.ntotal == N
    inv = index.invlists
    ids = np.concatenate(
        [faiss.rev_swig_ptr(inv.get_ids(l), inv.list_size(l)).copy() for l in range(NLIST) if inv.list_size(l)]
    )
    np.testing.assert_array_equal(np.sort(ids), np.arange(N))
    assert merged == merged_path(merged.parent, NLIST, 1)


@pytest.mark.parametrize("quantizer,fastscan", [("flat", False), ("flat", True), ("hnsw", True)])
def test_sharded_search_matches_merged(data, merged, quantizer, fastscan):
    """Searching the two shards side by side = searching the merged index."""
    fbin, _ = data
    shards = list_shards(merged.parent, NLIST, 1)
    assert len(shards) == 2
    q = _clustered_rows(20, D, seed=3)
    one = IVFRaBitQSearcher(merged, fbin, quantizer=quantizer, fastscan=fastscan)
    two = IVFRaBitQSearcher(shards, fbin, quantizer=quantizer, fastscan=fastscan)
    assert two.ntotal == N
    a = one.search(q, 10, nprobe=4, rerank=40, io_threads=4)
    b = two.search(q, 10, nprobe=4, rerank=40, io_threads=4)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-5, atol=1e-6)


def test_read_rows_by_id_matches_fbin(data):
    fbin, x = data
    ids = np.array([5, 3, 5, N - 1, 0])
    np.testing.assert_array_equal(read_rows_by_id(fbin, ids, threads=3), x[ids])


@pytest.mark.parametrize("quantizer,fastscan", [("flat", False), ("flat", True), ("hnsw", False)])
def test_full_probe_full_rerank_is_exact(data, merged, quantizer, fastscan):
    fbin, x = data
    q = _clustered_rows(20, D, seed=1)
    k = 10
    s = IVFRaBitQSearcher(merged, fbin, quantizer=quantizer, fastscan=fastscan)
    scores, ids = s.search(q, k, nprobe=NLIST, rerank=N, io_threads=4)
    exact = q @ x.T
    want = np.argsort(-exact, axis=1, kind="stable")[:, :k]
    np.testing.assert_allclose(scores, np.take_along_axis(exact, want, axis=1), rtol=1e-5, atol=1e-6)
    # ids may differ only between exactly tied scores
    np.testing.assert_allclose(np.take_along_axis(exact, ids, axis=1), scores, rtol=1e-5, atol=1e-6)


def test_partial_probe_scores_are_exact_and_sorted(data, merged):
    fbin, x = data
    q = _clustered_rows(20, D, seed=2)
    s = IVFRaBitQSearcher(merged, fbin)
    scores, ids = s.search(q, 10, nprobe=2, rerank=50, io_threads=4)
    ok = ids >= 0
    np.testing.assert_allclose(scores[ok], np.einsum("ij,ij->i", x[ids[ok]], np.repeat(q, 10, 0).reshape(20, 10, D)[ok]), rtol=1e-5)
    assert (np.diff(scores, axis=1)[np.isfinite(scores[:, 1:])] <= 1e-6).all()


def test_config_ivf_ids():
    cfg = DenseConfig(name="locale", checkpoint_path=None, use_ivf=True, ivf_nprobe=256, both_strands=True)
    assert "ivf16384rabitq1_np256_rr300_qb8" in cfg.hits_id
    assert cfg.experiment_id.endswith("_top100_bothstrands")
    with pytest.raises(ValueError):
        DenseConfig(name="locale", checkpoint_path=None, use_ivf=True, use_rabitq=True)
