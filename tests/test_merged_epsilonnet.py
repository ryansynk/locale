"""MergedEpsilonNetIndex: build reuses a finished sibling per-accession net (or
builds a private one), its counts matrix accounts for every level-1 center,
every level-1 center is covered, and topk_hits equals a brute-force expansion
of each query's top centers into accessions."""

import json

import numpy as np
import polars as pl
import pytest
import torch
from src.config import EpsilonNetIndex, MergedEpsilonNetIndex
from src.engines import make_engine
from src.epsilonnet import NET_FBIN, NET_IDS, EpsilonNetEngine
from src.fbin import _create_fbin_memmap, _load_fbin_mmap

# Enough rows that greedy_net gets past its first (never deduplicated) batch.
ROWS = [8000, 1, 9000, 3, 7000]
D = 8
L1_EPS, MERGE_EPS = 0.1, 0.3


@pytest.fixture
def fbin_dir(tmp_path):
    """Encoder dir whose accessions draw from 30 shared directions, so
    accessions overlap and the merge has something to collapse."""
    g = torch.Generator().manual_seed(1)
    base = torch.nn.functional.normalize(torch.randn(30, D, generator=g), dim=1)
    pick = torch.randint(0, 30, (sum(ROWS),), generator=g)
    noise = 0.15 * torch.randn(sum(ROWS), D, generator=g)
    X = torch.nn.functional.normalize(base[pick] + noise, dim=1).numpy()
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    starts = np.cumsum([0] + ROWS[:-1]).tolist()
    pl.DataFrame(
        {
            "srr_id": [f"acc{i}" for i in range(len(ROWS))],
            "start_row": starts,
            "num_rows": ROWS,
        }
    ).write_parquet(tmp_path / "meta.parquet")
    return tmp_path


def _sibling_net(fbin_dir):
    d = fbin_dir / "epsnet"
    EpsilonNetEngine(EpsilonNetIndex(epsilon=L1_EPS)).build(fbin_dir, d, 0, 1)
    (d / "config.json").write_text(
        json.dumps({"engine": "epsilonnet", "epsilon": L1_EPS})
    )
    (d / ".done").touch()
    return d


def _build(fbin_dir):
    cfg = MergedEpsilonNetIndex(epsilon=MERGE_EPS, per_accession_epsilon=L1_EPS)
    engine = make_engine(cfg)
    engine.build(fbin_dir, fbin_dir / "merged", 0, 1)
    engine.load(fbin_dir, fbin_dir / "merged", ["cpu"])
    return engine


def test_reuses_finished_sibling(fbin_dir):
    _sibling_net(fbin_dir)
    _build(fbin_dir)
    assert not (fbin_dir / "merged" / "level1").exists()


def test_builds_private_level1_without_sibling(fbin_dir):
    unfinished = fbin_dir / "unfinished"  # matching identity but no .done
    unfinished.mkdir()
    (unfinished / "config.json").write_text(
        json.dumps({"engine": "epsilonnet", "epsilon": L1_EPS})
    )
    _build(fbin_dir)
    assert (fbin_dir / "merged" / "level1" / ".done").exists()


def test_counts_cover_and_topk_hits(fbin_dir):
    level1 = _sibling_net(fbin_dir)
    engine = _build(fbin_dir)
    X = np.asarray(_load_fbin_mmap(fbin_dir / "embeddings.fbin"))
    starts = np.cumsum([0] + ROWS[:-1])
    l1_rows = torch.load(level1 / NET_IDS).numpy()
    centers = np.asarray(_load_fbin_mmap(fbin_dir / "merged" / NET_FBIN))
    C = engine.counts.to_dense().numpy()

    # Every level-1 center is counted once, under its own accession.
    l1_acc = np.searchsorted(starts, l1_rows, side="right") - 1
    assert np.array_equal(C.sum(0), np.bincount(l1_acc, minlength=len(ROWS)))
    assert (C.sum(1) > 0).all()
    assert len(centers) < len(l1_rows)  # the merge collapsed something
    # Centers are the level-1 vectors net_ids.pt names, and cover all of them.
    assert np.allclose(centers, X[torch.load(fbin_dir / "merged" / NET_IDS).numpy()])
    assert ((X[l1_rows] @ centers.T).max(1) >= 1 - MERGE_EPS - 1e-5).all()

    # topk_hits: per query, an accession's score is the best of its top_k
    # nearest centers that contain it; ids are accession start rows.
    top_k = 3
    Q = X[:: len(X) // 40][:40]
    scores, ids = engine.topk_hits(Q, top_k)
    sims = Q @ centers.T
    for q in range(len(Q)):
        best = {}
        for c in np.argsort(-sims[q])[:top_k]:
            for a in np.flatnonzero(C[c]):
                best[a] = max(best.get(a, -np.inf), sims[q, c])
        got = {
            int(np.flatnonzero(starts == i)[0]): s
            for i, s in zip(ids[q], scores[q])
            if i >= 0
        }
        assert len(got) == min(top_k, len(best))
        expected = sorted(best.values(), reverse=True)[: len(got)]
        assert np.allclose(sorted(got.values(), reverse=True), expected, atol=1e-5)
        assert all(abs(best[a] - s) < 1e-5 for a, s in got.items())
