"""RandomSampleIndex: per-accession subsample sizes, determinism, and the
build -> load -> topk_hits round trip it shares with the epsilon net."""

import numpy as np
import polars as pl
import pytest
import torch
from src.config import RandomSampleIndex
from src.engines import make_engine
from src.fbin import _create_fbin_memmap
from src.random_sample import RandomSampleEngine

ROWS = [300, 1, 500, 7]


@pytest.fixture
def fbin_dir(tmp_path):
    X = torch.nn.functional.normalize(torch.randn(sum(ROWS), 16), dim=1).numpy()
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    starts = np.cumsum([0] + ROWS[:-1]).tolist()
    pl.DataFrame(
        {"srr_id": list("ABCD"), "start_row": starts, "num_rows": ROWS}
    ).write_parquet(tmp_path / "meta.parquet")
    return tmp_path, X, starts


def _build(fbin_dir, name, **kw):
    d, _, _ = fbin_dir
    eng = make_engine(RandomSampleIndex(**kw))
    eng.build(d, d / name, 0, 1)
    return eng, torch.load(d / name / "net_ids.pt").numpy()


def test_make_engine_dispatch():
    assert isinstance(make_engine(RandomSampleIndex()), RandomSampleEngine)


def test_compression_ratio_below_one_raises():
    with pytest.raises(ValueError):
        RandomSampleIndex(compression_ratio=0.5)


def test_per_accession_counts_and_ranges(fbin_dir):
    _, _, starts = fbin_dir
    _, ids = _build(fbin_dir, "r3", compression_ratio=3.0)
    assert np.all(np.diff(ids) > 0)  # sorted, no duplicates
    acc = np.searchsorted(starts, ids, side="right") - 1
    counts = np.bincount(acc, minlength=len(ROWS))
    # round(n / 3), and every accession keeps at least one row
    assert counts.tolist() == [100, 1, 167, 2]


def test_ratio_one_keeps_everything(fbin_dir):
    _, ids = _build(fbin_dir, "r1", compression_ratio=1.0)
    assert ids.tolist() == list(range(sum(ROWS)))


def test_seed_determinism(fbin_dir):
    _, a = _build(fbin_dir, "s0a", compression_ratio=4.0, seed=0)
    _, b = _build(fbin_dir, "s0b", compression_ratio=4.0, seed=0)
    _, c = _build(fbin_dir, "s1", compression_ratio=4.0, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_round_trip_topk(fbin_dir):
    d, X, _ = fbin_dir
    eng, ids = _build(fbin_dir, "rt", compression_ratio=2.0)
    eng.load(d, d / "rt", ["cpu"])
    q = X[:12]
    scores, hits = eng.topk_hits(q, 10)
    assert scores.shape == hits.shape == (12, 10)
    assert np.isin(hits, ids).all()  # only kept rows come back
    assert np.allclose((X[hits] * q[:, None, :]).sum(-1), scores, atol=1e-5)
    assert np.all(np.diff(scores, axis=1) <= 1e-6)
    assert eng.size_gb(d, d / "rt") == pytest.approx(len(ids) * 16 * 4 / 1e9)
