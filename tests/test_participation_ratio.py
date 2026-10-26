"""participation_ratio streams an fbin into a centered-covariance spectrum."""

import numpy as np
import pytest

from lae.training.validation import participation_ratio


def _write_fbin(path, x: np.ndarray):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.astype(np.float32).tofile(f)


def _reference(x: np.ndarray) -> float:
    cov = np.cov(x.astype(np.float64), rowvar=False, bias=True)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float(lam.sum() ** 2 / (lam**2).sum())


def test_matches_dense_reference_across_block_boundaries(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 16)) @ rng.standard_normal((16, 16)) + 3.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # unit rows, like the index
    _write_fbin(tmp_path / "e.fbin", x)
    out = participation_ratio(tmp_path / "e.fbin", device="cpu", block_rows=333)
    assert out["val/participation_ratio"] == pytest.approx(_reference(x), rel=1e-6)
    assert out["val/participation_ratio_frac"] == pytest.approx(
        out["val/participation_ratio"] / 16
    )
    mu = x.mean(0)
    assert out["val/embed_cov_trace"] == pytest.approx(1 - mu @ mu, rel=1e-5)


def test_isotropic_is_near_d_and_rank_one_is_one(tmp_path):
    rng = np.random.default_rng(1)
    _write_fbin(tmp_path / "iso.fbin", rng.standard_normal((20000, 8)))
    iso = participation_ratio(tmp_path / "iso.fbin", device="cpu")
    assert iso["val/participation_ratio"] == pytest.approx(8, abs=0.05)

    line = np.outer(rng.standard_normal(500), rng.standard_normal(8)) + 1.0
    _write_fbin(tmp_path / "line.fbin", line)
    one = participation_ratio(tmp_path / "line.fbin", device="cpu")
    assert one["val/participation_ratio"] == pytest.approx(1, abs=1e-6)


def test_empty_file_raises(tmp_path):
    _write_fbin(tmp_path / "empty.fbin", np.zeros((0, 4)))
    with pytest.raises(ValueError):
        participation_ratio(tmp_path / "empty.fbin", device="cpu")
