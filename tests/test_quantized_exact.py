"""QuantizedExactIndex: the 2-byte copy of the encoder fbin it writes, and
the build -> load -> topk_hits round trip through ExactEngine's scan, which
must score the rounded vectors exactly (upcast to fp32 before the matmul).
Also the shared fbin_stream.transform_fbin across several blocks, and that
the fp32 exact scan is unchanged by the dtype-aware block loader."""

import numpy as np
import pytest
import torch
from src.config import ExactIndex, QuantizedExactIndex
from src.engines import make_engine
from src.exact import ExactEngine
from src.fbin import FBIN_HEADER_BYTES, _create_fbin_memmap
from src.fbin_stream import transform_fbin
from src.quantized_exact import QuantizedExactEngine, quantized_file

N, D = 1000, 24
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@pytest.fixture
def fbin_dir(tmp_path):
    g = torch.Generator().manual_seed(0)
    X = torch.nn.functional.normalize(torch.randn(N, D, generator=g), dim=1).numpy()
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    return tmp_path, X


def _build(fbin_dir, dtype):
    d, _ = fbin_dir
    eng = make_engine(QuantizedExactIndex(dtype=dtype, top_k=10))
    eng.build(d, d / dtype, 0, 1)
    eng.load(d, d / dtype, ["cpu"])
    return eng


def _brute_topk(Xs, q, k):
    scores = q @ Xs.T
    ids = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(scores, ids, axis=1), ids


def test_make_engine_dispatch():
    assert isinstance(make_engine(QuantizedExactIndex()), QuantizedExactEngine)


def test_unknown_dtype_raises():
    with pytest.raises(ValueError):
        QuantizedExactEngine(QuantizedExactIndex(dtype="int8"))


@pytest.mark.parametrize("dtype", DTYPES)
def test_file_is_header_plus_rounded_values(fbin_dir, dtype):
    d, X = fbin_dir
    _build(fbin_dir, dtype)
    raw = (d / dtype / quantized_file(dtype)).read_bytes()
    assert len(raw) == FBIN_HEADER_BYTES + N * D * 2
    assert np.frombuffer(raw[:8], np.uint32).tolist() == [N, D]
    stored = torch.frombuffer(bytearray(raw[8:]), dtype=torch.int16).view(DTYPES[dtype])
    want = torch.from_numpy(X).to(DTYPES[dtype]).reshape(-1)
    assert torch.equal(stored, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_scores_rounded_vectors_exactly(fbin_dir, dtype):
    _, X = fbin_dir
    eng = _build(fbin_dir, dtype)
    Xq = torch.from_numpy(X).to(DTYPES[dtype]).float().numpy()
    q = X[:20] + 0.05 * np.random.default_rng(1).standard_normal((20, D))
    q = q.astype(np.float32)
    # Several blocks, so the running top-k merge is exercised too.
    scores, ids = eng.topk_hits(q, 10, block_rows=128)
    want_s, want_i = _brute_topk(Xq, q, 10)
    assert scores.dtype == np.float32
    assert np.allclose(scores, want_s, atol=1e-5)
    assert np.array_equal(ids, want_i)
    # And it is a rounding of the fp32 answer, not something else.
    fp32_s, _ = _brute_topk(X, q, 10)
    tol = 2e-3 if dtype == "fp16" else 1e-2
    assert np.abs(scores - fp32_s).max() < tol


def test_vec_range_shard(fbin_dir):
    _, X = fbin_dir
    eng = _build(fbin_dir, "bf16")
    Xq = torch.from_numpy(X).to(torch.bfloat16).float().numpy()
    q = X[:5]
    scores, ids = eng.topk_hits(q, 10, vec_range=(300, 700))
    assert ((ids >= 300) & (ids < 700)).all()
    want_s, want_i = _brute_topk(Xq[300:700], q, 10)
    assert np.array_equal(ids, want_i + 300)
    assert np.allclose(scores, want_s, atol=1e-5)


def test_size_is_half_the_fbin(fbin_dir):
    d, _ = fbin_dir
    eng = _build(fbin_dir, "fp16")
    full_bytes = (d / "embeddings.fbin").stat().st_size
    half_bytes = FBIN_HEADER_BYTES + (full_bytes - FBIN_HEADER_BYTES) // 2
    assert eng.size_gb(d, d / "fp16") == half_bytes / (1024**3)


def test_fp16_overflow_fails_the_build(tmp_path):
    X = np.ones((10, 4), dtype=np.float32)
    X[3, 2] = 1e6  # past fp16's 65504
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    eng = make_engine(QuantizedExactIndex(dtype="fp16"))
    with pytest.raises(ValueError, match="fp16 cannot hold"):
        eng.build(tmp_path, tmp_path / "fp16", 0, 1)


def test_load_rejects_mismatched_copy(fbin_dir):
    d, _ = fbin_dir
    _build(fbin_dir, "bf16")
    # A different encoder fbin under the same index dir.
    fb = _create_fbin_memmap(d / "embeddings.fbin", N + 1, D)
    del fb
    eng = make_engine(QuantizedExactIndex(dtype="bf16"))
    with pytest.raises(ValueError, match="encoder fbin is"):
        eng.load(d, d / "bf16", ["cpu"])


def test_transform_fbin_multi_block(fbin_dir, tmp_path):
    d, X = fbin_dir
    P = torch.randn(D, 5, generator=torch.Generator().manual_seed(2))
    out = tmp_path / "proj.fbin"
    # 1000 rows in blocks of 300: three full blocks and a partial one, so both
    # buffers and the prefetch are used.
    transform_fbin(
        d / "embeddings.fbin", out, lambda x: x @ P, 5, torch.float32, "cpu", "t", 300
    )
    raw = out.read_bytes()
    assert np.frombuffer(raw[:8], np.uint32).tolist() == [N, 5]
    got = np.frombuffer(raw[8:], np.float32).reshape(N, 5)
    assert np.allclose(got, X @ P.numpy(), atol=1e-6)


def test_fp32_exact_scan_unchanged(fbin_dir):
    d, X = fbin_dir
    eng = ExactEngine(ExactIndex(top_k=10))
    eng.load(d, d, ["cpu"])
    assert eng.all_embeddings.dtype == torch.float32
    q = X[:8]
    scores, ids = eng.topk_hits(q, 10, block_rows=128)
    want_s, want_i = _brute_topk(X, q, 10)
    assert np.array_equal(ids, want_i)
    assert np.allclose(scores, want_s, atol=1e-6)
