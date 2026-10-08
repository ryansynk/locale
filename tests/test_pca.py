"""PCAIndex: the projector and projected file it writes (fp32 or rounded to
fp16 / bf16), and the build -> load -> topk_hits round trip through
ExactEngine's scan, which must score the stored (projected, rounded)
vectors exactly against the fp32-projected queries."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import pytest
import torch
from src.config import PCAIndex
from src.engines import make_engine
from src.fbin import FBIN_HEADER_BYTES, _create_fbin_memmap
from src.fbin_stream import create_fbin_file, transform_rows, transform_rows_parallel
from src.pca import PCA_BUILD_DONE, PCA_PROJECTOR, PCAEngine, pca_file
from src.rabitq import shard_bounds

ROWS = [400, 1, 0, 599]  # an empty and a one-row accession
N, D, K = sum(ROWS), 24, 6
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@pytest.fixture
def fbin_dir(tmp_path):
    g = torch.Generator().manual_seed(0)
    # Low-rank plus noise, so the top K components carry most of the energy.
    X = torch.randn(N, K, generator=g) @ torch.randn(K, D, generator=g)
    X = X + 0.05 * torch.randn(N, D, generator=g)
    X = torch.nn.functional.normalize(X, dim=1).numpy()
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    starts = np.cumsum([0] + ROWS[:-1]).tolist()
    pl.DataFrame(
        {"srr_id": list("ABCD"), "start_row": starts, "num_rows": ROWS}
    ).write_parquet(tmp_path / "meta.parquet")
    return tmp_path, X


def _build(fbin_dir, dtype, k=K):
    d, _ = fbin_dir
    name = f"pca-d{k}-{dtype}"
    eng = make_engine(PCAIndex(pca_dim=k, dtype=dtype, top_k=10))
    eng.build(d, d / name, 0, 1)
    eng.load(d, d / name, ["cpu"])
    return eng, d / name


def _stored(path, dtype):
    raw = path.read_bytes()
    n, k = np.frombuffer(raw[:8], np.uint32).tolist()
    t = torch.frombuffer(bytearray(raw[8:]), dtype=DTYPES[dtype]).reshape(n, k)
    return t.float().numpy()


def test_make_engine_dispatch():
    assert isinstance(make_engine(PCAIndex()), PCAEngine)


def test_unknown_dtype_raises():
    with pytest.raises(ValueError):
        PCAEngine(PCAIndex(dtype="int8"))


def test_pca_dim_out_of_range_raises(fbin_dir):
    d, _ = fbin_dir
    eng = make_engine(PCAIndex(pca_dim=D + 1))
    with pytest.raises(ValueError, match="pca_dim"):
        eng.build(d, d / "bad", 0, 1)


def _engine(dtype="fp32", k=K, timeout=30.0):
    eng = make_engine(PCAIndex(pca_dim=k, dtype=dtype, top_k=10))
    eng.POLL, eng.WAIT_TIMEOUT = 0.02, timeout
    return eng


def _build_ranks(d, idx, num_shards, ranks=None, dtype="fp32", timeout=30.0):
    """Run build for ``ranks`` (default all) of num_shards, one thread and
    engine per rank, as separate nodes would; re-raise the first failure."""
    ranks = range(num_shards) if ranks is None else ranks
    with ThreadPoolExecutor(len(ranks)) as pool:
        futs = [
            pool.submit(_engine(dtype, timeout=timeout).build, d, idx, r, num_shards)
            for r in ranks
        ]
        for f in futs:
            f.result()


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_sharded_build_matches_single_rank(fbin_dir, dtype):
    d = fbin_dir[0]
    _build_ranks(d, d / "one", 1, dtype=dtype)
    _build_ranks(d, d / "three", 3, dtype=dtype)
    # Same seeded fit, so the same projector; rows differ at most by matmul
    # noise across block shapes (then rounding, for bf16).
    assert torch.equal(
        torch.load(d / "one" / PCA_PROJECTOR), torch.load(d / "three" / PCA_PROJECTOR)
    )
    a = _stored(d / "one" / pca_file(dtype), dtype)
    b = _stored(d / "three" / pca_file(dtype), dtype)
    assert np.abs(a - b).max() <= (1e-6 if dtype == "fp32" else 8e-3)
    built = json.loads((d / "three" / PCA_BUILD_DONE).read_text())
    assert built == {"n": N, "pca_dim": K, "dtype": dtype, "num_shards": 3}
    bounds = shard_bounds(N, 3)
    for r in range(3):
        m = json.loads((d / "three" / f"shard_rank_{r}.json").read_text())
        assert (m["start"], m["end"]) == (bounds[r], bounds[r + 1])
    # and it loads and searches like any other PCA index
    eng = make_engine(PCAIndex(pca_dim=K, dtype=dtype, top_k=10))
    eng.load(d, d / "three", ["cpu"])
    _, X = fbin_dir
    scores, ids = eng.topk_hits(X[:8], 10)
    qp = (torch.from_numpy(X[:8]) @ eng.projector).numpy()
    assert np.allclose(scores, (qp[:, None, :] * b[ids]).sum(-1), atol=1e-5)


def test_every_rank_returns_only_when_complete(fbin_dir):
    d, _ = fbin_dir
    idx = d / "late"
    # Ranks 0 and 1 of 3 alone cannot finish: both must time out waiting.
    with pytest.raises(TimeoutError):
        _build_ranks(d, idx, 3, ranks=[0, 1], timeout=1.0)
    assert not (idx / PCA_BUILD_DONE).exists()
    # Rank 2 joins later; ranks 0 and 1 resume from their markers.
    _build_ranks(d, idx, 3)
    assert (idx / PCA_BUILD_DONE).exists()


def test_resume_skips_rows_already_marked(fbin_dir):
    d, _ = fbin_dir
    idx = d / "resume"
    _build_ranks(d, idx, 2)
    path = idx / pca_file("fp32")
    start = shard_bounds(N, 2)[1]
    row_bytes = K * 4
    with open(path, "r+b") as f:  # wipe rank 1's rows
        f.seek(FBIN_HEADER_BYTES + start * row_bytes)
        f.write(b"\0" * (N - start) * row_bytes)
    before = _stored(path, "fp32")
    (idx / PCA_BUILD_DONE).unlink()
    _build_ranks(d, idx, 2)  # rank 1's marker stands: its rows are not rewritten
    assert np.array_equal(_stored(path, "fp32"), before)
    (idx / PCA_BUILD_DONE).unlink()
    (idx / "shard_rank_1.json").unlink()
    _build_ranks(d, idx, 2)  # now they are
    assert np.abs(_stored(path, "fp32")[start:]).max() > 0


def test_done_index_is_left_alone(fbin_dir):
    d, _ = fbin_dir
    idx = d / "done"
    idx.mkdir()
    (idx / PCA_BUILD_DONE).write_text("{}")
    _build_ranks(d, idx, 2)  # returns at once: no projector, no file
    assert not (idx / PCA_PROJECTOR).exists()


def test_changed_num_shards_raises(fbin_dir):
    d, _ = fbin_dir
    idx = d / "reshard"
    _build_ranks(d, idx, 2)
    (idx / PCA_BUILD_DONE).unlink()
    with pytest.raises(ValueError, match="different num_shards"):
        _engine().build(d, idx, 0, 3)  # rank 0's marker covers half, not a third


def test_shard_out_of_range_raises(fbin_dir):
    d, _ = fbin_dir
    with pytest.raises(ValueError, match="outside"):
        _engine().build(d, d / "x", 2, 2)


def test_transform_rows_parallel_matches_one_device(fbin_dir, tmp_path):
    d, X = fbin_dir
    P = torch.randn(D, 5, generator=torch.Generator().manual_seed(3))
    one, three = tmp_path / "one.fbin", tmp_path / "three.fbin"
    for path, devs in ((one, ["cpu"]), (three, ["cpu"] * 3)):
        create_fbin_file(path, N, 5, torch.float32)
        transform_rows_parallel(
            d / "embeddings.fbin", path, lambda _dev: lambda x: x @ P,
            5, torch.float32, devs, 0, N, "t", block_rows=70,
        )  # fmt: skip
    a, b = _stored(one, "fp32"), _stored(three, "fp32")
    assert np.allclose(a, X @ P.numpy(), atol=1e-6)
    assert np.allclose(a, b, atol=1e-6)


def test_transform_rows_needs_a_created_file(fbin_dir, tmp_path):
    d, _ = fbin_dir
    path = tmp_path / "short.fbin"
    path.write_bytes(b"\0" * 8)
    with pytest.raises(ValueError, match="create_fbin_file"):
        transform_rows(
            d / "embeddings.fbin", path, lambda x: x, D, torch.float32, "cpu", 0, N
        )


def test_file_names():
    assert pca_file("fp32") == "embeddings.fbin"
    assert pca_file("bf16") == "embeddings.bf16bin"
    assert pca_file("fp16") == "embeddings.fp16bin"


@pytest.mark.parametrize("dtype", DTYPES)
def test_file_is_rounded_projection(fbin_dir, dtype):
    _, X = fbin_dir
    eng, idx = _build(fbin_dir, dtype)
    P = torch.load(idx / PCA_PROJECTOR)
    assert P.shape == (D, K)
    assert torch.allclose(P.T @ P, torch.eye(K), atol=1e-5)  # orthonormal
    want = (torch.from_numpy(X) @ P).to(DTYPES[dtype]).float().numpy()
    got = _stored(idx / pca_file(dtype), dtype)
    assert got.shape == (N, K)
    # The build projects on whichever device it finds; bitwise equality only
    # holds for the same matmul, so allow fp32 matmul noise before rounding.
    tol = {"fp32": 1e-6, "fp16": 1e-3, "bf16": 8e-3}[dtype]
    assert np.abs(got - want).max() <= tol
    expect = FBIN_HEADER_BYTES + N * K * DTYPES[dtype].itemsize
    assert (idx / pca_file(dtype)).stat().st_size == expect


@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_scores_stored_vectors_exactly(fbin_dir, dtype):
    _, X = fbin_dir
    eng, idx = _build(fbin_dir, dtype)
    stored = _stored(idx / pca_file(dtype), dtype)
    q = X[:16] + 0.02 * np.random.default_rng(1).standard_normal((16, D))
    q = q.astype(np.float32)
    scores, ids = eng.topk_hits(q, 10, block_rows=128)  # several blocks
    qp = (torch.from_numpy(q) @ eng.projector).numpy()
    want = qp @ stored.T
    assert np.allclose(scores, np.take_along_axis(want, ids, 1), atol=1e-5)
    # the returned ids are a top-10 of the stored vectors (ties aside)
    kth = np.sort(want, axis=1)[:, -10]
    assert (scores[:, -1] >= kth - 1e-5).all()


def test_rounded_matches_fp32_neighbors(fbin_dir):
    _, X = fbin_dir
    q = X[:16]
    _, ids32 = _build(fbin_dir, "fp32")[0].topk_hits(q, 10)
    _, ids16 = _build(fbin_dir, "bf16")[0].topk_hits(q, 10)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids32, ids16)])
    assert overlap >= 0.9


def test_vec_range_shard(fbin_dir):
    _, X = fbin_dir
    eng, idx = _build(fbin_dir, "bf16")
    stored = _stored(idx / pca_file("bf16"), "bf16")
    q = X[:5]
    scores, ids = eng.topk_hits(q, 10, vec_range=(300, 700))
    assert ((ids >= 300) & (ids < 700)).all()
    qp = (torch.from_numpy(q) @ eng.projector).numpy()
    assert np.allclose(scores, (qp[:, None, :] * stored[ids]).sum(-1), atol=1e-5)


def test_size_counts_the_dtype_file(fbin_dir):
    eng, idx = _build(fbin_dir, "bf16")
    want = (idx / pca_file("bf16")).stat().st_size + (
        idx / PCA_PROJECTOR
    ).stat().st_size
    assert eng.size_gb(idx.parent, idx) == want / (1024**3)


def test_load_rejects_wrong_dtype_file(fbin_dir):
    d, _ = fbin_dir
    _, idx = _build(fbin_dir, "fp32")
    eng = make_engine(PCAIndex(pca_dim=K, dtype="bf16"))
    with pytest.raises(FileNotFoundError):
        eng.load(d, idx, ["cpu"])


def test_identity_includes_dtype():
    from src.config import build_identity

    ident = build_identity(PCAIndex(pca_dim=64, dtype="bf16"))
    assert json.loads(json.dumps(ident)) == {
        "engine": "pca",
        "pca_dim": 64,
        "dtype": "bf16",
    }
    assert build_identity(PCAIndex(pca_dim=64))["dtype"] == "fp32"
