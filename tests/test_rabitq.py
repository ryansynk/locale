"""Sharded 1-bit RaBitQ build + packed-resident search, CPU only.

A synthetic fbin of random unit vectors stands in for the index. The
estimator is checked against a numpy re-implementation from the on-disk
codes, the multi-rank build against the single-rank build byte for byte, and
range-restricted loads against the full load, since those are the two
properties the 8-node build and search rely on.
"""

import json
import shutil
import threading
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from src.dense_index import DenseIndex
from src.fbin import _create_fbin_memmap, _load_fbin_mmap
from src.rabitq import (
    RaBitQIndex,
    build_rabitq_index,
    estimate_centroid,
    quantize_rows,
    shard_bounds,
    unpack_codes,
)

N, D = 4000, 128


def _unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def fbin(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "embeddings.fbin"
    mm = _create_fbin_memmap(path, N, D)
    mm[:] = _unit_rows(N, D, seed=0)
    mm.flush()
    del mm
    return path


@pytest.fixture(scope="module")
def built(fbin) -> Path:
    rabitq_dir = fbin.parent / "rabitq"
    build_rabitq_index(
        fbin, rabitq_dir, chunk_rows=1000, seed=0, centroid_sample_rows=N
    )
    return rabitq_dir


def _reference_estimates(index: RaBitQIndex, q: np.ndarray) -> np.ndarray:
    """RaBitQ estimator in numpy from the on-disk codes; q is (n_q, d)."""
    codes, norms, dots = index.read_rows(0, index.n)
    pm1 = (
        np.unpackbits(codes, axis=1, bitorder="little")[:, : index.d].astype(np.float32)
        * 2
        - 1
    )
    c, r = index._centroid_np, index._rotation_np
    q_rot = (q - c) @ r
    return (q_rot @ pm1.T) * (norms / (index.sqrt_d * dots)) + (q @ c)[:, None]


class TestCentroid:
    def test_exact_when_sample_covers_all(self, fbin):
        mm = _load_fbin_mmap(fbin)
        est = estimate_centroid(mm, sample_rows=N, seed=0)
        np.testing.assert_allclose(est, np.asarray(mm).mean(axis=0), atol=1e-6)

    def test_sampled_mean_is_close_and_seeded(self, tmp_path):
        n, d = 20_000, 16
        path = tmp_path / "e.fbin"
        mm = _create_fbin_memmap(path, n, d)
        mm[:] = _unit_rows(n, d, seed=1) + np.float32(0.3)  # non-zero mean
        mm.flush()
        del mm
        mm = _load_fbin_mmap(path)
        exact = np.asarray(mm).mean(axis=0)
        a = estimate_centroid(mm, sample_rows=5000, seed=7, block_rows=100)
        b = estimate_centroid(mm, sample_rows=5000, seed=7, block_rows=100)
        assert np.array_equal(a, b)
        assert np.abs(a - exact).max() < 0.05
        assert not np.allclose(a, exact)  # it really is a sample


class TestCodes:
    def test_unpack_matches_numpy(self):
        rng = np.random.default_rng(3)
        packed = rng.integers(0, 256, size=(7, D // 8), dtype=np.uint8)
        got = unpack_codes(torch.from_numpy(packed), D).numpy()
        want = (
            np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32) * 2 - 1
        )
        assert np.array_equal(got, want)

    def test_single_rank_layout(self, built):
        meta = json.loads((built / "meta.json").read_text())
        assert meta["n"] == N and meta["d"] == D and meta["bytes_per_vec"] == D // 8
        assert meta["shards"] == [{"rank": 0, "start": 0, "end": N}]
        for name in (
            "codes_rank_0.u8",
            "norms_rank_0.f32",
            "dots_rank_0.f32",
            "shard_rank_0.json",
            "centroid.npy",
            "rotation.npy",
        ):
            assert (built / name).exists(), name
        assert (built / "codes_rank_0.u8").stat().st_size == N * D // 8


class TestSearch:
    def test_scores_match_reference_estimator(self, built):
        index = RaBitQIndex.load(built, devices=["cpu"])
        q = _unit_rows(5, D, seed=11)
        scores, ids = index.search(torch.from_numpy(q), k=N)  # every row
        ref = _reference_estimates(index, q)
        for i in range(5):
            np.testing.assert_allclose(
                scores[i].numpy(), ref[i][ids[i].numpy()], atol=2e-2
            )
            assert len(set(ids[i].tolist())) == N

    def test_estimate_tracks_true_inner_product(self, built, fbin):
        index = RaBitQIndex.load(built, devices=["cpu"])
        x = np.asarray(_load_fbin_mmap(fbin))
        q = _unit_rows(1, D, seed=5)
        est = _reference_estimates(index, q)[0].astype(np.float64)
        true = (x @ q[0]).astype(np.float64)
        # For i.i.d. random unit vectors the true inner products have std
        # ~1/sqrt(d), the same order as the 1-bit estimator's error, so the
        # correlation is bounded well below 1 here (real embeddings, with
        # large structured similarities, do much better). What must hold: the
        # estimator is unbiased and its error is O(1/sqrt(d)).
        err = est - true
        assert abs(err.mean()) < 3 * err.std() / np.sqrt(len(err))
        assert err.std() < 1.5 / np.sqrt(D)
        assert np.corrcoef(est, true)[0, 1] > 0.7

    def test_query_finds_its_own_vector(self, built, fbin):
        index = RaBitQIndex.load(built, devices=["cpu"])
        x = np.asarray(_load_fbin_mmap(fbin))
        for row in (0, 1234, N - 1):
            _, ids = index.search(torch.from_numpy(x[row : row + 1].copy()), k=1)
            assert ids[0, 0].item() == row

    def test_row_ranges_merge_to_full(self, built):
        full = RaBitQIndex.load(built, devices=["cpu"])
        q = torch.from_numpy(_unit_rows(3, D, seed=2))
        k = 20
        fs, fi = full.search(q, k)
        parts = RaBitQIndex.open(built)
        cand_s, cand_i = [], []
        for lo, hi in [(0, 1500), (1500, 1501), (1501, N)]:
            parts.load_rows(lo, hi, devices=["cpu"])
            s, i = parts.search(q, k)
            assert i.min() >= lo and i.max() < hi
            cand_s.append(s)
            cand_i.append(i)
        cs, ci = torch.cat(cand_s, 1), torch.cat(cand_i, 1)
        ms, sel = cs.topk(k, dim=1)
        mi = ci.gather(1, sel)
        for qi in range(3):
            assert set(mi[qi].tolist()) == set(fi[qi].tolist())
            np.testing.assert_allclose(ms[qi].numpy(), fs[qi].numpy(), atol=2e-2)

    def test_empty_range(self, built):
        index = RaBitQIndex.open(built)
        index.load_rows(10, 10, devices=["cpu"])
        s, i = index.search(torch.from_numpy(_unit_rows(2, D, seed=0)), k=5)
        assert s.shape == (2, 0) and i.shape == (2, 0)


class TestShardedBuild:
    def test_three_ranks_equal_single_rank(self, fbin, built, tmp_path):
        out = tmp_path / "rabitq3"
        errors = []

        def run(rank):
            try:
                build_rabitq_index(
                    fbin,
                    out,
                    rank=rank,
                    num_ranks=3,
                    chunk_rows=700,
                    seed=0,
                    centroid_sample_rows=N,
                    devices=["cpu"],
                    wait_timeout=120,
                    poll=0.05,
                )
            except Exception as e:  # noqa: BLE001 - surfaced below
                errors.append((rank, e))

        threads = [threading.Thread(target=run, args=(r,)) for r in (2, 0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        meta = json.loads((out / "meta.json").read_text())
        b = shard_bounds(N, 3)
        assert [(s["start"], s["end"]) for s in meta["shards"]] == list(pairwise(b))

        one = RaBitQIndex.open(built)
        three = RaBitQIndex.open(out)
        for a, c in zip(one.read_rows(0, N), three.read_rows(0, N)):
            assert np.array_equal(a, c)
        assert np.array_equal(one._centroid_np, three._centroid_np)
        assert np.array_equal(one._rotation_np, three._rotation_np)

    def test_completed_shard_is_skipped(self, fbin, built):
        codes = built / "codes_rank_0.u8"
        before = codes.stat().st_mtime_ns
        mm = _load_fbin_mmap(fbin)
        index = RaBitQIndex.open(built)
        quantize_rows(
            mm, built, 0, 0, N, index._centroid_np, index._rotation_np, devices=["cpu"]
        )
        assert codes.stat().st_mtime_ns == before
        with pytest.raises(ValueError):  # different bounds under the same rank
            quantize_rows(
                mm,
                built,
                0,
                0,
                N - 1,
                index._centroid_np,
                index._rotation_np,
                devices=["cpu"],
            )
        build_rabitq_index(fbin, built)  # meta exists: no-op
        assert codes.stat().st_mtime_ns == before

    def test_legacy_single_file_layout_loads(self, built, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        for src, dst in (
            ("codes_rank_0.u8", "codes.u8"),
            ("norms_rank_0.f32", "norms.f32"),
            ("dots_rank_0.f32", "dots.f32"),
            ("centroid.npy", "centroid.npy"),
            ("rotation.npy", "rotation.npy"),
        ):
            shutil.copy(built / src, legacy / dst)
        (legacy / "meta.json").write_text(
            json.dumps({"n": N, "d": D, "bytes_per_vec": D // 8})
        )
        a = RaBitQIndex.load(legacy, devices=["cpu"])
        b = RaBitQIndex.load(built, devices=["cpu"])
        q = torch.from_numpy(_unit_rows(2, D, seed=9))
        sa, ia = a.search(q, 10)
        sb, ib = b.search(q, 10)
        assert torch.equal(ia, ib) and torch.allclose(sa, sb)


class _RowEncoder:
    """Returns fixed index rows as the 'embedding' of any query."""

    def __init__(self, rows: np.ndarray):
        self.rows = torch.from_numpy(rows)

    def encode(self, sequences):
        return self.rows[: len(sequences)]


def _dense_index(built: Path, fbin: Path, query_rows: list[int]) -> DenseIndex:
    x = np.asarray(_load_fbin_mmap(fbin))
    index = DenseIndex.__new__(DenseIndex)
    index.no_search = False
    index.k = 10
    index.use_ann = False
    index.use_rabitq = True
    index.exhaustive = False
    index.top_k = 10
    index.both_strands = False
    index.rabitq_sample_rows = N
    index.rabitq_index = RaBitQIndex.open(built)
    index.all_embeddings = None
    index.acc_names_flat = ["acc0", "acc1", "acc2", "acc3"]
    index.acc_offsets = [0, 1000, 2000, 3000, N]
    index.n_vectors = N
    index.model_cfg = SimpleNamespace(device="cpu", max_seq_len=50)
    index.chunk_type = "stride"
    index.chunk_overlap = 0
    index.contig_align_intervals = None
    index.model = _RowEncoder(x[query_rows].copy())
    return index


class TestDenseIndexRaBitQ:
    def _queries(self, n):
        return pl.DataFrame(
            {"query_sequence": ["ACGT"] * n, "query_id": [f"q{i}" for i in range(n)]}
        )

    def test_topk_hits_schema_and_top_hit(self, built, fbin):
        index = _dense_index(built, fbin, [1500, 3999])
        hits = index.topk_hits(self._queries(2))
        assert hits.columns == ["query_id", "hits"]
        top = [h[0] for h in hits["hits"].to_list()]
        assert (top[0]["vector_id"], top[0]["accession"]) == (1500, "acc1")
        assert (top[1]["vector_id"], top[1]["accession"]) == (3999, "acc3")
        assert all(len(h) == 10 for h in hits["hits"].to_list())

    def test_vec_range_restricts_and_merges(self, built, fbin):
        from src.topk_regroup import merge_topk_hits

        index = _dense_index(built, fbin, [1500])
        full = index.topk_hits(self._queries(1))
        parts = [
            index.topk_hits(self._queries(1), vec_range=r)
            for r in [(0, 1000), (1000, 2500), (2500, N)]
        ]
        for (lo, hi), p in zip([(0, 1000), (1000, 2500), (2500, N)], parts):
            assert all(lo <= h["vector_id"] < hi for h in p["hits"][0].to_list())
        merged = merge_topk_hits(parts, 10)
        assert {h["vector_id"] for h in merged["hits"][0].to_list()} == {
            h["vector_id"] for h in full["hits"][0].to_list()
        }

    def test_search_regroups_with_sentinel(self, built, fbin):
        index = _dense_index(built, fbin, [1500])
        index.top_k = 1
        res = index.search(self._queries(1))
        by_acc = {r["accession"]: r["score"] for r in res["results"][0]}
        assert set(by_acc) == {"acc0", "acc1", "acc2", "acc3"}
        assert by_acc["acc1"] > 0.5
        assert all(by_acc[a] == -2.0 for a in ("acc0", "acc2", "acc3"))
