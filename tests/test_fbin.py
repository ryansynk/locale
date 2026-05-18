import numpy as np
import polars as pl
import pytest

from src.dense_index import _create_fbin_memmap, _load_fbin_mmap, DenseIndex


# ---------------------------------------------------------------------------
# fbin round-trip
# ---------------------------------------------------------------------------


class TestFbinRoundTrip:
    def test_small_array_survives_round_trip(self, tmp_path):
        path = tmp_path / "test.fbin"
        data = np.arange(40, dtype=np.float32).reshape(10, 4)

        mmap = _create_fbin_memmap(path, 10, 4)
        mmap[:] = data
        mmap.flush()
        del mmap

        loaded = _load_fbin_mmap(path)
        np.testing.assert_array_equal(loaded, data)

    def test_header_encodes_shape_correctly(self, tmp_path):
        path = tmp_path / "test.fbin"
        _create_fbin_memmap(path, 7, 13).flush()

        with open(path, "rb") as f:
            n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        assert (int(n), int(d)) == (7, 13)

    def test_large_array_round_trip(self, tmp_path):
        path = tmp_path / "large.fbin"
        rng = np.random.default_rng(0)
        data = rng.standard_normal((1000, 128)).astype(np.float32)

        mmap = _create_fbin_memmap(path, 1000, 128)
        mmap[:] = data
        mmap.flush()
        del mmap

        loaded = _load_fbin_mmap(path)
        np.testing.assert_allclose(loaded, data, rtol=0, atol=0)

    def test_loaded_array_is_readonly(self, tmp_path):
        path = tmp_path / "test.fbin"
        _create_fbin_memmap(path, 4, 4).flush()
        loaded = _load_fbin_mmap(path)
        assert not loaded.flags.writeable


# ---------------------------------------------------------------------------
# merge_shards
# ---------------------------------------------------------------------------


def _write_shard(shard_dir, vectors: np.ndarray, meta_rows: list[dict]):
    """Helper: write embeddings.fbin + meta.parquet into shard_dir."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    n, d = vectors.shape
    mmap = _create_fbin_memmap(shard_dir / "embeddings.fbin", n, d)
    mmap[:] = vectors
    mmap.flush()
    del mmap
    pl.DataFrame(meta_rows).write_parquet(shard_dir / "meta.parquet")


class TestMergeShards:
    def test_vector_count_and_content(self, tmp_path):
        rng = np.random.default_rng(1)
        shard0_vecs = rng.standard_normal((3, 4)).astype(np.float32)
        shard1_vecs = rng.standard_normal((2, 4)).astype(np.float32)

        _write_shard(
            tmp_path / "shard_0",
            shard0_vecs,
            [{"srr_id": "acc0", "start_row": 0, "num_rows": 2},
             {"srr_id": "acc1", "start_row": 2, "num_rows": 1}],
        )
        _write_shard(
            tmp_path / "shard_1",
            shard1_vecs,
            [{"srr_id": "acc2", "start_row": 0, "num_rows": 2}],
        )

        DenseIndex.merge_shards(tmp_path, num_nodes=2)

        merged = _load_fbin_mmap(tmp_path / "embeddings.fbin")
        assert merged.shape == (5, 4)
        np.testing.assert_array_equal(merged[:3], shard0_vecs)
        np.testing.assert_array_equal(merged[3:], shard1_vecs)

    def test_meta_offsets_are_updated(self, tmp_path):
        rng = np.random.default_rng(2)
        _write_shard(
            tmp_path / "shard_0",
            rng.standard_normal((3, 4)).astype(np.float32),
            [{"srr_id": "acc0", "start_row": 0, "num_rows": 3}],
        )
        _write_shard(
            tmp_path / "shard_1",
            rng.standard_normal((2, 4)).astype(np.float32),
            [{"srr_id": "acc1", "start_row": 0, "num_rows": 2}],
        )

        DenseIndex.merge_shards(tmp_path, num_nodes=2)

        meta = pl.read_parquet(tmp_path / "meta.parquet")
        rows = meta.sort("srr_id").to_dicts()
        assert rows[0] == {"srr_id": "acc0", "start_row": 0, "num_rows": 3}
        # acc1 was at row 0 in shard_1; after merge it should start at row 3
        assert rows[1] == {"srr_id": "acc1", "start_row": 3, "num_rows": 2}

    def test_merged_accession_count(self, tmp_path):
        rng = np.random.default_rng(3)
        for i in range(3):
            _write_shard(
                tmp_path / f"shard_{i}",
                rng.standard_normal((2, 8)).astype(np.float32),
                [{"srr_id": f"acc{i}", "start_row": 0, "num_rows": 2}],
            )

        DenseIndex.merge_shards(tmp_path, num_nodes=3)

        meta = pl.read_parquet(tmp_path / "meta.parquet")
        assert len(meta) == 3
        assert set(meta["srr_id"].to_list()) == {"acc0", "acc1", "acc2"}
