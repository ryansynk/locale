import pytest

from src.dense_index import batched, chunk_sequence


def _seq(n: int) -> str:
    bases = "ACGT"
    return "".join(bases[i % 4] for i in range(n))


# ---------------------------------------------------------------------------
# batched
# ---------------------------------------------------------------------------


class TestBatched:
    def test_even_split(self):
        assert list(batched(range(6), 2)) == [[0, 1], [2, 3], [4, 5]]

    def test_uneven_last_batch_is_shorter(self):
        assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]

    def test_batch_larger_than_input(self):
        assert list(batched(range(3), 10)) == [[0, 1, 2]]

    def test_empty_iterable(self):
        assert list(batched([], 5)) == []

    def test_batch_size_one(self):
        assert list(batched("abc", 1)) == [["a"], ["b"], ["c"]]

    def test_output_covers_all_elements(self):
        items = list(range(17))
        batches = list(batched(items, 5))
        assert sum(len(b) for b in batches) == 17
        assert [x for b in batches for x in b] == items


# ---------------------------------------------------------------------------
# chunk_sequence — stride
# ---------------------------------------------------------------------------


class TestChunkSequenceStride:
    def test_all_chunks_are_exact_size(self):
        chunks = chunk_sequence(_seq(100), "c1", 30, 10, "stride", None)
        assert all(len(c) == 30 for c in chunks)

    def test_chunk_count(self):
        # step=20, range(0, 71, 20) = [0, 20, 40, 60] → 4 chunks
        chunks = chunk_sequence(_seq(100), "c1", 30, 10, "stride", None)
        assert len(chunks) == 4

    def test_adjacent_chunks_share_overlap_region(self):
        seq = _seq(100)
        chunks = chunk_sequence(seq, "c1", 30, 10, "stride", None)
        # last 10 chars of chunk[0] must equal first 10 chars of chunk[1]
        assert chunks[0][-10:] == chunks[1][:10]

    def test_no_overlap_produces_non_overlapping_chunks(self):
        seq = _seq(60)
        chunks = chunk_sequence(seq, "c1", 20, 0, "stride", None)
        assert chunks == [seq[0:20], seq[20:40], seq[40:60]]

    def test_sequence_shorter_than_chunk_returns_empty(self):
        # DenseIndex._iter_chunks routes short seqs around chunk_sequence,
        # so empty output here is the correct contract.
        assert chunk_sequence("ACGT", "c1", 30, 10, "stride", None) == []

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_sequence(_seq(100), "c1", 30, 30, "stride", None)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_sequence(_seq(100), "c1", 30, 31, "stride", None)

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            chunk_sequence(_seq(100), "c1", 0, 0, "stride", None)

    def test_exact_sequence_content(self):
        seq = "ACGTACGTACGT"  # 12 chars, chunk=8, overlap=4, step=4
        chunks = chunk_sequence(seq, "c1", 8, 4, "stride", None)
        # range(0, 12-8+1, 4) = range(0, 5, 4) = [0, 4] → 2 chunks
        assert chunks == ["ACGTACGT", "ACGTACGT"]


# ---------------------------------------------------------------------------
# chunk_sequence — exact_chunk
# ---------------------------------------------------------------------------


class TestChunkSequenceExactChunk:
    def test_empty_interval_list_chunks_full_sequence(self):
        import random

        random.seed(0)
        seq = _seq(200)
        # No intervals → entire seq is uncovered → chunked by size
        chunks = chunk_sequence(seq, "c1", 50, 0, "exact_chunk", {"c1": []})
        # range(0, 200, 50) = [0, 50, 100, 150] → 4 chunks
        assert len(chunks) == 4

    def test_unknown_contig_id_treats_as_no_intervals(self):
        import random

        random.seed(0)
        seq = _seq(100)
        chunks = chunk_sequence(seq, "missing", 50, 0, "exact_chunk", {"c1": [(0, 50)]})
        # unknown contig → no intervals → full seq chunked
        assert len(chunks) == 2

    def test_interval_produces_covering_chunk(self):
        import random

        random.seed(0)
        seq = _seq(200)
        # interval (75, 125), chunk_size=50:
        #   lo = max(0, 125-50) = 75
        #   hi = min(75, 200-50) = 75
        # so chunk_start is deterministically 75 → chunk covers [75, 125) exactly
        chunks = chunk_sequence(seq, "c1", 50, 0, "exact_chunk", {"c1": [(75, 125)]})
        assert seq[75:125] in chunks

    def test_produces_at_least_one_chunk_per_interval(self):
        import random

        random.seed(42)
        seq = _seq(300)
        intervals = {"c1": [(50, 100), (200, 250)]}
        chunks = chunk_sequence(seq, "c1", 50, 0, "exact_chunk", intervals)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# chunk_sequence — invalid type
# ---------------------------------------------------------------------------


class TestChunkSequenceInvalidType:
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="chunk_type"):
            chunk_sequence(_seq(100), "c1", 30, 10, "sliding_window", None)
