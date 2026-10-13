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
        chunks = chunk_sequence(_seq(100), "c1", 30, 10, "stride")
        assert all(len(c) == 30 for c in chunks)

    def test_chunk_count(self):
        # step=20, range(0, 71, 20) = [0, 20, 40, 60] → 4 chunks
        chunks = chunk_sequence(_seq(100), "c1", 30, 10, "stride")
        assert len(chunks) == 4

    def test_adjacent_chunks_share_overlap_region(self):
        seq = _seq(100)
        chunks = chunk_sequence(seq, "c1", 30, 10, "stride")
        # last 10 chars of chunk[0] must equal first 10 chars of chunk[1]
        assert chunks[0][-10:] == chunks[1][:10]

    def test_no_overlap_produces_non_overlapping_chunks(self):
        seq = _seq(60)
        chunks = chunk_sequence(seq, "c1", 20, 0, "stride")
        assert chunks == [seq[0:20], seq[20:40], seq[40:60]]

    def test_sequence_shorter_than_chunk_returns_empty(self):
        # DenseIndex._iter_chunks routes short seqs around chunk_sequence,
        # so empty output here is the correct contract.
        assert chunk_sequence("ACGT", "c1", 30, 10, "stride") == []

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_sequence(_seq(100), "c1", 30, 30, "stride")

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_sequence(_seq(100), "c1", 30, 31, "stride")

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            chunk_sequence(_seq(100), "c1", 0, 0, "stride")

    def test_exact_sequence_content(self):
        seq = "ACGTACGTACGT"  # 12 chars, chunk=8, overlap=4, step=4
        chunks = chunk_sequence(seq, "c1", 8, 4, "stride")
        # range(0, 12-8+1, 4) = range(0, 5, 4) = [0, 4] → 2 chunks
        assert chunks == ["ACGTACGT", "ACGTACGT"]


# ---------------------------------------------------------------------------
# chunk_sequence — invalid type
# ---------------------------------------------------------------------------


class TestChunkSequenceInvalidType:
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="chunk_type"):
            chunk_sequence(_seq(100), "c1", 30, 10, "sliding_window")


# ---------------------------------------------------------------------------
# chunk_sequence — end-aligned window (end_window_min_tail_bp)
# ---------------------------------------------------------------------------

# The index's real geometry: 256-bp windows, 150-bp overlap, 106-bp step.
W, OVERLAP, STEP = 256, 150, 106


def _covered(seq_len: int, chunks: list[str], seq: str) -> set[int]:
    """Positions of ``seq`` covered by ``chunks`` (each located by content)."""
    covered: set[int] = set()
    for i in range(0, seq_len - W + 1, STEP):  # strided windows, in order
        covered.update(range(i, i + W))
    if len(chunks) > (seq_len - W) // STEP + 1:  # plus the end window
        assert chunks[-1] == seq[-W:]
        covered.update(range(seq_len - W, seq_len))
    return covered


class TestEndWindow:
    def test_none_keeps_original_chunking(self):
        # 361 bp: one strided window (0-256), a 105-bp tail left uncovered.
        seq = _seq(361)
        assert chunk_sequence(seq, "c1", W, OVERLAP, "stride") == [seq[:W]]
        assert chunk_sequence(seq, "c1", W, OVERLAP, "stride", None) == [seq[:W]]

    def test_long_tail_gets_end_aligned_window(self):
        seq = _seq(361)  # tail 105 > 50
        chunks = chunk_sequence(seq, "c1", W, OVERLAP, "stride", 50)
        assert chunks == [seq[:W], seq[-W:]]

    def test_short_tail_gets_none(self):
        seq = _seq(280)  # tail 24 <= 50
        assert chunk_sequence(seq, "c1", W, OVERLAP, "stride", 50) == [seq[:W]]

    def test_threshold_is_strict(self):
        seq = _seq(W + 50)  # tail exactly 50: not added at T = 50, added at 49
        assert len(chunk_sequence(seq, "c1", W, OVERLAP, "stride", 50)) == 1
        assert len(chunk_sequence(seq, "c1", W, OVERLAP, "stride", 49)) == 2

    def test_no_tail_no_extra_window(self):
        seq = _seq(W + STEP)  # strided windows end exactly at the contig end
        chunks = chunk_sequence(seq, "c1", W, OVERLAP, "stride", 0)
        assert chunks == [seq[:W], seq[STEP : STEP + W]]

    def test_short_sequence_still_empty(self):
        assert chunk_sequence("ACGT", "c1", W, OVERLAP, "stride", 0) == []

    @pytest.mark.parametrize("threshold", [0, 50, 104])
    def test_uncovered_tail_never_exceeds_threshold(self, threshold):
        for n in range(W + 1, 1200):
            seq = _seq(n)
            chunks = chunk_sequence(seq, "c1", W, OVERLAP, "stride", threshold)
            assert all(len(c) == W for c in chunks)
            uncovered = n - (max(_covered(n, chunks, seq)) + 1)
            assert uncovered <= threshold, (n, uncovered)
            # and nothing is uncovered except possibly that 3' tail
            assert _covered(n, chunks, seq) == set(range(n - uncovered))


# ---------------------------------------------------------------------------
# chunk counting must match chunking exactly (the build preallocates the
# fbin from count_total_chunks, then fills it from _iter_chunks)
# ---------------------------------------------------------------------------


class TestChunkCountMatchesChunking:
    @pytest.mark.parametrize("threshold", [None, 0, 50, 104])
    def test_per_length(self, threshold):
        from src.dense_index import _chunks_for_length

        for n in range(0, 1200):
            got = _chunks_for_length(n, W, STEP, threshold)
            if n <= W:
                want = 1  # _iter_chunks passes these through whole
            else:
                want = len(
                    chunk_sequence(_seq(n), "c1", W, OVERLAP, "stride", threshold)
                )
            assert got == want, (n, threshold)

    @pytest.mark.parametrize("threshold", [None, 50])
    def test_per_file(self, tmp_path, threshold):
        from src.dense_index import _count_file_chunks

        lengths = [10, 256, 257, 300, 361, 362, 1000, 4096]
        fa = tmp_path / "x.contigs.fa"
        with open(fa, "w") as f:
            for i, n in enumerate(lengths):
                s = _seq(n)
                f.write(f">c{i}\n")
                for j in range(0, n, 60):  # wrapped like real FASTA
                    f.write(s[j : j + 60] + "\n")
        want = sum(
            1
            if n <= W
            else len(chunk_sequence(_seq(n), "c", W, OVERLAP, "stride", threshold))
            for n in lengths
        )
        assert _count_file_chunks(str(fa), W, STEP, threshold) == want
