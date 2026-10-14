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


# ---------------------------------------------------------------------------
# min_seq_len: contigs below it are not embedded at all; the count, the
# iterator and meta.parquet completion must agree
# ---------------------------------------------------------------------------


def _write_fasta(path, records: dict[str, int]):
    with open(path, "w") as f:
        for name, n in records.items():
            s = _seq(n)
            f.write(f">{name}\n")
            for j in range(0, n, 60):
                f.write(s[j : j + 60] + "\n")


class TestMinSeqLen:
    def test_per_length(self):
        from src.dense_index import _chunks_for_length

        for n in range(0, 600):
            without = _chunks_for_length(n, W, STEP, 50)
            with_cut = _chunks_for_length(n, W, STEP, 50, min_seq_len=100)
            assert with_cut == (0 if n < 100 else without), n
        # the boundary is inclusive: a contig exactly min_seq_len long stays
        assert _chunks_for_length(100, W, STEP, None, min_seq_len=100) == 1
        assert _chunks_for_length(99, W, STEP, None, min_seq_len=100) == 0

    def test_count_matches_iterator(self, tmp_path):
        from src.dense_index import _count_file_chunks, iter_chunks

        acc = tmp_path / "SRR1"
        acc.mkdir()
        fa = acc / "SRR1.contigs.fa"
        lengths = {"a": 10, "b": 99, "c": 100, "d": 256, "e": 361, "f": 1000}
        _write_fasta(fa, lengths)
        for cut in (None, 1, 100, 256):
            got = list(iter_chunks([fa], W, OVERLAP, 50, cut))
            assert _count_file_chunks(str(fa), W, STEP, 50, cut) == len(got)
            assert all(srr == "SRR1" for srr, _ in got)
            # nothing shorter than the cutoff is ever yielded
            assert all(len(chunk) >= (cut or 0) for _, chunk in got)
        # with the cutoff, a / b are gone and the rest are unchanged
        full = [c for _, c in iter_chunks([fa], W, OVERLAP, 50, None)]
        cut = [c for _, c in iter_chunks([fa], W, OVERLAP, 50, 100)]
        assert cut == full[2:]

    def test_file_with_nothing_surviving_counts_zero(self, tmp_path):
        from src.dense_index import _count_file_chunks, iter_chunks

        fa = tmp_path / "x.contigs.fa"
        _write_fasta(fa, {"a": 31, "b": 45, "c": 60})
        assert _count_file_chunks(str(fa), W, STEP, None, 100) == 0
        assert list(iter_chunks([fa], W, OVERLAP, None, 100)) == []

    def test_complete_meta_rows_inserts_empty_accessions(self):
        from src.dense_index import complete_meta_rows

        rows = [
            {"srr_id": "B", "start_row": 0, "num_rows": 5},
            {"srr_id": "D", "start_row": 5, "num_rows": 3},
        ]
        out = complete_meta_rows(rows, ["A", "B", "C", "D", "E"])
        assert out == [
            {"srr_id": "A", "start_row": 0, "num_rows": 0},
            {"srr_id": "B", "start_row": 0, "num_rows": 5},
            {"srr_id": "C", "start_row": 5, "num_rows": 0},
            {"srr_id": "D", "start_row": 5, "num_rows": 3},
            {"srr_id": "E", "start_row": 8, "num_rows": 0},
        ]
        # DenseIndex.load's offsets stay contiguous and map rows to the
        # accession that owns them, empty ranges included.
        import numpy as np
        from src.topk_regroup import vector_ids_to_accessions

        starts = [r["start_row"] for r in out]
        counts = [r["num_rows"] for r in out]
        offsets = np.asarray(starts + [starts[-1] + counts[-1]])
        assert all(offsets[i + 1] - offsets[i] == counts[i] for i in range(5))
        names = vector_ids_to_accessions(
            np.arange(8), offsets, [r["srr_id"] for r in out]
        )
        assert names == ["B"] * 5 + ["D"] * 3

    def test_complete_meta_rows_is_identity_when_all_present(self):
        from src.dense_index import complete_meta_rows

        rows = [
            {"srr_id": "A", "start_row": 0, "num_rows": 2},
            {"srr_id": "B", "start_row": 2, "num_rows": 4},
        ]
        assert complete_meta_rows(rows, ["A", "B"]) == rows

    def test_complete_meta_rows_rejects_out_of_order_rows(self):
        from src.dense_index import complete_meta_rows

        rows = [
            {"srr_id": "B", "start_row": 0, "num_rows": 2},
            {"srr_id": "A", "start_row": 2, "num_rows": 1},
        ]
        with pytest.raises(ValueError, match="accession order"):
            complete_meta_rows(rows, ["A", "B"])
