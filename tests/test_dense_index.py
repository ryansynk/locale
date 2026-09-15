"""
Synthetic DenseIndex search tests — no GPU or model download required.

We bypass __init__ with __new__ so no DenseEncoder is created, then wire in
a FakeEncoder that returns predetermined embeddings. Using standard basis
vectors makes the expected top-1 result deterministic: the query is e_2, so
only acc1 (whose vectors include e_2) can score 1.0.
"""

import torch
import polars as pl
import pytest
from types import SimpleNamespace

from src.dense_index import DenseIndex


def _basis(dim: int, i: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


class _FakeEncoder:
    """Returns a fixed embedding regardless of input sequences."""

    def __init__(self, vec: torch.Tensor):
        self._vec = vec

    def encode(self, sequences):
        return self._vec.unsqueeze(0).expand(len(sequences), -1)


def _make_index(query_vec: torch.Tensor, all_embeddings: torch.Tensor):
    """Construct a DenseIndex with synthetic state, bypassing __init__."""
    index = DenseIndex.__new__(DenseIndex)
    index.no_search = False
    index.k = 10
    index.use_ann = False
    index.exact_search = False
    index.use_rabitq = False
    index.top_k = 10
    index.all_embeddings = all_embeddings
    index.acc_names_flat = ["acc0", "acc1", "acc2"]
    index.acc_offsets = [0, 2, 4, 6]
    index.model_cfg = SimpleNamespace(device="cpu", max_seq_len=50)
    index.chunk_type = "stride"
    index.chunk_overlap = 0
    index.contig_align_intervals = None
    index.model = _FakeEncoder(query_vec)
    return index


# 3 accessions × 2 vectors each; dim=6 so we have one basis vector per slot.
# Slot layout: acc0=[e0,e1], acc1=[e2,e3], acc2=[e4,e5]
_DIM = 6
_ALL_EMBEDDINGS = torch.stack([_basis(_DIM, i) for i in range(6)])  # (6, 6)


class TestDenseIndexSearch:
    def test_exact_match_ranks_first(self):
        # Query = e2 → perfect dot-product score of 1.0 only against acc1
        query_vec = _basis(_DIM, 2)
        index = _make_index(query_vec, _ALL_EMBEDDINGS)

        queries = pl.DataFrame({"query_sequence": ["ACGT"], "query_id": ["q0"]})
        result = index.search(queries)

        top_acc = max(result["results"][0], key=lambda r: r["score"])["accession"]
        assert top_acc == "acc1"

    def test_different_query_hits_different_accession(self):
        # Query = e4 → perfect score only against acc2
        query_vec = _basis(_DIM, 4)
        index = _make_index(query_vec, _ALL_EMBEDDINGS)

        queries = pl.DataFrame({"query_sequence": ["ACGT"], "query_id": ["q0"]})
        result = index.search(queries)

        top_acc = max(result["results"][0], key=lambda r: r["score"])["accession"]
        assert top_acc == "acc2"

    def test_output_schema(self):
        query_vec = _basis(_DIM, 0)
        index = _make_index(query_vec, _ALL_EMBEDDINGS)

        queries = pl.DataFrame({"query_sequence": ["ACGT"], "query_id": ["q0"]})
        result = index.search(queries)

        assert result.columns == ["query_id", "results"]
        assert len(result) == 1

    def test_all_accessions_appear_in_results(self):
        query_vec = _basis(_DIM, 0)
        index = _make_index(query_vec, _ALL_EMBEDDINGS)

        queries = pl.DataFrame({"query_sequence": ["ACGT"], "query_id": ["q0"]})
        result = index.search(queries)

        returned_accs = {r["accession"] for r in result["results"][0]}
        assert returned_accs == {"acc0", "acc1", "acc2"}

    def test_multiple_queries_each_get_results(self):
        # Two queries: e0 → acc0, e4 → acc2
        class _MultiFakeEncoder:
            def __init__(self, vecs):
                self._vecs = vecs
                self._call = 0

            def encode(self, sequences):
                v = self._vecs[self._call % len(self._vecs)]
                self._call += 1
                return v.unsqueeze(0).expand(len(sequences), -1)

        index = _make_index(_basis(_DIM, 0), _ALL_EMBEDDINGS)
        index.model = _MultiFakeEncoder([_basis(_DIM, 0), _basis(_DIM, 4)])

        queries = pl.DataFrame(
            {"query_sequence": ["ACGT", "TTTT"], "query_id": ["q0", "q1"]}
        )
        result = index.search(queries)
        assert len(result) == 2

    def test_non_negative_scores_returned(self):
        query_vec = _basis(_DIM, 2)
        index = _make_index(query_vec, _ALL_EMBEDDINGS)

        queries = pl.DataFrame({"query_sequence": ["ACGT"], "query_id": ["q0"]})
        result = index.search(queries)

        scores = [r["score"] for r in result["results"][0]]
        # Basis vectors are orthonormal: dot products are 0.0 or 1.0
        assert all(s >= 0.0 for s in scores)


class TestExactTopKSearch:
    """Exact top-k scan on the same synthetic index, on CPU.

    Basis vectors give exact-arithmetic scores: the query e_i scores 1.0 against
    slot i and 0.0 against the other five, so ranks and ties are known.
    """

    def _queries(self, n=1):
        return pl.DataFrame(
            {"query_sequence": ["ACGT"] * n, "query_id": [f"q{i}" for i in range(n)]}
        )

    def test_hits_are_sorted_and_map_to_the_right_accession(self):
        index = _make_index(_basis(_DIM, 2), _ALL_EMBEDDINGS)
        index.exact_search = True
        index.top_k = 3
        hits = index.exact_topk_hits(self._queries())
        assert hits.columns == ["query_id", "hits"]
        rows = hits["hits"][0].to_list()
        assert len(rows) == 3
        assert rows[0] == {"accession": "acc1", "score": 1.0, "vector_id": 2}
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_index_returns_every_vector_once(self):
        index = _make_index(_basis(_DIM, 0), _ALL_EMBEDDINGS)
        index.exact_search = True
        index.top_k = 50
        hits = index.exact_topk_hits(self._queries())
        ids = sorted(r["vector_id"] for r in hits["hits"][0].to_list())
        assert ids == list(range(6))  # no -1 padding leaks through

    # A query with a distinct dot product against every slot, so top-k over the
    # six basis vectors has no ties for torch.topk to break arbitrarily.
    _DISTINCT = torch.tensor([0.1, 0.6, 0.3, 0.9, 0.2, 0.5])

    def test_block_streaming_matches_single_block(self):
        index = _make_index(self._DISTINCT, _ALL_EMBEDDINGS)
        index.exact_search = True
        index.top_k = 4
        one = index.exact_topk_hits(self._queries(), block_rows=1_000_000)
        many = index.exact_topk_hits(self._queries(), block_rows=1)
        assert one["hits"].to_list() == many["hits"].to_list()

    def test_sharded_ranges_merge_to_the_full_scan(self):
        from src.topk_regroup import merge_topk_hits

        index = _make_index(self._DISTINCT, _ALL_EMBEDDINGS)
        index.exact_search = True
        index.top_k = 4
        full = index.exact_topk_hits(self._queries())
        assert [r["vector_id"] for r in full["hits"][0].to_list()] == [3, 1, 5, 2]
        parts = [
            index.exact_topk_hits(self._queries(), vec_range=(0, 2)),
            index.exact_topk_hits(self._queries(), vec_range=(2, 5)),
            index.exact_topk_hits(self._queries(), vec_range=(5, 6)),
        ]
        assert all(len(p["hits"][0]) <= 4 for p in parts)
        merged = merge_topk_hits(parts, index.top_k)
        key = lambda r: (-r["score"], r["vector_id"])
        assert sorted(merged["hits"][0].to_list(), key=key) == sorted(
            full["hits"][0].to_list(), key=key
        )
        assert merged["hits"][0].to_list()[0]["vector_id"] == 3

    def test_empty_range_yields_empty_hits(self):
        index = _make_index(_basis(_DIM, 3), _ALL_EMBEDDINGS)
        index.exact_search = True
        hits = index.exact_topk_hits(self._queries(), vec_range=(4, 4))
        assert hits["hits"][0].to_list() == []

    def test_search_regroups_with_sentinel_for_misses(self):
        index = _make_index(_basis(_DIM, 2), _ALL_EMBEDDINGS)
        index.exact_search = True
        index.top_k = 1
        result = index.search(self._queries())
        assert result.columns == ["query_id", "results"]
        by_acc = {r["accession"]: r["score"] for r in result["results"][0]}
        assert set(by_acc) == {"acc0", "acc1", "acc2"}
        assert by_acc["acc1"] == 1.0
        assert by_acc["acc0"] == -2.0 and by_acc["acc2"] == -2.0

    def test_exact_search_top1_agrees_with_streaming_search(self):
        for slot in range(6):
            stream = _make_index(_basis(_DIM, slot), _ALL_EMBEDDINGS)
            exact = _make_index(_basis(_DIM, slot), _ALL_EMBEDDINGS)
            exact.exact_search = True

            def top(df):
                return max(df["results"][0], key=lambda r: r["score"])["accession"]

            assert top(stream.search(self._queries())) == top(
                exact.search(self._queries())
            )
