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
