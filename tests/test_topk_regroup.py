"""regroup_topk_hits / merge_topk_hits: pure polars, no model or GPU."""

import numpy as np
import polars as pl
import pytest
from src.topk_regroup import (
    HITS_DTYPE,
    MISS_SCORE,
    build_hits_frame,
    merge_topk_hits,
    regroup_topk_hits,
)

ACCS = ["accA", "accB", "accC", "accD"]


def _hits(rows_per_query: dict[str, list[tuple[str, float, int]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": list(rows_per_query),
            "hits": [
                [{"accession": a, "score": s, "vector_id": v} for a, s, v in rows]
                for rows in rows_per_query.values()
            ],
        },
        schema={"query_id": pl.String, "hits": HITS_DTYPE},
    )


def _scores(results: pl.DataFrame, query_id: str) -> dict[str, float]:
    row = results.filter(pl.col("query_id") == query_id)["results"][0]
    return {r["accession"]: r["score"] for r in row}


class TestRegroup:
    def test_every_accession_present_misses_get_sentinel(self):
        hits = _hits({"q0": [("accB", 0.9, 10), ("accB", 0.8, 11), ("accD", 0.7, 30)]})
        res = regroup_topk_hits(hits, ACCS)
        assert res.columns == ["query_id", "results"]
        s = _scores(res, "q0")
        assert set(s) == set(ACCS)
        assert s == {"accA": MISS_SCORE, "accB": 0.9, "accC": MISS_SCORE, "accD": 0.7}

    def test_truncation_to_k_changes_which_accessions_clear_the_cutoff(self):
        # Crowding: accB fills the first two slots, accD only enters at k=3.
        hits = _hits({"q0": [("accB", 0.9, 10), ("accB", 0.8, 11), ("accD", 0.7, 30)]})
        assert _scores(regroup_topk_hits(hits, ACCS, k=2), "q0")["accD"] == MISS_SCORE
        assert _scores(regroup_topk_hits(hits, ACCS, k=3), "q0")["accD"] == 0.7
        assert _scores(regroup_topk_hits(hits, ACCS, k=1), "q0")["accB"] == 0.9

    def test_full_k_equals_no_k(self):
        hits = _hits(
            {"q0": [("accB", 0.9, 10), ("accD", 0.7, 30)], "q1": [("accA", 0.5, 0)]}
        )
        assert regroup_topk_hits(hits, ACCS, k=2).equals(regroup_topk_hits(hits, ACCS))

    def test_results_follow_accession_order_and_query_order(self):
        hits = _hits({"q1": [("accC", 0.3, 20)], "q0": [("accA", 0.5, 0)]})
        res = regroup_topk_hits(hits, ACCS)
        assert res["query_id"].to_list() == ["q1", "q0"]
        assert [r["accession"] for r in res["results"][0]] == ACCS

    def test_empty_hits_list_is_all_sentinel(self):
        hits = _hits({"q0": []})
        assert set(_scores(regroup_topk_hits(hits, ACCS), "q0").values()) == {
            MISS_SCORE
        }

    def test_unknown_accession_raises(self):
        hits = _hits({"q0": [("accZ", 0.5, 0)]})
        with pytest.raises(ValueError):
            regroup_topk_hits(hits, ACCS)


class TestMerge:
    def test_union_retakes_global_top_k_and_dedups_vectors(self):
        a = _hits(
            {"q0": [("accA", 0.9, 1), ("accB", 0.5, 2)], "q1": [("accC", 0.2, 3)]}
        )
        b = _hits({"q0": [("accD", 0.7, 9), ("accA", 0.6, 8)], "q1": []})
        merged = merge_topk_hits([a, b], top_k=3)
        assert merged["query_id"].to_list() == ["q0", "q1"]
        q0 = merged["hits"][0].to_list()
        assert [r["vector_id"] for r in q0] == [1, 9, 8]
        assert merged["hits"][1].to_list() == [
            {"accession": "accC", "score": 0.2, "vector_id": 3}
        ]


class TestBuildHitsFrame:
    def test_dedups_chunks_by_vector_and_drops_padding(self):
        offsets = np.array([0, 2, 4])
        names = ["accA", "accB"]
        # query 0 has two chunks: both hit vector 3 (scores .5/.9), one padding.
        frame = build_hits_frame(
            query_ids=["q0", "q1"],
            flat_query_pos=np.array([0, 0, 0, 0]),
            flat_vector_ids=np.array([3, -1, 3, 0]),
            flat_scores=np.array([0.5, -np.inf, 0.9, 0.1]),
            acc_offsets=offsets,
            acc_names=names,
            top_k=5,
        )
        assert frame["hits"][0].to_list() == [
            {"accession": "accB", "score": 0.9, "vector_id": 3},
            {"accession": "accA", "score": 0.1, "vector_id": 0},
        ]
        assert frame["hits"][1].to_list() == []
