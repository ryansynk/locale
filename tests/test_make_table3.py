"""Recall@Rq is the headline number of the backbone-swap experiment, so pin its
behaviour on cases where the right answer is obvious by hand."""

import numpy as np
import polars as pl
import pytest

from make_table3 import _recall_at_rq


def _results(rows: list[tuple[str, list[tuple[str, float]]]]) -> pl.DataFrame:
    """Build a results frame in the shape run_benchmark.py writes."""
    return pl.DataFrame(
        [
            {
                "query_id": qid,
                "results": [{"accession": a, "score": s} for a, s in scored],
            }
            for qid, scored in rows
        ]
    )


def test_perfect_ranking_scores_one():
    results = _results([("q1", [("A", 0.9), ("B", 0.8), ("C", 0.1)])])
    values = _recall_at_rq(results, {"q1": {"A", "B"}})
    assert values.tolist() == [1.0]


def test_worst_ranking_scores_zero():
    results = _results([("q1", [("A", 0.1), ("B", 0.2), ("C", 0.9)])])
    values = _recall_at_rq(results, {"q1": {"A"}})
    assert values.tolist() == [0.0]


def test_partial_recall():
    # Rq = 2, top-2 by score is (C, A); only A is relevant -> 1/2.
    results = _results([("q1", [("A", 0.8), ("B", 0.1), ("C", 0.9)])])
    values = _recall_at_rq(results, {"q1": {"A", "B"}})
    assert values.tolist() == [0.5]


def test_queries_with_no_relevant_accessions_are_dropped():
    results = _results(
        [
            ("q1", [("A", 0.9), ("B", 0.1)]),
            ("q2", [("A", 0.9), ("B", 0.1)]),
        ]
    )
    values = _recall_at_rq(results, {"q1": {"A"}})
    assert values.tolist() == [1.0], "q2 has no gold set and must not count as 0.0"


def test_mean_is_averaged_over_queries_not_accessions():
    results = _results(
        [
            ("q1", [("A", 0.9), ("B", 0.1)]),
            ("q2", [("A", 0.1), ("B", 0.9)]),
        ]
    )
    values = _recall_at_rq(results, {"q1": {"A"}, "q2": {"A"}})
    assert np.isclose(values.mean(), 0.5)


@pytest.mark.parametrize("rung", ["none", "heavy"])
def test_recall_is_a_fraction(rung):
    rng = np.random.default_rng(0)
    accs = [f"acc{i}" for i in range(20)]
    results = _results(
        [(f"q{q}", list(zip(accs, rng.random(len(accs))))) for q in range(10)]
    )
    relevant = {f"q{q}": {accs[q % len(accs)]} for q in range(10)}
    values = _recall_at_rq(results, relevant)
    assert len(values) == 10
    assert ((values >= 0.0) & (values <= 1.0)).all()
