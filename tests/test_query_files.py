"""The benchmark selects a pre-mutated query file per rate; it never mutates.

Bundles hold queries_mut0.00/0.05/0.10.parquet (locale-data/benchmark/
mutate_queries.py). The float on the CLI arrives as 0.1 or 0.10 and both must
map to the same file, and a mutation_rate of 0 must still read a per-rate file
so the loader has one rule.
"""

import importlib.util
from pathlib import Path

import pytest

RUN_BENCHMARK = Path(__file__).parent.parent / "benchmark" / "run_benchmark.py"


def _load_run_benchmark():
    spec = importlib.util.spec_from_file_location("run_benchmark", RUN_BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "rate, expected",
    [
        (0.0, "queries_mut0.00.parquet"),
        (0.05, "queries_mut0.05.parquet"),
        (0.1, "queries_mut0.10.parquet"),
        (0.10, "queries_mut0.10.parquet"),
    ],
)
def test_query_file_name(rate, expected):
    assert _load_run_benchmark().query_file_name(rate) == expected


def test_benchmark_has_no_mutation_code():
    source = RUN_BENCHMARK.read_text()
    assert "Augmenter" not in source
    assert "apply_mutations" not in source
