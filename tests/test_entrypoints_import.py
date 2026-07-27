"""Entry points must at least import.

train.py sat un-importable on this branch from 3b00460 (2026-05-15) until it was
noticed by a failed 8-GPU launch: a cleanup deleted train_kl from
lae/training/training.py but left the import and call site behind. Nothing in
the suite touched the entry point, so nothing caught it. These tests are cheap
and catch that entire class of bug.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

ENTRYPOINTS = [
    REPO_ROOT / "train.py",
    REPO_ROOT / "benchmark" / "run_benchmark.py",
    REPO_ROOT / "benchmark" / "make_table3.py",
]


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_imports(path):
    """Importing must not raise. The __main__ guard keeps this side-effect free."""
    assert path.exists(), f"missing entry point: {path}"
    spec = importlib.util.spec_from_file_location(f"_entrypoint_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_train_entrypoint_exposes_main_and_train():
    path = REPO_ROOT / "train.py"
    spec = importlib.util.spec_from_file_location("_entrypoint_train_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module.train)
