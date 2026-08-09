"""Guard that every benchmark config names a dataset the registry knows.

`run_benchmark.main()` resolves `cfg.dataset_name` through the DATASETS dict on
its first line of real work, so a typo there raises KeyError only after SLURM has
granted the allocation and the process has imported torch. On a queued multi-GPU
job that is a slow, expensive way to learn about a misspelling.

DATASETS is read with `ast` rather than imported: importing run_benchmark pulls
in torch, the lae package and the whole src.* index stack, which is far too much
machinery for a string lookup and would make this test the slowest in the suite.
"""

import ast
from pathlib import Path

import pytest
import yaml

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"
CONFIG_DIR = BENCHMARK_DIR / "configs"


def _registry() -> dict[str, str]:
    tree = ast.parse((BENCHMARK_DIR / "run_benchmark.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "DATASETS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("DATASETS not found in run_benchmark.py")


CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_registry_is_non_empty():
    assert _registry()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_dataset_name_is_registered(path):
    cfg = yaml.safe_load(path.read_text()) or {}
    name = cfg.get("dataset_name")
    if name is None:
        pytest.skip("config does not set dataset_name")
    registry = _registry()
    assert name in registry, (
        f"{path.name}: dataset_name {name!r} is not in DATASETS "
        f"(known: {sorted(registry)})"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_dataset_dir_is_not_shared_across_datasets(path):
    """Two datasets must never materialise into the same dataset_dir.

    run_benchmark only downloads contigs when logan_accessions is empty, so a
    dataset_dir already populated by a different dataset is silently reused: the
    manifest check passes on the subset that happens to overlap and the run
    reports recall over the wrong accession set. Datasets drawn from a shared
    pool of distractors are the dangerous case -- the overlap is large enough to
    look healthy while the accessions unique to the new dataset are missing.
    """
    cfg = yaml.safe_load(path.read_text()) or {}
    name, ddir = cfg.get("dataset_name"), cfg.get("dataset_dir")
    if name is None or ddir is None:
        pytest.skip("config does not set both dataset_name and dataset_dir")

    conflicts = []
    for other in CONFIGS:
        ocfg = yaml.safe_load(other.read_text()) or {}
        if ocfg.get("dataset_dir") == ddir and ocfg.get("dataset_name") not in (
            None,
            name,
        ):
            conflicts.append(f"{other.name} ({ocfg['dataset_name']})")
    assert not conflicts, (
        f"{path.name}: dataset_dir {ddir} is also used by {conflicts} "
        f"for a different dataset"
    )
