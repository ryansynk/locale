"""Guards for the benchmark eval configs driven by eval_{nt50m,hyenadna}.sh.

The failure these exist to catch is a mislabelled rung: eval configs differ only
in one path, so a wrong run id still produces a complete, plausible Table 3 with
two columns swapped. Nothing downstream would flag it — make_table3 reads the
rung from the filename and the checkpoint from the config, and never checks that
the two agree. These tests do.
"""

from pathlib import Path

import pytest
import yaml

from tests.test_ladder_configs import EXPECTED_AUG

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"
CONFIG_DIR = BENCHMARK_DIR / "configs"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

BACKBONE_IDS = ["nt50m", "hyenadna", "dna2vec"]
RUNGS = ["none", "light", "medium", "heavy"]

# Everything the eval must hold fixed for the four columns to be comparable.
# max_seq_len/pooling additionally feed the index suffix that both the sweep
# script and make_table3 reconstruct by hand, so a change here silently
# desynchronises all three.
EXPECTED_FIXED = {
    "dataset_name": "sra50",
    "dataset_dir": "/fs/nexus-projects/sra_search/hf/sra50",
    "index_dir": "/fs/nexus-projects/sra_search/indexes_locale",
}
EXPECTED_MODEL_FIXED = {
    "name": "locale",
    "pooling": "mean",
    "max_seq_len": 256,
    "chunk_overlap": 150,
}


def _load(backbone: str, rung: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{backbone}_{rung}.yaml").read_text())


def _runs(backbone: str) -> dict:
    spec = yaml.safe_load((BENCHMARK_DIR / f"runs_{backbone}.yaml").read_text())
    return spec, spec.get("runs") or {}


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_config_exists_and_holds_the_recipe_fixed(backbone, rung):
    cfg = _load(backbone, rung)
    for key, expected in EXPECTED_FIXED.items():
        assert cfg[key] == expected, f"{backbone}/{rung}: {key}"
    for key, expected in EXPECTED_MODEL_FIXED.items():
        assert cfg["model"][key] == expected, f"{backbone}/{rung}: model.{key}"


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
def test_rungs_differ_only_in_checkpoint_path(backbone):
    configs = {rung: _load(backbone, rung) for rung in RUNGS}
    reference = configs["none"]
    for rung, cfg in configs.items():
        if rung == "none":
            continue
        assert cfg.keys() == reference.keys()
        for key in reference:
            if key == "model":
                continue
            assert cfg[key] == reference[key], f"{backbone}/{rung} differs in {key}"
        for key in reference["model"]:
            if key == "checkpoint_path":
                continue
            assert cfg["model"][key] == reference["model"][key], (
                f"{backbone}/{rung} differs in model.{key}"
            )


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_checkpoint_path_matches_recorded_run_id(backbone, rung):
    """The mislabelling guard: config rung -> run id must match runs_*.yaml."""
    spec, runs = _runs(backbone)
    if rung not in runs:
        pytest.skip(f"{backbone}/{rung} not yet trained")
    path = Path(_load(backbone, rung)["model"]["checkpoint_path"])
    assert path.parent.name == runs[rung], (
        f"{backbone}/{rung} eval config points at {path.parent.name}, "
        f"but runs_{backbone}.yaml records {runs[rung]}"
    )
    assert path.name == f"checkpoint{spec['step']}.pth.tar"


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
def test_run_ids_are_distinct(backbone):
    _, runs = _runs(backbone)
    assert len(set(runs.values())) == len(runs), f"duplicate run id in {backbone}"


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_checkpoint_on_disk_was_trained_by_this_backbone_and_rung(backbone, rung):
    """Cross-check the run id against the config train.py actually saved.

    runs_*.yaml is appended by the sweep scripts from whatever wandb id was on
    disk at the time; this is the independent confirmation that the id really
    belongs to the rung it was filed under.
    """
    _, runs = _runs(backbone)
    if rung not in runs:
        pytest.skip(f"{backbone}/{rung} not yet trained")
    train_config = CHECKPOINT_DIR / runs[rung] / "config.yaml"
    if not train_config.exists():
        pytest.skip(f"{train_config} not on this machine")
    cfg = yaml.safe_load(train_config.read_text())
    assert cfg["backbone"] == backbone
    augment = cfg["augment_config"]
    for key, expected in EXPECTED_AUG[rung].items():
        assert augment[key] == expected, (
            f"{backbone}/{rung} ({runs[rung]}) trained with {key}={augment[key]}, "
            f"expected {expected}"
        )


def test_every_backbone_and_rung_has_a_config():
    """The grid is complete, so no rung silently drops out of the table."""
    for backbone in BACKBONE_IDS:
        for rung in RUNGS:
            assert (CONFIG_DIR / f"{backbone}_{rung}.yaml").exists()
