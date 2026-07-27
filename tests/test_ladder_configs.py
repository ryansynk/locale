"""Guards for the backbone-swap experiment configs.

The experiment's whole claim rests on holding the recipe fixed and varying only
(a) the augmentation strength within a backbone and (b) the backbone itself. A
stray edit to one ladder rung would silently confound the result, so these tests
assert the configs stay identical except where they are supposed to differ.
"""

from pathlib import Path

import pytest
import yaml

from lae.config import AugmentConfig, TrainConfig
from lae.modeling.backbones import BACKBONES

CONFIG_DIR = Path(__file__).parent.parent / "configs"

BACKBONE_IDS = ["nt50m", "hyenadna"]
RUNGS = ["none", "light", "medium", "heavy"]

# The paper's augmentation ladder (method_context.md, Table 3 rows).
EXPECTED_AUG = {
    "none": {"disable_mutations": True},
    "light": {
        "disable_mutations": False,
        "identity_mean": 95.0,
        "identity_stdev": 2.5,
        "identity_max": 99.0,
    },
    "medium": {
        "disable_mutations": False,
        "identity_mean": 90.0,
        "identity_stdev": 6.0,
        "identity_max": 98.0,
    },
    "heavy": {
        "disable_mutations": False,
        "identity_mean": 80.0,
        "identity_stdev": 6.0,
        "identity_max": 88.0,
    },
}

# Only these may differ between rungs of the same ladder.
MUTABLE_AUG_KEYS = {
    "disable_mutations",
    "identity_mean",
    "identity_stdev",
    "identity_max",
}


def _load(backbone: str, rung: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{backbone}_{rung}.yaml").read_text())


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_config_parses_as_train_config(backbone, rung):
    cfg = _load(backbone, rung)
    aug = cfg.pop("augment_config")
    TrainConfig(**cfg, augment_config=AugmentConfig(**aug))


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_backbone_field_matches_filename(backbone, rung):
    assert _load(backbone, rung)["backbone"] == backbone
    assert backbone in BACKBONES


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_augmentation_matches_paper_ladder(backbone, rung):
    aug = _load(backbone, rung)["augment_config"]
    for key, expected in EXPECTED_AUG[rung].items():
        assert aug[key] == expected, f"{backbone}/{rung}: {key}"


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
def test_rungs_differ_only_in_mutation_settings(backbone):
    """The load-bearing invariant: within a ladder, only the ablated axis moves."""
    baseline = _load(backbone, "heavy")
    baseline_aug = baseline.pop("augment_config")

    for rung in RUNGS:
        if rung == "heavy":
            continue
        other = _load(backbone, rung)
        other_aug = other.pop("augment_config")

        assert other == baseline, f"{backbone}/{rung} differs outside augment_config"
        assert set(other_aug) == set(baseline_aug)
        for key in baseline_aug:
            if key in MUTABLE_AUG_KEYS:
                continue
            assert other_aug[key] == baseline_aug[key], f"{backbone}/{rung}: {key}"


def test_backbones_differ_only_in_backbone_field():
    """Across backbones, the recipe must be identical at every rung."""
    for rung in RUNGS:
        a = _load("nt50m", rung)
        b = _load("hyenadna", rung)
        assert a.pop("backbone") == "nt50m"
        assert b.pop("backbone") == "hyenadna"
        assert a == b, f"rung {rung} differs between backbones beyond the backbone id"


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
@pytest.mark.parametrize("rung", RUNGS)
def test_recipe_constants_match_paper(backbone, rung):
    cfg = _load(backbone, rung)
    aug = cfg["augment_config"]

    assert cfg["moco_softmax_temp"] == 0.05  # tau
    assert cfg["moco_queue_size"] == 0  # plain InfoNCE, MoCo path off
    assert cfg["backbone_lr"] == 6.0e-05
    assert cfg["pooling"] == "mean"
    assert cfg["use_projection_head"] is False
    assert cfg["total_samples"] == 6_000_000
    assert cfg["reference_global_batch_size"] == 1024

    assert aug["min_seq_len"] == 31  # crop lengths uniform in [31, 256]
    assert aug["max_seq_len"] == 256
    assert aug["containment_prob"] == 1.0  # containment cropping only
    assert aug["overlap_prob"] == 0.0
    assert aug["min_overlap_percent"] == 0.4


@pytest.mark.parametrize("backbone", BACKBONE_IDS)
def test_smoke_config_checkpoint_interval_is_safe(backbone):
    """checkpoint_interval_steps is an integer division in training.py; if it
    rounds to zero the run dies on a modulo-by-zero partway through."""
    cfg = _load_smoke(backbone)
    global_batch = cfg["per_device_batch_size"]  # smoke runs on a single GPU
    assert cfg["checkpoint_interval_samples"] // global_batch > 0


def _load_smoke(backbone: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"smoke_{backbone}.yaml").read_text())
