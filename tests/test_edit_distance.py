"""Exact edit-distance targets (lae/training/edit_distance.py), the regression
loss built on them (lae/training/losses.py), and the collate that carries them
into the training step."""

import random

import numpy as np
import pytest
import torch

from lae.config import AugmentConfig, TrainConfig
from lae.training.batcher import Augmenter
from lae.training.edit_distance import (
    edit_distance_matrix,
    length_matrix,
    normalized_edit_distance,
)
from lae.training.losses import edit_distance_loss, spearman


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _rnd(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


class TestEditDistanceMatrix:
    def test_identical_is_zero_and_symmetric(self):
        rng = random.Random(0)
        seqs = [_rnd(rng.randint(5, 60), rng) for _ in range(8)]
        d = edit_distance_matrix(seqs, seqs)
        assert d.dtype == np.int32 and d.shape == (8, 8)
        assert (np.diag(d) == 0).all()
        assert (d == d.T).all()

    def test_matches_pure_python_dp(self):
        rng = random.Random(1)
        a = [_rnd(rng.randint(1, 40), rng) for _ in range(6)]
        b = [_rnd(rng.randint(1, 40), rng) for _ in range(5)]
        d = edit_distance_matrix(a, b)
        for i, s in enumerate(a):
            for j, t in enumerate(b):
                assert d[i, j] == _levenshtein(s, t)

    def test_global_distance_counts_the_overhang(self):
        # The point of the experiment: a crop contained in a longer crop is
        # at least (long - short) away even with no mutation, unlike the
        # infix distance the MoCo filter uses.
        long = "ACGT" * 64
        short = long[50:150]
        assert edit_distance_matrix([short], [long])[0, 0] == len(long) - len(short)

    def test_containment_positive_lower_bound_under_recipe(self):
        torch.manual_seed(0)
        rng = random.Random(2)
        aug = Augmenter(
            AugmentConfig(
                max_len=256, max_seq_len=256, min_seq_len=31,
                min_overlap_percent=0.4, containment_prob=1.0, overlap_prob=0.0,
                disable_mutations=True,
            )
        )
        seq = _rnd(2000, rng)
        for _ in range(50):
            q, r = aug.get_pairs(seq)
            d = edit_distance_matrix([q], [r])[0, 0]
            assert d == abs(len(q) - len(r))  # unmutated containment: exactly the overhang

    def test_normalizations(self):
        a, b = ["AAAA", "CC"], ["AAAA", "GGGGGG"]
        assert (length_matrix(a, b, "max_len") == [[4, 6], [4, 6]]).all()
        assert (length_matrix(a, b, "sum_len") == [[8, 10], [6, 8]]).all()
        assert (length_matrix(a, b, "none") == 1).all()
        n = normalized_edit_distance(a, b, "max_len")
        assert n.dtype == np.float32
        assert n[0, 0] == 0.0 and 0.0 <= n.min() and n.max() <= 1.0
        with pytest.raises(ValueError):
            length_matrix(a, b, "bogus")

    def test_n_is_its_own_symbol(self):
        assert edit_distance_matrix(["ACNT"], ["ACGT"])[0, 0] == 1
        assert edit_distance_matrix(["ACNT"], ["ACNT"])[0, 0] == 0


class TestEditDistanceLoss:
    def test_zero_loss_at_exact_geometry(self):
        # cos = 1 - target / scale reproduces the targets exactly.
        torch.manual_seed(0)
        b, d = 4, 8
        q = torch.nn.functional.normalize(torch.randn(b, d), dim=1)
        k = torch.nn.functional.normalize(torch.randn(b, d), dim=1)
        log_scale = torch.tensor(0.0)  # scale 1
        target = 1.0 - q @ k.T
        loss, pred, tgt, m = edit_distance_loss(q, k, None, target, None, log_scale)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert torch.allclose(pred, tgt)
        assert m["train/ed_spearman"] == pytest.approx(1.0, abs=1e-5)
        assert m["train/ed_scale"] == pytest.approx(1.0)

    def test_negatives_are_scored_and_scale_gets_gradient(self):
        torch.manual_seed(1)
        b, d = 3, 5
        q = torch.nn.functional.normalize(torch.randn(b, d), dim=1)
        k = torch.nn.functional.normalize(torch.randn(b, d), dim=1)
        n = torch.nn.functional.normalize(torch.randn(b, d), dim=1)
        log_scale = torch.nn.Parameter(torch.tensor(0.3))
        t_qk = torch.full((b, b), 0.5); t_qk.fill_diagonal_(0.05)
        t_qn = torch.full((b, b), 0.55)
        loss, pred, tgt, m = edit_distance_loss(q, k, n, t_qk, t_qn, log_scale)
        assert pred.shape == (b, 2 * b) and tgt.shape == (b, 2 * b)
        loss.backward()
        assert log_scale.grad is not None and log_scale.grad.abs() > 0
        assert m["train/ed_target_pos"] == pytest.approx(0.05)
        assert m["train/ed_target_neg"] == pytest.approx((0.5 * 6 + 0.55 * 9) / 15)
        assert m["train/ed_pos_beyond_median_neg"] == 0.0

    def test_spearman(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert spearman(x, x * 10).item() == pytest.approx(1.0)
        assert spearman(x, -x).item() == pytest.approx(-1.0)


class TestConfigAndCollate:
    def test_queue_rejected_with_edit_distance_loss(self):
        with pytest.raises(ValueError):
            TrainConfig(dataset_path="x", loss="edit_distance", moco_queue_size=16)
        TrainConfig(dataset_path="x", loss="edit_distance", moco_queue_size=0)

    def test_collate_attaches_targets(self):
        from lae.training.training import collate_with_edit_distance

        class Tok:
            def __call__(self, seqs, return_tensors, padding):
                return {"n": len(seqs)}

        batch = [
            {"query": "ACGTACGT", "ref": "ACGTTCGT", "negative": "GGGGGGGG"},
            {"query": "TTTT", "ref": "TTTTAA", "negative": None},
        ]
        out = collate_with_edit_distance(batch, Tok(), True, "max_len")
        *rest, targets = out
        assert len(rest) == 6
        # The None-negative row is dropped before the targets are built.
        assert targets["qk"].shape == (1, 1) and targets["qn"].shape == (1, 1)
        assert targets["qk"][0, 0] == pytest.approx(1 / 8)
        assert targets["qn"][0, 0] == pytest.approx(
            _levenshtein("ACGTACGT", "GGGGGGGG") / 8
        )
        out = collate_with_edit_distance(batch, Tok(), False, "none")
        *rest, targets = out
        assert len(rest) == 4
        assert targets["qn"] is None
        assert targets["qk"].shape == (2, 2)
        assert targets["qk"][1, 1] == 2 and targets["qk"][0, 0] == 1
