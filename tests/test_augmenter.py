import pytest
import torch

from lae.config import AugmentConfig
from lae.training.batcher import Augmenter, beta_parameters

_BASES = set("ACGT")


# ---------------------------------------------------------------------------
# Augmenter._uniform_random_mutation
# ---------------------------------------------------------------------------


class TestUniformRandomMutation:
    def test_identity_one_returns_original_sequence(self):
        seq = "ACGTACGTACGT"
        result, actual_id = Augmenter._uniform_random_mutation(seq, 1.0)
        assert result == seq
        assert actual_id == 1.0

    def test_output_contains_only_acgt(self):
        torch.manual_seed(0)
        seq = "ACGT" * 50
        result, _ = Augmenter._uniform_random_mutation(seq, 0.7)
        assert set(result).issubset(_BASES)

    def test_actual_identity_close_to_target(self):
        torch.manual_seed(1)
        seq = "ACGT" * 100  # 400 bp — large enough for stable estimate
        _, actual_id = Augmenter._uniform_random_mutation(seq, 0.9)
        assert 0.75 <= actual_id <= 1.0

    def test_zero_identity_mutates_all_positions(self):
        torch.manual_seed(2)
        seq = "A" * 50
        result, actual_id = Augmenter._uniform_random_mutation(seq, 0.0)
        assert actual_id < 0.5

    def test_single_character_sequence_does_not_crash(self):
        result, _ = Augmenter._uniform_random_mutation("A", 0.5)
        assert isinstance(result, str)

    def test_returns_string_and_float(self):
        result, actual_id = Augmenter._uniform_random_mutation("ACGT", 0.8)
        assert isinstance(result, str)
        assert isinstance(actual_id, float)

    def test_identity_bounds_are_valid(self):
        torch.manual_seed(3)
        seq = "ACGT" * 25
        _, actual_id = Augmenter._uniform_random_mutation(seq, 0.5)
        assert 0.0 <= actual_id <= 1.0


# ---------------------------------------------------------------------------
# beta_parameters
# ---------------------------------------------------------------------------


class TestBetaParameters:
    def test_valid_params_return_positive_values(self):
        a, b = beta_parameters(beta_mean=0.8, beta_stdev=0.05, beta_max=1.0)
        assert a > 0
        assert b > 0

    def test_invalid_params_raise_value_error(self):
        # stdev=20 relative to mean=95, max=99 makes beta_a negative
        with pytest.raises(ValueError):
            beta_parameters(beta_mean=95, beta_stdev=20.0, beta_max=99)

    def test_returns_two_values(self):
        result = beta_parameters(beta_mean=0.8, beta_stdev=0.05, beta_max=1.0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Augmenter._sample_identity (Badread semantics: draw scaled by identity_max)
# ---------------------------------------------------------------------------


class TestSampleIdentity:
    @staticmethod
    def _draws(mean, stdev, maximum, n=20000):
        torch.manual_seed(0)
        aug = Augmenter(
            AugmentConfig(identity_mean=mean, identity_stdev=stdev, identity_max=maximum)
        )
        return torch.tensor([aug._sample_identity() for _ in range(n)])

    def test_heavy_rung_matches_badread_mean_and_max(self):
        x = self._draws(80.0, 6.0, 88.0)
        assert abs(x.mean().item() - 0.80) < 0.01
        assert abs(x.std().item() - 0.06) < 0.01
        assert x.max().item() <= 0.88
        assert x.min().item() >= 0.0

    def test_light_rung_matches_badread_mean_and_max(self):
        x = self._draws(95.0, 2.5, 99.0)
        assert abs(x.mean().item() - 0.95) < 0.005
        assert x.max().item() <= 0.99

    def test_fractional_config_is_not_rescaled(self):
        x = self._draws(0.8, 0.05, 1.0)
        assert abs(x.mean().item() - 0.80) < 0.01
        assert x.max().item() <= 1.0
