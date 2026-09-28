import torch
import pytest

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
