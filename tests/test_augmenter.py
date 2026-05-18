import torch
import pytest

from lae.training.batcher import Augmenter, beta_parameters
from src.dense_index import _strings_to_one_hot

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
# _strings_to_one_hot
# ---------------------------------------------------------------------------


class TestStringsToOneHot:
    def test_known_bases_produce_correct_one_hot(self):
        result = _strings_to_one_hot(["ACGT"], len_sequence=4)
        expected = torch.eye(4)
        torch.testing.assert_close(result[0], expected)

    def test_unknown_character_produces_zero_row(self):
        result = _strings_to_one_hot(["N"], len_sequence=1)
        torch.testing.assert_close(result[0, 0], torch.zeros(4))

    def test_sequence_shorter_than_len_is_padded_with_zeros(self):
        result = _strings_to_one_hot(["AC"], len_sequence=4)
        # Positions 2 and 3 should be all-zero (padding)
        torch.testing.assert_close(result[0, 2], torch.zeros(4))
        torch.testing.assert_close(result[0, 3], torch.zeros(4))

    def test_sequence_longer_than_len_is_truncated(self):
        result = _strings_to_one_hot(["ACGTTTTT"], len_sequence=4)
        assert result.shape == (1, 4, 4)

    def test_output_shape_is_batch_x_len_x_4(self):
        batch = ["ACGT", "AAAA", "CCCC"]
        result = _strings_to_one_hot(batch, len_sequence=4)
        assert result.shape == (3, 4, 4)

    def test_case_insensitive(self):
        lower = _strings_to_one_hot(["acgt"], len_sequence=4)
        upper = _strings_to_one_hot(["ACGT"], len_sequence=4)
        torch.testing.assert_close(lower, upper)

    def test_each_row_sums_to_one_for_known_bases(self):
        result = _strings_to_one_hot(["ACGT"], len_sequence=4)
        row_sums = result[0].sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(4))
