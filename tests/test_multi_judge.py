"""
Unit tests for multi-judge reward aggregation functions.

Tests the core multi-judge functionality including:
- Pairwise accuracy computation
- Kendall's tau agreement computation
- Fleiss's kappa computation
- Reward aggregation
"""

import pytest
from open_instruct.search_rewards.rubric_judge_rewards import (
    compute_binary_judge_votes,
    compute_pairwise_accuracy,
    compute_fleiss_kappa,
    compute_judge_agreement,
    aggregate_multi_judge_reward,
    resolve_multi_judge_votes,
    aggregate_multi_judge_comparison,
)


class TestComputePairwiseAccuracy:
    """Tests for compute_pairwise_accuracy function."""

    def test_perfect_agreement(self):
        """All judges rank accepted > rejected."""
        accepted = [0.8, 0.9, 0.7]
        rejected = [0.5, 0.6, 0.4]
        accuracy = compute_pairwise_accuracy(accepted, rejected)
        assert accuracy == 1.0, "All judges ranked correctly, should be 100%"

    def test_zero_agreement(self):
        """No judges rank accepted > rejected."""
        accepted = [0.3, 0.4, 0.2]
        rejected = [0.5, 0.6, 0.7]
        accuracy = compute_pairwise_accuracy(accepted, rejected)
        assert accuracy == 0.0, "No judges ranked correctly, should be 0%"

    def test_partial_agreement(self):
        """Some judges rank correctly."""
        accepted = [0.8, 0.4, 0.7]  # Judge 2 ranks incorrectly
        rejected = [0.5, 0.6, 0.4]
        accuracy = compute_pairwise_accuracy(accepted, rejected)
        assert accuracy == pytest.approx(2.0 / 3.0), "2 out of 3 judges correct"

    def test_single_judge(self):
        """Single judge case."""
        accepted = [0.8]
        rejected = [0.5]
        accuracy = compute_pairwise_accuracy(accepted, rejected)
        assert accuracy == 1.0

    def test_empty_lists(self):
        """Empty lists should return 0.0."""
        accepted = []
        rejected = []
        accuracy = compute_pairwise_accuracy(accepted, rejected)
        assert accuracy == 0.0

    def test_mismatched_lengths(self):
        """Mismatched lengths should raise ValueError."""
        accepted = [0.8, 0.9]
        rejected = [0.5]
        with pytest.raises(ValueError):
            compute_pairwise_accuracy(accepted, rejected)


class TestComputeBinaryJudgeVotes:
    """Tests for explicit binary vote extraction from judge scores."""

    def test_binary_votes_are_zero_one_vector(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]
        assert compute_binary_judge_votes(accepted, rejected) == [1, 0, 1]

    def test_ties_map_to_zero(self):
        accepted = [0.8, 0.5, 0.3]
        rejected = [0.5, 0.5, 0.7]
        assert compute_binary_judge_votes(accepted, rejected) == [1, 0, 0]

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_binary_judge_votes([0.8, 0.9], [0.5])


class TestComputeJudgeAgreement:
    """Tests for compute_judge_agreement function using Kendall's tau."""

    def test_perfect_agreement(self):
        """All judges have identical score differences."""
        accepted = [0.8, 0.8, 0.8]
        rejected = [0.5, 0.5, 0.5]
        agreement = compute_judge_agreement(accepted, rejected)
        assert agreement == 1.0, "Perfect agreement should be 1.0"

    def test_single_judge(self):
        """Single judge should return 1.0 (no disagreement possible)."""
        accepted = [0.8]
        rejected = [0.5]
        agreement = compute_judge_agreement(accepted, rejected)
        assert agreement == 1.0

    def test_two_judges_same_margin(self):
        """Two judges with same score margin."""
        accepted = [0.8, 0.7]
        rejected = [0.5, 0.4]
        # Both have margin of 0.3
        # Note: Kendall's tau may not be 1.0 for this case due to how it's computed
        # The agreement should be reasonable (not necessarily perfect)
        agreement = compute_judge_agreement(accepted, rejected)
        assert 0.0 <= agreement <= 1.0, "Agreement should be in valid range"

    def test_two_judges_different_margins(self):
        """Two judges with different score margins."""
        accepted = [0.9, 0.6]  # Margins: 0.4, 0.1
        rejected = [0.5, 0.5]
        agreement = compute_judge_agreement(accepted, rejected)
        # Should be between 0 and 1 (partial agreement on ranking)
        assert 0.0 <= agreement <= 1.0

    def test_empty_lists(self):
        """Empty lists should return 1.0 (no judges = perfect agreement by default)."""
        accepted = []
        rejected = []
        # Function returns 1.0 for less than 2 judges (no disagreement possible)
        agreement = compute_judge_agreement(accepted, rejected)
        assert agreement == 1.0

    def test_mismatched_lengths(self):
        """Mismatched lengths should raise ValueError."""
        accepted = [0.8, 0.9]
        rejected = [0.5]
        with pytest.raises(ValueError):
            compute_judge_agreement(accepted, rejected)


class TestComputeFleissKappa:
    """Tests for compute_fleiss_kappa on binary judge votes."""

    def test_unanimous_positive_votes(self):
        accepted = [0.8, 0.9, 0.7]
        rejected = [0.5, 0.6, 0.4]
        assert compute_fleiss_kappa(accepted, rejected) == pytest.approx(1.0)

    def test_split_votes(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]
        assert compute_fleiss_kappa(accepted, rejected) == pytest.approx(-0.5)

    def test_single_judge(self):
        accepted = [0.8]
        rejected = [0.5]
        assert compute_fleiss_kappa(accepted, rejected) == pytest.approx(1.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_fleiss_kappa([0.8, 0.9], [0.5])


class TestResolveMultiJudgeVotes:
    """Tests for explicit winner resolution from multiple judges."""

    def test_majority_accepted(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = resolve_multi_judge_votes(accepted, rejected)

        assert result["binary_votes"] == [1, 0, 1]
        assert result["winner"] == "accepted"
        assert result["accepted_votes"] == 2
        assert result["rejected_votes"] == 1
        assert result["tied_judges"] == 0
        assert result["tie_breaker_used"] is None

    def test_vote_tie_broken_by_mean_score(self):
        accepted = [0.9, 0.3]
        rejected = [0.2, 0.8]

        result = resolve_multi_judge_votes(accepted, rejected, tie_breaker="mean_score")

        assert result["winner"] == "accepted"
        assert result["is_vote_tie"] is True
        assert result["tie_breaker_used"] == "mean_score"

    def test_vote_tie_broken_by_first_judge(self):
        accepted = [0.9, 0.3]
        rejected = [0.2, 0.8]

        result = resolve_multi_judge_votes(accepted, rejected, tie_breaker="first_judge")

        assert result["winner"] == "accepted"
        assert result["is_vote_tie"] is True
        assert result["tie_breaker_used"] == "first_judge"

    def test_invalid_tie_breaker(self):
        with pytest.raises(ValueError):
            resolve_multi_judge_votes([0.8], [0.2], tie_breaker="random")


class TestAggregateMultiJudgeReward:
    """Tests for aggregate_multi_judge_reward function."""

    def test_perfect_accuracy_perfect_agreement(self):
        """All judges rank correctly with identical margins."""
        accepted = [0.8, 0.8, 0.8]
        rejected = [0.5, 0.5, 0.5]
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=0.7, beta=0.3)

        assert result["pairwise_accuracy"] == 1.0
        assert result["agreement"] == pytest.approx(1.0)
        assert result["reward"] == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)
        assert result["num_judges"] == 3
        assert result["mean_accepted_score"] == pytest.approx(0.8)
        assert result["mean_rejected_score"] == pytest.approx(0.5)

    def test_zero_accuracy(self):
        """No judges rank correctly."""
        accepted = [0.3, 0.4, 0.2]
        rejected = [0.5, 0.6, 0.7]
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=0.7, beta=0.3)

        assert result["pairwise_accuracy"] == 0.0
        # Agreement may vary but reward should be low
        assert result["reward"] <= 0.3  # At most beta * agreement

    def test_partial_accuracy(self):
        """Mixed rankings from judges."""
        accepted = [0.8, 0.4, 0.7]  # Judge 2 incorrect
        rejected = [0.5, 0.6, 0.4]
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=0.7, beta=0.3)

        assert result["pairwise_accuracy"] == pytest.approx(2.0 / 3.0)
        assert 0.0 <= result["agreement"] <= 1.0
        assert 0.0 <= result["reward"] <= 1.0

    def test_alpha_beta_normalization(self):
        """Alpha and beta should be normalized to sum to 1.0."""
        accepted = [0.8, 0.9]
        rejected = [0.5, 0.6]

        # Unnormalized weights
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=1.0, beta=1.0)

        # After normalization: alpha=0.5, beta=0.5
        # Reward should be 0.5 * pairwise_accuracy + 0.5 * agreement
        assert result["reward"] == pytest.approx(
            0.5 * result["pairwise_accuracy"] + 0.5 * result["agreement"]
        )

    def test_alpha_only(self):
        """Only pairwise accuracy (beta=0)."""
        accepted = [0.8, 0.9, 0.7]
        rejected = [0.5, 0.6, 0.4]
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=1.0, beta=0.0)

        # After normalization: alpha=1.0, beta=0.0
        assert result["reward"] == pytest.approx(result["pairwise_accuracy"])

    def test_beta_only(self):
        """Only agreement (alpha=0)."""
        accepted = [0.8, 0.9, 0.7]
        rejected = [0.5, 0.6, 0.4]
        result = aggregate_multi_judge_reward(accepted, rejected, alpha=0.0, beta=1.0)

        # After normalization: alpha=0.0, beta=1.0
        assert result["reward"] == pytest.approx(result["agreement"])

    def test_return_structure(self):
        """Result should contain all expected keys."""
        accepted = [0.8, 0.9]
        rejected = [0.5, 0.6]
        result = aggregate_multi_judge_reward(accepted, rejected)

        expected_keys = [
            "reward",
            "pairwise_accuracy",
            "agreement",
            "accepted_scores",
            "rejected_scores",
            "mean_accepted_score",
            "mean_rejected_score",
            "num_judges",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert result["accepted_scores"] == accepted
        assert result["rejected_scores"] == rejected


class TestAggregateMultiJudgeComparison:
    """Tests for configurable multi-judge aggregation modes."""

    def test_majority_vote_reward(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="majority_vote",
        )

        assert result["winner"] == "accepted"
        assert result["reward"] == 1.0
        assert result["aggregation_mode"] == "majority_vote"

    def test_average_vote_reward(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="average_vote",
        )

        assert result["binary_votes"] == [1, 0, 1]
        assert result["average_vote_reward"] == pytest.approx(2.0 / 3.0)
        assert result["reward"] == pytest.approx(2.0 / 3.0)
        assert result["aggregation_mode"] == "average_vote"

    def test_agreement_bonus_reward(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="agreement_bonus",
            alpha=0.7,
            beta=0.3,
        )

        expected = aggregate_multi_judge_reward(accepted, rejected, alpha=0.7, beta=0.3)
        assert result["reward"] == pytest.approx(expected["reward"])
        assert result["agreement_bonus_reward"] == pytest.approx(expected["reward"])

    def test_average_minus_variance_reward(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="average_minus_variance",
        )

        assert result["aggregation_mode"] == "average_minus_variance"
        assert result["binary_votes"] == [1, 0, 1]
        assert result["average_vote_reward"] == pytest.approx(2.0 / 3.0)
        assert result["pairwise_accuracy"] == pytest.approx(2.0 / 3.0)
        assert result["fleiss_kappa"] == pytest.approx(-0.5)
        assert result["clipped_fleiss_kappa"] == pytest.approx(0.0)
        assert result["variance_penalty"] == pytest.approx(1.0)
        assert result["average_minus_variance_reward"] == pytest.approx(-1.0 / 3.0)
        assert result["reward"] == pytest.approx(-1.0 / 3.0)

    def test_margin_kappa_format_reward(self):
        accepted = [0.8, 0.9, 0.7]
        rejected = [0.3, 0.4, 0.2]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="margin_kappa_format",
            rubric_format_score=1.0,
            margin_weight=0.5,
            format_weight=0.3,
            kappa_weight=0.2,
        )

        assert result["aggregation_mode"] == "margin_kappa_format"
        avg_margin = sum(a - r for a, r in zip(accepted, rejected)) / 3
        assert result["avg_margin"] == pytest.approx(avg_margin)
        assert result["rubric_format_score"] == 1.0
        # All judges agree (all binary votes = 1), so kappa = 1.0
        assert result["clipped_fleiss_kappa"] == pytest.approx(1.0)
        expected_reward = 0.5 * avg_margin + 0.3 * 1.0 + 0.2 * 1.0
        assert result["reward"] == pytest.approx(expected_reward)
        assert result["margin_kappa_format_reward"] == pytest.approx(expected_reward)

    def test_margin_kappa_format_no_format(self):
        accepted = [0.8, 0.4, 0.7]
        rejected = [0.5, 0.6, 0.4]

        result = aggregate_multi_judge_comparison(
            accepted,
            rejected,
            aggregation_mode="margin_kappa_format",
            rubric_format_score=0.0,
        )

        avg_margin = sum(a - r for a, r in zip(accepted, rejected)) / 3
        assert result["avg_margin"] == pytest.approx(avg_margin)
        assert result["rubric_format_score"] == 0.0
        expected_reward = 0.5 * avg_margin + 0.3 * 0.0 + 0.2 * result["clipped_fleiss_kappa"]
        assert result["reward"] == pytest.approx(expected_reward)

    def test_invalid_aggregation_mode(self):
        with pytest.raises(ValueError):
            aggregate_multi_judge_comparison([0.8], [0.2], aggregation_mode="median_vote")

    def test_empty_scores_return_zero_reward(self):
        result = aggregate_multi_judge_comparison([], [], aggregation_mode="majority_vote")

        assert result["reward"] == 0.0
        assert result["winner"] is None
        assert result["num_judges"] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_identical_scores(self):
        """Accepted and rejected have identical scores (tie)."""
        accepted = [0.5, 0.5, 0.5]
        rejected = [0.5, 0.5, 0.5]
        result = aggregate_multi_judge_reward(accepted, rejected)

        # All ties, pairwise accuracy should be 0
        assert result["pairwise_accuracy"] == 0.0
        # Agreement should be high (all judges agree it's a tie)
        assert result["agreement"] >= 0.9

    def test_very_small_differences(self):
        """Very small score differences."""
        accepted = [0.5001, 0.5002, 0.5003]
        rejected = [0.5000, 0.5001, 0.5002]
        result = aggregate_multi_judge_reward(accepted, rejected)

        # All judges rank correctly (accepted > rejected by tiny margin)
        assert result["pairwise_accuracy"] == 1.0

    def test_large_score_range(self):
        """Large range of scores."""
        accepted = [0.1, 0.5, 0.9]
        rejected = [0.0, 0.4, 0.8]
        result = aggregate_multi_judge_reward(accepted, rejected)

        # All rank correctly, but different margins
        assert result["pairwise_accuracy"] == 1.0
        assert 0.0 <= result["agreement"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
