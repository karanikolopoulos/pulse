import math

import pytest

from pulse.domain.scoring import elbow_rank, norm_prob_diff, ground_truth_diff


def test_norm_prob_diff_positive_favors_a():
    # P(A) = e^-1 / (e^-1 + e^-2)
    p_a = 1 / (1 + math.exp(-1))
    assert norm_prob_diff([-1.0, -3.0], [-2.0, -3.0]) == pytest.approx([2 * p_a - 1, 0.0])


def test_norm_prob_diff_handles_very_negative_logprobs():
    assert norm_prob_diff([-1000.0], [-1001.0]) == pytest.approx(norm_prob_diff([-1.0], [-2.0]))


def test_ground_truth_diff_same_scale():
    assert ground_truth_diff([0.6, 0.25], [0.4, 0.75]) == pytest.approx([0.2, -0.5])


def test_elbow_rank():
    # probabilities 0.5, 0.3, 0.2 -> cumulative 0.5, 0.8, 1.0
    logprobs = [math.log(0.5), math.log(0.3), math.log(0.2)]
    assert elbow_rank(logprobs, min_p=0.4) == 1
    assert elbow_rank(logprobs, min_p=0.75) == 2
    assert elbow_rank(logprobs, min_p=0.99) == 3
