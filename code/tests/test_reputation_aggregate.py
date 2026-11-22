"""Unit tests for the reputation aggregation used by the robust-functional baseline.

    pytest tests/test_reputation_aggregate.py        # ps env; CPU only
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mycode.reputation import aggregate_reputation  # noqa: E402

P = torch.tensor([0.9, 0.8, 0.1, 0.7, 0.6])


def test_mean_is_the_arithmetic_mean():
    assert aggregate_reputation(P, "mean").item() == pytest.approx(0.62)


def test_median_ignores_a_single_outlying_rater():
    assert aggregate_reputation(P, "median").item() == pytest.approx(0.7)
    assert aggregate_reputation(torch.tensor([0.8, 0.8, 0.0]), "median").item() == pytest.approx(0.8)


def test_median_of_even_count_averages_the_middle_pair():
    assert aggregate_reputation(torch.tensor([0.2, 0.4, 0.6, 0.8]), "median").item() == pytest.approx(0.5)


def test_trimmed_mean_drops_the_extremes():
    x = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    assert aggregate_reputation(x, "trimmed", trim=0.1).item() == pytest.approx(0.5)


def test_trimmed_mean_keeps_small_sets_intact():
    x = torch.tensor([0.2, 0.9])
    assert aggregate_reputation(x, "trimmed", trim=0.1).item() == pytest.approx(0.55)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        aggregate_reputation(P, "bogus")
