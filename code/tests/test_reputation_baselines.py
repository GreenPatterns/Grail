"""Unit tests for the paired reputation-baseline runner's pure helpers
(experiments/run_reputation_baselines.py).

    pytest tests/test_reputation_baselines.py        # ps env; CPU only
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

from run_reputation_baselines import (  # noqa: E402
    FUNCS, _attacked_reps, _clean_reps, _fg_goodness_after)
from mycode.reputation_systems import fairness_goodness  # noqa: E402


def _toy_graph():
    # raters 0-3 trust node 5, and each also trusts node 6; node 4 has rated nothing
    src = np.array([0, 1, 2, 3, 0, 1, 2, 3]); dst = np.array([5, 5, 5, 5, 6, 6, 6, 6])
    w = np.ones(8)
    return src, dst, w, 8


def test_fg_injection_lowers_only_through_the_new_distrust_edges():
    src, dst, w, n = _toy_graph()
    _, g0 = fairness_goodness(src, dst, w, n)
    g1 = _fg_goodness_after(src, dst, w, n, target=5, srcs=[4, 7])
    assert g0[5] == pytest.approx(1.0)
    assert g1[5] < g0[5]
    # the inputs are not mutated
    assert len(src) == 8 and w.min() == 1.0


def test_clean_and_attacked_reps_cover_every_function():
    src, dst, w, n = _toy_graph()
    _, g0 = fairness_goodness(src, dst, w, n)
    p0 = torch.tensor([0.9, 0.8, 0.7, 0.6])
    r0 = _clean_reps(p0, g0, t=5, k_in=4, n_in=4)
    assert set(r0) == set(FUNCS)
    assert r0["gdte_mean_total"] == pytest.approx(0.75)
    assert r0["gdte_median_prop"] == pytest.approx(0.75)
    assert r0["fraction"] == 1.0 and r0["beta"] == pytest.approx(5 / 6)

    g1 = _fg_goodness_after(src, dst, w, n, target=5, srcs=[4])
    p_full = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1])
    p_pre = torch.tensor([0.5, 0.5, 0.5, 0.5])
    r1 = _attacked_reps(p_full, p_pre, g1, t=5, k_in=4, n_in=4, B=1)
    assert set(r1) == set(FUNCS)
    assert r1["gdte_mean_total"] == pytest.approx(0.42)
    assert r1["gdte_mean_prop"] == pytest.approx(0.5)
    assert r1["gdte_median_total"] == pytest.approx(0.5)
    assert r1["fraction"] == pytest.approx(4 / 5)       # one distrust among five ratings
    assert r1["beta"] == pytest.approx(5 / 7)


def test_prop_is_nan_without_pre_existing_raters():
    src, dst, w, n = _toy_graph()
    _, g0 = fairness_goodness(src, dst, w, n)
    r1 = _attacked_reps(torch.tensor([0.2]), None, g0, t=4, k_in=0, n_in=0, B=1)
    assert r1["gdte_mean_prop"] != r1["gdte_mean_prop"]
