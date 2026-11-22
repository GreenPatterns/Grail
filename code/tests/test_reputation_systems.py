"""Unit tests for the non-GNN reputation baselines (mycode/reputation_systems.py).

    pytest tests/test_reputation_systems.py        # ps env; CPU only
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mycode.reputation_systems import fairness_goodness  # noqa: E402


def test_fixed_point_on_a_hand_solved_graph():
    # raters 0 and 1 trust node 3 (+1), rater 2 distrusts it (-1).
    # Fixed point: g3 = (f0 + f1 - f2)/3 with f0 = f1 = 1 - |1 - g3|/2, f2 = 1 - |-1 - g3|/2
    # -> g3 = 1/3, f0 = f1 = 2/3, f2 = 1/3.
    src = np.array([0, 1, 2]); dst = np.array([3, 3, 3]); w = np.array([1.0, 1.0, -1.0])
    f, g = fairness_goodness(src, dst, w, n_nodes=4)
    assert g[3] == pytest.approx(1 / 3, abs=1e-6)
    assert f[0] == pytest.approx(2 / 3, abs=1e-6) and f[1] == pytest.approx(2 / 3, abs=1e-6)
    assert f[2] == pytest.approx(1 / 3, abs=1e-6)


def test_ranges_and_isolated_nodes():
    src = np.array([0, 1, 1]); dst = np.array([1, 2, 0]); w = np.array([0.5, -1.0, 1.0])
    f, g = fairness_goodness(src, dst, w, n_nodes=4)
    assert ((f >= 0) & (f <= 1)).all() and ((g >= -1) & (g <= 1)).all()
    assert g[3] == 0.0 and f[3] == 1.0          # no ratings in or out


def test_a_fair_rater_moves_goodness_more_than_an_unfair_one():
    # node 5 is rated +1 by raters 0-3; add one distrust from a consensus-agreeing rater (4,
    # who also rates node 6 like everyone else) vs. from a rater with only that one rating.
    base_src = [0, 1, 2, 3, 0, 1, 2, 3, 4]; base_dst = [5, 5, 5, 5, 6, 6, 6, 6, 6]
    base_w = [1.0] * 9
    f, g = fairness_goodness(np.array(base_src + [4]), np.array(base_dst + [5]),
                             np.array(base_w + [-1.0]), n_nodes=8)
    f2, g2 = fairness_goodness(np.array(base_src + [7]), np.array(base_dst + [5]),
                               np.array(base_w + [-1.0]), n_nodes=8)
    assert g[5] < g2[5]


def test_warm_start_reaches_the_same_fixed_point():
    rng = np.random.RandomState(0)
    n = 60
    src = rng.randint(0, n, 400); dst = rng.randint(0, n, 400)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = np.where(rng.rand(len(src)) < 0.8, 1.0, -1.0)
    f0, g0 = fairness_goodness(src, dst, w, n)
    # one extra distrust edge, solved cold and warm-started from the clean fixed point
    s2, d2, w2 = np.append(src, 3), np.append(dst, 7), np.append(w, -1.0)
    f_cold, g_cold = fairness_goodness(s2, d2, w2, n)
    f_warm, g_warm = fairness_goodness(s2, d2, w2, n, init=(f0, g0))
    assert np.allclose(g_warm, g_cold, atol=1e-7) and np.allclose(f_warm, f_cold, atol=1e-7)
    # the caller's arrays are not modified
    assert np.array_equal(f0, fairness_goodness(src, dst, w, n)[0])
