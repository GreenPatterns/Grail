"""Unit tests for attacker source selection in run_p3_defense.py.

'high_history' is the adaptive attacker against rater-history weighting: it rates
only from accounts the rule weights fully (out-degree >= h0). 'random' and
'low_history' must keep their original behavior so earlier results reproduce.

    pytest tests/test_p3_defense_sources.py        # ps env (module imports torch)
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from run_p3_defense import pick_sources, H0, SEED  # noqa: E402

OUTDEG = np.array([0, 1, 7, 5, 9, 2, 12, 0])
CANDS = list(range(8))


def test_high_history_only_picks_fully_weighted_accounts():
    srcs = pick_sources("high_history", CANDS, OUTDEG, v=3, B=3)
    assert len(srcs) == 3
    assert all(OUTDEG[s] >= H0 for s in srcs)


def test_low_history_picks_lowest_outdegree():
    assert pick_sources("low_history", CANDS, OUTDEG, v=3, B=2) == [0, 7]


def test_random_matches_original_selection():
    expected = [int(s) for s in np.random.RandomState(SEED + 3).permutation(CANDS)[:4]]
    assert pick_sources("random", CANDS, OUTDEG, v=3, B=4) == expected


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        pick_sources("bogus", CANDS, OUTDEG, v=3, B=1)
