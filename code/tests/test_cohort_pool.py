"""The cohort runner's candidate pool must be fixed per target and exclude raters and self.

    pytest tests/test_cohort_pool.py        # ps env; CPU only
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

from run_cohort_seeds import candidate_pool  # noqa: E402


def test_pool_is_deterministic_per_target_and_excludes_raters_and_self():
    raters = [3, 4, 5]
    p1 = candidate_pool(7, raters, 200, C=50)
    p2 = candidate_pool(7, raters, 200, C=50)
    assert p1 == p2 and len(p1) == 50 and len(set(p1)) == 50
    assert not (set(p1) & {3, 4, 5, 7})


def test_pools_differ_across_targets():
    assert candidate_pool(7, [], 200) != candidate_pool(8, [], 200)
