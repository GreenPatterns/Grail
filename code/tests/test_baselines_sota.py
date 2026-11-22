"""Interface smoke test for the SOTA 2024--26 attack baselines.

Validates that each new ranker in ``mycode/baselines_sota.py`` returns a clean
ranking over the candidate pool and integrates with the shared processed-placement
delta-R scorer (``oracle.score_edge_set_processed``). It does NOT assert specific
numbers -- only the interface contract used by the experiment runners, plus the
label-sensitivity check that a ranker actually scores at the processed placement.

Runs on a torch-enabled (GPU) machine. On a host without torch (e.g. the
Python-3.6 authoring box) it skips cleanly:

    python tests/test_baselines_sota.py        # standalone
    pytest tests/test_baselines_sota.py         # under pytest
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)                       # GRAIL/code
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "common"))
os.environ.setdefault(
    "GRAIL_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(CODE_DIR)), "data", "cyberdata"),
)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _first_target(strata, edges):
    for targets in strata.values():
        for t in targets:
            return int(t)
    # fallback: any destination node that has an incoming edge
    return int(edges[1][0].item())


def _run():
    from experiment_common import train_model, get_targets
    from mycode.trust_influence import ReputationAttackOracle
    from mycode import baselines_sota

    trained = train_model("alpha", epochs=3)          # quick smoke, not full training
    oracle = ReputationAttackOracle(
        trained["model"], trained["index_list"], trained["device"]
    )
    edges, labels = trained["edges"], trained["labels"]
    strata = get_targets(oracle, edges, labels, per_stratum=2)
    target = _first_target(strata, edges)
    pool = list(range(oracle.num_nodes))

    for name, fn in baselines_sota.SOTA_RANKERS.items():
        ranking = fn(
            oracle, edges, labels, target,
            candidate_sources=pool, top_k=5, max_candidates=40,
        )
        assert isinstance(ranking, list), f"{name}: ranking is not a list"
        assert all(isinstance(s, int) for s in ranking), f"{name}: non-int source ids"
        assert len(ranking) == len(set(ranking)), f"{name}: duplicate source ids"
        assert target not in ranking, f"{name}: target leaked into ranking"
        dr = oracle.score_edge_set_processed(edges, labels, ranking[:5], target, "distrust")
        assert dr == dr, f"{name}: delta-R is NaN"     # NaN != NaN
        print(f"  {name:14s} top5={ranking[:5]}  deltaR={dr:+.5f}  OK")

    print("ALL BASELINE INTERFACE CHECKS PASSED")


def _run_node_injection_uses_processed_placement():
    from experiment_common import train_model, get_targets
    from mycode.trust_influence import ReputationAttackOracle
    from mycode import baselines_sota

    trained = train_model("alpha", epochs=3)
    oracle = ReputationAttackOracle(
        trained["model"], trained["index_list"], trained["device"]
    )
    edges, labels = trained["edges"], trained["labels"]
    target = _first_target(get_targets(oracle, edges, labels, per_stratum=2), edges)
    pool = list(range(oracle.num_nodes))

    # rank_new_edges appends past the processed window, where the label never
    # reaches the encoder (Proposition 1); its scores are label-blind.
    def _appended_scorer_forbidden(*args, **kwargs):
        raise AssertionError(
            "rank_node_injection used rank_new_edges (appended placement)")
    oracle.rank_new_edges = _appended_scorer_forbidden

    seen = {}
    processed = oracle.score_candidates_counterfactual_processed

    def _spy(edges_, labels_, target_, candidates_, sign="distrust"):
        seen["scores"] = processed(edges_, labels_, target_, candidates_, sign=sign)
        return seen["scores"]
    oracle.score_candidates_counterfactual_processed = _spy

    ranking = baselines_sota.rank_node_injection(
        oracle, edges, labels, target, candidate_sources=pool,
        top_k=10_000, max_candidates=40)
    assert "scores" in seen, "processed-placement counterfactual was never called"
    expected = [s for s, _ in sorted(seen["scores"].items(), key=lambda x: x[1])]
    assert ranking == expected, "ranking is not the processed counterfactual order"


def test_baselines_interface():
    if not HAVE_TORCH:
        try:
            import pytest
            pytest.skip("torch unavailable on this host")
        except Exception:
            return
    _run()


def test_node_injection_ranks_by_processed_counterfactual():
    if not HAVE_TORCH:
        import pytest
        pytest.skip("torch unavailable on this host")
    _run_node_injection_uses_processed_placement()


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("SKIP: torch unavailable on this host (run on the GPU box).")
        sys.exit(0)
    _run()
