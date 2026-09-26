"""The fast last-snapshot scorer (mycode/fast_s6.py) must reproduce the full forward pass.

An edge spliced at the end of the last processed snapshot s_{T-1} changes only that
snapshot's structural embedding (the structural layer is stateless per snapshot), so the
cached s_0..s_{T-2} embeddings can be reused. This test compares the fast embeddings and
reputations with the oracle's full _compute_node_embeddings on real injections.

    CUDA_VISIBLE_DEVICES=<free> GRAIL_DATA_ROOT=... pytest tests/test_fast_s6.py
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

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from experiment_common import train_model  # noqa: E402
from mycode.trust_influence import ReputationAttackOracle  # noqa: E402
from mycode.fast_s6 import S6Scorer  # noqa: E402
from run_loo_flip_table import _variants  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    tr = train_model("alpha", seed=42, epochs=3)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    return o, tr["edges"], tr["labels"]


def test_fast_reputations_match_the_full_forward_pass(setup):
    o, edges, labels = setup
    fast = S6Scorer(o, edges, labels)
    rng = np.random.RandomState(0)
    indeg = np.bincount(edges[1].cpu().numpy(), minlength=o.num_nodes)
    targets = [int(t) for t in rng.choice(np.nonzero(indeg >= 2)[0], 6, replace=False)]
    for t in targets:
        srcs = [int(s) for s in rng.choice(o.num_nodes, 2, replace=False) if s != t][:2]
        r0_full, rf_full, rp_full = _variants(o, edges, labels, t, srcs)
        r0, rf, rp = fast.clean(t), *fast.attacked(t, srcs)
        assert abs(r0 - r0_full) < 1e-5 and abs(rf - rf_full) < 1e-5 and abs(rp - rp_full) < 1e-5, \
            (t, r0 - r0_full, rf - rf_full, rp - rp_full)


def test_best_single_edge_matches_the_oracle_counterfactual(setup):
    o, edges, labels = setup
    fast = S6Scorer(o, edges, labels)
    t = int(np.nonzero(np.bincount(edges[1].cpu().numpy(), minlength=o.num_nodes) >= 5)[0][3])
    pool = [s for s in range(40) if s != t]
    cf = o.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
    r0 = fast.clean(t)
    for s in pool[:10]:
        assert abs((fast.attacked(t, [s])[0] - r0) - cf[s]) < 1e-5, s


def test_rater_vectors_match_the_full_forward_pass(setup):
    """attacked_vectors returns (pre-existing raters' P(trust), injected edges' P(trust));
    their means are the propagation-only and total-effect reputations."""
    o, edges, labels = setup
    fast = S6Scorer(o, edges, labels)
    rng = np.random.RandomState(1)
    indeg = np.bincount(edges[1].cpu().numpy(), minlength=o.num_nodes)
    for t in [int(x) for x in rng.choice(np.nonzero(indeg >= 3)[0], 4, replace=False)]:
        srcs = [int(s) for s in rng.choice(o.num_nodes, 3, replace=False) if s != t][:3]
        p_pre, p_inj = fast.attacked_vectors(t, srcs)
        rf, rp = fast.attacked(t, srcs)
        assert len(p_pre) == indeg[t] and len(p_inj) == len(srcs)
        assert abs(p_pre.mean().item() - rp) < 1e-6
        assert abs(torch.cat([p_pre, p_inj]).mean().item() - rf) < 1e-6
        c = fast.clean_vector(t)
        assert len(c) == indeg[t] and abs(c.mean().item() - fast.clean(t)) < 1e-6
