"""Regression test for the seed bug: train_mainz_snapshot(seed=s) must train a model from
seed s. Before the fix, args.seed stayed at the args.py default 42 and the DGTEN / SL
constructors reseeded every RNG from it, so every "seed" built the same model.

    CUDA_VISIBLE_DEVICES=<free> GRAIL_DATA_ROOT=... pytest tests/test_seed_propagation.py
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

if not torch.cuda.is_available():
    pytest.skip("GCNTrainer needs a GPU", allow_module_level=True)

from mycode.mainz_protocol import build_mainz_args, train_mainz_snapshot  # noqa: E402
from mycode.utils import read_graph  # noqa: E402


def _args():
    args = build_mainz_args("alpha", pred_snap="single", attack=False, atype="bad_mouthing",
                            victim_percentage=0.0, homogeneous_edges=False, lambda_temp=0.01,
                            epochs=1)
    args.train_time_slots = 7
    return args


def _checksum(model):
    return sum(float(p.detach().double().abs().sum()) for p in model.parameters())


@pytest.fixture(scope="module")
def runs():
    args = _args()
    graph = read_graph(args)
    out = {}
    for tag, seed in (("a", 1), ("b", 2), ("c", 1)):
        trainer, _ = train_mainz_snapshot(args, graph, seed=seed, startmsg=f"seedtest-{tag}")
        out[tag] = (trainer.X.detach().cpu().clone(), _checksum(trainer.model), trainer.args.seed)
    out["caller_seed"] = args.seed
    return out


def test_the_trainer_runs_with_the_requested_seed(runs):
    assert runs["a"][2] == 1 and runs["b"][2] == 2


def test_different_seeds_give_different_features_and_weights(runs):
    assert not torch.allclose(runs["a"][0], runs["b"][0])
    assert abs(runs["a"][1] - runs["b"][1]) > 1e-3


def test_same_seed_gives_identical_features(runs):
    assert torch.equal(runs["a"][0], runs["c"][0])


def test_caller_args_are_not_mutated(runs):
    assert runs["caller_seed"] == 42
