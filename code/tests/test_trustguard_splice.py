"""Unit tests for the edge-splicing helper of the TrustGuard placement harness
(experiments/run_trustguard_placement.py). TrustGuard's forward pass reads snapshot i as
edges[index_list[i-1]:index_list[i]] (index_list[-1] := 0) for i < T only.

    pytest tests/test_trustguard_splice.py        # ps env; CPU only
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

from run_trustguard_placement import splice_edge  # noqa: E402

EDGES = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 6, 7, 7, 8, 8]])
LABELS = torch.tensor([[1., 0.]] * 6)
INDEX = [2, 4, 6, 9, 12]           # snapshots 0..2 processed (T = 3); 3..4 are past the window
T = 3


def _slice(index, i):
    return (0 if i == 0 else index[i - 1]), index[i]


def test_processed_placement_lands_inside_that_snapshot():
    for pos in range(T):
        e, l, idx = splice_edge(EDGES, LABELS, INDEX, T, pos, src=9, dst=6, label=[0., 1.])
        lo, hi = _slice(idx, pos)
        k = [j for j in range(e.shape[1]) if e[0, j] == 9][0]
        assert lo <= k < hi, (pos, k, lo, hi)
        assert l[k].tolist() == [0., 1.]
        assert e.shape[1] == 7 and idx[T - 1] == INDEX[T - 1] + 1


def test_appended_placement_is_past_the_processed_window():
    e, l, idx = splice_edge(EDGES, LABELS, INDEX, T, 'appended', src=9, dst=6, label=[0., 1.])
    k = [j for j in range(e.shape[1]) if e[0, j] == 9][0]
    assert k >= idx[T - 1]                   # never read by the encoder
    assert idx[:T] == INDEX[:T]              # processed boundaries unchanged


def test_inputs_are_not_mutated():
    before = (EDGES.clone(), LABELS.clone(), list(INDEX))
    splice_edge(EDGES, LABELS, INDEX, T, 1, src=9, dst=6, label=[0., 1.])
    assert torch.equal(EDGES, before[0]) and torch.equal(LABELS, before[1]) and INDEX == before[2]
