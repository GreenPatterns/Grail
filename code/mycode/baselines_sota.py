"""
SOTA 2024--2026 attack baselines for the GRAIL utility comparison.

Each ranker returns a ranked ``List[int]`` of source node ids (most-damaging
first), matching the interface of ``ReputationAttackOracle.baseline_*_ranking``
so the experiment runners can score the chosen top-B via
``oracle.score_edge_set_processed(...)`` (the exact-counterfactual delta-R scorer
at the processed placement). Do not score them with ``oracle.score_edge_set``:
it appends past the processed window, where every edge is a silent no-op.

The three baselines correspond to the dominant recent attack families
(see paper Sec. 2 and the project plan):

  * ``rank_prbcd``           -- PR/GR-BCD style *iterative* gradient attack
                               (Geisler et al., NeurIPS'21) adapted to the
                               model's differentiable edge-LABEL relaxation.
  * ``rank_influence_edge_edit`` -- influence-function edge-insertion ranking,
                               in the spirit of "Influence Functions for Edge
                               Edits in Non-Convex GNNs" (arXiv 2506.04694).
  * ``rank_node_injection``  -- fresh-account / node-injection source selection
                               under a stealth (low-degree) constraint, in the
                               spirit of JANUS (2509.13266) / LPGIA (2405.18824).

Design notes (why these adaptations):
  * GDTE's structural convolution (mycode/SL.py ``Convolution.forward``)
    aggregates with plain ``scatter_mean`` and does NOT consume a per-edge
    weight in the non-robust path, so a classic adjacency-weight PRBCD is not
    available without changing message passing. The model's exposed
    differentiable handle is the edge-label tensor (the same one GRAIL-Grad
    uses), so PRBCD here optimises continuous distrust-label *strengths* and
    discretises to a budget-B edge set -- a faithful PRBCD analogue.
  * The model is frozen at inference, so an edge "influence" is the change it
    induces in the forward pass (no retraining); the influence baseline is a
    curvature-corrected first-order approximation of that change, a fairer
    "smart gradient" competitor to single-pass GRAIL-Grad.
  * The frozen model cannot index brand-new node embeddings, so node-injection
    is modelled by constraining sources to low-degree ("fresh-like") accounts,
    consistent with the threat model (Sec. 3: attacker controls existing
    accounts; no zero-history Sybil creation).

NOTE: This module was authored against the verified oracle/model interfaces but
requires the torch/GDTE stack to execute. It MUST be smoke-tested on a
torch-enabled (GPU) machine via tests/test_baselines_sota.py before reporting
numbers; it cannot run on the Python-3.6/no-torch host used for authoring.
"""

from typing import List, Optional

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────

def _eligible_candidates(
    oracle,
    edges: torch.Tensor,
    target_node: int,
    candidate_sources: Optional[List[int]],
    max_candidates: int,
    seed: int = 42,
) -> List[int]:
    """Filter to non-target, non-existing sources and subsample to a fixed pool.

    Mirrors the candidate filtering used by ``rank_new_edges`` /
    ``rank_new_edges_gradient`` so every method ranks the SAME pool.
    """
    if candidate_sources is None:
        candidate_sources = list(range(oracle.num_nodes))

    existing = set()
    dst_row = edges[1]
    src_row = edges[0]
    for i in range(edges.shape[1]):
        existing.add((int(src_row[i].item()), int(dst_row[i].item())))

    candidates = [
        int(s) for s in candidate_sources
        if int(s) != target_node and (int(s), target_node) not in existing
    ]
    if len(candidates) > max_candidates:
        rng = np.random.RandomState(seed)
        candidates = [int(s) for s in rng.choice(candidates, max_candidates, replace=False)]
    return candidates


def _node_total_degree(oracle, edges: torch.Tensor) -> torch.Tensor:
    """Total (in+out) degree per node as a float tensor on the oracle device."""
    deg = torch.zeros(oracle.num_nodes, device=oracle.device)
    ones = torch.ones(edges.shape[1], device=oracle.device)
    deg.scatter_add_(0, edges[0].long(), ones)
    deg.scatter_add_(0, edges[1].long(), ones)
    return deg


def _candidate_snapshot_layout(
    oracle,
    edges: torch.Tensor,
    labels: torch.Tensor,
    candidates: List[int],
    target_node: int,
):
    """Replicate the verified augmentation from ``score_candidates_gradient``.

    Distributes the candidate edges across the T training snapshots
    proportionally to each snapshot's edge count (required for gradient signal
    to propagate through the temporal backbone), and returns the pieces needed
    to build a (re-parameterisable) label tensor.

    Returns
    -------
    aug_edges : LongTensor [2, E+C]
    aug_index : List[int]            updated snapshot boundaries
    base_parts_l : List[FloatTensor] per-snapshot label blocks (original rows
                   followed by candidate rows already set to the distrust
                   one-hot); concatenation order matches ``cand_label_indices``
    cand_label_indices : List[int]   rows in the concatenated label tensor that
                   correspond to candidate edges (in ``candidates`` order)
    """
    C = len(candidates)
    cand_edges = torch.tensor(
        [candidates, [target_node] * C], dtype=torch.long, device=oracle.device
    )

    T = oracle.model.args.train_time_slots
    orig_idx_list = list(oracle.model.index_list)
    edges_per_snap, prev = [], -1
    for idx in orig_idx_list[:T]:
        edges_per_snap.append(idx - prev)
        prev = idx
    total = sum(edges_per_snap)
    props = [max(1, int(C * e / total)) for e in edges_per_snap]
    props[-1] += C - sum(props)

    aug_parts_e, aug_parts_l = [], []
    orig_offset, cand_offset = 0, 0
    distrust_row = torch.tensor([0.0, 1.0], device=oracle.device, dtype=torch.float)
    for t in range(T):
        end = orig_idx_list[t] + 1
        part_e = edges[:, orig_offset:end]
        part_l = labels[orig_offset:end, :]
        n_new = props[t]
        if n_new > 0:
            new_e = cand_edges[:, cand_offset:cand_offset + n_new]
            soft_l = distrust_row.repeat(n_new, 1)
            part_e = torch.cat([part_e, new_e], dim=1)
            part_l = torch.cat([part_l, soft_l], dim=0)
            cand_offset += n_new
        aug_parts_e.append(part_e)
        aug_parts_l.append(part_l)
        orig_offset = end

    aug_edges = torch.cat(aug_parts_e, dim=1)

    cand_label_indices, row_offset = [], 0
    for t in range(T):
        n_orig_actual = aug_parts_l[t].shape[0] - props[t]
        row_offset += n_orig_actual
        if props[t] > 0:
            cand_label_indices.extend(range(row_offset, row_offset + props[t]))
            row_offset += props[t]

    aug_index, cumsum = [], 0
    for part in aug_parts_e:
        cumsum += part.shape[1]
        aug_index.append(cumsum - 1)
    aug_index = aug_index[:T]

    return aug_edges, aug_index, aug_parts_l, cand_label_indices


def _reputation_under_labels(oracle, aug_edges, soft_labels, aug_index, target_node):
    """Forward pass -> differentiable target reputation, with index_list guarded."""
    original_idx = oracle.model.index_list
    try:
        oracle.model.index_list = aug_index
        mu = oracle.model._process_structural_layer(aug_edges, soft_labels)
        temporal_all = oracle.model.s3r(mu)
        node_emb = temporal_all[:, oracle.model.args.train_time_slots - 1, :].squeeze()
        rep = oracle._reputation_score(node_emb, target_node, aug_edges)
    finally:
        oracle.model.index_list = original_idx
    return rep


# ────────────────────────────────────────────────────────────────
# 1. PRBCD / GRBCD  (iterative gradient attack)
# ────────────────────────────────────────────────────────────────

def rank_prbcd(
    oracle,
    edges: torch.Tensor,
    labels: torch.Tensor,
    target_node: int,
    candidate_sources: Optional[List[int]] = None,
    *,
    sign: str = 'distrust',
    budget: int = 5,
    n_iter: int = 30,
    block_frac: float = 0.5,
    step: float = 0.25,
    minimize: bool = True,
    seed: int = 42,
    max_candidates: int = 100,
    top_k: int = 10,
) -> List[int]:
    """PR/GR-BCD-style iterative gradient attack over edge-label strengths.

    Maintains a continuous distrust strength ``p_s in [0, 1]`` per candidate
    (label row ``[1-p, p]`` for distrust), runs ``n_iter`` projected gradient
    steps that MINIMISE the target reputation, updating a random block of
    coordinates each step (R-BCD), and ranks candidates by final strength.
    The budget enters as the L0 selection (top-``budget``) performed by the
    caller via ``score_edge_set``; we additionally bias the projection toward a
    sparse solution by renormalising the active mass toward ``budget``.

    Returns the full ranked candidate list (top_k) of source ids.
    """
    if sign != 'distrust':
        raise NotImplementedError("rank_prbcd currently supports sign='distrust'.")

    candidates = _eligible_candidates(
        oracle, edges, target_node, candidate_sources, max_candidates, seed
    )
    C = len(candidates)
    if C == 0:
        return []

    aug_edges, aug_index, aug_parts_l, cand_label_indices = _candidate_snapshot_layout(
        oracle, edges, labels, candidates, target_node
    )
    base_labels = torch.cat(aug_parts_l, dim=0).clone().detach().float()
    cand_idx = torch.tensor(cand_label_indices, device=oracle.device, dtype=torch.long)

    rng = np.random.RandomState(seed)
    # initialise strengths near the budget mass spread over the pool
    p = torch.full((C,), min(1.0, budget / max(C, 1)), device=oracle.device)

    for _ in range(n_iter):
        p_var = p.clone().detach().requires_grad_(True)
        # distrust label rows parameterised by p:  [1-p, p]
        cand_rows = torch.stack([1.0 - p_var, p_var], dim=1)
        soft_labels = base_labels.index_copy(0, cand_idx, cand_rows)

        rep = _reputation_under_labels(oracle, aug_edges, soft_labels, aug_index, target_node)
        grad = torch.autograd.grad(rep, p_var)[0]

        # randomized block-coordinate mask
        mask = torch.from_numpy(
            (rng.rand(C) < block_frac).astype(np.float32)
        ).to(oracle.device)

        # projected gradient step; minimize => descend reputation (more damage).
        # Exposed as a flag so the descent direction can be validated (diag_prbcd_sign.py).
        sign_dir = -1.0 if minimize else 1.0
        p = (p + sign_dir * step * grad * mask).clamp_(0.0, 1.0)

        # soft budget projection: if total mass exceeds the budget, rescale
        total_mass = p.sum()
        if float(total_mass.item()) > budget:
            p = p * (budget / total_mass)

    order = torch.argsort(p, descending=True)
    return [candidates[int(i)] for i in order[:top_k]]


# ────────────────────────────────────────────────────────────────
# 2. Influence-function edge-edit ranking
# ────────────────────────────────────────────────────────────────

def rank_influence_edge_edit(
    oracle,
    edges: torch.Tensor,
    labels: torch.Tensor,
    target_node: int,
    candidate_sources: Optional[List[int]] = None,
    *,
    sign: str = 'distrust',
    damping: float = 1e-2,
    seed: int = 42,
    max_candidates: int = 100,
    top_k: int = 10,
) -> List[int]:
    """Influence-function edge-insertion ranking (arXiv 2506.04694 spirit).

    A curvature-corrected first-order estimate of each candidate's effect on
    the target reputation: a single batched backward pass gives the per-edge
    first-order influence g_s = dR/d(distrust-strength) (as in
    ``score_candidates_gradient``); we then apply a damped Newton rescaling
    g_s / (h_s + damping), where h_s is a Gauss-Newton curvature proxy from the
    second derivative of R along each candidate's strength. This is a fairer
    "smart gradient" competitor to single-pass GRAIL-Grad without the full
    per-candidate counterfactual cost of GRAIL-Fwd.
    """
    if sign != 'distrust':
        raise NotImplementedError("rank_influence_edge_edit supports sign='distrust'.")

    candidates = _eligible_candidates(
        oracle, edges, target_node, candidate_sources, max_candidates, seed
    )
    C = len(candidates)
    if C == 0:
        return []

    aug_edges, aug_index, aug_parts_l, cand_label_indices = _candidate_snapshot_layout(
        oracle, edges, labels, candidates, target_node
    )
    base_labels = torch.cat(aug_parts_l, dim=0).clone().detach().float()
    cand_idx = torch.tensor(cand_label_indices, device=oracle.device, dtype=torch.long)

    # evaluate at a small insertion strength so the Taylor expansion is local
    eps0 = 0.5
    p = torch.full((C,), eps0, device=oracle.device, requires_grad=True)
    cand_rows = torch.stack([1.0 - p, p], dim=1)
    soft_labels = base_labels.index_copy(0, cand_idx, cand_rows)

    rep = _reputation_under_labels(oracle, aug_edges, soft_labels, aug_index, target_node)
    g = torch.autograd.grad(rep, p, create_graph=True)[0]          # first order dR/dp_s
    # diagonal curvature proxy h_s = d^2R/dp_s^2 via grad of sum(g)
    h = torch.autograd.grad(g.sum(), p, retain_graph=False)[0].detach()
    g = g.detach()

    # damped Newton influence; lower (more negative) => more reputation damage
    influence = g / (h.abs() + damping)
    order = torch.argsort(influence, descending=False)             # most negative first
    return [candidates[int(i)] for i in order[:top_k]]


# ────────────────────────────────────────────────────────────────
# 3. Node-injection (fresh-account) ranking under a stealth constraint
# ────────────────────────────────────────────────────────────────

def rank_node_injection(
    oracle,
    edges: torch.Tensor,
    labels: torch.Tensor,
    target_node: int,
    candidate_sources: Optional[List[int]] = None,
    *,
    sign: str = 'distrust',
    degree_percentile: float = 25.0,
    min_pool: int = 10,
    seed: int = 42,
    max_candidates: int = 100,
    top_k: int = 10,
) -> List[int]:
    """Node-injection source selection (JANUS/LPGIA spirit) under stealth.

    Models injected fresh accounts by restricting sources to low-degree
    ("fresh-like") nodes -- the stealth/imperceptibility constraint -- then
    ranks that constrained pool by the exact counterfactual reputation shift at
    the processed placement (``oracle.score_candidates_counterfactual_processed``).
    An earlier version reused ``oracle.rank_new_edges``, which appends past the
    processed window, where the label never reaches the encoder, so its ranking
    was label-blind. Returns ranked source ids.
    """
    candidates = _eligible_candidates(
        oracle, edges, target_node, candidate_sources, max_candidates, seed
    )
    if not candidates:
        return []

    deg = _node_total_degree(oracle, edges)
    cand_deg = np.array([float(deg[s].item()) for s in candidates])
    thresh = np.percentile(cand_deg, degree_percentile)
    fresh = [s for s, d in zip(candidates, cand_deg) if d <= thresh]

    # relax the stealth constraint if the pool is too small to be meaningful
    if len(fresh) < min_pool:
        order = np.argsort(cand_deg)
        fresh = [candidates[int(i)] for i in order[:max(min_pool, top_k)]]

    cf = oracle.score_candidates_counterfactual_processed(
        edges, labels, target_node, fresh, sign=sign,
    )
    # distrust: most negative shift first; trust: most positive first
    ranked = sorted(cf.items(), key=lambda x: x[1] if sign == 'distrust' else -x[1])
    return [int(src) for src, _shift in ranked[:top_k]]


# Registry so runners can iterate uniformly alongside oracle.baseline_* methods.
SOTA_RANKERS = {
    "PRBCD": rank_prbcd,
    "InfluenceEdit": rank_influence_edge_edit,
    "NodeInjection": rank_node_injection,
}
