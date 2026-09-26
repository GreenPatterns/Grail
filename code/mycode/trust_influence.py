"""
Reputation Attack Oracle
========================
Core module for "From Trust Prediction to Trust Manipulation."

Uses a frozen, trained DGTEN as a differentiable oracle to compute
how any interaction in a trust network influences any node's reputation.

The key method: compute ∂reputation(target)/∂edge_label in a SINGLE
backward pass — producing a full influence map over the entire graph
without re-running the model per edge.

Classes:
    ReputationAttackOracle  — gradient-based influence computation
    CounterfactualSimulator — multi-step what-if planning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from mycode.reputation import aggregate_reputation


# ────────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────────

@dataclass
class InfluenceMap:
    """Result of a single-backward-pass influence computation."""
    target_node: int
    edge_influences: torch.Tensor    # [E, 2] gradient per edge label
    influence_magnitudes: torch.Tensor  # [E] scalar influence per edge
    top_k_indices: torch.Tensor      # indices of most influential edges
    top_k_scores: torch.Tensor       # influence scores for top-k
    reputation_score: float          # current reputation of target


@dataclass
class AttackMetrics:
    """Normalized metrics for evaluating a reputation attack."""
    delta_r: float           # Raw reputation shift (R_after - R_before)
    delta_r_normalized: float  # Normalized drop (ΔR / R_before)
    flipped: bool            # Did reputation cross 0.5 threshold?
    rank_before: int         # Percentile rank before attack (1 = top)
    rank_after: int          # Percentile rank after attack
    rank_shift: int          # rank_after - rank_before
    efficiency: float        # |ΔR| / budget
    r_before: float          # Reputation before
    r_after: float           # Reputation after


@dataclass
class AttackPlan:
    """A planned set of fake interactions to manipulate reputation."""
    target_node: int
    interventions: List[Tuple[int, int, str]]  # (src, dst, sign)
    predicted_shift: float
    actual_shift: float = 0.0
    steps: List[Dict] = field(default_factory=list)
    metrics: Optional[AttackMetrics] = None


# ────────────────────────────────────────────────────────────────
# Reputation Attack Oracle
# ────────────────────────────────────────────────────────────────

class ReputationAttackOracle:
    """
    Gradient-based influence oracle for reputation manipulation.

    Given a frozen GDTE model (implemented as DGTEN class), computes how
    every edge in the network influences a target node's reputation ---
    in a SINGLE backward pass.

    The model's differentiability is used to measure local influence
    patterns through gradient-based influence analysis.
    """

    def __init__(self, trained_model, index_list, device,
                 rep_mode: str = 'mean', decay_lambda: float = 0.8,
                 rater_h0: float = 5.0):
        self.model = trained_model
        self.model.eval()
        self.index_list = index_list
        self.device = device
        self.num_nodes = self.model.nodes
        self.embed_dim = self.model.input_dim
        self.num_labels = self.model.num_labels
        # Reputation functional over incoming raters of v:
        #   'mean'          = unweighted incoming mean (Eq. 1);
        #   'decayed'       = recency-weighted mean, weight decay_lambda**age
        #                     (age = snapshots-since-most-recent); lambda=1.0
        #                     reduces exactly to 'mean';
        #   'rater_weighted'= history-weighted mean, weight = min(1, hist(u)/h0)
        #                     where hist(u) is rater u's out-degree (number of
        #                     ratings issued). Low-history / fresh accounts are
        #                     down-weighted -- the defense recommended in S7.
        #   'median'/'trimmed' = robust aggregation of the raters' predicted trust
        #                     (mycode/reputation.py), the robust-functional baseline.
        assert rep_mode in ('mean', 'decayed', 'rater_weighted', 'median', 'trimmed')
        self.rep_mode = rep_mode
        self.decay_lambda = float(decay_lambda)
        self.rater_h0 = float(rater_h0)

        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ── Reputation Scoring ──────────────────────────────────────

    def _compute_node_embeddings(self, edges, labels):
        """
        Run the frozen model's forward pass and return the
        final temporal node embeddings.

        We bypass the loss computation — we only need embeddings.
        """
        mu = self.model._process_structural_layer(edges, labels)
        temporal_all = self.model.s3r(mu)
        node_emb = temporal_all[:, self.model.args.train_time_slots - 1, :]
        return node_emb.squeeze()

    def _reputation_score(self, node_emb, target_node, edges):
        """
        Compute the reputation of target_node: the average trust
        probability assigned to it across all edges where it appears
        as a destination.

        reputation(v) = mean_over_{u: (u,v) ∈ E} P(trust | u, v)

        This is differentiable w.r.t. node_emb.
        """
        dst_mask = (edges[1] == target_node)

        if dst_mask.sum() == 0:
            # Fallback: use the node's own embedding projected
            # through the destination half of regression weights
            dst_weights = self.model.regression_weights[self.embed_dim:, :]
            score = node_emb[target_node] @ dst_weights
            return F.softmax(score.unsqueeze(0), dim=1)[0, 0]

        # Get all source nodes that rate this target
        src_indices = edges[0, dst_mask]
        src_emb = node_emb[src_indices]
        dst_emb = node_emb[target_node].unsqueeze(0).expand_as(src_emb)
        features = torch.cat((src_emb, dst_emb), dim=1)
        logits = features @ self.model.regression_weights
        trust_probs = F.softmax(logits, dim=1)
        p_trust = trust_probs[:, 0]

        if self.rep_mode == 'mean':
            # Reputation = unweighted mean trust probability (Eq. 1)
            return p_trust.mean()

        if self.rep_mode in ('median', 'trimmed'):
            return aggregate_reputation(p_trust, self.rep_mode)

        if self.rep_mode == 'rater_weighted':
            # History-weighted mean: weight each rater u by min(1, hist(u)/h0)
            # where hist(u) = u's out-degree (ratings issued) in the current
            # graph. Fresh / low-history accounts (the naive Sybil attacker)
            # are down-weighted. Weights are constants w.r.t. node_emb.
            outdeg = torch.bincount(edges[0], minlength=self.num_nodes).to(p_trust.dtype)
            hist = outdeg[src_indices]
            weights = (hist / self.rater_h0).clamp(max=1.0)
            return (weights * p_trust).sum() / weights.sum().clamp(min=1e-12)

        # 'decayed': recency-weighted mean over incoming raters. Each rater
        # edge's snapshot age is derived from the (current) boundary list;
        # weight = decay_lambda**age, so older ratings count less. Weights are
        # constants w.r.t. node_emb, so the score stays differentiable.
        cols = dst_mask.nonzero(as_tuple=True)[0]
        boundaries = torch.as_tensor(self.model.index_list, device=cols.device)
        snap = torch.searchsorted(boundaries, cols)
        T = self.model.args.train_time_slots
        age = (T - 1 - snap).clamp(min=0).to(p_trust.dtype)
        weights = self.decay_lambda ** age
        return (weights * p_trust).sum() / weights.sum().clamp(min=1e-12)

    def compute_reputation(self, edges, labels, target_node):
        """Public method: get a node's current reputation score."""
        with torch.no_grad():
            node_emb = self._compute_node_embeddings(edges, labels)
            rep = self._reputation_score(node_emb, target_node, edges)
        return rep.item()

    def compute_all_reputations(self, edges, labels):
        """Compute reputation scores for all nodes."""
        with torch.no_grad():
            node_emb = self._compute_node_embeddings(edges, labels)
            scores = torch.zeros(self.num_nodes, device=self.device)
            for v in range(self.num_nodes):
                scores[v] = self._reputation_score(node_emb, v, edges)
        return scores

    # ── Normalized Attack Metrics ───────────────────────────────

    def compute_attack_metrics(
        self,
        edges_before: torch.Tensor,
        labels_before: torch.Tensor,
        edges_after: torch.Tensor,
        labels_after: torch.Tensor,
        target_node: int,
        budget: int,
        all_reputations_before: Optional[torch.Tensor] = None,
    ) -> AttackMetrics:
        """
        Compute the full suite of normalized metrics for an attack.

        Args:
            edges_before: graph before attack
            labels_before: labels before attack
            edges_after: graph after injecting fake edges
            labels_after: labels after injection
            target_node: attacked node
            budget: number of fake edges injected
            all_reputations_before: precomputed reputations (saves time)
        """
        # Compute reputations before and after
        r_before = self.compute_reputation(
            edges_before, labels_before, target_node
        )
        r_after = self.compute_reputation(
            edges_after, labels_after, target_node
        )
        delta_r = r_after - r_before
        delta_r_n = delta_r / max(r_before, 1e-8)

        # Flipping: did it cross the 0.5 threshold?
        flipped = (r_before >= 0.5 and r_after < 0.5) or \
                  (r_before < 0.5 and r_after >= 0.5)

        # Rank shift: compute percentile rank
        if all_reputations_before is None:
            all_reputations_before = self.compute_all_reputations(
                edges_before, labels_before
            )
        all_reputations_after = self.compute_all_reputations(
            edges_after, labels_after
        )

        # Rank 1 = highest reputation
        sorted_before = all_reputations_before.argsort(descending=True)
        sorted_after = all_reputations_after.argsort(descending=True)
        rank_before = (sorted_before == target_node).nonzero(
            as_tuple=True
        )[0].item() + 1
        rank_after = (sorted_after == target_node).nonzero(
            as_tuple=True
        )[0].item() + 1

        return AttackMetrics(
            delta_r=delta_r,
            delta_r_normalized=delta_r_n,
            flipped=flipped,
            rank_before=rank_before,
            rank_after=rank_after,
            rank_shift=rank_after - rank_before,
            efficiency=abs(delta_r) / max(budget, 1),
            r_before=r_before,
            r_after=r_after,
        )

    # ── Influence Map Validation (E9) ──────────────────────────

    def validate_influence_map(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        top_n: int = 100,
    ) -> Dict:
        """
        Validate gradient-based influence map against brute-force
        ground truth.

        For the top-N edges by gradient magnitude, perturb each one
        (flip its label) and measure the ACTUAL reputation shift.
        Then compute Pearson correlation between gradient magnitude
        and actual |ΔR|.

        This is Experiment E9 — the method validation.

        Args:
            edges: current graph
            labels: current labels
            target_node: node to analyze
            top_n: how many edges to validate

        Returns:
            Dict with 'pearson_r', 'p_value', 'recall_at_10',
            'gradient_magnitudes', 'actual_shifts'
        """
        from scipy import stats

        # Step 1: Compute influence map
        imap = self.compute_influence_map(
            edges, labels, target_node, top_k=top_n
        )

        # Baseline reputation
        base_rep = imap.reputation_score

        # Step 2: For each top-N edge, flip its label and measure actual ΔR
        gradient_mags = []
        actual_shifts = []
        edge_indices = []

        top_indices = imap.influence_magnitudes.argsort(
            descending=True
        )[:top_n]

        for idx in top_indices:
            idx_val = idx.item()
            grad_mag = imap.influence_magnitudes[idx_val].item()

            # Flip this edge's label
            perturbed_labels = labels.clone()
            perturbed_labels[idx_val] = 1.0 - perturbed_labels[idx_val]

            # Measure actual reputation shift
            new_rep = self.compute_reputation(
                edges, perturbed_labels, target_node
            )
            actual_delta = abs(new_rep - base_rep)

            gradient_mags.append(grad_mag)
            actual_shifts.append(actual_delta)
            edge_indices.append(idx_val)

        gradient_mags = np.array(gradient_mags)
        actual_shifts = np.array(actual_shifts)

        # Step 3: Pearson correlation
        if len(gradient_mags) > 2 and np.std(gradient_mags) > 0:
            pearson_r, p_value = stats.pearsonr(
                gradient_mags, actual_shifts
            )
        else:
            pearson_r, p_value = 0.0, 1.0

        # Step 4: Recall@10
        # Are the gradient's top-10 among the brute-force top-10?
        gradient_top10 = set(np.argsort(gradient_mags)[-10:])
        actual_top10 = set(np.argsort(actual_shifts)[-10:])
        recall_at_10 = len(gradient_top10 & actual_top10) / 10.0

        return {
            'target_node': target_node,
            'pearson_r': pearson_r,
            'p_value': p_value,
            'recall_at_10': recall_at_10,
            'num_edges_validated': len(gradient_mags),
            'gradient_magnitudes': gradient_mags,
            'actual_shifts': actual_shifts,
            'edge_indices': edge_indices,
        }

    # ── Gradient-Based Influence Map ────────────────────────────

    def compute_influence_map(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        top_k: int = 20,
    ) -> InfluenceMap:
        """
        CORE METHOD: Single-backward-pass influence computation.

        Computes ∂reputation(target_node)/∂edge_label for EVERY edge
        in the graph simultaneously. One forward + one backward pass.

        The gradient magnitude of each edge tells us:
        "How much does this edge's trust/distrust signal affect
         the target node's reputation?"

        Args:
            edges: [2, E] edge index
            labels: [E, 2] one-hot trust/distrust labels
            target_node: whose reputation we're analyzing
            top_k: return this many most influential edges

        Returns:
            InfluenceMap with per-edge influence scores
        """
        # Create differentiable copy of edge labels
        # Model params are frozen — only labels get gradients
        soft_labels = labels.clone().detach().float().requires_grad_(True)

        # Forward pass through frozen model
        node_emb = self._compute_node_embeddings(edges, soft_labels)

        # Compute differentiable reputation score
        reputation = self._reputation_score(node_emb, target_node, edges)

        # Single backward pass: gradient flows through
        # S3R → StructuralNetwork → SpectralGatedMapping → edge labels
        reputation.backward()

        # Extract influence from gradients
        edge_grads = soft_labels.grad.detach()  # [E, 2]

        # Influence magnitude = L2 norm of gradient per edge
        magnitudes = edge_grads.norm(dim=1)  # [E]

        # Top-k most influential edges
        top_vals, top_idx = magnitudes.topk(min(top_k, len(magnitudes)))

        return InfluenceMap(
            target_node=target_node,
            edge_influences=edge_grads,
            influence_magnitudes=magnitudes,
            top_k_indices=top_idx,
            top_k_scores=top_vals,
            reputation_score=reputation.item(),
        )

    # ── New-Edge Influence Scoring ──────────────────────────────

    def rank_new_edges_gradient(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: Optional[List[int]] = None,
        sign: str = 'distrust',
        top_k: int = 10,
        max_candidates: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Genuine gradient-guided candidate ranking (GRAIL core method).
        
        Adds ALL candidate edges simultaneously as differentiable soft-label
        edges distributed across training snapshots, runs ONE forward+backward
        pass, and ranks candidates by gradient magnitude on their labels.
        
        Key insight: candidate edges must be distributed proportionally across
        ALL temporal snapshots (not just the last one) for gradient signal to
        propagate through the temporal GDTE backbone.
        
        Complexity: O(T_fwd + T_bwd) for all C candidates.
        """
        if candidate_sources is None:
            candidate_sources = list(range(self.num_nodes))
        
        existing = set()
        for i in range(edges.shape[1]):
            existing.add((edges[0, i].item(), edges[1, i].item()))
        
        candidates = [
            s for s in candidate_sources
            if s != target_node and (s, target_node) not in existing
        ]
        
        if len(candidates) > max_candidates:
            rng = np.random.RandomState(42)
            candidates = list(rng.choice(
                candidates, max_candidates, replace=False
            ))
        
        C = len(candidates)
        if C == 0:
            return []
        
        cand_edges = torch.tensor(
            [candidates, [target_node] * C],
            dtype=torch.long, device=self.device
        )
        
        # Distribute candidate edges across ALL training snapshots proportionally
        T = self.model.args.train_time_slots
        orig_idx_list = list(self.model.index_list)
        edges_per_snap = []
        prev = -1
        for idx in orig_idx_list[:T]:
            edges_per_snap.append(idx - prev)
            prev = idx
        total = sum(edges_per_snap)
        props = [max(1, int(C * e / total)) for e in edges_per_snap]
        # Adjust to sum exactly to C
        diff = C - sum(props)
        props[-1] += diff
        
        # Build augmented edges with candidates distributed across snapshots
        aug_parts_e, aug_parts_l = [], []
        orig_offset, cand_offset = 0, 0
        for t in range(T):
            end = orig_idx_list[t] + 1
            part_e = edges[:, orig_offset:end]
            part_l = labels[orig_offset:end, :]
            n_new = props[t]
            if n_new > 0:
                new_e = cand_edges[:, cand_offset:cand_offset+n_new]
                soft_l = torch.tensor(
                    [[0.0, 1.0] if sign == 'distrust' else [1.0, 0.0]],
                    device=self.device, dtype=torch.float
                ).repeat(n_new, 1)
                part_e = torch.cat([part_e, new_e], dim=1)
                part_l = torch.cat([part_l, soft_l], dim=0)
                cand_offset += n_new
            aug_parts_e.append(part_e)
            aug_parts_l.append(part_l)
            orig_offset = end
        
        aug_edges = torch.cat(aug_parts_e, dim=1)
        aug_labels = torch.cat(aug_parts_l, dim=0)
        
        # Track which rows in the concatenated labels belong to candidates
        candidate_label_indices = []
        row_offset = 0
        for t in range(T):
            n_orig = orig_idx_list[t] + 1 - (0 if t == 0 else orig_idx_list[t-1] + 1)
            # Original edges come first in each snapshot part
            n_orig_actual = aug_parts_l[t].shape[0] - props[t]
            row_offset += n_orig_actual
            # Candidate edges follow in the same snapshot
            n_new = props[t]
            if n_new > 0:
                new_indices = list(range(row_offset, row_offset + n_new))
                candidate_label_indices.extend(new_indices)
                row_offset += n_new
        
        # Make ALL labels differentiable
        soft_all = aug_labels.clone().detach().float().requires_grad_(True)
        
        # Update snapshot boundaries for augmented graph
        aug_index = []
        cumsum = 0
        for part in aug_parts_e:
            cumsum += part.shape[1]
            aug_index.append(cumsum - 1)
        aug_index = aug_index[:T]
        
        # Forward + backward pass (with try/finally for index_list safety)
        original_idx = self.model.index_list
        try:
            self.model.index_list = aug_index
            node_emb = self._compute_node_embeddings(aug_edges, soft_all)
            reputation = self._reputation_score(node_emb, target_node, aug_edges)
            reputation.backward()
        finally:
            self.model.index_list = original_idx
        
        # Extract gradients for candidate edges using tracked indices
        idx_tensor = torch.tensor(candidate_label_indices, device=self.device)
        cand_grads = soft_all.grad.index_select(0, idx_tensor)  # [C, 2]
        distrust_grad = cand_grads[:, 1]  # gradient w.r.t. distrust prob
        
        # Rank: most negative distrust gradient = most damaging
        if sign == 'distrust':
            ranked_indices = distrust_grad.argsort()
        else:
            ranked_indices = (-distrust_grad).argsort()
        
        results = []
        for idx in ranked_indices[:top_k]:
            src = candidates[idx.item()]
            results.append((int(src), float(distrust_grad[idx].item())))
        
        soft_all.grad = None
        return results

    def score_candidates_gradient(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        sign: str = 'distrust',
    ) -> Dict[int, float]:
        """Return gradient scores for ALL candidates (for proper correlation)."""
        if len(candidate_sources) == 0:
            return {}
        
        C = len(candidate_sources)
        cand_edges = torch.tensor(
            [candidate_sources, [target_node] * C],
            dtype=torch.long, device=self.device
        )
        
        T = self.model.args.train_time_slots
        orig_idx_list = list(self.model.index_list)
        edges_per_snap = []
        prev = -1
        for idx in orig_idx_list[:T]:
            edges_per_snap.append(idx - prev)
            prev = idx
        total = sum(edges_per_snap)
        props = [max(1, int(C * e / total)) for e in edges_per_snap]
        diff = C - sum(props)
        props[-1] += diff
        
        aug_parts_e, aug_parts_l = [], []
        orig_offset, cand_offset = 0, 0
        for t in range(T):
            end = orig_idx_list[t] + 1
            part_e = edges[:, orig_offset:end]
            part_l = labels[orig_offset:end, :]
            n_new = props[t]
            if n_new > 0:
                new_e = cand_edges[:, cand_offset:cand_offset+n_new]
                soft_l = torch.tensor(
                    [[0.0, 1.0] if sign == 'distrust' else [1.0, 0.0]],
                    device=self.device, dtype=torch.float
                ).repeat(n_new, 1)
                part_e = torch.cat([part_e, new_e], dim=1)
                part_l = torch.cat([part_l, soft_l], dim=0)
                cand_offset += n_new
            aug_parts_e.append(part_e)
            aug_parts_l.append(part_l)
            orig_offset = end
        
        aug_edges = torch.cat(aug_parts_e, dim=1)
        
        candidate_label_indices = []
        row_offset = 0
        for t in range(T):
            n_orig_actual = aug_parts_l[t].shape[0] - props[t]
            row_offset += n_orig_actual
            if props[t] > 0:
                new_indices = list(range(row_offset, row_offset + props[t]))
                candidate_label_indices.extend(new_indices)
                row_offset += props[t]
        
        soft_all = torch.cat(aug_parts_l, dim=0).clone().detach().float().requires_grad_(True)
        
        aug_index = []
        cumsum = 0
        for part in aug_parts_e:
            cumsum += part.shape[1]
            aug_index.append(cumsum - 1)
        aug_index = aug_index[:T]
        
        original_idx = self.model.index_list
        try:
            self.model.index_list = aug_index
            node_emb = self._compute_node_embeddings(aug_edges, soft_all)
            reputation = self._reputation_score(node_emb, target_node, aug_edges)
            reputation.backward()
        finally:
            self.model.index_list = original_idx
        
        idx_tensor = torch.tensor(candidate_label_indices, device=self.device)
        cand_grads = soft_all.grad.index_select(0, idx_tensor)
        distrust_grad = cand_grads[:, 1]
        
        result = {}
        for i, src in enumerate(candidate_sources):
            result[int(src)] = float(distrust_grad[i].item())
        
        soft_all.grad = None
        return result

    def score_candidates_gradient_single(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        sign: str = 'distrust',
        preinserted_sources: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Single-edge CLEAN-GRAPH gradient score for each candidate.

        Unlike ``score_candidates_gradient`` (which inserts ALL candidates at
        once, saturating the operating point), this inserts ONE soft distrust
        edge at a time and reads d R(target) / d y_{distrust}. The edge is placed
        in the LAST snapshot to match the counterfactual ``score_new_edge`` /
        ``rank_new_edges`` insertion exactly — removing BOTH confounds in the
        original gradient attack (joint saturation AND cross-snapshot placement).

        ``preinserted_sources``: optional list of sources to add as HARD distrust
        edges BEFORE measuring (the saturation "background"); used by the
        operating-point sweep. Empty/None => the clean single-edge gradient
        (operating point k=0).

        Returns ``{src: grad}`` where grad = d R / d y_{src, distrust}. More
        negative => inserting this distrust edge is predicted to reduce
        reputation more (same sign convention as the distrust counterfactual).

        Cost: one forward+backward per candidate. This is the diagnostic /
        reference extraction; the deployed efficient variants (integrated and
        greedy gradients) build on the same single-edge clean-graph operating
        point.
        """
        soft_vec = [0.0, 1.0] if sign == 'distrust' else [1.0, 0.0]
        label_component = 1 if sign == 'distrust' else 0

        # The candidate edge MUST be inserted into a snapshot the structural
        # layer actually processes (i = 0..train_time_slots-1). Appending past
        # the global last boundary (index_list[-1]) lands the edge in an
        # unprocessed snapshot, so its LABEL has zero gradient — the bug that
        # silently neutered the original attack. We splice into the last TRAINED
        # snapshot, the realistic position for a fresh attack edge.
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1  # end of last trained snapshot

        # Optional saturation background of HARD distrust edges, spliced in too.
        pre = list(preinserted_sources) if preinserted_sources else []
        n_bg = len(pre)
        if pre:
            bg_edges = torch.tensor(
                [pre, [target_node] * n_bg], dtype=torch.long, device=self.device,
            )
            bg_labels = torch.tensor(
                [soft_vec] * n_bg, dtype=torch.float, device=self.device,
            )
        original_idx = self.model.index_list
        result = {}
        for src in candidate_sources:
            new_edge = torch.tensor(
                [[src], [target_node]], dtype=torch.long, device=self.device,
            )
            soft_label = torch.tensor(
                [soft_vec], dtype=torch.float, device=self.device,
            ).requires_grad_(True)
            # Splice candidate (+ optional hard background) at the end of the
            # last trained snapshot, then shift boundaries >= T-1.
            if pre:
                ins_edges = torch.cat([bg_edges, new_edge], dim=1)
                ins_labels = torch.cat([bg_labels, soft_label], dim=0)
            else:
                ins_edges, ins_labels = new_edge, soft_label
            aug_edges = torch.cat([edges[:, :split], ins_edges, edges[:, split:]], dim=1)
            aug_labels = torch.cat([labels[:split], ins_labels, labels[split:]], dim=0)
            shift = n_bg + 1
            aug_index = [b + shift if i >= T - 1 else b
                         for i, b in enumerate(original_idx)]
            try:
                self.model.index_list = aug_index
                node_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                rep = self._reputation_score(node_emb, target_node, aug_edges)
                grad = torch.autograd.grad(rep, soft_label)[0]
            finally:
                self.model.index_list = original_idx
            result[int(src)] = float(grad[0, label_component].item())
        return result

    def rank_new_edges_gradient_single(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: Optional[List[int]] = None,
        sign: str = 'distrust',
        top_k: int = 10,
        max_candidates: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Corrected single-edge clean-graph gradient ranking.

        Ranks candidates by the single-edge clean-graph gradient (most negative
        distrust gradient first = most damaging), mirroring the counterfactual's
        ordering. This is the "break" method: the gradient extracted at the
        correct operating point.
        """
        if candidate_sources is None:
            candidate_sources = list(range(self.num_nodes))

        existing = set()
        for i in range(edges.shape[1]):
            existing.add((edges[0, i].item(), edges[1, i].item()))
        candidates = [
            s for s in candidate_sources
            if s != target_node and (s, target_node) not in existing
        ]
        if len(candidates) > max_candidates:
            rng = np.random.RandomState(42)
            candidates = list(rng.choice(candidates, max_candidates, replace=False))
        if not candidates:
            return []

        scores = self.score_candidates_gradient_single(
            edges, labels, target_node, candidates, sign=sign,
        )
        # Ascending: most-negative (most damaging) distrust gradient first.
        items = sorted(scores.items(), key=lambda x: x[1])
        if sign != 'distrust':
            items = items[::-1]
        return [(int(s), float(v)) for s, v in items[:top_k]]

    def score_candidates_counterfactual_processed(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        sign: str = 'distrust',
    ) -> Dict[int, float]:
        """
        Exact single-edge counterfactual ΔR with the edge spliced into the last
        TRAINED snapshot (so the structural layer processes it) — the fair,
        label-sensitive reference matching ``score_candidates_gradient_single``.

        Contrast with ``rank_new_edges`` / ``score_new_edge``, which append the
        edge past the trained horizon: there the structural layer ignores it and
        ΔR reduces to the reputation-average effect alone (label-independent).
        """
        lab_vec = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1
        original_idx = self.model.index_list

        base_rep = self.compute_reputation(edges, labels, target_node)
        result = {}
        for src in candidate_sources:
            new_edge = torch.tensor(
                [[src], [target_node]], dtype=torch.long, device=self.device,
            )
            new_label = torch.tensor([lab_vec], dtype=torch.float, device=self.device)
            aug_edges = torch.cat([edges[:, :split], new_edge, edges[:, split:]], dim=1)
            aug_labels = torch.cat([labels[:split], new_label, labels[split:]], dim=0)
            aug_index = [b + 1 if i >= T - 1 else b for i, b in enumerate(original_idx)]
            try:
                self.model.index_list = aug_index
                with torch.no_grad():
                    node_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                    new_rep = self._reputation_score(node_emb, target_node, aug_edges).item()
            finally:
                self.model.index_list = original_idx
            result[int(src)] = float(new_rep - base_rep)
        return result

    def score_candidates_gradient_batch_processed(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        baseline: str = 'trust',
        sign: str = 'distrust',
    ) -> Dict[int, float]:
        """
        EFFICIENT one-backward-pass gradient ranking at the CORRECT (processed)
        placement. All C candidate edges are spliced into the last trained snapshot
        with a NON-saturating baseline label (trust [1,0] or neutral [0.5,0.5]) so
        the operating point is not distrust-saturated (the bug that inverted the
        original attack). A single forward+backward yields d R / d y_distrust for
        ALL candidates at once — O(1) backward vs the counterfactual's O(C) forward.

        Returns {src: grad}; most-negative = most damaging.
        """
        C = len(candidate_sources)
        if C == 0:
            return {}
        base_vec = [1.0, 0.0] if baseline == 'trust' else [0.5, 0.5]
        comp = 1 if sign == 'distrust' else 0
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1

        cand_e = torch.tensor(
            [list(candidate_sources), [target_node] * C],
            dtype=torch.long, device=self.device)
        soft = torch.tensor(
            [base_vec] * C, dtype=torch.float, device=self.device).requires_grad_(True)
        aug_e = torch.cat([edges[:, :split], cand_e, edges[:, split:]], dim=1)
        aug_l = torch.cat([labels[:split], soft, labels[split:]], dim=0)
        aug_index = [b + C if i >= T - 1 else b
                     for i, b in enumerate(self.model.index_list)]
        orig = self.model.index_list
        try:
            self.model.index_list = aug_index
            node_emb = self._compute_node_embeddings(aug_e, aug_l)
            rep = self._reputation_score(node_emb, target_node, aug_e)
            grad = torch.autograd.grad(rep, soft)[0]  # [C, 2]
        finally:
            self.model.index_list = orig
        return {int(s): float(grad[i, comp].item())
                for i, s in enumerate(candidate_sources)}

    def score_candidates_integrated_gradient_processed(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        steps: int = 16,
        sign: str = 'distrust',
    ) -> Dict[int, float]:
        """
        Integrated-gradient ranking at the processed placement: integrate
        d R / d y along the label path from trust [1,0] to distrust [0,1] for all
        candidates jointly. ``steps`` backward passes total (not C*steps) — still
        far cheaper than per-candidate counterfactuals when C >> steps.

        IG attribution along this path telescopes toward R(distrust)-R(trust); we
        use it as a smoothed, less-saturation-sensitive ranking signal.
        """
        C = len(candidate_sources)
        if C == 0:
            return {}
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1
        cand_e = torch.tensor(
            [list(candidate_sources), [target_node] * C],
            dtype=torch.long, device=self.device)
        aug_index = [b + C if i >= T - 1 else b
                     for i, b in enumerate(self.model.index_list)]
        orig = self.model.index_list
        ig = torch.zeros(C, device=self.device)
        try:
            self.model.index_list = aug_index
            for k in range(steps):
                alpha = (k + 0.5) / steps  # midpoint Riemann
                lab = torch.tensor([[1.0 - alpha, alpha]] * C,
                                   dtype=torch.float, device=self.device).requires_grad_(True)
                aug_e = torch.cat([edges[:, :split], cand_e, edges[:, split:]], dim=1)
                aug_l = torch.cat([labels[:split], lab, labels[split:]], dim=0)
                node_emb = self._compute_node_embeddings(aug_e, aug_l)
                rep = self._reputation_score(node_emb, target_node, aug_e)
                g = torch.autograd.grad(rep, lab)[0]  # [C,2]
                # path direction trust->distrust: d/dalpha = g_distrust - g_trust
                ig += (g[:, 1] - g[:, 0]) / steps
        finally:
            self.model.index_list = orig
        s = 1.0 if sign == 'distrust' else -1.0
        return {int(src): float(s * ig[i].item()) for i, src in enumerate(candidate_sources)}

    def rank_new_edges_greedy_processed(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: List[int],
        budget: int,
        sign: str = 'distrust',
    ) -> List[int]:
        """
        Greedy guided attack at the processed placement. Each step ranks remaining
        candidates by the one-pass batch gradient, commits the most-damaging one as
        a HARD edge into the snapshot, and re-ranks against the updated graph.
        ``budget`` backward passes total. Returns the chosen source set.
        """
        chosen = []
        cur_edges, cur_labels = edges, labels
        remaining = list(candidate_sources)
        lab_hard = [0.0, 1.0] if sign == 'distrust' else [1.0, 0.0]
        T = self.model.args.train_time_slots
        for _ in range(budget):
            if not remaining:
                break
            scores = self.score_candidates_gradient_batch_processed(
                cur_edges, cur_labels, target_node, remaining, baseline='trust', sign=sign)
            best = min(scores, key=scores.get)  # most negative
            chosen.append(best)
            remaining.remove(best)
            # commit as hard edge in last trained snapshot
            split = self.model.index_list[T - 1] + 1
            ne = torch.tensor([[best], [target_node]], dtype=torch.long, device=self.device)
            nl = torch.tensor([lab_hard], dtype=torch.float, device=self.device)
            cur_edges = torch.cat([cur_edges[:, :split], ne, cur_edges[:, split:]], dim=1)
            cur_labels = torch.cat([cur_labels[:split], nl, cur_labels[split:]], dim=0)
            self.model.index_list = [b + 1 if i >= T - 1 else b
                                     for i, b in enumerate(self.model.index_list)]
        # restore index_list (greedy mutated it by len(chosen))
        self.model.index_list = [b - len(chosen) if i >= T - 1 else b
                                 for i, b in enumerate(self.model.index_list)]
        return chosen

    def score_edge_set_processed(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        sources: List[int],
        target_node: int,
        sign: str = 'distrust',
        return_after: bool = False,
    ):
        """
        Joint ΔR of injecting B edges spliced into the last TRAINED snapshot
        (re-embedding placement), the TM-B realization. Mirrors ``score_edge_set``
        but places edges where the structural layer processes them rather than
        appending past the trained horizon.

        With ``return_after=True`` returns (delta_r, r_before, r_after) so callers
        can compute threshold-flip rates.
        """
        B = len(sources)
        if B == 0:
            return (0.0, self.compute_reputation(edges, labels, target_node),
                    self.compute_reputation(edges, labels, target_node)) if return_after else 0.0
        lab_vec = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1
        original_idx = self.model.index_list

        new_edges = torch.tensor(
            [list(sources), [target_node] * B], dtype=torch.long, device=self.device)
        new_labels = torch.tensor([lab_vec] * B, dtype=torch.float, device=self.device)
        aug_edges = torch.cat([edges[:, :split], new_edges, edges[:, split:]], dim=1)
        aug_labels = torch.cat([labels[:split], new_labels, labels[split:]], dim=0)
        aug_index = [b + B if i >= T - 1 else b for i, b in enumerate(original_idx)]

        base_rep = self.compute_reputation(edges, labels, target_node)
        try:
            self.model.index_list = aug_index
            with torch.no_grad():
                node_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                new_rep = self._reputation_score(node_emb, target_node, aug_edges).item()
        finally:
            self.model.index_list = original_idx
        dr = float(new_rep - base_rep)
        return (dr, float(base_rep), float(new_rep)) if return_after else dr

    def decompose_processed(self, edges, labels, sources, target_node, sign='distrust'):
        """Decompose processed-placement \\Delta R into two additive parts:
          averaging effect  = adding the new raters to the mean at FROZEN (clean)
                              embeddings (label-independent dilution), and
          embedding effect  = the change in trust-probabilities of the (augmented)
                              rater set caused by re-embedding the labeled edges.
        Returns (dR_total, dR_avg, dR_emb) with dR_avg + dR_emb = dR_total.
        A dominant embedding effect shows the drop is a genuine model response to
        the processed label, not fragility of the unweighted mean."""
        B = len(sources)
        if B == 0:
            return 0.0, 0.0, 0.0
        lab_vec = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
        T = self.model.args.train_time_slots
        split = self.model.index_list[T - 1] + 1
        original_idx = self.model.index_list
        new_edges = torch.tensor(
            [list(sources), [target_node] * B], dtype=torch.long, device=self.device)
        new_labels = torch.tensor([lab_vec] * B, dtype=torch.float, device=self.device)
        aug_edges = torch.cat([edges[:, :split], new_edges, edges[:, split:]], dim=1)
        aug_labels = torch.cat([labels[:split], new_labels, labels[split:]], dim=0)
        aug_index = [b + B if i >= T - 1 else b for i, b in enumerate(original_idx)]
        with torch.no_grad():
            z0 = self._compute_node_embeddings(edges, labels)                  # clean
            R0 = self._reputation_score(z0, target_node, edges).item()
            R_avg = self._reputation_score(z0, target_node, aug_edges).item()  # dilution @ z0
            try:
                self.model.index_list = aug_index
                z1 = self._compute_node_embeddings(aug_edges, aug_labels)      # re-embedded
                R1 = self._reputation_score(z1, target_node, aug_edges).item()
            finally:
                self.model.index_list = original_idx
        return float(R1 - R0), float(R_avg - R0), float(R1 - R_avg)

    def score_edge_set(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        sources: List[int],
        target_node: int,
        sign: str = 'distrust',
    ) -> float:
        """Score JOINT effect of injecting B edges simultaneously."""
        B = len(sources)
        if B == 0:
            return 0.0
        new_edges = torch.tensor(
            [sources, [target_node] * B],
            dtype=torch.long, device=self.device
        )
        new_labels = torch.tensor(
            [[1.0, 0.0] if sign == 'trust' else [0.0, 1.0]] * B,
            dtype=torch.float, device=self.device
        )
        aug_edges = torch.cat([edges, new_edges], dim=1)
        aug_labels = torch.cat([labels, new_labels], dim=0)
        aug_index = list(self.model.index_list)
        aug_index[-1] += B
        
        original_idx = self.model.index_list
        try:
            self.model.index_list = aug_index
            with torch.no_grad():
                base_rep = self.compute_reputation(edges, labels, target_node)
                node_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                new_rep = self._reputation_score(node_emb, target_node, aug_edges).item()
        finally:
            self.model.index_list = original_idx
        return new_rep - base_rep

    def score_new_edge(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        source: int,
        target_node: int,
        sign: str = 'distrust',
    ) -> float:
        """Compute influence of a single NEW edge on target's reputation."""
        new_edge = torch.tensor(
            [[source], [target_node]], dtype=torch.long, device=self.device
        )
        new_label = torch.tensor(
            [[1.0, 0.0] if sign == 'trust' else [0.0, 1.0]],
            device=self.device,
        )

        # Augment graph
        aug_edges = torch.cat([edges, new_edge], dim=1)
        aug_labels = torch.cat([labels, new_label], dim=0)

        # Update last snapshot boundary
        aug_index = list(self.model.index_list)
        aug_index[-1] += 1

        # Compute reputation with and without the new edge
        with torch.no_grad():
            # Baseline
            base_emb = self._compute_node_embeddings(edges, labels)
            base_rep = self._reputation_score(
                base_emb, target_node, edges
            ).item()

            # With new edge
            original_idx = self.model.index_list
            try:
                self.model.index_list = aug_index
                new_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                new_rep = self._reputation_score(
                    new_emb, target_node, aug_edges
                ).item()
            finally:
                self.model.index_list = original_idx

        return new_rep - base_rep

    def rank_new_edges(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        candidate_sources: Optional[List[int]] = None,
        sign: str = 'distrust',
        top_k: int = 10,
        max_candidates: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate NEW edges by their impact on target's reputation.

        Returns list of (source_node, reputation_shift) sorted by
        magnitude of shift.
        """
        if candidate_sources is None:
            candidate_sources = list(range(self.num_nodes))

        # Filter out existing edges and self-loops
        existing = set()
        for i in range(edges.shape[1]):
            existing.add((edges[0, i].item(), edges[1, i].item()))

        candidates = [
            s for s in candidate_sources
            if s != target_node and (s, target_node) not in existing
        ]

        # Sample if too many candidates
        if len(candidates) > max_candidates:
            rng = np.random.RandomState(42)
            candidates = list(rng.choice(
                candidates, max_candidates, replace=False
            ))

        # Score each candidate
        results = []
        with torch.no_grad():
            base_emb = self._compute_node_embeddings(edges, labels)
            base_rep = self._reputation_score(
                base_emb, target_node, edges
            ).item()

        for src in candidates:
            new_edge = torch.tensor(
                [[src], [target_node]], dtype=torch.long, device=self.device
            )
            new_label = torch.tensor(
                [[1.0, 0.0] if sign == 'trust' else [0.0, 1.0]],
                device=self.device,
            )
            aug_edges = torch.cat([edges, new_edge], dim=1)
            aug_labels = torch.cat([labels, new_label], dim=0)
            aug_index = list(self.model.index_list)
            aug_index[-1] += 1

            original_idx = self.model.index_list
            try:
                self.model.index_list = aug_index
                with torch.no_grad():
                    new_emb = self._compute_node_embeddings(aug_edges, aug_labels)
                    new_rep = self._reputation_score(
                        new_emb, target_node, aug_edges
                    ).item()
            finally:
                self.model.index_list = original_idx
            results.append((src, new_rep - base_rep))

        # Sort by magnitude of shift (most damaging first for distrust)
        if sign == 'distrust':
            results.sort(key=lambda x: x[1])  # most negative first
        else:
            results.sort(key=lambda x: -x[1])  # most positive first

        return results[:top_k]

    # ── Baseline Comparisons ────────────────────────────────────

    def baseline_degree_ranking(
        self,
        edges: torch.Tensor,
        target_node: int,
        top_k: int = 10,
    ) -> List[int]:
        """Baseline: rank candidate attackers by degree centrality."""
        degrees = torch.zeros(self.num_nodes, device=self.device)
        src_nodes = edges[0]
        for s in src_nodes:
            degrees[s] += 1
        # Exclude target and existing edges
        existing_src = set(
            edges[0, edges[1] == target_node].tolist()
        )
        candidates = [
            (int(n), degrees[n].item())
            for n in range(self.num_nodes)
            if n != target_node and n not in existing_src
        ]
        candidates.sort(key=lambda x: -x[1])
        return [c[0] for c in candidates[:top_k]]

    def baseline_pagerank_ranking(
        self,
        edges: torch.Tensor,
        target_node: int,
        top_k: int = 10,
        num_iterations: int = 20,
        damping: float = 0.85,
    ) -> List[int]:
        """Baseline: rank candidate attackers by PageRank."""
        N = self.num_nodes
        pr = torch.ones(N, device=self.device) / N

        # Build adjacency as source → destination
        src = edges[0].long()
        dst = edges[1].long()
        out_degree = torch.zeros(N, device=self.device)
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        out_degree = out_degree.clamp(min=1)

        for _ in range(num_iterations):
            contributions = pr[src] / out_degree[src]
            new_pr = torch.zeros(N, device=self.device)
            new_pr.scatter_add_(0, dst, contributions)
            pr = (1 - damping) / N + damping * new_pr

        # Exclude target and existing edges
        existing_src = set(
            edges[0, edges[1] == target_node].tolist()
        )
        candidates = [
            (int(n), pr[n].item())
            for n in range(N)
            if n != target_node and n not in existing_src
        ]
        candidates.sort(key=lambda x: -x[1])
        return [c[0] for c in candidates[:top_k]]

    def baseline_betweenness_ranking(
        self,
        edges: torch.Tensor,
        target_node: int,
        top_k: int = 10,
    ) -> List[int]:
        """
        Baseline: rank candidate attackers by approximate
        betweenness centrality (sampled BFS).
        """
        N = self.num_nodes
        src = edges[0].long().cpu().numpy()
        dst = edges[1].long().cpu().numpy()

        # Build adjacency list
        adj = {i: [] for i in range(N)}
        for s, d in zip(src, dst):
            adj[s].append(d)

        # Approximate betweenness via sampled BFS
        betweenness = np.zeros(N)
        sample_size = min(50, N)
        rng = np.random.RandomState(42)
        sample_nodes = rng.choice(N, sample_size, replace=False)

        from collections import deque
        for start in sample_nodes:
            visited = {start: 0}
            queue = deque([start])
            predecessors = {start: []}
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        predecessors[neighbor] = [node]
                        queue.append(neighbor)
                    elif visited[neighbor] == visited[node] + 1:
                        predecessors[neighbor].append(node)
            # Backtrack to accumulate betweenness
            delta = {v: 0.0 for v in visited}
            nodes_by_dist = sorted(
                visited.keys(), key=lambda v: -visited[v]
            )
            for v in nodes_by_dist:
                for pred in predecessors.get(v, []):
                    delta[pred] += (1 + delta[v]) / len(
                        predecessors[v]
                    )
                if v != start:
                    betweenness[v] += delta[v]

        # Exclude target and existing edges
        existing_src = set(
            edges[0, edges[1] == target_node].tolist()
        )
        candidates = [
            (int(n), betweenness[n])
            for n in range(N)
            if n != target_node and n not in existing_src
        ]
        candidates.sort(key=lambda x: -x[1])
        return [c[0] for c in candidates[:top_k]]

    def get_high_reputation_nodes(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.9,
    ) -> List[int]:
        """Return nodes with reputation > threshold (matching paper definition)."""
        reps = self.compute_all_reputations(edges, labels)
        return [int(v) for v in range(self.num_nodes) if reps[v].item() > threshold]

    def baseline_expert_heuristic_ranking(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        good_nodes: List[int],
        top_k: int = 10,
    ) -> List[int]:
        """
        Expert Heuristic: rank attackers by how many distrust edges
        they already have toward other good nodes.

        A model-free but savvy adversary would pick accounts that
        already have a track record of distrusting reputable users.
        """
        good_set = set(good_nodes)
        # For each node, count distrust edges toward good nodes
        distrust_score = np.zeros(self.num_nodes)
        for i in range(edges.shape[1]):
            src = edges[0, i].item()
            dst = edges[1, i].item()
            # labels[:, 1] > labels[:, 0] means distrust
            if labels[i, 1] > labels[i, 0] and dst in good_set:
                distrust_score[src] += 1

        existing_src = set(
            edges[0, edges[1] == target_node].tolist()
        )
        candidates = [
            (int(n), distrust_score[n])
            for n in range(self.num_nodes)
            if n != target_node and n not in existing_src
        ]
        candidates.sort(key=lambda x: -x[1])
        return [c[0] for c in candidates[:top_k]]

    # ── Stratified Target Selection ─────────────────────────────

    def select_stratified_targets(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        per_stratum: int = 5,
    ) -> Dict[str, List[int]]:
        """
        Select target nodes across three reputation strata.
        
        Instead of attacking only ultra-robust nodes, we find targets
        across the full range of reputations — focusing on the moderate
        and low strata where real-world damage is most plausible.

        Returns dict with keys 'high', 'moderate', 'low', each
        containing a list of node IDs.
        """
        all_reps = self.compute_all_reputations(edges, labels)

        strata = {'high': [], 'moderate': [], 'low': []}

        for v in range(self.num_nodes):
            r = all_reps[v].item()
            # Must have at least some incoming edges to be meaningful
            in_count = (edges[1] == v).sum().item()
            if in_count < 2:
                continue
            if 0.9 <= r <= 1.0:
                strata['high'].append((v, r, in_count))
            elif 0.6 <= r < 0.9:
                strata['moderate'].append((v, r, in_count))
            elif 0.5 <= r < 0.6:
                strata['low'].append((v, r, in_count))

        # Sort by reputation and pick per_stratum from each
        result = {}
        for key in strata:
            # Prefer nodes with moderate in-degree (10-100)
            nodes = sorted(strata[key], key=lambda x: abs(x[2] - 30))
            result[key] = [n[0] for n in nodes[:per_stratum]]

        return result

    def compute_confidence_weighted_influence(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        top_k: int = 20,
    ) -> 'InfluenceMap':
        """
        Confidence-weighted influence map: multiply gradient magnitude
        by the model's prediction certainty for each edge.

        This improves reliability on weaker models (like Alpha) by
        downweighting edges where the model is uncertain.
        """
        # Get standard influence map
        imap = self.compute_influence_map(edges, labels, target_node, top_k)

        # Compute prediction certainty for each edge
        with torch.no_grad():
            node_emb = self._compute_node_embeddings(edges, labels)
            src_emb = node_emb[edges[0]]
            dst_emb = node_emb[edges[1]]
            features = torch.cat((src_emb, dst_emb), dim=1)
            logits = features @ self.model.regression_weights
            probs = F.softmax(logits, dim=1)
            # Certainty = max(p_trust, p_distrust) — ranges from 0.5 to 1.0
            certainty = probs.max(dim=1).values

        # Weight gradients by certainty
        weighted_magnitudes = imap.influence_magnitudes * certainty
        top_vals, top_idx = weighted_magnitudes.topk(
            min(top_k, len(weighted_magnitudes))
        )

        return InfluenceMap(
            target_node=target_node,
            edge_influences=imap.edge_influences,
            influence_magnitudes=weighted_magnitudes,
            top_k_indices=top_idx,
            top_k_scores=top_vals,
            reputation_score=imap.reputation_score,
        )


# ────────────────────────────────────────────────────────────────
# Counterfactual Simulator
# ────────────────────────────────────────────────────────────────

class CounterfactualSimulator:
    """
    Multi-step reputation attack planner.

    Simulates sequences of fake interactions and measures how
    the target's reputation evolves after each injection.
    """

    def __init__(self, oracle: ReputationAttackOracle):
        self.oracle = oracle

    def simulate_sequence(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        interventions: List[Tuple[int, str]],
    ) -> AttackPlan:
        """
        Simulate injecting a sequence of fake edges targeting
        a specific node and track reputation change at each step.

        Args:
            edges: current graph
            labels: current labels
            target_node: node whose reputation we're attacking
            interventions: [(source_node, sign), ...] in order

        Returns:
            AttackPlan with per-step reputation tracking
        """
        base_rep = self.oracle.compute_reputation(edges, labels, target_node)

        current_edges = edges.clone()
        current_labels = labels.clone()
        aug_index = list(self.oracle.model.index_list)
        original_idx = self.oracle.model.index_list

        steps = []
        prev_rep = base_rep

        for i, (src, sign) in enumerate(interventions):
            # Inject edge
            new_edge = torch.tensor(
                [[src], [target_node]],
                dtype=torch.long, device=self.oracle.device,
            )
            new_label = torch.tensor(
                [[1.0, 0.0] if sign == 'trust' else [0.0, 1.0]],
                device=self.oracle.device,
            )
            current_edges = torch.cat([current_edges, new_edge], dim=1)
            current_labels = torch.cat([current_labels, new_label], dim=0)
            aug_index[-1] += 1

            # Measure reputation after this injection
            self.oracle.model.index_list = list(aug_index)
            new_rep = self.oracle.compute_reputation(
                current_edges, current_labels, target_node
            )

            steps.append({
                'step': i + 1,
                'source': src,
                'sign': sign,
                'reputation_before': prev_rep,
                'reputation_after': new_rep,
                'step_delta': new_rep - prev_rep,
                'cumulative_delta': new_rep - base_rep,
            })
            prev_rep = new_rep

        self.oracle.model.index_list = original_idx

        full_interventions = [(s, target_node, sign) for s, sign in interventions]
        actual_shift = steps[-1]['cumulative_delta'] if steps else 0.0

        return AttackPlan(
            target_node=target_node,
            interventions=full_interventions,
            predicted_shift=actual_shift,
            actual_shift=actual_shift,
            steps=steps,
        )

    def greedy_plan(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        target_node: int,
        budget: int = 5,
        sign: str = 'distrust',
        candidate_pool_size: int = 50,
    ) -> AttackPlan:
        """
        Greedily build an optimal attack plan.

        At each step, uses the influence map to pick the edge
        that would most damage the target's reputation, injects it,
        and repeats.

        Args:
            target_node: whose reputation to attack
            budget: max fake edges to inject
            sign: 'distrust' to damage, 'trust' to boost
            candidate_pool_size: how many candidates to evaluate per step
        """
        current_edges = edges.clone()
        current_labels = labels.clone()
        aug_index = list(self.oracle.model.index_list)
        original_idx = self.oracle.model.index_list

        base_rep = self.oracle.compute_reputation(edges, labels, target_node)
        plan_steps = []
        used_sources = set()

        for step in range(budget):
            # Use influence map to identify promising candidates
            self.oracle.model.index_list = list(aug_index)
            imap = self.oracle.compute_influence_map(
                current_edges, current_labels, target_node
            )

            # Get existing sources for this target to avoid duplicates
            existing_src = set(
                current_edges[0, current_edges[1] == target_node].tolist()
            )
            existing_src.update(used_sources)

            # Select candidates: nodes with high influence on target
            # that don't already have an edge to target
            candidate_nodes = [
                n for n in range(self.oracle.num_nodes)
                if n != target_node and n not in existing_src
            ]

            if not candidate_nodes:
                break

            # Score candidates by new-edge influence
            sample = candidate_nodes[:candidate_pool_size]
            best_src = None
            best_shift = 0.0

            for src in sample:
                shift = self.oracle.score_new_edge(
                    current_edges, current_labels,
                    src, target_node, sign
                )
                if sign == 'distrust' and shift < best_shift:
                    best_shift = shift
                    best_src = src
                elif sign == 'trust' and shift > best_shift:
                    best_shift = shift
                    best_src = src

            if best_src is None:
                break

            # Inject the best edge
            new_edge = torch.tensor(
                [[best_src], [target_node]],
                dtype=torch.long, device=self.oracle.device,
            )
            new_label = torch.tensor(
                [[1.0, 0.0] if sign == 'trust' else [0.0, 1.0]],
                device=self.oracle.device,
            )
            current_edges = torch.cat([current_edges, new_edge], dim=1)
            current_labels = torch.cat([current_labels, new_label], dim=0)
            aug_index[-1] += 1
            used_sources.add(best_src)

            # Measure actual reputation after injection
            self.oracle.model.index_list = list(aug_index)
            new_rep = self.oracle.compute_reputation(
                current_edges, current_labels, target_node
            )

            plan_steps.append({
                'step': step + 1,
                'source': best_src,
                'sign': sign,
                'predicted_shift': best_shift,
                'reputation_after': new_rep,
                'cumulative_delta': new_rep - base_rep,
            })

        self.oracle.model.index_list = original_idx

        interventions = [
            (s['source'], target_node, s['sign']) for s in plan_steps
        ]
        actual_shift = plan_steps[-1]['cumulative_delta'] if plan_steps else 0.0

        return AttackPlan(
            target_node=target_node,
            interventions=interventions,
            predicted_shift=actual_shift,
            actual_shift=actual_shift,
            steps=plan_steps,
        )


# ────────────────────────────────────────────────────────────────
# Defence & Analysis Utilities
# ────────────────────────────────────────────────────────────────

def detect_attack_edges(
    oracle: ReputationAttackOracle,
    real_edges: torch.Tensor,
    real_labels: torch.Tensor,
    attack_edges: torch.Tensor,
    attack_labels: torch.Tensor,
) -> Dict:
    """
    Defence experiment: train a logistic regression to distinguish
    real edges from injected attack edges using structural features.

    Features per edge: src_degree, dst_degree, src_reputation,
    dst_reputation, edge_symmetry (does reverse edge exist?).

    Returns dict with AUC, accuracy, feature importances.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    device = oracle.device
    all_edges = torch.cat([real_edges, attack_edges], dim=1)

    # Compute degrees
    src_deg = np.bincount(all_edges[0].cpu().numpy(), minlength=oracle.num_nodes)
    dst_deg = np.bincount(all_edges[1].cpu().numpy(), minlength=oracle.num_nodes)

    # Compute reputations (on real graph only)
    all_reps = oracle.compute_all_reputations(real_edges, real_labels)
    reps = all_reps.cpu().numpy()

    def edge_features(edges_t):
        feats = []
        src = edges_t[0].cpu().numpy()
        dst = edges_t[1].cpu().numpy()
        # Build reverse edge set for symmetry check
        rev_set = set()
        for i in range(real_edges.shape[1]):
            rev_set.add((real_edges[1, i].item(), real_edges[0, i].item()))
        for i in range(len(src)):
            s, d = src[i], dst[i]
            sym = 1.0 if (s, d) in rev_set else 0.0
            feats.append([
                src_deg[s], dst_deg[d],
                reps[s] if s < len(reps) else 0.5,
                reps[d] if d < len(reps) else 0.5,
                sym,
            ])
        return np.array(feats)

    X_real = edge_features(real_edges)
    X_attack = edge_features(attack_edges)

    # Subsample real edges to balance
    n_attack = len(X_attack)
    if len(X_real) > n_attack * 5:
        idx = np.random.choice(len(X_real), n_attack * 5, replace=False)
        X_real = X_real[idx]

    X = np.vstack([X_real, X_attack])
    y = np.array([0] * len(X_real) + [1] * len(X_attack))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    feature_names = ['src_degree', 'dst_degree', 'src_reputation',
                     'dst_reputation', 'edge_symmetry']

    return {
        'auc': float(roc_auc_score(y, y_prob)),
        'accuracy': float(accuracy_score(y, y_pred)),
        'n_real': len(X_real),
        'n_attack': n_attack,
        'feature_importances': {
            name: float(coef) for name, coef
            in zip(feature_names, clf.coef_[0])
        },
    }


def analyze_gradient_failures(
    oracle: ReputationAttackOracle,
    edges: torch.Tensor,
    labels: torch.Tensor,
    target_node: int,
    top_n: int = 50,
) -> Dict:
    """
    Diagnose when and why gradient-based influence fails.

    Identifies edges where gradient magnitude is high but actual
    |ΔR| is low (false positives) and vice versa (false negatives).
    Characterises them by edge features.
    """
    from scipy import stats

    imap = oracle.compute_influence_map(edges, labels, target_node, top_k=top_n)
    base_rep = imap.reputation_score

    top_indices = imap.influence_magnitudes.argsort(descending=True)[:top_n]

    grad_mags, actual_shifts = [], []
    edge_info = []

    for idx in top_indices:
        idx_val = idx.item()
        grad = imap.influence_magnitudes[idx_val].item()

        perturbed = labels.clone()
        perturbed[idx_val] = 1.0 - perturbed[idx_val]
        new_rep = oracle.compute_reputation(edges, perturbed, target_node)
        actual = abs(new_rep - base_rep)

        grad_mags.append(grad)
        actual_shifts.append(actual)
        edge_info.append({
            'edge_idx': idx_val,
            'src': edges[0, idx_val].item(),
            'dst': edges[1, idx_val].item(),
            'gradient': grad,
            'actual_shift': actual,
        })

    grad_mags = np.array(grad_mags)
    actual_shifts = np.array(actual_shifts)

    # Identify false positives (high gradient, low actual)
    if len(grad_mags) > 5:
        grad_pct = np.percentile(grad_mags, 75)
        actual_pct = np.percentile(actual_shifts, 25)
        false_positives = [
            e for e, g, a in zip(edge_info, grad_mags, actual_shifts)
            if g >= grad_pct and a <= actual_pct
        ]
        false_negatives = [
            e for e, g, a in zip(edge_info, grad_mags, actual_shifts)
            if g <= np.percentile(grad_mags, 25) and a >= np.percentile(actual_shifts, 75)
        ]
    else:
        false_positives, false_negatives = [], []

    # Compute prediction certainty for false-positive edges
    src_degrees = np.bincount(
        edges[0].cpu().numpy(), minlength=oracle.num_nodes
    )
    fp_degrees = [src_degrees[e['src']] for e in false_positives]
    fn_degrees = [src_degrees[e['src']] for e in false_negatives]

    return {
        'target_node': target_node,
        'n_false_positives': len(false_positives),
        'n_false_negatives': len(false_negatives),
        'fp_mean_src_degree': float(np.mean(fp_degrees)) if fp_degrees else 0,
        'fn_mean_src_degree': float(np.mean(fn_degrees)) if fn_degrees else 0,
        'base_reputation': float(base_rep),
        'pearson_r': float(stats.pearsonr(grad_mags, actual_shifts)[0])
            if len(grad_mags) > 2 else 0.0,
    }
