# ──────────────────────────────────────────────────────────────────────────────
#  CUDA-Accelerated Trust-Attack Graph Utilities
#  Requires: PyTorch ≥ 1.10 with CUDA
# ──────────────────────────────────────────────────────────────────────────────

import random
import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"


# ──────────────────────────────────────────────────────────────────────────────
# Internal GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_one_hot_labels(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    if labels is None:
        raise ValueError("labels is None")
    if labels.dim() == 1:
        return F.one_hot(labels.long(), num_classes=num_classes).to(torch.int64)
    if labels.dim() == 2 and labels.shape[1] == 1:
        return F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).to(torch.int64)
    if labels.dim() == 2 and labels.shape[1] == num_classes:
        return F.one_hot(
            torch.argmax(labels, dim=1).long(), num_classes=num_classes
        ).to(torch.int64)
    raise ValueError(f"Unsupported label shape {labels.shape}")


def _deduplicate_edges_and_labels(
    edges: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if edges.numel() == 0:
        return edges, labels
    seen = set()
    keep_indices: List[int] = []
    for idx, (src, dst) in enumerate(edges.t().tolist()):
        edge = (int(src), int(dst))
        if edge not in seen:
            seen.add(edge)
            keep_indices.append(idx)
    if len(keep_indices) == edges.shape[1]:
        return edges, labels
    keep_idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=edges.device)
    dedup_edges = edges.index_select(1, keep_idx_tensor)
    dedup_labels = labels.index_select(0, keep_idx_tensor.to(labels.device))
    return dedup_edges, dedup_labels


def _compute_node_ratings_gpu(
    edges_long: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    device = edges_long.device
    labels_one_hot = _ensure_one_hot_labels(labels, num_classes=2)
    node_ratings = torch.zeros((num_nodes, 2), dtype=torch.int64, device=device)
    dst = edges_long[1]
    node_ratings.scatter_add_(
        0,
        dst.unsqueeze(1).expand(-1, 2),
        labels_one_hot.to(dtype=torch.int64),
    )
    return node_ratings


def _future_target_signal(
    target_edges: Optional[torch.Tensor],
    target_labels: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.zeros(num_nodes, dtype=torch.long, device=device)
    neg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if target_edges is None or target_labels is None:
        return pos, neg
    if target_edges.numel() == 0 or target_labels.numel() == 0:
        return pos, neg
    edges = target_edges.long().to(device)
    labels = _ensure_one_hot_labels(target_labels.to(device), num_classes=2)
    dst = edges[1].clamp(min=0, max=num_nodes - 1)
    pos.scatter_add_(0, dst, labels[:, 0].long())
    neg.scatter_add_(0, dst, labels[:, 1].long())
    return pos, neg


def _targeted_future_edge_attack(
    edges_long: torch.Tensor,
    original_labels: torch.Tensor,
    target_edges: Optional[torch.Tensor],
    target_labels: Optional[torch.Tensor],
    malicious_label_row: Optional[torch.Tensor],
    victim_percentage: float,
    attack_percentage: float,
    seed: Optional[int],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if target_edges is None or target_labels is None:
        return None, None
    if target_edges.numel() == 0 or target_labels.numel() == 0:
        return None, None

    target_edges = target_edges.long().to(device)
    target_labels_oh = _ensure_one_hot_labels(target_labels.to(device), num_classes=2)
    target_class = torch.argmax(target_labels_oh, dim=1)

    # Inject the same future edge early with a contradictory sign.  This is a
    # task-aligned poisoning attack for MCC recovery experiments.  When no
    # fixed malicious class is supplied, flip each target edge individually.
    if malicious_label_row is None:
        contradictory = torch.ones_like(target_class, dtype=torch.bool)
    else:
        malicious_class = int(torch.argmax(malicious_label_row.squeeze(0)).item())
        contradictory = target_class != malicious_class
    if contradictory.sum() == 0:
        return None, None

    existing = set((int(u), int(v)) for u, v in edges_long.t().detach().cpu().tolist())
    candidate_idx = []
    for idx, (u, v) in enumerate(target_edges.t().detach().cpu().tolist()):
        if not bool(contradictory[idx].item()):
            continue
        edge = (int(u), int(v))
        if edge in existing:
            continue
        existing.add(edge)
        candidate_idx.append(idx)

    if not candidate_idx:
        return None, None

    rng = random.Random(seed)
    rng.shuffle(candidate_idx)
    budget = max(1, int(len(candidate_idx) * victim_percentage * max(1.0, attack_percentage)))
    selected = candidate_idx[: min(len(candidate_idx), budget)]
    selected_idx = torch.tensor(selected, dtype=torch.long, device=device)
    malicious_edges = target_edges.index_select(1, selected_idx)
    if malicious_label_row is None:
        selected_targets = target_labels_oh.index_select(0, selected_idx)
        malicious_onehot = torch.stack([selected_targets[:, 1], selected_targets[:, 0]], dim=1)
    else:
        malicious_onehot = malicious_label_row.expand(malicious_edges.shape[1], -1)

    if original_labels.dim() == 1:
        malicious_labels = torch.argmax(malicious_onehot, dim=1).to(dtype=original_labels.dtype)
    elif original_labels.dim() == 2 and original_labels.shape[1] == 1:
        malicious_labels = torch.argmax(malicious_onehot, dim=1).unsqueeze(1).to(dtype=original_labels.dtype)
    else:
        malicious_labels = malicious_onehot.to(dtype=original_labels.dtype)

    updated_edges = torch.cat([edges_long, malicious_edges], dim=1)
    updated_labels = torch.cat([original_labels, malicious_labels], dim=0)
    return updated_edges, updated_labels


def _build_reverse_sparse_adj(
    edges_long: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    device = edges_long.device
    rev_indices = torch.stack([edges_long[0], edges_long[1]], dim=0)
    values = torch.ones(edges_long.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        rev_indices, values, size=(num_nodes, num_nodes), device=device
    ).coalesce()


def _bfs_distances_gpu(
    reverse_adj: torch.Tensor,
    source: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    distances[source] = 0
    visited = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    visited[source] = 1.0
    frontier = visited.clone()
    for depth in range(1, num_nodes):
        new_reach = torch.sparse.mm(reverse_adj, frontier.unsqueeze(1)).squeeze(1)
        new_mask = (new_reach > 0) & (visited == 0)
        if not new_mask.any():
            break
        distances[new_mask] = depth
        visited[new_mask] = 1.0
        frontier = new_mask.float()
    return distances


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def identify_good_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    if edges.numel() == 0:
        return []
    edges_long = edges.long()
    if num_nodes is None:
        num_nodes = int(edges_long.max().item()) + 1
    node_ratings = _compute_node_ratings_gpu(edges_long, labels, num_nodes)
    return torch.where(node_ratings[:, 0] > node_ratings[:, 1])[0].tolist()


def identify_bad_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    if edges.numel() == 0:
        return []
    edges_long = edges.long()
    if num_nodes is None:
        num_nodes = int(edges_long.max().item()) + 1
    node_ratings = _compute_node_ratings_gpu(edges_long, labels, num_nodes)
    return torch.where(node_ratings[:, 1] > node_ratings[:, 0])[0].tolist()


def initiate_attack(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_type: str = "bad_mouthing",
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
    snapshot_idx: int = 0,
    total_snapshots: int = 7,
    target_edges: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # ── Validation ────────────────────────────────────────────────────────────
    if not 0.0 <= attack_percentage <= 10.0:
        raise ValueError(f"attack_percentage must be >= 0.0 (you passed {attack_percentage})")
    if not 0.0 <= victim_percentage <= 1.0:
        raise ValueError(f"victim_percentage must be between 0.0 and 1.0, got {victim_percentage}")
    if original_edges.numel() == 0:
        raise ValueError("Empty edge tensor provided")
    if original_edges.shape[0] != 2:
        raise ValueError(f"Expected edges with shape (2, num_edges), got {original_edges.shape}")

    device = original_edges.device
    edges_long = original_edges.long()
    edges_long, original_labels = _deduplicate_edges_and_labels(edges_long, original_labels)
    num_nodes = int(edges_long.max().item()) + 1

    node_ratings = _compute_node_ratings_gpu(edges_long, original_labels, num_nodes)
    future_pos, future_neg = _future_target_signal(target_edges, target_labels, num_nodes, device)

    attack_type_normalized = attack_type
    if attack_type_normalized == "on_off":
        attack_type_normalized = "on-off"
    if attack_type_normalized not in {"on-off"}:
        attack_type_normalized = attack_type_normalized.replace("-", "_")

    if attack_type_normalized == "bad_mouthing":
        candidate_mask = future_pos > future_neg
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 0] > node_ratings[:, 1]   # good nodes
        malicious_label_row = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        attack_name = "Bad-Mouthing"
    elif attack_type_normalized in {"good_mouthing", "ballot_stuffing"}:
        candidate_mask = future_neg > future_pos
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 1] > node_ratings[:, 0]   # bad nodes
        malicious_label_row = torch.tensor([[1, 0]], dtype=torch.int64, device=device)
        attack_name = "Ballot-Stuffing" if attack_type_normalized == "ballot_stuffing" else "Good-Mouthing"
    elif attack_type_normalized == "sybil_boosting":
        candidate_mask = future_neg > future_pos
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 1] > node_ratings[:, 0]   # bad nodes
        malicious_label_row = torch.tensor([[1, 0]], dtype=torch.int64, device=device)
        attack_name = "Sybil-Boosting"
    elif attack_type_normalized == "on-off":
        if snapshot_idx % 2 == 1:
            return edges_long, original_labels
        candidate_mask = future_pos > future_neg
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 0] > node_ratings[:, 1]
        malicious_label_row = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        attack_name = "On-Off (Bad-Mouthing on)"
    elif attack_type_normalized == "slow_poisoning":
        if total_snapshots <= 1:
            progress = 1.0
        else:
            progress = (snapshot_idx + 1) / total_snapshots
        current_victim_pct = min(1.0, victim_percentage * (0.35 + progress))
        if current_victim_pct < 0.01:
            return edges_long, original_labels
        victim_percentage = current_victim_pct
        candidate_mask = (future_neg + future_pos) > 0
        if not candidate_mask.any():
            candidate_mask = (node_ratings[:, 0] + node_ratings[:, 1]) > 0
        malicious_label_row = None
        if target_edges is not None and target_labels is not None:
            attack_percentage = max(attack_percentage, 10.0)
        attack_name = "Slow-Poisoning (future contradiction drift)"
    elif attack_type_normalized == "camouflage":
        progress = (snapshot_idx + 1) / max(1, total_snapshots)
        candidate_mask = future_pos > future_neg
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 0] > node_ratings[:, 1]
        if progress < 0.50:
            malicious_label_row = torch.tensor([[1, 0]], dtype=torch.int64, device=device)
            attack_percentage = min(attack_percentage, 0.35)
            attack_name = "Camouflage warm-up"
        else:
            malicious_label_row = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
            attack_name = "Camouflage strike"
    elif attack_type_normalized in {"adaptive_stealth", "adaptive_gradient", "low_budget_stealth"}:
        candidate_mask = future_pos > future_neg
        if not candidate_mask.any():
            candidate_mask = node_ratings[:, 0] > node_ratings[:, 1]
        malicious_label_row = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        attack_percentage = min(attack_percentage, 0.40)
        attack_name = "Adaptive-Stealth"
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    candidate_victims_tensor = torch.where(candidate_mask)[0]

    if candidate_victims_tensor.numel() == 0:
        if verbose:
            print(f"No victim candidates for {attack_name}. Returning original graph.")
        return edges_long, original_labels

    targeted_edges, targeted_labels = _targeted_future_edge_attack(
        edges_long=edges_long,
        original_labels=original_labels,
        target_edges=target_edges,
        target_labels=target_labels,
        malicious_label_row=malicious_label_row,
        victim_percentage=victim_percentage,
        attack_percentage=attack_percentage,
        seed=seed,
        device=device,
    )
    if targeted_edges is not None:
        return targeted_edges, targeted_labels

    num_to_select = max(1, int(candidate_victims_tensor.numel() * victim_percentage))
    candidate_victims_list = candidate_victims_tensor.tolist()
    victim_nodes_list = random.sample(candidate_victims_list, k=num_to_select)
    victim_nodes_tensor = torch.tensor(victim_nodes_list, dtype=torch.long, device=device)

    src_nodes = edges_long[0]
    dst_nodes = edges_long[1]
    ones_e = torch.ones(edges_long.shape[1], dtype=torch.long, device=device)
    in_degrees = torch.zeros(num_nodes, dtype=torch.long, device=device).scatter_add_(0, dst_nodes, ones_e)
    out_degrees = torch.zeros(num_nodes, dtype=torch.long, device=device).scatter_add_(0, src_nodes, ones_e)
    total_degrees = in_degrees + out_degrees

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj_matrix[src_nodes, dst_nodes] = True

    reverse_adj = _build_reverse_sparse_adj(edges_long, num_nodes)

    all_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
    newly_added_edges_list: List[torch.Tensor] = []
    sybil_sources_list: List[torch.Tensor] = []
    total_added = 0

    if verbose:
        print(f"\n--- Generating Malicious Edges ({attack_name}) ---")
        print(f"Selected victims: {num_to_select} ({victim_percentage * 100:.1f}%)")
        print(f"Attack intensity: {attack_percentage * 100:.1f}% of victim's total-degree")

    for victim_idx in range(victim_nodes_tensor.numel()):
        victim = int(victim_nodes_tensor[victim_idx].item())
        v_tot = int(total_degrees[victim].item())
        if v_tot == 0:
            continue
        attack_volume = max(1, int(v_tot * attack_percentage))

        not_self = all_nodes != victim
        no_existing_edge = ~adj_matrix[:, victim]
        potential_mask = not_self & no_existing_edge
        potential_att = all_nodes[potential_mask]
        if potential_att.numel() == 0:
            potential_att = all_nodes[not_self]

        distances = _bfs_distances_gpu(reverse_adj, victim, num_nodes, device)

        potential_list = potential_att.tolist()
        cand_with_dist = [(n, int(distances[n].item())) for n in potential_list]
        reachable = [(n, d) for n, d in cand_with_dist if d != -1]
        unreachable = [(n, d) for n, d in cand_with_dist if d == -1]

        if reachable:
            reachable.sort(key=lambda x: x[1], reverse=True)
            random.shuffle(unreachable)
            combined = reachable + unreachable
            attackers_selected_list = [n for n, _ in combined[:attack_volume]]
        else:
            random.shuffle(unreachable)
            attackers_selected_list = [n for n, _ in unreachable[:attack_volume]]

        attackers_selected = torch.tensor(attackers_selected_list, dtype=torch.long, device=device)

        already_exist = adj_matrix[attackers_selected, victim]
        new_attackers = attackers_selected[~already_exist]
        n_added = new_attackers.numel()

        if n_added > 0:
            adj_matrix[new_attackers, victim] = True
            victim_col = torch.full((n_added,), victim, dtype=torch.long, device=device)
            newly_added_edges_list.append(torch.stack([new_attackers, victim_col], dim=0))
            if attack_type_normalized == "sybil_boosting":
                sybil_sources_list.append(new_attackers)
            total_added += n_added

    if attack_type_normalized == "sybil_boosting" and sybil_sources_list:
        sybil_sources = torch.unique(torch.cat(sybil_sources_list))
        max_sybil = min(int(sybil_sources.numel()), max(2, int(total_added ** 0.5) + 2))
        sybil_sources = sybil_sources[:max_sybil]
        collusive_edges = []
        for i in range(int(sybil_sources.numel())):
            u = int(sybil_sources[i].item())
            v = int(sybil_sources[(i + 1) % int(sybil_sources.numel())].item())
            if u == v or adj_matrix[u, v]:
                continue
            adj_matrix[u, v] = True
            collusive_edges.append([u, v])
        if collusive_edges:
            collusive_tensor = torch.tensor(collusive_edges, dtype=torch.long, device=device).t().contiguous()
            newly_added_edges_list.append(collusive_tensor)
            total_added += collusive_tensor.shape[1]

    if not newly_added_edges_list:
        if verbose:
            print(f"No new malicious edges generated for {attack_name}.")
        return edges_long, original_labels

    malicious_edges_tensor = torch.cat(newly_added_edges_list, dim=1)
    num_malicious_edges = malicious_edges_tensor.shape[1]

    orig_label_dtype = original_labels.dtype
    malicious_onehot = malicious_label_row.expand(num_malicious_edges, -1)

    if original_labels.dim() == 1:
        malicious_labels_tensor = torch.argmax(malicious_onehot, dim=1).to(dtype=orig_label_dtype)
    elif original_labels.dim() == 2 and original_labels.shape[1] == 1:
        malicious_labels_tensor = torch.argmax(malicious_onehot, dim=1).unsqueeze(1).to(dtype=orig_label_dtype)
    else:
        malicious_labels_tensor = malicious_onehot.to(dtype=orig_label_dtype)

    updated_edges = torch.cat([edges_long, malicious_edges_tensor], dim=1)
    updated_labels = torch.cat([original_labels, malicious_labels_tensor], dim=0)

    if verbose:
        print(f"\n--- {attack_name} Attack Summary ---")
        print(f"Original edges:      {original_edges.shape[1]}")
        print(f"Malicious edges added: {num_malicious_edges}")
        print(f"New total edges:     {updated_edges.shape[1]}")

    return updated_edges, updated_labels


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────

def initiate_bad_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="bad_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose, seed=seed,
    )


def initiate_good_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="good_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose, seed=seed,
    )


def calculate_victim_drift(
    current_time: int,
    total_time_steps: int,
    max_victim_percentage: float,
    mode: str = "linear",
) -> float:
    if total_time_steps <= 1:
        return max_victim_percentage
    progress = (current_time + 1) / total_time_steps
    if mode == "linear":
        return float(max_victim_percentage * progress)
    elif mode == "exponential":
        return float(max_victim_percentage * (progress ** 2))
    return float(max_victim_percentage)
