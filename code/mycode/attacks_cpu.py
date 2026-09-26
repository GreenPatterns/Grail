import random
import os
from typing import List, Tuple, Optional

import networkx as nx
import torch

os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"


def _ensure_one_hot_labels(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    if labels is None:
        raise ValueError("labels is None")

    if labels.dim() == 1:
        idx = labels.long()
        one_hot = torch.nn.functional.one_hot(idx, num_classes=num_classes)
        return one_hot.to(dtype=torch.int64, device=labels.device)
    if labels.dim() == 2 and labels.shape[1] == 1:
        idx = labels.squeeze(1).long()
        one_hot = torch.nn.functional.one_hot(idx, num_classes=num_classes)
        return one_hot.to(dtype=torch.int64, device=labels.device)
    if labels.dim() == 2 and labels.shape[1] == num_classes:
        idx = torch.argmax(labels, dim=1).long()
        one_hot = torch.nn.functional.one_hot(idx, num_classes=num_classes)
        return one_hot.to(dtype=torch.int64, device=labels.device)
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


def identify_good_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    if edges.numel() == 0:
        return []

    if num_nodes is None:
        num_nodes = int(edges.max().cpu().item()) + 1

    edges_cpu = edges.cpu().long()
    labels_one_hot = _ensure_one_hot_labels(labels.cpu(), num_classes=2)

    if edges_cpu.min().item() < 0 or edges_cpu.max().item() >= num_nodes:
        raise ValueError(f"Invalid edge indices: min={int(edges_cpu.min().item())}, max={int(edges_cpu.max().item())}, num_nodes={num_nodes}")

    node_ratings = torch.zeros((num_nodes, 2), dtype=torch.int64, device=edges_cpu.device)
    destination_nodes = edges_cpu[1]
    node_ratings.scatter_add_(0, destination_nodes.unsqueeze(1).expand(-1, 2),
                              labels_one_hot.to(dtype=node_ratings.dtype))

    if verbose:
        print(f"--- Identifying Good Nodes (Inferred {num_nodes} nodes) ---")
        for i in range(num_nodes):
            pos_count = int(node_ratings[i, 0].item())
            neg_count = int(node_ratings[i, 1].item())
            status = "Good" if pos_count > neg_count else "Not Good"
            print(f"Node {i}: Received {pos_count} Trust, {neg_count} Distrust. -> {status}")
        print("-" * 55 + "\n")

    good_nodes_mask = node_ratings[:, 0] > node_ratings[:, 1]
    good_nodes_indices = torch.where(good_nodes_mask)[0]
    return good_nodes_indices.tolist()


def identify_bad_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    if edges.numel() == 0:
        return []

    if num_nodes is None:
        num_nodes = int(edges.max().cpu().item()) + 1

    edges_cpu = edges.cpu().long()
    labels_one_hot = _ensure_one_hot_labels(labels.cpu(), num_classes=2)

    if edges_cpu.min().item() < 0 or edges_cpu.max().item() >= num_nodes:
        raise ValueError(f"Invalid edge indices: min={int(edges_cpu.min().item())}, max={int(edges_cpu.max().item())}, num_nodes={num_nodes}")

    node_ratings = torch.zeros((num_nodes, 2), dtype=torch.int64, device=edges_cpu.device)
    destination_nodes = edges_cpu[1]
    node_ratings.scatter_add_(0, destination_nodes.unsqueeze(1).expand(-1, 2),
                              labels_one_hot.to(dtype=node_ratings.dtype))

    if verbose:
        print(f"--- Identifying Bad Nodes (Inferred {num_nodes} nodes) ---")
        for i in range(num_nodes):
            pos_count = int(node_ratings[i, 0].item())
            neg_count = int(node_ratings[i, 1].item())
            status = "Bad" if neg_count > pos_count else "Not Bad"
            print(f"Node {i}: Received {pos_count} Trust, {neg_count} Distrust. -> {status}")
        print("-" * 55 + "\n")

    bad_nodes_mask = node_ratings[:, 1] > node_ratings[:, 0]
    bad_nodes_indices = torch.where(bad_nodes_mask)[0]
    return bad_nodes_indices.tolist()


def initiate_attack(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_type: str = "bad_mouthing",
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if not 0.0 <= attack_percentage <= 10.0:
        raise ValueError(f"attack_percentage must be >= 0.0 (you passed {attack_percentage})")
    if not 0.0 <= victim_percentage <= 1.0:
        raise ValueError(f"victim_percentage must be between 0.0 and 1.0, got {victim_percentage}")
    if original_edges.numel() == 0:
        raise ValueError("Empty edge tensor provided")
    if original_edges.shape[0] != 2:
        raise ValueError(f"Expected edges with shape (2, num_edges), got {original_edges.shape}")

    device = original_edges.device
    orig_edges_cpu = original_edges.cpu().long()
    orig_labels_cpu = original_labels.cpu()
    orig_edges_cpu, orig_labels_cpu = _deduplicate_edges_and_labels(orig_edges_cpu, orig_labels_cpu)
    num_nodes = int(orig_edges_cpu.max().item()) + 1

    if attack_type == "bad_mouthing":
        candidate_victims = identify_good_nodes_from_tensors(orig_edges_cpu, orig_labels_cpu, num_nodes=num_nodes, verbose=verbose)
        malicious_label = [0, 1]
        attack_name = "Bad-Mouthing"
    elif attack_type == "good_mouthing":
        candidate_victims = identify_bad_nodes_from_tensors(orig_edges_cpu, orig_labels_cpu, num_nodes=num_nodes, verbose=verbose)
        malicious_label = [1, 0]
        attack_name = "Good-Mouthing"
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. Use 'bad_mouthing' or 'good_mouthing'")

    if not candidate_victims:
        if verbose:
            print(f"No victim candidates identified for {attack_name}. Returning original graph.")
        return orig_edges_cpu.to(device), orig_labels_cpu.to(device)

    num_to_select = max(1, int(len(candidate_victims) * victim_percentage))
    victim_nodes = random.sample(candidate_victims, k=num_to_select)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([tuple(e) for e in orig_edges_cpu.t().tolist()])
    existing_edges = set(tuple(e) for e in orig_edges_cpu.t().tolist())
    newly_added_edges: List[List[int]] = []

    if verbose:
        print(f"\n--- Generating Malicious Edges ({attack_name}) ---")
        print(f"Total candidate nodes: {len(candidate_victims)}")
        print(f"Selected victims: {len(victim_nodes)} ({victim_percentage*100:.1f}%)")
        print(f"Attack intensity: {attack_percentage*100:.1f}% of victim's in-degree")
        print(f"Victim nodes: {victim_nodes}")

    total_attempted_edges = 0
    total_added = 0

    for victim in victim_nodes:
        victim_in_degree = G.in_degree(victim)
        victim_out_degree = G.out_degree(victim)
        victim_total_degree = victim_in_degree + victim_out_degree

        if victim_total_degree == 0:
            if verbose:
                print(f"Skipping victim {victim}: total degree is zero.")
            continue

        attack_volume = max(1, int(victim_total_degree * attack_percentage))

        potential_attackers = [
            node for node in range(num_nodes)
            if node != victim and (node, victim) not in existing_edges
        ]

        if not potential_attackers:
            potential_attackers = [node for node in range(num_nodes) if node != victim]

        G_reverse = G.reverse()

        try:
            distances = dict(nx.shortest_path_length(G_reverse, source=victim))

            cand_with_dist = [(n, distances.get(n, -1)) for n in potential_attackers]
            reachable = [(n, d) for n, d in cand_with_dist if d != -1]
            unreachable = [(n, d) for n, d in cand_with_dist if d == -1]

            if reachable:
                reachable.sort(key=lambda x: x[1], reverse=True)
                random.shuffle(unreachable)
                combined = reachable + unreachable
                attackers_selected = [n for n, _ in combined[:attack_volume]]
            else:
                random.shuffle(unreachable)
                attackers_selected = [n for n, _ in unreachable[:attack_volume]]

        except nx.NetworkXError:
            attackers_selected = random.sample(
                potential_attackers,
                min(attack_volume, len(potential_attackers))
            )

        total_attempted_edges += attack_volume

        edges_added_for_victim = 0
        for attacker in attackers_selected:
            edge = (attacker, victim)
            if edge not in existing_edges:
                newly_added_edges.append([attacker, victim])
                existing_edges.add(edge)
                edges_added_for_victim += 1
                total_added += 1

        if verbose:
            print(
                f" -> Victim {victim}: in-degree={victim_in_degree}, "
                f"out-degree={victim_out_degree}, total-degree={victim_total_degree}, "
                f"attack_volume={attack_volume}, added={edges_added_for_victim}"
        )

    if not newly_added_edges:
        if verbose:
            print(f"\nNo new malicious edges were generated for {attack_name}.")
        return orig_edges_cpu.to(device), orig_labels_cpu.to(device)

    malicious_edges_tensor = torch.tensor(newly_added_edges, dtype=orig_edges_cpu.dtype).t()
    num_malicious_edges = malicious_edges_tensor.shape[1]
    orig_label_dtype = orig_labels_cpu.dtype
    malicious_labels_onehot = torch.tensor([malicious_label] * num_malicious_edges, dtype=torch.int64)

    if orig_labels_cpu.dim() == 1:
        malicious_labels_tensor = torch.argmax(malicious_labels_onehot, dim=1).to(dtype=orig_label_dtype)
    elif orig_labels_cpu.dim() == 2 and orig_labels_cpu.shape[1] == 1:
        malicious_labels_tensor = torch.argmax(malicious_labels_onehot, dim=1).unsqueeze(1).to(dtype=orig_label_dtype)
    else:
        malicious_labels_tensor = malicious_labels_onehot.to(dtype=orig_label_dtype)

    updated_edges = torch.cat([orig_edges_cpu, malicious_edges_tensor], dim=1).to(device)
    updated_labels = torch.cat([orig_labels_cpu, malicious_labels_tensor], dim=0).to(device)

    if verbose:
        print(f"\n--- {attack_name} Attack Summary ---")
        print(f"Original edges: {original_edges.shape[1]}")
        print(f"Malicious edges added: {num_malicious_edges}")
        print(f"New total edges: {updated_edges.shape[1]}")
        print(f"Graph size increase: {(num_malicious_edges/original_edges.shape[1])*100:.2f}%")
        print("-" * 40)

    return updated_edges, updated_labels


def initiate_bad_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="bad_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose,
        seed=seed
    )


def initiate_good_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="good_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose,
        seed=seed
    )


def calculate_victim_drift(
    current_time: int,
    total_time_steps: int,
    max_victim_percentage: float,
    mode: str = 'linear'
) -> float:
    if total_time_steps <= 1:
        return max_victim_percentage
    progress = (current_time + 1) / total_time_steps

    if mode == 'linear':
        current_coverage = max_victim_percentage * progress
    elif mode == 'exponential':
        current_coverage = max_victim_percentage * (progress ** 2)
    else:
        current_coverage = max_victim_percentage

    return float(current_coverage)