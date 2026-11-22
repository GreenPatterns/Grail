import gc
import copy
from typing import Any, Dict, Optional

import numpy as np
import torch

from mycode.args import parameter_parser
from mycode.gcn3 import GCNTrainer
from mycode.utils import read_graph


BEST_EPOCH_METRICS_IDX = [1, 2, 3, 6]
_ORACLE_RUN_CACHE: Dict[tuple, Dict[str, Any]] = {}
_ATTACKED_RUN_CACHE: Dict[tuple, Dict[str, Any]] = {}


def build_mainz_args(
    dataset: str,
    *,
    pred_snap: str = "single",
    attack: bool = False,
    atype: str = "bad_mouthing",
    victim_percentage: float = 0.0,
    homogeneous_edges: bool = False,
    lambda_temp: float = 0.01,
    epochs: int = 50,
    layers: Optional[list[int]] = None,
):
    """Create args using the same single-snapshot defaults as mainZ.py."""
    args = parameter_parser(
        dataset=dataset,
        pred_snap=pred_snap,
        attack=attack,
        atype=atype,
        victim_percentage=victim_percentage,
        homogeneous_edges=homogeneous_edges,
        lambda_temp=lambda_temp,
        epochs=epochs,
    )
    if layers is not None:
        args.layers = layers
    args.best_model_path = None
    return args


def best_epoch_from_logs(logs: Dict[str, Any]) -> Optional[list]:
    """Pick the best epoch exactly like mainZ.py."""
    best_epoch_data = None
    best_epoch_score = -1.0
    for epoch_log in logs["performance"][1:]:
        if any(epoch_log[i] is None for i in BEST_EPOCH_METRICS_IDX):
            continue
        current_score = sum(float(epoch_log[i]) for i in BEST_EPOCH_METRICS_IDX)
        if current_score > best_epoch_score:
            best_epoch_score = current_score
            best_epoch_data = list(epoch_log)
    return best_epoch_data


def perf_dict(epoch_log: Optional[list]) -> Dict[str, float]:
    if not epoch_log:
        return {
            "MCC": 0.0,
            "AUC": 0.0,
            "BAcc": 0.0,
            "AP": 0.0,
            "F1_Micro": 0.0,
            "F1_Macro": 0.0,
            "F1": 0.0,
            "epoch": -1,
        }
    return {
        "epoch": int(epoch_log[0]),
        "MCC": float(epoch_log[1]),
        "AUC": float(epoch_log[2]),
        "BAcc": float(epoch_log[3]),
        "AP": float(epoch_log[4]),
        "F1_Micro": float(epoch_log[5]),
        "F1_Macro": float(epoch_log[6]),
        "F1": float(epoch_log[5]),
    }


def train_mainz_snapshot(args, graph, *, seed: int = 42, startmsg: str = ""):
    """Train one mainZ-style snapshot and return trainer plus best-epoch metrics.

    The trainer gets a copy of args with args.seed = seed: DGTEN, the structural layer and
    setup_features all (re)seed from args.seed, so without it every seed built one model.
    """
    args = copy.deepcopy(args)
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = GCNTrainer(args, graph, use_GPU=True)
    trainer.setup_dataset()
    trainer.create_and_train_model(startmsg=startmsg or f"TrainSnapshots = {args.train_time_slots}")
    return trainer, perf_dict(best_epoch_from_logs(trainer.logs))


def clean_forward_sweep(
    dataset: str,
    *,
    seed: int = 42,
    epochs: int = 50,
    layers: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Run the clean mainZ.py forward-chaining sweep: train_time_slots 2..9."""
    args = build_mainz_args(
        dataset,
        pred_snap="single",
        attack=False,
        atype="bad_mouthing",
        victim_percentage=0.0,
        homogeneous_edges=False,
        lambda_temp=0.01,
        epochs=epochs,
        layers=layers,
    )
    graph = read_graph(args)
    per_step = []
    num_time_steps = args.time_slots - 2
    for t_slots in range(2, 2 + num_time_steps):
        step_args = copy.deepcopy(args)
        step_args.train_time_slots = t_slots
        trainer, perf = train_mainz_snapshot(
            step_args, graph, seed=seed, startmsg=f"FV-{dataset}-s{seed}-t{t_slots}"
        )
        perf["train_time_slots"] = t_slots
        per_step.append(perf)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    keys = ["MCC", "AUC", "BAcc", "AP", "F1_Micro", "F1_Macro", "F1"]
    avg = {k: float(np.mean([p[k] for p in per_step])) for k in keys}
    avg.update({f"{k}_std": float(np.std([p[k] for p in per_step])) for k in keys})
    avg["n_time_steps"] = len(per_step)
    avg["per_step"] = per_step
    return avg


def train_fixed_mainz_model(
    dataset: str,
    *,
    seed: int = 42,
    train_time_slots: int = 7,
    attack: bool = False,
    atype: str = "bad_mouthing",
    victim_percentage: float = 0.0,
    epochs: int = 50,
    layers: Optional[list[int]] = None,
    start_prefix: str = "ATK",
):
    """Train the fixed single-snapshot model used by mainZ.py attack mode."""
    args = build_mainz_args(
        dataset,
        pred_snap="single",
        attack=attack,
        atype=atype,
        victim_percentage=victim_percentage,
        homogeneous_edges=False,
        lambda_temp=0.01,
        epochs=epochs,
        layers=layers,
    )
    args.train_time_slots = train_time_slots
    graph = read_graph(args)
    trainer, perf = train_mainz_snapshot(
        args,
        graph,
        seed=seed,
        startmsg=f"{start_prefix}-{dataset}-s{seed}-t{train_time_slots}",
    )
    return trainer, perf


def train_oracle_mainz_model(
    dataset: str,
    *,
    seed: int = 42,
    epochs: int = 50,
    layers: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """
    Main paper setup: clean mainZ performance sweep plus fixed clean t=7 model
    for frozen-oracle/counterfactual experiments.
    """
    cache_key = (dataset, seed, epochs, tuple(layers) if layers is not None else None)
    if cache_key in _ORACLE_RUN_CACHE:
        return _ORACLE_RUN_CACHE[cache_key]

    perf = clean_forward_sweep(dataset, seed=seed, epochs=epochs, layers=layers)
    trainer, fixed_perf = train_fixed_mainz_model(
        dataset,
        seed=seed,
        train_time_slots=7,
        attack=False,
        atype="bad_mouthing",
        victim_percentage=0.0,
        epochs=epochs,
        layers=layers,
        start_prefix="ATK",
    )
    result = {
        "trainer": trainer,
        "model": trainer.model,
        "device": trainer.device,
        "edges": trainer.train_edges_final,
        "labels": trainer.train_labels_final,
        "index_list": trainer.index_list,
        "perf": perf,
        "performance": perf,
        "fixed_perf": fixed_perf,
    }
    _ORACLE_RUN_CACHE[cache_key] = result
    return result


def train_attacked_mainz_model(
    dataset: str,
    *,
    seed: int = 42,
    atype: str = "bad_mouthing",
    victim_percentage: float = 0.10,
    epochs: int = 50,
    layers: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Built-in GDTE attacked setup from mainZ.py: attack=True, one t=7 run."""
    cache_key = (
        dataset,
        seed,
        atype,
        float(victim_percentage),
        epochs,
        tuple(layers) if layers is not None else None,
    )
    if cache_key in _ATTACKED_RUN_CACHE:
        return _ATTACKED_RUN_CACHE[cache_key]

    trainer, perf = train_fixed_mainz_model(
        dataset,
        seed=seed,
        train_time_slots=7,
        attack=True,
        atype=atype,
        victim_percentage=victim_percentage,
        epochs=epochs,
        layers=layers,
        start_prefix=f"ATTACK-{atype}",
    )
    result = {
        "trainer": trainer,
        "model": trainer.model,
        "device": trainer.device,
        "edges": trainer.train_edges_final,
        "labels": trainer.train_labels_final,
        "index_list": trainer.index_list,
        "perf": perf,
        "performance": perf,
    }
    _ATTACKED_RUN_CACHE[cache_key] = result
    return result
