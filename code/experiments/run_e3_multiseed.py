"""Multi-seed E3 utility comparison (for cross-seed robustness / CIs).

Re-runs the E3 attack-effectiveness comparison -- including the three SOTA
baselines (PRBCD, InfluenceEdit, NodeInjection) -- across several seeds, so the
hardened-null claim ("SOTA attacks do not beat the Expert heuristic") can be
shown to hold beyond the single seed-42 run. Per-seed results are saved to
results/unified/e3_multiseed.json.

Designed to run on a dedicated GPU concurrently with the main suite, e.g.:
    CUDA_VISIBLE_DEVICES=1 GRAIL_SEEDS=1,2,3 python experiments/run_e3_multiseed.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from project_paths import RESULTS_DIR
from experiment_common import (
    train_model, get_targets, run_E3_attack_effectiveness, log,
)
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors


def main():
    seeds = [int(s) for s in os.environ.get("GRAIL_SEEDS", "1,2,3").split(",")]
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    out = {}
    for seed in seeds:
        out[str(seed)] = {}
        for ds in datasets:
            log(f"[multiseed] seed={seed} ds={ds}")
            tr = train_model(ds, seed=seed)
            oracle = ReputationAttackOracle(
                tr["model"], tr["index_list"], tr["device"]
            )
            edges, labels = tr["edges"], tr["labels"]
            strata = get_targets(oracle, edges, labels, per_stratum=10)
            good_nodes = identify_good_nodes_from_tensors(
                edges, labels, num_nodes=oracle.num_nodes
            )
            out[str(seed)][ds] = {
                "B1": run_E3_attack_effectiveness(
                    oracle, edges, labels, strata, good_nodes, ds, budget=1
                ),
                "B5": run_E3_attack_effectiveness(
                    oracle, edges, labels, strata, good_nodes, ds, budget=5
                ),
            }

    out_path = RESULTS_DIR / "unified" / "e3_multiseed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
