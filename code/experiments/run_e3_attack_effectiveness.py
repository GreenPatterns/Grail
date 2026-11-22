import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json
import gc
import torch

from project_paths import RESULTS_DIR
from experiment_common import run_E3_attack_effectiveness, train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors


def main():
    results = {}
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha,epn").split(","):
        trained = train_model(ds)
        oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
        edges, labels = trained["edges"], trained["labels"]
        strata = get_targets(oracle, edges, labels, per_stratum=10)
        good_nodes = identify_good_nodes_from_tensors(edges, labels, num_nodes=oracle.num_nodes)
        results[ds] = {
            "B1": run_E3_attack_effectiveness(oracle, edges, labels, strata, good_nodes, ds, budget=1),
            "B5": run_E3_attack_effectiveness(oracle, edges, labels, strata, good_nodes, ds, budget=5),
        }
        del trained, oracle
        gc.collect()
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "unified" / "e3_attack_effectiveness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
