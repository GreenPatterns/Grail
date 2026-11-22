import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json
import gc
import torch

from project_paths import RESULTS_DIR
from experiment_common import run_E2_grad_vs_fwd, train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle


def main():
    results = {}
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha,epn").split(","):
        trained = train_model(ds)
        oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
        results[ds] = run_E2_grad_vs_fwd(oracle, trained["edges"], trained["labels"], ds)
        del trained, oracle
        gc.collect()
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "unified" / "e2_grad_vs_fwd.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
