"""
A1: Model depth ablation — 1, 2, 3 GCN layers.
Evaluates how depth affects gradient oracle quality and attack effectiveness.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import gc, json, time, warnings
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import RESULTS_DIR
from mycode.mainz_protocol import train_oracle_mainz_model
from mycode.trust_influence import ReputationAttackOracle

warnings.filterwarnings('ignore')

RDIR = str(RESULTS_DIR / "unified" / "ablation")
FDIR = os.path.join(RDIR, "figures")
os.makedirs(FDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def train_and_evaluate(ds, seed=42, layers=None):
    trained = train_oracle_mainz_model(ds, seed=seed, layers=layers, epochs=50)
    oracle = ReputationAttackOracle(
        trained['trainer'].model, trained['trainer'].index_list, trained['trainer'].device
    )
    return {
        'model': trained['trainer'].model,
        'oracle': oracle,
        'edges': trained['trainer'].train_edges_final,
        'labels': trained['trainer'].train_labels_final,
        'device': trained['trainer'].device,
        'index_list': trained['trainer'].index_list,
        'performance': trained['performance'],
    }


def validate_oracle(oracle, edges, labels, strata):
    results = {}
    for sn, targets in strata.items():
        if not targets:
            continue
        rs = []
        for v in targets[:5]:
            r = oracle.validate_influence_map(edges, labels, v, top_n=50)
            rs.append(r['pearson_r'])
        results[sn] = {
            'r': float(np.mean(rs)),
            'r_std': float(np.std(rs)),
        }
    return results


def run_depth_ablation(ds):
    log(f"\n{'=' * 50}\n  A1: Depth Ablation — {ds}\n{'=' * 50}")
    configs = [
        (1, [32, 32]),
        (2, [32, 64, 32]),
        (3, [32, 64, 64, 32]),
    ]
    results = {}
    for n_layers, layers in configs:
        log(f"  {n_layers} layer(s): {layers}")
        trained = train_and_evaluate(ds, layers=layers)
        strata = trained['oracle'].select_stratified_targets(
            trained['edges'], trained['labels'], per_stratum=5
        )
        val = validate_oracle(trained['oracle'], trained['edges'],
                              trained['labels'], strata)
        results[str(n_layers)] = {
            'layers': layers,
            'performance': trained['performance'],
            'validation': val,
        }
        log(f"    Perf: MCC={trained['performance']['MCC']:.3f}, AUC={trained['performance']['AUC']:.3f}")
        for sn, v in val.items():
            log(f"    {sn}: r={v['r']:.3f}±{v['r_std']:.3f}")
        del trained
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    results = {ds: run_depth_ablation(ds) for ds in ["otc", "alpha"]}
    out_path = RESULTS_DIR / "unified" / "ablation" / "a1_model_depth.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
