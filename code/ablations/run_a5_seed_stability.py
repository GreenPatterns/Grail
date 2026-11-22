"""
A5: Seed stability — 5 different random seeds.
Evaluates gradient oracle quality consistency across training seeds.
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


def run_seed_stability(ds, n_seeds=5):
    log(f"\n{'=' * 50}\n  A5: Seed Stability — {ds}\n{'=' * 50}")
    seeds = [42, 123, 456, 789, 101112][:n_seeds]
    results = {}
    for seed in seeds:
        trained = train_and_evaluate(ds, seed=seed)
        strata = trained['oracle'].select_stratified_targets(
            trained['edges'], trained['labels'], per_stratum=5
        )
        val = validate_oracle(trained['oracle'], trained['edges'],
                              trained['labels'], strata)
        results[str(seed)] = {
            'performance': trained['performance'],
            'validation': val,
        }
        log(f"  Seed {seed}: MCC={trained['performance']['MCC']:.3f}")
        for sn, v in val.items():
            log(f"    {sn}: r={v['r']:.3f}±{v['r_std']:.3f}")
        del trained
        gc.collect()
        torch.cuda.empty_cache()

    agg = {}
    for sn in ['moderate', 'low']:
        rs = [results[str(s)]['validation'].get(sn, {}).get('r', 0) for s in seeds]
        agg[sn] = {
            'r_mean': float(np.mean(rs)),
            'r_std': float(np.std(rs)),
            'r_min': float(np.min(rs)),
            'r_max': float(np.max(rs)),
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    strata_names = [s for s in ['moderate', 'low'] if s in agg]
    data = []
    for sn in strata_names:
        data.append([results[str(s)]['validation'].get(sn, {}).get('r', 0) for s in seeds])
    bp = ax.boxplot(data, labels=[s.capitalize() for s in strata_names], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#E63946', '#457B9D']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Pearson r across seeds')
    ax.set_title(f'Seed Stability — {ds} ({n_seeds} seeds)')
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{FDIR}/ablation_seed_{ds}.pdf')
    fig.savefig(f'{FDIR}/ablation_seed_{ds}.png')
    plt.close()

    return {'per_seed': results, 'aggregated': agg}


def main():
    results = {ds: run_seed_stability(ds, n_seeds=5) for ds in ["otc", "alpha"]}
    out_path = RESULTS_DIR / "unified" / "ablation" / "a5_seed_stability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
