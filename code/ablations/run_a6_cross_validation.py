"""
A6: Forward-chaining cross-validation — 5-fold temporal CV with confidence intervals.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import gc, json, time, warnings
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import RESULTS_DIR
from mycode.mainz_protocol import build_mainz_args, train_mainz_snapshot
from mycode.utils import read_graph

warnings.filterwarnings('ignore')

RDIR = str(RESULTS_DIR / "unified" / "ablation")
os.makedirs(RDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def temporal_cv_fold(ds, train_slots, seed=42):
    args = build_mainz_args(
        dataset=ds, pred_snap='single', attack=False,
        atype='bad_mouthing', victim_percentage=0.0,
        homogeneous_edges=False, lambda_temp=0.01, epochs=50,
    )
    args.best_model_path = None
    args.train_time_slots = train_slots
    graph = read_graph(args)
    trainer, perf = train_mainz_snapshot(args, graph, seed=seed,
                                         startmsg=f"CV-{ds}-{train_slots}snaps")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return perf


def run_cross_validation(ds, n_folds=5, seeds=(42, 123)):
    log(f"\n{'=' * 50}\n  Cross-Validation — {ds}\n{'=' * 50}")

    all_folds = []
    for fold in range(1, n_folds + 1):
        train_slots = fold + 1
        for seed in seeds:
            log(f"  Fold {fold}/{n_folds}, seed {seed}: train_time_slots={train_slots}")
            result = temporal_cv_fold(ds, train_slots=train_slots, seed=seed)
            result['fold'] = fold
            result['seed'] = seed
            all_folds.append(result)
            log(f"    MCC={result['MCC']:.4f} AUC={result['AUC']:.4f} "
                f"BAcc={result['BAcc']:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

    metrics = ['MCC', 'AUC', 'BAcc', 'F1']
    aggregated = {}
    for m in metrics:
        values = [f[m] for f in all_folds]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        ci_95 = 1.96 * std / np.sqrt(len(values))
        aggregated[m] = {
            'mean': mean,
            'std': std,
            'ci_95_low': mean - ci_95,
            'ci_95_high': mean + ci_95,
            'values': values,
        }
        log(f"  {m}: {mean:.4f} ± {std:.4f} [95% CI: {mean - ci_95:.4f}, {mean + ci_95:.4f}]")

    per_fold = {}
    for fold in range(1, n_folds + 1):
        fold_data = [f for f in all_folds if f['fold'] == fold]
        per_fold[str(fold)] = {
            m: {
                'mean': float(np.mean([f[m] for f in fold_data])),
                'std': float(np.std([f[m] for f in fold_data], ddof=1)),
            }
            for m in metrics
        }

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    means = [aggregated[m]['mean'] for m in metrics]
    errors = [aggregated[m]['ci_95_high'] - aggregated[m]['mean'] for m in metrics]
    bars = ax.bar(x, means, yerr=errors, capsize=5,
                  color=['#E63946', '#457B9D', '#2A9D8F', '#F4A261'],
                  edgecolor='w')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score')
    ax.set_title(f'Temporal CV Performance — {ds} ({n_folds} folds × {len(seeds)} seeds)')
    ax.grid(True, axis='y', alpha=.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{RDIR}/cv_performance_{ds}.pdf')
    fig.savefig(f'{RDIR}/cv_performance_{ds}.png')
    plt.close()

    return {
        'n_folds': n_folds,
        'n_seeds': len(seeds),
        'n_total_runs': len(all_folds),
        'aggregated': aggregated,
        'per_fold': per_fold,
        'raw': all_folds,
    }


def main():
    results = {}
    for ds in ['otc', 'alpha']:
        results[ds] = run_cross_validation(ds, n_folds=5, seeds=(42, 123))

    out_path = RESULTS_DIR / "unified" / "ablation" / "a6_cross_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for ds, data in results.items():
        clean[ds] = {
            'n_folds': data['n_folds'],
            'n_seeds': data['n_seeds'],
            'n_total_runs': data['n_total_runs'],
            'aggregated': {
                m: {k: v for k, v in d.items() if k != 'values'}
                for m, d in data['aggregated'].items()
            },
            'per_fold': data['per_fold'],
        }
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
