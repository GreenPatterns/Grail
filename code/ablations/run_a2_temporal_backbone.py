"""
A2: Temporal backbone ablation — S3R vs. identity (no temporal processing).
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


def run_temporal_ablation(ds):
    log(f"\n{'=' * 50}\n  A2: Temporal Ablation — {ds}\n{'=' * 50}")
    results = {}

    full = train_and_evaluate(ds)
    strata = full['oracle'].select_stratified_targets(
        full['edges'], full['labels'], per_stratum=5
    )
    full_val = validate_oracle(full['oracle'], full['edges'],
                                full['labels'], strata)
    results['with_temporal'] = {
        'performance': full['performance'],
        'validation': full_val,
    }
    log(f"  With S3R: MCC={full['performance']['MCC']:.3f}")
    for sn, v in full_val.items():
        log(f"    {sn}: r={v['r']:.3f}±{v['r_std']:.3f}")

    class IdentityTemporal(torch.nn.Module):
        def __init__(self, input_dim, n_heads, max_seq_len, dropout):
            super().__init__()
            self.input_dim = input_dim
        def forward(self, x):
            return x

    trained = train_and_evaluate(ds, seed=123)
    orig_s3r = trained['model'].s3r
    trained['model'].s3r = IdentityTemporal(
        trained['model'].input_dim, trained['model'].num_heads,
        trained['model'].timeslots, trained['model'].dropout
    ).to(trained['device'])

    oracle_notemp = ReputationAttackOracle(
        trained['model'], trained['index_list'], trained['device']
    )
    no_val = validate_oracle(oracle_notemp, trained['edges'],
                              trained['labels'], strata)
    results['no_temporal'] = {
        'performance': {'MCC': 0.0, 'AUC': 0.0, 'BAcc': 0.0, 'F1': 0.0},
        'validation': no_val,
    }
    log(f"  Without S3R (identity):")
    for sn, v in no_val.items():
        log(f"    {sn}: r={v['r']:.3f}±{v['r_std']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    strata_names = [s for s in ['moderate', 'low'] if s in full_val]
    x = np.arange(len(strata_names))
    w = 0.35
    full_rs = [full_val[s]['r'] for s in strata_names]
    no_rs = [no_val.get(s, {}).get('r', 0) for s in strata_names]
    ax.bar(x - w / 2, full_rs, w, label='With S3R', color='#E63946', edgecolor='w')
    ax.bar(x + w / 2, no_rs, w, label='No Temporal', color='#A8DADC', edgecolor='w')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strata_names])
    ax.set_ylabel('Pearson r')
    ax.set_title(f'Temporal Ablation — {ds}')
    ax.legend()
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{FDIR}/ablation_temporal_{ds}.pdf')
    fig.savefig(f'{FDIR}/ablation_temporal_{ds}.png')
    plt.close()

    del trained, full
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    results = {ds: run_temporal_ablation(ds) for ds in ["otc", "alpha"]}
    out_path = RESULTS_DIR / "unified" / "ablation" / "a2_temporal_backbone.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
