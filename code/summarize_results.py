"""
ReputationAttack Results Summarizer
====================================
Reads all JSON results from the unified results directory and prints
a formatted summary table.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))

import json
from pathlib import Path
import numpy as np

from project_paths import RESULTS_DIR

BASE = RESULTS_DIR / "unified"


def load(name):
    path = BASE / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def main():
    print("ReputationAttack Unified Results Summary")
    print("=" * 50)

    for ds in ["otc", "alpha"]:
        print(f"\n--- {ds.upper()} ---")

        # E1: Gradient Validation
        e1 = load("e1_gradient_validation")
        if e1 and ds in e1:
            print("\nE1 gradient validation:")
            for stratum in ["high", "moderate", "low"]:
                row = e1[ds].get(stratum)
                if row:
                    print(f"  {stratum}: r={fmt(row['r'])}±{fmt(row['r_std'])} "
                          f"R@10={fmt(row['recall'], 2)} n={row['n']}")

        # E2: GRAIL-GRAD vs GRAIL-FWD
        e2 = load("e2_grad_vs_fwd")
        if e2 and ds in e2:
            print("\nE2 gradient shortcut vs forward scoring:")
            for c in ["C50", "C100"]:
                row = e2[ds].get(c)
                if row:
                    print(f"  {c}: Grad ΔR={row['grad_dr_mean']:+.4f}, "
                          f"Fwd ΔR={row['fwd_dr_mean']:+.4f}, "
                          f"overlap={row['overlap_mean']:.0%}, speedup={row['speedup']:.1f}x")

        # E3: Attack Effectiveness
        e3 = load("e3_attack_effectiveness")
        if e3 and ds in e3:
            print("\nE3 attack effectiveness:")
            for budget in ["B1", "B5"]:
                for stratum in ["moderate", "low"]:
                    key = f"{budget}_{stratum}"
                    row = e3[ds].get(key)
                    if row:
                        ai = row.get("AI Oracle", {}).get("mean", 0)
                        expert = row.get("Expert", {}).get("mean", 0)
                        degree = row.get("Degree", {}).get("mean", 0)
                        rand = row.get("Random", {}).get("mean", 0)
                        print(f"  {key}: AI={ai:+.4f} Expert={expert:+.4f} "
                              f"Degree={degree:+.4f} Random={rand:+.4f}")

        # E4: Budget Efficiency
        e4 = load("e4_budget_efficiency")
        if e4 and ds in e4:
            print("\nE4 budget efficiency:")
            for method in ["AI Oracle", "Degree", "PageRank"]:
                parts = []
                for b, row in e4[ds].get(method, {}).items():
                    parts.append(f"B{b} {row['mean']:+.4f}")
                print(f"  {method}: " + ", ".join(parts))

        # E5: Transfer
        e5 = load("e5_transfer_blackbox")
        if e5 and ds in e5:
            row = e5[ds]
            print(f"\nE5 transfer: WB={row.get('wb_mean', 0):+.4f} "
                  f"BB={row.get('bb_mean', 0):+.4f} "
                  f"ratio={row.get('transfer_ratio', 0):.3f}")

        # E6: Stronger + Stealth
        e6 = load("e6_stronger_stealth")
        if e6 and ds in e6:
            stronger = e6[ds].get("stronger_model", {})
            stealth = e6[ds].get("degree_matched_stealth", {})
            print(f"\nE6 robustness: "
                  f"deeper AUC={stronger.get('deeper_auc', 0):.3f}, "
                  f"deeper ΔR={stronger.get('deeper_dr', 0):+.4f}")
            for b in ["B1", "B3", "B5"]:
                s = stealth.get(b, {})
                if s:
                    print(f"  {b}: Unconst={s.get('unconstrained', 0):+.4f}, "
                          f"DM={s.get('degree_matched', 0):+.4f}")

        # E7: Influence Distribution
        e7 = load("e7_influence_distribution")
        if e7 and ds in e7:
            row = e7[ds]
            print(f"\nE7 influence: Gini={row.get('gini', 0):.3f} "
                  f"deg_corr={row.get('degree_corr', 0):+.3f} "
                  f"top5={row.get('top5_share', 0):.3f}")

        # E8: Detection
        e8 = load("e8_detection")
        if e8 and ds in e8:
            row = e8[ds]
            print(f"\nE8 detection: AUC={row.get('auc', 0):.3f} "
                  f"Acc={row.get('accuracy', 0):.3f}")

        # E9-E10: Diagnostics
        e9e10 = load("e9_e10_diagnostics")
        if e9e10 and ds in e9e10:
            bc = e9e10[ds].get("bootstrap_ci", {})
            if bc:
                print("\nE10 bootstrap CI:")
                for b in ["B1", "B3", "B5"]:
                    row = bc.get(b, {})
                    if row:
                        print(f"  {b}: {row.get('mean', 0):+.4f} "
                              f"[{row.get('ci_lo', 0):+.4f}, {row.get('ci_hi', 0):+.4f}]")

    # ── Ablations ───────────────────────────────────────────────
    print("\n\n--- ABLATIONS ---")

    a1 = load("ablation/a1_model_depth")
    if a1:
        print("\nA1 model depth:")
        for ds in ["otc", "alpha"]:
            if ds in a1:
                for depth, row in a1[ds].items():
                    val = row.get("validation", {})
                    mcc = row.get("performance", {}).get("MCC", 0)
                    bits = ", ".join(f"{sn}={v['r']:.3f}" for sn, v in val.items())
                    print(f"  {ds} depth={depth} MCC={mcc:.3f} {bits}")

    a2 = load("ablation/a2_temporal_backbone")
    if a2:
        print("\nA2 temporal backbone:")
        for ds in ["otc", "alpha"]:
            if ds in a2:
                for name, row in a2[ds].items():
                    val = row.get("validation", {})
                    bits = ", ".join(f"{sn}={v['r']:.3f}" for sn, v in val.items())
                    print(f"  {ds} {name}: {bits}")

    a5 = load("ablation/a5_seed_stability")
    if a5:
        print("\nA5 seed stability:")
        for ds in ["otc", "alpha"]:
            if ds in a5:
                agg = a5[ds].get("aggregated", {})
                bits = ", ".join(
                    f"{sn}={v['r_mean']:.3f}±{v['r_std']:.3f}" for sn, v in agg.items()
                )
                print(f"  {ds} {bits}")

    a6 = load("ablation/a6_cross_validation")
    if a6:
        print("\nA6 cross-validation:")
        for ds in ["otc", "alpha"]:
            if ds in a6:
                agg = a6[ds].get("aggregated", {})
                print(f"  {ds} MCC={agg.get('MCC', {}).get('mean', 0):.3f}"
                      f"±{agg.get('MCC', {}).get('std', 0):.3f} "
                      f"AUC={agg.get('AUC', {}).get('mean', 0):.3f}"
                      f"±{agg.get('AUC', {}).get('std', 0):.3f} "
                      f"BAcc={agg.get('BAcc', {}).get('mean', 0):.3f}"
                      f"±{agg.get('BAcc', {}).get('std', 0):.3f}")

    print(f"\nResults directory: {BASE.resolve()}")


if __name__ == "__main__":
    main()
