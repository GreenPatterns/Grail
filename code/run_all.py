"""
ReputationAttack Unified Runner
================================
Runs all experiments (E1-E10) and ablations (A1, A2, A5, A6).
Saves combined results to ReputationAttack/results/unified/
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ablations'))

from project_paths import RESULTS_DIR


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("Starting ReputationAttack unified evaluation...")
    t_start = time.time()

    results = {}

    # ── Experiments ──────────────────────────────────────────────
    log("\n\n===== E1: Gradient Validation =====")
    from experiments.run_e1_gradient_validation import main as run_e1
    run_e1()

    log("\n\n===== E2: GRAIL-GRAD vs GRAIL-FWD =====")
    from experiments.run_e2_grad_vs_fwd import main as run_e2
    run_e2()

    log("\n\n===== E3: Attack Effectiveness (B=1 & B=5) =====")
    from experiments.run_e3_attack_effectiveness import main as run_e3
    run_e3()

    log("\n\n===== E4: Budget Efficiency =====")
    from experiments.run_e4_budget_efficiency import main as run_e4
    run_e4()

    log("\n\n===== E5: Surrogate Transfer =====")
    from experiments.run_e5_transfer_blackbox import main as run_e5
    run_e5()

    log("\n\n===== E6: Stronger Model + Stealth =====")
    from experiments.run_e6_stronger_stealth import main as run_e6
    run_e6()

    log("\n\n===== E7: Influence Distribution =====")
    from experiments.run_e7_influence_distribution import main as run_e7
    run_e7()

    log("\n\n===== E8: Attack Edge Detection =====")
    from experiments.run_e8_detection import main as run_e8
    run_e8()

    log("\n\n===== E9-E10: Sparsity Ablation + Bootstrap CI =====")
    from experiments.run_e9_e10_diagnostics import main as run_e9_e10
    run_e9_e10()

    # ── Ablations ───────────────────────────────────────────────
    log("\n\n===== A1: Model Depth =====")
    from ablations.run_a1_model_depth import main as run_a1
    run_a1()

    log("\n\n===== A2: Temporal Backbone =====")
    from ablations.run_a2_temporal_backbone import main as run_a2
    run_a2()

    log("\n\n===== A5: Seed Stability =====")
    from ablations.run_a5_seed_stability import main as run_a5
    run_a5()

    log("\n\n===== A6: Cross-Validation =====")
    from ablations.run_a6_cross_validation import main as run_a6
    run_a6()

    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"All experiments complete. Total time: {elapsed / 60:.1f} minutes")
    log(f"Results saved to: {RESULTS_DIR / 'unified'}")
    log(f"Run summarize_results.py to view summary.")


if __name__ == "__main__":
    main()
