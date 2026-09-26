"""
Phase 3 — temporal-window sensitivity. Vary the processed-window length
$T_{\\text{train}}$ (with $K{=}10$ snapshots) and confirm the silent-no-op
signature and the corrected vulnerability are stable: appended placement stays
null and label-dead at every $T$, while a single processed distrust edge stays
damaging and label-sensitive.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_window.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))
import json, gc
import numpy as np
import torch
from project_paths import RESULTS_DIR
from experiment_common import get_targets, log
from mycode.mainz_protocol import train_fixed_mainz_model
from mycode.trust_influence import ReputationAttackOracle
from run_p0b_verify import insert_at_snapshot

SEED = 42
T_GRID = [int(x) for x in os.environ.get("GRAIL_TGRID", "5,6,7,8").split(",")]
DIS, TRU = [0., 1.], [1., 0.]


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 3 WINDOW SENSITIVITY — {ds}\n{'='*64}")
    out = {}
    for T in T_GRID:
        trainer, perf = train_fixed_mainz_model(ds, seed=SEED, train_time_slots=T,
                                                start_prefix=f"WIN-t{T}")
        o = ReputationAttackOracle(trainer.model, trainer.index_list, trainer.device)
        edges, labels = trainer.train_edges_final, trainer.train_labels_final
        strata = get_targets(o, edges, labels, per_stratum=8)
        targets = (strata.get("moderate", []) + strata.get("low", []))[:10]
        rng = np.random.RandomState(SEED)
        proc_dr, proc_ld, app_dr, app_ld = [], [], [], []
        for t in targets:
            existing = set(edges[0, edges[1] == t].tolist())
            src = next(s for s in rng.permutation(o.num_nodes)
                       if s != t and s not in existing)
            pd = insert_at_snapshot(o, edges, labels, t, src, T - 1, DIS)
            pt = insert_at_snapshot(o, edges, labels, t, src, T - 1, TRU)
            ad = insert_at_snapshot(o, edges, labels, t, src, "appended", DIS)
            at = insert_at_snapshot(o, edges, labels, t, src, "appended", TRU)
            proc_dr.append(pd); proc_ld.append(abs(pd - pt))
            app_dr.append(ad); app_ld.append(abs(ad - at))
        m = lambda x: float(np.mean(x))
        out[f"T{T}"] = {"auc": float(perf.get("AUC", float('nan'))),
                        "proc_dr": m(proc_dr), "proc_labeldelta": m(proc_ld),
                        "app_dr": m(app_dr), "app_labeldelta": m(app_ld)}
        log(f"  T={T} (AUC {perf.get('AUC',0):.2f}): processed ΔR={m(proc_dr):+.4f} "
            f"label-Δ={m(proc_ld):.4f} | appended ΔR={m(app_dr):+.4f} label-Δ={m(app_ld):.5f}")
        del trainer, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p3_window.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
