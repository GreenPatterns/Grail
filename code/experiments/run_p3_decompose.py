"""
Phase 3 — decompose the processed-placement drop (is the GNN vulnerable, or is
the mean fragile?). For each victim we split the realized $\\Delta R$ into:
  averaging  = new raters added to the mean at FROZEN clean embeddings
               (label-independent dilution; equals the appended/no-op effect), and
  embedding  = change in the (augmented) rater set's trust-probabilities caused by
               re-embedding the labeled edges (the genuine model response).
A dominant embedding term proves the vulnerability is the GNN's response to the
processed label, not fragility of the unweighted average.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_decompose.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import json, gc
import numpy as np
import torch
from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
PER_STRATUM = 25
CAND = 60


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 3 DECOMPOSE — {ds}\n{'='*64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    rng = np.random.RandomState(SEED)
    strata = o.select_stratified_targets(edges, labels, per_stratum=PER_STRATUM)
    out = {}
    for stratum in ["low", "moderate"]:
        vics = strata.get(stratum, [])
        if not vics:
            continue
        for B in (1, 5):
            tot, avg, emb = [], [], []
            for v in vics:
                existing = set(edges[0, edges[1] == v].tolist())
                pool = [int(s) for s in rng.permutation(o.num_nodes)
                        if s != v and s not in existing][:CAND]
                sc = o.score_candidates_counterfactual_processed(edges, labels, v, pool, sign='distrust')
                opt = [s for s, _ in sorted(sc.items(), key=lambda x: x[1])][:B]
                dR, dRa, dRe = o.decompose_processed(edges, labels, opt, v, 'distrust')
                tot.append(dR); avg.append(dRa); emb.append(dRe)
            mt, ma, me = float(np.mean(tot)), float(np.mean(avg)), float(np.mean(emb))
            frac = me / mt if abs(mt) > 1e-9 else float('nan')
            out[f"{stratum}_B{B}"] = {"dR": mt, "dR_avg": ma, "dR_emb": me, "emb_frac": frac}
            log(f"  {stratum:>8} B={B}: ΔR={mt:+.4f} = avg {ma:+.4f} + emb {me:+.4f}  "
                f"(embedding {frac:.0%} of total)")
    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p3_decompose.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
