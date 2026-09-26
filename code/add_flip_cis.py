"""
Add exact (Clopper-Pearson) binomial 95% CIs to every flip-rate cell.

Flip rate is k/n with k=round(flip_rate * n_eligible) successes out of
n=n_eligible eligible (rb>=0.5) targets. The runners store the rate and
n_eligible, so the CI is recoverable post-hoc with no reruns.

Usage: python code/add_flip_cis.py
Writes <name>.flipci.json next to each input and prints headline cells.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from scipy.stats import beta

from project_paths import RESULTS_DIR
UNI = RESULTS_DIR / "unified"


def cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial CI for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (k / n, float(lo), float(hi))


def annotate_p0d(d):
    """p0d_*.json: {ds: {stratum_Bx: {flip_rate_*, n_eligible_flip, ...}}}."""
    headlines = []
    for ds, cells in d.items():
        if not isinstance(cells, dict):
            continue
        for cell, e in cells.items():
            n = int(e.get("n_eligible_flip", 0))
            for key in ("flip_rate_random", "flip_rate_optimized"):
                if key not in e:
                    continue
                k = round(e[key] * n)
                p, lo, hi = cp_ci(k, n)
                e[key + "_ci"] = [round(lo, 4), round(hi, 4)]
                e[key + "_k_n"] = [k, n]
                if key == "flip_rate_optimized":
                    headlines.append((ds, cell, k, n, p, lo, hi))
    return headlines


def annotate_p1(d):
    """p1_efficient.json: {ds: {flip_rate: {method: rate}, n_eligible}}."""
    headlines = []
    for ds, e in d.items():
        if not isinstance(e, dict) or "flip_rate" not in e:
            continue
        n = int(e.get("n_eligible", 0))
        ci = {}
        for method, rate in e["flip_rate"].items():
            k = round(rate * n)
            p, lo, hi = cp_ci(k, n)
            ci[method] = {"k_n": [k, n], "ci": [round(lo, 4), round(hi, 4)]}
            headlines.append((ds, method, k, n, p, lo, hi))
        e["flip_rate_ci"] = ci
    return headlines


def main():
    targets = (sorted(glob.glob(str(UNI / "p0d*.json")))
               + sorted(glob.glob(str(UNI / "p1_efficient*.json"))))
    print(f"{'file':<22}{'cell/method':<16}{'k/n':>8}{'rate':>8}   95% CI (Clopper-Pearson)")
    print("-" * 78)
    for path in targets:
        if path.endswith(".flipci.json"):
            continue
        with open(path) as f:
            d = json.load(f)
        name = os.path.basename(path)
        is_p1 = "p1_efficient" in name
        heads = annotate_p1(d) if is_p1 else annotate_p0d(d)
        out = path.replace(".json", ".flipci.json")
        with open(out, "w") as f:
            json.dump(d, f, indent=2)
        # print only the headline-relevant rows: B1/B5 low/moderate for p0d,
        # optimized/random/expert for p1
        for ds, cell, k, n, p, lo, hi in heads:
            show = is_p1 and cell in ("optimized", "random", "expert", "counterfactual") \
                or (not is_p1 and ("B1" in str(cell) or "B5" in str(cell)))
            if show:
                print(f"{name:<22}{ds+' '+str(cell):<16}{f'{k}/{n}':>8}{p:>8.0%}"
                      f"   [{lo:.0%}, {hi:.0%}]")
    print("\nWrote *.flipci.json alongside each input.")


if __name__ == "__main__":
    main()
