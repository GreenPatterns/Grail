"""Emit the authoritative tab:flip cells (ΔR / flip% [CP-CI]) from current JSON."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from scipy.stats import beta
from project_paths import RESULTS_DIR
UNI = RESULTS_DIR / "unified"


def ci(k, n):
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return (lo, hi)


SRC = {"otc": ("p0d_otcalpha.json", "otc"),
       "alpha": ("p0d_otcalpha.json", "alpha"),
       "epn": ("p0d_harden.json", "epn30000")}


def main():
    for label, (fn, key) in SRC.items():
        d = json.load(open(UNI / fn))[key]
        print(f"\n### {label}")
        for stratum in ("high", "moderate", "low"):
            cells = []
            for B in (1, 3, 5, 10):
                e = d.get(f"{stratum}_B{B}")
                if not e:
                    cells.append("--"); continue
                n = e["n_eligible_flip"]; fr = e["flip_rate_optimized"]
                k = round(fr * n); lo, hi = ci(k, n)
                cells.append(f"${e['dr_optimized']:.2f}$ / {fr*100:.0f}"
                             f" [{lo*100:.0f},{hi*100:.0f}]")
            print(f"  {stratum:>8}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
