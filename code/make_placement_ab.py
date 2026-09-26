"""
Placement-only A/B (reviewer Q1).

For every (stratum, budget) cell the hardening sweep already scores the SAME
optimized counterfactual source set at two placements:
    dr_optimized      -> processed window (re-embedded)   [TM-B]
    dr_appended_paper -> appended past the window (paper)  [TM-A]
Identical targets, identical budget, identical sources -> placement is the
ONLY toggled variable. This isolates placement's contribution to the ~4x gap.

Usage: python code/make_placement_ab.py
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from project_paths import RESULTS_DIR
UNI = RESULTS_DIR / "unified"

FILES = ["p0d_otcalpha.json", "p0d_harden.json", "p0d_decayed.json"]


def main():
    print(f"{'dataset':<10}{'stratum':<10}{'B':>3}  {'ΔR processed':>13}  "
          f"{'ΔR appended':>12}  {'ratio':>7}")
    print("-" * 64)
    rows = []
    for fn in FILES:
        path = UNI / fn
        if not path.exists():
            continue
        d = json.load(open(path))
        tag = "" if fn == "p0d_otcalpha.json" else f"  [{fn.replace('p0d_','').replace('.json','')}]"
        for ds, cells in d.items():
            if not isinstance(cells, dict):
                continue
            for stratum in ("high", "moderate", "low"):
                for B in (1, 3, 5, 10):
                    e = cells.get(f"{stratum}_B{B}")
                    if not e:
                        continue
                    proc = e.get("dr_optimized")
                    app = e.get("dr_appended_paper")
                    if proc is None or app is None:
                        continue
                    ratio = (proc / app) if abs(app) > 1e-6 else float('inf')
                    rows.append((ds + tag, stratum, B, proc, app, ratio))
    for ds, stratum, B, proc, app, ratio in rows:
        rstr = "n/a" if ratio == float('inf') else f"{ratio:6.1f}x"
        print(f"{ds:<10}{stratum:<10}{B:>3}  {proc:>+13.4f}  {app:>+12.4f}  {rstr:>7}")

    print("\nHeadline (B=5, optimized): processed vs appended, placement-only.")
    for ds, stratum, B, proc, app, ratio in rows:
        if B == 5 and stratum in ("low", "moderate") and "decayed" not in ds and "epn" not in ds:
            print(f"  {ds} {stratum}: processed {proc:+.3f} vs appended {app:+.3f} "
                  f"(|ratio| {abs(ratio):.0f}x), sign-reversed")


if __name__ == "__main__":
    main()
