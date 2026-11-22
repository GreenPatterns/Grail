"""
Prints the tables of the TrustGuard case study from results/trustguard_casestudy_*.json (written by
run_trustguard_casestudy.py): (1) reproduction of TrustGuard's Fig. 3 MCC next to the published bars,
(2) the poisoning checks (label swap, reachability, targets' reputation), (3) the single-edge evasion test
(flip rate and shift under R_tot and R_prop). Values are mean +- SD over seeds; pooled rates carry exact
95% intervals.

Run: python code/experiments/report_trustguard_casestudy.py
"""
import glob
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')


def ms(x, scale=1.0, nd=3):
    if not x:
        return '--'
    sd = '' if x['sd'] is None else f"+-{scale * x['sd']:.{nd}f}"
    return f"{scale * x['mean']:.{nd}f}{sd}"


def rate(b):
    if not b or not b['pooled_n']:
        return '--'
    lo, hi = b['pooled_cp95_pct']
    return f"{ms(b['rate_pct'], 1, 0)} ({b['pooled_k']}/{b['pooled_n']}, [{lo:.0f},{hi:.0f}])"


def main():
    for path in sorted(glob.glob(os.path.join(RES, 'trustguard_casestudy_*.json'))):
        d = json.load(open(path))
        c, s = d['config'], d.get('summary', {})
        tag = f"{d['dataset']} features={c.get('features')} degree={c.get('attack_degree', 'train')}"
        print(f'\n## {tag}  ({os.path.basename(path)})')
        att = d['attack']
        print('attack edges per seed:', {k: v['n_attack_edges'] for k, v in att.items()},
              'targets:', {k: v['n_targets'] for k, v in att.items()})
        print('published MCC (Fig. 3):', d['published_fig3_single_timeslot_MCC'])
        for agg, a in s.items():
            r = a['repro']
            print(f"\n[{agg}] seeds {r['seeds_done']}")
            print('  MCC best epoch:', {k[9:]: ms(v) for k, v in r.items() if k.startswith('MCC_best_')})
            print('  MCC last epoch:', {k[9:]: ms(v) for k, v in r.items() if k.startswith('MCC_last_')})
            print('  dMCC best vs clean:', {k[10:]: ms(v) for k, v in r.items() if k.startswith('dMCC_best_')})
            print('  Lambda_MCC best:', {k[16:]: ms(v) for k, v in r.items() if k.startswith('Lambda_MCC_best_')},
                  ' Lambda_R targets:', {k[16:]: ms(v) for k, v in r.items() if k.startswith('Lambda_R_targets_')})
            for cond in ('inwin_dis', 'inwin_tru', 'app_dis', 'app_tru', 'graph_dis', 'graph_tru'):
                t = r.get(f'targets_{cond}')
                if not t:
                    continue
                print(f"  targets {cond}: flip prop {rate(t['flip_prop'])} tot {rate(t['flip_tot'])}; "
                      f"shift prop {ms(t['shift_prop'])} tot {ms(t['shift_tot'])}; reach |dz| {ms(t['reach_dz_mean'])}; "
                      f"coef L1/L2/L3 in {[ms(t['coef'].get(f'L{i}_in')) for i in (1, 2, 3)]} "
                      f"pruned share in L1/L2/L3: injected {[ms(t['coef_share_zero'].get(f'L{i}_in')) for i in (1, 2, 3)]} "
                      f"organic {[ms(t.get('coef_organic_share_zero', {}).get(f'L{i}_in')) for i in (1, 2, 3)]}")
            b = a.get('evasion', {})
            if b and b.get('n_low_evaluated'):
                print(f"  evasion: low targets per seed {b['n_low_evaluated']} (of {b['n_low_all']})")
                for src in ('random', 'optimized'):
                    for sc in ('tot', 'prop'):
                        print(f"    {src:9s} R_{sc:4s} flip {rate(b[f'{src}_{sc}_flip'])}  shift {ms(b[f'{src}_{sc}_shift'])}")
                print(f"    all 50 candidates: flip tot {rate(b['all_candidates_flip_tot'])} prop {rate(b['all_candidates_flip_prop'])}")
                print(f"    label swap: Lambda_in {ms(b['Lambda_in'])} (min {b['Lambda_in_min']:.2e}), Lambda_out {ms(b['Lambda_out'], 1, 9)}"
                      f" (max {b['Lambda_out_max']:.1e}); |dz| in {ms(b['dz_in'])}, out max {b['dz_out_max']:.1e}; "
                      f"dR_tot out {ms(b['dR_tot_dis_out'])}, dR_tot trust-in {ms(b['dR_tot_tru_in'])}")
                recs = [x for s_ in d['runs'].values() if agg in s_ and 'evasion' in s_[agg] for x in s_[agg]['evasion']['targets']]
                pr = [x for x in recs if x['dz_in'] < 1e-5]          # random in-window edge left the target's embedding unchanged
                if pr:
                    mt = sum(x['random']['R_tot'] - x['R_clean'] for x in pr) / len(pr)
                    mp = sum(x['random']['R_prop'] - x['R_clean'] for x in pr) / len(pr)
                    print(f"    target embedding unchanged (|dz|<1e-5) for {len(pr)}/{len(recs)} random in-window edges: "
                          f"mean shift R_tot {mt:+.4f}, R_prop {mp:+.5f}; flips R_tot {sum(x['random']['R_tot'] < 0.5 for x in pr)}, "
                          f"R_prop {sum(x['random']['R_prop'] < 0.5 for x in pr)}; mean Lambda_in {sum(x['Lambda_in'] for x in pr) / len(pr):.3f}")


def tables():
    """Markdown tables: MCC reproduction (all runs) and the single-edge evasion test (runs with part B)."""
    print('| data | features | attack size | agg | clean MCC | attacked MCC | published (clean / attacked) |')
    print('|---|---|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(RES, 'trustguard_casestudy_*.json'))):
        d = json.load(open(path))
        c, p = d['config'], d['published_fig3_single_timeslot_MCC']
        for agg, a in d.get('summary', {}).items():
            r = a['repro']
            pub = p['attack_with_defense'] if agg == 'robust' else p['attack_without_defense']
            print(f"| {d['dataset']} | {c['features']} | {c.get('attack_degree', 'train')} | {agg} | {ms(r['MCC_best_clean'])} | "
                  f"{ms(r['MCC_best_inwin_dis'])} | {p['no_attack']:.3f} / {pub:.3f} |")
    print('\n| data | features | agg | source | score | flip % | pooled k/n [95%] | mean shift | n/seed |')
    print('|---|---|---|---|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(RES, 'trustguard_casestudy_*.json'))):
        d = json.load(open(path))
        for agg, a in d.get('summary', {}).items():
            b = a.get('evasion', {})
            if not b or not b.get('n_low_evaluated'):
                continue
            for src in ('random', 'optimized'):
                for sc in ('tot', 'prop'):
                    f = b[f'{src}_{sc}_flip']
                    lo, hi = f['pooled_cp95_pct']
                    print(f"| {d['dataset']} | {d['config']['features']} | {agg} | {src} | R_{sc} | {ms(f['rate_pct'], 1, 0)} | "
                          f"{f['pooled_k']}/{f['pooled_n']} [{lo:.0f},{hi:.0f}] | {ms(b[f'{src}_{sc}_shift'])} | {b['n_low_evaluated']} |")


if __name__ == '__main__':
    tables() if '--tables' in os.sys.argv else main()
