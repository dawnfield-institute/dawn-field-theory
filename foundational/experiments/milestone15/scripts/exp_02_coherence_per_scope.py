"""
exp_02 -- Coherence Limits Per-Scope

Milestone 15 (The Representative Problem)

PRE-REGISTERED: journals/2026-06-11_m15-exp01-03-preregistration.md (same commit).
Re-poses M13.5 exp_15 (0/4): the "non-universal coherence limit" was class pooling.
Same rate (max single-step complement deformation), extended ranks 3..28, claims
per-scope and relational.

Tests:
  T1: A-family even and odd parity classes EACH converge (last-5 CV < 0.05)
  T2: class-limit ratios r1 = lim(A_even)/lim(A_odd), r2 = lim(D)/lim(A_even)
      agree across disjoint rank windows (14-20 vs 22-28) within 5%
  T3: each ADE class last-5 CV < CV of 20 random connected graphs at matched size

Outputs: results/exp_02_coherence_per_scope_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from representative import (
    build_path, build_d, max_deformation_rate, save_m15_results, _convert_numpy)

_M13_CORE = Path(__file__).resolve().parent.parent.parent / "milestone13" / "core"
sys.path.insert(0, str(_M13_CORE))
from identity_complement import generate_random_connected_graph

RANKS = range(3, 29)
W1 = (14, 20)
W2 = (22, 28)


def family_rates(builder, ranks):
    out = {}
    for n in ranks:
        try:
            out[n] = float(max_deformation_rate(builder(n)))
        except Exception:
            continue
    return out


def cv(vals):
    vals = np.asarray(vals, float)
    return float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else np.inf


def window_limit(rates, lo, hi):
    vals = [r for n, r in rates.items() if lo <= n <= hi]
    return float(np.mean(vals)) if vals else np.nan


def run():
    print("\n  Computing rates (A_3..A_28, D_4..D_28)...")
    a_rates = family_rates(build_path, RANKS)
    d_rates = family_rates(build_d, range(4, 29))

    a_even = {n: r for n, r in a_rates.items() if n % 2 == 0}
    a_odd = {n: r for n, r in a_rates.items() if n % 2 == 1}

    # T1: per-class convergence (last 5 ranks of each class)
    print("\n  T1: parity-class convergence")
    classes = {'A_even': a_even, 'A_odd': a_odd, 'D': d_rates}
    t1_detail = {}
    for name, rates in classes.items():
        last5 = [rates[n] for n in sorted(rates)[-5:]]
        c = cv(last5)
        t1_detail[name] = {'last5': last5, 'cv': c, 'converges': bool(c < 0.05)}
        print(f"    {name:<7} last-5 CV = {c:.4f} -> "
              f"{'converges' if c < 0.05 else 'FAILS'}")
    t1 = t1_detail['A_even']['converges'] and t1_detail['A_odd']['converges']
    print(f"    -> {'PASS' if t1 else 'FAIL'}")

    # T2: class-limit ratios stable across disjoint windows
    print("\n  T2: class-limit ratios across windows")
    ratios = {}
    stable = []
    for label, (num, den) in {'r1_Aeven_over_Aodd': (a_even, a_odd),
                              'r2_D_over_Aeven': (d_rates, a_even)}.items():
        rw1 = window_limit(num, *W1) / window_limit(den, *W1)
        rw2 = window_limit(num, *W2) / window_limit(den, *W2)
        rel = abs(rw1 - rw2) / np.mean([rw1, rw2])
        ratios[label] = {'w1': rw1, 'w2': rw2, 'rel_diff': rel,
                         'stable': bool(rel < 0.05)}
        stable.append(rel < 0.05)
        print(f"    {label}: w1={rw1:.4f} w2={rw2:.4f} rel-diff={rel:.4f}")
    t2 = all(stable)
    print(f"    -> {'PASS' if t2 else 'FAIL'}")

    # T3: per-scope constraint vs random controls
    print("\n  T3: class CV vs random connected controls (matched size)")
    rng_seed = 1500
    t3_detail = {}
    t3_flags = []
    for name, rates in classes.items():
        last_ranks = sorted(rates)[-5:]
        rand_cvs = []
        for rep in range(20):
            vals = []
            for n in last_ranks:
                g = generate_random_connected_graph(n, density=0.3,
                                                    seed=rng_seed + rep * 100 + n)
                vals.append(float(max_deformation_rate(g)))
            rand_cvs.append(cv(vals))
        class_cv = t1_detail[name]['cv']
        med_rand = float(np.median(rand_cvs))
        tighter = bool(class_cv < med_rand)
        t3_detail[name] = {'class_cv': class_cv, 'random_cv_median': med_rand,
                           'tighter': tighter}
        t3_flags.append(tighter)
        print(f"    {name:<7} class CV={class_cv:.4f} vs random median={med_rand:.4f} "
              f"-> {'tighter' if tighter else 'NOT tighter'}")
    t3 = all(t3_flags)
    print(f"    -> {'PASS' if t3 else 'FAIL'}")

    score = sum([t1, t2, t3])
    verdict = 'SUPPORTED' if (t1 and t2) else ('KILLED' if not t1 else 'PARTIAL')
    print(f"\n  Overall: {score}/3  VERDICT: {verdict}")
    return {
        'experiment': 'exp_02_coherence_per_scope', 'milestone': 'M15',
        'a_rates': a_rates, 'd_rates': d_rates,
        'T1': {'detail': t1_detail, 'PASS': t1},
        'T2': {'ratios': ratios, 'PASS': t2},
        'T3': {'detail': t3_detail, 'PASS': t3},
        'score': f"{score}/3", 'verdict': verdict,
    }


def selftest():
    print("SELFTEST: builders + rate plumbing only")
    r = max_deformation_rate(build_path(6))
    print(f"  A_6 rate = {r:.4f} (must be finite, positive)")
    assert 0 < r < 10
    r = max_deformation_rate(build_d(6))
    print(f"  D_6 rate = {r:.4f}")
    assert 0 < r < 10
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_02: Coherence Limits Per-Scope")
    print("Milestone 15 -- pre-registered")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_m15_results('exp_02_coherence_per_scope', _convert_numpy(data))
