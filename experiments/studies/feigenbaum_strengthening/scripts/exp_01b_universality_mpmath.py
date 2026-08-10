#!/usr/bin/env python3
"""
exp_01b_universality_mpmath.py
================================

HIGH-PRECISION FEIGENBAUM UNIVERSALITY (mpmath, 50+ dps)

Extends exp_01 by using mpmath to push the cascade to n_max=14-16,
computing delta to 10+ digits from independently extracted superstable
parameters. This exceeds float64 verification limits.

METHOD:
    For 3 map families (logistic, quadratic, sine):
    - Scan narrow window for sign changes of f^{2^n}(x_c) - x_c
    - Bisect each bracket to 50+ dps precision
    - Filter for genuine period-2^n (not lower-period alias)
    - Compute delta_n and apply Aitken extrapolation for r_inf

SUCCESS CRITERION: independently computed delta matches known value to 10+ digits.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from mpmath import mp, mpf, pi as mppi, sin as mpsin, log10 as mplog10, fabs

DELTA_KNOWN_STR = "4.6692016091029906718532038204473235465537569385573790853930777530"
R_INF_LOGISTIC_STR = "3.5699456718709449018420051513864989367638369115148323781380304413"


# =============================================================================
# MAP DEFINITIONS (mpmath)
# =============================================================================

def logistic_mp(x, r):
    return r * x * (mpf(1) - x)

def quadratic_mp(x, r):
    return r - x * x

def sine_mp(x, r):
    return r * mpsin(mppi * x)

MAP_DEFS_MP = [
    {'name': 'logistic',  'f': logistic_mp,  'x_c': mpf('0.5'),
     'r_lo': mpf('1.5'), 'r_hi': mpf('3.5700'), 'first_n': 0},
    {'name': 'quadratic', 'f': quadratic_mp, 'x_c': mpf('0'),
     'r_lo': mpf('0.01'), 'r_hi': mpf('1.40116'), 'first_n': 1},
    {'name': 'sine',      'f': sine_mp,      'x_c': mpf('0.5'),
     'r_lo': mpf('0.3'), 'r_hi': mpf('0.8350'), 'first_n': 0},
]


# =============================================================================
# ROOT FINDING
# =============================================================================

def iterate_mp(f, x_c, r, period):
    """Compute f^{period}(x_c) in mpmath."""
    x = x_c
    for _ in range(period):
        x = f(x, r)
    return x


def is_genuine_period_mp(f, x_c, r, n, dps):
    """Check genuine period-2^n: f^{2^{n-1}}(x_c) NOT close to x_c."""
    if n == 0:
        return True
    half = 2 ** (n - 1)
    x_half = iterate_mp(f, x_c, r, half)
    # For genuine period 2^n, the half-period iterate should be far from x_c
    return fabs(x_half - x_c) > mpf(10) ** (-dps // 3)


def bisect_mp(f, x_c, period, lo, hi, dps):
    """Bisect to find r where f^{period}(x_c) = x_c in [lo, hi]."""
    mp.dps = dps + 20
    tol = mpf(10) ** (-(dps + 5))

    h_lo = iterate_mp(f, x_c, lo, period) - x_c
    h_hi = iterate_mp(f, x_c, hi, period) - x_c

    if h_lo * h_hi > 0:
        return None

    for _ in range(4 * dps):
        mid = (lo + hi) / 2
        h_mid = iterate_mp(f, x_c, mid, period) - x_c

        if fabs(h_mid) < tol or hi - lo < tol:
            return mid

        if h_lo * h_mid < 0:
            hi = mid
        else:
            lo = mid
            h_lo = h_mid

    return (lo + hi) / 2


def find_superstable_scan(f, x_c, n, scan_lo, scan_hi, dps, n_scan=500):
    """
    Scan for sign changes of f^{2^n}(x_c) - x_c, then bisect each bracket.
    Filter for genuine period. Return smallest genuine root above scan_lo.
    """
    mp.dps = dps + 20
    period = 2 ** n

    # Scan for sign changes
    rs = [scan_lo + (scan_hi - scan_lo) * mpf(i) / n_scan for i in range(n_scan + 1)]
    hs = []
    for r in rs:
        try:
            h = iterate_mp(f, x_c, r, period) - x_c
            hs.append(h)
        except Exception:
            hs.append(None)

    # Find brackets
    brackets = []
    for i in range(len(hs) - 1):
        if hs[i] is not None and hs[i + 1] is not None:
            if hs[i] * hs[i + 1] < 0:
                brackets.append((rs[i], rs[i + 1]))

    if not brackets:
        return None

    # Bisect each bracket and filter for genuine period
    candidates = []
    for lo, hi in brackets:
        root = bisect_mp(f, x_c, period, lo, hi, dps)
        if root is not None:
            if is_genuine_period_mp(f, x_c, root, n, dps):
                candidates.append(root)

    if not candidates:
        return None

    # Return smallest genuine root
    return min(candidates)


def find_cascade_mp(mdef, n_max, dps):
    """Find superstable cascade for a map family using mpmath."""
    f = mdef['f']
    x_c = mdef['x_c']
    name = mdef['name']
    first_n = mdef['first_n']

    print(f"\n  {'='*60}")
    print(f"  Map: {name} (mpmath, {dps} dps, n_max={n_max})")
    print(f"  {'='*60}")

    results = []

    for n in range(first_n, n_max + 1):
        # Adaptive scan density: fewer points at high n (function evals are expensive)
        # At n, each eval costs 2^n iterations. Budget: ~500K total iterations per scan.
        period = 2 ** n
        base_scan = max(50, min(2000, 500000 // max(period, 1)))

        # Determine search window
        if len(results) >= 3:
            r_prev = results[-1][1]
            r_prev2 = results[-2][1]
            dr = r_prev - r_prev2
            r_pp = results[-3][1]
            dr_prev = r_prev2 - r_pp
            if fabs(dr) > 0:
                delta_est = dr_prev / dr
            else:
                delta_est = mpf('4.67')
            dr_next = fabs(dr) / delta_est
            scan_lo = r_prev + dr_next * mpf('0.1')
            scan_hi = r_prev + dr_next * mpf('8.0')
            n_scan = base_scan
        elif len(results) >= 2:
            r_prev = results[-1][1]
            r_prev2 = results[-2][1]
            dr = r_prev - r_prev2
            dr_next = fabs(dr) / mpf('4.67')
            scan_lo = r_prev + dr_next * mpf('0.1')
            scan_hi = r_prev + dr_next * mpf('8.0')
            n_scan = base_scan
        elif len(results) == 1:
            scan_lo = results[0][1] * mpf('1.001')
            scan_hi = mdef['r_hi']
            n_scan = min(2000, base_scan * 2)
        else:
            scan_lo = mdef['r_lo']
            scan_hi = mdef['r_hi']
            n_scan = min(2000, base_scan * 2)

        t0 = time.time()
        r_ss = find_superstable_scan(f, x_c, n, scan_lo, scan_hi, dps, n_scan)

        # Fallback: wider window
        if r_ss is None and len(results) >= 1:
            r_prev = results[-1][1]
            r_ss = find_superstable_scan(f, x_c, n, r_prev, mdef['r_hi'], dps, 5000)

        dt = time.time() - t0

        if r_ss is not None:
            results.append((n, r_ss))
            r_str = mp.nstr(r_ss, 22)
            period = 2 ** n
            print(f"    n={n:2d}, 2^n={period:>6d}: r = {r_str}  ({dt:.1f}s)")
        else:
            print(f"    n={n:2d}, 2^n={2**n:>6d}: NOT FOUND  ({dt:.1f}s)")
            # Don't stop -- try next n with wider fallback

    return results


def analyze_cascade_mp(points, map_name, dps):
    """Extract delta and r_inf from superstable points."""
    mp.dps = dps

    if len(points) < 4:
        print(f"    {map_name}: only {len(points)} points, need >= 4")
        return None

    rs = [p[1] for p in points]
    ns = [p[0] for p in points]
    delta_known = mpf(DELTA_KNOWN_STR)

    # Compute delta_n from successive ratios
    deltas = []
    for i in range(2, len(rs)):
        dr_prev = rs[i - 1] - rs[i - 2]
        dr_curr = rs[i] - rs[i - 1]
        if fabs(dr_curr) > mpf(10) ** (-dps) and fabs(dr_prev) > mpf(10) ** (-dps):
            delta_n = dr_prev / dr_curr
            err = fabs(delta_n - delta_known) / delta_known
            digits = float(-mplog10(err)) if err > 0 else dps
            deltas.append({'n': ns[i], 'delta': delta_n, 'error': float(err), 'digits': digits})

    # Aitken extrapolation for r_inf (use last 3 points)
    r_inf_estimates = []
    for i in range(2, len(rs)):
        r0, r1, r2 = rs[i - 2], rs[i - 1], rs[i]
        denom = (r2 - r1) - (r1 - r0)
        if fabs(denom) > mpf(10) ** (-dps):
            r_inf = r2 - (r2 - r1) ** 2 / denom
            r_inf_estimates.append({'from_n': ns[i], 'r_inf': r_inf})

    # Print
    print(f"\n    Delta convergence ({map_name}):")
    print(f"    {'n':>4s}  {'delta':>26s}  {'digits':>8s}")
    print(f"    {'-'*4}  {'-'*26}  {'-'*8}")
    for d in deltas:
        d_str = mp.nstr(d['delta'], 20)
        print(f"    {d['n']:4d}  {d_str:>26s}  {d['digits']:8.1f}")

    best = deltas[-1] if deltas else None

    if r_inf_estimates:
        best_rinf = r_inf_estimates[-1]['r_inf']
        print(f"\n    r_inf (Aitken from n={r_inf_estimates[-1]['from_n']}): "
              f"{mp.nstr(best_rinf, 25)}")
        if map_name == 'logistic':
            r_inf_known = mpf(R_INF_LOGISTIC_STR)
            rinf_err = fabs(best_rinf - r_inf_known) / r_inf_known
            rinf_digits = float(-mplog10(rinf_err)) if rinf_err > 0 else dps
            print(f"    r_inf known:                      {mp.nstr(r_inf_known, 25)}")
            print(f"    r_inf error: {float(rinf_err):.2e} ({rinf_digits:.1f} digits)")

    return {
        'map': map_name,
        'n_points': len(rs),
        'max_n': max(ns),
        'superstable_points': [(n, mp.nstr(r, dps)) for n, r in points],
        'deltas': deltas,
        'r_inf_aitken': mp.nstr(best_rinf, dps) if r_inf_estimates else None,
        'delta_best': {'value': mp.nstr(best['delta'], 22),
                       'digits': best['digits']} if best else None,
    }


def run():
    DPS = 50
    N_MAX = 16

    mp.dps = DPS + 20

    print("=" * 72)
    print(f"  EXPERIMENT 01b: HIGH-PRECISION UNIVERSALITY (mpmath, {DPS} dps)")
    print("=" * 72)

    # Formula values
    delta_formula = (mpf(50050) + 32 * mppi) / (mpf(10725) + 5 * mppi)
    delta_known = mpf(DELTA_KNOWN_STR)
    formula_err = fabs(delta_formula - delta_known) / delta_known
    formula_digits = float(-mplog10(formula_err))
    print(f"\n  Formula delta:  {mp.nstr(delta_formula, 22)}")
    print(f"  Known delta:    {mp.nstr(delta_known, 22)}")
    print(f"  Formula error:  {float(formula_err):.2e} ({formula_digits:.1f} digits)")

    all_results = []
    t0 = time.time()

    for mdef in MAP_DEFS_MP:
        try:
            points = find_cascade_mp(mdef, N_MAX, DPS)
            result = analyze_cascade_mp(points, mdef['name'], DPS)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  {mdef['name']}: FAILED -- {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    # =========================================================================
    # CROSS-MAP COMPARISON
    # =========================================================================
    print("\n" + "=" * 72)
    print("  CROSS-MAP COMPARISON")
    print("=" * 72)

    print(f"\n  {'Map':<12} {'max_n':>6} {'Pts':>4} {'Best delta':>24} {'Digits':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*4} {'-'*24} {'-'*8}")
    for r in all_results:
        if r['delta_best']:
            print(f"  {r['map']:<12} {r['max_n']:>6} {r['n_points']:>4} "
                  f"{r['delta_best']['value']:>24} {r['delta_best']['digits']:>8.1f}")
        else:
            print(f"  {r['map']:<12} {r.get('max_n','?'):>6} {r['n_points']:>4} "
                  f"{'N/A':>24} {'N/A':>8}")

    print(f"\n  Formula delta: {formula_digits:.1f} digits")

    # Check if any map independently exceeds formula precision
    max_ind_digits = 0
    for r in all_results:
        if r['delta_best'] and r['delta_best']['digits'] > max_ind_digits:
            max_ind_digits = r['delta_best']['digits']

    if max_ind_digits >= 10:
        print(f"\n  ACHIEVED: {max_ind_digits:.1f} digits from independent cascade computation")
        print(f"  (Exceeds formula's {formula_digits:.1f} digits)")
    else:
        print(f"\n  Best independent: {max_ind_digits:.1f} digits "
              f"(target: 10+ to exceed formula)")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_01b_universality_mpmath.py',
        'parameters': {'dps': DPS, 'n_max': N_MAX},
        'formula': {
            'delta': mp.nstr(delta_formula, DPS),
            'digits': formula_digits,
        },
        'map_results': all_results,
        'max_independent_digits': max_ind_digits,
        'elapsed_seconds': elapsed,
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_01b_universality_mpmath_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"\n  Saved: {fpath}")


if __name__ == "__main__":
    run()
