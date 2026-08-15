#!/usr/bin/env python3
"""
exp_01_universality_maps.py
============================

FEIGENBAUM UNIVERSALITY: INDEPENDENT VERIFICATION ACROSS MAP FAMILIES

HYPOTHESIS:
    The Fibonacci closed-form formula for delta matches the universal
    bifurcation ratio computed independently from the period-doubling
    cascade of 5+ unimodal map families.

METHOD:
    For each map, find superstable parameters r_n where f^{2^n}(x_c) = x_c.

    KEY: f^{2^n}(x_c) = x_c has roots at ALL lower-period superstable params
    (period-k divides period-2^n) AND at chaotic-window params beyond r_inf.
    To find the GENUINE period-2^n superstable, we check that
    f^{2^{n-1}}(x_c) != x_c (the orbit is truly period 2^n, not 2^{n-1}).

    All map iterations vectorised with numpy for speed.

MAP FAMILIES (all unimodal with quadratic maximum):
    1. Logistic:   f(x) = r*x*(1-x),   x_c = 1/2
    2. Quadratic:  f(x) = r - x^2,      x_c = 0
    3. Sine:       f(x) = r*sin(pi*x),  x_c = 1/2
    4. Cubic-max:  f(x) = r*x^2*(3-2x), x_c = 1
    5. Exp-quad:   f(x) = r*x*exp(1-x), x_c = 1
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


# =============================================================================
# KNOWN VALUES
# =============================================================================

DELTA_KNOWN = 4.669201609102990671853203820447323546553756938557
R_INF_LOGISTIC = 3.569945671870944901842005151386498936763836911515


# =============================================================================
# FORMULA (from exp_07)
# =============================================================================

def formula_r_inf():
    F, P = 55, 17
    d = np.sqrt(52 + 2 * np.pi / 55)
    inner = P - np.pi / (F * d)
    r_base = np.pi * (F + np.sqrt(inner)) * (F + np.pi) / F**2
    xi_m1 = np.pi / 55
    k = np.sqrt(3/5 - xi_m1**2 / 7)
    return r_base - k * np.pi**4 / 55**6


def formula_delta():
    return (50050 + 32 * np.pi) / (10725 + 5 * np.pi)


# =============================================================================
# MAP FAMILIES
# =============================================================================

def logistic_f(x, r):    return r * x * (1.0 - x)
def quadratic_f(x, r):   return r - x * x
def sine_f(x, r):        return r * np.sin(np.pi * x)
def cubicmax_f(x, r):    return r * x * x * (3.0 - 2.0 * x)
def expquad_f(x, r):     return r * x * np.exp(np.clip(1.0 - x, -50, 50))

MAP_DEFS = [
    {'name': 'logistic',  'f': logistic_f,  'x_c': 0.5, 'r_lo': 1.5,  'r_hi': 3.58, 'first_n': 0},
    {'name': 'quadratic', 'f': quadratic_f, 'x_c': 0.0, 'r_lo': 0.01, 'r_hi': 1.42, 'first_n': 1},
    {'name': 'sine',      'f': sine_f,      'x_c': 0.5, 'r_lo': 0.3,  'r_hi': 0.84, 'first_n': 0},
    {'name': 'cubic-max', 'f': cubicmax_f,  'x_c': 1.0, 'r_lo': 0.8,  'r_hi': 1.26, 'first_n': 0},
    {'name': 'exp-quad',  'f': expquad_f,   'x_c': 1.0, 'r_lo': 0.5,  'r_hi': 3.58, 'first_n': 0},
]


# =============================================================================
# CORE: ITERATE AND FIND GENUINE PERIOD-2^n SUPERSTABLE
# =============================================================================

def scalar_iterate(f, x_c, r, period):
    """Compute f^{period}(x_c) at scalar r."""
    x = x_c
    for _ in range(period):
        x = f(x, r)
        if not np.isfinite(x) or abs(x) > 1e12:
            return np.nan
    return x


def is_genuine_period(f, x_c, r, n):
    """
    Check if r gives a genuine period-2^n orbit (not period-2^{n-1} or lower).
    True if f^{2^n}(x_c) ≈ x_c but f^{2^{n-1}}(x_c) is NOT ≈ x_c.
    """
    if n == 0:
        return True  # period-1 has no lower period to check

    half_period = 2**(n-1)
    x_half = scalar_iterate(f, x_c, r, half_period)

    if not np.isfinite(x_half):
        return False

    # If the half-period iterate returns to x_c, this is period-2^{n-1}, not 2^n
    return abs(x_half - x_c) > 1e-8


def find_superstable_genuine(f, x_c, n, r_lo, r_hi, n_scan=20000):
    """
    Find the genuine period-2^n superstable in [r_lo, r_hi].

    Scans for ALL roots of f^{2^n}(x_c) - x_c, filters for genuine period
    (not lower-period alias), returns the smallest genuine root above r_lo.
    """
    period = 2**n
    rs = np.linspace(r_lo, r_hi, n_scan)

    # Vectorised iteration
    x = np.full_like(rs, x_c)
    for _ in range(period):
        with np.errstate(over='ignore', invalid='ignore'):
            x = f(x, rs)
        bad = ~np.isfinite(x) | (np.abs(x) > 1e12)
        x[bad] = np.nan

    h = x - x_c

    # Find all sign changes
    candidates = []
    for i in range(1, len(h)):
        if (np.isfinite(h[i]) and np.isfinite(h[i-1]) and h[i] * h[i-1] < 0):
            try:
                def h_scalar(r):
                    return scalar_iterate(f, x_c, r, period) - x_c
                r_root = brentq(h_scalar, rs[i-1], rs[i], xtol=1e-14, rtol=1e-14)
                candidates.append(r_root)
            except (ValueError, FloatingPointError):
                pass

    if not candidates:
        return None

    # Filter for genuine period-2^n (not lower period alias)
    genuine = [r for r in candidates if is_genuine_period(f, x_c, r, n)]

    if not genuine:
        return None

    # Return the smallest genuine root (closest to previous superstable)
    return min(genuine)


# =============================================================================
# CASCADE
# =============================================================================

def find_cascade(mdef, n_max=12, verbose=True):
    f = mdef['f']
    x_c = mdef['x_c']
    name = mdef['name']
    first_n = mdef['first_n']

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Map: {name},  x_c = {x_c}")
        print(f"  {'='*60}")

    results = []

    for n in range(first_n, n_max + 1):
        # Determine scan window
        if len(results) >= 2:
            r_prev = results[-1][1]
            r_prev2 = results[-2][1]
            dr = r_prev - r_prev2
            # Expect next gap ≈ dr / delta
            dr_next = abs(dr) / 4.669
            # Search from just above r_prev to estimated r_inf + margin
            scan_lo = r_prev + dr_next * 0.05
            scan_hi = r_prev + dr_next * 3.0
            n_scan = 20000
        elif len(results) == 1:
            scan_lo = results[0][1] * 1.001
            scan_hi = mdef['r_hi']
            n_scan = 40000
        else:
            scan_lo = mdef['r_lo']
            scan_hi = mdef['r_hi']
            n_scan = 40000

        t0 = time.time()
        r_ss = find_superstable_genuine(f, x_c, n, scan_lo, scan_hi, n_scan)

        # Fallback: full range
        if r_ss is None:
            r_ss = find_superstable_genuine(f, x_c, n, mdef['r_lo'], mdef['r_hi'], 80000)

        dt = time.time() - t0

        if r_ss is not None and (not results or r_ss > results[-1][1]):
            results.append((n, r_ss))
            if verbose:
                print(f"    n={n:2d}, 2^n={2**n:>6d}: r = {r_ss:.15f}  ({dt:.2f}s)")
        else:
            if verbose:
                status = "no genuine root" if r_ss is None else f"not increasing ({r_ss:.10f})"
                print(f"    n={n:2d}, 2^n={2**n:>6d}: STOPPED — {status}  ({dt:.2f}s)")
            break

    return results


# =============================================================================
# EXTRACT CONSTANTS
# =============================================================================

def extract_constants(points, map_name="", verbose=True):
    if len(points) < 4:
        if verbose:
            print(f"    {map_name}: only {len(points)} points, need >= 4")
        return None

    rs = [p[1] for p in points]
    ns = [p[0] for p in points]

    deltas = []
    for i in range(2, len(rs)):
        dr_prev = rs[i-1] - rs[i-2]
        dr_curr = rs[i] - rs[i-1]
        if abs(dr_curr) > 1e-30:
            deltas.append((ns[i], dr_prev / dr_curr))

    # Aitken extrapolation
    r_n, r_n1, r_n2 = rs[-1], rs[-2], rs[-3]
    denom = (r_n - r_n1) - (r_n1 - r_n2)
    r_inf = r_n - (r_n - r_n1)**2 / denom if abs(denom) > 1e-30 else r_n

    delta_best = deltas[-1][1] if deltas else None

    if verbose and deltas:
        print(f"\n    Delta convergence ({map_name}):")
        for n, d in deltas:
            err = abs(d - DELTA_KNOWN) / DELTA_KNOWN
            digits = max(0, -int(np.log10(err))) if 0 < err < 1 else 0
            print(f"      n={n:2d}: delta = {d:12.8f}  ({err:.2e}, ~{digits} dig)")
        print(f"    r_inf (Aitken) = {r_inf:.15f}")

    return {
        'map': map_name,
        'n_points': len(rs),
        'superstable_points': [(n, float(r)) for n, r in points],
        'deltas': [(n, float(d)) for n, d in deltas],
        'r_inf_extrapolated': float(r_inf),
        'delta_best': float(delta_best) if delta_best else None,
        'delta_best_error': float(abs(delta_best - DELTA_KNOWN) / DELTA_KNOWN) if delta_best else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(n_max=12):
    print()
    print("=" * 72)
    print("  EXPERIMENT 01: FEIGENBAUM UNIVERSALITY ACROSS MAP FAMILIES")
    print("=" * 72)
    print(f"\n  Cascade depth: up to period 2^{n_max} = {2**n_max}")
    print(f"  Root filter: genuine period check (f^{{2^{{n-1}}}}(x_c) != x_c)")
    print()

    all_results = []
    t0 = time.time()

    for mdef in MAP_DEFS:
        try:
            points = find_cascade(mdef, n_max=n_max)
            result = extract_constants(points, map_name=mdef['name'])
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  {mdef['name']}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    # =========================================================================
    # COMPARISON
    # =========================================================================

    print("\n" + "=" * 72)
    print("  CROSS-MAP COMPARISON")
    print("=" * 72)

    r_inf_formula = formula_r_inf()
    delta_formula = formula_delta()
    formula_err = abs(delta_formula - DELTA_KNOWN) / DELTA_KNOWN

    print(f"\n  Formula delta = {delta_formula:.15f}  (error: {formula_err:.2e})")
    print(f"  Known   delta = {DELTA_KNOWN:.15f}")
    print()

    print(f"  {'Map':<14} {'Pts':>4} {'Best delta':>14} {'Error':>12} {'Dig':>4}")
    print(f"  {'-'*14} {'-'*4} {'-'*14} {'-'*12} {'-'*4}")

    delta_values = []
    for r in all_results:
        d = r['delta_best']
        if d is not None and d > 0:
            d_err = abs(d - DELTA_KNOWN) / DELTA_KNOWN
            digits = max(0, -int(np.log10(d_err))) if 0 < d_err < 1 else 0
            delta_values.append(d)
            print(f"  {r['map']:<14} {r['n_points']:>4} {d:>14.8f} {d_err:>12.2e} {digits:>4}")
        else:
            print(f"  {r['map']:<14} {r['n_points']:>4} {'N/A':>14}")

    if len(delta_values) >= 2:
        arr = np.array(delta_values)
        print(f"\n  Cross-map:  mean = {np.mean(arr):.10f}, "
              f"std = {np.std(arr):.4e}, "
              f"spread = {np.std(arr)/np.mean(arr):.4e}")

    # Logistic r_inf
    logistic = [r for r in all_results if r['map'] == 'logistic']
    if logistic:
        r_comp = logistic[0]['r_inf_extrapolated']
        print(f"\n  Logistic r_inf:")
        print(f"    Known:    {R_INF_LOGISTIC:.15f}")
        print(f"    Formula:  {r_inf_formula:.15f}  "
              f"({abs(r_inf_formula-R_INF_LOGISTIC)/R_INF_LOGISTIC:.2e})")
        print(f"    Computed: {r_comp:.15f}  "
              f"({abs(r_comp-R_INF_LOGISTIC)/R_INF_LOGISTIC:.2e})")

    # =========================================================================
    # VERDICT
    # =========================================================================

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72 + "\n")

    n_with = len(delta_values)
    if n_with >= 3:
        errs = [abs(d - DELTA_KNOWN)/DELTA_KNOWN for d in delta_values]
        max_err = max(errs)
        min_dig = min(max(0, -int(np.log10(e))) if 0 < e < 1 else 0 for e in errs)
        all_1pct = all(e < 0.01 for e in errs)

        print(f"  Maps with delta:  {n_with}/{len(all_results)}")
        print(f"  Worst error:      {max_err:.2e}")
        print(f"  Min digits:       {min_dig}")
        print(f"  Universality:     {'CONFIRMED' if all_1pct else 'PARTIAL'}")
    else:
        print(f"  Insufficient data ({n_with} maps with delta)")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_01_universality_maps.py',
        'parameters': {'n_max': n_max},
        'formula': {
            'r_inf': float(r_inf_formula), 'delta': float(delta_formula),
            'delta_error': float(formula_err),
        },
        'map_results': all_results,
        'universality': {
            'n_maps': len(all_results),
            'delta_values': [float(d) for d in delta_values],
            'delta_mean': float(np.mean(delta_values)) if delta_values else None,
            'delta_std': float(np.std(delta_values)) if len(delta_values) >= 2 else None,
        },
        'elapsed_seconds': elapsed,
    }
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_01_universality_maps_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"  Saved: {fpath}")
    return output


if __name__ == "__main__":
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment(n_max=n_max)
