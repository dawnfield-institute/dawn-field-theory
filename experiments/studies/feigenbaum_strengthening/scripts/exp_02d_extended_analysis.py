#!/usr/bin/env python3
"""
exp_02d_extended_analysis.py
==============================

EXTENDED COEFFICIENT ANALYSIS: permutation test + march to N=80 + suppression pattern.

Three analyses Peter requested:

1. PERMUTATION TEST: With 6 Fibonacci and ~35 non-Fibonacci positions from N=40 data,
   test whether Fibonacci mean marginal information is significantly low.
   Pick 6 random positions 100K times, compute how often their mean marginal
   falls as low as the Fibonacci mean.

2. EXTEND TO N=80: Use Newton solve at N=40 (120 dps), then march coefficients
   from 41 to 80 using the triangular structure of the Feigenbaum equation.
   Given alpha (known to 64 digits), each c_N can be extracted one at a time.

3. SUPPRESSION PATTERN: Plot marginal digits vs Fibonacci index.
   Fit decay, test if related to phi.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from itertools import combinations

from mpmath import mp, mpf, matrix as mpmatrix, lu_solve, log10 as mplog10, fabs, pi as mppi

ALPHA_EXACT_STR = "2.5029078750958928222839028732182157863646264378071999356799944268"
DELTA_KNOWN_STR = "4.6692016091029906718532038204473235465537569385573790853930777530"
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
FIB_SET = set(FIBONACCI)
FIB_INDEX = {1: 1, 2: 3, 3: 4, 5: 5, 8: 6, 13: 7, 21: 8, 34: 9, 55: 10, 89: 11}
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# POLYNOMIAL TOOLS (mpmath)
# =============================================================================

def poly_mul_trunc(a, b, max_idx):
    """Multiply two even-power coefficient arrays, truncated to index max_idx.
    a[j] = coefficient of x^{2j} in the polynomial."""
    result = [mpf(0)] * (max_idx + 1)
    la = min(len(a), max_idx + 1)
    lb = len(b)
    for i in range(la):
        if a[i] == 0:
            continue
        for j in range(min(lb, max_idx + 1 - i)):
            if b[j] == 0:
                continue
            result[i + j] += a[i] * b[j]
    return result


def feigenbaum_residual(c_vec, N, dps):
    """Compute residual of Feigenbaum equation at truncation N."""
    mp.dps = dps
    c = [mpf(1)] + list(c_vec)
    g_at_1 = sum(c)
    alpha = mpf(-1) / g_at_1

    # y(x) = g(-x/alpha) = sum c[k] x^{2k} / alpha^{2k}
    y = [mpf(0)] * (N + 1)
    for k in range(N + 1):
        y[k] = c[k] / alpha ** (2 * k)

    # Build y^{2k} for k=0..N
    y_sq = poly_mul_trunc(y, y, N)
    y_pow = [None] * (N + 1)
    y_pow[0] = [mpf(0)] * (N + 1)
    y_pow[0][0] = mpf(1)
    y_pow[1] = y_sq[:]
    for k in range(2, N + 1):
        y_pow[k] = poly_mul_trunc(y_pow[k - 1], y_sq, N)

    # g(y) = sum c[k] * (y^2)^k = sum c[k] * y_pow[k]
    g_of_y = [mpf(0)] * (N + 1)
    for k in range(N + 1):
        if c[k] == 0:
            continue
        for j in range(N + 1):
            g_of_y[j] += c[k] * y_pow[k][j]

    # T[g] = -alpha * g(y)
    T_g = [-alpha * g_of_y[j] for j in range(N + 1)]

    # Residual at each order
    residual = [T_g[k] - c[k] for k in range(1, N + 1)]
    return residual, alpha


def solve_newton(N, seed_c, dps, max_iter=30):
    """Newton solve at truncation N from a seed."""
    mp.dps = dps
    c = [mpf(str(x)) for x in seed_c[:N]]
    while len(c) < N:
        c.append(mpf(0))

    for iteration in range(max_iter):
        res, alpha = feigenbaum_residual(c, N, dps)
        res_norm = max(fabs(r) for r in res)

        alpha_exact = mpf(ALPHA_EXACT_STR)
        err = fabs(fabs(alpha) - alpha_exact) / alpha_exact
        digits = float(-mplog10(err)) if err > 0 else dps

        if iteration == 0 or iteration % 5 == 0 or res_norm < mpf(10) ** (-dps + 10):
            print(f"      iter {iteration}: residual={float(res_norm):.2e}, "
                  f"alpha={digits:.1f} digits")

        if res_norm < mpf(10) ** (-dps + 10):
            break

        # Numerical Jacobian
        eps = mpf(10) ** (-(dps * 2) // 3)
        J_cols = []
        for j in range(N):
            c_p = list(c)
            c_p[j] += eps
            res_p, _ = feigenbaum_residual(c_p, N, dps)
            J_cols.append([(res_p[i] - res[i]) / eps for i in range(N)])

        J_mat = mpmatrix(N, N)
        for i in range(N):
            for j in range(N):
                J_mat[i, j] = J_cols[j][i]

        rhs = mpmatrix(N, 1)
        for i in range(N):
            rhs[i] = -res[i]

        try:
            delta = lu_solve(J_mat, rhs)
            for j in range(N):
                c[j] += delta[j]
        except Exception:
            break

    c_full = [mpf(1)] + c
    alpha_final = fabs(mpf(-1) / sum(c_full))
    return c, alpha_final


def march_coefficients(c_known, alpha, N_from, N_to, dps):
    """
    Extend coefficients from N_from to N_to using triangular extraction.

    Given alpha and c[0]..c[N_from-1], extract c[N_from]..c[N_to-1] one at a time.
    At each order 2N, the equation is (approximately) linear in c_N:
        c_N = -alpha * A / (1 + alpha * (P + B))
    where A, P, B depend only on known lower coefficients.
    """
    mp.dps = dps
    c = list(c_known)  # c[0]=1, c[1], ..., c[N_from-1]

    for N in range(N_from, N_to + 1):
        # y_0(x) = sum_{k=0}^{N-1} c[k] * x^{2k} / alpha^{2k}
        # as even-power coefficient array
        y0 = [mpf(0)] * (N + 1)
        for k in range(min(len(c), N)):
            y0[k] = c[k] / alpha ** (2 * k)

        # Build y0_sq = y0^2, then y0_pow[k] = y0^{2k} = (y0_sq)^k
        y0_sq = poly_mul_trunc(y0, y0, N)
        y_pow = [None] * (N + 1)
        y_pow[0] = [mpf(0)] * (N + 1)
        y_pow[0][0] = mpf(1)
        if N >= 1:
            y_pow[1] = y0_sq[:]
        for k in range(2, N + 1):
            y_pow[k] = poly_mul_trunc(y_pow[k - 1], y0_sq, N)

        # A = sum_{k=0}^{N-1} c[k] * y_pow[k][N]
        # (x^{2N} coefficient of sum c[k] * y^{2k} with c_N = 0)
        A = mpf(0)
        for k in range(min(len(c), N)):
            if k < len(y_pow) and N < len(y_pow[k]):
                A += c[k] * y_pow[k][N]

        # P = y_pow[N][N] (x^{2N} coefficient of y0^{2N})
        P = mpf(0)
        if N < len(y_pow) and y_pow[N] is not None and N < len(y_pow[N]):
            P = y_pow[N][N]

        # B = (1/alpha^{2N}) * sum_{k=1}^{N-1} 2*k * c[k]
        B_sum = mpf(0)
        for k in range(1, min(len(c), N)):
            B_sum += 2 * k * c[k]
        B = B_sum / alpha ** (2 * N)

        # c_N = -alpha * A / (1 + alpha * (P + B))
        denom = 1 + alpha * (P + B)
        if fabs(denom) < mpf(10) ** (-dps + 5):
            print(f"    N={N}: denominator near zero, stopping")
            break

        c_N = -alpha * A / denom
        c.append(c_N)

        if N % 10 == 0:
            log_cn = float(mplog10(fabs(c_N))) if fabs(c_N) > 0 else -999
            print(f"    Marched to N={N}: log10|c_N| = {log_cn:.2f}")

    return c


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def permutation_test(marginals, fib_indices, n_perms=100000):
    """
    Permutation test: how often does a random set of len(fib_indices) positions
    have mean marginal as low as the Fibonacci mean?
    """
    n_fib = len(fib_indices)
    fib_mean = np.mean([marginals[i] for i in fib_indices])

    all_indices = list(range(len(marginals)))
    rng = np.random.default_rng(42)

    count_as_low = 0
    random_means = []
    for _ in range(n_perms):
        sample = rng.choice(all_indices, size=n_fib, replace=False)
        sample_mean = np.mean([marginals[i] for i in sample])
        random_means.append(sample_mean)
        if sample_mean <= fib_mean:
            count_as_low += 1

    p_value = count_as_low / n_perms
    return {
        'fib_mean': fib_mean,
        'p_value': p_value,
        'n_perms': n_perms,
        'random_mean': np.mean(random_means),
        'random_std': np.std(random_means),
        'z_score': (fib_mean - np.mean(random_means)) / np.std(random_means)
    }


def suppression_pattern(fib_marginals):
    """Analyze whether Fibonacci marginal digits decay with Fibonacci index."""
    # fib_marginals: list of (fib_index_k, marginal_digits)
    if len(fib_marginals) < 3:
        return None

    ks = np.array([k for k, _ in fib_marginals], dtype=float)
    ms = np.array([m for _, m in fib_marginals], dtype=float)

    # Linear fit: marginal ~ a * k + b
    fit = np.polyfit(ks, ms, 1)
    predicted = np.polyval(fit, ks)
    residuals = ms - predicted
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((ms - np.mean(ms)) ** 2)

    # What k gives marginal = 0? (crossover to dead zone)
    k_zero = -fit[1] / fit[0] if abs(fit[0]) > 1e-10 else None

    # Compare slope to -1/phi, -ln(phi), etc.
    comparisons = {
        'slope': fit[0],
        'intercept': fit[1],
        'r_squared': r_squared,
        'k_zero_crossing': k_zero,
        'slope_vs_neg_inv_phi': fit[0] / (-1 / PHI),
        'slope_vs_neg_ln_phi': fit[0] / (-np.log(PHI)),
    }

    return comparisons


# =============================================================================
# MAIN
# =============================================================================

def run():
    DPS = 120
    N_NEWTON = 40
    N_MARCH = 80

    print("=" * 72)
    print("  EXTENDED FEIGENBAUM COEFFICIENT ANALYSIS")
    print("=" * 72)

    # =========================================================================
    # PHASE 1: Newton solve at N=40
    # =========================================================================
    print(f"\n  Phase 1: Float64 bootstrap to N={N_NEWTON}...")

    from exp_02_lanford_truncation import solve_f64
    prev_c = None
    for n in range(3, N_NEWTON + 1):
        alpha, c_sol, ok = solve_f64(n, initial_guess=prev_c)
        if ok:
            prev_c = c_sol
        else:
            N_NEWTON = n - 1
            print(f"    Float64 stopped at N={N_NEWTON}")
            break

    print(f"    Float64: N={N_NEWTON}, seeding mpmath...")

    print(f"\n  Phase 2: mpmath Newton at N={N_NEWTON}, {DPS} dps...")
    t0 = time.time()
    c_mp40, alpha_mp = solve_newton(N_NEWTON, prev_c, DPS)
    t_newton = time.time() - t0

    mp.dps = DPS
    alpha_exact = mpf(ALPHA_EXACT_STR)
    err40 = fabs(alpha_mp - alpha_exact) / alpha_exact
    digits40 = float(-mplog10(err40)) if err40 > 0 else DPS
    print(f"    Newton done: {digits40:.1f} digits of alpha in {t_newton:.1f}s")

    # =========================================================================
    # PHASE 3: March coefficients from 41 to 80
    # =========================================================================
    print(f"\n  Phase 3: Marching coefficients from {N_NEWTON + 1} to {N_MARCH}...")
    t0 = time.time()
    c_full_list = [mpf(1)] + list(c_mp40)
    c_all = march_coefficients(c_full_list, alpha_mp, N_NEWTON + 1, N_MARCH, DPS)
    t_march = time.time() - t0
    print(f"    Marched to N={len(c_all) - 1} in {t_march:.1f}s")

    # Verify: alpha from full truncation
    g1_full = sum(c_all[:N_MARCH + 1])
    alpha_check = fabs(mpf(-1) / g1_full)
    err_check = fabs(alpha_check - alpha_exact) / alpha_exact
    digits_check = float(-mplog10(err_check)) if err_check > 0 else DPS
    print(f"    Alpha from N={N_MARCH}: {digits_check:.1f} digits")

    # =========================================================================
    # PHASE 4: Extract coefficient spectrum
    # =========================================================================
    N_eff = min(N_MARCH, len(c_all) - 1)
    abs_c = [fabs(c_all[k]) for k in range(1, N_eff + 1)]  # c_1..c_{N_eff}
    log_c = []
    for k in range(N_eff):
        if abs_c[k] > 0:
            log_c.append(float(mplog10(abs_c[k])))
        else:
            log_c.append(None)

    print(f"\n  Phase 4: Coefficient spectrum (N=1..{N_eff})")
    print(f"\n  {'k':>4s}  {'log10|c_k|':>14s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*5}")
    for k in range(N_eff):
        idx = k + 1
        lv = log_c[k]
        if lv is not None and lv > -110:
            is_fib = idx in FIB_SET and idx > 2
            marker = " <<<" if is_fib else ""
            if idx <= 25 or idx >= N_eff - 5 or is_fib or idx % 10 == 0:
                print(f"  {idx:4d}  {lv:14.4f}  {marker}")

    # =========================================================================
    # PHASE 5: Marginal information per coefficient
    # =========================================================================
    print(f"\n  Phase 5: Marginal digits of alpha per truncation order")

    digits_data = []
    prev_d = 0
    for N_cur in range(3, N_eff + 1):
        c_trunc = c_all[:N_cur + 1]
        g1 = sum(c_trunc)
        if fabs(g1) < mpf(10) ** (-80):
            continue
        alpha_N = fabs(mpf(-1) / g1)
        err_N = fabs(alpha_N - alpha_exact) / alpha_exact
        d = float(-mplog10(err_N)) if err_N > 0 else DPS
        marginal = d - prev_d
        prev_d = d
        is_fib = N_cur in FIB_SET
        digits_data.append({
            'N': N_cur, 'digits': d, 'marginal': marginal, 'is_fib': is_fib
        })

    print(f"\n  {'N':>4s}  {'total':>10s}  {'marginal':>10s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*5}")
    for d in digits_data:
        marker = " <<<" if d['is_fib'] else ""
        if d['N'] <= 15 or d['is_fib'] or d['N'] % 10 == 0 or d['N'] >= N_eff - 5:
            print(f"  {d['N']:4d}  {d['digits']:10.3f}  {d['marginal']:+10.4f}  {marker}")

    # Separate Fibonacci vs non-Fibonacci
    fib_m = [(d['N'], d['marginal']) for d in digits_data if d['is_fib'] and d['N'] > 3]
    nonfib_m = [(d['N'], d['marginal']) for d in digits_data if not d['is_fib'] and d['N'] > 3]
    print(f"\n  Fibonacci marginals (N>3): {', '.join(f'N={n}:{m:+.3f}' for n, m in fib_m)}")
    print(f"  Fibonacci mean: {np.mean([m for _, m in fib_m]):+.4f}")
    print(f"  Non-Fibonacci mean: {np.mean([m for _, m in nonfib_m]):+.4f}")

    # =========================================================================
    # PHASE 6: PERMUTATION TEST
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PERMUTATION TEST: Is Fibonacci marginal suppression significant?")
    print("=" * 72)

    # Build arrays for permutation test
    all_marginals = [d['marginal'] for d in digits_data if d['N'] > 3]
    all_N_vals = [d['N'] for d in digits_data if d['N'] > 3]
    fib_positions = [i for i, N in enumerate(all_N_vals) if N in FIB_SET]

    if len(fib_positions) >= 3 and len(all_marginals) >= 10:
        ptest = permutation_test(all_marginals, fib_positions, n_perms=100000)
        print(f"\n  Fibonacci positions tested: {[all_N_vals[i] for i in fib_positions]}")
        print(f"  Fibonacci mean marginal: {ptest['fib_mean']:+.4f}")
        print(f"  Random mean marginal:    {ptest['random_mean']:+.4f} +/- {ptest['random_std']:.4f}")
        print(f"  Z-score: {ptest['z_score']:.2f}")
        print(f"  p-value: {ptest['p_value']:.6f} ({ptest['n_perms']} permutations)")
        if ptest['p_value'] < 0.01:
            print(f"  SIGNIFICANT at p < 0.01")
        elif ptest['p_value'] < 0.05:
            print(f"  SIGNIFICANT at p < 0.05")
        elif ptest['p_value'] < 0.1:
            print(f"  MARGINAL (p < 0.1)")
        else:
            print(f"  NOT SIGNIFICANT (p = {ptest['p_value']:.3f})")
    else:
        ptest = None
        print(f"\n  Insufficient data for permutation test")

    # =========================================================================
    # PHASE 7: SUPPRESSION PATTERN vs FIBONACCI INDEX
    # =========================================================================
    print("\n" + "=" * 72)
    print("  SUPPRESSION PATTERN: marginal digits vs Fibonacci index")
    print("=" * 72)

    fib_marginals_indexed = []
    for d in digits_data:
        if d['is_fib'] and d['N'] > 3 and d['N'] in FIB_INDEX:
            fib_marginals_indexed.append((FIB_INDEX[d['N']], d['marginal']))

    if fib_marginals_indexed:
        print(f"\n  {'F_k':>4s}  {'k_idx':>6s}  {'marginal':>10s}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*10}")
        for k_idx, marg in fib_marginals_indexed:
            F_k = FIBONACCI[k_idx] if k_idx < len(FIBONACCI) else '?'
            print(f"  {F_k:>4}  {k_idx:>6d}  {marg:+10.4f}")

        sp = suppression_pattern(fib_marginals_indexed)
        if sp:
            print(f"\n  Linear fit: marginal = {sp['slope']:.4f} * k + {sp['intercept']:.4f}")
            print(f"  R^2 = {sp['r_squared']:.4f}")
            if sp['k_zero_crossing']:
                print(f"  Zero crossing at k = {sp['k_zero_crossing']:.1f} "
                      f"(F_{int(sp['k_zero_crossing'])} ~ {int(PHI**sp['k_zero_crossing']/np.sqrt(5)+0.5)})")
            print(f"\n  Slope comparisons:")
            print(f"    slope / (-1/phi) = {sp['slope_vs_neg_inv_phi']:.4f}")
            print(f"    slope / (-ln(phi)) = {sp['slope_vs_neg_ln_phi']:.4f}")
            print(f"    (1.0 would mean slope = -1/phi or -ln(phi) respectively)")

            # Predict marginal at F_10=55 (k=10)
            pred_10 = sp['slope'] * 10 + sp['intercept']
            print(f"\n  Predicted marginal at F_10=55 (k=10): {pred_10:+.3f} digits")
            if 55 in FIB_SET:
                actual = [m for d in digits_data if d['N'] == 55 for m in [d['marginal']]]
                if actual:
                    print(f"  Actual marginal at N=55: {actual[0]:+.3f} digits")

    # =========================================================================
    # PHASE 8: NEIGHBOR ANALYSIS (extended to N=80)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  NEIGHBOR ANALYSIS (extended)")
    print("=" * 72)

    # Build log|c_k| lookup
    log_lookup = {}
    for k in range(N_eff):
        if log_c[k] is not None and log_c[k] > -110:
            log_lookup[k + 1] = log_c[k]

    fibs_extended = sorted([f for f in FIB_SET if f > 2 and f in log_lookup])
    print(f"\n  {'F_k':>4s}  {'k_idx':>6s}  {'log|c_k|':>10s}  {'log(nbrs)':>10s}  "
          f"{'delta':>8s}  {'suppressed?':>12s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")

    neighbor_data = []
    for f in fibs_extended:
        nbrs = []
        for offset in [-2, -1, 1, 2]:
            nk = f + offset
            if nk in log_lookup:
                nbrs.append(log_lookup[nk])
        if len(nbrs) < 2:
            continue
        nbr_mean = np.mean(nbrs)
        delta = log_lookup[f] - nbr_mean
        suppressed = "YES" if delta < -0.15 else "no"
        k_idx = FIB_INDEX.get(f, '?')
        neighbor_data.append({'F_k': f, 'k_idx': k_idx, 'delta': delta})
        print(f"  {f:4d}  {str(k_idx):>6s}  {log_lookup[f]:10.4f}  {nbr_mean:10.4f}  "
              f"{delta:+8.4f}  {suppressed:>12s}")

    if neighbor_data:
        mean_delta = np.mean([d['delta'] for d in neighbor_data])
        print(f"\n  Mean neighbor delta: {mean_delta:+.4f}")
        n_suppressed = sum(1 for d in neighbor_data if d['delta'] < -0.15)
        print(f"  Suppressed (delta < -0.15): {n_suppressed}/{len(neighbor_data)}")

    # =========================================================================
    # SAVE
    # =========================================================================
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    fib_mean_marg = np.mean([m for _, m in fib_m]) if fib_m else None
    nonfib_mean_marg = np.mean([m for _, m in nonfib_m]) if nonfib_m else None
    print(f"\n  Coefficients extracted: N=1..{N_eff}")
    print(f"  Alpha precision: {digits_check:.1f} digits")
    if fib_mean_marg is not None:
        print(f"  Fibonacci mean marginal: {fib_mean_marg:+.4f} digits")
        print(f"  Non-Fibonacci mean marginal: {nonfib_mean_marg:+.4f} digits")
    if ptest:
        print(f"  Permutation test p-value: {ptest['p_value']:.6f}")
    if sp:
        print(f"  Suppression slope: {sp['slope']:.4f} per Fibonacci index")

    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_02d_extended_analysis.py',
        'parameters': {'N_newton': N_NEWTON, 'N_march': N_MARCH, 'dps': DPS},
        'alpha_digits': digits_check,
        'n_coefficients': N_eff,
        'marginals': {
            'fib_mean': float(fib_mean_marg) if fib_mean_marg else None,
            'nonfib_mean': float(nonfib_mean_marg) if nonfib_mean_marg else None,
        },
        'permutation_test': ptest,
        'suppression_pattern': sp,
        'neighbor_analysis': neighbor_data,
        'digits_per_N': digits_data,
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_02d_extended_analysis_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"\n  Saved: {fpath}")


if __name__ == "__main__":
    run()
