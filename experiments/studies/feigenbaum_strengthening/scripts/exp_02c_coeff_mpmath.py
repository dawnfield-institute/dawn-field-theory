#!/usr/bin/env python3
"""
exp_02c_coeff_mpmath.py
========================

High-precision coefficient extraction from Feigenbaum RG fixed point.

Strategy: solve at N=40 in float64 (fast), then refine with mpmath at 80 dps
(2-3 Newton steps). Extract |c_k| for k=1..40 and analyze Fibonacci gap structure.

KEY QUESTION: Are Fibonacci-indexed coefficients c_{F_k} systematically
suppressed relative to the smooth envelope? If so, this explains why the
Feigenbaum formula uses 55 = F_10.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from mpmath import mp, mpf, matrix as mpmatrix, lu_solve, log10 as mplog10, fabs

ALPHA_EXACT_STR = "2.5029078750958928222839028732182157863646264378071999356799944268"
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
FIB_SET = set(FIBONACCI)
FIB_INDEX = {1:1, 2:3, 3:4, 5:5, 8:6, 13:7, 21:8, 34:9, 55:10, 89:11}


def poly_mul_trunc_mp(a, b, maxd):
    """Multiply two polynomials in mpf, truncated to degree maxd."""
    result = [mpf(0)] * (maxd + 1)
    la = min(len(a), maxd + 1)
    lb = len(b)
    for i in range(la):
        if a[i] == 0: continue
        for j in range(min(lb, maxd + 1 - i)):
            if b[j] == 0: continue
            result[i + j] += a[i] * b[j]
    return result


def feigenbaum_residual_mp(c_vec, N):
    """Compute Feigenbaum RG residual in mpmath."""
    c = [mpf(1)] + list(c_vec)
    g_at_1 = sum(c)
    alpha = mpf(-1) / g_at_1
    max_pow = 2 * N

    # y(x) = g(-x/alpha) = g(x/alpha) since g is even
    y = [mpf(0)] * (max_pow + 1)
    for k in range(N + 1):
        if 2 * k <= max_pow:
            y[2 * k] = c[k] / alpha ** (2 * k)

    # Build y^j for j=0..2N
    y_pow = [[mpf(0)] * (max_pow + 1) for _ in range(2 * N + 1)]
    y_pow[0][0] = mpf(1)
    y_pow[1] = y[:]
    for j in range(2, 2 * N + 1):
        y_pow[j] = poly_mul_trunc_mp(y_pow[j - 1], y, max_pow)

    # g(y) = sum c[k] * y^{2k}
    g_of_y = [mpf(0)] * (max_pow + 1)
    for k in range(N + 1):
        if c[k] == 0: continue
        for i in range(max_pow + 1):
            g_of_y[i] += c[k] * y_pow[2 * k][i]

    # T[g](x) = -alpha * g(g(-x/alpha))
    T_g = [-alpha * g_of_y[i] for i in range(max_pow + 1)]

    # Residual: T[g]_{2k} - c[k] for k=1..N
    residual = []
    for k in range(1, N + 1):
        T_k = T_g[2 * k] if 2 * k <= max_pow else mpf(0)
        residual.append(T_k - c[k])

    return residual, alpha


def solve_mpmath_from_seed(N, seed_c, dps=80, max_iter=30):
    """Refine float64 solution using mpmath Newton iteration."""
    mp.dps = dps
    c = [mpf(str(x)) for x in seed_c[:N]]
    while len(c) < N:
        c.append(mpf(0))

    for iteration in range(max_iter):
        res, alpha = feigenbaum_residual_mp(c, N)
        res_norm = max(fabs(r) for r in res)

        if iteration % 5 == 0:
            alpha_exact = mpf(ALPHA_EXACT_STR)
            err = fabs(fabs(alpha) - alpha_exact) / alpha_exact
            digits = float(-mplog10(err)) if err > 0 else dps
            print(f"    iter {iteration:2d}: residual = {float(res_norm):.2e}, "
                  f"alpha digits = {digits:.1f}")

        if res_norm < mpf(10) ** (-dps + 10):
            break

        # Numerical Jacobian
        eps = mpf(10) ** (-(dps * 2) // 3)
        J_cols = []
        for j in range(N):
            c_pert = list(c)
            c_pert[j] += eps
            res_p, _ = feigenbaum_residual_mp(c_pert, N)
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
        except Exception as e:
            print(f"    Newton failed at iter {iteration}: {e}")
            break

    alpha_final = fabs(mpf(-1) / sum([mpf(1)] + c))
    return c, alpha_final


def run():
    print("=" * 72)
    print("  FEIGENBAUM COEFFICIENT GAP ANALYSIS")
    print("=" * 72)

    # Step 1: Float64 bootstrap to N=40
    from exp_02_lanford_truncation import solve_f64
    N = 40
    DPS = 80

    print(f"\n  Phase 1: Float64 bootstrap to N={N}...")
    prev_c = None
    for n in range(3, N + 1):
        alpha, c_sol, ok = solve_f64(n, initial_guess=prev_c)
        if ok:
            prev_c = c_sol
        else:
            print(f"    Float64 failed at N={n}")
            N = n - 1
            break

    f64_alpha_err = abs(alpha - float(ALPHA_EXACT_STR[:18])) / float(ALPHA_EXACT_STR[:18])
    print(f"  Float64: N={N}, alpha error = {f64_alpha_err:.2e}")

    # Step 2: mpmath refinement
    print(f"\n  Phase 2: mpmath refinement at {DPS} dps...")
    t0 = time.time()
    c_mp, alpha_mp = solve_mpmath_from_seed(N, prev_c, dps=DPS)
    elapsed = time.time() - t0

    mp.dps = DPS
    alpha_exact = mpf(ALPHA_EXACT_STR)
    final_err = fabs(alpha_mp - alpha_exact) / alpha_exact
    final_digits = float(-mplog10(final_err)) if final_err > 0 else DPS
    print(f"  mpmath: {final_digits:.1f} digits of alpha in {elapsed:.1f}s")

    # Step 3: Extract and analyze coefficient spectrum
    print("\n" + "=" * 72)
    print("  COEFFICIENT SPECTRUM")
    print("=" * 72)

    abs_c = [fabs(x) for x in c_mp]
    log_c = []
    for k in range(N):
        if abs_c[k] > 0:
            log_c.append(float(mplog10(abs_c[k])))
        else:
            log_c.append(None)

    print(f"\n  {'k':>4s}  {'log10|c_k|':>14s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*5}")

    valid_k = []
    valid_log = []
    for k in range(N):
        idx = k + 1
        lv = log_c[k]
        if lv is not None and lv > -70:
            valid_k.append(idx)
            valid_log.append(lv)
            is_fib = idx in FIB_SET and idx > 1
            marker = " <<<" if is_fib else ""
            print(f"  {idx:4d}  {lv:14.4f}  {marker}")

    # Step 4: Envelope fit
    print("\n" + "=" * 72)
    print("  ENVELOPE FIT")
    print("=" * 72)

    vk = np.array(valid_k, dtype=float)
    vl = np.array(valid_log, dtype=float)

    # Try linear and quadratic
    c1 = np.polyfit(vk, vl, 1)
    c2 = np.polyfit(vk, vl, 2)
    rss1 = np.sum((vl - np.polyval(c1, vk)) ** 2)
    rss2 = np.sum((vl - np.polyval(c2, vk)) ** 2)

    print(f"\n  Linear:    log10|c_k| = {c1[0]:.6f}*k + {c1[1]:.4f}  (RSS={rss1:.4f})")
    print(f"  Quadratic: log10|c_k| = {c2[0]:.6f}*k^2 + {c2[1]:.4f}*k + {c2[2]:.4f}  (RSS={rss2:.4f})")

    use_quad = rss2 < rss1 * 0.7
    envelope = np.polyval(c2 if use_quad else c1, vk)
    fit_label = "quadratic" if use_quad else "linear"
    print(f"  -> Using {fit_label} envelope")

    phi = (1 + np.sqrt(5)) / 2
    print(f"\n  Decay base: 10^({c1[0]:.6f}) = {10**c1[0]:.6f}")
    print(f"  Compare: 1/phi = {1/phi:.6f}, 1/phi^2 = {1/phi**2:.6f}")

    # Step 5: Deviations from envelope
    print("\n" + "=" * 72)
    print("  DEVIATIONS FROM ENVELOPE")
    print("=" * 72)

    deviations = vl - envelope
    fib_mask = np.array([k in FIB_SET and k > 2 for k in valid_k])
    nonfib_mask = np.array([k not in FIB_SET and k > 2 for k in valid_k])

    print(f"\n  {'k':>4s}  {'log10|c_k|':>12s}  {'envelope':>10s}  {'deviation':>10s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*5}")
    for i in range(len(valid_k)):
        k = valid_k[i]
        is_fib = k in FIB_SET and k > 2
        marker = " <<<" if is_fib else ""
        print(f"  {k:4d}  {vl[i]:12.4f}  {envelope[i]:10.4f}  {deviations[i]:+10.4f}  {marker}")

    if np.any(fib_mask) and np.any(nonfib_mask):
        fib_dev_mean = np.mean(deviations[fib_mask])
        nonfib_dev_mean = np.mean(deviations[nonfib_mask])
        nonfib_dev_std = np.std(deviations[nonfib_mask])
        z = (fib_dev_mean - nonfib_dev_mean) / nonfib_dev_std if nonfib_dev_std > 0 else 0

        print(f"\n  Fibonacci mean deviation:     {fib_dev_mean:+.4f} orders of magnitude")
        print(f"  Non-Fibonacci mean deviation: {nonfib_dev_mean:+.4f} +/- {nonfib_dev_std:.4f}")
        print(f"  Z-score: {z:.2f}")
        if z < -2:
            print(f"  SIGNIFICANT: Fibonacci coefficients systematically suppressed (p < 0.02)")
        elif z < -1:
            print(f"  SUGGESTIVE: Fibonacci coefficients tend to be suppressed")

    # Step 6: Neighbor ratios (model-free)
    print("\n" + "=" * 72)
    print("  NEIGHBOR ANALYSIS (model-free)")
    print("=" * 72)

    # Build lookup: k -> log10|c_k|
    log_lookup = {valid_k[i]: valid_log[i] for i in range(len(valid_k))}

    fibs_in_range = sorted([f for f in FIB_SET if f > 2 and f in log_lookup])
    print(f"\n  For each Fibonacci k, compare |c_k| to geometric mean of c_{{k-2}}, c_{{k-1}}, c_{{k+1}}, c_{{k+2}}")
    print(f"\n  {'F_k':>4s}  {'k_idx':>6s}  {'log|c_k|':>10s}  {'log(nbrs)':>10s}  "
          f"{'delta':>8s}  {'ratio':>10s}  {'suppressed?':>12s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")

    neighbor_deltas = []
    for f in fibs_in_range:
        k_idx = FIB_INDEX.get(f, '?')
        nbrs = []
        for offset in [-2, -1, 1, 2]:
            nk = f + offset
            if nk in log_lookup:
                nbrs.append(log_lookup[nk])
        if not nbrs:
            continue

        nbr_mean = np.mean(nbrs)
        delta = log_lookup[f] - nbr_mean
        ratio = 10 ** delta
        suppressed = "YES" if delta < -0.15 else "no"
        neighbor_deltas.append({'F_k': f, 'k_idx': k_idx, 'delta': delta, 'ratio': ratio})
        print(f"  {f:4d}  {str(k_idx):>6s}  {log_lookup[f]:10.4f}  {nbr_mean:10.4f}  "
              f"{delta:+8.4f}  {ratio:10.4f}  {suppressed:>12s}")

    # Control: same analysis for ALL non-Fibonacci k > 2
    all_non_fib_deltas = []
    for k in valid_k:
        if k <= 2 or k in FIB_SET:
            continue
        nbrs = []
        for offset in [-2, -1, 1, 2]:
            nk = k + offset
            if nk in log_lookup:
                nbrs.append(log_lookup[nk])
        if len(nbrs) >= 2:
            delta = log_lookup[k] - np.mean(nbrs)
            all_non_fib_deltas.append(delta)

    if neighbor_deltas and all_non_fib_deltas:
        fib_mean_delta = np.mean([d['delta'] for d in neighbor_deltas])
        ctrl_mean = np.mean(all_non_fib_deltas)
        ctrl_std = np.std(all_non_fib_deltas)
        z_nb = (fib_mean_delta - ctrl_mean) / ctrl_std if ctrl_std > 0 else 0

        print(f"\n  Fibonacci mean delta: {fib_mean_delta:+.4f}")
        print(f"  Control mean delta:   {ctrl_mean:+.4f} +/- {ctrl_std:.4f}")
        print(f"  Z-score (vs control): {z_nb:.2f}")

    # Step 7: Marginal digits from truncation
    print("\n" + "=" * 72)
    print("  MARGINAL INFORMATION: digits of alpha per truncation order")
    print("=" * 72)

    digits_data = []
    prev_d = 0
    for N_cur in range(3, N + 1):
        c_trunc = c_mp[:N_cur]
        c_full = [mpf(1)] + list(c_trunc)
        g1 = sum(c_full)
        if fabs(g1) < mpf(10) ** (-50):
            continue
        alpha_N = fabs(mpf(-1) / g1)
        err = fabs(alpha_N - alpha_exact) / alpha_exact
        d = float(-mplog10(err)) if err > 0 else DPS
        marginal = d - prev_d
        prev_d = d
        is_fib = N_cur in FIB_SET
        digits_data.append({'N': N_cur, 'digits': d, 'marginal': marginal, 'is_fib': is_fib})

    print(f"\n  {'N':>4s}  {'total_digits':>12s}  {'marginal':>10s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*5}")
    for d in digits_data:
        marker = " <<<" if d['is_fib'] else ""
        print(f"  {d['N']:4d}  {d['digits']:12.3f}  {d['marginal']:+10.4f}  {marker}")

    fib_m = [d['marginal'] for d in digits_data if d['is_fib'] and d['N'] > 3]
    nonfib_m = [d['marginal'] for d in digits_data if not d['is_fib'] and d['N'] > 3]
    if fib_m and nonfib_m:
        print(f"\n  Fibonacci mean marginal:     {np.mean(fib_m):+.4f} digits")
        print(f"  Non-Fibonacci mean marginal: {np.mean(nonfib_m):+.4f} digits")

    # Step 8: Synthesis
    print("\n" + "=" * 72)
    print("  SYNTHESIS: Why does 55 = F_10 appear in the formula?")
    print("=" * 72)

    last_informative = 3
    for d in digits_data:
        if d['marginal'] > 0.01:
            last_informative = d['N']

    total = digits_data[-1]['digits'] if digits_data else 0
    fib_total = sum(d['marginal'] for d in digits_data if d['is_fib'])
    nonfib_total = sum(d['marginal'] for d in digits_data if not d['is_fib'])

    print(f"\n  Effective polynomial degree: N={last_informative} "
          f"(last coefficient adding >0.01 digits)")
    print(f"  Total precision: {total:.1f} digits")
    print(f"  Fibonacci terms contribute: {fib_total:.2f} digits ({100*fib_total/total:.1f}%)")
    print(f"  Non-Fibonacci terms contribute: {nonfib_total:.2f} digits ({100*nonfib_total/total:.1f}%)")

    # =========================================================================
    # SAVE
    # =========================================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_02c_coeff_mpmath.py',
        'parameters': {'N': N, 'dps': DPS},
        'alpha_digits': final_digits,
        'coefficients': [
            {'k': valid_k[i], 'log10_abs': valid_log[i],
             'is_fibonacci': valid_k[i] in FIB_SET,
             'deviation_from_envelope': float(deviations[i])}
            for i in range(len(valid_k))
        ],
        'neighbor_analysis': neighbor_deltas,
        'digits_per_N': digits_data,
        'synthesis': {
            'last_informative_N': last_informative,
            'total_digits': total,
            'fib_contribution_pct': 100 * fib_total / total if total > 0 else 0,
        }
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_02c_coeff_mpmath_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"\n  Saved: {fpath}")


if __name__ == "__main__":
    run()
