#!/usr/bin/env python3
"""
exp_02_lanford_truncation.py
==============================

FEIGENBAUM RG FIXED POINT AT FIBONACCI TRUNCATION DIMENSIONS

HYPOTHESIS:
    The convergence of |alpha_N| (from solving the Feigenbaum functional equation
    at polynomial truncation order N) shows structure at Fibonacci dimensions --
    specifically at N = F_10 = 55.

BACKGROUND:
    The Feigenbaum functional equation T[g](x) = -alpha * g(g(-x/alpha))
    has a unique fixed point g* with g*(0) = 1. Lanford (1982) proved existence
    by truncating g to degree 2N even polynomial and solving the nonlinear system.

    At truncation order N, we get alpha_N. As N increases, alpha_N -> alpha_exact.
    The question: does convergence have structure at Fibonacci N?

METHOD:
    g(x) = 1 + c_1*x^2 + c_2*x^4 + ... + c_N*x^{2N}
    alpha = -1/g(1) = -1/(1 + sum c_k)
    Match T[g] = g coefficient by coefficient -> nonlinear system in {c_k}.
    Sweep N = 3 to n_max. Use mpmath for extended precision if available.

THREE POSSIBLE OUTCOMES:
    1. Local minimum/inflection at N=55: Fibonacci dimensionality in RG
    2. Smooth convergence through 55: F_10 connects elsewhere
    3. Saturation before 55: F_10 about orbit structure, not truncation
"""

import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

# Try mpmath for extended precision
try:
    from mpmath import mp, mpf, matrix as mpmatrix, lu_solve, norm
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# =============================================================================
# KNOWN VALUES (|alpha|, positive)
# =============================================================================

# High-precision string for mpmath comparison (OEIS A006891)
ALPHA_EXACT_STR = "2.5029078750958928222839028732182157863646264378071999356799944268"
ALPHA_EXACT = float(ALPHA_EXACT_STR[:18])  # float64 for quick comparison
DELTA_EXACT = 4.669201609102990671853203820447323546553756938557

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
FIB_SET = set(FIBONACCI)


# =============================================================================
# FLOAT64 IMPLEMENTATION
# =============================================================================

def feigenbaum_system_f64(c_vec, N):
    """Feigenbaum functional equation residual at truncation N (float64)."""
    c = np.zeros(N + 1)
    c[0] = 1.0
    c[1:] = c_vec

    g_at_1 = np.sum(c)
    if abs(g_at_1) < 1e-15:
        return np.ones(N) * 1e10

    alpha = -1.0 / g_at_1
    max_deg = 2 * N

    # g(x) as full polynomial (even powers only)
    g_full = np.zeros(max_deg + 1)
    for k in range(N + 1):
        g_full[2*k] = c[k]

    # g(-x/alpha): since g is even, g(-x/alpha) = g(x/alpha)
    # coefficient of x^{2k} is c[k] / alpha^{2k}
    g_of_u = np.zeros(max_deg + 1)
    for k in range(N + 1):
        g_of_u[2*k] = c[k] / alpha**(2*k)

    # Compose g(g_of_u(x)): polynomial composition truncated to max_deg
    result = np.zeros(max_deg + 1)
    power = np.zeros(max_deg + 1)
    power[0] = 1.0  # (g_of_u)^0

    for k in range(N + 1):
        if abs(c[k]) > 1e-300:
            result[:max_deg+1] += c[k] * power[:max_deg+1]
        if k < N:
            # Multiply power by g_of_u^2 (since g has even terms, power index = x^{2k})
            # Actually we need (g_of_u)^1 at each step for general composition
            # g(y) = sum c[k] y^{2k}, y = g_of_u(x)
            # We need y^{2k} = (g_of_u)^{2k}
            # This means power should track (g_of_u)^{2k}
            pass

    # SIMPLER APPROACH: direct composition via coefficient matching
    # g(y) = 1 + c1*y^2 + c2*y^4 + ... where y = g(-x/alpha)
    # y^2 = (g(-x/alpha))^2, etc.
    # Build powers of y = g_of_u(x) up to y^{2N}

    # Reset and do properly
    y = g_of_u  # y(x) = g(-x/alpha) as polynomial in x

    # Compute y^j for j = 0, 2, 4, ..., 2N
    # y_powers[j] = y^j as polynomial truncated to degree max_deg
    y_pow = [None] * (2*N + 1)
    y_pow[0] = np.zeros(max_deg + 1)
    y_pow[0][0] = 1.0
    y_pow[1] = y.copy()

    for j in range(2, 2*N + 1):
        yp = np.convolve(y_pow[j-1], y)[:max_deg+1]
        if len(yp) < max_deg + 1:
            yp = np.concatenate([yp, np.zeros(max_deg + 1 - len(yp))])
        y_pow[j] = yp

    # g(y) = sum_{k=0}^{N} c[k] * y^{2k}
    g_of_y = np.zeros(max_deg + 1)
    for k in range(N + 1):
        if abs(c[k]) > 1e-300:
            g_of_y += c[k] * y_pow[2*k][:max_deg+1]

    # T[g](x) = -alpha * g(g(-x/alpha))
    T_g = -alpha * g_of_y

    # Extract even coefficients and compute residual
    residual = np.zeros(N)
    for k in range(1, N + 1):
        T_k = T_g[2*k] if 2*k <= max_deg else 0.0
        residual[k-1] = T_k - c[k]

    return residual


def solve_f64(N, initial_guess=None):
    """Solve Feigenbaum at truncation N using float64."""
    if initial_guess is None:
        c0 = np.zeros(N)
        c0[0] = -1.528
        if N > 1: c0[1] = 0.105
        if N > 2: c0[2] = 0.027
    else:
        c0 = np.zeros(N)
        n_copy = min(len(initial_guess), N)
        c0[:n_copy] = initial_guess[:n_copy]

    try:
        c_sol, info, ier, msg = fsolve(
            feigenbaum_system_f64, c0, args=(N,),
            full_output=True, maxfev=10000
        )
        if ier != 1:
            return None, None, False

        c_full = np.zeros(N + 1)
        c_full[0] = 1.0
        c_full[1:] = c_sol
        alpha = abs(-1.0 / np.sum(c_full))

        residual = feigenbaum_system_f64(c_sol, N)
        max_res = float(np.max(np.abs(residual)))

        return float(alpha), c_sol, max_res < 1e-8

    except Exception:
        return None, None, False


# =============================================================================
# MPMATH IMPLEMENTATION (extended precision)
# =============================================================================

def solve_mpmath(N, dps=50, initial_guess=None):
    """Solve Feigenbaum at truncation N using mpmath arbitrary precision."""
    if not HAS_MPMATH:
        return None, None, False

    mp.dps = dps

    # Initial guess
    if initial_guess is not None:
        c_init = [mpf(str(x)) for x in initial_guess]
        while len(c_init) < N:
            c_init.append(mpf(0))
        c_init = c_init[:N]
    else:
        c_init = [mpf(0)] * N
        c_init[0] = mpf('-1.528')
        if N > 1: c_init[1] = mpf('0.105')
        if N > 2: c_init[2] = mpf('0.027')

    def system(c_vec):
        """Compute residual using mpmath."""
        c = [mpf(1)] + list(c_vec)
        g_at_1 = sum(c)
        if abs(g_at_1) < mpf(10)**(-dps+5):
            return [mpf(10)**10] * N

        alpha = mpf(-1) / g_at_1
        max_pow = 2 * N

        # y(x) = g(-x/alpha) = sum c[k] x^{2k} / alpha^{2k} (even function)
        y_coeffs = [mpf(0)] * (max_pow + 1)
        for k in range(N + 1):
            if 2*k <= max_pow:
                y_coeffs[2*k] = c[k] / alpha**(2*k)

        # Build powers of y(x) up to y^{2N}
        def poly_mul_trunc(a, b, maxd):
            result = [mpf(0)] * (maxd + 1)
            for i in range(min(len(a), maxd+1)):
                if a[i] == 0:
                    continue
                for j in range(min(len(b), maxd+1-i)):
                    if b[j] == 0:
                        continue
                    result[i+j] += a[i] * b[j]
            return result

        y_pow = [[mpf(0)] * (max_pow + 1) for _ in range(2*N + 1)]
        y_pow[0][0] = mpf(1)
        if N > 0:
            y_pow[1] = y_coeffs[:]

        for j in range(2, 2*N + 1):
            y_pow[j] = poly_mul_trunc(y_pow[j-1], y_coeffs, max_pow)

        # g(y) = sum c[k] * y^{2k}
        g_of_y = [mpf(0)] * (max_pow + 1)
        for k in range(N + 1):
            if c[k] == 0:
                continue
            for i in range(max_pow + 1):
                g_of_y[i] += c[k] * y_pow[2*k][i]

        # T[g] = -alpha * g(g(-x/alpha))
        T_g = [-alpha * g_of_y[i] for i in range(max_pow + 1)]

        # Residual: T[g]_{2k} - c[k] for k=1..N
        residual = []
        for k in range(1, N + 1):
            T_k = T_g[2*k] if 2*k <= max_pow else mpf(0)
            residual.append(T_k - c[k])

        return residual

    # Simple Newton iteration
    c = list(c_init)
    converged = False

    for iteration in range(100):
        res = system(c)
        res_norm = max(abs(r) for r in res)

        if res_norm < mpf(10)**(-dps + 10):
            converged = True
            break

        # Numerical Jacobian
        eps = mpf(10)**(-dps//2)
        J = []
        for j in range(N):
            c_pert = list(c)
            c_pert[j] += eps
            res_pert = system(c_pert)
            J_col = [(res_pert[i] - res[i]) / eps for i in range(N)]
            J.append(J_col)

        # Solve J * delta = -res
        J_mat = mpmatrix(N, N)
        for i in range(N):
            for j in range(N):
                J_mat[i, j] = J[j][i]  # J[j] is column j

        rhs = mpmatrix(N, 1)
        for i in range(N):
            rhs[i] = -res[i]

        try:
            delta = lu_solve(J_mat, rhs)
            for j in range(N):
                c[j] += delta[j]
        except Exception:
            break

    if not converged:
        return None, None, False

    c_full = [mpf(1)] + c
    alpha_mp = abs(mpf(-1) / sum(c_full))

    # Compute error in full precision
    alpha_exact_mp = mpf(ALPHA_EXACT_STR)
    error_mp = abs(alpha_mp - alpha_exact_mp) / alpha_exact_mp

    return float(alpha_mp), [float(x) for x in c], True, float(error_mp)


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_experiment(n_min=3, n_max=60, use_mpmath=False, dps=50):
    print()
    print("=" * 72)
    print("  EXPERIMENT 02: LANFORD TRUNCATION AT FIBONACCI DIMENSIONS")
    print("=" * 72)
    print()
    print(f"  Sweeping N = {n_min} to {n_max}")
    print(f"  Fibonacci markers: {[f for f in FIBONACCI if n_min <= f <= n_max]}")
    print(f"  Precision: {'mpmath ' + str(dps) + ' dps' if use_mpmath else 'float64'}")
    print(f"  |alpha_exact| = {ALPHA_EXACT:.15f}")
    print()

    results = []
    prev_c = None
    t0 = time.time()

    for N in range(n_min, n_max + 1):
        t1 = time.time()

        if use_mpmath and HAS_MPMATH:
            result = solve_mpmath(N, dps=dps, initial_guess=prev_c)
            alpha_N, c_sol, ok = result[0], result[1], result[2]
            mp_error = result[3] if len(result) > 3 else None
        else:
            alpha_N, c_sol, ok = solve_f64(N, initial_guess=prev_c)
            mp_error = None

        dt = time.time() - t1

        if alpha_N is not None and ok:
            # Use mpmath error if available (higher precision)
            error = mp_error if mp_error is not None else abs(alpha_N - ALPHA_EXACT) / ALPHA_EXACT
            is_fib = N in FIB_SET
            log_err = np.log10(error) if error > 1e-300 else -300

            results.append({
                'N': N, 'alpha_N': alpha_N, 'error': error,
                'log10_error': log_err, 'is_fibonacci': is_fib,
                'converged': True, 'time': dt,
            })

            fib_mark = "  <<<" if is_fib else ""
            print(f"    N={N:3d}: |alpha| = {alpha_N:.15f}, "
                  f"error = {error:.2e}, "
                  f"log10 = {log_err:.2f}  "
                  f"({dt:.2f}s){fib_mark}")

            if c_sol is not None:
                prev_c = c_sol if isinstance(c_sol, list) else c_sol.tolist()
        else:
            results.append({
                'N': N, 'alpha_N': None, 'error': None,
                'log10_error': None, 'is_fibonacci': N in FIB_SET,
                'converged': False, 'time': dt,
            })
            print(f"    N={N:3d}: FAILED  ({dt:.2f}s)")

    elapsed = time.time() - t0

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    converged = [r for r in results if r['converged'] and r['error'] is not None]

    print("\n" + "=" * 72)
    print("  CONVERGENCE ANALYSIS")
    print("=" * 72 + "\n")

    if not converged:
        print("  No converged results!")
        return

    best_err = min(r['error'] for r in converged)
    best_digits = max(0, -int(np.log10(best_err))) if best_err > 0 else '>50'
    print(f"  Best |alpha| error: {best_err:.2e} (~{best_digits} digits)")
    print()

    # Per-step convergence (the KEY observable)
    pre_sat = [r for r in converged if r['log10_error'] > -200]
    if len(pre_sat) >= 2:
        print("  Per-step improvement (digits gained at each N):")
        increments = []
        for i in range(1, len(pre_sat)):
            gain = pre_sat[i-1]['log10_error'] - pre_sat[i]['log10_error']
            is_fib = pre_sat[i]['is_fibonacci']
            marker = "  <<< FIBONACCI" if is_fib else ""
            increments.append((pre_sat[i]['N'], gain, is_fib))
            print(f"    N={pre_sat[i]['N']:3d}: +{gain:.4f} digits{marker}")

        # Compare Fibonacci vs non-Fibonacci gains
        fib_gains = [g for _, g, fib in increments if fib]
        nonfib_gains = [g for _, g, fib in increments if not fib]
        if fib_gains and nonfib_gains:
            print(f"\n  Fibonacci mean gain:     {np.mean(fib_gains):.4f} digits")
            print(f"  Non-Fibonacci mean gain: {np.mean(nonfib_gains):.4f} digits")
            ratio = np.mean(fib_gains) / np.mean(nonfib_gains)
            print(f"  Ratio (Fib/non-Fib):     {ratio:.4f}")
            if ratio < 0.5:
                print(f"  OBSERVATION: Fibonacci N values show reduced convergence rate")
            elif ratio > 1.5:
                print(f"  OBSERVATION: Fibonacci N values show enhanced convergence rate")

    # Fibonacci analysis
    fib_results = [r for r in converged if r['is_fibonacci'] and r['N'] > 2]
    if fib_results:
        print(f"\n  Fibonacci N error values:")
        for fr in fib_results:
            print(f"    N={fr['N']:3d}: error = {fr['error']:.6e}, "
                  f"log10 = {fr['log10_error']:.4f}")

    # Convergence rate (pre-saturation)
    pre_sat = [r for r in converged if r['error'] > 1e-14]
    if len(pre_sat) >= 3:
        Ns = np.array([r['N'] for r in pre_sat])
        log_errs = np.array([r['log10_error'] for r in pre_sat])
        valid = np.isfinite(log_errs)
        if np.sum(valid) >= 3:
            coeffs = np.polyfit(Ns[valid], log_errs[valid], 1)
            print(f"\n  Convergence rate (pre-saturation):")
            print(f"    log10(error) ~ {coeffs[0]:.4f} * N + {coeffs[1]:.4f}")
            print(f"    Rate: ~{abs(coeffs[0]):.3f} digits per unit N")

    # Check if we need mpmath
    if not use_mpmath:
        print(f"\n  NOTE: Float64 precision limited. "
              f"Re-run with --mpmath to see full convergence structure.")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # =========================================================================
    # SAVE
    # =========================================================================

    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_02_lanford_truncation.py',
        'parameters': {
            'n_min': n_min, 'n_max': n_max,
            'precision': f'mpmath {dps} dps' if use_mpmath else 'float64',
            'alpha_exact': ALPHA_EXACT,
        },
        'results': results,
        'analysis': {
            'best_N': max(r['N'] for r in converged) if converged else None,
            'best_error': min(r['error'] for r in converged) if converged else None,
            'n_converged': len(converged),
        },
        'elapsed_seconds': elapsed,
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'mpmath' if use_mpmath else 'f64'
    fpath = results_dir / f'exp_02_lanford_truncation_{mode}_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"  Saved: {fpath}")
    return output


if __name__ == "__main__":
    use_mp = '--mpmath' in sys.argv
    n_max = 60
    dps = 100

    for arg in sys.argv[1:]:
        if arg.startswith('--') and arg != '--mpmath':
            continue
        try:
            n_max = int(arg)
        except ValueError:
            pass

    if use_mp and not HAS_MPMATH:
        print("ERROR: mpmath not installed. pip install mpmath")
        sys.exit(1)

    run_experiment(n_min=3, n_max=n_max, use_mpmath=use_mp, dps=dps)
