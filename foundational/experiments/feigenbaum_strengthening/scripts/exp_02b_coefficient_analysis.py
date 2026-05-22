#!/usr/bin/env python3
"""
exp_02b_coefficient_analysis.py
================================

COEFFICIENT GAP ANALYSIS: Why F_10 = 55 appears in the Feigenbaum formula.

The Feigenbaum RG fixed point g*(x) = 1 + sum c_k x^{2k} has Taylor
coefficients c_k that decay rapidly. The question: do Fibonacci-indexed
coefficients c_{F_k} sit systematically BELOW the smooth envelope?

If yes, Fibonacci positions mark "spectral dead spots" in the polynomial
expansion of g* -- positions where adding a new term contributes essentially
nothing. This would explain why the formula uses 55 = F_10: truncating at
a Fibonacci dimension captures the same information as a much higher-order
non-Fibonacci truncation.

METHOD:
    1. Solve g* at N=80 (float64), extract |c_1|...|c_80|
    2. Fit smooth envelope: log10|c_k| ~ a*k + b (exponential decay)
    3. Compute gap ratio: |c_k| / envelope(k) for each k
    4. Compare gap ratios at Fibonacci vs non-Fibonacci positions
    5. Check if gap depths follow a pattern (e.g., phi^{-k})
"""

import numpy as np
from scipy.optimize import fsolve
import json
from datetime import datetime
from pathlib import Path

# Import the solver from exp_02
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_02_lanford_truncation import solve_f64, FIBONACCI, FIB_SET, ALPHA_EXACT

def analyze_coefficients():
    print()
    print("=" * 72)
    print("  COEFFICIENT GAP ANALYSIS: g*(x) Taylor coefficients")
    print("=" * 72)

    # Solve at high truncation order to get stable coefficients
    N_max = 80
    print(f"\n  Solving Feigenbaum RG at N={N_max} (float64)...")

    # Bootstrap from low N to high N for stability
    prev_c = None
    last_good_c = None
    last_good_N = 0

    for N in range(3, N_max + 1):
        alpha_N, c_sol, ok = solve_f64(N, initial_guess=prev_c)
        if ok and alpha_N is not None:
            prev_c = c_sol
            last_good_c = c_sol
            last_good_N = N
        else:
            print(f"    N={N}: solver failed, using last good (N={last_good_N})")
            break

    if last_good_c is None:
        print("  ERROR: No solutions found!")
        return

    N_eff = last_good_N
    c = last_good_c
    alpha_err = abs(float(abs(-1.0 / (1.0 + np.sum(c)))) - ALPHA_EXACT) / ALPHA_EXACT
    print(f"  Solved to N={N_eff}, alpha error = {alpha_err:.2e}")

    # Extract |c_k| values
    abs_c = np.abs(c)
    indices = np.arange(1, len(c) + 1)

    print(f"\n  Coefficients c_1 through c_{len(c)}:")
    print(f"  {'k':>4s}  {'c_k':>14s}  {'|c_k|':>12s}  {'log10|c_k|':>12s}  {'Fib?':>5s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*5}")

    for k in range(len(c)):
        idx = k + 1
        is_fib = idx in FIB_SET
        if abs_c[k] > 0:
            log_val = np.log10(abs_c[k])
        else:
            log_val = -999
        marker = " <<<" if is_fib else ""
        if idx <= 40 or is_fib:  # Show first 40 + all Fibonacci
            print(f"  {idx:4d}  {c[k]:14.6e}  {abs_c[k]:12.4e}  {log_val:12.4f}  {marker}")

    # =========================================================================
    # ENVELOPE FIT
    # =========================================================================
    print("\n" + "=" * 72)
    print("  ENVELOPE ANALYSIS")
    print("=" * 72)

    # Use only non-zero coefficients for fit
    valid = abs_c > 1e-300
    k_valid = indices[valid]
    log_abs_c = np.log10(abs_c[valid])

    # Fit log10|c_k| = a*k + b (simple exponential decay)
    if len(k_valid) >= 5:
        coeffs = np.polyfit(k_valid, log_abs_c, 1)
        a, b = coeffs
        print(f"\n  Linear fit: log10|c_k| = {a:.4f} * k + ({b:.4f})")
        print(f"  Decay rate: |c_k| ~ 10^({a:.4f}*k) = {10**a:.6f}^k")
        print(f"  Compare to 1/phi^k: (1/phi)^k has base {1/((1+np.sqrt(5))/2):.6f}")

        envelope = 10**(a * indices + b)

        # Also try quadratic fit for better envelope
        if len(k_valid) >= 8:
            coeffs2 = np.polyfit(k_valid, log_abs_c, 2)
            a2, b2, c2_fit = coeffs2
            envelope_quad = 10**(a2 * indices**2 + b2 * indices + c2_fit)
            residuals_lin = np.sum((log_abs_c - (a * k_valid + b))**2)
            residuals_quad = np.sum((log_abs_c - (a2 * k_valid**2 + b2 * k_valid + c2_fit))**2)
            print(f"  Quadratic fit: log10|c_k| = {a2:.6f}*k^2 + {b2:.4f}*k + ({c2_fit:.4f})")
            print(f"  RSS linear: {residuals_lin:.4f}, quadratic: {residuals_quad:.4f}")

            # Use whichever fits better
            if residuals_quad < residuals_lin * 0.8:
                print(f"  Using quadratic envelope (significantly better fit)")
                envelope = envelope_quad
                fit_type = "quadratic"
            else:
                print(f"  Using linear envelope")
                fit_type = "linear"
        else:
            fit_type = "linear"

        # =====================================================================
        # GAP RATIOS
        # =====================================================================
        print("\n" + "=" * 72)
        print("  GAP RATIOS: |c_k| / envelope(k)")
        print("=" * 72)

        gap_ratio = abs_c / envelope[:len(abs_c)]

        print(f"\n  {'k':>4s}  {'|c_k|':>12s}  {'envelope':>12s}  {'ratio':>10s}  "
              f"{'log10(ratio)':>12s}  {'Fib?':>5s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*5}")

        fib_ratios = []
        nonfib_ratios = []
        fib_log_ratios = []
        nonfib_log_ratios = []

        for k in range(len(c)):
            idx = k + 1
            is_fib = idx in FIB_SET and idx > 2  # Skip 1,1,2 (too small to be meaningful)
            if abs_c[k] < 1e-300:
                continue
            r = gap_ratio[k]
            lr = np.log10(r) if r > 0 else -999

            if is_fib:
                fib_ratios.append(r)
                fib_log_ratios.append(lr)
            elif idx > 2:  # Same lower bound
                nonfib_ratios.append(r)
                nonfib_log_ratios.append(lr)

            if idx <= 40 or is_fib:
                marker = " <<<" if is_fib else ""
                print(f"  {idx:4d}  {abs_c[k]:12.4e}  {envelope[k]:12.4e}  "
                      f"{r:10.4f}  {lr:12.4f}  {marker}")

        print()
        if fib_ratios and nonfib_ratios:
            fib_mean = np.mean(fib_log_ratios)
            nonfib_mean = np.mean(nonfib_log_ratios)
            print(f"  Fibonacci mean log10(ratio):     {fib_mean:.4f}")
            print(f"  Non-Fibonacci mean log10(ratio): {nonfib_mean:.4f}")
            print(f"  Fibonacci geometric mean ratio:  {10**fib_mean:.6f}")
            print(f"  Non-Fibonacci geometric mean ratio: {10**nonfib_mean:.6f}")
            print(f"  Gap factor: {10**(nonfib_mean - fib_mean):.4f}x "
                  f"(Fibonacci coefficients are this much smaller than envelope)")

        # =====================================================================
        # NEIGHBOR COMPARISON (more robust than envelope)
        # =====================================================================
        print("\n" + "=" * 72)
        print("  NEIGHBOR COMPARISON: |c_{F_k}| vs geometric mean of neighbors")
        print("=" * 72)

        fibs_in_range = [f for f in FIBONACCI if 3 <= f <= N_eff - 2 and f < len(c)]
        print(f"\n  {'F_k':>4s}  {'|c_{F_k}|':>12s}  {'nbr_mean':>12s}  {'ratio':>10s}  "
              f"{'F_k is dead spot?':>18s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*18}")

        neighbor_ratios = []
        for f in fibs_in_range:
            idx = f - 1  # 0-based
            if abs_c[idx] < 1e-300:
                continue
            # Geometric mean of k-2, k-1, k+1, k+2
            neighbors = []
            for offset in [-2, -1, 1, 2]:
                ni = idx + offset
                if 0 <= ni < len(abs_c) and abs_c[ni] > 1e-300:
                    neighbors.append(abs_c[ni])
            if not neighbors:
                continue
            nbr_mean = np.exp(np.mean(np.log(neighbors)))
            ratio = abs_c[idx] / nbr_mean
            is_dead = ratio < 0.5
            neighbor_ratios.append((f, ratio))
            print(f"  {f:4d}  {abs_c[idx]:12.4e}  {nbr_mean:12.4e}  "
                  f"{ratio:10.6f}  {'YES' if is_dead else 'no':>18s}")

        # Random comparison: pick same number of random non-Fibonacci positions
        np.random.seed(42)
        non_fib_indices = [k for k in range(3, min(N_eff-2, len(c))+1) if k not in FIB_SET]
        n_fib = len(fibs_in_range)
        if len(non_fib_indices) >= n_fib * 10:
            random_ratios_all = []
            for _ in range(1000):  # Monte Carlo
                sample = np.random.choice(non_fib_indices, size=n_fib, replace=False)
                sample_ratios = []
                for s in sample:
                    idx = s - 1
                    if abs_c[idx] < 1e-300:
                        continue
                    neighbors = []
                    for offset in [-2, -1, 1, 2]:
                        ni = idx + offset
                        if 0 <= ni < len(abs_c) and abs_c[ni] > 1e-300:
                            neighbors.append(abs_c[ni])
                    if neighbors:
                        nbr_mean = np.exp(np.mean(np.log(neighbors)))
                        sample_ratios.append(abs_c[idx] / nbr_mean)
                if sample_ratios:
                    random_ratios_all.append(np.mean(sample_ratios))

            fib_mean_ratio = np.mean([r for _, r in neighbor_ratios])
            random_mean = np.mean(random_ratios_all)
            random_std = np.std(random_ratios_all)
            z_score = (fib_mean_ratio - random_mean) / random_std if random_std > 0 else 0

            print(f"\n  Monte Carlo comparison (1000 random samples of {n_fib} positions):")
            print(f"    Fibonacci mean neighbor ratio: {fib_mean_ratio:.6f}")
            print(f"    Random mean neighbor ratio:    {random_mean:.6f} +/- {random_std:.6f}")
            print(f"    Z-score: {z_score:.2f}")
            if z_score < -2:
                print(f"    SIGNIFICANT: Fibonacci coefficients are systematically below neighbors")
            elif z_score < -1:
                print(f"    MARGINAL: Some evidence for Fibonacci suppression")
            else:
                print(f"    NOT SIGNIFICANT at this truncation order")

        # =====================================================================
        # GAP DEPTH PATTERN
        # =====================================================================
        print("\n" + "=" * 72)
        print("  GAP DEPTH PATTERN: Do Fibonacci gaps deepen as phi^{-k}?")
        print("=" * 72)

        if len(neighbor_ratios) >= 3:
            fib_positions = [f for f, _ in neighbor_ratios]
            ratios_arr = [r for _, r in neighbor_ratios]
            log_ratios = [np.log10(r) for r in ratios_arr if r > 0]

            print(f"\n  {'F_k':>4s}  {'ratio':>10s}  {'log10(ratio)':>12s}  "
                  f"{'k (Fib index)':>14s}")
            print(f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*14}")

            fib_idx_map = {1:1, 2:3, 3:4, 5:5, 8:6, 13:7, 21:8, 34:9, 55:10, 89:11}
            for f, r in neighbor_ratios:
                k = fib_idx_map.get(f, '?')
                lr = np.log10(r) if r > 0 else -999
                print(f"  {f:4d}  {r:10.6f}  {lr:12.4f}  {str(k):>14s}")

            # Check if log10(ratio) is linear in k (i.e., ratio ~ base^k)
            k_vals = [fib_idx_map[f] for f, _ in neighbor_ratios if f in fib_idx_map]
            r_vals = [np.log10(r) for f, r in neighbor_ratios if f in fib_idx_map and r > 0]
            if len(k_vals) >= 3 and len(r_vals) >= 3:
                fit = np.polyfit(k_vals[:len(r_vals)], r_vals, 1)
                base = 10**fit[0]
                print(f"\n  Fit: log10(gap_ratio) ~ {fit[0]:.4f} * k + {fit[1]:.4f}")
                print(f"  Gap ratio ~ {base:.4f}^k")
                print(f"  Compare to 1/phi = {1/((1+np.sqrt(5))/2):.4f}")
                print(f"  Compare to phi^{-2} = {((1+np.sqrt(5))/2)**(-2):.4f}")

        # =====================================================================
        # INFORMATION CONTENT: bits per coefficient
        # =====================================================================
        print("\n" + "=" * 72)
        print("  INFORMATION CONTENT: cumulative digits of alpha from N terms")
        print("=" * 72)

        print(f"\n  Solving at each N to track alpha convergence + coefficient contribution...")
        prev_c2 = None
        digits_by_N = []
        for N in range(3, min(N_eff + 1, 61)):
            alpha_N, c_sol, ok = solve_f64(N, initial_guess=prev_c2)
            if ok and alpha_N is not None:
                err = abs(alpha_N - ALPHA_EXACT) / ALPHA_EXACT
                digits = -np.log10(err) if err > 1e-300 else 16
                digits_by_N.append((N, digits, N in FIB_SET))
                prev_c2 = c_sol

        if digits_by_N:
            print(f"\n  {'N':>4s}  {'digits':>8s}  {'marginal':>10s}  {'Fib?':>5s}")
            print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*5}")
            for i, (N, d, fib) in enumerate(digits_by_N):
                marginal = d - digits_by_N[i-1][1] if i > 0 else d
                marker = " <<<" if fib else ""
                print(f"  {N:4d}  {d:8.3f}  {marginal:+10.4f}  {marker}")

            # At what N does adding terms stop helping?
            for i, (N, d, fib) in enumerate(digits_by_N):
                if d >= 14:
                    print(f"\n  14+ digits reached at N={N}")
                    fib_above = [f for f in FIBONACCI if f >= N]
                    print(f"  Next Fibonacci above: {fib_above[0] if fib_above else 'none'}")
                    print(f"  F_10 = 55: {'above saturation' if N < 55 else 'below saturation'}")
                    break

    # =========================================================================
    # SAVE
    # =========================================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_02b_coefficient_analysis.py',
        'N_effective': int(N_eff),
        'alpha_error': float(alpha_err),
        'fit_type': fit_type if 'fit_type' in dir() else 'none',
        'coefficients': {
            'indices': [int(i) for i in indices[:len(c)]],
            'values': [float(v) for v in c],
            'abs_values': [float(v) for v in abs_c],
        },
    }

    if 'neighbor_ratios' in dir() and neighbor_ratios:
        output['fibonacci_neighbor_ratios'] = [
            {'F_k': int(f), 'ratio': float(r)} for f, r in neighbor_ratios
        ]

    if 'z_score' in dir():
        output['monte_carlo'] = {
            'z_score': float(z_score),
            'fib_mean_ratio': float(fib_mean_ratio),
            'random_mean_ratio': float(random_mean),
            'random_std': float(random_std),
        }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_02b_coefficient_analysis_{ts}.json'
    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\n  Saved: {fpath}")


if __name__ == "__main__":
    analyze_coefficients()
