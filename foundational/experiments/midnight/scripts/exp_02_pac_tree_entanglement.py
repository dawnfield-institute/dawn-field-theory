"""
exp_02 -- PAC Tree Correlation Depth Profile

Midnight Initiative, Thread 5 (Phase-Rate Primitive)

Hypothesis: The PAC tree's golden-ratio branching creates phi-structured
correlations between depth levels. When PAC conservation (V_parent = V_left +
V_right) operates with split ratio 1/phi + noise, the correlation between
depth levels decays as phi^(-delta). This is distinct from equal-split trees
(decay as 2^(-delta)) and confirms that the phi structure in PAC conservation
propagates into observable correlations.

Tests:
  T1: PAC conservation holds exactly despite noisy splitting
  T2: Correlation decays as phi^(-delta), not 2^(-delta)
  T3: Fibonacci-index separations show enhanced correlation above trend
  T4: Phi-split uniquely produces phi-decay — other splits don't

Source: exp_01 findings, journals/2026-06-03_phase-rate-primitive.md
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from phase_rate import (
    PHI, INV_PHI, LN_PHI,
    build_weighted_pac_tree,
    stochastic_pac_tree,
    pac_tree_correlation_profile,
    save_midnight_results, _convert_numpy,
)


def test_T1_pac_conservation():
    """T1: Conservation holds exactly in stochastic PAC trees."""
    print("\n  T1: PAC conservation holds in stochastic trees")

    depth = 6
    rng = np.random.RandomState(42)
    max_error = 0.0
    n_checks = 0

    for trial in range(100):
        potentials, levels = stochastic_pac_tree(depth, INV_PHI, 0.05, rng)
        n = len(potentials)
        for i in range(n):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and right < n:
                error = abs(potentials[i] - potentials[left] - potentials[right])
                max_error = max(max_error, error)
                n_checks += 1

    conservation_ok = max_error < 1e-14
    print(f"    Conservation max error ({n_checks} checks): {max_error:.2e} (<1e-14: {conservation_ok})")

    # Also verify the deterministic tree
    potentials_det, _, levels_det = build_weighted_pac_tree(depth, INV_PHI)
    det_errors = []
    for k in range(depth + 1):
        mask = levels_det == k
        if np.any(mask):
            major_path_v = INV_PHI**k
            det_errors.append(abs(potentials_det[2**k - 1] - major_path_v))

    max_det_error = max(det_errors)
    det_ok = max_det_error < 1e-14
    print(f"    Deterministic V(k) = phi^(-k) error: {max_det_error:.2e} (<1e-14: {det_ok})")

    passed = conservation_ok and det_ok
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_pac_conservation',
        'n_trials': 100,
        'n_checks': n_checks,
        'max_conservation_error': float(max_error),
        'max_deterministic_error': float(max_det_error),
        'PASS': passed,
    }


def test_T2_phi_decay():
    """T2: Correlation between depth levels decays as phi^(-delta)."""
    print("\n  T2: Correlation decays as phi^(-delta)")

    depth = 8
    profile = pac_tree_correlation_profile(depth, INV_PHI, noise_scale=0.05,
                                            n_trials=3000, seed=42)

    deltas = sorted(d for d in profile.keys() if d > 0)
    C_values = [profile[d]['mean_correlation'] for d in deltas]

    print(f"    Correlation profile:")
    for d in deltas:
        p = profile[d]
        print(f"      delta={d}: C={p['mean_correlation']:.6f} (n={p['n_pairs']})")

    if len(deltas) < 3 or all(c < 1e-10 for c in C_values):
        print("    Insufficient data for fit")
        return {'test': 'T2_phi_decay', 'PASS': False}

    # Fit: log(C) = log(A) - delta * log(r)
    valid = [(d, c) for d, c in zip(deltas, C_values) if c > 1e-10]
    if len(valid) < 3:
        print("    Not enough nonzero correlations")
        return {'test': 'T2_phi_decay', 'PASS': False}

    d_arr = np.array([v[0] for v in valid], dtype=float)
    log_C = np.log([v[1] for v in valid])

    A_mat = np.vstack([np.ones_like(d_arr), -d_arr]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_C, rcond=None)
    r_fit = np.exp(coeffs[1])

    phi_error = abs(r_fit - PHI) / PHI
    two_error = abs(r_fit - 2.0) / 2.0

    is_phi = phi_error < 0.10
    print(f"    Best-fit decay base: {r_fit:.4f}")
    print(f"    Deviation from phi ({PHI:.4f}): {phi_error:.1%} (<10%: {is_phi})")
    print(f"    Deviation from 2: {two_error:.1%}")

    passed = is_phi
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_phi_decay',
        'decay_base_fit': float(r_fit),
        'phi_deviation': float(phi_error),
        'two_deviation': float(two_error),
        'n_deltas_used': len(valid),
        'PASS': passed,
    }


def test_T3_fibonacci_enhancement():
    """T3: Fibonacci-index separations show enhanced correlation above trend."""
    print("\n  T3: Fibonacci separations show enhanced correlation")

    depth = 8
    profile = pac_tree_correlation_profile(depth, INV_PHI, noise_scale=0.05,
                                            n_trials=3000, seed=42)

    deltas = sorted(d for d in profile.keys() if d > 0)
    C_dict = {d: profile[d]['mean_correlation'] for d in deltas}
    C_arr = np.array([C_dict[d] for d in deltas])
    d_arr = np.array(deltas, dtype=float)

    # Fit exponential trend
    valid = [(d, c) for d, c in zip(deltas, C_arr) if c > 1e-10]
    if len(valid) < 4:
        print("    Not enough data for trend fit")
        return {'test': 'T3_fibonacci_enhancement', 'PASS': False}

    d_fit = np.array([v[0] for v in valid], dtype=float)
    log_C = np.log([v[1] for v in valid])
    A_mat = np.vstack([np.ones_like(d_fit), -d_fit]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_C, rcond=None)

    # Residuals from trend
    residuals = {}
    for d in deltas:
        if C_dict[d] > 1e-10:
            predicted = np.exp(coeffs[0] - coeffs[1] * d)
            residuals[d] = C_dict[d] - predicted

    fibs = set()
    a, b = 1, 2
    while a <= max(deltas):
        fibs.add(a)
        a, b = b, a + b

    fib_resid = [residuals[d] for d in residuals if d in fibs]
    non_fib_resid = [residuals[d] for d in residuals if d not in fibs]

    if not fib_resid or not non_fib_resid:
        print("    Cannot separate Fibonacci from non-Fibonacci")
        return {'test': 'T3_fibonacci_enhancement', 'PASS': False}

    mean_fib = np.mean(fib_resid)
    mean_non_fib = np.mean(non_fib_resid)
    enhanced = mean_fib > mean_non_fib

    for d in sorted(residuals.keys()):
        marker = " [F]" if d in fibs else "    "
        print(f"    delta={d}{marker}: C={C_dict[d]:.6f}  residual={residuals[d]:+.6f}")

    print(f"    Mean Fibonacci residual: {mean_fib:+.6f}")
    print(f"    Mean non-Fibonacci:      {mean_non_fib:+.6f}")
    print(f"    Fibonacci enhanced: {enhanced}")

    passed = enhanced
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_fibonacci_enhancement',
        'mean_fib_residual': float(mean_fib),
        'mean_non_fib_residual': float(mean_non_fib),
        'enhanced': enhanced,
        'residuals': {str(d): float(v) for d, v in residuals.items()},
        'PASS': passed,
    }


def test_T4_phi_uniqueness():
    """T4: Only the phi-split produces correlation decay matching phi."""
    print("\n  T4: Phi-split uniquely produces phi-decay")

    depth = 7
    split_ratios = {
        '1/phi': INV_PHI,
        '1/2': 0.5,
        '0.4': 0.4,
        '0.3': 0.3,
        '1/e': 1.0 / np.e,
        '1/3': 1.0 / 3.0,
    }

    results = {}
    for name, ratio in split_ratios.items():
        profile = pac_tree_correlation_profile(depth, ratio, noise_scale=0.05,
                                                n_trials=2000, seed=42)
        deltas = sorted(d for d in profile.keys() if d > 0)
        valid = [(d, profile[d]['mean_correlation']) for d in deltas
                 if profile[d]['mean_correlation'] > 1e-10]

        if len(valid) < 3:
            results[name] = {'decay_base': 0.0, 'phi_error': 1.0, 'split': float(ratio)}
            continue

        d_arr = np.array([v[0] for v in valid], dtype=float)
        log_C = np.log([v[1] for v in valid])
        A_mat = np.vstack([np.ones_like(d_arr), -d_arr]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_C, rcond=None)
        r_fit = np.exp(coeffs[1])

        phi_error = abs(r_fit - PHI) / PHI
        results[name] = {
            'split': float(ratio),
            'decay_base': float(r_fit),
            'phi_error': float(phi_error),
        }

    phi_result = results['1/phi']
    others = {k: v for k, v in results.items() if k != '1/phi'}
    phi_best = all(phi_result['phi_error'] < v['phi_error'] for v in others.values())
    others_far = all(v['phi_error'] > 0.10 for v in others.values())

    print(f"    {'Split':>8}  {'Ratio':>8}  {'Decay':>8}  {'Phi err':>8}")
    for name in split_ratios:
        r = results[name]
        marker = " <--" if name == '1/phi' else ""
        print(f"    {name:>8}  {r['split']:>8.4f}  {r['decay_base']:>8.4f}  {r['phi_error']:>8.1%}{marker}")

    print(f"    Phi-split best match: {phi_best}")
    print(f"    All others >10% from phi: {others_far}")

    passed = phi_best and others_far
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_phi_uniqueness',
        'results': results,
        'phi_best': phi_best,
        'others_far': others_far,
        'PASS': passed,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("exp_02: PAC Tree Correlation Depth Profile")
    print("Midnight Initiative, Thread 5 (Phase-Rate Primitive)")
    print("=" * 70)

    t1 = test_T1_pac_conservation()
    t2 = test_T2_phi_decay()
    t3 = test_T3_fibonacci_enhancement()
    t4 = test_T4_phi_uniqueness()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_02_pac_tree_correlation',
        'initiative': 'midnight',
        'thread': 'phase_rate_primitive',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_02_pac_tree_correlation', _convert_numpy(data))
