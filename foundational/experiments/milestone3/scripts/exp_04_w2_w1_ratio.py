"""
exp_04: w2/w1 Ratio from First Principles

HYPOTHESIS: The cascade weight ratio w2/w1 can be derived from
Landauer erasure physics alone, and its value determines whether
the limiting ratio approaches φ, 1/φ, or something else.

SOURCE: internal/energy_equivilance/cascade_deep_dive.py (ratio gap analysis)
TARGET: Paper 6 - E-I-S dynamics, connecting cascade ratio to 0.600 vs 0.618

CONTEXT: The energy_equivalence session found w1 ≈ 0.6 consistently.
The gap between 0.600 (observed cascade decay) and 0.618 (1/φ) is
either physically meaningful or a finite-size artifact.

METHOD:
1. Compute w1, w2 from Landauer physics across many configurations
2. Measure w2/w1 and its distribution
3. Test whether w2/w1 → 1 (required for Fibonacci) in any limit
4. Characterize the 0.600 vs 0.618 gap analytically
5. Test n_modes dependence (does gap close as modes → ∞?)
"""

import sys
import os
import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, INV_PHI, LN_PHI
from core.xi_calculator import single_landauer_event
from core.utils import save_results, experiment_header


def measure_cascade_ratio(T, n_modes, n_samples, n_trials=50, rng_seed=42):
    """
    Measure the effective cascade decay ratio from Landauer events.

    The cascade ratio is Θ/P — what fraction of potential survives
    each erasure step.
    """
    ratios = []
    w1_values = []
    w2_values = []

    for trial in range(n_trials):
        rng = np.random.default_rng(rng_seed + trial)
        event = single_landauer_event(T, n_modes=n_modes,
                                       n_samples=n_samples, rng=rng)
        P = T * np.log(2)
        w1 = event['theta'] / P  # Thermal remainder fraction
        w2 = event['xi'] / P     # Structure fraction

        ratios.append(w1)
        w1_values.append(w1)
        w2_values.append(w2)

    return {
        'cascade_ratio_mean': float(np.mean(ratios)),
        'cascade_ratio_std': float(np.std(ratios)),
        'w1_mean': float(np.mean(w1_values)),
        'w2_mean': float(np.mean(w2_values)),
        'w2_over_w1': float(np.mean(w2_values)) / float(np.mean(w1_values))
        if np.mean(w1_values) != 0 else float('inf'),
    }


def analytical_limiting_ratio(w1, w2):
    """
    Analytical limiting ratio for P(n) = w1·P(n-1) + w2·P(n-2).

    Solution: r = (w1 + sqrt(w1² + 4w2)) / 2
    """
    discriminant = w1**2 + 4*w2
    if discriminant < 0:
        return float('nan')
    return (w1 + np.sqrt(discriminant)) / 2


def main():
    meta = experiment_header(
        'exp_04_w2_w1_ratio',
        'w2/w1 ratio from first principles — closing the 0.600 vs 0.618 gap',
        paper='Paper 6',
        section='§8 (E-I-S dynamics)'
    )

    results = {**meta, 'tests': {}}

    # --- Test 1: n_modes sweep ---
    print("Test 1: n_modes dependence of cascade ratio")
    mode_sweep = {}
    for n_modes in [2, 4, 8, 16, 32, 64]:
        data = measure_cascade_ratio(T=1.0, n_modes=n_modes,
                                      n_samples=30000, n_trials=50)
        limiting = analytical_limiting_ratio(data['w1_mean'], data['w2_mean'])
        data['limiting_ratio'] = float(limiting)
        data['gap_to_inv_phi'] = float(abs(data['cascade_ratio_mean'] - INV_PHI))
        data['gap_to_0_6'] = float(abs(data['cascade_ratio_mean'] - 0.6))
        mode_sweep[f'modes_{n_modes}'] = data
        print(f"  n_modes={n_modes:3d}: cascade_ratio={data['cascade_ratio_mean']:.6f}, "
              f"w2/w1={data['w2_over_w1']:.4f}, "
              f"limiting={limiting:.6f}")

    results['tests']['mode_sweep'] = mode_sweep

    # Does the gap close as n_modes increases?
    cascade_ratios = [mode_sweep[f'modes_{n}']['cascade_ratio_mean']
                      for n in [2, 4, 8, 16, 32, 64]]
    gaps = [abs(r - INV_PHI) for r in cascade_ratios]
    closing = all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1))

    results['tests']['gap_convergence'] = {
        'gaps_by_modes': {f'modes_{n}': float(g) for n, g in
                          zip([2, 4, 8, 16, 32, 64], gaps)},
        'monotonically_closing': closing,
    }
    print(f"\n  Gap monotonically closing: {closing}")

    # --- Test 2: Temperature sweep ---
    print("\nTest 2: Temperature dependence")
    temp_sweep = {}
    for T in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        data = measure_cascade_ratio(T=T, n_modes=8,
                                      n_samples=30000, n_trials=30)
        temp_sweep[f'T_{T}'] = data
        print(f"  T={T:6.2f}: cascade_ratio={data['cascade_ratio_mean']:.6f}, "
              f"w2/w1={data['w2_over_w1']:.4f}")

    results['tests']['temperature_sweep'] = temp_sweep

    # --- Test 3: Analytical characterization ---
    print("\nTest 3: Analytical gap characterization")
    # For the Fibonacci recurrence, w2/w1 must equal 1 exactly.
    # The Landauer model's w1 = Θ/P and w2 = ξ/P.
    # Question: under what conditions does ξ → Θ?
    # ξ comes from inter-mode correlation; Θ from thermal remainder.
    # As n_modes → ∞, coupling becomes more diffuse, ξ may change.

    # Best-case w2/w1 from the sweep
    best_modes = max(mode_sweep.keys(),
                     key=lambda k: mode_sweep[k]['w2_over_w1'])
    best_ratio = mode_sweep[best_modes]['w2_over_w1']

    results['tests']['analytical'] = {
        'fibonacci_requires_w2_over_w1': 1.0,
        'best_observed_w2_over_w1': best_ratio,
        'best_observed_at': best_modes,
        'gap_to_fibonacci': abs(best_ratio - 1.0),
        'note': (
            'For the limiting ratio to equal φ, we need w2/w1 = 1 exactly. '
            'The Landauer model gives w2/w1 that depends on n_modes and T. '
            'The 0.600 vs 0.618 gap may be: (a) a finite-mode artifact that '
            'closes as n_modes → ∞, (b) a physical prediction distinguishing '
            'the cascade ratio from 1/φ, or (c) an artifact of the exponential '
            'coupling topology. This experiment characterizes which.'
        ),
    }
    print(f"  Best w2/w1: {best_ratio:.6f} at {best_modes}")
    print(f"  Gap to Fibonacci (w2/w1=1): {abs(best_ratio - 1.0):.6f}")

    save_results(results, 'exp_04_w2_w1_ratio')


if __name__ == '__main__':
    main()
