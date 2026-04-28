"""
Milestone 10 -- Exp 15: 4π Holonomy and Half-Integer Modes

EXTENSION — testing Möbius extension §3 (closure claim) and §4 (π claim)

The extension claims:
  - 4π closure is the geometric form of M10 §4 (iteration requires two passes)
  - Half-integer modes are the Möbius spectral signature
  - π enters physics as the first numerical residue of Möbius traversal

This experiment verifies these structural claims by comparing the
spectral and dynamical properties of self-referential systems on
Möbius (anti-periodic) vs circle (periodic) topologies.

Tests:
  1. Eigenvalue formulas: Möbius matches cos(π(2k+1)/N), circle matches
     cos(2πk/N) — mathematical identities that confirm topology encoding.
  2. Mode doubling: Möbius has no zero eigenvalue; circle has one.
  3. 4π recurrence: autocorrelation period on Möbius ≈ 2× that on circle.
  4. Half-integer mode spacing: Möbius mode frequency ratios are (2k+3)/(2k+1),
     never integer ratios.

Builds on: exp_14 (topology comparison), exp_03 (iteration engine)
Extension: Möbius extension §3 (4π ↔ M10 §4), §4 (π as first residue)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    SelfApplicator,
    build_topology_matrix,
    measure_mode_structure,
    measure_holonomy_period,
    save_results, setup_experiment,
    PHI, LN_PHI, GAMMA_EM, XI_BALANCE, PI,
)

_, RESULTS_DIR = setup_experiment(__file__)


# ============================================================
# Test 1: Eigenvalue Classification
# ============================================================
def test1_eigenvalue_formulas():
    """
    Verify that topology matrices have the predicted eigenvalue structure.
    Möbius: cos(π(2k+1)/N), circle: cos(2πk/N).
    These are mathematical identities — should be exact to machine precision.
    """
    print("\n=== Test 1: Eigenvalue Classification ===")

    max_errors = []

    for N in [16, 32, 64]:
        for topology, formula_fn in [
            ('mobius', lambda k, N: np.cos(np.pi * (2 * k + 1) / N)),
            ('circle', lambda k, N: np.cos(2 * np.pi * k / N)),
        ]:
            W = build_topology_matrix(N, topology, sr=1.0)
            # Get eigenvalues of the un-normalized matrix
            # The sr normalization scales all eigenvalues, so compare shapes
            eigvals = np.sort(np.linalg.eigvalsh(W))

            # Expected eigenvalues (topology encodes nearest-neighbor with coupling c)
            # For sr=1.0 normalization, the max eigenvalue magnitude is 1.0
            expected = np.sort(np.array([formula_fn(k, N) for k in range(N)]))

            # Normalize both to same scale
            e_scale = np.max(np.abs(eigvals))
            x_scale = np.max(np.abs(expected))
            if e_scale > 1e-10 and x_scale > 1e-10:
                eigvals_n = eigvals / e_scale
                expected_n = expected / x_scale
            else:
                eigvals_n = eigvals
                expected_n = expected

            max_err = np.max(np.abs(eigvals_n - expected_n))
            max_errors.append(max_err)
            print(f"  {topology:>8s} N={N:3d}: max_error = {max_err:.2e}")

    passed = all(e < 1e-10 for e in max_errors)
    print(f"\n  PASS: {passed} (need all errors < 1e-10)")

    return {
        'test': 'eigenvalue_formulas',
        'passed': passed,
        'max_errors': [float(e) for e in max_errors],
    }


# ============================================================
# Test 2: Mode Doubling (No Zero Mode on Möbius)
# ============================================================
def test2_mode_doubling():
    """
    Möbius has no zero eigenvalue (no constant mode).
    Circle has a zero eigenvalue (constant mode exists).

    This is because anti-periodic boundaries kill the k=0 mode:
    cos(π·1/N) ≠ 0 for any finite N.
    Periodic boundaries allow k=0: cos(0) = 1.
    """
    print("\n=== Test 2: Mode Doubling (Zero Mode) ===")

    results = {}
    for N in [16, 32, 64]:
        for topology in ['mobius', 'circle']:
            ms = measure_mode_structure(build_topology_matrix(N, topology))
            key = f"{topology}_N{N}"
            results[key] = {
                'has_zero_mode': ms['has_zero_mode'],
                'min_eigval': float(np.min(np.abs(ms['eigenvalues']))),
            }
            print(f"  {topology:>8s} N={N:3d}: has_zero_mode={ms['has_zero_mode']}, "
                  f"min|λ|={np.min(np.abs(ms['eigenvalues'])):.6f}")

    # Pass if: all Möbius have NO zero mode, all circle DO have a zero mode
    mobius_ok = all(not results[k]['has_zero_mode']
                    for k in results if k.startswith('mobius'))
    circle_ok = all(results[k]['has_zero_mode']
                    for k in results if k.startswith('circle'))

    passed = mobius_ok and circle_ok
    print(f"\n  Möbius: all non-zero = {mobius_ok}")
    print(f"  Circle: all have zero = {circle_ok}")
    print(f"  PASS: {passed}")

    return {
        'test': 'mode_doubling',
        'passed': passed,
        'results': results,
    }


# ============================================================
# Test 3: 4π Recurrence
# ============================================================
def test3_holonomy_period():
    """
    Autocorrelation period on Möbius ≈ 2× that on circle.
    This is the dynamical manifestation of 4π holonomy.
    """
    print("\n=== Test 3: 4π Recurrence (Holonomy Period) ===")

    period_ratios = []

    for N in [16, 32]:
        for seed in range(15):
            periods = {}
            for topology in ['mobius', 'circle']:
                sa = SelfApplicator(rule_seed=seed, self_applies=True,
                                    symmetric=True, size=N)
                sa.W = build_topology_matrix(N, topology, sr=1.2)

                # Run to attractor
                for _ in range(500):
                    sa.step()

                # Record trajectory
                traj = np.zeros((2000, N))
                for t in range(2000):
                    sa.step()
                    traj[t] = sa.state

                hol = measure_holonomy_period(traj)
                periods[topology] = hol['period']

            if periods['circle'] > 0 and periods['mobius'] > 0:
                ratio = periods['mobius'] / periods['circle']
                period_ratios.append(ratio)
                if seed < 3:  # Print a few examples
                    print(f"  N={N}, seed={seed}: Möbius={periods['mobius']}, "
                          f"Circle={periods['circle']}, ratio={ratio:.2f}")

    if period_ratios:
        mean_ratio = np.mean(period_ratios)
        std_ratio = np.std(period_ratios)
        n_in_range = sum(1 for r in period_ratios if 1.5 <= r <= 2.5)
        frac_in_range = n_in_range / len(period_ratios)

        print(f"\n  Mean period ratio: {mean_ratio:.2f} ± {std_ratio:.2f}")
        print(f"  Fraction in [1.5, 2.5]: {frac_in_range:.2f} ({n_in_range}/{len(period_ratios)})")

        passed = frac_in_range > 0.5
    else:
        mean_ratio = 0.0
        frac_in_range = 0.0
        passed = False
        print("\n  No valid period ratios found")

    print(f"  PASS: {passed} (need >50% of ratios in [1.5, 2.5])")

    return {
        'test': 'holonomy_period',
        'passed': passed,
        'mean_ratio': float(mean_ratio),
        'n_ratios': len(period_ratios),
        'frac_in_range': float(frac_in_range),
    }


# ============================================================
# Test 4: Half-Integer Mode Spacing
# ============================================================
def test4_half_integer_spacing():
    """
    Möbius mode frequencies are proportional to (2k+1), giving ratios
    (2k+3)/(2k+1) = 3/1, 5/3, 7/5, 9/7, ... — never integers.
    Circle mode frequencies are proportional to k, giving integer ratios.
    """
    print("\n=== Test 4: Half-Integer Mode Spacing ===")

    results = {}

    for N in [16, 32, 64]:
        for topology in ['mobius', 'circle']:
            W = build_topology_matrix(N, topology, sr=1.0)
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(W)))

            # Remove near-zero eigenvalues
            eigvals = eigvals[eigvals > 1e-10]

            if len(eigvals) < 4:
                continue

            # Compute consecutive ratios
            ratios = eigvals[1:] / eigvals[:-1]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

            # Check if ratios are close to integers
            integer_nearness = np.mean(np.abs(ratios - np.round(ratios)))

            # Check if ratios match (2k+3)/(2k+1) pattern
            half_int_ratios = np.array(
                [(2 * k + 3) / (2 * k + 1) for k in range(len(ratios))]
            )
            half_int_nearness = np.mean(np.abs(ratios[:len(half_int_ratios)]
                                               - half_int_ratios[:len(ratios)]))

            results[f"{topology}_N{N}"] = {
                'integer_nearness': float(integer_nearness),
                'half_int_nearness': float(half_int_nearness),
                'n_ratios': len(ratios),
                'sample_ratios': [float(r) for r in ratios[:5]],
            }

            print(f"  {topology:>8s} N={N:3d}: integer_nearness={integer_nearness:.4f}, "
                  f"half_int_nearness={half_int_nearness:.4f}")
            print(f"           first ratios: {[f'{r:.3f}' for r in ratios[:5]]}")

    # Check the mode structure classification from measure_mode_structure
    print("\n  Mode structure classification:")
    for N in [16, 32]:
        for topology in ['mobius', 'circle']:
            ms = measure_mode_structure(build_topology_matrix(N, topology))
            print(f"    {topology:>8s} N={N}: half_int_check={ms['half_integer_check']:.4f}, "
                  f"integer_check={ms['integer_check']:.4f}")

    # Pass if Möbius matches half-integer pattern better than integer,
    # and circle matches integer pattern better than half-integer
    mobius_correct = all(
        results.get(f"mobius_N{N}", {}).get('half_int_nearness', 1.0) <
        results.get(f"mobius_N{N}", {}).get('integer_nearness', 0.0)
        for N in [16, 32, 64] if f"mobius_N{N}" in results
    )
    circle_correct = all(
        results.get(f"circle_N{N}", {}).get('integer_nearness', 1.0) <
        results.get(f"circle_N{N}", {}).get('half_int_nearness', 0.0)
        for N in [16, 32, 64] if f"circle_N{N}" in results
    )

    passed = mobius_correct and circle_correct
    print(f"\n  Möbius matches half-integer: {mobius_correct}")
    print(f"  Circle matches integer: {circle_correct}")
    print(f"  PASS: {passed}")

    return {
        'test': 'half_integer_spacing',
        'passed': passed,
        'results': results,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Exp 15: 4π Holonomy and Half-Integer Modes")
    print("  Testing Möbius extension §3 (closure) and §4 (π harmonics)")
    print("=" * 70)

    tests = [
        test1_eigenvalue_formulas,
        test2_mode_doubling,
        test3_holonomy_period,
        test4_half_integer_spacing,
    ]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_15_holonomy_half_integer_modes',
        'type': 'extension',
        'extension_section': '§3 (closure), §4 (π harmonics)',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'timestamp': datetime.now().isoformat(),
    }
    save_results(output, RESULTS_DIR, 'exp_15_holonomy_half_integer_modes')
