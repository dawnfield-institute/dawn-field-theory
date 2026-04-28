"""
Milestone 10 -- Exp 11: Phi Emergence Conditions

INVESTIGATIVE — probing exp_03 T4 failure (phi ratio = 2.0 from period-2)

The scalar two-circle system produces period-2 oscillation. Phi never appears.
What structural conditions are actually required for phi to emerge from
mutual reference?

Two hypotheses:
  H1: Dimensionality — mutual reference needs enough DoF for scale hierarchy
  H2: Self-modification — the coupling must evolve (not just the state)

If H2 is correct, it strengthens exp_01's uniqueness argument: self-application
of the RULE (not just mutual reference of states) is the critical ingredient.
The SelfApplicator produces phi because its eigenvalue modulation creates
spectral structure. Fixed-coupling two-circle, at any N, should not.

Tests:
  1. Fixed-coupling two-circle, N=1..64: always period-2/chaos, never phi
  2. Self-modifying two-circle, N=1..64: find critical N where phi appears
  3. At critical N, which self-modification rules produce phi?
  4. Eigenspectrum at phi-producing N: consecutive ratios cluster near phi

Builds on: exp_03 T4 failure, exp_01 (SelfApplicator uniqueness)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    run_matrix_two_circle,
    save_results, setup_experiment,
    PHI, INV_PHI, LN_PHI,
)

_, RESULTS_DIR = setup_experiment(__file__)


def extract_dominant_ratio(norms, skip_transient=100):
    """
    Extract dominant inter-scale ratio from norm trajectory autocorrelation.

    Returns the ratio of the first two autocorrelation peak lags.
    Period-2 gives ratio = 2.0. Phi-structured gives ratio near 1.618.
    """
    signal = norms[skip_transient:]
    if len(signal) < 50:
        return None

    signal = signal - np.mean(signal)
    if np.std(signal) < 1e-12:
        return None

    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-30)

    # Find peaks in autocorrelation
    peaks = []
    for i in range(2, min(len(autocorr) - 1, len(signal) // 2)):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.05:
                peaks.append(i)
        if len(peaks) >= 3:
            break

    if len(peaks) >= 2:
        return peaks[1] / peaks[0] if peaks[0] > 0 else None
    return None


def test1_fixed_coupling_no_phi():
    """Fixed-coupling two-circle at all N: phi should NOT appear."""
    print("\n" + "=" * 70)
    print("TEST 1: FIXED COUPLING — No Phi at Any Dimension")
    print("=" * 70)

    dimensions = [1, 2, 4, 8, 16, 32, 64]
    n_seeds = 10
    n_steps = 2000

    results_by_N = {}
    phi_count_total = 0
    total_trials = 0

    for N in dimensions:
        ratios = []
        phi_hits = 0
        for seed in range(n_seeds):
            result = run_matrix_two_circle(N, n_steps=n_steps, seed=seed, evolving=False)
            ratio = extract_dominant_ratio(result['norms'])
            if ratio is not None:
                ratios.append(ratio)
                if abs(ratio - PHI) / PHI < 0.15:
                    phi_hits += 1
            total_trials += 1

        mean_ratio = np.mean(ratios) if ratios else 0
        results_by_N[N] = {
            'n_valid': len(ratios),
            'mean_ratio': float(mean_ratio),
            'phi_hits': phi_hits,
            'ratios': [float(r) for r in ratios],
        }
        phi_count_total += phi_hits
        print(f"  N={N:3d}: {len(ratios):2d} valid, mean ratio = {mean_ratio:.3f}, "
              f"phi hits = {phi_hits}/{len(ratios)}")

    frac_phi = phi_count_total / max(total_trials, 1)
    print(f"\n  Total phi hits: {phi_count_total}/{total_trials} ({frac_phi:.1%})")

    # PASS: phi is rare or absent with fixed coupling
    passed = frac_phi < 0.05
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_phi:.1%} < 5% phi hits")

    return {
        'test': 'fixed_coupling_no_phi',
        'dimensions': dimensions,
        'results_by_N': results_by_N,
        'total_phi_hits': phi_count_total,
        'total_trials': total_trials,
        'fraction_phi': float(frac_phi),
        'passed': bool(passed),
    }


def test2_self_modifying_phi_threshold():
    """Self-modifying two-circle: find critical N where phi appears."""
    print("\n" + "=" * 70)
    print("TEST 2: SELF-MODIFYING COUPLING — Phi Emergence Threshold")
    print("=" * 70)

    dimensions = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    n_seeds = 10
    n_steps = 2000

    results_by_N = {}
    critical_N = None

    for N in dimensions:
        ratios = []
        phi_hits = 0
        for seed in range(n_seeds):
            result = run_matrix_two_circle(N, n_steps=n_steps, seed=seed, evolving=True)
            ratio = extract_dominant_ratio(result['norms'])
            if ratio is not None:
                ratios.append(ratio)
                if abs(ratio - PHI) / PHI < 0.15:
                    phi_hits += 1

        mean_ratio = np.mean(ratios) if ratios else 0
        phi_frac = phi_hits / max(len(ratios), 1)
        results_by_N[N] = {
            'n_valid': len(ratios),
            'mean_ratio': float(mean_ratio),
            'phi_hits': phi_hits,
            'phi_fraction': float(phi_frac),
            'ratios': [float(r) for r in ratios],
        }

        marker = " <-- PHI" if phi_frac > 0.3 else ""
        print(f"  N={N:3d}: {len(ratios):2d} valid, mean ratio = {mean_ratio:.3f}, "
              f"phi = {phi_hits}/{len(ratios)} ({phi_frac:.0%}){marker}")

        if critical_N is None and phi_frac > 0.3:
            critical_N = N

    print(f"\n  Critical N (>30% phi): {critical_N}")

    # PASS: there exists a critical N where phi emerges
    passed = critical_N is not None and critical_N <= 32
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: "
          f"{'critical N = ' + str(critical_N) if passed else 'no phi emergence found'}")

    return {
        'test': 'self_modifying_phi_threshold',
        'dimensions': dimensions,
        'results_by_N': results_by_N,
        'critical_N': critical_N,
        'passed': bool(passed),
    }


def test3_modification_rule_comparison():
    """Which self-modification rules produce phi?"""
    print("\n" + "=" * 70)
    print("TEST 3: MODIFICATION RULES — What Produces Phi?")
    print("=" * 70)

    N = 16  # Use moderate dimension
    n_seeds = 15
    n_steps = 2000

    rules = {}

    # Rule A: Anti-Hebbian (the SelfApplicator mechanism) — use evolving=True
    phi_hits_A = 0
    ratios_A = []
    for seed in range(n_seeds):
        result = run_matrix_two_circle(N, n_steps=n_steps, seed=seed, evolving=True)
        ratio = extract_dominant_ratio(result['norms'])
        if ratio is not None:
            ratios_A.append(ratio)
            if abs(ratio - PHI) / PHI < 0.15:
                phi_hits_A += 1
    rules['anti_hebbian'] = {
        'phi_hits': phi_hits_A,
        'n_valid': len(ratios_A),
        'mean_ratio': float(np.mean(ratios_A)) if ratios_A else 0,
    }

    # Rule B: Random perturbation (no structure in modification)
    phi_hits_B = 0
    ratios_B = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N) / np.sqrt(N)
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 1e-10:
            W = W * (1.2 / sr)
        x = rng.randn(N) * 0.5
        y = rng.randn(N) * 0.5
        norms = [np.linalg.norm(x)]

        for step in range(n_steps):
            x_new = np.tanh(W @ y)
            y_new = np.tanh(W.T @ x)
            # Random perturbation to W
            W += rng.randn(N, N) * 0.001
            sr = np.max(np.abs(np.linalg.eigvals(W)))
            if sr > 1e-10:
                W = W * (1.2 / sr)
            x, y = x_new, y_new
            norms.append(np.linalg.norm(x))

        ratio = extract_dominant_ratio(np.array(norms))
        if ratio is not None:
            ratios_B.append(ratio)
            if abs(ratio - PHI) / PHI < 0.15:
                phi_hits_B += 1
    rules['random_perturbation'] = {
        'phi_hits': phi_hits_B,
        'n_valid': len(ratios_B),
        'mean_ratio': float(np.mean(ratios_B)) if ratios_B else 0,
    }

    # Rule C: Hebbian (opposite of anti-Hebbian — strengthen active modes)
    phi_hits_C = 0
    ratios_C = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N) / np.sqrt(N)
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 1e-10:
            W = W * (1.2 / sr)
        x = rng.randn(N) * 0.5
        y = rng.randn(N) * 0.5
        norms = [np.linalg.norm(x)]

        for step in range(n_steps):
            x_new = np.tanh(W @ y)
            y_new = np.tanh(W.T @ x)
            # Hebbian: strengthen active, weaken inactive
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            activity = (x_new + y_new) / 2
            projections = (Vt @ activity) ** 2
            total = np.sum(projections) + 1e-10
            activities = projections / total
            mean_act = 1.0 / N
            mod = np.ones(len(S))
            mod[activities[:len(S)] > 2.0 * mean_act] = 1.01  # Strengthen active
            mod[activities[:len(S)] < 0.5 * mean_act] = 0.95  # Weaken inactive
            S_new = S * mod
            sr = np.max(S_new)
            if sr > 1e-10:
                S_new = S_new * (1.2 / sr)
            W = U @ np.diag(S_new) @ Vt
            x, y = x_new, y_new
            norms.append(np.linalg.norm(x))

        ratio = extract_dominant_ratio(np.array(norms))
        if ratio is not None:
            ratios_C.append(ratio)
            if abs(ratio - PHI) / PHI < 0.15:
                phi_hits_C += 1
    rules['hebbian'] = {
        'phi_hits': phi_hits_C,
        'n_valid': len(ratios_C),
        'mean_ratio': float(np.mean(ratios_C)) if ratios_C else 0,
    }

    print(f"\n  {'Rule':<25s} {'Phi hits':>10s}  {'Mean ratio':>12s}")
    print(f"  {'-'*50}")
    for name, data in rules.items():
        print(f"  {name:<25s} {data['phi_hits']:>3d}/{data['n_valid']:<6d}  "
              f"{data['mean_ratio']:>12.3f}")

    # PASS: anti-Hebbian produces phi, others don't (or much less)
    ah_frac = rules['anti_hebbian']['phi_hits'] / max(rules['anti_hebbian']['n_valid'], 1)
    others_max = max(
        rules['random_perturbation']['phi_hits'] / max(rules['random_perturbation']['n_valid'], 1),
        rules['hebbian']['phi_hits'] / max(rules['hebbian']['n_valid'], 1),
    )

    passed = ah_frac > 0.20 and ah_frac > others_max * 2
    print(f"\n  Anti-Hebbian phi rate:  {ah_frac:.1%}")
    print(f"  Best other phi rate:   {others_max:.1%}")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: anti-Hebbian "
          f"{'dominates' if passed else 'does not dominate'}")

    return {
        'test': 'modification_rule_comparison',
        'N': N,
        'rules': rules,
        'anti_hebbian_phi_rate': float(ah_frac),
        'others_max_phi_rate': float(others_max),
        'passed': bool(passed),
    }


def test4_eigenspectrum_phi_structure():
    """At phi-producing N, eigenvalue ratios cluster near phi."""
    print("\n" + "=" * 70)
    print("TEST 4: EIGENSPECTRUM — Phi in Eigenvalue Ratios")
    print("=" * 70)

    # Run self-modifying two-circle at N=16,32 and collect final eigenspectra
    N_values = [16, 32]
    n_seeds = 20
    n_steps = 2000

    all_ratios = []
    all_eigenvalues = []

    for N in N_values:
        for seed in range(n_seeds):
            result = run_matrix_two_circle(N, n_steps=n_steps, seed=seed, evolving=True)
            eigvals = result['eigenvalues']
            # Consecutive eigenvalue ratios (sorted descending)
            eigvals = eigvals[eigvals > 0.01]  # Filter near-zero
            if len(eigvals) >= 3:
                ratios = eigvals[:-1] / eigvals[1:]
                all_ratios.extend(ratios.tolist())
                all_eigenvalues.append(eigvals.tolist())

    if not all_ratios:
        print("  No eigenvalue ratios collected")
        return {'test': 'eigenspectrum_phi_structure', 'passed': False}

    all_ratios = np.array(all_ratios)

    # What fraction of consecutive eigenvalue ratios are near phi?
    phi_close = np.sum(np.abs(all_ratios - PHI) / PHI < 0.15)
    phi_frac = phi_close / len(all_ratios)

    # Compare to how many are near 2.0 (period-doubling) or 1.0 (degenerate)
    two_close = np.sum(np.abs(all_ratios - 2.0) / 2.0 < 0.15)
    one_close = np.sum(np.abs(all_ratios - 1.0) < 0.15)

    # Histogram of ratios
    mean_ratio = float(np.mean(all_ratios))
    median_ratio = float(np.median(all_ratios))

    print(f"\n  Total eigenvalue ratios: {len(all_ratios)}")
    print(f"  Mean ratio:             {mean_ratio:.3f}")
    print(f"  Median ratio:           {median_ratio:.3f}")
    print(f"  Near phi (±15%):        {phi_close} ({phi_frac:.1%})")
    print(f"  Near 2.0 (±15%):        {two_close} ({two_close/len(all_ratios):.1%})")
    print(f"  Near 1.0 (±0.15):       {one_close} ({one_close/len(all_ratios):.1%})")

    # PASS: phi enrichment above random expectation
    # Random uniform in [1, 3] would give ~10% in any 30% window
    passed = phi_frac > 0.15
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: phi enrichment {phi_frac:.1%} > 15%")

    return {
        'test': 'eigenspectrum_phi_structure',
        'n_ratios': len(all_ratios),
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'phi_fraction': float(phi_frac),
        'two_fraction': float(two_close / len(all_ratios)),
        'one_fraction': float(one_close / len(all_ratios)),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 11: PHI EMERGENCE CONDITIONS")
    print("Investigative — probing exp_03 T4 failure")
    print("=" * 70)

    r1 = test1_fixed_coupling_no_phi()
    r2 = test2_self_modifying_phi_threshold()
    r3 = test3_modification_rule_comparison()
    r4 = test4_eigenspectrum_phi_structure()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    # Interpretation
    print("\n  INTERPRETATION:")
    if r1['passed'] and r2['passed']:
        print("  -> Fixed coupling never produces phi (at any N).")
        print("  -> Self-modification of coupling IS the critical ingredient.")
        print("  -> Dimensionality provides capacity; self-modification provides mechanism.")
        print("  -> This sharpens iddea.md §4: mutual reference alone is insufficient.")
    elif r1['passed'] and not r2['passed']:
        print("  -> Fixed coupling doesn't produce phi, but self-modification alone")
        print("     isn't enough either. Something else is needed.")
    elif not r1['passed']:
        print("  -> Surprising: fixed coupling CAN produce phi at high N.")
        print("  -> Dimensionality alone may suffice — self-modification not required.")

    results = {
        'experiment': 'exp_11_phi_emergence_conditions',
        'milestone': 10,
        'block': 'investigative',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_11_phi_emergence_conditions', RESULTS_DIR)


if __name__ == '__main__':
    main()
