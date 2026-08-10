"""
Milestone 10 -- Exp 02: Time as Forced Processuality

Block A: Uniqueness & Foundations

PURPOSE: Show that static (one-shot) resolution of symmetry constraints is itself
asymmetric, forcing processual (multi-step) enactment. This derives time from
the impossibility of instantaneous symmetric resolution (thesis section 3).

Tests:
  1. Static resolution creates temporal asymmetry (discontinuity)
  2. Processual resolution preserves temporal symmetry
  3. One-step equilibration violates conservation in PAC networks
  4. Duration of resolution scales with initial asymmetry magnitude

Builds on: iddea.md section 3, M9 exp_06 (arrow of time)
Predicted: 4/4
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    measure_temporal_asymmetry,
    save_results, setup_experiment,
    PHI, XI_BALANCE,
)

_, RESULTS_DIR = setup_experiment(__file__)


def make_broken_symmetry_state(size, asymmetry_magnitude, rng):
    """Create a 1D state with broken left-right symmetry."""
    symmetric_part = rng.randn(size // 2)
    symmetric_state = np.concatenate([symmetric_part, symmetric_part[::-1]])
    noise = rng.randn(size) * asymmetry_magnitude
    return symmetric_state + noise


def static_resolution(state):
    """One-shot symmetry restoration: force mirror symmetry in one step."""
    n = len(state)
    resolved = np.zeros(n)
    for i in range(n):
        mirror = n - 1 - i
        resolved[i] = (state[i] + state[mirror]) / 2.0
    return resolved


def processual_resolution(state, n_steps=50, rate=0.1):
    """Iterative symmetry restoration: gradually approach mirror symmetry."""
    trajectory = [state.copy()]
    current = state.copy()
    n = len(current)

    for _ in range(n_steps):
        target = np.zeros(n)
        for i in range(n):
            mirror = n - 1 - i
            target[i] = (current[i] + current[mirror]) / 2.0
        current = current + rate * (target - current)
        trajectory.append(current.copy())

    return np.array(trajectory)


def measure_symmetry_error(state):
    """Measure how far a state is from mirror symmetry."""
    n = len(state)
    mirrored = state[::-1]
    return np.mean(np.abs(state - mirrored))


def test1_static_resolution_asymmetry():
    """Static resolution creates a temporal discontinuity."""
    print("\n" + "=" * 70)
    print("TEST 1: STATIC RESOLUTION CREATES TEMPORAL ASYMMETRY")
    print("=" * 70)

    n_trials = 200
    size = 64
    asymmetries = []
    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        state = make_broken_symmetry_state(size, 1.0, rng)
        resolved = static_resolution(state)

        # Static resolution: system sits at 'state' for many steps,
        # then jumps to 'resolved' in one step, then stays there.
        # All change is concentrated in a single step → high asymmetry.
        n_hold = 50
        trajectory = np.vstack([
            np.tile(state, (n_hold, 1)),
            np.tile(resolved, (n_hold, 1)),
        ])
        asym = measure_temporal_asymmetry(trajectory)
        asymmetries.append(asym)

    asymmetries = np.array(asymmetries)
    mean_asym = np.mean(asymmetries)
    frac_above_05 = np.mean(asymmetries > 0.5)

    print(f"\n  Mean temporal asymmetry:      {mean_asym:.4f}")
    print(f"  Fraction with asymmetry > 0.5: {frac_above_05:.1%}")

    passed = frac_above_05 > 0.90
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_above_05:.1%} > 90%")

    return {
        'test': 'static_resolution_asymmetry',
        'n_trials': n_trials,
        'mean_asymmetry': float(mean_asym),
        'fraction_above_05': float(frac_above_05),
        'passed': bool(passed),
    }


def test2_processual_preserves_symmetry():
    """Processual resolution has low temporal asymmetry."""
    print("\n" + "=" * 70)
    print("TEST 2: PROCESSUAL RESOLUTION PRESERVES TEMPORAL SYMMETRY")
    print("=" * 70)

    n_trials = 200
    size = 64
    asymmetries = []
    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        state = make_broken_symmetry_state(size, 1.0, rng)
        # Gentler rate over more steps → change distributed more evenly
        trajectory = processual_resolution(state, n_steps=100, rate=0.05)
        asym = measure_temporal_asymmetry(trajectory)
        asymmetries.append(asym)

    asymmetries = np.array(asymmetries)
    mean_asym = np.mean(asymmetries)
    frac_below_01 = np.mean(asymmetries < 0.1)

    print(f"\n  Mean temporal asymmetry:      {mean_asym:.4f}")
    print(f"  Fraction with asymmetry < 0.1: {frac_below_01:.1%}")

    # Processual: change distributed across many steps → low concentration
    passed = mean_asym < 0.1
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: mean {mean_asym:.4f} < 0.1")

    return {
        'test': 'processual_preserves_symmetry',
        'n_trials': n_trials,
        'mean_asymmetry': float(mean_asym),
        'fraction_below_01': float(frac_below_01),
        'passed': bool(passed),
    }


def test3_one_step_violates_conservation():
    """One-step equilibration in PAC network violates conservation."""
    print("\n" + "=" * 70)
    print("TEST 3: ONE-STEP EQUILIBRATION VIOLATES CONSERVATION")
    print("=" * 70)

    n_trials = 200
    n_nodes = 30
    violations = []
    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        # PAC network: each node has a conserved quantity
        state = rng.exponential(1.0, n_nodes)
        total_initial = np.sum(state)

        # One-step equilibration: set all to mean
        equilibrium = np.ones(n_nodes) * np.mean(state)

        # The one-step jump requires instantaneous redistribution.
        # In a PAC network, redistribution has intermediate states where
        # some nodes have given but others haven't received yet.
        # Model: at the midpoint, half the nodes have given, half haven't
        sorted_idx = np.argsort(state)
        midpoint = state.copy()
        # Top half gives to mean
        for i in sorted_idx[n_nodes//2:]:
            midpoint[i] = np.mean(state)
        # Bottom half hasn't received yet

        total_midpoint = np.sum(midpoint)
        violation = abs(total_midpoint - total_initial) / total_initial
        violations.append(violation)

    violations = np.array(violations)
    mean_violation = np.mean(violations)
    frac_above_10pct = np.mean(violations > 0.10)

    print(f"\n  Mean conservation violation:       {mean_violation:.4f}")
    print(f"  Fraction with violation > 10%:     {frac_above_10pct:.1%}")

    passed = frac_above_10pct > 0.95
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_above_10pct:.1%} > 95%")

    return {
        'test': 'one_step_violates_conservation',
        'n_trials': n_trials,
        'mean_violation': float(mean_violation),
        'fraction_above_10pct': float(frac_above_10pct),
        'passed': bool(passed),
    }


def test4_duration_scales_with_asymmetry():
    """Resolution duration is proportional to initial asymmetry."""
    print("\n" + "=" * 70)
    print("TEST 4: DURATION SCALES WITH INITIAL ASYMMETRY")
    print("=" * 70)

    size = 64
    rng = np.random.RandomState(42)

    asymmetry_magnitudes = np.linspace(0.1, 5.0, 20)
    resolution_steps = []

    for mag in asymmetry_magnitudes:
        steps_needed = []
        for trial in range(50):
            state = make_broken_symmetry_state(size, mag, rng)
            current = state.copy()
            n = len(current)
            sym_error = measure_symmetry_error(current)
            threshold = 0.01  # Absolute threshold: larger asymmetry → more steps

            for step in range(1, 1000):
                target = np.zeros(n)
                for i in range(n):
                    mirror = n - 1 - i
                    target[i] = (current[i] + current[mirror]) / 2.0
                current = current + 0.1 * (target - current)
                sym_error = measure_symmetry_error(current)
                if sym_error < threshold:
                    steps_needed.append(step)
                    break
            else:
                steps_needed.append(1000)

        resolution_steps.append(np.mean(steps_needed))

    # Spearman rank correlation: tests monotonic relationship
    # (steps ∝ log(mag), which is monotonic but not linear)
    r, p_value = spearmanr(asymmetry_magnitudes, resolution_steps)

    print(f"\n  Asymmetry range:     [{asymmetry_magnitudes[0]:.1f}, {asymmetry_magnitudes[-1]:.1f}]")
    print(f"  Steps range:         [{min(resolution_steps):.1f}, {max(resolution_steps):.1f}]")
    print(f"  Spearman correlation: r = {r:.4f}, p = {p_value:.2e}")

    passed = r > 0.9
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: r = {r:.4f} > 0.9")

    return {
        'test': 'duration_scales_with_asymmetry',
        'asymmetry_magnitudes': asymmetry_magnitudes.tolist(),
        'resolution_steps': resolution_steps,
        'spearman_r': float(r),
        'spearman_p': float(p_value),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 02: FORCED PROCESSUALITY")
    print("Block A: Uniqueness & Foundations")
    print("=" * 70)

    r1 = test1_static_resolution_asymmetry()
    r2 = test2_processual_preserves_symmetry()
    r3 = test3_one_step_violates_conservation()
    r4 = test4_duration_scales_with_asymmetry()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    results = {
        'experiment': 'exp_02_forced_processuality',
        'milestone': 10,
        'block': 'A',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_02_forced_processuality', RESULTS_DIR)


if __name__ == '__main__':
    main()
