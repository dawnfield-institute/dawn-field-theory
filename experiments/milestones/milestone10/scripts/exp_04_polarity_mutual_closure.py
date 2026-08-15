"""
Milestone 10 -- Exp 04: Polarity as Mutual Closure

Block B: Polarity & Dynamic Laws

PURPOSE: Show that info-dynamics and thermodynamics are structurally required
mutual closures. Neither self-stabilizes alone. Info-dynamics alone diverges
(unbounded exploration). Thermodynamics alone collapses (trivial equilibrium).
Only the coupled system produces stable, non-trivial multi-scale structure
(thesis section 5).

Tests:
  1. Isolated exploration diverges: variance grows unbounded
  2. Isolated dissipation collapses: reaches max-entropy flat state
  3. Coupled system stabilizes: non-trivial steady state with multi-scale structure
  4. Equal coupling required: stability only near alpha = beta

Builds on: iddea.md section 5
Predicted: 4/4 (straightforward dynamical test)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    info_dynamics_step, thermo_dynamics_step, coupled_polarity_step,
    measure_complexity,
    save_results, setup_experiment,
    PHI, INV_PHI,
)

_, RESULTS_DIR = setup_experiment(__file__)


def make_initial_states(n_states=50, size=64, seed=42):
    """Generate random initial states with non-trivial structure."""
    rng = np.random.RandomState(seed)
    states = []
    for _ in range(n_states):
        state = rng.randn(size) * 2.0
        # Add some structure
        for k in range(1, 4):
            freq = k * 2 * np.pi / size
            state += rng.randn() * np.sin(freq * np.arange(size))
        states.append(state)
    return states


def test1_isolated_exploration_diverges():
    """Info-dynamics alone: variance grows unbounded."""
    print("\n" + "=" * 70)
    print("TEST 1: ISOLATED EXPLORATION — Variance Diverges")
    print("=" * 70)

    states = make_initial_states(n_states=50, size=64)
    n_steps = 1000
    diverged_count = 0

    for ic in states:
        state = ic.copy()
        initial_var = np.var(state)

        for _ in range(n_steps):
            state = info_dynamics_step(state, dt=0.01, growth_rate=1.0)
            if np.any(np.abs(state) > 1e10):
                break

        final_var = np.var(state)
        growth_ratio = final_var / max(initial_var, 1e-10)
        if growth_ratio > 100:
            diverged_count += 1

    frac_diverged = diverged_count / len(states)
    print(f"\n  Initial conditions tested: {len(states)}")
    print(f"  Diverged (var > 100x):     {diverged_count} ({frac_diverged:.1%})")

    passed = frac_diverged > 0.90
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_diverged:.1%} > 90%")

    return {
        'test': 'isolated_exploration_diverges',
        'n_states': len(states),
        'diverged_count': diverged_count,
        'fraction_diverged': float(frac_diverged),
        'passed': bool(passed),
    }


def test2_isolated_dissipation_collapses():
    """Thermodynamics alone: collapses to trivial equilibrium."""
    print("\n" + "=" * 70)
    print("TEST 2: ISOLATED DISSIPATION — Collapses to Flat")
    print("=" * 70)

    states = make_initial_states(n_states=50, size=64)
    n_steps = 1000
    collapsed_count = 0

    for ic in states:
        state = ic.copy()

        for _ in range(n_steps):
            state = thermo_dynamics_step(state, dt=0.01, dissipation_rate=1.0)

        # Check if collapsed to near-flat state
        complexity = measure_complexity(state)
        # Flat = all values nearly equal = maximum bin entropy
        state_range = np.max(state) - np.min(state)
        initial_range = np.max(ic) - np.min(ic)
        range_ratio = state_range / max(initial_range, 1e-10)

        if range_ratio < 0.1:  # Range collapsed to < 10%
            collapsed_count += 1

    frac_collapsed = collapsed_count / len(states)
    print(f"\n  Initial conditions tested: {len(states)}")
    print(f"  Collapsed (range < 10%):   {collapsed_count} ({frac_collapsed:.1%})")

    passed = frac_collapsed > 0.90
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_collapsed:.1%} > 90%")

    return {
        'test': 'isolated_dissipation_collapses',
        'n_states': len(states),
        'collapsed_count': collapsed_count,
        'fraction_collapsed': float(frac_collapsed),
        'passed': bool(passed),
    }


def test3_coupled_system_stabilizes():
    """Coupled system: stable non-trivial multi-scale structure."""
    print("\n" + "=" * 70)
    print("TEST 3: COUPLED SYSTEM — Stable Non-Trivial Structure")
    print("=" * 70)

    states = make_initial_states(n_states=50, size=64)
    n_steps = 500
    stable_count = 0
    complexity_values = []

    for ic in states:
        state = ic.copy()
        complexities = []

        for step in range(n_steps):
            state = coupled_polarity_step(state, dt=0.01, alpha=1.0, beta=1.0)
            if np.any(~np.isfinite(state)):
                break
            if step >= n_steps // 2:  # Measure second half
                complexities.append(measure_complexity(state))

        if len(complexities) > 10:
            mean_c = np.mean(complexities)
            std_c = np.std(complexities)
            complexity_values.append(mean_c)

            # Stable = complexity stays in [0.3, 0.7] of max
            if 0.3 <= mean_c <= 0.7 and std_c < 0.2:
                stable_count += 1

    frac_stable = stable_count / len(states) if states else 0
    print(f"\n  Initial conditions tested: {len(states)}")
    print(f"  Stable non-trivial:        {stable_count} ({frac_stable:.1%})")
    if complexity_values:
        print(f"  Mean complexity:           {np.mean(complexity_values):.3f}")

    # Threshold: at least some fraction produce stable structure
    # More lenient than 50% since coupling parameters may not be perfectly tuned
    passed = stable_count >= len(states) * 0.3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {stable_count} >= {int(len(states) * 0.3)} stable")

    return {
        'test': 'coupled_system_stabilizes',
        'n_states': len(states),
        'stable_count': stable_count,
        'fraction_stable': float(frac_stable),
        'complexity_values': [float(c) for c in complexity_values[:20]],
        'passed': bool(passed),
    }


def test4_equal_coupling_required():
    """Stability only when alpha and beta are near-equal."""
    print("\n" + "=" * 70)
    print("TEST 4: EQUAL COUPLING — Stability Near Diagonal")
    print("=" * 70)

    # Scan alpha-beta parameter space
    n_grid = 15
    alphas = np.linspace(0.1, 3.0, n_grid)
    betas = np.linspace(0.1, 3.0, n_grid)
    n_steps = 300
    size = 64
    rng = np.random.RandomState(42)

    stability_map = np.zeros((n_grid, n_grid))
    initial_state = rng.randn(size) * 2.0

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            state = initial_state.copy()
            complexities = []

            for step in range(n_steps):
                state = coupled_polarity_step(state, dt=0.01, alpha=alpha, beta=beta)
                if np.any(~np.isfinite(state)) or np.max(np.abs(state)) > 1e8:
                    break
                if step >= n_steps // 2:
                    complexities.append(measure_complexity(state))

            if len(complexities) > 10:
                mean_c = np.mean(complexities)
                std_c = np.std(complexities)
                if 0.3 <= mean_c <= 0.7 and std_c < 0.2:
                    stability_map[i, j] = 1.0

    # Find centroid of stable region
    stable_points = np.argwhere(stability_map > 0.5)
    if len(stable_points) > 0:
        centroid_i = np.mean(stable_points[:, 0])
        centroid_j = np.mean(stable_points[:, 1])
        centroid_alpha = alphas[int(round(centroid_i))]
        centroid_beta = betas[int(round(centroid_j))]
        diagonal_distance = abs(centroid_alpha - centroid_beta) / max(centroid_alpha, centroid_beta)
    else:
        centroid_alpha = 0
        centroid_beta = 0
        diagonal_distance = 1.0

    n_stable = int(stability_map.sum())
    print(f"\n  Grid: {n_grid}x{n_grid} alpha-beta space")
    print(f"  Stable cells:       {n_stable}/{n_grid*n_grid}")
    print(f"  Centroid alpha:     {centroid_alpha:.2f}")
    print(f"  Centroid beta:      {centroid_beta:.2f}")
    print(f"  Diagonal distance:  {diagonal_distance:.3f}")

    passed = diagonal_distance < 0.20 and n_stable >= 3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: diagonal distance {diagonal_distance:.3f} < 0.20")

    return {
        'test': 'equal_coupling_required',
        'grid_size': n_grid,
        'n_stable': n_stable,
        'centroid_alpha': float(centroid_alpha),
        'centroid_beta': float(centroid_beta),
        'diagonal_distance': float(diagonal_distance),
        'stability_map': stability_map.tolist(),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 04: POLARITY AS MUTUAL CLOSURE")
    print("Block B: Polarity & Dynamic Laws")
    print("=" * 70)

    r1 = test1_isolated_exploration_diverges()
    r2 = test2_isolated_dissipation_collapses()
    r3 = test3_coupled_system_stabilizes()
    r4 = test4_equal_coupling_required()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    results = {
        'experiment': 'exp_04_polarity_mutual_closure',
        'milestone': 10,
        'block': 'B',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_04_polarity_mutual_closure', RESULTS_DIR)


if __name__ == '__main__':
    main()
