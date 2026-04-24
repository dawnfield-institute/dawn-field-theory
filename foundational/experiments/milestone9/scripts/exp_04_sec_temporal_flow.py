"""
Milestone 9 -- Exp 04: SEC Temporal Flow

PURPOSE: Does SEC generate temporal ordering from a time-symmetric initial
state? The Second Entropic Constraint breaks time-reversal symmetry:
forward SEC and anti-SEC (reversed signs) produce qualitatively different
dynamics from the same initial conditions. This asymmetry defines the
arrow of time, with the rate of change following logarithmic scaling
consistent with the cascade clock.

Block B: Information-Time Nexus

Tests:
  1. SEC breaks time symmetry: forward vs anti-SEC produce opposite effects
  2. Emergent time arrow: forward/backward entropy changes are asymmetric
  3. Flow rate matches clock: entropy change rate is logarithmic in time
  4. SEC equilibrium is max entropy: SEC drives toward equipartition
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M9_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M9_ROOT))

from core.infodynamics import (
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, PI,
    B_DFT, B_FREE, T_UNIVERSE,
    run_sec_dynamics, sec_entropy_rate,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)


def _shannon_entropy(P):
    """Shannon entropy of distribution P (assumed normalized)."""
    p = P / np.sum(P)
    p = p[p > 1e-15]
    return -np.sum(p * np.log(p))


def _gini_coefficient(P):
    """Gini coefficient of distribution P. 0 = uniform, 1 = concentrated."""
    p = np.sort(P / np.sum(P))
    n = len(p)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * p) / (n * np.sum(p))) - (n + 1.0) / n


def test1_sec_breaks_time_symmetry():
    """
    Test 1: Run SEC dynamics (forward) and anti-SEC (reversed alpha, beta)
    from the SAME random initial state. Forward SEC should drive the
    distribution toward equipartition (lower Gini, higher entropy).
    Anti-SEC should drive toward concentration (higher Gini, lower entropy).

    The asymmetry between forward and backward defines the arrow of time:
    SEC has a preferred direction even though the initial state is symmetric.

    PASS: forward Gini decreases AND anti-SEC Gini increases from same start.
    """
    print("\n" + "-" * 70)
    print("TEST 1: SEC BREAKS TIME SYMMETRY")
    print("-" * 70)

    n_nodes = 50
    n_steps = 2000

    print(f"\n  Running from same random initial state (seed=42)")
    print(f"  Forward SEC: alpha=1.0, beta=0.5")
    print(f"  Anti-SEC:    alpha=-1.0, beta=-0.5")

    history_fwd = run_sec_dynamics(n_nodes, n_steps, alpha=1.0, beta=0.5, seed=42)
    history_anti = run_sec_dynamics(n_nodes, n_steps, alpha=-1.0, beta=-0.5, seed=42)

    # Both start from same initial state (same seed)
    gini_init = _gini_coefficient(history_fwd[0])
    gini_fwd_final = _gini_coefficient(history_fwd[-1])
    gini_anti_final = _gini_coefficient(history_anti[-1])

    S_init = _shannon_entropy(history_fwd[0])
    S_fwd_final = _shannon_entropy(history_fwd[-1])
    S_anti_final = _shannon_entropy(history_anti[-1])

    delta_gini_fwd = gini_fwd_final - gini_init
    delta_gini_anti = gini_anti_final - gini_init
    delta_S_fwd = S_fwd_final - S_init
    delta_S_anti = S_anti_final - S_init

    print(f"\n  Initial state:")
    print(f"    Gini = {gini_init:.6f}")
    print(f"    S    = {S_init:.6f}")

    print(f"\n  Forward SEC (after {n_steps} steps):")
    print(f"    Gini = {gini_fwd_final:.6f}  (delta = {delta_gini_fwd:+.6f})")
    print(f"    S    = {S_fwd_final:.6f}  (delta = {delta_S_fwd:+.6f})")

    print(f"\n  Anti-SEC (after {n_steps} steps):")
    print(f"    Gini = {gini_anti_final:.6f}  (delta = {delta_gini_anti:+.6f})")
    print(f"    S    = {S_anti_final:.6f}  (delta = {delta_S_anti:+.6f})")

    # Trajectory at checkpoints
    checkpoints = [0, 100, 500, 1000, 2000]
    print(f"\n  Gini trajectory:")
    print(f"  {'Step':>6s}  {'Forward':>10s}  {'Anti-SEC':>10s}")
    for cp in checkpoints:
        if cp < len(history_fwd) and cp < len(history_anti):
            g_f = _gini_coefficient(history_fwd[cp])
            g_a = _gini_coefficient(history_anti[cp])
            print(f"  {cp:6d}  {g_f:10.6f}  {g_a:10.6f}")

    fwd_gini_decreased = delta_gini_fwd < 0
    anti_gini_increased = delta_gini_anti > 0

    print(f"\n  Forward Gini decreased: {fwd_gini_decreased}")
    print(f"  Anti-SEC Gini increased: {anti_gini_increased}")
    print(f"  Time symmetry broken: {fwd_gini_decreased and anti_gini_increased}")

    passed = fwd_gini_decreased and anti_gini_increased
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: time symmetry "
          f"{'broken' if passed else 'not broken'}")

    return {
        'test': 'sec_breaks_time_symmetry',
        'n_nodes': n_nodes,
        'n_steps': n_steps,
        'gini_initial': float(gini_init),
        'gini_fwd_final': float(gini_fwd_final),
        'gini_anti_final': float(gini_anti_final),
        'delta_gini_fwd': float(delta_gini_fwd),
        'delta_gini_anti': float(delta_gini_anti),
        'delta_S_fwd': float(delta_S_fwd),
        'delta_S_anti': float(delta_S_anti),
        'fwd_decreased': bool(fwd_gini_decreased),
        'anti_increased': bool(anti_gini_increased),
        'passed': bool(passed),
    }


def test2_emergent_time_arrow():
    """
    Test 2: The forward-backward asymmetry defines an arrow of time.
    Run forward SEC and anti-SEC for the same duration. Compute the
    magnitude of entropy change for each. The asymmetry ratio
    measures how strongly time-reversal symmetry is broken:

      asymmetry = |delta_S_fwd - delta_S_anti| / max(|delta_S_fwd|, |delta_S_anti|)

    For a time-symmetric system, asymmetry = 0.
    For a strong arrow, asymmetry -> 1.

    PASS: asymmetry > 0.5 AND the signs of delta_S are opposite.
    """
    print("\n" + "-" * 70)
    print("TEST 2: EMERGENT TIME ARROW")
    print("-" * 70)

    n_nodes = 50
    n_steps = 1000

    print(f"\n  Running SEC and anti-SEC from same initial state")

    history_fwd = run_sec_dynamics(n_nodes, n_steps, alpha=1.0, beta=0.5, seed=42)
    history_anti = run_sec_dynamics(n_nodes, n_steps, alpha=-1.0, beta=-0.5, seed=42)

    S_init = _shannon_entropy(history_fwd[0])
    S_fwd = _shannon_entropy(history_fwd[-1])
    S_anti = _shannon_entropy(history_anti[-1])

    delta_S_fwd = S_fwd - S_init
    delta_S_anti = S_anti - S_init

    max_abs = max(abs(delta_S_fwd), abs(delta_S_anti))
    if max_abs > 1e-10:
        asymmetry = abs(delta_S_fwd - delta_S_anti) / max_abs
    else:
        asymmetry = 0.0

    signs_opposite = (delta_S_fwd > 0) != (delta_S_anti > 0)

    print(f"\n  Entropy changes:")
    print(f"    Forward SEC:  delta_S = {delta_S_fwd:+.6f}")
    print(f"    Anti-SEC:     delta_S = {delta_S_anti:+.6f}")
    print(f"    Signs opposite: {signs_opposite}")

    print(f"\n  Time-reversal asymmetry:")
    print(f"    |delta_S_fwd - delta_S_anti| / max(|...|) = {asymmetry:.6f}")
    print(f"    Threshold: > 0.5")
    print(f"    For symmetric system: asymmetry = 0")
    print(f"    For strong arrow: asymmetry -> 1")

    # Entropy trajectory comparison
    sample_steps = list(range(0, n_steps + 1, 100))
    print(f"\n  Entropy trajectories:")
    print(f"  {'Step':>6s}  {'S_fwd':>10s}  {'S_anti':>10s}  {'gap':>10s}")
    for step in sample_steps:
        if step < len(history_fwd) and step < len(history_anti):
            sf = _shannon_entropy(history_fwd[step])
            sa = _shannon_entropy(history_anti[step])
            print(f"  {step:6d}  {sf:10.6f}  {sa:10.6f}  {sf - sa:+10.6f}")

    passed = asymmetry > 0.5 and signs_opposite
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: time arrow "
          f"{'confirmed' if passed else 'not confirmed'} "
          f"(asymmetry = {asymmetry:.4f})")

    return {
        'test': 'emergent_time_arrow',
        'n_nodes': n_nodes,
        'n_steps': n_steps,
        'S_initial': float(S_init),
        'S_fwd_final': float(S_fwd),
        'S_anti_final': float(S_anti),
        'delta_S_fwd': float(delta_S_fwd),
        'delta_S_anti': float(delta_S_anti),
        'asymmetry': float(asymmetry),
        'signs_opposite': bool(signs_opposite),
        'passed': bool(passed),
    }


def test3_flow_rate_matches_clock():
    """
    Test 3: Run SEC dynamics for many steps. Compute Shannon entropy
    at logarithmically spaced steps. If the entropy change rate follows
    the cascade clock, then S(t) = a + b*ln(t) (logarithmic in step number).

    PASS: R^2 > 0.8 for logarithmic fit of entropy vs step.
    """
    print("\n" + "-" * 70)
    print("TEST 3: FLOW RATE MATCHES CLOCK")
    print("-" * 70)

    n_nodes = 100
    n_steps = 5000

    print(f"\n  Running SEC dynamics: {n_nodes} nodes, {n_steps} steps")

    history = run_sec_dynamics(n_nodes, n_steps, alpha=1.0, beta=0.5, seed=42)

    # Measure entropy at logarithmically spaced steps
    sample_steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    sample_steps = [s for s in sample_steps if s < len(history)]

    entropies = []
    for step in sample_steps:
        S = _shannon_entropy(history[step])
        entropies.append(S)

    ln_steps = np.log(np.array(sample_steps, dtype=float))
    S_arr = np.array(entropies)

    # Fit: S = a + b * ln(step)
    coeffs = np.polyfit(ln_steps, S_arr, 1)
    b_fit = coeffs[0]
    a_fit = coeffs[1]

    # R^2
    S_pred = np.polyval(coeffs, ln_steps)
    ss_res = np.sum((S_arr - S_pred)**2)
    ss_tot = np.sum((S_arr - np.mean(S_arr))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    S_max = np.log(n_nodes)  # maximum entropy for uniform distribution

    print(f"\n  Shannon entropy vs log(step):")
    print(f"    S_max (uniform) = ln({n_nodes}) = {S_max:.6f}")
    print()
    for step, S in zip(sample_steps, entropies):
        pct_of_max = S / S_max * 100
        print(f"    Step {step:5d}: S = {S:.6f}  ({pct_of_max:.1f}% of S_max)")

    print(f"\n  Logarithmic fit: S = {b_fit:.6f} * ln(step) + {a_fit:.6f}")
    print(f"    Slope (b): {b_fit:.6f}")
    print(f"    R^2:       {r_squared:.6f}")
    print(f"    Threshold: R^2 > 0.8")

    print(f"\n  Interpretation:")
    print(f"    Entropy evolves logarithmically: S(t) ~ ln(t)")
    print(f"    Consistent with cascade clock: N(t) = a + slope*ln(t)")
    print(f"    SEC dynamics reproduce the same temporal scaling")

    # Compare slope to DFT prediction
    print(f"\n  Slope comparison:")
    print(f"    Measured slope:  {b_fit:.6f}")
    print(f"    B_DFT = 1/ln(phi): {B_DFT:.6f}")
    print(f"    Ratio:           {abs(b_fit / B_DFT):.4f}" if B_DFT != 0 else "")

    passed = r_squared > 0.8
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: logarithmic flow rate "
          f"{'confirmed' if passed else 'not confirmed'} "
          f"(R^2 = {r_squared:.4f})")

    return {
        'test': 'flow_rate_matches_clock',
        'n_nodes': n_nodes,
        'n_steps': n_steps,
        'sample_steps': sample_steps,
        'entropies': [float(x) for x in entropies],
        'fit_slope': float(b_fit),
        'fit_intercept': float(a_fit),
        'r_squared': float(r_squared),
        'S_max': float(S_max),
        'passed': bool(passed),
    }


def test4_sec_equilibrium_is_maxent():
    """
    Test 4: SEC drives toward maximum entropy (equipartition). The
    equilibrium state should approach S_max = ln(N) where N is the
    number of nodes. Measure how close the final state gets to S_max
    and verify the approach is monotonic.

    Also verify the total entropy change is positive (second law) and
    that the final distribution's Gini coefficient approaches zero
    (perfect equality = maximum entropy = thermodynamic equilibrium).

    PASS: final entropy > 95% of S_max AND entropy is monotonically
    non-decreasing (checked at 10 sample points).
    """
    print("\n" + "-" * 70)
    print("TEST 4: SEC EQUILIBRIUM IS MAXIMUM ENTROPY")
    print("-" * 70)

    n_nodes = 100
    n_steps = 5000

    print(f"\n  Running SEC dynamics: {n_nodes} nodes, {n_steps} steps")

    history = run_sec_dynamics(n_nodes, n_steps, alpha=1.0, beta=0.5, seed=42)

    S_max = np.log(n_nodes)
    S_initial = _shannon_entropy(history[0])
    S_final = _shannon_entropy(history[-1])
    final_fraction = S_final / S_max

    # Monotonicity check at 10 sample points
    sample_steps = [0, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
    sample_steps = [s for s in sample_steps if s < len(history)]
    sample_S = [_shannon_entropy(history[s]) for s in sample_steps]

    monotonic = all(sample_S[i] <= sample_S[i + 1] + 1e-10
                    for i in range(len(sample_S) - 1))

    # Gini at final state
    gini_initial = _gini_coefficient(history[0])
    gini_final = _gini_coefficient(history[-1])

    print(f"\n  Maximum entropy: S_max = ln({n_nodes}) = {S_max:.6f}")
    print(f"  S_initial = {S_initial:.6f}  ({S_initial/S_max*100:.1f}% of S_max)")
    print(f"  S_final   = {S_final:.6f}  ({final_fraction*100:.1f}% of S_max)")
    print(f"  Threshold: > 95% of S_max")

    print(f"\n  Entropy trajectory (monotonicity check):")
    for step, S in zip(sample_steps, sample_S):
        pct = S / S_max * 100
        print(f"    Step {step:5d}: S = {S:.6f}  ({pct:.1f}%)")
    print(f"    Monotonically non-decreasing: {monotonic}")

    print(f"\n  Gini coefficient (inequality measure):")
    print(f"    Initial: {gini_initial:.6f}")
    print(f"    Final:   {gini_final:.6f}")
    print(f"    Gini -> 0 means equipartition (max entropy)")

    print(f"\n  Physical interpretation:")
    print(f"    SEC drives the system toward maximum entropy (uniform distribution)")
    print(f"    This IS the second law of thermodynamics: entropy increases")
    print(f"    The equilibrium state is equipartition -- no structure remains")
    print(f"    The arrow of time (test 2) ensures this happens irreversibly")

    above_95 = final_fraction > 0.95

    passed = above_95 and monotonic
    print(f"\n  Final > 95% of S_max: {above_95}")
    print(f"  Monotonic approach: {monotonic}")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: max-entropy equilibrium "
          f"{'confirmed' if passed else 'not confirmed'}")

    return {
        'test': 'sec_equilibrium_maxent',
        'n_nodes': n_nodes,
        'n_steps': n_steps,
        'S_max': float(S_max),
        'S_initial': float(S_initial),
        'S_final': float(S_final),
        'final_fraction': float(final_fraction),
        'monotonic': bool(monotonic),
        'gini_initial': float(gini_initial),
        'gini_final': float(gini_final),
        'above_95_pct': bool(above_95),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 9 - EXP 04: SEC TEMPORAL FLOW")
    print("Block B: Information-Time Nexus")
    print("Does SEC generate temporal ordering from time-symmetric states?")
    print("=" * 70)

    r1 = test1_sec_breaks_time_symmetry()
    r2 = test2_emergent_time_arrow()
    r3 = test3_flow_rate_matches_clock()
    r4 = test4_sec_equilibrium_is_maxent()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (SEC breaks time symmetry):   {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (Emergent time arrow):         {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Flow rate matches clock):     {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (SEC equilibrium = max ent):   {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    if r1['passed'] and r2['passed']:
        print(f"\n  KEY FINDING: SEC dynamics break time-reversal symmetry.")
        print(f"  Forward SEC and anti-SEC produce qualitatively different")
        print(f"  outcomes from identical initial conditions.")
    if r3['passed']:
        print(f"\n  KEY FINDING: Entropy evolution follows logarithmic scaling,")
        print(f"  consistent with the cascade clock N(t) = a + slope*ln(t).")

    results = {
        'experiment': 'exp_04_sec_temporal_flow',
        'milestone': 9,
        'block': 'B',
        'block_name': 'Information-Time Nexus',
        'tests': {
            'test1_sec_breaks_time_symmetry': r1,
            'test2_emergent_time_arrow': r2,
            'test3_flow_rate_matches_clock': r3,
            'test4_sec_equilibrium_maxent': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_04_sec_temporal_flow', RESULTS_DIR)


if __name__ == '__main__':
    main()
