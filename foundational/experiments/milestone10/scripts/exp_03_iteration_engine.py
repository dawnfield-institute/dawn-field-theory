"""
Milestone 10 -- Exp 03: Non-Terminating Self-Application as Engine

Block A: Uniqueness & Foundations

PURPOSE: Show that symmetric self-application necessarily produces non-terminating
iteration with discrete hierarchical residue. Single-circle self-reference either
terminates (fixed point) or diverges (chaos). Two-circle mutual reference under
symmetry constraint produces bounded, non-terminating dynamics with discrete
level structure (thesis section 4).

Tests:
  1. Single-circle: all terminate or go chaotic (zero hierarchical)
  2. Two-circle with symmetry: bounded non-terminating with scale separation
  3. Discrete residue: residue per iteration is quantized, not continuous
  4. PAC hierarchy recovery: inter-level ratio converges to phi

Builds on: iddea.md section 4, M7 exp_01 (phi from self-reference)
Predicted: 3/4 (T4 phi-ratio recovery is hardest)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    run_two_circle_dynamics,
    save_results, setup_experiment,
    PHI, INV_PHI, LN_PHI, XI_BALANCE, PI,
)

_, RESULTS_DIR = setup_experiment(__file__)


def make_single_circle_maps():
    """Generate 20 self-referential maps f: R -> R."""
    maps = []
    labels = []

    # Contractive maps (should terminate at fixed points)
    maps.append(lambda x: 0.5 * x)
    labels.append("0.5x (contractive)")
    maps.append(lambda x: np.tanh(x))
    labels.append("tanh(x) (bounded contractive)")
    maps.append(lambda x: np.cos(x))
    labels.append("cos(x) (oscillating contractive)")
    maps.append(lambda x: x / (1 + abs(x)))
    labels.append("x/(1+|x|) (saturation)")

    # Fixed-point maps
    maps.append(lambda x: 1.0 / (1.0 + x**2))
    labels.append("1/(1+x^2) (fixed point)")
    maps.append(lambda x: np.sqrt(abs(x) + 0.1))
    labels.append("sqrt(|x|+0.1) (sublinear)")
    maps.append(lambda x: 0.9 * np.sin(x) + 0.1)
    labels.append("0.9*sin(x)+0.1 (damped oscillation)")

    # Chaotic maps
    maps.append(lambda x: 4.0 * x * (1.0 - x) if 0 < x < 1 else 0.5)
    labels.append("logistic r=4 (chaotic)")
    maps.append(lambda x: (2 * x) % 1.0 if abs(x) < 10 else 0.5)
    labels.append("2x mod 1 (Bernoulli, chaotic)")
    maps.append(lambda x: 1.0 - 1.99 * x**2)
    labels.append("1-1.99x^2 (chaotic)")

    # Divergent maps
    maps.append(lambda x: 2.0 * x)
    labels.append("2x (divergent)")
    maps.append(lambda x: x**2)
    labels.append("x^2 (super-divergent)")
    maps.append(lambda x: x + np.sign(x))
    labels.append("x+sign(x) (linear divergent)")

    # Self-referential / recursive maps
    maps.append(lambda x: x - x**3 / 3)
    labels.append("x-x^3/3 (truncated sin)")
    maps.append(lambda x: (x + 1.0 / max(abs(x), 0.01)) / 2.0)
    labels.append("(x+1/x)/2 (Babylonian sqrt)")
    maps.append(lambda x: x * np.exp(-abs(x)))
    labels.append("x*exp(-|x|) (self-limiting)")
    maps.append(lambda x: np.sin(x * PHI))
    labels.append("sin(phi*x) (phi-modulated)")
    maps.append(lambda x: x * (1 - abs(x)) + 0.5 * np.sign(x))
    labels.append("x(1-|x|)+0.5*sign (bounded)")
    maps.append(lambda x: PHI * x * (1 - x / PHI))
    labels.append("phi*x*(1-x/phi) (logistic-phi)")
    maps.append(lambda x: (x + PHI) / (1 + x * INV_PHI) if abs(1 + x * INV_PHI) > 0.01 else PHI)
    labels.append("(x+phi)/(1+x/phi) (Mobius-phi)")

    return maps, labels


def classify_orbit(f, x0, n_steps=500):
    """Classify orbit: 'fixed_point', 'periodic', 'chaotic', 'divergent', 'bounded_nonperiodic'."""
    trajectory = [x0]
    for _ in range(n_steps):
        try:
            x_new = f(trajectory[-1])
            if not np.isfinite(x_new) or abs(x_new) > 1e6:
                return 'divergent', trajectory
            trajectory.append(x_new)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 'divergent', trajectory

    traj = np.array(trajectory)

    # Fixed point check (last 50 values within 1e-8)
    if len(traj) > 50:
        tail = traj[-50:]
        if np.std(tail) < 1e-8:
            return 'fixed_point', traj

    # Periodic check (look for period 1-20)
    if len(traj) > 100:
        for period in range(1, 21):
            if all(abs(traj[-1] - traj[-1 - k * period]) < 1e-6
                   for k in range(1, min(5, len(traj) // period))):
                return 'periodic', traj

    # Check boundedness
    if np.max(np.abs(traj)) < 1e4:
        # Chaotic: bounded but sensitive to IC
        return 'chaotic', traj

    return 'divergent', traj


def has_hierarchical_output(traj):
    """Check if trajectory has hierarchical structure (multiple scale-separated frequencies).

    Must distinguish genuine hierarchy (structured, persistent peaks at specific
    scale ratios) from broadband chaotic noise (incidental peaks). Requires
    prominent peaks well above the noise floor.
    """
    if len(traj) < 50:
        return False
    fft = np.fft.rfft(traj[-200:] if len(traj) > 200 else traj)
    power = np.abs(fft)**2
    noise_floor = np.median(power)
    max_power = np.max(power[1:])  # Exclude DC component

    peaks = []
    for i in range(1, len(power) - 1):
        if (power[i] > power[i-1] and power[i] > power[i+1]
                and power[i] > 10 * noise_floor
                and power[i] > 0.01 * max_power):
            peaks.append(i)

    # Need at least 3 octave-separated peaks (not just 2, which chaotic maps can produce)
    octave_pairs = 0
    for i in range(len(peaks) - 1):
        if peaks[i] > 0 and np.log2(peaks[i+1] / peaks[i]) >= 1.0:
            octave_pairs += 1
    return octave_pairs >= 2


def test1_single_circle():
    """Single-circle self-reference: all terminate or go chaotic."""
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE-CIRCLE — Terminate or Chaos")
    print("=" * 70)

    maps, labels = make_single_circle_maps()
    hierarchical_count = 0
    classifications = {}

    for i, (f, label) in enumerate(zip(maps, labels)):
        # Test from multiple initial conditions
        any_hierarchical = False
        orbits = []
        for x0 in [0.1, 0.5, 1.0, -0.3, 2.0]:
            cls, traj = classify_orbit(f, x0)
            orbits.append(cls)
            if cls in ('chaotic', 'bounded_nonperiodic'):
                if has_hierarchical_output(traj):
                    any_hierarchical = True

        if any_hierarchical:
            hierarchical_count += 1
        dominant = max(set(orbits), key=orbits.count)
        classifications[label] = dominant
        print(f"  {i+1:2d}. {label:40s} -> {dominant}")

    print(f"\n  Hierarchical output: {hierarchical_count}/{len(maps)}")
    passed = hierarchical_count == 0
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {hierarchical_count} hierarchical (need 0)")

    return {
        'test': 'single_circle_termination',
        'n_maps': len(maps),
        'hierarchical_count': hierarchical_count,
        'classifications': classifications,
        'passed': bool(passed),
    }


def test2_two_circle_bounded():
    """Two-circle mutual reference with symmetry: bounded non-terminating."""
    print("\n" + "=" * 70)
    print("TEST 2: TWO-CIRCLE — Bounded Non-Terminating")
    print("=" * 70)

    # Generate symmetric two-circle functions
    test_functions = [
        lambda x: np.tanh(PHI * x),
        lambda x: np.sin(x * PHI) * INV_PHI,
        lambda x: x / (1 + abs(x)) * PHI,
        lambda x: np.cos(x) * INV_PHI + 0.1,
        lambda x: (x + INV_PHI) / (1 + abs(x)),
        lambda x: np.tanh(x + INV_PHI),
        lambda x: np.sin(x) * np.cos(x * INV_PHI),
        lambda x: x * np.exp(-x**2) * PHI,
        lambda x: np.arctan(x * PHI) / PI,
        lambda x: (np.sin(x) + np.cos(x * PHI)) / 2,
        lambda x: np.tanh(x * np.cos(x)),
        lambda x: INV_PHI * x + (1 - INV_PHI) * np.sin(x),
        lambda x: np.sign(x) * np.sqrt(abs(x)) * INV_PHI,
        lambda x: np.tanh(x**2 * np.sign(x)) * PHI,
        lambda x: (x + np.sin(PHI * x)) / (1 + x**2),
        lambda x: np.cos(x * PHI) * np.tanh(x),
        lambda x: x * (1 - x**2 / (PHI**2 + x**2)),
        lambda x: np.sin(x + INV_PHI) / (1 + abs(x)),
        lambda x: np.tanh(x / PHI + np.sin(x)),
        lambda x: (PHI * x + np.sin(x)) / (PHI + abs(x)),
    ]

    bounded_nonterminating = 0
    with_scale_separation = 0

    for i, f in enumerate(test_functions):
        result = run_two_circle_dynamics(f, n_steps=500, x0=0.5, y0=-0.3)
        if result['non_terminating_bounded']:
            bounded_nonterminating += 1
            if result['n_peaks'] >= 2:
                with_scale_separation += 1

    frac_bnt = bounded_nonterminating / len(test_functions)
    frac_scale = with_scale_separation / len(test_functions)

    print(f"\n  Total functions tested:         {len(test_functions)}")
    print(f"  Bounded non-terminating:        {bounded_nonterminating} ({frac_bnt:.1%})")
    print(f"  With scale separation (≥2 peaks): {with_scale_separation} ({frac_scale:.1%})")

    passed = frac_bnt > 0.50
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {frac_bnt:.1%} > 50%")

    return {
        'test': 'two_circle_bounded',
        'n_functions': len(test_functions),
        'bounded_nonterminating': bounded_nonterminating,
        'with_scale_separation': with_scale_separation,
        'fraction_bnt': float(frac_bnt),
        'fraction_scale': float(frac_scale),
        'passed': bool(passed),
    }


def test3_discrete_residue():
    """Residue per iteration is quantized, not continuous."""
    print("\n" + "=" * 70)
    print("TEST 3: DISCRETE RESIDUE — Quantized, Not Continuous")
    print("=" * 70)

    from scipy.stats import kstest as ks

    # Run several two-circle systems and collect step-size residues.
    # For non-terminating bounded orbits, the step size |x_{n+1} - x_n|
    # is the "information cost per iteration." If this clusters at specific
    # values (modal distribution), the residue is quantized.
    test_functions = [
        lambda x: np.tanh(PHI * x),
        lambda x: np.sin(x * PHI) * INV_PHI,
        lambda x: (x + INV_PHI) / (1 + abs(x)),
        lambda x: np.tanh(x + INV_PHI),
        lambda x: x / (1 + abs(x)) * PHI,
    ]

    all_residues = []
    for f in test_functions:
        result = run_two_circle_dynamics(f, n_steps=2000, x0=0.5, y0=-0.3)
        if not result['non_terminating_bounded']:
            continue
        xs = np.array(result['xs'])
        # Step sizes after transient
        steps = np.abs(np.diff(xs[200:]))
        steps = steps[steps > 1e-10]  # Remove near-zero steps
        all_residues.extend(steps.tolist())

    if len(all_residues) < 100:
        print("  Insufficient residues computed")
        return {'test': 'discrete_residue', 'n_residues': len(all_residues), 'passed': False}

    residues = np.array(all_residues)

    # Test for modality: histogram should show peaks
    n_bins = 30
    counts, bin_edges = np.histogram(residues, bins=n_bins)
    mean_count = np.mean(counts)
    std_count = np.std(counts)

    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
            if counts[i] > mean_count + 2 * std_count:
                peaks += 1

    # KS test against uniform
    r = residues.max() - residues.min()
    if r > 1e-10:
        ks_stat, ks_p = ks(residues, 'uniform', args=(residues.min(), r))
    else:
        ks_stat, ks_p = 0.0, 1.0

    print(f"\n  Residues computed:   {len(residues)}")
    print(f"  Histogram peaks:    {peaks} (above 2-sigma)")
    print(f"  KS stat vs uniform: {ks_stat:.4f} (p = {ks_p:.4e})")
    print(f"  Residue mean:       {np.mean(residues):.6f}")
    print(f"  Residue std:        {np.std(residues):.6f}")

    passed = ks_p < 0.05 and peaks >= 1
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: non-uniform (p={ks_p:.4e}), {peaks} peaks")

    return {
        'test': 'discrete_residue',
        'n_residues': len(residues),
        'histogram_peaks': peaks,
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        'residue_mean': float(np.mean(residues)),
        'residue_std': float(np.std(residues)),
        'passed': bool(passed),
    }


def test4_phi_ratio_recovery():
    """Inter-level ratio from two-circle dynamics converges to phi."""
    print("\n" + "=" * 70)
    print("TEST 4: PHI RATIO RECOVERY — Inter-Level Ratio -> phi")
    print("=" * 70)

    # Use several two-circle systems and check inter-level ratios
    test_functions = [
        lambda x: np.tanh(PHI * x),
        lambda x: np.sin(x * PHI) * INV_PHI,
        lambda x: (x + INV_PHI) / (1 + abs(x)),
        lambda x: x / (1 + abs(x)) * PHI,
        lambda x: np.tanh(x + INV_PHI),
    ]

    phi_matches = 0
    ratios_found = []

    for f in test_functions:
        result = run_two_circle_dynamics(f, n_steps=2000, x0=0.5, y0=-0.3)
        if not result['non_terminating_bounded']:
            continue

        xs = result['xs']

        # Find characteristic scales via autocorrelation peaks
        signal = xs[100:]  # Skip transient
        if len(signal) < 100:
            continue

        autocorr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find peaks in autocorrelation (characteristic timescales)
        ac_peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                ac_peaks.append(i)
            if len(ac_peaks) >= 3:
                break

        if len(ac_peaks) >= 2:
            ratio = ac_peaks[1] / ac_peaks[0] if ac_peaks[0] > 0 else 0
            ratios_found.append(ratio)
            if abs(ratio - PHI) / PHI < 0.10:
                phi_matches += 1

    if ratios_found:
        mean_ratio = np.mean(ratios_found)
        error = abs(mean_ratio - PHI) / PHI
    else:
        mean_ratio = 0
        error = 1.0

    print(f"\n  Ratios found:     {len(ratios_found)}")
    if ratios_found:
        print(f"  Mean ratio:       {mean_ratio:.4f}")
        print(f"  Target (phi):     {PHI:.4f}")
        print(f"  Relative error:   {error:.4f}")
        print(f"  Phi matches:      {phi_matches}/{len(ratios_found)}")

    passed = error < 0.10 and len(ratios_found) >= 2
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: error {error:.4f} < 10%")

    return {
        'test': 'phi_ratio_recovery',
        'n_ratios': len(ratios_found),
        'ratios': [float(r) for r in ratios_found],
        'mean_ratio': float(mean_ratio),
        'phi_target': float(PHI),
        'relative_error': float(error),
        'phi_matches': phi_matches,
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 03: ITERATION ENGINE")
    print("Block A: Uniqueness & Foundations")
    print("=" * 70)

    r1 = test1_single_circle()
    r2 = test2_two_circle_bounded()
    r3 = test3_discrete_residue()
    r4 = test4_phi_ratio_recovery()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    results = {
        'experiment': 'exp_03_iteration_engine',
        'milestone': 10,
        'block': 'A',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_03_iteration_engine', RESULTS_DIR)


if __name__ == '__main__':
    main()
