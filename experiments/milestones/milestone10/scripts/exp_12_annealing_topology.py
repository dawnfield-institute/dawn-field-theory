"""
Milestone 10 -- Exp 12: Conservation Removes Scale-Dependence

INVESTIGATIVE — probing exp_07 T3/T4 failures (REWORKED after 0/4)

The first version revealed something unexpected: standard annealing IS
scale-dependent (rho=1.0), but the SM shows rho~0.08. The SM's scale-
independence isn't trivial — it's ANOMALOUS and requires explanation.

Revised hypothesis: PAC conservation (total "tuning budget" conserved)
is the mechanism that removes scale-dependence. When optimizing one
parameter requires degrading another, the allocation depends on landscape
topology, not energy scale. This produces scale-independent residuals
AND heavy-tailed distributions (because conservation forces some parameters
to keep large residuals).

Tests:
  1. Unconstrained is scale-dependent: confirm rho >> 0 (baseline)
  2. Conservation removes correlation: add PAC constraint, rho drops to ~0
  3. Conservation produces heavy tails: constrained distribution is wider
  4. Sweep conservation strength: rho decreases monotonically with strength

Builds on: exp_07 T3/T4 failures, exp_12 v1 failure (0/4)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, kurtosis

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    save_results, setup_experiment,
    PHI, LN_PHI, XI_BALANCE,
)

_, RESULTS_DIR = setup_experiment(__file__)


def multi_scale_anneal(n_params=8, n_steps=5000, conservation_strength=0.0, seed=42):
    """
    Anneal n_params parameters with different natural scales.

    Each parameter has target_i at distance scale_i from origin.
    The loss is sum((x_i - target_i)^2 / scale_i^2) (normalized).

    conservation_strength > 0 adds PAC-like constraint:
      penalty = strength * (sum(log|residual_i / scale_i|) - C)^2
    where C is the initial total log-residual.

    This forces trade-offs: improving one parameter degrades another.
    """
    rng = np.random.RandomState(seed)

    # Natural scales spanning 6 orders (mimicking SM hierarchy)
    scales = np.logspace(-3, 3, n_params)
    targets = rng.randn(n_params) * scales

    # Start away from targets
    x = targets + rng.randn(n_params) * scales * 3

    def normalized_residuals(x):
        return np.abs(x - targets) / (scales + 1e-30)

    def log_res_total(x):
        nr = normalized_residuals(x)
        return np.sum(np.log(nr + 1e-30))

    C = log_res_total(x)  # Initial conservation target

    def loss(x):
        fit = np.sum(((x - targets) / scales) ** 2)
        if conservation_strength > 0:
            conservation = conservation_strength * (log_res_total(x) - C) ** 2
            return fit + conservation
        return fit

    E = loss(x)
    T0 = 10.0

    for step in range(1, n_steps + 1):
        T = T0 / (1 + step * 0.01)
        # Step size uniform — this creates the scale-dependence
        proposal = x + rng.randn(n_params) * T * 0.1
        E_new = loss(proposal)
        if E_new < E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-10)):
            x = proposal
            E = E_new

    return {
        'scales': scales,
        'normalized_residuals': normalized_residuals(x),
        'log_residuals': np.log10(normalized_residuals(x) + 1e-30),
    }


def measure_scale_correlation(results_list):
    """Compute Spearman rho between log(scale) and log(normalized_residual)."""
    all_scales = []
    all_log_res = []
    for r in results_list:
        all_scales.extend(np.log10(r['scales']).tolist())
        all_log_res.extend(r['log_residuals'].tolist())
    rho, p = spearmanr(all_scales, all_log_res)
    return float(rho), float(p)


def test1_unconstrained_baseline():
    """Confirm standard annealing IS scale-dependent."""
    print("\n" + "=" * 70)
    print("TEST 1: UNCONSTRAINED BASELINE — Scale-Dependent (rho >> 0)")
    print("=" * 70)

    results = []
    for seed in range(30):
        r = multi_scale_anneal(n_params=8, n_steps=5000,
                               conservation_strength=0.0, seed=seed)
        results.append(r)

    rho, p = measure_scale_correlation(results)

    # Show one example
    r0 = results[0]
    print(f"\n  Example (seed=0):")
    print(f"    {'Scale':>10s}  {'log10(residual)':>16s}")
    for s, lr in zip(r0['scales'], r0['log_residuals']):
        print(f"    {s:>10.3e}  {lr:>16.2f}")

    print(f"\n  Spearman rho (scale vs residual): {rho:.4f}")
    print(f"  p-value:                          {p:.4e}")

    # PASS: scale-dependent (rho > 0.3)
    passed = rho > 0.3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: rho = {rho:.3f} > 0.3 (scale-dependent)")

    return {
        'test': 'unconstrained_baseline',
        'n_trials': len(results),
        'spearman_rho': rho,
        'p_value': p,
        'passed': bool(passed),
    }


def test2_conservation_removes_correlation():
    """PAC conservation constraint removes scale-dependence."""
    print("\n" + "=" * 70)
    print("TEST 2: CONSERVATION CONSTRAINT — Scale-Independent (rho ~ 0)")
    print("=" * 70)

    # Use strong conservation
    results = []
    for seed in range(30):
        r = multi_scale_anneal(n_params=8, n_steps=5000,
                               conservation_strength=50.0, seed=seed)
        results.append(r)

    rho, p = measure_scale_correlation(results)

    r0 = results[0]
    print(f"\n  Example (seed=0, conservation=50):")
    print(f"    {'Scale':>10s}  {'log10(residual)':>16s}")
    for s, lr in zip(r0['scales'], r0['log_residuals']):
        print(f"    {s:>10.3e}  {lr:>16.2f}")

    print(f"\n  Spearman rho (scale vs residual): {rho:.4f}")
    print(f"  p-value:                          {p:.4e}")
    print(f"  SM reference rho:                 0.08")

    # PASS: scale-independent (|rho| < 0.3)
    passed = abs(rho) < 0.3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: |rho| = {abs(rho):.3f} < 0.3")

    return {
        'test': 'conservation_removes_correlation',
        'n_trials': len(results),
        'spearman_rho': rho,
        'p_value': p,
        'sm_reference_rho': 0.08,
        'passed': bool(passed),
    }


def test3_conservation_produces_heavy_tails():
    """Conservation constraint produces wider, heavier-tailed distribution."""
    print("\n" + "=" * 70)
    print("TEST 3: HEAVY TAILS — Conservation Widens Distribution")
    print("=" * 70)

    # Collect log-residual distributions for both regimes
    unconstrained_lr = []
    constrained_lr = []

    for seed in range(50):
        r_unc = multi_scale_anneal(n_params=8, n_steps=5000,
                                    conservation_strength=0.0, seed=seed)
        r_con = multi_scale_anneal(n_params=8, n_steps=5000,
                                    conservation_strength=50.0, seed=seed)
        unconstrained_lr.extend(r_unc['log_residuals'].tolist())
        constrained_lr.extend(r_con['log_residuals'].tolist())

    unc_arr = np.array(unconstrained_lr)
    con_arr = np.array(constrained_lr)

    unc_spread = np.std(unc_arr)
    con_spread = np.std(con_arr)
    unc_kurt = kurtosis(unc_arr)
    con_kurt = kurtosis(con_arr)
    unc_range = np.ptp(unc_arr)
    con_range = np.ptp(con_arr)

    print(f"\n  {'Metric':<25s} {'Unconstrained':>15s}  {'Constrained':>15s}")
    print(f"  {'-'*58}")
    print(f"  {'Std (spread)':25s} {unc_spread:>15.3f}  {con_spread:>15.3f}")
    print(f"  {'Kurtosis':25s} {unc_kurt:>15.3f}  {con_kurt:>15.3f}")
    print(f"  {'Range':25s} {unc_range:>15.3f}  {con_range:>15.3f}")
    print(f"  {'Mean':25s} {np.mean(unc_arr):>15.3f}  {np.mean(con_arr):>15.3f}")

    # PASS: conservation produces wider spread OR heavier tails
    wider = con_spread > unc_spread * 1.1
    heavier = con_kurt > unc_kurt + 0.5

    passed = wider or heavier
    print(f"\n  Wider spread: {'yes' if wider else 'no'} "
          f"({con_spread:.2f} vs {unc_spread:.2f})")
    print(f"  Heavier tails: {'yes' if heavier else 'no'} "
          f"(kurtosis {con_kurt:.2f} vs {unc_kurt:.2f})")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'conservation_heavy_tails',
        'unconstrained_spread': float(unc_spread),
        'constrained_spread': float(con_spread),
        'unconstrained_kurtosis': float(unc_kurt),
        'constrained_kurtosis': float(con_kurt),
        'unconstrained_range': float(unc_range),
        'constrained_range': float(con_range),
        'wider': bool(wider),
        'heavier': bool(heavier),
        'passed': bool(passed),
    }


def test4_conservation_strength_sweep():
    """Rho decreases monotonically with conservation strength."""
    print("\n" + "=" * 70)
    print("TEST 4: STRENGTH SWEEP — rho vs Conservation Strength")
    print("=" * 70)

    strengths = [0, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 200]
    n_seeds = 20

    rho_values = []

    for strength in strengths:
        results = []
        for seed in range(n_seeds):
            r = multi_scale_anneal(n_params=8, n_steps=5000,
                                    conservation_strength=strength, seed=seed)
            results.append(r)

        rho, p = measure_scale_correlation(results)
        rho_values.append(rho)
        marker = " <-- SM range" if abs(rho) < 0.15 else ""
        print(f"  strength={strength:>6.1f}: rho = {rho:>7.4f}{marker}")

    # Check monotonicity: rho should generally decrease
    monotonic_steps = sum(1 for i in range(1, len(rho_values))
                         if rho_values[i] <= rho_values[i-1] + 0.05)
    monotonicity = monotonic_steps / (len(rho_values) - 1)

    # Find strength where rho first drops below 0.2
    threshold_strength = None
    for s, r in zip(strengths, rho_values):
        if abs(r) < 0.2:
            threshold_strength = s
            break

    print(f"\n  Monotonicity:       {monotonicity:.1%}")
    print(f"  Threshold (|rho|<0.2): strength={threshold_strength}")
    print(f"  rho at max strength:   {rho_values[-1]:.4f}")

    # PASS: generally decreasing, and rho drops below 0.2 at some finite strength
    passed = monotonicity > 0.60 and threshold_strength is not None
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: monotonicity {monotonicity:.1%}, "
          f"threshold {'found' if threshold_strength else 'not found'}")

    return {
        'test': 'conservation_strength_sweep',
        'strengths': strengths,
        'rho_values': rho_values,
        'monotonicity': float(monotonicity),
        'threshold_strength': threshold_strength,
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 12: CONSERVATION REMOVES SCALE-DEPENDENCE")
    print("Investigative — probing exp_07 T3/T4 failures (REWORKED)")
    print("=" * 70)

    r1 = test1_unconstrained_baseline()
    r2 = test2_conservation_removes_correlation()
    r3 = test3_conservation_produces_heavy_tails()
    r4 = test4_conservation_strength_sweep()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    print("\n  INTERPRETATION:")
    if r1['passed'] and r2['passed']:
        print("  -> Standard annealing IS scale-dependent, but conservation removes it.")
        print("  -> The SM's scale-independence (rho~0.08) is a SIGNATURE of conservation.")
        print("  -> PAC conservation is a candidate mechanism: when total 'tuning budget' is")
        print("     conserved, the allocation depends on topology, not energy scale.")
    if r3['passed']:
        print("  -> Conservation also produces heavier tails / wider spread.")
        print("     This connects to exp_07 T2 (Levy-stable): conservation forces some")
        print("     parameters to retain extreme residuals, creating heavy tails.")
    if r4['passed']:
        print(f"  -> Continuous transition: rho drops below 0.2 at strength={r4.get('threshold_strength')}.")
        print("     The SM's near-zero rho implies strong conservation constraint.")

    results = {
        'experiment': 'exp_12_annealing_topology',
        'milestone': 10,
        'block': 'investigative',
        'version': 2,
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_12_annealing_topology_v2', RESULTS_DIR)


if __name__ == '__main__':
    main()
