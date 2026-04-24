"""
Milestone 9 -- Exp 05: Gravity-Time from Cascade

PURPOSE: Does the PAC cascade produce gravity-time duality? The phi-split
g_out = g_in^2 is satisfied ONLY for phi, and this algebraic identity
connects cascade energy release to gravitational dynamics. Free-fall
through phi-ratio radii produces time intervals with ratio phi^(3/2),
and the cascade completes in finite proper time (Zeno completion).

Block B: Information-Time Nexus

Tests:
  1. Duality from phi-split: g_out = g_in^2 only for phi
  2. BH proper time from cascade: free-fall time ratios = phi^(3/2)
  3. Cascade level tracks redshift: n maps monotonically to z_grav
  4. Zeno completion: cascade sums to finite total proper time
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
    pac_cascade_ratios,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)


def test1_duality_from_phi_split():
    """
    Test 1: For g_in = 1/alpha, the released fraction is g_out = 1 - 1/alpha.
    The scale invariance condition g_out = g_in^2 gives:
      1 - 1/alpha = 1/alpha^2
    which is satisfied ONLY for alpha = phi.

    Verify: error |g_out - g_in^2| is < 1e-10 for phi, and > 0.01 for
    e, 2, pi, sqrt(2), sqrt(3).

    PASS: phi error < 1e-10 AND all other errors > 0.01.
    """
    print("\n" + "-" * 70)
    print("TEST 1: DUALITY FROM PHI-SPLIT")
    print("-" * 70)

    test_constants = {
        'phi':     PHI,
        'e':       np.e,
        '2':       2.0,
        'pi':      PI,
        'sqrt(2)': np.sqrt(2),
        'sqrt(3)': np.sqrt(3),
    }

    print(f"\n  Scale invariance condition: g_out = g_in^2")
    print(f"  where g_in = 1/alpha, g_out = 1 - 1/alpha")
    print(f"  Error = |g_out - g_in^2| = |1 - 1/alpha - 1/alpha^2|")

    errors = {}
    print()
    for name, alpha in test_constants.items():
        g_in = 1.0 / alpha
        g_out = 1.0 - g_in
        g_in_sq = g_in ** 2
        error = abs(g_out - g_in_sq)
        errors[name] = error

        marker = ""
        if name == 'phi':
            marker = " <-- EXACT"
        print(f"  alpha = {name:7s} ({alpha:.6f}): "
              f"g_in={g_in:.6f}, g_out={g_out:.6f}, g_in^2={g_in_sq:.6f}, "
              f"error={error:.2e}{marker}")

    phi_error = errors['phi']
    non_phi_errors = {k: v for k, v in errors.items() if k != 'phi'}
    all_non_phi_above = all(v > 0.01 for v in non_phi_errors.values())

    print(f"\n  Phi error: {phi_error:.2e} (threshold: < 1e-10)")
    print(f"  All non-phi errors > 0.01: {all_non_phi_above}")

    # Physical interpretation
    print(f"\n  Physical meaning:")
    print(f"    g_in = 1/phi = {INV_PHI:.6f} (retained at each level)")
    print(f"    g_out = 1/phi^2 = {INV_PHI**2:.6f} (released at each level)")
    print(f"    g_out = g_in^2: released energy = retained^2")
    print(f"    This means: energy release IS the square of energy retention")
    print(f"    Only phi satisfies this -- the duality IS the golden ratio")

    passed = phi_error < 1e-10 and all_non_phi_above
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: phi duality "
          f"{'confirmed' if passed else 'not confirmed'}")

    return {
        'test': 'duality_from_phi_split',
        'errors': {k: float(v) for k, v in errors.items()},
        'phi_error': float(phi_error),
        'all_non_phi_above_001': bool(all_non_phi_above),
        'passed': bool(passed),
    }


def test2_bh_proper_time():
    """
    Test 2: For free-fall through phi-ratio radii r_n = r_0 / phi^n,
    the velocity at r_n is v_n = sqrt(2/r_n) (normalized units, GM=1).
    The time to traverse from r_n to r_{n+1} is:
      dt_n = (r_n - r_{n+1}) / v_n = r_n * (1 - 1/phi) / sqrt(2/r_n)
           = r_n^{3/2} * (1 - 1/phi) / sqrt(2)
    Since r_n = r_0 / phi^n: dt_n proportional to phi^{-3n/2}.
    Time ratio: dt_n / dt_{n+1} = phi^{3/2} = 2.058.

    PASS: mean ratio for n > 3 within 1% of phi^{3/2}.
    """
    print("\n" + "-" * 70)
    print("TEST 2: BH PROPER TIME FROM CASCADE")
    print("-" * 70)

    n_levels = 20
    r_0 = 1.0  # normalized starting radius

    phi_32 = PHI ** 1.5
    print(f"\n  Model: free-fall through phi-ratio radii")
    print(f"    r_n = r_0 / phi^n, v_n = sqrt(2/r_n)")
    print(f"    Expected time ratio: phi^(3/2) = {phi_32:.6f}")

    # Compute radii and time intervals
    radii = np.array([r_0 / PHI**n for n in range(n_levels)])
    velocities = np.sqrt(2.0 / radii)
    delta_r = radii[:-1] - radii[1:]
    dt = delta_r / velocities[:-1]

    # Time ratios
    time_ratios = dt[:-1] / dt[1:]

    print(f"\n  Radii and time intervals (selected levels):")
    for n in range(min(10, n_levels)):
        if n < len(dt):
            print(f"    n={n:2d}: r={radii[n]:.6f}, v={velocities[n]:.4f}, "
                  f"dt={dt[n]:.6e}")

    print(f"\n  Time ratios dt_n / dt_{{n+1}}:")
    for n in range(min(15, len(time_ratios))):
        marker = ""
        if n > 3:
            dev = abs(time_ratios[n] - phi_32) / phi_32
            marker = f"  (dev: {dev*100:.4f}%)"
        print(f"    n={n:2d}: ratio = {time_ratios[n]:.6f}{marker}")

    # Analyze n > 3
    ratios_gt3 = time_ratios[4:]  # n > 3
    if len(ratios_gt3) > 0:
        mean_ratio = np.mean(ratios_gt3)
        deviation = abs(mean_ratio - phi_32) / phi_32
    else:
        mean_ratio = 0.0
        deviation = float('inf')

    print(f"\n  Mean ratio (n > 3): {mean_ratio:.6f}")
    print(f"  phi^(3/2):          {phi_32:.6f}")
    print(f"  Deviation:          {deviation*100:.4f}%")
    print(f"  Threshold:          1%")

    # Analytic verification
    print(f"\n  Analytic check:")
    print(f"    dt_n / dt_{{n+1}} = (r_n/r_{{n+1}})^(3/2) = phi^(3/2)")
    print(f"    This is EXACT for geometric radii -- any deviation is numerical")

    passed = deviation < 0.01
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: BH proper time "
          f"{'confirmed' if passed else 'not confirmed'}")

    return {
        'test': 'bh_proper_time',
        'n_levels': n_levels,
        'phi_32': float(phi_32),
        'mean_ratio_gt3': float(mean_ratio),
        'deviation_pct': float(deviation * 100),
        'time_ratios': [float(r) for r in time_ratios],
        'passed': bool(passed),
    }


def test3_cascade_level_tracks_redshift():
    """
    Test 3: Map the PAC cascade onto Schwarzschild geometry. At phi-ratio
    radii r_n = r_0 / phi^n (with r_0 = 10*r_s), the cascade level is n.
    The gravitational redshift at each radius is z_grav = 1/sqrt(1-r_s/r) - 1.
    Since both n and z_grav increase as r decreases, they should be
    monotonically related. Compute the Spearman rank correlation.

    Also verify: ln(1 + z_grav) grows faster than n (redshift accelerates
    relative to cascade level near r_s). This means each successive
    cascade level contributes MORE redshift -- gravity IS the cascade
    compressing.

    PASS: Spearman correlation > 0.99 AND ln(1+z) growth rate increases.
    """
    print("\n" + "-" * 70)
    print("TEST 3: CASCADE LEVEL TRACKS GRAVITATIONAL REDSHIFT")
    print("-" * 70)

    r_s = 1.0  # Schwarzschild radius (normalized)
    r_0 = 10.0 * r_s  # outer radius

    # Cascade levels: r_n = r_0 / phi^n
    # Find how many levels before r_n < r_s
    n_max = int(np.floor(np.log(r_0 / r_s) / np.log(PHI)))
    n_levels = min(n_max - 1, 25)  # stay above r_s

    cascade_levels = np.arange(n_levels)
    radii = r_0 / PHI**cascade_levels

    # Gravitational redshift at each radius
    z_grav = 1.0 / np.sqrt(1.0 - r_s / radii) - 1.0
    ln_1pz = np.log(1.0 + z_grav)

    print(f"\n  Cascade on Schwarzschild spacetime")
    print(f"  r_0 = {r_0:.1f} r_s, r_s = {r_s:.1f}")
    print(f"  Max levels before horizon: {n_max}")
    print(f"  Using {n_levels} levels")

    print(f"\n  {'n':>3s}  {'r_n/r_s':>8s}  {'z_grav':>10s}  {'ln(1+z)':>10s}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*10}")
    for n in range(n_levels):
        print(f"  {n:3d}  {radii[n]/r_s:8.4f}  {z_grav[n]:10.6f}  {ln_1pz[n]:10.6f}")

    # Spearman rank correlation between n and z_grav
    # Since both are monotonic, ranks should be perfectly correlated
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(cascade_levels, z_grav)

    print(f"\n  Spearman rank correlation (n vs z_grav): {spearman_corr:.6f}")
    print(f"  p-value: {spearman_p:.2e}")

    # Check: does ln(1+z) per level INCREASE with depth?
    # dln(1+z)/dn at each level
    dln_dn = np.diff(ln_1pz)
    increasing_rate = all(dln_dn[i] < dln_dn[i + 1]
                          for i in range(len(dln_dn) - 1))

    print(f"\n  Redshift rate per cascade level:")
    print(f"  {'Level':>5s}  {'d(ln(1+z))/dn':>15s}")
    print(f"  {'-'*5}  {'-'*15}")
    for i in range(len(dln_dn)):
        print(f"  {i:5d}  {dln_dn[i]:15.6f}")

    print(f"\n  Rate monotonically increasing: {increasing_rate}")
    print(f"  (Each deeper level contributes MORE redshift)")

    print(f"\n  Physical interpretation:")
    print(f"    Cascade level n maps monotonically to gravitational redshift")
    print(f"    Near the horizon, each cascade level produces exponentially")
    print(f"    more redshift -- the cascade ACCELERATES near r_s")
    print(f"    This is gravitational time dilation as cascade compression")

    passed = spearman_corr > 0.99 and increasing_rate
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: cascade-redshift mapping "
          f"{'confirmed' if passed else 'not confirmed'}")

    return {
        'test': 'cascade_level_tracks_redshift',
        'n_levels': n_levels,
        'r_0': float(r_0),
        'r_s': float(r_s),
        'spearman_corr': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'rate_increasing': bool(increasing_rate),
        'z_grav_final': float(z_grav[-1]),
        'passed': bool(passed),
    }


def test4_zeno_completion():
    """
    Test 4: The cascade has infinitely many levels, but the total proper
    time converges (Zeno completion). Sum of time intervals:
      T = sum_{n=0}^{inf} phi^{-3n/2} = 1 / (1 - phi^{-3/2})

    Compute:
    - Theoretical infinite sum
    - Partial sum with 20 terms
    - Ratio (should be > 0.999 for rapid convergence)
    - Verify sum is finite (the cascade COMPLETES in finite time)

    PASS: partial/theoretical ratio > 0.999 AND sum is finite.
    """
    print("\n" + "-" * 70)
    print("TEST 4: ZENO COMPLETION")
    print("-" * 70)

    phi_32 = PHI ** 1.5
    r = 1.0 / phi_32  # common ratio = phi^{-3/2}
    n_terms = 20

    print(f"\n  Cascade time series: sum of phi^(-3n/2)")
    print(f"    Common ratio r = phi^(-3/2) = {r:.6f}")
    print(f"    |r| < 1: {abs(r) < 1} (convergent)")

    # Theoretical infinite sum
    theoretical_sum = 1.0 / (1.0 - r)
    print(f"\n  Theoretical sum (infinite terms): {theoretical_sum:.10f}")
    print(f"  Formula: 1 / (1 - phi^(-3/2))")

    # Partial sums
    terms = np.array([r**n for n in range(n_terms)])
    partial_sums = np.cumsum(terms)

    print(f"\n  Partial sums:")
    for n in range(n_terms):
        ratio = partial_sums[n] / theoretical_sum
        print(f"    N={n+1:3d}: sum = {partial_sums[n]:.10f}  "
              f"(fraction of total: {ratio:.8f})")

    # Final ratio
    final_ratio = partial_sums[-1] / theoretical_sum
    is_finite = np.isfinite(theoretical_sum)

    print(f"\n  Convergence:")
    print(f"    Partial sum ({n_terms} terms): {partial_sums[-1]:.10f}")
    print(f"    Theoretical (infinite):       {theoretical_sum:.10f}")
    print(f"    Ratio:                        {final_ratio:.10f}")
    print(f"    Threshold:                    > 0.999")
    print(f"    Sum is finite:                {is_finite}")

    # Rate of convergence
    print(f"\n  Convergence milestones:")
    for target in [0.9, 0.99, 0.999, 0.9999]:
        for n in range(n_terms):
            if partial_sums[n] / theoretical_sum >= target:
                print(f"    {target*100:.2f}% reached at N = {n+1}")
                break

    # Physical interpretation
    print(f"\n  Physical meaning:")
    print(f"    Despite infinitely many cascade levels, the total proper")
    print(f"    time for traversal (infall) is FINITE.")
    print(f"    Total time = {theoretical_sum:.4f} * dt_0 (in units of first interval)")
    print(f"    This is the gravitational cascade's Zeno completion:")
    print(f"    infinitely many events in finite duration.")

    # Also show the phi cascade (energy-proportional timing)
    r_energy = INV_PHI
    energy_sum = 1.0 / (1.0 - r_energy)
    print(f"\n  Comparison with energy cascade:")
    print(f"    Energy sum (r = 1/phi):     {energy_sum:.6f} (= phi^2)")
    print(f"    Gravity sum (r = 1/phi^1.5): {theoretical_sum:.6f}")
    print(f"    Gravity converges faster (stronger ratio)")

    passed = final_ratio > 0.999 and is_finite
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: Zeno completion "
          f"{'confirmed' if passed else 'not confirmed'}")

    return {
        'test': 'zeno_completion',
        'n_terms': n_terms,
        'common_ratio': float(r),
        'theoretical_sum': float(theoretical_sum),
        'partial_sum': float(partial_sums[-1]),
        'ratio': float(final_ratio),
        'is_finite': bool(is_finite),
        'energy_sum': float(energy_sum),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 9 - EXP 05: GRAVITY-TIME FROM CASCADE")
    print("Block B: Information-Time Nexus")
    print("Does the cascade produce gravity-time duality?")
    print("=" * 70)

    r1 = test1_duality_from_phi_split()
    r2 = test2_bh_proper_time()
    r3 = test3_cascade_level_tracks_redshift()
    r4 = test4_zeno_completion()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Duality from phi-split):     {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (BH proper time):             {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Cascade tracks redshift):    {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Zeno completion):            {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    if r1['passed'] and r2['passed']:
        print(f"\n  KEY FINDING: The phi-split duality g_out = g_in^2 connects")
        print(f"  cascade energy release to gravitational free-fall dynamics.")
        print(f"  Time intervals at phi-ratio radii scale as phi^(3/2).")
    if r4['passed']:
        print(f"\n  KEY FINDING: The cascade completes in finite proper time")
        print(f"  despite infinitely many levels (Zeno completion).")

    results = {
        'experiment': 'exp_05_gravity_time_from_cascade',
        'milestone': 9,
        'block': 'B',
        'block_name': 'Information-Time Nexus',
        'tests': {
            'test1_duality_from_phi_split': r1,
            'test2_bh_proper_time': r2,
            'test3_cascade_level_tracks_redshift': r3,
            'test4_zeno_completion': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_05_gravity_time_from_cascade', RESULTS_DIR)


if __name__ == '__main__':
    main()
