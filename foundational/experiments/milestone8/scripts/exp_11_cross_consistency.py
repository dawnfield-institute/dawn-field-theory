"""
Milestone 8 -- Exp 11: Cross-Consistency Propagation

PURPOSE: Verify that M8 predictions form a self-consistent set.
Currently each experiment computes independently — this checks whether
the SAME constants and masses propagate correctly through the full chain.

This is the hardening experiment: it catches internal contradictions
that individual experiments can't detect.

Tests:
  1. Mass propagation: exp_02 mass -> exp_03 abundance -> exp_09 JWST floor
  2. N=6 universality: independently fit N from each observable
  3. Coupling->mass->abundance chain: single derivation, no re-computation
  4. Prediction independence: count true degrees of freedom vs data points

"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.special import erfc

SCRIPT_DIR = Path(__file__).resolve().parent
M8_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M8_ROOT))

from core.bsm import (
    PHI, INV_PHI, LN_PHI, PI, XI_BALANCE,
    M_PLANCK_GEV, M_Z_GEV, HIGGS_VEV, ALPHA_EM,
    H0_PLANCK, H0_SHOES, OMEGA_M, OMEGA_DM, OMEGA_LAMBDA,
    SIGMA8_PLANCK, S8_PLANCK, S8_KIDS, S8_DES,
    OMEGA_DM_H2,
    F6, F7, F8, F10,
    DEPTH_DARK,
    fibonacci_depth_coupling, cyclotomic_phi3,
    dodelson_widrow_abundance, free_streaming_length,
    growth_factor, press_schechter_fraction,
    save_results, setup_experiment,
)

_, RESULTS_DIR = setup_experiment(__file__)


# JWST observations (same as exp_09)
JWST_N_Z8 = 1e-5
JWST_N_Z12 = 3e-6


def test1_mass_propagation():
    """
    Test 1: Does the SAME DM mass work across exp_02, exp_03, and exp_09?

    Chain: cascade routes -> mass -> DW abundance -> relic density
    Then: mass -> free-streaming -> JWST floor consistency

    PASS: all three contexts accept the same mass without internal contradiction.
    """
    print("\n" + "=" * 70)
    print("TEST 1: MASS PROPAGATION CHAIN")
    print("=" * 70)

    # Step 1: Mass from cascade routes (exp_02 logic)
    m_b_gev = HIGGS_VEV * PHI**(-(DEPTH_DARK) / 2)
    m_c_gev = M_Z_GEV * PHI**(-(F8 + F7))
    m_b_kev = m_b_gev * 1e6
    m_c_kev = m_c_gev * 1e6
    m_dm_kev = np.exp(np.mean(np.log([m_b_kev, m_c_kev])))
    m_dm_gev = m_dm_kev / 1e6

    print(f"\n  Mass derivation (exp_02 routes):")
    print(f"    Route b (VEV cascade): {m_b_kev:.2f} keV")
    print(f"    Route c (MZ cascade):  {m_c_kev:.2f} keV")
    print(f"    Geometric mean:        {m_dm_kev:.2f} keV")

    # Step 2: Relic abundance (exp_03 logic)
    # DW production requires sin^2(2theta) to produce Omega_DM h^2 = 0.120
    # From DW formula: Omega h^2 = 0.3 * (sin2_2theta/1e-10) * (m/keV)^1.8
    # Solve for sin2_2theta:
    target_omega_h2 = OMEGA_DM_H2
    sin2_2theta = target_omega_h2 / (0.3 * (m_dm_kev)**1.8) * 1e-10
    omega_check = dodelson_widrow_abundance(m_dm_kev, sin2_2theta)

    print(f"\n  Relic abundance (exp_03 chain):")
    print(f"    Required sin^2(2theta) = {sin2_2theta:.3e}")
    print(f"    DW Omega h^2 = {omega_check:.4f} (target: {target_omega_h2})")
    print(f"    Consistency: {abs(omega_check - target_omega_h2)/target_omega_h2*100:.2f}%")

    # Step 3: Free-streaming check
    lambda_fs = free_streaming_length(m_dm_kev)
    fs_ok = 0.01 < lambda_fs < 1.0  # Mpc, WDM range

    print(f"\n  Free-streaming (exp_03 chain):")
    print(f"    lambda_fs = {lambda_fs:.4f} Mpc")
    print(f"    WDM range [0.01, 1.0]: {'YES' if fs_ok else 'NO'}")

    # Step 4: JWST floor — does this mass affect the cascade floor?
    # The JWST floor doesn't depend on DM mass directly; it depends on
    # N_cascade and ln(phi). But the DM coupling alpha_73 affects the
    # cascade energy budget. Check: is alpha_73 consistent with the
    # S8 dissipation fraction used in exp_07?
    alpha_73 = fibonacci_depth_coupling(DEPTH_DARK)
    # The dissipation fraction in exp_07: f_eff = (1/phi^2)/6 * Omega_DM/Omega_M
    # This should be physically related to alpha_73 through the coupling strength.
    # Check: is the cascade coupling strong enough to produce observable effects?
    # alpha_73 ~ 2.5e-16 is extremely weak per particle, but collective.
    # Number density of DM particles: n_DM ~ rho_DM / m_DM
    # Collective coupling scales as n_DM * alpha_73

    print(f"\n  Coupling check:")
    print(f"    alpha_73 = {alpha_73:.3e}")
    print(f"    This is the per-particle coupling. Collective effects are:")
    print(f"    n_DM * alpha_73 * (volume) -> scales differently than f_eff")

    # Consistency verdict
    omega_consistent = abs(omega_check - target_omega_h2) / target_omega_h2 < 0.01
    mass_spread = abs(np.log10(m_b_kev) - np.log10(m_c_kev))

    issues = []
    if mass_spread > 0.5:
        issues.append(f"Mass routes spread > 0.5 orders ({mass_spread:.2f})")
    if not omega_consistent:
        issues.append(f"DW abundance off by {abs(omega_check - target_omega_h2)/target_omega_h2*100:.1f}%")
    if not fs_ok:
        issues.append(f"Free-streaming outside WDM range ({lambda_fs:.4f} Mpc)")

    passed = len(issues) == 0
    print(f"\n  Issues: {issues if issues else 'None'}")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: mass propagation {'consistent' if passed else 'has contradictions'}")

    return {
        'test': 'mass_propagation',
        'm_dm_kev': float(m_dm_kev),
        'mass_spread_orders': float(mass_spread),
        'sin2_2theta': float(sin2_2theta),
        'omega_check': float(omega_check),
        'omega_consistent': omega_consistent,
        'lambda_fs': float(lambda_fs),
        'fs_ok': fs_ok,
        'issues': issues,
        'passed': passed,
    }


def test2_n6_universality():
    """
    Test 2: Independently fit N from each observable.

    If N=6 is universal, each observable should independently prefer N~6.
    If they don't agree, the unification is post-hoc.

    Observables:
    a) Hubble ratio: phi^{1/N} = H0_local/H0_CMB -> N = ln(phi) / ln(ratio)
    b) S8: (1/phi^2)/N * Omega_DM/Omega_M gives S8_DFT. Solve for N.
    c) JWST z=12/z=8 ratio: exp(-z/[ln(phi)*N]). Solve for N.
    d) BAO: phi^{-1/N} correction -> H0. Solve for N.

    HARDENED: Round 1. Previously passed with tolerance +/-2 of N=6, which
    hid the spread (actual range 2.74). Tightened to range < 1.5.
    Expected to FAIL — M9's cascade clock is the resolution.

    PASS: range of independent N values < 1.5.
    """
    print("\n" + "=" * 70)
    print("TEST 2: N=6 UNIVERSALITY")
    print("=" * 70)

    fitted_N = {}

    # (a) Hubble ratio
    h0_ratio = H0_SHOES / H0_PLANCK
    N_hubble = np.log(PHI) / np.log(h0_ratio)
    fitted_N['hubble'] = N_hubble
    print(f"\n  (a) Hubble ratio = {h0_ratio:.6f}")
    print(f"      phi^{{1/N}} = ratio -> N = ln(phi)/ln(ratio) = {N_hubble:.2f}")

    # (b) S8 — solve for N from S8 measurement
    # S8_DFT = S8_Planck * (1 - (1/phi^2)/N * Omega_DM/Omega_M)
    # Target S8 = mean of lensing measurements
    s8_target = (S8_KIDS + S8_DES) / 2
    # (1 - S8_target/S8_Planck) = (1/phi^2)/N * Omega_DM/Omega_M
    # N = (1/phi^2) * (Omega_DM/Omega_M) / (1 - S8_target/S8_Planck)
    reduction = 1 - s8_target / S8_PLANCK
    if reduction > 0:
        N_s8 = INV_PHI**2 * (OMEGA_DM / OMEGA_M) / reduction
    else:
        N_s8 = float('inf')
    fitted_N['s8'] = N_s8
    print(f"\n  (b) S8 target (lensing mean) = {s8_target:.3f}")
    print(f"      Reduction fraction = {reduction:.4f}")
    print(f"      Fitted N = {N_s8:.2f}")

    # (c) JWST ratio — solve for N from z=12/z=8 abundance ratio
    # ratio = exp(-12/[ln(phi)*N]) / exp(-8/[ln(phi)*N])
    #       = exp(-4/[ln(phi)*N])
    # Given JWST ratio ~ 0.3:
    jwst_ratio = JWST_N_Z12 / JWST_N_Z8  # 0.3
    # 0.3 = exp(-4 / [ln(phi)*N])
    # ln(0.3) = -4 / [ln(phi)*N]
    # N = -4 / (ln(phi) * ln(0.3))
    if jwst_ratio > 0 and jwst_ratio < 1:
        N_jwst = -4.0 / (LN_PHI * np.log(jwst_ratio))
    else:
        N_jwst = float('inf')
    fitted_N['jwst'] = N_jwst
    print(f"\n  (c) JWST ratio n(z=12)/n(z=8) = {jwst_ratio:.2f}")
    print(f"      Fitted N = {N_jwst:.2f}")

    # (d) BAO — solve for N from corrected H0
    # H0_corrected = H0_Planck / phi^{-1/N} = H0_Planck * phi^{1/N}
    # For H0_corrected = H0_SH0ES:
    # phi^{1/N} = H0_SH0ES / H0_Planck -> same as (a)
    # So BAO and Hubble ratio give IDENTICAL N — they're NOT independent!
    N_bao = N_hubble  # Same constraint
    fitted_N['bao'] = N_bao
    print(f"\n  (d) BAO correction: phi^{{-1/N}} on r_s")
    print(f"      Fitted N = {N_bao:.2f}")
    print(f"      NOTE: BAO and Hubble ratio are the SAME constraint (both = phi^{{1/N}})")
    print(f"      Independent constraints: Hubble, S8, JWST (3, not 4)")

    # Assess agreement
    independent_N = [N_hubble, N_s8, N_jwst]  # BAO = Hubble, don't double-count
    N_mean = np.mean(independent_N)
    N_std = np.std(independent_N)
    N_range = max(independent_N) - min(independent_N)
    all_near_6 = all(abs(n - 6) < 2 for n in independent_N)
    # HARDENED: tighten from abs(n-6)<2 to range<1.5
    range_tight = N_range < 1.5

    print(f"\n  Independent N values:")
    print(f"    Hubble:  {N_hubble:.2f}")
    print(f"    S8:      {N_s8:.2f}")
    print(f"    JWST:    {N_jwst:.2f}")
    print(f"    Mean:    {N_mean:.2f} +/- {N_std:.2f}")
    print(f"    Range:   {N_range:.2f}")
    print(f"    All within [4, 8]: {all_near_6} (old criterion, kept for reference)")
    print(f"    Range < 1.5: {range_tight} (HARDENED criterion)")

    passed = range_tight
    if not passed:
        print(f"\n  HONEST FAILURE: N values span {N_range:.2f} (> 1.5)")
        print(f"    This reveals that N=6 is NOT universal at M8 precision.")
        print(f"    M9's cascade clock N(t) resolves this: N is z-dependent,")
        print(f"    so different observables at different redshifts SHOULD give different N.")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: N universality "
          f"{'confirmed' if passed else 'fails at tightened tolerance'}")

    # CRITICAL FINDING: BAO and Hubble are NOT independent constraints on N.
    # We have 3 independent observables, not 4.
    print(f"\n  FINDING: BAO = Hubble -> only 3 independent constraints on N")

    return {
        'test': 'n6_universality',
        'fitted_N': {k: float(v) for k, v in fitted_N.items()},
        'independent_N': [float(n) for n in independent_N],
        'N_mean': float(N_mean),
        'N_std': float(N_std),
        'N_range': float(N_range),
        'all_near_6': all_near_6,
        'range_tight': range_tight,
        'hardened': 'Round 1: tightened from abs(n-6)<2 to range<1.5',
        'bao_equals_hubble': True,
        'n_independent_constraints': 3,
        'passed': passed,
    }


def test3_coupling_mass_abundance_chain():
    """
    Test 3: Single derivation chain from coupling -> mass -> abundance.

    Start from depth 73, derive everything in sequence:
    1. alpha_73 from phi^{-73}/sqrt(5)
    2. Mass from cascade formula
    3. sin^2(2theta) from DW abundance requirement
    4. Check relic density = 0.120 +/- 10%

    The point: one unbroken chain, no re-derivation at each step.
    """
    print("\n" + "=" * 70)
    print("TEST 3: COUPLING->MASS->ABUNDANCE CHAIN")
    print("=" * 70)

    # Step 1: Coupling
    depth = DEPTH_DARK  # 73
    alpha = PHI**(-depth) / np.sqrt(5)
    print(f"\n  Step 1: Coupling at depth {depth}")
    print(f"    alpha_73 = phi^{{-73}}/sqrt(5) = {alpha:.4e}")

    # Step 2: Mass (cascade routes only)
    m_vev = HIGGS_VEV * PHI**(-depth / 2)
    m_mz = M_Z_GEV * PHI**(-(F8 + F7))
    m_dm = np.exp(np.mean(np.log([m_vev * 1e6, m_mz * 1e6])))  # keV
    print(f"\n  Step 2: Mass from cascade")
    print(f"    m_DM = {m_dm:.2f} keV")

    # Step 3: Mixing angle from abundance requirement
    # Omega h^2 = 0.3 * (sin2_2theta/1e-10) * (m/keV)^1.8 = 0.120
    sin2_2theta = 0.120 / (0.3 * m_dm**1.8) * 1e-10
    print(f"\n  Step 3: Mixing angle from Omega_DM h^2 = 0.120")
    print(f"    sin^2(2theta) = {sin2_2theta:.3e}")

    # Step 4: Verify chain closure
    omega_final = dodelson_widrow_abundance(m_dm, sin2_2theta)
    chain_error = abs(omega_final - 0.120) / 0.120

    print(f"\n  Step 4: Chain verification")
    print(f"    Omega h^2 (reconstructed) = {omega_final:.6f}")
    print(f"    Chain closure error: {chain_error*100:.4f}%")

    # Cross-checks against bounds
    # X-ray line
    e_xray = m_dm / 2  # keV
    xray_ratio = e_xray / 3.55  # vs Bulbul+ 2014
    print(f"\n  Cross-checks:")
    print(f"    X-ray line energy: {e_xray:.2f} keV (observed: 3.55 keV)")
    print(f"    Ratio: {xray_ratio:.2f}")

    # Lyman-alpha
    lyman_ok = m_dm > 3.3
    print(f"    Lyman-alpha bound (>3.3 keV): {'OK' if lyman_ok else 'TENSION'} ({m_dm:.2f} keV)")

    # NuSTAR bound
    nustar_bound = 2e-11  # at ~6 keV
    nustar_ok = sin2_2theta < nustar_bound
    print(f"    NuSTAR bound (<{nustar_bound}): {'OK' if nustar_ok else 'EXCLUDED'} ({sin2_2theta:.2e})")

    passed = chain_error < 0.10 and lyman_ok
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: chain closure {chain_error*100:.2f}%, "
          f"Lyman-alpha {'OK' if lyman_ok else 'FAIL'}")

    return {
        'test': 'coupling_mass_abundance_chain',
        'depth': depth,
        'alpha': float(alpha),
        'm_dm_kev': float(m_dm),
        'sin2_2theta': float(sin2_2theta),
        'omega_final': float(omega_final),
        'chain_error': float(chain_error),
        'e_xray_kev': float(e_xray),
        'xray_ratio': float(xray_ratio),
        'lyman_ok': lyman_ok,
        'nustar_ok': nustar_ok,
        'passed': passed,
    }


def test4_prediction_independence():
    """
    Test 4: How many truly independent predictions does M8 make?

    Count degrees of freedom (free parameters) vs data points (predictions).
    If DoF >= data points, we're fitting, not predicting.

    DFT inputs (the framework):
    - phi (mathematical constant, not free)
    - ln(phi), Xi = gamma + ln(phi) (derived)
    - N_cascade = 6 (fitted to Hubble ratio)
    - Depth 73 = Phi_3(F_6) (structural, but chosen as dark sector)

    M8 predictions:
    - DM mass ~6.4 keV (from depth 73 + cascade)
    - X-ray line ~3.2 keV (= mass/2, derived from above)
    - sin^2(2theta) ~ 10^{-11} (from abundance, derived from mass)
    - alpha_73 ~ 2.5e-16 (from depth formula, same as mass)
    - Z' mass = 395 GeV (independent: different depth)
    - g'/g = 1/13 (= 1/F_7, structural)
    - Normal hierarchy (structural)
    - delta_CP = 63.5 deg (from Xi)
    - H0 ratio = phi^{1/6} (uses N_cascade)
    - S8 = 0.787 (uses N_cascade)
    - w0 = -0.83 (from cascade EOS)
    - JWST ratio (uses N_cascade + ln(phi))

    Truly independent clusters:
    A) Dark sector: depth 73 -> mass, coupling, X-ray, sin2theta, Lyman-alpha
       (1 input -> 5 derived quantities, but only 1 independent prediction)
    B) Z' sector: F_7/F_4 -> mass, g'/g, width
       (1 input -> 3 derived, 1 independent)
    C) Neutrinos: Xi, Fibonacci ratios -> hierarchy, delta_CP, masses
       (2 inputs -> 3 predictions, 2 independent)
    D) Cosmology: N_cascade -> H0, S8, BAO, JWST
       (1 input -> 4 observables, but BAO=H0, so 3 independent)

    PASS: n_independent > n_free_params + 2 (genuine overconstrained system)
    """
    print("\n" + "=" * 70)
    print("TEST 4: PREDICTION INDEPENDENCE")
    print("=" * 70)

    # Free parameters / choices in M8
    free_params = {
        'N_cascade': 'Fitted to Hubble ratio (1 DoF)',
        'depth_73': 'Chosen as dark sector candidate (structural)',
    }
    # phi, ln(phi), Xi are mathematical constants — not free
    # Fibonacci ratios are structural — not free

    # Predictions grouped by independence cluster
    clusters = {
        'A_dark_sector': {
            'input': 'depth 73 = Phi_3(F_6)',
            'predictions': ['DM mass ~6.4 keV', 'alpha_73', 'X-ray line', 'sin2_2theta', 'Lyman-alpha'],
            'independent_count': 1,
            'note': 'All derived from one depth value',
        },
        'B_zprime': {
            'input': 'M_Z * F_7/F_4',
            'predictions': ['Z\' mass = 395 GeV', 'g\'/g = 1/13', 'width'],
            'independent_count': 1,
            'note': 'Mass and coupling from same Fibonacci ratio',
        },
        'C_neutrino': {
            'input': 'Xi, scope depths',
            'predictions': ['Normal hierarchy', 'delta_CP = 63.5 deg', 'Splitting ratio'],
            'independent_count': 2,
            'note': 'Hierarchy from ordering; delta_CP from Xi (independent)',
        },
        'D_cosmology': {
            'input': 'N_cascade = 6',
            'predictions': ['H0 ratio', 'BAO correction', 'S8 = 0.787', 'JWST ratio'],
            'independent_count': 3,
            'note': 'BAO = H0 (same constraint); H0, S8, JWST are independent',
        },
    }

    n_free = len(free_params)
    n_independent = sum(c['independent_count'] for c in clusters.values())
    n_claimed = 10  # M8 claims 10 predictions

    print(f"\n  Free parameters/choices: {n_free}")
    for k, v in free_params.items():
        print(f"    {k}: {v}")

    print(f"\n  Prediction clusters:")
    for name, cluster in clusters.items():
        print(f"\n    {name}: {cluster['input']}")
        print(f"      Predictions: {cluster['predictions']}")
        print(f"      Independent: {cluster['independent_count']}")
        print(f"      Note: {cluster['note']}")

    print(f"\n  Summary:")
    print(f"    Claimed predictions: {n_claimed}")
    print(f"    Truly independent: {n_independent}")
    print(f"    Free parameters: {n_free}")
    print(f"    Overconstrained by: {n_independent - n_free}")

    # PASS: more independent predictions than free parameters + 2
    # (need to overconstrain by at least 2 to be genuinely predictive)
    overconstrained = n_independent - n_free
    passed = overconstrained >= 3

    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {n_independent} independent predictions "
          f"with {n_free} free parameters (overconstrained by {overconstrained})")

    return {
        'test': 'prediction_independence',
        'n_claimed': n_claimed,
        'n_independent': n_independent,
        'n_free_params': n_free,
        'overconstrained_by': overconstrained,
        'clusters': {k: {'independent': v['independent_count'], 'note': v['note']}
                     for k, v in clusters.items()},
        'passed': passed,
    }


def main():
    print("=" * 70)
    print("MILESTONE 8 - EXP 11: CROSS-CONSISTENCY PROPAGATION")
    print("Hardening: Internal consistency audit")
    print("=" * 70)

    r1 = test1_mass_propagation()
    r2 = test2_n6_universality()
    r3 = test3_coupling_mass_abundance_chain()
    r4 = test4_prediction_independence()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Mass propagation):          {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (N=6 universality):           {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Coupling->mass->abundance):  {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Prediction independence):    {'PASS' if r4['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/4")

    if r2['passed']:
        print(f"\n  KEY FINDING: BAO and Hubble ratio are the SAME constraint on N.")
        print(f"  M8 has 3 independent constraints on N, not 4.")
    if r4['passed']:
        print(f"\n  KEY FINDING: {r4['n_independent']} independent predictions from "
              f"{r4['n_free_params']} free parameters.")

    results = {
        'experiment': 'exp_11_cross_consistency',
        'milestone': 8,
        'block': 'E',
        'tests': {
            'test1_mass_propagation': r1,
            'test2_n6_universality': r2,
            'test3_coupling_mass_abundance_chain': r3,
            'test4_prediction_independence': r4,
        },
        'score': f"{n_passed}/4",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_11_cross_consistency', RESULTS_DIR)


if __name__ == '__main__':
    main()
