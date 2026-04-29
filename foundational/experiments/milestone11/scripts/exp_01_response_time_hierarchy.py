"""
exp_01 — Gravitational Response-Time Hierarchy

Milestone 11, Block A (Response-Time Foundations)

Hypothesis: The response time of each fundamental force is determined by its
cascade depth, and the ordering reproduces the known force hierarchy. The
crossover where perturbation rate exceeds response time predicts where each
force's "law" breaks down. For gravity at depth 183, this defines the quantum
gravity regime.

Tests:
  T1: Response-time ordering reproduces force hierarchy (Planck < EM < gravity < Hubble)
  T2: Response-time RATIOS match Fibonacci depth ratios
  T3: LawNegotiator violation fraction monotonic above crossover energy
  T4: Known QG phenomena cluster at gravitational crossover

From M10 Section 6: laws are not rules but maintained negotiations with
characteristic response times. Forces with deeper cascade depth have longer
response times (weaker coupling = slower negotiation).
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, PI, LN2,
    ALPHA_EM, G_NEWTON, HBAR, C_LIGHT,
    T_PLANCK_S, T_EM_S, T_GRAVITY_S, T_HUBBLE_S,
    E_PLANCK_GEV, L_PLANCK_M, DEPTH_EM, DEPTH_GRAVITY,
    RESPONSE_TIMES,
    force_response_hierarchy, crossover_energy, cascade_depth_response_time,
    LawNegotiator,
    save_results, setup_experiment, PredictionRegistry,
    fib,
)

# ============================================================
# Setup
# ============================================================
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_ordering():
    """
    T1: Response-time ordering reproduces force hierarchy.

    Expected: strong < weak < EM < gravity, matching known force ordering
    from strongest to weakest. Response time = negotiation time = 1/coupling.
    """
    hierarchy = force_response_hierarchy()

    # Extract (name, depth, tau) and sort by tau
    forces = [(k, v['depth'], v['tau_seconds']) for k, v in hierarchy.items()]
    by_tau = sorted(forces, key=lambda x: x[2])

    # Expected ordering by depth (shallow = fast response = strong force)
    expected_order = ['strong', 'weak', 'em', 'gravity']
    actual_order = [f[0] for f in by_tau]

    ordering_correct = actual_order == expected_order

    # Also check that Planck time < all force response times < Hubble time
    all_taus = [v['tau_seconds'] for v in hierarchy.values()]
    planck_bounded = all(t > T_PLANCK_S for t in all_taus)
    hubble_bounded = all(t < T_HUBBLE_S for t in all_taus)

    # Spearman rank correlation between depth and log(tau)
    depths = [v['depth'] for v in hierarchy.values()]
    log_taus = [np.log10(v['tau_seconds']) for v in hierarchy.values()]
    rho, p_val = spearmanr(depths, log_taus)

    # Quantitative coupling test: depth-183 vs known alpha_grav(proton)
    # This is genuinely non-tautological: phi^(-183) is a DFT prediction,
    # alpha_grav = G*m_p^2/(hbar*c) is independently measured.
    M_PROTON_KG = 1.67262192e-27  # kg (PDG 2022)
    alpha_grav_proton = G_NEWTON * M_PROTON_KG**2 / (HBAR * C_LIGHT)
    alpha_grav_dft = PHI ** (-DEPTH_GRAVITY)  # phi^(-183)

    log_known = np.log10(alpha_grav_proton)
    log_dft = np.log10(alpha_grav_dft)
    log_agreement = abs(log_known - log_dft) / abs(log_known)
    coupling_match = log_agreement < 0.05  # Within 5% in log space

    result = {
        'test': 'T1_ordering',
        'expected_order': expected_order,
        'actual_order': actual_order,
        'ordering_correct': ordering_correct,
        'planck_bounded': planck_bounded,
        'hubble_bounded': hubble_bounded,
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'alpha_grav_proton': float(alpha_grav_proton),
        'alpha_grav_dft': float(alpha_grav_dft),
        'log10_known': float(log_known),
        'log10_dft': float(log_dft),
        'log_agreement_pct': float(log_agreement * 100),
        'coupling_match': coupling_match,
        'hierarchy': {k: {'depth': v['depth'], 'tau_s': v['tau_seconds'],
                          'log10_tau': v['log10_tau']}
                      for k, v in hierarchy.items()},
        'note_ordering': 'Ordering is structural (deeper depth = longer tau by construction). '
                         'Coupling match is the genuine test.',
        'PASS': ordering_correct and coupling_match,
    }
    return result


def test_T2_coupling_ratios():
    """
    T2: DFT depth-difference predictions match known coupling ratios.

    DFT: alpha_grav / alpha_EM = phi^(-183) / phi^(-13) = phi^(-170)
    Known: (G*m_p^2 / hbar*c) / alpha_EM

    Previous version tested tau2/tau1 = phi^(d2-d1) vs phi^(d2-d1) — same formula
    both sides (tautological). This version compares DFT to independent measurements.
    """
    M_PROTON_KG = 1.67262192e-27  # kg
    alpha_grav_proton = G_NEWTON * M_PROTON_KG**2 / (HBAR * C_LIGHT)

    # Test 1: Gravity/EM coupling ratio
    depth_diff_ge = DEPTH_GRAVITY - DEPTH_EM  # 183 - 13 = 170
    ratio_dft_ge = PHI ** (-depth_diff_ge)
    ratio_known_ge = alpha_grav_proton / ALPHA_EM

    log_dft_ge = np.log10(ratio_dft_ge)
    log_known_ge = np.log10(ratio_known_ge)
    log_error_ge = abs(log_dft_ge - log_known_ge) / abs(log_known_ge)

    # Test 2: Gravity/Weak coupling ratio
    # alpha_weak ~ alpha_EM / sin^2(theta_W), using DFT's sin^2(theta_W) = 3/13
    sin2_theta_w = 3.0 / 13.0
    alpha_weak = ALPHA_EM / sin2_theta_w
    depth_diff_gw = DEPTH_GRAVITY - 7  # 183 - 7 = 176
    ratio_dft_gw = PHI ** (-depth_diff_gw)
    ratio_known_gw = alpha_grav_proton / alpha_weak

    log_dft_gw = np.log10(ratio_dft_gw)
    log_known_gw = np.log10(ratio_known_gw)
    log_error_gw = abs(log_dft_gw - log_known_gw) / abs(log_known_gw)

    ratio_tests = [
        {
            'pair': 'gravity / EM',
            'depth_diff': depth_diff_ge,
            'log10_dft': float(log_dft_ge),
            'log10_known': float(log_known_ge),
            'log_relative_error': float(log_error_ge),
            'match': log_error_ge < 0.05,
        },
        {
            'pair': 'gravity / weak',
            'depth_diff': depth_diff_gw,
            'log10_dft': float(log_dft_gw),
            'log10_known': float(log_known_gw),
            'log_relative_error': float(log_error_gw),
            'match': log_error_gw < 0.10,
        },
    ]

    primary_match = ratio_tests[0]['match']

    result = {
        'test': 'T2_coupling_ratios',
        'ratio_tests': ratio_tests,
        'primary_match': primary_match,
        'note': 'Replaced tautological ratio test (same formula both sides) '
                'with comparison to independently measured coupling constants.',
        'PASS': primary_match,
    }
    return result


def test_T3_violation_monotonic():
    """
    T3: LawNegotiator violation fraction monotonic above crossover.

    Run LawNegotiator at increasing perturbation rates. Violation should
    increase monotonically once rate exceeds 1/tau.
    """
    # Use gravitational response time as the negotiation timescale
    tau_grav = 1.0  # Normalized

    perturbation_rates = np.logspace(-2, 2, 20)  # 0.01 to 100 × 1/tau

    violations = []
    for rate in perturbation_rates:
        negotiator = LawNegotiator(
            n_participants=50,
            response_time=tau_grav,
            conserved_total=100.0,
        )
        result = negotiator.perturb_and_negotiate(
            perturbation_rate=rate,
            n_steps=500,
            amplitude=1.0,
        )
        violations.append(result['mean_violation'])

    violations = np.array(violations)

    # Check monotonicity above crossover (rate > 1/tau = 1.0)
    above_crossover = perturbation_rates > 1.0
    violations_above = violations[above_crossover]

    if len(violations_above) > 2:
        # Check that violations are non-decreasing (with tolerance for noise)
        diffs = np.diff(violations_above)
        monotonic_fraction = np.mean(diffs >= -0.001)  # Allow tiny dips
        is_monotonic = monotonic_fraction > 0.8
    else:
        is_monotonic = False
        monotonic_fraction = 0.0

    # Also check: below crossover, violations are small
    below_crossover = perturbation_rates < 0.5
    violations_below = violations[below_crossover]
    below_are_small = np.mean(violations_below) < 0.05 if len(violations_below) > 0 else False

    # Fit logistic to violation curve
    from scipy.optimize import curve_fit

    def logistic(x, k, x0, L):
        return L / (1 + np.exp(-k * (np.log10(x) - x0)))

    try:
        popt, _ = curve_fit(logistic, perturbation_rates, violations,
                            p0=[2.0, 0.0, np.max(violations)], maxfev=5000)
        logistic_fit = True
        r2 = 1 - np.sum((violations - logistic(perturbation_rates, *popt))**2) / \
             np.sum((violations - np.mean(violations))**2)
    except Exception:
        logistic_fit = False
        r2 = 0.0

    result = {
        'test': 'T3_violation_monotonic',
        'n_rates': len(perturbation_rates),
        'monotonic_fraction_above_crossover': float(monotonic_fraction),
        'is_monotonic': is_monotonic,
        'mean_violation_below_crossover': float(np.mean(violations_below)) if len(violations_below) > 0 else None,
        'below_are_small': below_are_small,
        'logistic_fit_r2': float(r2),
        'violation_at_rates': {f"{r:.3f}": float(v) for r, v in
                               zip(perturbation_rates.tolist(), violations.tolist())},
        'PASS': is_monotonic and below_are_small,
    }
    return result


def test_T4_qg_clustering():
    """
    T4: Known QG phenomena cluster at gravitational crossover.

    QG signatures (BH information paradox, spacetime foam, non-renormalizability,
    trans-Planckian problem) all appear at energies where tau_pert < tau_grav.
    Non-QG anomalies (g-2 anomaly, proton radius puzzle) appear elsewhere.

    Test: Spearman correlation between "is QG-related" and "proximity to crossover".
    """
    # Known phenomena with their characteristic energy scales (GeV)
    phenomena = [
        # QG phenomena (should cluster near Planck scale)
        {'name': 'BH information paradox', 'E_gev': 1e19, 'is_qg': True},
        {'name': 'Spacetime foam', 'E_gev': 1e19, 'is_qg': True},
        {'name': 'Non-renormalizability of gravity', 'E_gev': 1e19, 'is_qg': True},
        {'name': 'Trans-Planckian problem (inflation)', 'E_gev': 1e16, 'is_qg': True},
        {'name': 'Singularity (BH/Big Bang)', 'E_gev': 1e19, 'is_qg': True},
        # Non-QG phenomena (should NOT cluster near Planck scale)
        {'name': 'Muon g-2 anomaly', 'E_gev': 0.106, 'is_qg': False},
        {'name': 'Proton radius puzzle', 'E_gev': 0.938, 'is_qg': False},
        {'name': 'Neutrino mass origin', 'E_gev': 1e-10, 'is_qg': False},
        {'name': 'Strong CP problem', 'E_gev': 1.0, 'is_qg': False},
        {'name': 'Electroweak hierarchy', 'E_gev': 125.0, 'is_qg': False},
    ]

    # Compute proximity to Planck scale (gravitational crossover)
    E_planck = E_PLANCK_GEV  # ~1.22e19 GeV

    for p in phenomena:
        p['log_E'] = np.log10(p['E_gev'])
        p['log_distance_to_planck'] = abs(np.log10(p['E_gev']) - np.log10(E_planck))
        p['at_crossover'] = p['log_distance_to_planck'] < 4  # Within 4 orders

    # QG phenomena should be at crossover, non-QG should not
    is_qg = np.array([p['is_qg'] for p in phenomena], dtype=float)
    at_crossover = np.array([p['at_crossover'] for p in phenomena], dtype=float)

    # Spearman correlation
    rho, p_val = spearmanr(is_qg, -np.array([p['log_distance_to_planck'] for p in phenomena]))

    # Direct check: all QG phenomena within 4 orders of Planck
    qg_at_crossover = all(p['at_crossover'] for p in phenomena if p['is_qg'])
    # Most non-QG NOT at crossover
    non_qg_away = sum(1 for p in phenomena if not p['is_qg'] and not p['at_crossover'])
    non_qg_total = sum(1 for p in phenomena if not p['is_qg'])

    result = {
        'test': 'T4_qg_clustering',
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'qg_all_at_crossover': qg_at_crossover,
        'non_qg_away_fraction': float(non_qg_away / non_qg_total) if non_qg_total > 0 else 0,
        'phenomena': [{k: v for k, v in p.items()} for p in phenomena],
        'PASS': rho > 0.7 and qg_at_crossover,
    }
    return result


# ============================================================
# Main
# ============================================================
def main():
    setup = setup_experiment(__file__)
    registry = PredictionRegistry()

    print("=" * 70)
    print("EXP 01 — Gravitational Response-Time Hierarchy")
    print("Milestone 11, Block A")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    # T1: Response-time ordering + coupling match
    print("\n--- T1: Response-time ordering + coupling match ---")
    t1 = test_T1_ordering()
    results['T1'] = t1
    if t1['PASS']:
        score += 1
        print(f"  PASS: ordering correct, coupling match {t1['log_agreement_pct']:.2f}% in log space")
    else:
        print(f"  FAIL: order={t1['actual_order']}, coupling_match={t1['coupling_match']}")
    for k, v in t1['hierarchy'].items():
        print(f"    {k:>10s}: depth={v['depth']:>4d}, log10(tau)={v['log10_tau']:>8.2f}")
    print(f"    alpha_grav(DFT)={t1['alpha_grav_dft']:.3e} vs alpha_grav(proton)={t1['alpha_grav_proton']:.3e}")

    # T2: Coupling ratios vs known physics
    print("\n--- T2: Coupling ratios vs known physics ---")
    t2 = test_T2_coupling_ratios()
    results['T2'] = t2
    if t2['PASS']:
        score += 1
        print("  PASS: coupling ratios match known physics")
    else:
        print("  FAIL: coupling ratios don't match")
    for r in t2['ratio_tests']:
        status = "OK" if r['match'] else "FAIL"
        print(f"    {r['pair']:>20s}: log10(DFT)={r['log10_dft']:.2f} "
              f"vs log10(known)={r['log10_known']:.2f} "
              f"(err={r['log_relative_error']:.1%}) [{status}]")

    # T3: Violation monotonicity
    print("\n--- T3: LawNegotiator violation monotonicity ---")
    t3 = test_T3_violation_monotonic()
    results['T3'] = t3
    if t3['PASS']:
        score += 1
        print(f"  PASS: monotonic above crossover ({t3['monotonic_fraction_above_crossover']:.1%}), "
              f"small below ({t3['mean_violation_below_crossover']:.4f})")
    else:
        print(f"  FAIL: monotonic={t3['is_monotonic']}, below_small={t3['below_are_small']}")
    print(f"    Logistic fit R²: {t3['logistic_fit_r2']:.4f}")

    # T4: QG clustering
    print("\n--- T4: QG phenomena clustering ---")
    t4 = test_T4_qg_clustering()
    results['T4'] = t4
    if t4['PASS']:
        score += 1
        print(f"  PASS: rho={t4['spearman_rho']:.3f}, all QG at crossover={t4['qg_all_at_crossover']}")
    else:
        print(f"  FAIL: rho={t4['spearman_rho']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print(f"EXP 01 SCORE: {score}/{total}")
    print("=" * 70)

    results['score'] = score
    results['total'] = total
    results['pass_rate'] = score / total

    # Save
    save_results(results, RESULTS_DIR, "exp_01_response_time_hierarchy")
    return results


if __name__ == "__main__":
    main()
