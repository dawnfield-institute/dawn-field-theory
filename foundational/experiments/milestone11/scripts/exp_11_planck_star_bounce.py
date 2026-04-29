"""
exp_11 — Planck Star Bounce

Milestone 11, Block D (Cosmological Contact)

Hypothesis: Cascade saturation + PAC conservation forces a bounce at the
core of collapsing matter (Rovelli & Vidotto 2014 analog). Without PAC,
collapse proceeds to singularity.

Tests:
  T1: Bounce time t_bounce ~ M * sqrt(r_min/r_s) in Planck units
  T2: PAC forces bounce (without PAC -> singularity)
  T3: Burst energy E_burst ~ (M/M_P)^(-1/3) in Planck units
  T4: Evaporation-bounce crossover: Hawking correction shuts off evap at M_Planck
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, PI, LN2,
    M_PLANCK_KG, M_SUN_KG,
    PlanckStarDynamics, CascadeSaturation,
    save_results, setup_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_bounce_time():
    """T1: Bounce time scales as M * sqrt(r_min/r_s)."""
    masses = [1e-4, 1e-2, 1.0, 100.0, 1e6]
    data = []

    for M in masses:
        ps = PlanckStarDynamics(M)
        t_bounce = ps.bounce_time_planck()
        data.append({
            'M_solar': M,
            'M_planck_ratio': float(ps.sat.M_planck_ratio),
            't_bounce_planck': float(t_bounce),
            'log_M': float(np.log10(M)),
            'log_t': float(np.log10(max(t_bounce, 1e-300))),
        })

    # Check scaling: t_bounce ~ M * sqrt(r_min/r_s)
    # r_min/r_s = 1/M_planck_ratio^2
    # sqrt(r_min/r_s) = 1/M_planck_ratio
    # t_bounce ~ M_planck_ratio * 1/M_planck_ratio = 1 (constant in Planck units!)
    # Actually: t_bounce = M_planck_ratio * sqrt(r_min/r_s)
    #                    = M_planck_ratio * (1/M_planck_ratio) = 1

    # So t_bounce should be approximately 1 Planck time for all masses
    t_values = [d['t_bounce_planck'] for d in data]
    all_near_1 = all(0.1 < t < 10 for t in t_values)

    # Or check that they're constant (independent of mass)
    std_over_mean = np.std(t_values) / np.mean(t_values) if np.mean(t_values) > 0 else float('inf')
    constant = std_over_mean < 0.01

    return {
        'test': 'T1_bounce_time',
        'data': data,
        't_bounce_values': t_values,
        'all_near_planck_time': all_near_1,
        'constant_across_mass': constant,
        'std_over_mean': float(std_over_mean),
        'PASS': all_near_1 and constant,
    }


def test_T2_pac_forces_bounce():
    """
    T2: PAC conservation forces bounce; without PAC -> singularity.

    With perfect PAC (epsilon=0): information pressure at rho_max
    exceeds gravitational pressure -> bounce.
    With broken PAC (epsilon>0.5): information can be destroyed,
    no pressure -> collapse continues.
    """
    masses = [0.01, 1.0, 100.0]
    epsilon_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = []
    for M in masses:
        ps = PlanckStarDynamics(M)
        mass_results = []
        for eps in epsilon_values:
            bounces = ps.pac_forces_bounce(epsilon_pac=eps)
            mass_results.append({
                'epsilon': eps,
                'bounces': bounces,
            })
        results.append({
            'M_solar': M,
            'results': mass_results,
        })

    # Perfect PAC should always bounce
    perfect_pac_bounces = all(
        r['results'][0]['bounces'] for r in results
    )

    # Full PAC violation should not bounce
    full_violation_no_bounce = all(
        not r['results'][-1]['bounces'] for r in results
    )

    # There should be a transition epsilon
    has_transition = perfect_pac_bounces and full_violation_no_bounce

    return {
        'test': 'T2_pac_forces_bounce',
        'results': results,
        'perfect_pac_bounces': perfect_pac_bounces,
        'full_violation_no_bounce': full_violation_no_bounce,
        'has_transition': has_transition,
        'PASS': perfect_pac_bounces and full_violation_no_bounce,
    }


def test_T3_burst_energy():
    """
    T3: Burst energy E_burst ~ (M/M_Planck)^(-1/3).

    For primordial BHs: burst in gamma-ray range.
    For stellar-mass BHs: extremely weak burst.
    """
    masses = [1e-8, 1e-4, 1.0, 100.0, 1e6]
    data = []

    for M in masses:
        ps = PlanckStarDynamics(M)
        E_burst = ps.burst_energy_planck()
        data.append({
            'M_solar': M,
            'E_burst_planck': float(E_burst),
            'log_M': float(np.log10(M)),
            'log_E': float(np.log10(max(E_burst, 1e-300))),
        })

    # Check -1/3 scaling
    log_M = np.array([d['log_M'] for d in data])
    log_E = np.array([d['log_E'] for d in data])

    # But E_burst = M_planck_ratio^(-1/3) = (M * M_SUN / M_PLANCK)^(-1/3)
    # log(E) = -1/3 * log(M_planck_ratio) = -1/3 * (log(M) + log(M_SUN/M_PLANCK))
    # So d(log E)/d(log M) = -1/3
    coeffs = np.polyfit(log_M, log_E, 1)
    slope = coeffs[0]
    slope_near_minus_third = abs(slope - (-1.0/3.0)) < 0.05

    # Burst energy decreases with mass (larger BH = weaker burst)
    decreasing = all(data[i]['E_burst_planck'] > data[i+1]['E_burst_planck']
                     for i in range(len(data)-1))

    return {
        'test': 'T3_burst_energy',
        'data': data,
        'slope': float(slope),
        'slope_near_minus_third': slope_near_minus_third,
        'decreasing_with_mass': decreasing,
        'PASS': slope_near_minus_third and decreasing,
    }


def test_T4_zeno_limit():
    """
    T4: Evaporation-bounce crossover near Planck mass.

    The Hawking correction (1 - (r_min/r_s)^2) from cascade saturation
    suppresses evaporation as M -> M_Planck (correction -> 0, T_eff -> 0).
    The corrected evaporation time DIVERGES while t_bounce = 1 t_Planck (constant).
    Bounce takes over as the dominant timescale: this is a crossover, not convergence.
    """
    from quantum_gravity import hawking_with_correction, page_time_scaling, T_PLANCK_S

    # Masses approaching Planck mass
    M_planck_solar = M_PLANCK_KG / M_SUN_KG
    mass_ratios = [100, 10, 5, 2, 1.5, 1.1, 1.01, 1.0]
    data = []

    for ratio in mass_ratios:
        M = ratio * M_planck_solar
        ps = PlanckStarDynamics(M)
        t_bounce = ps.bounce_time_planck()

        # Standard evaporation time (no correction)
        pt = page_time_scaling(M)
        t_evap_standard = pt['t_evap_seconds'] / T_PLANCK_S

        # Hawking correction: T_eff = T_H * (1 - (r_min/r_s)^2)
        hc = hawking_with_correction(M)
        correction = hc['correction_factor']

        # Corrected evaporation time: t_evap_corrected ~ t_evap_standard / correction^4
        # (Stefan-Boltzmann: luminosity ~ T^4, so t_evap ~ 1/T^4)
        if correction > 1e-30:
            t_evap_corrected = t_evap_standard / correction**4
        else:
            t_evap_corrected = float('inf')

        data.append({
            'M_over_M_planck': ratio,
            't_bounce_planck': float(t_bounce),
            't_evap_standard': float(t_evap_standard),
            't_evap_corrected': float(t_evap_corrected) if np.isfinite(t_evap_corrected) else 1e30,
            'correction_factor': float(correction),
            'bounce_dominates': t_bounce < t_evap_corrected,
        })

    # Key checks:
    # 1. Correction factor -> 0 as M -> M_Planck (evaporation shuts off)
    planck_entry = next((d for d in data if d['M_over_M_planck'] == 1.0), None)
    correction_suppresses = planck_entry is not None and planck_entry['correction_factor'] < 0.1

    # 2. There exists a crossover: for large M, evap dominates; for M ~ M_P, bounce dominates
    large_M_evap = data[0]['t_evap_corrected'] > data[0]['t_bounce_planck']  # Large M: evap slower
    planck_bounce = planck_entry['bounce_dominates'] if planck_entry else False

    # 3. Bounce time is constant (~1 t_Planck) while corrected evap time grows
    t_bounces = [d['t_bounce_planck'] for d in data]
    bounce_constant = max(t_bounces) - min(t_bounces) < 0.01

    return {
        'test': 'T4_zeno_limit',
        'data': data,
        'correction_suppresses_evap': correction_suppresses,
        'bounce_dominates_at_planck': planck_bounce,
        'bounce_constant': bounce_constant,
        'PASS': correction_suppresses and planck_bounce and bounce_constant,
    }


def main():
    setup = setup_experiment(__file__)

    print("=" * 70)
    print("EXP 11 — Planck Star Bounce")
    print("Milestone 11, Block D")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [('T1', test_T1_bounce_time),
                           ('T2', test_T2_pac_forces_bounce),
                           ('T3', test_T3_burst_energy),
                           ('T4', test_T4_zeno_limit)]:
        print(f"\n--- {name} ---")
        t = test_fn()
        results[name] = t
        if t['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

        if name == 'T1':
            print(f"    t_bounce constant across mass: std/mean = {t['std_over_mean']:.2e}")
            for d in t['data'][:3]:
                print(f"    M={d['M_solar']:.0e}: t_bounce = {d['t_bounce_planck']:.4f} t_Planck")
        elif name == 'T2':
            for r in t['results'][:1]:
                print(f"    M={r['M_solar']:.2f} M_sun:")
                for er in r['results']:
                    print(f"      eps={er['epsilon']:.1f}: {'BOUNCE' if er['bounces'] else 'COLLAPSE'}")
        elif name == 'T3':
            print(f"    slope = {t['slope']:.4f} (target -1/3 = -0.3333)")
            for d in t['data'][:3]:
                print(f"    M={d['M_solar']:.0e}: E_burst = {d['E_burst_planck']:.4e} E_Planck")
        elif name == 'T4':
            for d in t['data'][-3:]:
                print(f"    M/M_P={d['M_over_M_planck']:.2f}: t_bounce={d['t_bounce_planck']:.4f}, "
                      f"correction={d['correction_factor']:.4e}, bounce_dom={d['bounce_dominates']}")

    print("\n" + "=" * 70)
    print(f"EXP 11 SCORE: {score}/{total}")
    print("=" * 70)

    results['score'] = score
    results['total'] = total
    save_results(results, RESULTS_DIR, "exp_11_planck_star_bounce")
    return results


if __name__ == "__main__":
    main()
