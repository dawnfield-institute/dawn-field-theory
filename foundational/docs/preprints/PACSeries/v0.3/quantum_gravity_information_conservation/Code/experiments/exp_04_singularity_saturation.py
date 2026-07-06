"""
exp_04 — Singularity Resolution via Cascade Saturation

Milestone 11, Block B (Black Hole Resolution)

Hypothesis: MVAE sets rho_max. Cascade density saturates before singularity.
The interior transitions from Schwarzschild to a de Sitter core with constant
Planck density. The metric is non-singular everywhere.

Tests:
  T1: Saturation radius derivable for any M
  T2: Modified metric non-singular (Kretschner scalar finite everywhere)
  T3: Matches Schwarzschild for r >> r_min
  T4: Interior information scales as M^2 (area, not volume)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, PI, LN2,
    L_PLANCK_M, M_PLANCK_KG, M_SUN_KG,
    CascadeSaturation,
    save_results, setup_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_saturation_radius():
    """T1: Saturation radius derivable for any M."""
    masses = [1e-8, 1e-4, 1.0, 10.0, 1e6, 1e9]  # Solar masses
    results = []

    for M in masses:
        sat = CascadeSaturation(M)
        results.append({
            'M_solar': M,
            'r_s_meters': float(sat.r_s),
            'r_min_meters': float(sat.r_min_meters),
            'r_min_over_r_s': float(sat.r_min_planck / sat.r_s_planck),
            'r_min_planck_units': float(sat.r_min_planck),
        })

    # r_min should be much smaller than r_s for stellar-mass and above
    all_derivable = all(r['r_min_over_r_s'] < 1.0 for r in results)
    # r_min should decrease relative to r_s for larger masses
    ratios = [r['r_min_over_r_s'] for r in results]
    monotonic = all(ratios[i] >= ratios[i+1] for i in range(len(ratios)-1))

    return {
        'test': 'T1_saturation_radius',
        'masses': results,
        'all_derivable': all_derivable,
        'ratio_decreases_with_mass': monotonic,
        'PASS': all_derivable and monotonic,
    }


def test_T2_non_singular():
    """T2: Modified metric non-singular (Kretschner finite everywhere)."""
    sat = CascadeSaturation(1.0)  # Solar mass BH

    # Evaluate Kretschner scalar from r_min/10 to 10*r_s
    r_values = np.logspace(
        np.log10(sat.r_min_planck * 0.1),
        np.log10(sat.r_s_planck * 10),
        1000,
    )

    K = sat.kretschner_scalar(r_values)
    g_tt = sat.metric_g_tt(r_values)

    # Key checks
    all_finite = np.all(np.isfinite(K)) and np.all(np.isfinite(g_tt))
    K_at_center = sat.kretschner_scalar(np.array([sat.r_min_planck * 0.01]))[0]
    K_finite_at_center = np.isfinite(K_at_center)

    # The de Sitter interior has constant K
    interior = r_values < sat.r_min_planck
    if np.sum(interior) > 1:
        K_interior = K[interior]
        K_interior_std = np.std(K_interior)
        K_interior_mean = np.mean(K_interior)
        interior_constant = K_interior_std / K_interior_mean < 0.01 if K_interior_mean > 0 else False
    else:
        interior_constant = True  # Can't test with insufficient points

    return {
        'test': 'T2_non_singular',
        'all_finite': all_finite,
        'K_at_center': float(K_at_center),
        'K_finite_at_center': K_finite_at_center,
        'interior_K_constant': interior_constant,
        'r_min_planck': float(sat.r_min_planck),
        'r_s_planck': float(sat.r_s_planck),
        'PASS': all_finite and K_finite_at_center,
    }


def test_T3_matches_schwarzschild():
    """T3: Matches Schwarzschild for r >> r_min."""
    sat = CascadeSaturation(1.0)

    # Test at r = 10*r_min, 100*r_min, 0.5*r_s, r_s, 2*r_s, 10*r_s
    test_radii = np.array([
        10 * sat.r_min_planck,
        100 * sat.r_min_planck,
        0.5 * sat.r_s_planck,
        2.0 * sat.r_s_planck,
        10.0 * sat.r_s_planck,
    ])

    deviations = []
    for r in test_radii:
        g_tt_mod = sat.metric_g_tt(np.array([r]))[0]
        g_tt_schw = 1.0 - sat.r_s_planck / r
        if abs(g_tt_schw) > 1e-20:
            dev = abs(g_tt_mod - g_tt_schw) / abs(g_tt_schw)
        else:
            dev = abs(g_tt_mod - g_tt_schw)
        deviations.append({
            'r_planck': float(r),
            'r_over_r_min': float(r / sat.r_min_planck),
            'g_tt_modified': float(g_tt_mod),
            'g_tt_schwarzschild': float(g_tt_schw),
            'relative_deviation': float(dev),
        })

    # At r >> r_min, deviation should be tiny
    far_field = [d for d in deviations if d['r_over_r_min'] > 50]
    far_match = all(d['relative_deviation'] < 1e-10 for d in far_field) if far_field else False

    return {
        'test': 'T3_matches_schwarzschild',
        'deviations': deviations,
        'far_field_match': far_match,
        'PASS': far_match,
    }


def test_T4_area_scaling():
    """T4: Interior information scales as M^2 (area, not volume)."""
    masses = [0.1, 1.0, 10.0, 100.0, 1000.0]

    info_values = []
    for M in masses:
        sat = CascadeSaturation(M)
        info = sat.information_content(n_shells=500)
        info_values.append({
            'M_solar': M,
            'info': float(info),
            'log_M': float(np.log10(M)),
            'log_info': float(np.log10(max(info, 1e-300))),
        })

    # Fit log(info) vs log(M): should get slope ~ 2 (area) not ~ 3 (volume)
    log_M = np.array([v['log_M'] for v in info_values])
    log_info = np.array([v['log_info'] for v in info_values])

    valid = np.isfinite(log_info)
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(log_M[valid], log_info[valid], 1)
        slope = coeffs[0]
        # Predict
        log_info_pred = np.polyval(coeffs, log_M[valid])
        ss_res = np.sum((log_info[valid] - log_info_pred)**2)
        ss_tot = np.sum((log_info[valid] - np.mean(log_info[valid]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        slope = 0.0
        r2 = 0.0

    # Area scaling: slope ~ 2 (±0.05, tightened from ±0.5)
    # Volume scaling would be slope ~ 3
    is_area = abs(slope - 2.0) < 0.05
    not_volume = abs(slope - 3.0) > 0.3

    # PROFILE-INDEPENDENCE CHECK:
    # If the gradient method gives slope 2 for ANY density profile, it's
    # measuring integration range, not physics. Test with rho ~ 1/r^2:
    # |d(rho)/dr| = 2*r_s/r^3, integrand = 4*pi*r^2 * 2*r_s/r^3 = 8*pi*r_s/r
    # integral ~ ln(r_max/r_min) ~ ln(M), so slope should be ~1, NOT 2.
    from quantum_gravity import PI as _PI
    info_alt = []
    for M_val in masses:
        sat_alt = CascadeSaturation(M_val)
        r_alt = np.linspace(sat_alt.r_min_planck, sat_alt.r_s_planck, 500)
        rho_alt = sat_alt.r_s_planck / r_alt**2  # 1/r^2 profile
        drho_dr_alt = np.abs(np.gradient(rho_alt, r_alt))
        integrand_alt = 4 * _PI * r_alt**2 * drho_dr_alt
        info_a = np.trapz(integrand_alt, r_alt)
        info_alt.append(float(info_a))

    log_info_alt = np.array([np.log10(max(x, 1e-300)) for x in info_alt])
    valid_alt = np.isfinite(log_info_alt)
    if np.sum(valid_alt) >= 3:
        coeffs_alt = np.polyfit(log_M[valid_alt], log_info_alt[valid_alt], 1)
        slope_alt = coeffs_alt[0]
    else:
        slope_alt = slope  # Can't discriminate

    # The method should give a DIFFERENT slope for a different profile
    profile_discriminates = abs(slope_alt - slope) > 0.3

    return {
        'test': 'T4_area_scaling',
        'info_values': info_values,
        'slope': float(slope),
        'r2': float(r2),
        'is_area_scaling': is_area,
        'not_volume_scaling': not_volume,
        'slope_alt_profile': float(slope_alt),
        'profile_discriminates': profile_discriminates,
        'PASS': is_area and r2 > 0.95 and profile_discriminates,
    }


def main():
    setup = setup_experiment(__file__)

    print("=" * 70)
    print("EXP 04 — Singularity Resolution via Cascade Saturation")
    print("Milestone 11, Block B")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [('T1', test_T1_saturation_radius),
                           ('T2', test_T2_non_singular),
                           ('T3', test_T3_matches_schwarzschild),
                           ('T4', test_T4_area_scaling)]:
        print(f"\n--- {name} ---")
        t = test_fn()
        results[name] = t
        if t['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

        # Print key details
        if name == 'T1':
            for m in t['masses'][:3]:
                print(f"    M={m['M_solar']:.0e} M_sun: r_min/r_s = {m['r_min_over_r_s']:.2e}")
        elif name == 'T2':
            print(f"    K at center: {t['K_at_center']:.4e} (finite={t['K_finite_at_center']})")
        elif name == 'T3':
            for d in t['deviations']:
                print(f"    r/r_min={d['r_over_r_min']:.0f}: deviation={d['relative_deviation']:.2e}")
        elif name == 'T4':
            print(f"    slope={t['slope']:.3f} (target 2.0), R²={t['r2']:.4f}")

    print("\n" + "=" * 70)
    print(f"EXP 04 SCORE: {score}/{total}")
    print("=" * 70)

    results['score'] = score
    results['total'] = total
    save_results(results, RESULTS_DIR, "exp_04_singularity_saturation")
    return results


if __name__ == "__main__":
    main()
