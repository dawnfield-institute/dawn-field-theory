"""
exp_01 -- Gravity Gates Entanglement

Midnight Initiative, Thread 5 (Phase-Rate Primitive)

Hypothesis: Entanglement formation probability across a gravitational gradient
shows structure -- peaks at commensurate phase-rate ratios phi^n -- not a flat
background. Standard QM predicts flat; DFT predicts peaks.

The phase-rate field w(d) = (mc^2/hbar) * sqrt(1 - 2*epsilon*phi^(-d)) is the
rate at which an internal quantum clock advances at PAC depth d. Gravitational
embedding sets the mean of w via g_out = g_in^2 (M9/M12). Entanglement forms
when two nodes' phase-rates harmonically lock (commensurate ratio).

Tests:
  T1: Phase-rate maps to depth-weighted SEC clock rate
  T2: Commensurate ratios from g_out = g_in^2 are phi-based
  T3: Entanglement shows structure at phi-harmonic ratios
  T4: Structure discriminates DFT from flat QM

Source: journals/2026-06-03_phase-rate-primitive.md (Parts III, IX)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from phase_rate import (
    PHI, INV_PHI, LN_PHI,
    COMPTON_FREQ,
    phase_rate_at_depth, phase_rate_ratio,
    commensurate_ratios, is_commensurate,
    phase_coupling_kernel,
    entanglement_vs_ratio,
    fit_flat_model, fit_phi_harmonic_model, bic,
    save_midnight_results, _convert_numpy,
    DynkinDiagram,
)


# ============================================================
# T1: Phase-rate maps to depth-weighted SEC clock rate
# ============================================================

def test_T1_phase_rate_definition():
    """T1: Phase-rate w(d) is monotonically decreasing, converges to
    phi-dependent limit, and matches weak-field expansion."""
    print("\n  T1: Phase-rate maps to depth-weighted SEC clock rate")

    epsilon = 0.01
    depths = list(range(1, 21))
    w_values = [phase_rate_at_depth(d, epsilon) for d in depths]

    # Check 1: monotonically increasing (deeper = weaker gravity = faster clock)
    diffs = [w_values[i+1] - w_values[i] for i in range(len(w_values)-1)]
    is_monotone = all(d > 0 for d in diffs)
    print(f"    Monotonically increasing with depth: {is_monotone}")

    # Check 2: ratio w(d)/w(d+1) converges for large d (approaches 1 from below)
    ratios_adjacent = []
    for d in range(10, 20):
        r = phase_rate_ratio(d, d+1, epsilon)
        ratios_adjacent.append(r)

    ratio_spread = max(ratios_adjacent) - min(ratios_adjacent)
    converges = ratio_spread < 1e-4
    print(f"    Ratio convergence spread (d=10..19): {ratio_spread:.2e} (<1e-4: {converges})")

    # Check 3: weak-field expansion matches analytic form
    # In weak field: w(d) ~ COMPTON_FREQ * (1 - epsilon * phi^(-d))
    max_rel_error = 0.0
    for d in depths[5:]:
        w_exact = phase_rate_at_depth(d, epsilon)
        w_approx = COMPTON_FREQ * (1 - epsilon * PHI**(-d))
        rel_error = abs(w_exact - w_approx) / w_exact
        max_rel_error = max(max_rel_error, rel_error)

    weak_field_ok = max_rel_error < 1e-6
    print(f"    Weak-field expansion max error (d>=6): {max_rel_error:.2e} (<1e-6: {weak_field_ok})")

    passed = is_monotone and converges and weak_field_ok
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_phase_rate_definition',
        'is_monotone': is_monotone,
        'ratio_convergence_spread': float(ratio_spread),
        'weak_field_max_error': float(max_rel_error),
        'n_depths': len(depths),
        'PASS': passed,
    }


# ============================================================
# T2: Commensurate ratios from g_out = g_in^2 are phi-based
# ============================================================

def test_T2_commensurate_ratios():
    """T2: g_out = g_in^2 uniqueness produces phi-based commensurate ratios."""
    print("\n  T2: Commensurate ratios from g_out = g_in^2 are phi-based")

    # Check 1: g_out = g_in^2 for g_in = 1/phi
    g_in = INV_PHI
    g_out = 1.0 - g_in
    g_in_squared = g_in**2
    duality_error = abs(g_out - g_in_squared)
    duality_ok = duality_error < 1e-14
    print(f"    g_out = g_in^2 error: {duality_error:.2e} (<1e-14: {duality_ok})")

    # Check 2: potential ratios at adjacent depths = phi
    epsilon = 0.01
    potential_ratios = []
    for d in range(10, 19):
        pot_d = epsilon * PHI**(-d)
        pot_d1 = epsilon * PHI**(-(d+1))
        potential_ratios.append(pot_d / pot_d1)

    max_phi_error = max(abs(r - PHI) for r in potential_ratios)
    potentials_phi = max_phi_error < 1e-10
    print(f"    Potential ratio deviation from phi: {max_phi_error:.2e} (<1e-10: {potentials_phi})")

    # Check 3: commensurate ratios form geometric series
    cr = commensurate_ratios(max_n=8)
    positive_ratios = sorted([(n, v) for n, v in cr if n > 0], key=lambda x: x[0])
    geometric_errors = []
    for i in range(len(positive_ratios) - 1):
        n1, v1 = positive_ratios[i]
        n2, v2 = positive_ratios[i+1]
        series_ratio = v2 / v1
        geometric_errors.append(abs(series_ratio - PHI))

    max_geo_error = max(geometric_errors)
    geometric_ok = max_geo_error < 1e-10
    print(f"    Geometric series error: {max_geo_error:.2e} (<1e-10: {geometric_ok})")

    passed = duality_ok and potentials_phi and geometric_ok
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_commensurate_ratios',
        'duality_error': float(duality_error),
        'max_potential_ratio_error': float(max_phi_error),
        'max_geometric_series_error': float(max_geo_error),
        'commensurate_values': [(n, float(v)) for n, v in positive_ratios[:5]],
        'PASS': passed,
    }


# ============================================================
# T3: Entanglement shows structure at phi-harmonic ratios
# ============================================================

def test_T3_entanglement_structure():
    """T3: Entanglement entropy vs phase-rate ratio shows peaks at phi^n."""
    print("\n  T3: Entanglement shows structure at phi-harmonic ratios")

    # E_6 has 5 non-degenerate positive Laplacian eigenvalues — richer
    # spectral structure than D_4 (which has degenerate eigenvalues at 0.25).
    # D_4 is key for quantum uncertainty (non-abelian Aut) but spectrally poor.
    e6 = DynkinDiagram('E', 6)
    adj = e6.adjacency

    ratio_array = np.linspace(0.3, 4.0, 300)
    result = entanglement_vs_ratio(adj, ratio_array, sigma_log=0.15, max_n=6)
    S = result['entropy_array']

    # Peak detection: find local maxima above half-maximum
    half_max = (np.max(S) + np.min(S)) / 2
    peaks = []
    for i in range(1, len(S) - 1):
        if S[i] > S[i-1] and S[i] > S[i+1] and S[i] > half_max:
            peaks.append((ratio_array[i], S[i]))

    n_peaks = len(peaks)
    has_peaks = n_peaks >= 3
    print(f"    Peaks detected (above half-max): {n_peaks} (>=3: {has_peaks})")

    # Check each peak is near phi^n
    peaks_at_phi = []
    for r_peak, s_peak in peaks:
        near, n, dev = is_commensurate(r_peak, tolerance=0.05, max_n=6)
        peaks_at_phi.append({
            'ratio': float(r_peak),
            'entropy': float(s_peak),
            'near_phi_n': near,
            'nearest_n': n,
            'deviation': float(dev),
        })

    all_at_phi = all(p['near_phi_n'] for p in peaks_at_phi) if peaks_at_phi else False
    print(f"    All peaks near phi^n (within 5%): {all_at_phi}")

    # Check phi^1 peak is tallest (if present)
    phi1_peaks = [p for p in peaks_at_phi if p['nearest_n'] == 1]
    if phi1_peaks and peaks_at_phi:
        phi1_entropy = phi1_peaks[0]['entropy']
        max_other = max((p['entropy'] for p in peaks_at_phi if p['nearest_n'] != 1), default=0)
        phi1_tallest = phi1_entropy >= max_other * 0.95
    else:
        phi1_tallest = False
    print(f"    phi^1 peak is tallest: {phi1_tallest}")

    passed = has_peaks and all_at_phi and phi1_tallest
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_entanglement_structure',
        'graph': 'E_6',
        'n_ratio_points': len(ratio_array),
        'n_peaks': n_peaks,
        'peaks': peaks_at_phi,
        'n_spectral_modes': result['n_spectral'],
        'eigenvalues': result['eigenvalues'],
        'has_peaks': has_peaks,
        'all_at_phi': all_at_phi,
        'phi1_tallest': phi1_tallest,
        'entropy_min': float(np.min(S)),
        'entropy_max': float(np.max(S)),
        'PASS': passed,
    }


# ============================================================
# T4: Structure discriminates DFT from flat QM
# ============================================================

def test_T4_model_discrimination():
    """T4: Phi-harmonic model fits entanglement curve significantly
    better than flat (constant) model."""
    print("\n  T4: Structure discriminates DFT from flat QM")

    e6 = DynkinDiagram('E', 6)
    adj = e6.adjacency

    ratio_array = np.linspace(0.3, 4.0, 300)
    result = entanglement_vs_ratio(adj, ratio_array, sigma_log=0.15, max_n=6)
    S = result['entropy_array']
    N = len(S)

    # Flat model: S(r) = C
    C_flat, chi2_flat = fit_flat_model(S)
    bic_flat = bic(chi2_flat, 1, N)

    # Phi-harmonic model: S(r) = B + A * K(r)
    params_phi, chi2_phi = fit_phi_harmonic_model(ratio_array, S, max_n=6)
    bic_phi = bic(chi2_phi, 3, N)

    # Check 1: chi2 ratio
    if chi2_phi > 0:
        chi2_ratio = chi2_flat / chi2_phi
    else:
        chi2_ratio = float('inf')
    ratio_ok = chi2_ratio > 5.0
    print(f"    chi2_flat / chi2_phi = {chi2_ratio:.1f} (>5.0: {ratio_ok})")

    # Check 2: delta BIC
    delta_bic = bic_flat - bic_phi
    bic_ok = delta_bic > 10.0
    print(f"    delta_BIC = {delta_bic:.1f} (>10: {bic_ok})")

    print(f"    Flat:  chi2={chi2_flat:.4f}, BIC={bic_flat:.1f}")
    print(f"    Phi:   chi2={chi2_phi:.4f}, BIC={bic_phi:.1f}, sigma={params_phi['sigma_log']:.3f}")

    passed = ratio_ok and bic_ok
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_model_discrimination',
        'flat_model': {'C': C_flat, 'chi2': chi2_flat, 'bic': bic_flat},
        'phi_model': {**params_phi, 'chi2': chi2_phi, 'bic': bic_phi},
        'chi2_ratio': float(chi2_ratio),
        'delta_bic': float(delta_bic),
        'n_data_points': N,
        'ratio_ok': ratio_ok,
        'bic_ok': bic_ok,
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_01: Gravity Gates Entanglement")
    print("Midnight Initiative, Thread 5 (Phase-Rate Primitive)")
    print("=" * 70)

    t1 = test_T1_phase_rate_definition()
    t2 = test_T2_commensurate_ratios()
    t3 = test_T3_entanglement_structure()
    t4 = test_T4_model_discrimination()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_01_gravity_gates_entanglement',
        'initiative': 'midnight',
        'thread': 'phase_rate_primitive',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_01_gravity_gates_entanglement', _convert_numpy(data))
