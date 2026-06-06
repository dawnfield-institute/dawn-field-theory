"""
exp_04 -- Cascade Width Oscillation Profile

Midnight Initiative, Thread 1 (Photon Archaeology)

Hypothesis: The cascade clock N(z) = a + (1/ln(phi)) * ln(t_lookback) creates
a non-monotonic line-width profile W(z) — widths peak at cascade transitions
(integer N) and collapse between them. This oscillatory z-dependence is
qualitatively different from any standard broadening mechanism (thermal,
turbulent, instrumental — all monotonic or random).

This experiment maps the oscillation precisely: transition redshifts, the full
W(z) curve, the discriminating signature vs monotonic models, and observational
predictions for specific spectral lines.

Tests:
  T1: Map cascade transition redshifts (peaks at integer N, troughs at half-integer)
  T2: High-resolution W(z) curve showing oscillatory structure
  T3: Oscillatory model fits dramatically better than monotonic
  T4: Observational predictions — width contrasts at specific redshifts

Sources: exp_03 T3 (220% variation, Spearman rho=1.000), M9 cascade clock
"""

import sys
import numpy as np
from pathlib import Path
from scipy.optimize import brentq
from scipy.stats import spearmanr

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = MIDNIGHT_ROOT.parent

sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
sys.path.insert(0, str(EXPERIMENTS_ROOT / "milestone-r" / "core"))
sys.path.insert(0, str(EXPERIMENTS_ROOT / "milestone9" / "core"))

from phase_rate import (
    PHI, INV_PHI, LN_PHI, PI,
    save_midnight_results, _convert_numpy,
)
from radiation_physics import line_width_from_disequilibrium
from infodynamics import (
    z_to_lookback, B_DFT, cascade_clock, cascade_clock_fit,
    N_physical, T_UNIVERSE,
)


def n_at_z(z, a_clock):
    """Cascade level at redshift z (with boundary handling)."""
    t_look = z_to_lookback(z)
    if t_look <= 0.001:
        t_look = 0.001
    N_raw = cascade_clock(t_look, a_clock, B_DFT)
    return max(N_raw, 1.0)


def z_for_cascade_level(target_n, a_clock, z_min=0.001, z_max=20.0):
    """Find redshift z where N(z) = target_n via bisection."""
    n_lo = n_at_z(z_min, a_clock)
    n_hi = n_at_z(z_max, a_clock)

    if target_n < n_lo or target_n > n_hi:
        return None

    def residual(z):
        return n_at_z(z, a_clock) - target_n

    try:
        return brentq(residual, z_min, z_max, xtol=1e-6)
    except ValueError:
        return None


def disequilibrium_at_n(N):
    """Disequilibrium: 1.0 at integer N (transition), 0.0 at half-integer (settled)."""
    dist_to_int = abs(N - round(N))
    return max(0.0, 1.0 - 2.0 * dist_to_int)


def width_at_z(z, a_clock, adj, n_trials=200):
    """Compute line width at redshift z using the cascade disequilibrium model."""
    N = n_at_z(z, a_clock)
    diseq = disequilibrium_at_n(N)
    diseq_frac = 0.01 + 0.19 * diseq
    lw = line_width_from_disequilibrium(adj, vertex=0,
                                         disequilibrium_frac=diseq_frac,
                                         n_trials=n_trials, seed=42)
    return lw['variance'], N, diseq


# ============================================================
# T1: Map cascade transition redshifts
# ============================================================

def test_T1_transition_redshifts():
    """T1: Find exact redshifts for cascade transitions (peaks) and settled states (troughs)."""
    print("\n  T1: Map cascade transition redshifts")

    a_clock, slope, rms = cascade_clock_fit(constrained=True)
    print(f"    Cascade clock: a={a_clock:.3f}, slope={slope:.4f}")

    peaks = []  # integer N
    troughs = []  # half-integer N

    print(f"\n    {'N':>6}  {'Type':>8}  {'z':>8}  {'t_look (Gyr)':>14}")
    print(f"    {'-'*42}")

    for n_target in [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
        z = z_for_cascade_level(n_target, a_clock)
        if z is not None:
            t_look = z_to_lookback(z)
            is_peak = n_target == int(n_target)
            label = "PEAK" if is_peak else "trough"
            entry = {'N': float(n_target), 'z': float(z), 't_lookback_gyr': float(t_look)}
            if is_peak:
                peaks.append(entry)
            else:
                troughs.append(entry)
            print(f"    {n_target:>6.1f}  {label:>8}  {z:>8.3f}  {t_look:>14.2f}")
        else:
            print(f"    {n_target:>6.1f}  {'---':>8}  {'N/A':>8}")

    correctly_ordered = all(
        peaks[i]['z'] < peaks[i+1]['z'] for i in range(len(peaks)-1)
    )
    has_enough = len(peaks) >= 4

    print(f"\n    Peaks found: {len(peaks)}, Troughs found: {len(troughs)}")
    print(f"    Correctly ordered by z: {correctly_ordered}")

    passed = correctly_ordered and has_enough
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T1_transition_redshifts',
        'a_clock': float(a_clock),
        'peaks': peaks,
        'troughs': troughs,
        'correctly_ordered': correctly_ordered,
        'PASS': passed,
    }


# ============================================================
# T2: High-resolution W(z) curve
# ============================================================

def test_T2_oscillation_curve(t1_result):
    """T2: Compute W(z) at high resolution showing oscillatory structure."""
    print("\n  T2: High-resolution width oscillation curve")

    a_clock = t1_result['a_clock']

    n_graph = 6
    adj = np.zeros((n_graph, n_graph))
    for i in range(n_graph - 1):
        adj[i, i+1] = adj[i+1, i] = 1.0

    z_array = np.linspace(0.05, 4.0, 150)
    w_data = []

    for z in z_array:
        width, N, diseq = width_at_z(z, a_clock, adj, n_trials=200)
        w_data.append({
            'z': float(z), 'width': float(width),
            'N': float(N), 'disequilibrium': float(diseq),
        })

    widths = np.array([d['width'] for d in w_data])
    N_values = np.array([d['N'] for d in w_data])

    # Count oscillation cycles by counting zero-crossings of the derivative
    dw = np.diff(widths)
    sign_changes = np.sum(np.diff(np.sign(dw)) != 0)
    n_cycles = sign_changes // 2
    print(f"    Points computed: {len(z_array)}")
    print(f"    Width range: [{np.min(widths):.6f}, {np.max(widths):.6f}]")
    print(f"    Sign changes in dW/dz: {sign_changes}")
    print(f"    Oscillation cycles: {n_cycles}")

    # Check peaks align with T1 transition redshifts
    peak_zs = [p['z'] for p in t1_result['peaks']]
    peak_hits = 0
    for pz in peak_zs:
        if pz < z_array[0] or pz > z_array[-1]:
            continue
        idx = np.argmin(np.abs(z_array - pz))
        local_max = (idx > 0 and idx < len(widths)-1 and
                     widths[idx] >= widths[idx-1] * 0.9 and
                     widths[idx] >= widths[idx+1] * 0.9)
        if local_max:
            peak_hits += 1

    n_testable_peaks = sum(1 for pz in peak_zs if z_array[0] <= pz <= z_array[-1])
    alignment_frac = peak_hits / max(n_testable_peaks, 1)
    print(f"    Peaks aligned with transitions: {peak_hits}/{n_testable_peaks}")

    has_cycles = n_cycles >= 3
    peaks_align = alignment_frac >= 0.5

    passed = has_cycles and peaks_align
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T2_oscillation_curve',
        'n_points': len(z_array),
        'n_cycles': int(n_cycles),
        'sign_changes': int(sign_changes),
        'peak_alignment': float(alignment_frac),
        'width_min': float(np.min(widths)),
        'width_max': float(np.max(widths)),
        'curve_data': w_data[::10],  # save every 10th point
        'PASS': passed,
    }


# ============================================================
# T3: Oscillatory vs monotonic discrimination
# ============================================================

def test_T3_discrimination(t1_result):
    """T3: Oscillatory model fits dramatically better than monotonic."""
    print("\n  T3: Oscillatory vs monotonic discrimination")

    a_clock = t1_result['a_clock']

    n_graph = 6
    adj = np.zeros((n_graph, n_graph))
    for i in range(n_graph - 1):
        adj[i, i+1] = adj[i+1, i] = 1.0

    z_array = np.linspace(0.1, 3.5, 80)
    widths = []
    N_values = []

    for z in z_array:
        w, N, _ = width_at_z(z, a_clock, adj, n_trials=200)
        widths.append(w)
        N_values.append(N)

    widths = np.array(widths)
    N_values = np.array(N_values)

    # Monotonic model: W = A * z^B
    log_w = np.log(widths + 1e-15)
    log_z = np.log(z_array)
    valid = widths > 1e-10
    if np.sum(valid) > 2:
        A_mat = np.vstack([np.ones(np.sum(valid)), log_z[valid]]).T
        coeffs_mono, _, _, _ = np.linalg.lstsq(A_mat, log_w[valid], rcond=None)
        mono_pred = np.exp(coeffs_mono[0] + coeffs_mono[1] * log_z)
        chi2_mono = float(np.sum((widths - mono_pred)**2))
    else:
        chi2_mono = float('inf')
        mono_pred = np.zeros_like(widths)

    # Oscillatory model: W = baseline * (1 + amp * cos(2*pi*N))
    # Grid search over amplitude
    best_chi2_osc = float('inf')
    best_amp = 0
    baseline = np.mean(widths)
    for amp in np.linspace(0.1, 2.0, 50):
        osc_pred = baseline * (1 + amp * np.cos(2 * PI * N_values))
        osc_pred = np.maximum(osc_pred, 0)
        chi2 = float(np.sum((widths - osc_pred)**2))
        if chi2 < best_chi2_osc:
            best_chi2_osc = chi2
            best_amp = amp

    osc_pred_best = baseline * (1 + best_amp * np.cos(2 * PI * N_values))
    osc_pred_best = np.maximum(osc_pred_best, 0)

    chi2_ratio = chi2_mono / max(best_chi2_osc, 1e-15)
    ratio_ok = chi2_ratio > 10

    # Autocorrelation of monotonic residuals
    residuals = widths - mono_pred
    residuals = residuals - np.mean(residuals)
    acf = np.correlate(residuals, residuals, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / max(acf[0], 1e-15)

    # Find first significant peak after lag 0
    peak_lag = 0
    for i in range(2, min(len(acf)-1, 40)):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:
            peak_lag = i
            break

    has_periodic = peak_lag > 0

    print(f"    Monotonic chi2: {chi2_mono:.4f}")
    print(f"    Oscillatory chi2: {best_chi2_osc:.4f} (amp={best_amp:.2f})")
    print(f"    Chi2 ratio (mono/osc): {chi2_ratio:.1f} (>10: {ratio_ok})")
    print(f"    Autocorrelation periodic peak at lag {peak_lag} ({has_periodic})")

    passed = ratio_ok
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T3_discrimination',
        'chi2_monotonic': chi2_mono,
        'chi2_oscillatory': best_chi2_osc,
        'chi2_ratio': float(chi2_ratio),
        'best_amplitude': float(best_amp),
        'autocorrelation_peak_lag': peak_lag,
        'PASS': passed,
    }


# ============================================================
# T4: Observational predictions
# ============================================================

def test_T4_predictions(t1_result):
    """T4: Concrete observational predictions — width contrasts at transition redshifts."""
    print("\n  T4: Observational predictions")

    a_clock = t1_result['a_clock']
    peaks = t1_result['peaks']
    troughs = t1_result['troughs']

    n_graph = 6
    adj = np.zeros((n_graph, n_graph))
    for i in range(n_graph - 1):
        adj[i, i+1] = adj[i+1, i] = 1.0

    # Compute widths at peaks and nearest troughs
    predictions = []
    for peak in peaks:
        w_peak, _, _ = width_at_z(peak['z'], a_clock, adj, n_trials=500)

        nearest_trough = min(troughs, key=lambda t: abs(t['z'] - peak['z']),
                            default=None)
        if nearest_trough:
            w_trough, _, _ = width_at_z(nearest_trough['z'], a_clock, adj, n_trials=500)
        else:
            w_trough = 1e-10

        contrast = w_peak / max(w_trough, 1e-10)

        # Observable spectral lines at this redshift
        lines = []
        z = peak['z']
        if z >= 1.7:
            lines.append('Lyman-alpha (1216A)')
        if 0.3 <= z <= 2.5:
            lines.append('MgII (2796A)')
        if 1.0 <= z <= 3.5:
            lines.append('CIV (1549A)')
        if z <= 1.0:
            lines.append('CaII K (3934A)')

        pred = {
            'cascade_level': float(peak['N']),
            'z_peak': float(peak['z']),
            'z_trough': float(nearest_trough['z']) if nearest_trough else None,
            'width_peak': float(w_peak),
            'width_trough': float(w_trough),
            'contrast': float(contrast),
            'observable_lines': lines,
        }
        predictions.append(pred)

        print(f"    N={peak['N']:.0f}: z_peak={peak['z']:.3f}, "
              f"contrast={contrast:.1f}x, lines={', '.join(lines) if lines else 'none'}")

    all_contrasts_high = all(p['contrast'] > 2.0 for p in predictions)
    self_consistent = all(p['width_peak'] > p['width_trough'] for p in predictions)

    print(f"\n    All contrasts > 2x: {all_contrasts_high}")
    print(f"    Self-consistent (peak > trough): {self_consistent}")

    # Summary prediction statement
    print(f"\n    === OBSERVATIONAL PREDICTIONS ===")
    for p in predictions:
        if p['z_trough']:
            print(f"    At z={p['z_peak']:.3f} (N={p['cascade_level']:.0f} transition), "
                  f"absorption lines should be {p['contrast']:.1f}x broader "
                  f"than at z={p['z_trough']:.3f}")

    passed = all_contrasts_high and self_consistent
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'T4_predictions',
        'predictions': predictions,
        'all_contrasts_high': all_contrasts_high,
        'self_consistent': self_consistent,
        'PASS': passed,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("exp_04: Cascade Width Oscillation Profile")
    print("Midnight Initiative, Thread 1 (Photon Archaeology)")
    print("=" * 70)

    t1 = test_T1_transition_redshifts()
    t2 = test_T2_oscillation_curve(t1)
    t3 = test_T3_discrimination(t1)
    t4 = test_T4_predictions(t1)

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 70}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 70}")

    data = {
        'experiment': 'exp_04_cascade_width_oscillation',
        'initiative': 'midnight',
        'thread': 'photon_archaeology',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'score': f"{score}/4",
        'n_pass': score,
        'n_total': 4,
    }

    save_midnight_results('exp_04_cascade_width_oscillation', _convert_numpy(data))
