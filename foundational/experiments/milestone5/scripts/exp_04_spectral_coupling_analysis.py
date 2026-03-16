#!/usr/bin/env python3
"""
Milestone 5 - Exp 04: Spectral Coupling Analysis
=================================================

KEY INSIGHT from exp_01-03:
  Adding the strong force as a perturbation doesn't work:
  - Spectral mass enhancement trades gamma for G
  - Binding operators fight gravity
  - Parameter modulation is too subtle

NEW HYPOTHESIS: The strong force is already IMPLICIT in the gravity operator's
cascade-depth tiling filter. The filter creates scale-dependent coupling:
  suppression(k) = (ln^2(2))^(Xi * n(k))
  n(k) = log_phi(k_max / |k|)

This means:
  - High |k| (short range): coupling ~ 1 (strong!)
  - Low |k| (long range): coupling ~ 0 (weak!)

That's exactly the running coupling of QCD: alpha_s(Q) is large at low Q (= high |k|
in position space) and falls logarithmically at high Q.

This experiment:
1. Compute the tiling filter's effective coupling profile g(k)
2. Compare g(k) to QCD's running coupling alpha_s(Q)
3. Identify the "confinement scale" where g(k) transitions
4. Check if the transition scale encodes alpha_s = 0.1179
5. Run the simulator and measure the ACTUAL spectral coupling from field dynamics
"""

import os, sys, json, time, math
import numpy as np
from datetime import datetime

# --- path setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_ws   = os.path.join(_here, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_ws, 'reality-engine'))

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.sec_tracking import SECTrackingOperator

# ---- constants ----
PHI   = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2   = math.log(2)
LN2_SQ = LN2**2
GAMMA_EM = 0.5772156649
XI    = GAMMA_EM + LN_PHI
_EPS  = 1e-12

# DFT targets
TARGETS = {
    'f_local':      LN_PHI,
    'gamma_local':  1.0 / PHI,
    'alpha_local':  LN2,
    'G_local':      1.0 / (PHI * PHI),
    'lambda_local': 1.0 - LN2,
}

# strong coupling
ALPHA_S_PDG = 0.1179
ALPHA_S_C3  = 0.117214


# ============================================================================
# PART 1: Analytical tiling filter profile
# ============================================================================

def analyze_tiling_filter(nu=128, nv=64):
    """Compute and analyze the gravity operator's spectral tiling filter."""
    print(f"\n  {'='*90}")
    print(f"  PART 1: TILING FILTER SPECTRAL PROFILE (analytical)")
    print(f"  Grid: {nu}x{nv}")
    print(f"  {'='*90}")

    # Build the filter (same code as gravity operator)
    ku = np.arange(nu, dtype=np.float64)
    kv = np.arange(nv, dtype=np.float64)
    ku = np.where(ku > nu // 2, ku - nu, ku)
    kv = np.where(kv > nv // 2, kv - nv, kv)
    ku_grid, kv_grid = np.meshgrid(ku, kv, indexing='ij')

    k_mag = np.sqrt(ku_grid**2 + kv_grid**2)
    k_max = math.sqrt((nu // 2)**2 + (nv // 2)**2)

    k_safe = np.maximum(k_mag, 1.0)
    cascade_depth = np.log(k_max / k_safe) / LN_PHI

    log_suppression = XI * cascade_depth * math.log(LN2_SQ)
    tiling_filter = np.exp(log_suppression)
    tiling_filter[0, 0] = 0.0

    # Radial profile: bin by |k| and average the filter
    k_flat = k_mag.flatten()
    f_flat = tiling_filter.flatten()

    # Create radial bins
    k_bins = np.linspace(0.5, k_max, 50)
    k_centers = []
    f_means = []

    for i in range(len(k_bins) - 1):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if mask.sum() > 0:
            k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            f_means.append(f_flat[mask].mean())

    k_centers = np.array(k_centers)
    f_means = np.array(f_means)

    # Print the profile
    print(f"\n  Radial tiling filter profile:")
    print(f"  {'|k|':>8}  {'n(k)':>8}  {'filter':>10}  {'coupling':>10}  {'bar':>40}")
    for kc, fm in zip(k_centers, f_means):
        n_k = math.log(k_max / max(kc, 1)) / LN_PHI
        bar_len = int(fm * 40)
        bar = '#' * bar_len
        print(f"  {kc:8.1f}  {n_k:8.2f}  {fm:10.6f}  {fm:10.6f}  {bar}")

    # Key scale analysis
    print(f"\n  Key scales:")
    print(f"    k_max = {k_max:.1f} (Nyquist)")
    print(f"    k_max/phi = {k_max/PHI:.1f} (first cascade level)")
    print(f"    k_max/phi^2 = {k_max/PHI**2:.1f} (second cascade level)")

    # Find the "confinement scale" where filter = alpha_s
    # g(k_conf) = alpha_s means: exp(Xi * n(k_conf) * ln(LN2_SQ)) = alpha_s
    # -> n(k_conf) = ln(alpha_s) / (Xi * ln(LN2_SQ))
    # -> k_conf = k_max / phi^n(k_conf)
    n_conf = math.log(ALPHA_S_PDG) / (XI * math.log(LN2_SQ))
    k_conf = k_max / PHI**n_conf
    print(f"\n  Confinement scale (where filter = alpha_s = {ALPHA_S_PDG}):")
    print(f"    n(k_conf) = {n_conf:.4f}")
    print(f"    k_conf = {k_conf:.2f}")
    print(f"    k_conf/k_max = {k_conf/k_max:.4f}")

    # What fraction of modes are "confined" (filter < alpha_s)?
    confined = (tiling_filter.flatten() < ALPHA_S_PDG).sum()
    total_modes = tiling_filter.size
    print(f"    Confined modes: {confined}/{total_modes} ({100*confined/total_modes:.1f}%)")

    # Find where filter = 0.5 (transition scale)
    n_half = math.log(0.5) / (XI * math.log(LN2_SQ))
    k_half = k_max / PHI**n_half
    print(f"\n  Transition scale (filter = 0.5):")
    print(f"    n(k_half) = {n_half:.4f}")
    print(f"    k_half = {k_half:.2f}")
    print(f"    k_half/k_max = {k_half/k_max:.4f}")

    # What is the filter value at k = 1 (longest wavelength)?
    f_at_k1 = math.exp(XI * math.log(k_max) / LN_PHI * math.log(LN2_SQ))
    print(f"\n  Filter at k=1 (longest mode): {f_at_k1:.2e}")
    print(f"  Filter at k=k_max: 1.0 (by construction)")

    # The effective coupling profile IS the running coupling!
    # g(k) = tiling_filter(k)
    # In QCD: alpha_s(Q) ~ 1/ln(Q/Lambda_QCD) (one-loop)
    # In DFT: g(k) = exp(Xi * log_phi(k_max/k) * ln(ln^2(2)))
    #        = exp(-0.773 * log_phi(k_max/k))  [since Xi*ln(ln^2(2)) = 1.058*(-0.731) = -0.773]

    coeff = XI * math.log(LN2_SQ)
    print(f"\n  Running coupling formula:")
    print(f"    g(k) = exp({coeff:.4f} * log_phi(k_max/k))")
    print(f"    = (k/k_max)^({coeff/LN_PHI:.4f})  [power law form]")
    print(f"    = (k/k_max)^({-coeff/LN_PHI:.4f})  [effective exponent, positive for decay]")

    # Compare to QCD one-loop: alpha_s(Q) = alpha_s(M_Z) / (1 + b*alpha_s*ln(Q/M_Z))
    # Our form is: g(k) = (k/k_max)^gamma_eff where gamma_eff = -Xi*ln(LN2_SQ)/LN_PHI
    gamma_eff = -coeff / LN_PHI
    print(f"\n  Effective running exponent: {gamma_eff:.4f}")
    print(f"  QCD one-loop beta coefficient b0 = 11 - 2nf/3 = {11 - 2*6/3:.1f} (for nf=6)")
    print(f"  QCD running: alpha_s(Q) ~ 1/(b0 * ln(Q/Lambda))")
    print(f"  DFT running: g(k) ~ (k/k_max)^{gamma_eff:.3f}")

    return {
        'k_centers': k_centers.tolist(),
        'filter_profile': f_means.tolist(),
        'k_max': k_max,
        'k_conf': k_conf,
        'k_half': k_half,
        'n_conf': n_conf,
        'gamma_eff': gamma_eff,
        'coeff': coeff,
    }


# ============================================================================
# PART 2: Measure effective coupling from running simulator
# ============================================================================

def measure_spectral_coupling(device, ticks=3000):
    """Run the simulator and measure the actual effective coupling at each scale."""
    print(f"\n  {'='*90}")
    print(f"  PART 2: MEASURED SPECTRAL COUPLING FROM FIELD DYNAMICS")
    print(f"  Running {ticks} ticks...")
    print(f"  {'='*90}")

    grid = (128, 64)
    config = SimulationConfig(
        nu=grid[0], nv=grid[1], dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)

    ops = [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ]
    pipeline = Pipeline(ops)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Run to equilibrium-ish
    checkpoints = [500, 1000, 2000, 3000]
    spectral_data = {}

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick in checkpoints:
            state = engine.state
            E, I, M = state.E, state.I, state.M

            # Spectral analysis of mass field
            M_fft = torch.fft.fft2(M.double())
            M_power = (M_fft.real**2 + M_fft.imag**2)

            # Spectral analysis of E-I disequilibrium
            D = (E - I).double()
            D_fft = torch.fft.fft2(D)
            D_power = (D_fft.real**2 + D_fft.imag**2)

            # Effective coupling at each scale:
            # How much mass power is there relative to disequilibrium power?
            # This is analogous to G_local = M^2 / (M^2 + (E-I)^2) but in spectral domain
            coupling_spectrum = M_power / (M_power + D_power + _EPS)

            # Compute |k| grid
            nu, nv = M.shape
            ku = torch.arange(nu, device=M.device, dtype=torch.float64)
            kv = torch.arange(nv, device=M.device, dtype=torch.float64)
            ku = torch.where(ku > nu // 2, ku - nu, ku)
            kv = torch.where(kv > nv // 2, kv - nv, kv)
            ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')
            k_mag = torch.sqrt(ku_grid**2 + kv_grid**2)
            k_max = math.sqrt((nu // 2)**2 + (nv // 2)**2)

            # Radial binning
            k_flat = k_mag.flatten().cpu().numpy()
            c_flat = coupling_spectrum.flatten().cpu().numpy()
            mp_flat = M_power.flatten().cpu().numpy()
            dp_flat = D_power.flatten().cpu().numpy()

            k_bins = np.linspace(0.5, k_max, 30)
            k_ctr = []
            c_avg = []
            mp_avg = []
            dp_avg = []

            for j in range(len(k_bins) - 1):
                mask = (k_flat >= k_bins[j]) & (k_flat < k_bins[j+1])
                if mask.sum() > 0:
                    k_ctr.append((k_bins[j] + k_bins[j+1]) / 2)
                    c_avg.append(c_flat[mask].mean())
                    mp_avg.append(mp_flat[mask].mean())
                    dp_avg.append(dp_flat[mask].mean())

            spectral_data[tick] = {
                'k': k_ctr,
                'coupling': c_avg,
                'M_power': mp_avg,
                'D_power': dp_avg,
            }

            elapsed = time.time() - t0
            print(f"    tick {tick}: {elapsed:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"  Total: {total_time:.0f}s")

    # Print results
    for tick, data in spectral_data.items():
        print(f"\n  Spectral coupling at tick {tick}:")
        print(f"  {'|k|':>8}  {'g_eff(k)':>10}  {'M_power':>12}  {'D_power':>12}  {'bar':>30}")
        for kc, gc, mp, dp in zip(data['k'], data['coupling'], data['M_power'], data['D_power']):
            bar_len = int(gc * 30)
            bar = '#' * bar_len
            print(f"  {kc:8.1f}  {gc:10.6f}  {mp:12.4e}  {dp:12.4e}  {bar}")

    return spectral_data


# ============================================================================
# PART 3: Compare analytical filter to measured coupling
# ============================================================================

def compare_analytical_vs_measured(analytical, measured, ticks=[1000, 3000]):
    """Compare the tiling filter profile to measured spectral coupling."""
    print(f"\n  {'='*90}")
    print(f"  PART 3: ANALYTICAL FILTER vs MEASURED COUPLING")
    print(f"  {'='*90}")

    a_k = np.array(analytical['k_centers'])
    a_f = np.array(analytical['filter_profile'])

    for tick in ticks:
        if tick not in measured:
            continue
        m_k = np.array(measured[tick]['k'])
        m_c = np.array(measured[tick]['coupling'])

        # Interpolate analytical to measured k values
        a_interp = np.interp(m_k, a_k, a_f)

        print(f"\n  Tick {tick}:")
        print(f"  {'|k|':>8}  {'tiling':>10}  {'measured':>10}  {'ratio':>10}  {'diff':>10}")
        for kc, af, mc in zip(m_k, a_interp, m_c):
            ratio = mc / (af + 1e-12)
            diff = mc - af
            print(f"  {kc:8.1f}  {af:10.6f}  {mc:10.6f}  {ratio:10.3f}  {diff:+10.6f}")

        # Correlation
        valid = (a_interp > 0.001) & (m_c > 0.001)
        if valid.sum() > 3:
            corr = np.corrcoef(np.log(a_interp[valid] + 1e-12),
                               np.log(m_c[valid] + 1e-12))[0, 1]
            print(f"\n  Log-log correlation: {corr:.4f}")

        # Where does measured coupling = alpha_s?
        for i in range(len(m_k) - 1):
            if m_c[i] > ALPHA_S_PDG >= m_c[i+1]:
                k_cross = m_k[i] + (m_k[i+1] - m_k[i]) * (m_c[i] - ALPHA_S_PDG) / (m_c[i] - m_c[i+1])
                print(f"  Measured confinement scale (g=alpha_s): k = {k_cross:.1f}")
                break


# ============================================================================
# PART 4: Test if alpha_s emerges from spectral ratio
# ============================================================================

def test_alpha_s_emergence(analytical):
    """Check if alpha_s can be derived from the tiling filter's spectral structure."""
    print(f"\n  {'='*90}")
    print(f"  PART 4: DOES alpha_s EMERGE FROM THE SPECTRAL STRUCTURE?")
    print(f"  {'='*90}")

    k_max = analytical['k_max']
    gamma_eff = analytical['gamma_eff']

    # The tiling filter at any scale k is: g(k) = (k/k_max)^gamma_eff
    # Key question: is there a NATURAL scale where g(k) = alpha_s?

    # Hypothesis 1: alpha_s = filter at k = phi (the fundamental cascade scale)
    g_at_phi = (PHI / k_max)**gamma_eff
    print(f"\n  H1: alpha_s = g(k=phi) = ({PHI:.3f}/{k_max:.1f})^{gamma_eff:.4f}")
    print(f"      g(phi) = {g_at_phi:.6f}  (PDG: {ALPHA_S_PDG}, err: {abs(g_at_phi-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 2: alpha_s = filter at k = 1/phi (reciprocal cascade scale)
    g_at_inv_phi = (1.0/PHI / k_max)**gamma_eff if k_max > 0 else 0
    print(f"\n  H2: alpha_s = g(k=1/phi)")
    print(f"      g(1/phi) = {g_at_inv_phi:.6f}  (err: {abs(g_at_inv_phi-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 3: alpha_s = integral of filter over one phi-period
    # i.e., average coupling over one cascade level
    # integral from k/phi to k of g(k')/k' dk' normalized
    # For g(k) = (k/kmax)^gamma: integral = kmax^(-gamma)/(gamma+1) * [k^(gamma+1) - (k/phi)^(gamma+1)]
    # Normalized: avg_coupling_per_level = [1 - phi^(-gamma-1)] / [1 - phi^(-1)] * g(k)
    # At k=k_max: avg = [1 - phi^(-gamma-1)] / [1 - 1/phi]
    factor = (1 - PHI**(-(gamma_eff+1))) / (1 - 1/PHI)
    print(f"\n  H3: alpha_s = avg coupling over one phi-period at k_max")
    print(f"      avg = {factor:.6f}  (err: {abs(factor-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 4: alpha_s is the ratio of adjacent cascade levels
    # g(k/phi) / g(k) = phi^(-gamma_eff) (constant ratio!)
    ratio = PHI**(-gamma_eff)
    print(f"\n  H4: alpha_s = g(k/phi)/g(k) = phi^(-gamma_eff)")
    print(f"      ratio = {ratio:.6f}  (err: {abs(ratio-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 5: alpha_s = ln^2(2)^Xi (the tiling cost per unit cascade depth)
    tiling_unit_cost = LN2_SQ**XI
    print(f"\n  H5: alpha_s = ln^2(2)^Xi = {LN2_SQ:.6f}^{XI:.5f}")
    print(f"      = {tiling_unit_cost:.6f}  (err: {abs(tiling_unit_cost-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 6: alpha_s = exp(Xi * ln(ln^2(2))) (the e-folding rate)
    e_fold = math.exp(XI * math.log(LN2_SQ))
    print(f"\n  H6: alpha_s = exp(Xi * ln(ln^2(2)))")
    print(f"      = {e_fold:.6f}  (err: {abs(e_fold-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # Hypothesis 7: alpha_s = the filter value at the geometric mean scale
    # k_geo = sqrt(1 * k_max) = sqrt(k_max)
    k_geo = math.sqrt(k_max)
    g_at_geo = (k_geo / k_max)**gamma_eff
    print(f"\n  H7: alpha_s = g(sqrt(k_max)) = g({k_geo:.2f})")
    print(f"      = {g_at_geo:.6f}  (err: {abs(g_at_geo-ALPHA_S_PDG)/ALPHA_S_PDG*100:.1f}%)")

    # What cascade depth n gives alpha_s?
    n_for_alpha_s = math.log(ALPHA_S_PDG) / (XI * math.log(LN2_SQ))
    k_for_alpha_s = k_max / PHI**n_for_alpha_s
    print(f"\n  Exact: g(k) = alpha_s when:")
    print(f"    n(k) = {n_for_alpha_s:.4f} cascade levels")
    print(f"    k = {k_for_alpha_s:.2f} = k_max/phi^{n_for_alpha_s:.4f}")
    print(f"    k/k_max = {k_for_alpha_s/k_max:.6f}")

    return {
        'g_at_phi': g_at_phi,
        'ratio_adjacent': ratio,
        'tiling_unit_cost': tiling_unit_cost,
        'e_fold': e_fold,
        'n_for_alpha_s': n_for_alpha_s,
        'k_for_alpha_s': k_for_alpha_s,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*110}")
    print(f"  MILESTONE 5 -- EXP 04: SPECTRAL COUPLING ANALYSIS")
    print(f"  Device: {device}")
    print(f"  Question: Is the strong force already implicit in the tiling filter?")
    print(f"{'='*110}")

    # Part 1: Analytical tiling filter
    analytical = analyze_tiling_filter(128, 64)

    # Part 2: Measured spectral coupling from simulator
    measured = measure_spectral_coupling(device, ticks=3000)

    # Part 3: Compare
    compare_analytical_vs_measured(analytical, measured)

    # Part 4: Does alpha_s emerge?
    emergence = test_alpha_s_emergence(analytical)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_here, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_04_spectral_{ts}.json")

    save_data = {
        'experiment': 'exp_04_spectral_coupling_analysis',
        'timestamp': ts,
        'analytical': analytical,
        'measured': {str(k): v for k, v in measured.items()},
        'emergence': emergence,
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
