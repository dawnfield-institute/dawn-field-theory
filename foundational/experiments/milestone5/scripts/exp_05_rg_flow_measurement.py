#!/usr/bin/env python3
"""
Milestone 5 - Exp 05: Renormalization Group Flow Measurement
============================================================

The tiling filter creates a scale-dependent coupling (exp_04 result).
Now we measure the EFFECTIVE beta function: how does the coupling RUN
with scale in the simulator?

In QCD (one-loop): beta(g) = -b0 * g^3 / (16*pi^2)
  where b0 = 11 - 2*nf/3 = 7 (for nf=6)

In DFT: the tiling filter gives g(k) = (k/k_max)^gamma_eff
  where gamma_eff = 1.6123 (from Xi, phi, ln^2(2))
  So: dg/d(ln k) = gamma_eff * g   [constant coefficient!]

This is a POWER-LAW running, not logarithmic like QCD. But maybe the
measured coupling has corrections from field dynamics?

This experiment:
1. Run simulator to multiple equilibrium points at DIFFERENT GRID SIZES
   (equivalent to different UV cutoffs = different renormalization scales)
2. Measure coupling constants at each scale
3. Extract the beta function: beta(g) = dg/d(ln(mu)) where mu ~ k_max
4. Compare to QCD's beta function
5. Check for asymptotic freedom (does coupling decrease at high energy?)
"""

import os, sys, json, time, math
import numpy as np
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__))
_ws   = os.path.join(_here, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_ws, 'reality-engine'))

import torch
from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
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
PHI    = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2    = math.log(2)
LN2_SQ = LN2**2
GAMMA_EM = 0.5772156649
XI     = GAMMA_EM + LN_PHI
_EPS   = 1e-8

TARGETS = {
    'f_local':      LN_PHI,
    'gamma_local':  1.0 / PHI,
    'alpha_local':  LN2,
    'G_local':      1.0 / (PHI * PHI),
    'lambda_local': 1.0 - LN2,
}

# QCD reference
ALPHA_S_PDG = 0.1179
ALPHA_S_MZ  = 0.1179   # at M_Z = 91.2 GeV
B0_QCD = 7.0            # one-loop beta coefficient (nf=6)


def build_pipeline():
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
    return Pipeline(ops)


def measure_couplings(state):
    E, I, M = state.E, state.I, state.M
    E2, I2, M2 = E**2, I**2, M**2
    total = E2 + I2 + M2 + _EPS
    diseq2 = (E - I)**2

    f = (E2 / (E2 + I2 + _EPS)).mean().item()
    gamma_l = (I2 / (I2 + M2 + _EPS)).mean().item()
    alpha_l = (total - M2).mean().item() / total.mean().item()

    G_mass = M2 / (M2 + diseq2 + _EPS)
    xi_s = I2 / (E2 + _EPS)
    xi_pow = xi_s.pow(1.0 / PHI)
    xi_mod = (xi_pow / (xi_pow + 1.0)).sqrt()
    G = (G_mass * xi_mod).mean().item()

    lam = (M2 / total).mean().item()

    return {'f_local': f, 'gamma_local': gamma_l, 'alpha_local': alpha_l,
            'G_local': G, 'lambda_local': lam, 'M_mean': M.mean().item()}


def measure_spectral_coupling(state, n_bins=20):
    """Measure the effective spectral coupling g(k) = M_power / (M_power + D_power)."""
    E, I, M = state.E, state.I, state.M
    nu, nv = M.shape

    M_fft = torch.fft.fft2(M.double())
    M_power = M_fft.real**2 + M_fft.imag**2
    D = (E - I).double()
    D_fft = torch.fft.fft2(D)
    D_power = D_fft.real**2 + D_fft.imag**2

    coupling_spectrum = M_power / (M_power + D_power + _EPS)

    ku = torch.arange(nu, device=M.device, dtype=torch.float64)
    kv = torch.arange(nv, device=M.device, dtype=torch.float64)
    ku = torch.where(ku > nu // 2, ku - nu, ku)
    kv = torch.where(kv > nv // 2, kv - nv, kv)
    ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')
    k_mag = torch.sqrt(ku_grid**2 + kv_grid**2)
    k_max = math.sqrt((nu // 2)**2 + (nv // 2)**2)

    k_flat = k_mag.flatten().cpu().numpy()
    c_flat = coupling_spectrum.flatten().cpu().numpy()

    k_bins = np.linspace(0.5, k_max, n_bins + 1)
    k_centers = []
    g_values = []

    for j in range(len(k_bins) - 1):
        mask = (k_flat >= k_bins[j]) & (k_flat < k_bins[j+1])
        if mask.sum() > 0:
            k_centers.append((k_bins[j] + k_bins[j+1]) / 2)
            g_values.append(c_flat[mask].mean())

    return np.array(k_centers), np.array(g_values), k_max


def pct_err(val, target):
    return abs(val - target) / abs(target) * 100.0


def run_at_scale(nu, nv, device, ticks=3000):
    """Run the simulator at a specific grid size and measure couplings."""
    config = SimulationConfig(
        nu=nu, nv=nv, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    for tick in range(1, ticks + 1):
        engine.tick()

    couplings = measure_couplings(engine.state)
    k_centers, g_values, k_max = measure_spectral_coupling(engine.state)

    return couplings, k_centers, g_values, k_max


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*110}")
    print(f"  MILESTONE 5 -- EXP 05: RENORMALIZATION GROUP FLOW MEASUREMENT")
    print(f"  Device: {device}")
    print(f"  Method: Run at multiple grid sizes (UV cutoffs), measure coupling vs scale")
    print(f"{'='*110}")

    # ============================================================================
    # PART 1: Multi-scale coupling measurement
    # Different grid sizes = different UV cutoffs = different renormalization scales
    # ============================================================================

    print(f"\n  {'='*90}")
    print(f"  PART 1: COUPLING CONSTANTS vs GRID SIZE (UV CUTOFF)")
    print(f"  {'='*90}")

    # Grid sizes from small (low resolution = IR) to large (high resolution = UV)
    # Keep aspect ratio ~2:1 to match Mobius topology
    scales = [
        (32, 16),
        (48, 24),
        (64, 32),
        (96, 48),
        (128, 64),
        (192, 96),
    ]

    results = {}
    for nu, nv in scales:
        k_max = math.sqrt((nu//2)**2 + (nv//2)**2)
        print(f"\n  Grid {nu}x{nv} (k_max={k_max:.1f}) ...", end=" ", flush=True)
        t0 = time.time()
        couplings, k_centers, g_values, km = run_at_scale(nu, nv, device, ticks=3000)
        elapsed = time.time() - t0

        results[(nu, nv)] = {
            'couplings': couplings,
            'k_centers': k_centers.tolist(),
            'g_values': g_values.tolist(),
            'k_max': km,
            'elapsed': elapsed,
        }

        errs = {k: pct_err(couplings[k], TARGETS[k]) for k in TARGETS}
        avg = sum(errs.values()) / len(errs)
        print(f"{elapsed:.0f}s  avg_err={avg:.1f}%  "
              f"f={couplings['f_local']:.4f}({errs['f_local']:.1f}%)  "
              f"G={couplings['G_local']:.4f}({errs['G_local']:.1f}%)  "
              f"M={couplings['M_mean']:.4f}")

    # ============================================================================
    # PART 2: Coupling flow with scale
    # ============================================================================

    print(f"\n  {'='*90}")
    print(f"  PART 2: COUPLING FLOW (how couplings change with UV cutoff)")
    print(f"  {'='*90}")

    print(f"\n  {'Grid':>10}  {'k_max':>6}  {'ln(k_max)':>9}  "
          f"{'f':>8}  {'gamma':>8}  {'alpha':>8}  {'G':>8}  {'lambda':>8}  {'avg_err':>8}")

    scale_data = []
    for (nu, nv), data in results.items():
        c = data['couplings']
        km = data['k_max']
        errs = {k: pct_err(c[k], TARGETS[k]) for k in TARGETS}
        avg = sum(errs.values()) / len(errs)
        scale_data.append({
            'nu': nu, 'nv': nv, 'k_max': km, 'ln_k_max': math.log(km),
            **c, 'avg_err': avg
        })
        print(f"  {nu}x{nv:>3}  {km:6.1f}  {math.log(km):9.4f}  "
              f"{c['f_local']:8.4f}  {c['gamma_local']:8.4f}  {c['alpha_local']:8.4f}  "
              f"{c['G_local']:8.4f}  {c['lambda_local']:8.4f}  {avg:8.1f}%")

    # ============================================================================
    # PART 3: Beta function extraction
    # ============================================================================

    print(f"\n  {'='*90}")
    print(f"  PART 3: BETA FUNCTIONS (dg/d ln(k_max))")
    print(f"  {'='*90}")

    for coupling_name in ['f_local', 'gamma_local', 'alpha_local', 'G_local', 'lambda_local']:
        target = TARGETS[coupling_name]
        values = [d[coupling_name] for d in scale_data]
        ln_k = [d['ln_k_max'] for d in scale_data]

        print(f"\n  {coupling_name} (target={target:.6f}):")
        print(f"    {'ln(k_max)':>10}  {'g(k)':>10}  {'g-target':>10}  {'dg/dlnk':>10}")

        for i in range(len(values)):
            delta = values[i] - target
            if i > 0:
                beta = (values[i] - values[i-1]) / (ln_k[i] - ln_k[i-1])
            else:
                beta = float('nan')
            print(f"    {ln_k[i]:10.4f}  {values[i]:10.6f}  {delta:+10.6f}  {beta:10.6f}")

        # Linear fit to extract overall beta
        if len(values) >= 3:
            coeffs = np.polyfit(ln_k, values, 1)
            print(f"    Linear fit: g = {coeffs[0]:.6f} * ln(k_max) + {coeffs[1]:.6f}")
            print(f"    Overall beta = dg/dlnk = {coeffs[0]:.6f}")
            # At what scale does the fit predict g = target?
            if abs(coeffs[0]) > 1e-6:
                ln_k_target = (target - coeffs[1]) / coeffs[0]
                print(f"    g = target when ln(k_max) = {ln_k_target:.4f} (k_max = {math.exp(ln_k_target):.1f})")

    # ============================================================================
    # PART 4: Spectral coupling profiles across scales
    # ============================================================================

    print(f"\n  {'='*90}")
    print(f"  PART 4: SPECTRAL COUPLING PROFILES (rescaled by k_max)")
    print(f"  {'='*90}")

    # For each grid, plot g(k/k_max) — if physics is scale-invariant,
    # these should collapse onto a universal curve
    print(f"\n  Do spectral profiles collapse when rescaled by k_max?")
    print(f"  {'k/k_max':>10}", end="")
    for (nu, nv) in list(results.keys()):
        print(f"  {nu}x{nv:>3}", end="")
    print()

    # Sample at normalized k/k_max = [0.1, 0.2, ..., 0.9]
    norm_k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for nk in norm_k:
        print(f"  {nk:10.1f}", end="")
        for (nu, nv), data in results.items():
            kc = np.array(data['k_centers'])
            gv = np.array(data['g_values'])
            km = data['k_max']
            # Interpolate to find g(nk * k_max)
            target_k = nk * km
            if len(kc) > 1 and target_k >= kc[0] and target_k <= kc[-1]:
                g_interp = np.interp(target_k, kc, gv)
            else:
                g_interp = float('nan')
            print(f"  {g_interp:7.4f}", end="")
        print()

    # ============================================================================
    # PART 5: Asymptotic freedom check
    # ============================================================================

    print(f"\n  {'='*90}")
    print(f"  PART 5: ASYMPTOTIC FREEDOM CHECK")
    print(f"  {'='*90}")

    # In QCD, alpha_s DECREASES at higher energies (larger k_max)
    # Does our gravity coupling (G_local) show this behavior?
    G_values = [d['G_local'] for d in scale_data]
    k_max_values = [d['k_max'] for d in scale_data]

    # G should approach 1/phi^2 = 0.382 as scale increases
    G_target = 1.0 / PHI**2
    print(f"  G_local target (DFT): {G_target:.6f}")
    print(f"  G_local values by scale:")
    for km, G in zip(k_max_values, G_values):
        err = pct_err(G, G_target)
        trend = "converging" if err < 10 else "not yet"
        print(f"    k_max={km:6.1f}: G={G:.6f} ({err:.1f}% err) [{trend}]")

    # Check overall monotonicity (is the coupling running in the right direction?)
    monotone_up = all(G_values[i] <= G_values[i+1] for i in range(len(G_values)-1))
    monotone_down = all(G_values[i] >= G_values[i+1] for i in range(len(G_values)-1))
    if monotone_up:
        print(f"  G_local is MONOTONICALLY INCREASING with scale (anti-screening)")
    elif monotone_down:
        print(f"  G_local is MONOTONICALLY DECREASING with scale (asymptotic freedom!)")
    else:
        print(f"  G_local is NON-MONOTONIC — no simple asymptotic behavior")

    # Compare to QCD running at one loop
    print(f"\n  QCD one-loop comparison (alpha_s(Q) = alpha_s(M_Z) / (1 + b0*alpha_s*ln(Q/M_Z)/(2*pi))):")
    for km in k_max_values:
        Q_ratio = km / k_max_values[-1]  # normalize to largest scale
        if Q_ratio > 0:
            alpha_qcd = ALPHA_S_MZ / (1 + B0_QCD * ALPHA_S_MZ * math.log(Q_ratio) / (2 * math.pi))
            print(f"    k_max={km:6.1f}: alpha_QCD = {alpha_qcd:.6f}")

    # ============================================================================
    # SAVE
    # ============================================================================

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_here, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_05_rg_flow_{ts}.json")

    save_data = {
        'experiment': 'exp_05_rg_flow_measurement',
        'timestamp': ts,
        'scales': [{'nu': nu, 'nv': nv, **data['couplings'],
                     'k_max': data['k_max']}
                    for (nu, nv), data in results.items()],
        'scale_data': scale_data,
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
