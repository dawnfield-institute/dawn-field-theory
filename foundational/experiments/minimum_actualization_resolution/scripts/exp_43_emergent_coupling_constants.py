"""
Emergent Coupling Constants — Experiment Script 43

PURPOSE:
    Tests whether the coupling constants in the RBF equation, gravity, and mass
    generation converge to attractor values when derived per cell from field state
    rather than hardcoded.

    Following exp_42's confirmation that the actualization ratio f = E^2/(E^2+I^2)
    converges to ln(phi) as an attractor, we now test ALL coupling constants:

    1. alpha_local = (E^2+I^2) / (E^2+I^2+M^2)  — RBF collapse attraction
    2. lambda_local = M^2 / (E^2+I^2+M^2)        — RBF memory coupling
    3. G_local = M^2 / (M^2+(E-I)^2)              — gravitational coupling
    4. gamma_local = (E-I)^2 / (E^2+I^2)          — mass generation coefficient

    Note: alpha + lambda ~ 1 (they partition total field energy).

HYPOTHESIS:
    1. Each emergent coupling converges to a stable attractor value.
    2. The attractors are related to DFT constants (phi, Xi, ln(phi), etc).
    3. The system boils (nonzero variance) for all couplings.
    4. Convergence is grid-independent and IC-independent.

DESIGN:
    Part A — Track all 4 emergent couplings over 10000 ticks
    Part B — Boiling test: variance stays nonzero for all
    Part C — Grid independence
    Part D — Check relationships between attractor values and DFT constants

OUTPUT:
    Results saved to results/exp_43_emergent_coupling_constants.json

Simulation units (not Planck). PAC = E + I + M conserved.
"""

import json
import math
import os
import sys
import time
from datetime import datetime

# ============================================================
# Constants
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015328
XI = GAMMA_EM + LN_PHI
XI_PAC = 1 + (7 / 8) * LN2 * (1 - LN2) ** 2

# DFT constants to compare against
DFT_CONSTANTS = {
    'ln_phi': LN_PHI,           # 0.4812
    '1-ln_phi': 1 - LN_PHI,    # 0.5188
    'xi_pac': XI_PAC,           # 1.0571
    'xi_pac_inv': 1/XI_PAC,     # 0.9460
    'alpha_pac_config': 0.964,  # old hardcoded value
    'lambda_config': 0.020,     # old hardcoded value
    'mass_gen_config': 0.63,    # old hardcoded value
    'G_config': 0.15,           # old hardcoded value
    'phi_inv': 1/PHI,           # 0.6180
    'phi_inv_sq': 1/PHI**2,     # 0.3820
    'ln2': LN2,                 # 0.6931
    '1/phi^2': 1/(PHI**2),      # 0.3820
    'gamma_em': GAMMA_EM,       # 0.5772
}

# ============================================================
# Helpers
# ============================================================
def print_header(title, subtitle=None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70 + "\n")


def setup_engine():
    """Import and return Reality Engine v3 modules."""
    import torch
    re_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', '..', 'reality-engine'))
    if re_path not in sys.path:
        sys.path.insert(0, re_path)

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

    return {
        'torch': torch,
        'Engine': Engine,
        'SimulationConfig': SimulationConfig,
        'Pipeline': Pipeline,
        'operators': [
            RBFOperator, QBEOperator, ActualizationOperator,
            MemoryOperator, GravitationalCollapseOperator, FusionOperator,
            ConfluenceOperator, TemperatureOperator, ThermalNoiseOperator,
            NormalizationOperator, AdaptiveOperator, TimeEmergenceOperator,
        ],
    }


def build_engine(mods, nu=64, nv=64, dt=0.001, temperature=2.0):
    """Create and initialize a Reality Engine v3 instance."""
    torch = mods['torch']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = mods['SimulationConfig'](
        nu=nu, nv=nv, dt=dt, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )
    pipeline = mods['Pipeline']([cls() for cls in mods['operators']])
    engine = mods['Engine'](config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=temperature)
    return engine


def run_with_tracking(engine, ticks, sample_every=10):
    """Run engine for N ticks, sampling all emergent coupling metrics."""
    history = {
        'f_local_mean': [], 'f_local_std': [],
        'alpha_local_mean': [], 'alpha_local_std': [],
        'lambda_local_mean': [], 'lambda_local_std': [],
        'G_local_mean': [], 'G_local_std': [],
        'gamma_local_mean': [], 'gamma_local_std': [],
        'diseq': [], 'pac_drift': [],
        'ticks': [],
    }
    pac_initial = engine.state.pac_total

    for t in range(1, ticks + 1):
        engine.tick()
        if t % sample_every == 0:
            m = engine.state.metrics
            for key in history:
                if key == 'ticks':
                    history['ticks'].append(t)
                elif key == 'diseq':
                    history['diseq'].append(
                        (engine.state.E - engine.state.I).abs().mean().item())
                elif key == 'pac_drift':
                    history['pac_drift'].append(
                        engine.state.pac_total - pac_initial)
                else:
                    history[key].append(m.get(key, 0.0))

    return history


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def find_closest_constant(value, constants):
    """Find the DFT constant closest to the given value."""
    best_name, best_err = None, float('inf')
    for name, c in constants.items():
        if c == 0:
            continue
        err = abs(value - c) / abs(c)
        if err < best_err:
            best_name, best_err = name, err
    return best_name, best_err


# ============================================================
# Part A: Convergence of all emergent couplings
# ============================================================
def part_A(mods):
    print_header("Part A: Emergent Coupling Convergence",
                 "Track all couplings over 10000 ticks")

    engine = build_engine(mods, nu=64, nv=64)
    t0 = time.time()
    history = run_with_tracking(engine, ticks=10000, sample_every=50)
    elapsed = time.time() - t0

    n = len(history['ticks'])
    tail_start = int(n * 0.8)

    couplings = {
        'f_local': ('f_local_mean', 'f_local_std', 'actualization ratio'),
        'alpha': ('alpha_local_mean', 'alpha_local_std', 'collapse attraction'),
        'lambda': ('lambda_local_mean', 'lambda_local_std', 'memory coupling'),
        'G': ('G_local_mean', 'G_local_std', 'gravitational coupling'),
        'gamma': ('gamma_local_mean', 'gamma_local_std', 'mass generation'),
    }

    results = {}
    print(f"  Elapsed: {elapsed:.1f}s\n")
    print(f"  {'Coupling':15s} | {'Early':>8s} | {'Converged':>10s} | {'Std':>8s} | {'Closest DFT':>20s} | {'Err':>6s} | {'Drift':>8s}")
    print("  " + "-" * 90)

    for name, (mean_key, std_key, desc) in couplings.items():
        early = mean(history[mean_key][:int(n * 0.2)])
        tail = history[mean_key][tail_start:]
        tail_std_vals = history[std_key][tail_start:]
        converged = mean(tail)
        converged_std = mean(tail_std_vals)

        closest, err = find_closest_constant(converged, DFT_CONSTANTS)
        drift = converged - early

        results[name] = {
            'description': desc,
            'early': early,
            'converged': converged,
            'converged_std': converged_std,
            'closest_dft': closest,
            'closest_dft_value': DFT_CONSTANTS[closest],
            'closest_err_pct': err * 100,
            'drift': drift,
        }
        print(f"  {name:15s} | {early:8.4f} | {converged:10.6f} | {converged_std:8.4f} | {closest:>20s} | {err*100:5.1f}% | {drift:+8.4f}")

    # PAC conservation
    pac_drift = history['pac_drift'][-1] if history['pac_drift'] else 0.0
    print(f"\n  PAC drift: {pac_drift:.4e}")

    results['pac_drift'] = pac_drift
    results['history_summary'] = {
        'n_samples': n,
        'f_mean_last5': history['f_local_mean'][-5:],
        'alpha_mean_last5': history['alpha_local_mean'][-5:],
        'lambda_mean_last5': history['lambda_local_mean'][-5:],
        'G_mean_last5': history['G_local_mean'][-5:],
        'gamma_mean_last5': history['gamma_local_mean'][-5:],
    }

    return results


# ============================================================
# Part B: Boiling test — all couplings fluctuate
# ============================================================
def part_B(mods):
    print_header("Part B: Boiling Test",
                 "Do all couplings maintain nonzero variance?")

    engine = build_engine(mods, nu=64, nv=64)
    history = run_with_tracking(engine, ticks=10000, sample_every=50)

    n = len(history['ticks'])
    std_keys = ['f_local_std', 'alpha_local_std', 'lambda_local_std',
                'G_local_std', 'gamma_local_std']
    names = ['f_local', 'alpha', 'lambda', 'G', 'gamma']

    results = {}
    all_boiling = True
    print(f"  {'Coupling':12s} | {'Q1 std':>8s} | {'Q2 std':>8s} | {'Q3 std':>8s} | {'Q4 std':>8s} | {'Boiling':>8s}")
    print("  " + "-" * 65)

    for name, key in zip(names, std_keys):
        vals = history[key]
        q1 = mean(vals[:n//4])
        q2 = mean(vals[n//4:n//2])
        q3 = mean(vals[n//2:3*n//4])
        q4 = mean(vals[3*n//4:])
        min_std = min(q1, q2, q3, q4)
        is_boiling = min_std > 0.001
        if not is_boiling:
            all_boiling = False

        results[name] = {
            'quarters': [q1, q2, q3, q4],
            'min_std': min_std,
            'is_boiling': is_boiling,
        }
        status = "YES" if is_boiling else "NO"
        print(f"  {name:12s} | {q1:8.4f} | {q2:8.4f} | {q3:8.4f} | {q4:8.4f} | {status:>8s}")

    print(f"\n  All couplings boiling: {'YES' if all_boiling else 'NO'}")
    results['all_boiling'] = all_boiling
    return results


# ============================================================
# Part C: Grid independence
# ============================================================
def part_C(mods):
    print_header("Part C: Grid Independence",
                 "Do attractor values hold at different grid sizes?")

    grids = [(32, 32), (64, 64), (128, 32)]
    coupling_keys = ['f_local_mean', 'alpha_local_mean', 'lambda_local_mean',
                     'G_local_mean', 'gamma_local_mean']
    coupling_names = ['f_local', 'alpha', 'lambda', 'G', 'gamma']

    results = {}

    for nu, nv in grids:
        label = f"{nu}x{nv}"
        print(f"\n  Grid {label}...", end="", flush=True)
        t0 = time.time()
        engine = build_engine(mods, nu=nu, nv=nv)
        history = run_with_tracking(engine, ticks=5000, sample_every=50)
        elapsed = time.time() - t0

        n = len(history['ticks'])
        tail_start = int(n * 0.8)
        grid_results = {}
        parts = []
        for name, key in zip(coupling_names, coupling_keys):
            tail = history[key][tail_start:]
            conv = mean(tail)
            grid_results[name] = conv
            parts.append(f"{name}={conv:.4f}")
        results[label] = grid_results
        print(f" done ({elapsed:.1f}s) | {' '.join(parts)}")

    # Check spread across grids for each coupling
    print(f"\n  {'Coupling':12s} | {'Spread':>8s} | {'Grid-indep':>10s}")
    print("  " + "-" * 35)
    grid_independent = True
    for name in coupling_names:
        values = [results[g][name] for g in results]
        spread = max(values) - min(values)
        indep = spread < 0.05
        if not indep:
            grid_independent = False
        print(f"  {name:12s} | {spread:8.4f} | {'YES' if indep else 'NO':>10s}")

    results['grid_independent'] = grid_independent
    return results


# ============================================================
# Part D: Relationship to DFT constants
# ============================================================
def part_D(converged_values):
    print_header("Part D: DFT Constant Relationships",
                 "How do attractor values relate to known DFT constants?")

    print(f"  {'Coupling':12s} | {'Value':>10s} | {'Closest DFT':>20s} | {'DFT Value':>10s} | {'Error':>6s}")
    print("  " + "-" * 70)

    for name, info in converged_values.items():
        if name in ('pac_drift', 'history_summary'):
            continue
        val = info['converged']
        closest = info['closest_dft']
        dft_val = info['closest_dft_value']
        err = info['closest_err_pct']
        print(f"  {name:12s} | {val:10.6f} | {closest:>20s} | {dft_val:10.6f} | {err:5.1f}%")

    # Check alpha + lambda ~ 1
    alpha_conv = converged_values.get('alpha', {}).get('converged', 0)
    lambda_conv = converged_values.get('lambda', {}).get('converged', 0)
    partition_sum = alpha_conv + lambda_conv
    print(f"\n  alpha + lambda = {partition_sum:.6f} (should be ~1.0, err = {abs(1 - partition_sum)*100:.2f}%)")

    # Check if f_local ~ ln(phi)
    f_conv = converged_values.get('f_local', {}).get('converged', 0)
    print(f"  f_local deviation from ln(phi): {f_conv - LN_PHI:+.6f} ({abs(f_conv - LN_PHI)/LN_PHI*100:.1f}%)")

    return {
        'alpha_plus_lambda': partition_sum,
        'f_deviation_from_ln_phi': f_conv - LN_PHI,
    }


# ============================================================
# Main
# ============================================================
def main():
    print_header("Experiment 43: Emergent Coupling Constants",
                 "All physics parameters emerge from field state")

    print(f"  DFT Reference Constants:")
    for name, val in sorted(DFT_CONSTANTS.items()):
        print(f"    {name:25s} = {val:.10f}")

    mods = setup_engine()

    results = {
        'experiment': 'exp_43_emergent_coupling_constants',
        'timestamp': datetime.now().isoformat(),
        'dft_constants': DFT_CONSTANTS,
    }

    results['part_A'] = part_A(mods)
    results['part_B'] = part_B(mods)
    results['part_C'] = part_C(mods)
    results['part_D'] = part_D(results['part_A'])

    # ============================================================
    # Synthesis
    # ============================================================
    print_header("SYNTHESIS")

    # Count how many couplings converge to within 20% of a DFT constant
    n_close = 0
    for name, info in results['part_A'].items():
        if name in ('pac_drift', 'history_summary'):
            continue
        if info.get('closest_err_pct', 100) < 20:
            n_close += 1

    all_boiling = results['part_B']['all_boiling']
    grid_indep = results['part_C']['grid_independent']

    print(f"  Couplings within 20% of DFT constant: {n_close}/5")
    print(f"  All couplings boiling:                {'YES' if all_boiling else 'NO'}")
    print(f"  Grid independent:                     {'YES' if grid_indep else 'NO'}")
    print(f"  alpha + lambda:                       {results['part_D']['alpha_plus_lambda']:.6f}")

    if n_close >= 3 and all_boiling:
        verdict = f"CONFIRMED: {n_close}/5 couplings converge near DFT constants, all boiling"
    elif n_close >= 2:
        verdict = f"PARTIAL: {n_close}/5 near DFT constants"
    else:
        verdict = f"NOT CONFIRMED: only {n_close}/5 near DFT constants"

    print(f"\n  VERDICT: {verdict}")

    results['synthesis'] = {
        'n_close_to_dft': n_close,
        'all_boiling': all_boiling,
        'grid_independent': grid_indep,
        'verdict': verdict,
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"exp_43_emergent_coupling_constants_{ts}.json")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
