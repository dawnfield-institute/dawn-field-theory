#!/usr/bin/env python3
"""
Milestone 5 - Exp 03: Strong Force as Coupling Modulation
=========================================================

KEY INSIGHT from exp_01 and exp_02:
  - exp_01 (spectral mass enhancement): fixes gamma but breaks G. C3 > C2.
  - exp_02 (binding operators): all WORSE than baseline. C2 > C3 (reversal!).
  - Problem: both approaches ADD mass redistribution that fights gravity.

NEW APPROACH: Don't move mass. Modulate EXISTING operator parameters.

The RBF operator has:
    B = lap(E-I) + lambda_local * M * lap(M) - alpha_local * ||E-I||^2 - gamma * (E-I)

Where gamma (damping) is the ONLY global scalar. Everything else is already per-cell.

Strong force = confinement = reduced dissipation in bound states.
  -> gamma_local = gamma * (1 - alpha_s * confinement_field)
  -> confinement_field = high where mass gradients are steep (mass clumps bound together)

Three modulation strategies:
  A) Baseline (no strong force)
  B) GammaModulation: spatially vary gamma via mass gradient magnitude
  C) ThresholdModulation: spatially vary actualization threshold via local mass density
  D) CombinedModulation: both gamma + threshold modulation
  E) GammaModulation with C2 (for C3 vs C2 discrimination with new mechanism)
"""

import os, sys, json, time, math
from datetime import datetime

# --- path setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_ws   = os.path.join(_here, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_ws, 'reality-engine'))

import torch
from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state  import FieldState
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
PHI  = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2  = math.log(2)
GAMMA_EM = 0.5772156649
XI   = GAMMA_EM + LN_PHI
_EPS = 1e-8

# DFT targets
TARGETS = {
    'f_local':      LN_PHI,
    'gamma_local':  1.0 / PHI,
    'alpha_local':  LN2,
    'G_local':      1.0 / (PHI * PHI),
    'lambda_local': 1.0 - LN2,
}

# strong coupling candidates
ALPHA_S_C2 = 0.118239  # C2 correction (n=3, fundamental)
ALPHA_S_C3 = 0.117214  # C3 correction (n=8, adjoint)
ALPHA_S_PDG = 0.1179

# ---- operator wrappers that modulate existing parameters ----

class GammaModulationOperator:
    """
    Modulates the RBF damping rate gamma spatially based on mass gradient magnitude.

    Physics: confinement = reduced dissipation in bound states.
    Where mass gradients are steep (boundaries of bound structures), damping is reduced,
    allowing disequilibrium to persist and structure to form.

    gamma_local(x) = gamma_base * (1 - alpha_s * strength * confinement(x))
    confinement(x) = |grad(M)|^2 / (|grad(M)|^2 + M^2 + eps)  [normalized 0-1]
    """
    def __init__(self, alpha_s: float, strength: float = 1.0, label: str = ""):
        self.alpha_s = alpha_s
        self.strength = strength
        self.label = label

    def __call__(self, state, config, bus=None):
        M = state.M
        # compute mass gradient magnitude (periodic boundaries)
        grad_u = torch.roll(M, -1, 0) - torch.roll(M, 1, 0)
        grad_v = torch.roll(M, -1, 1) - torch.roll(M, 1, 1)
        grad_mag_sq = grad_u**2 + grad_v**2

        # confinement field: normalized gradient magnitude
        # high where mass gradients are steep (edges of bound structures)
        confinement = grad_mag_sq / (grad_mag_sq + M**2 + _EPS)

        # modulation factor: reduce damping where confinement is high
        # gamma_mod in [1 - alpha_s*strength, 1]
        gamma_mod = 1.0 - self.alpha_s * self.strength * confinement
        gamma_mod = gamma_mod.clamp(min=0.1)  # never negative or zero damping

        # store modulation in config for RBF to pick up
        # We inject into state metrics so RBF can read it
        metrics = dict(state.metrics) if state.metrics else {}
        metrics['gamma_modulation'] = gamma_mod
        metrics['confinement_mean'] = confinement.mean().item()
        metrics['confinement_max'] = confinement.max().item()
        metrics['gamma_mod_mean'] = gamma_mod.mean().item()

        return state.replace(metrics=metrics)


class ThresholdModulationOperator:
    """
    Modulates the actualization threshold based on local mass density.

    Physics: strong force lowers the barrier to mass formation in dense regions
    (easier to create quark-antiquark pairs where color field is strong).

    threshold_mod(x) = 1 - alpha_s * strength * density(x)
    density(x) = M^2 / (M^2 + (E^2 + I^2) + eps)  [normalized 0-1]
    """
    def __init__(self, alpha_s: float, strength: float = 1.0, label: str = ""):
        self.alpha_s = alpha_s
        self.strength = strength
        self.label = label

    def __call__(self, state, config, bus=None):
        M = state.M
        E = state.E
        I = state.I

        # mass density field (normalized)
        M2 = M**2
        density = M2 / (M2 + E**2 + I**2 + _EPS)

        # threshold modulation: lower threshold where mass is dense
        thresh_mod = 1.0 - self.alpha_s * self.strength * density
        thresh_mod = thresh_mod.clamp(min=0.3)  # never reduce by more than 70%

        metrics = dict(state.metrics) if state.metrics else {}
        metrics['threshold_modulation'] = thresh_mod
        metrics['density_mean'] = density.mean().item()
        metrics['thresh_mod_mean'] = thresh_mod.mean().item()

        return state.replace(metrics=metrics)


class ModulatedRBFOperator:
    """
    Wrapper around RBFOperator that applies gamma modulation from state metrics.

    If gamma_modulation is present in metrics, applies it as a per-cell scaling
    of the damping term. Otherwise behaves identically to base RBF.
    """
    def __init__(self, base_rbf: RBFOperator):
        self.base_rbf = base_rbf

    def __call__(self, state, config, bus=None):
        gamma_mod = None
        if state.metrics and 'gamma_modulation' in state.metrics:
            gamma_mod = state.metrics['gamma_modulation']

        if gamma_mod is None:
            # no modulation, use base RBF directly
            return self.base_rbf(state, config, bus)

        # Apply modulation: temporarily scale gamma in config
        # The RBF operator reads config.gamma_damping as a scalar.
        # We can't easily make it per-cell through config, so we modify the
        # damping term in the RBF output by post-correction.

        # Run base RBF to get the standard B field
        new_state = self.base_rbf(state, config, bus)

        # The RBF computes: B = lap(E-I) + lambda*M*lap(M) - alpha*||E-I||^2 - gamma*(E-I)
        # The damping contribution is: -gamma * (E-I)
        # We want: -gamma * gamma_mod * (E-I)
        # So the correction is: -gamma * (gamma_mod - 1) * (E-I)
        # Applied to E: dE_correction = -gamma * (gamma_mod - 1) * (E-I) * dt

        gamma = config.gamma_damping
        dt = config.dt
        disequilibrium = state.E - state.I

        # correction to make damping spatially varying
        correction = gamma * (1.0 - gamma_mod) * disequilibrium * dt
        # positive correction where gamma_mod < 1 (less damping -> more E retained)

        E_new = new_state.E + correction
        I_new = new_state.I - correction  # PAC conservation

        return new_state.replace(E=E_new, I=I_new)


class ModulatedActualizationOperator:
    """
    Wrapper around ActualizationOperator that applies threshold modulation.

    If threshold_modulation is in metrics, scales the actualization threshold per-cell.
    """
    def __init__(self, base_act: ActualizationOperator):
        self.base_act = base_act

    def __call__(self, state, config, bus=None):
        thresh_mod = None
        if state.metrics and 'threshold_modulation' in state.metrics:
            thresh_mod = state.metrics['threshold_modulation']

        if thresh_mod is None:
            return self.base_act(state, config, bus)

        # Scale the threshold: lower it where thresh_mod < 1
        # We modify P (potential) to effectively lower the threshold:
        # Instead of |P| > threshold, we check |P/thresh_mod| > threshold
        # Which is equivalent to boosting P by 1/thresh_mod

        P_boosted = state.P / thresh_mod.clamp(min=0.3)
        boosted_state = state.replace(P=P_boosted)

        new_state = self.base_act(boosted_state, config, bus)

        return new_state


# ---- experiment infrastructure ----

def build_pipeline(gamma_op=None, thresh_op=None):
    """Build the full 16-operator pipeline with optional modulation wrappers."""
    rbf_base = RBFOperator()
    act_base = ActualizationOperator()

    # Phase 1: RBF (with optional gamma modulation)
    if gamma_op is not None:
        ops = [gamma_op, ModulatedRBFOperator(rbf_base)]
    else:
        ops = [rbf_base]

    ops.append(QBEOperator())

    # Phase 2: Actualization (with optional threshold modulation)
    if thresh_op is not None:
        ops.extend([thresh_op, ModulatedActualizationOperator(act_base)])
    else:
        ops.append(act_base)

    # Phase 3: rest of pipeline (matches exp_01 exactly)
    ops.extend([
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])

    return Pipeline(ops)


def measure_couplings(state):
    """Extract coupling constants from field state."""
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


def pct_err(val, target):
    return abs(val - target) / abs(target) * 100.0


def grade(err):
    if err < 3:  return 'A'
    if err < 10: return 'B'
    if err < 20: return 'C'
    if err < 30: return 'D'
    return 'F'


def run_variant(name, device, gamma_op=None, thresh_op=None,
                grid=(128, 64), ticks=5000, sample_every=500):
    """Run one variant and return sampled coupling history."""
    pipe = build_pipeline(gamma_op, thresh_op)
    cfg  = SimulationConfig(nu=grid[0], nv=grid[1], device=device,
                            dt=0.001, enable_actualization=True,
                            actualization_threshold=0.05)
    torch.manual_seed(42)
    eng  = Engine(config=cfg, pipeline=pipe)
    eng.initialize("big_bang", temperature=3.0)

    history = []
    t0 = time.time()
    for t in range(1, ticks + 1):
        eng.tick()
        if t % sample_every == 0:
            c = measure_couplings(eng.state)
            c['tick'] = t
            history.append(c)
    elapsed = time.time() - t0
    return history, elapsed


def print_evolution(name, history):
    print(f"\n  {name}:")
    print(f"      {'tick':>5}  {'f_local':>12}  {'gamma':>12}  {'alpha':>12}  {'G_local':>12}  {'lambda':>12}  {'avg_err':>8}  {'M_mean':>8}")
    for h in history:
        errs = [pct_err(h[k], TARGETS[k]) for k in TARGETS]
        avg = sum(errs) / len(errs)
        parts = []
        for k in ['f_local', 'gamma_local', 'alpha_local', 'G_local', 'lambda_local']:
            e = pct_err(h[k], TARGETS[k])
            parts.append(f"{h[k]:7.4f}({e:4.1f}%)")
        print(f"      {h['tick']:5d}  {'  '.join(parts)}  {avg:6.1f}%  {h['M_mean']:7.4f}")


# ---- main ----

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid   = (128, 64)
    ticks  = 5000

    print(f"\n{'='*110}")
    print(f"  MILESTONE 5 -- EXP 03: STRONG FORCE AS COUPLING MODULATION")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks per variant")
    print(f"  Approach: modulate existing operator parameters, not mass redistribution")
    print(f"{'='*110}")

    variants = {}

    # A: baseline
    print(f"\n  [A_baseline] ...", end=" ", flush=True)
    h, t = run_variant("A_baseline", device)
    variants["A_baseline"] = h
    print(f"{t:.0f}s")

    # B: gamma modulation with C3
    print(f"  [B_gammaMod_C3] ...", end=" ", flush=True)
    g_op = GammaModulationOperator(ALPHA_S_C3, strength=1.0, label="C3_gamma")
    h, t = run_variant("B_gammaMod_C3", device, gamma_op=g_op)
    variants["B_gammaMod_C3"] = h
    print(f"{t:.0f}s")

    # C: threshold modulation with C3
    print(f"  [C_threshMod_C3] ...", end=" ", flush=True)
    t_op = ThresholdModulationOperator(ALPHA_S_C3, strength=1.0, label="C3_thresh")
    h, t = run_variant("C_threshMod_C3", device, thresh_op=t_op)
    variants["C_threshMod_C3"] = h
    print(f"{t:.0f}s")

    # D: combined modulation with C3
    print(f"  [D_combined_C3] ...", end=" ", flush=True)
    g_op2 = GammaModulationOperator(ALPHA_S_C3, strength=1.0, label="C3_gamma")
    t_op2 = ThresholdModulationOperator(ALPHA_S_C3, strength=1.0, label="C3_thresh")
    h, t = run_variant("D_combined_C3", device, gamma_op=g_op2, thresh_op=t_op2)
    variants["D_combined_C3"] = h
    print(f"{t:.0f}s")

    # E: gamma modulation with C2 (for discrimination)
    print(f"  [E_gammaMod_C2] ...", end=" ", flush=True)
    g_op3 = GammaModulationOperator(ALPHA_S_C2, strength=1.0, label="C2_gamma")
    h, t = run_variant("E_gammaMod_C2", device, gamma_op=g_op3)
    variants["E_gammaMod_C2"] = h
    print(f"{t:.0f}s")

    # --- print evolution tables ---
    print(f"\n{'='*110}")
    print(f"  COUPLING EVOLUTION (errors vs DFT targets)")
    print(f"{'='*110}")
    for name, hist in variants.items():
        print_evolution(name, hist)

    # --- final comparison ---
    print(f"\n{'='*110}")
    print(f"  FINAL STATE (tick {ticks})")
    print(f"{'='*110}")

    best_name, best_err = None, 999
    final_data = {}

    for name, hist in variants.items():
        h = hist[-1]
        errs = {k: pct_err(h[k], TARGETS[k]) for k in TARGETS}
        avg = sum(errs.values()) / len(errs)

        # stability: std of avg_err over last 3 samples
        last3 = hist[-3:]
        avgs_last3 = []
        for s in last3:
            e = [pct_err(s[k], TARGETS[k]) for k in TARGETS]
            avgs_last3.append(sum(e)/len(e))
        stab = max(avgs_last3) - min(avgs_last3) if len(avgs_last3) > 1 else 0

        # drift: avg_err at tick 5000 vs tick 2000
        if len(hist) >= 4:
            mid = hist[3]  # tick 2000
            mid_errs = [pct_err(mid[k], TARGETS[k]) for k in TARGETS]
            mid_avg = sum(mid_errs) / len(mid_errs)
            drift = avg - mid_avg
        else:
            drift = 0

        g_f = grade(errs['f_local'])
        g_g = grade(errs['gamma_local'])
        g_a = grade(errs['alpha_local'])
        g_G = grade(errs['G_local'])
        g_l = grade(errs['lambda_local'])

        print(f"  {name:22s}  avg={avg:5.1f}%  "
              f"f={errs['f_local']:5.1f}%({g_f})  "
              f"g={errs['gamma_local']:5.1f}%({g_g})  "
              f"a={errs['alpha_local']:5.1f}%({g_a})  "
              f"G={errs['G_local']:5.1f}%({g_G})  "
              f"L={errs['lambda_local']:5.1f}%({g_l})  "
              f"stab={stab:.2f}%  drift={drift:+.1f}%")

        final_data[name] = {
            'couplings': h,
            'errors': errs,
            'avg_err': avg,
            'stability': stab,
            'drift': drift,
        }

        if avg < best_err:
            best_err = avg
            best_name = name

    print(f"\n  BEST: {best_name} (avg_err = {best_err:.1f}%)")

    # C3 vs C2 comparison (gamma modulation)
    b_err = final_data["B_gammaMod_C3"]["avg_err"]
    e_err = final_data["E_gammaMod_C2"]["avg_err"]
    b_errs = final_data["B_gammaMod_C3"]["errors"]
    e_errs = final_data["E_gammaMod_C2"]["errors"]

    c3_wins = sum(1 for k in TARGETS if b_errs[k] < e_errs[k])
    c2_wins = 5 - c3_wins
    winner = "C3 (adjoint)" if c3_wins > c2_wins else "C2 (fundamental)"
    print(f"\n  C3 vs C2 (gamma modulation): C3 {c3_wins}/5, C2 {c2_wins}/5")
    print(f"  {winner} wins in modulation mode")

    # --- improvement over baseline ---
    base_err = final_data["A_baseline"]["avg_err"]
    print(f"\n  Improvement over baseline ({base_err:.1f}%):")
    for name, fd in final_data.items():
        if name == "A_baseline":
            continue
        delta = fd["avg_err"] - base_err
        direction = "BETTER" if delta < 0 else "WORSE"
        print(f"    {name:22s}: {delta:+5.1f}% ({direction})")

    # --- save results ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_here, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_03_modulation_{ts}.json")

    save_data = {
        'experiment': 'exp_03_coupling_modulation',
        'timestamp': ts,
        'grid': list(grid),
        'ticks': ticks,
        'device': str(device),
        'alpha_s_C2': ALPHA_S_C2,
        'alpha_s_C3': ALPHA_S_C3,
        'approach': 'coupling_modulation_not_mass_redistribution',
        'variants': {},
    }
    for name, hist in variants.items():
        save_data['variants'][name] = {
            'history': hist,
            'final': final_data[name],
        }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
