#!/usr/bin/env python3
"""
DFT Milestone 5 — Experiment 10: Gravity xi_mod Variants

The gravity operator already has entropy-coherence modulation:
  xi_s = I^2 / (E^2 + eps)
  xi_mod = sqrt(xi_s^(1/phi) / (xi_s^(1/phi) + 1))
  G_local = G_mass * xi_mod

This experiment tests whether the current xi_mod is optimal by comparing
5 variants of the gravity coupling formula:

  A) Current (sqrt + phi-scaled sigmoid)
  B) No xi_mod (G_local = G_mass only — raw mass dominance)
  C) Simple sigmoid: xi_s / (xi_s + 1)
  D) Stronger modulation: xi_s^2 / (xi_s^2 + 1)
  E) Asymmetric: boost coherence, xi_s^2 / (xi_s^2 + 0.5)

If B (no xi_mod) is better, the modulation is harmful.
If A is best, the current formula is already optimal.
If C/D/E win, the modulation shape matters and needs tuning.
"""

import json, math, os, sys, time, copy
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__))
_ws   = os.path.join(_here, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_ws, 'reality-engine'))

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.sec_tracking import SECTrackingOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.protocol import Pipeline

# ── DFT targets ────────────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
TARGETS = {
    "f_local":     LN_PHI,              # 0.4812
    "gamma_local": 1.0 / PHI,           # 0.6180
    "alpha_local": math.log(2),          # 0.6931
    "G_local":     1.0 / PHI**2,         # 0.3820
    "lambda_local": 1.0 - math.log(2),   # 0.3069
}
_EPS = 1e-12


# ── Gravity variants ──────────────────────────────────────────────

class GravityNoXiMod(GravitationalCollapseOperator):
    """Variant B: No xi_mod — G_local = G_mass only."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_local = M2 / (M2 + diseq2 + _EPS)  # No xi_mod

        phi = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = (M_new - M_candidate)
        pac_leak = mass_created * 0.5
        E_new = state.E - pac_leak
        I_new = state.I - pac_leak

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["xi_mod_mean"] = 1.0  # no modulation
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class GravitySimpleSigmoid(GravitationalCollapseOperator):
    """Variant C: Simple sigmoid xi_s / (xi_s + 1)."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_mass = M2 / (M2 + diseq2 + _EPS)

        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        xi_mod = xi_s / (xi_s + 1.0)  # simple sigmoid
        G_local = G_mass * xi_mod

        phi = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = (M_new - M_candidate)
        pac_leak = mass_created * 0.5
        E_new = state.E - pac_leak
        I_new = state.I - pac_leak

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class GravityStronger(GravitationalCollapseOperator):
    """Variant D: Stronger modulation xi_s^2 / (xi_s^2 + 1)."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_mass = M2 / (M2 + diseq2 + _EPS)

        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        xi_s2 = xi_s.pow(2)
        xi_mod = xi_s2 / (xi_s2 + 1.0)  # sharper sigmoid
        G_local = G_mass * xi_mod

        phi = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = (M_new - M_candidate)
        pac_leak = mass_created * 0.5
        E_new = state.E - pac_leak
        I_new = state.I - pac_leak

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class GravityAsymmetric(GravitationalCollapseOperator):
    """Variant E: Asymmetric — xi_s^2 / (xi_s^2 + 0.5), favors coherence."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_mass = M2 / (M2 + diseq2 + _EPS)

        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        xi_s2 = xi_s.pow(2)
        xi_mod = xi_s2 / (xi_s2 + 0.5)  # shifted sigmoid — more gravity for same xi_s
        G_local = G_mass * xi_mod

        phi = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = (M_new - M_candidate)
        pac_leak = mass_created * 0.5
        E_new = state.E - pac_leak
        I_new = state.I - pac_leak

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ── Measurement ────────────────────────────────────────────────────

def measure_couplings(state):
    E, I, M = state.E, state.I, state.M
    E2 = E.pow(2).mean().item()
    I2 = I.pow(2).mean().item()
    M2 = M.pow(2).mean().item()
    diseq2 = (E - I).pow(2).mean().item()

    f = E2 / (E2 + I2 + _EPS)
    gamma = I2 / (I2 + M2 + _EPS)
    alpha = E2 / (E2 + M2 + _EPS)
    G = M2 / (M2 + diseq2 + _EPS)
    lam = diseq2 / (E2 + I2 + M2 + _EPS)

    return {"f_local": f, "gamma_local": gamma, "alpha_local": alpha,
            "G_local": G, "lambda_local": lam}


def grade(err):
    if err < 2: return "A"
    if err < 5: return "A-"
    if err < 10: return "B"
    if err < 15: return "C"
    if err < 25: return "D"
    return "F"


# ── Pipeline builder ───────────────────────────────────────────────

def build_pipeline(grav_operator):
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        ActualizationOperator(),
        MemoryOperator(),
        PhiCascadeOperator(),
        grav_operator,
        SpinStatisticsOperator(),
        ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        NormalizationOperator(),
        SECTrackingOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])


# ── Run variant ────────────────────────────────────────────────────

def run_variant(name, grav_operator, grid=(128, 64), ticks=5000, device="cuda"):
    cfg = SimulationConfig(nu=grid[0], nv=grid[1], device=device,
                           dt=0.001, enable_actualization=True,
                           actualization_threshold=0.05)
    pipe = build_pipeline(grav_operator)
    torch.manual_seed(42)
    eng = Engine(config=cfg, pipeline=pipe)
    eng.initialize("big_bang", temperature=3.0)

    history = []
    t0 = time.time()

    for tick in range(1, ticks + 1):
        eng.tick()
        if tick % 500 == 0:
            st = eng.state
            c = measure_couplings(st)
            errs = {}
            for k, v in c.items():
                errs[k + "_err"] = abs(v - TARGETS[k]) / TARGETS[k] * 100
            avg_err = sum(errs.values()) / len(TARGETS)

            EI_ratio = st.E.pow(2).mean().item() / (st.I.pow(2).mean().item() + _EPS)

            record = {"tick": tick, "EI_ratio": EI_ratio, "avg_err": avg_err, **c, **errs}
            history.append(record)
            elapsed = time.time() - t0
            print(f"  [{name}] tick {tick:5d}  EI={EI_ratio:.4f}  avg={avg_err:.1f}%  [{elapsed:.0f}s]")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = (128, 64)
    ticks = 5000
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 90)
    print("  DFT Experiment 10: Gravity xi_mod Variants")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks")
    print("=" * 90)

    variants = {
        "A_current":     GravitationalCollapseOperator(),
        "B_no_ximod":    GravityNoXiMod(),
        "C_simple_sig":  GravitySimpleSigmoid(),
        "D_stronger":    GravityStronger(),
        "E_asymmetric":  GravityAsymmetric(),
    }

    all_results = {}
    for vname, grav_op in variants.items():
        print(f"\n--- Variant {vname} ---")
        history = run_variant(vname, grav_op, grid=grid, ticks=ticks, device=device)
        all_results[vname] = history

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  FINAL STATE COMPARISON (tick {ticks})")
    print("=" * 90)
    print(f"  {'Variant':<16s} {'avg':>6s} {'f':>8s} {'gamma':>8s} {'alpha':>8s} {'G':>8s} {'lambda':>8s} {'E/I':>8s}")
    print("  " + "-" * 78)

    best_avg = 999
    best_name = ""
    for vname, history in all_results.items():
        f = history[-1]
        fe = f["f_local_err"]
        ge = f["gamma_local_err"]
        ae = f["alpha_local_err"]
        Ge = f["G_local_err"]
        le = f["lambda_local_err"]
        avg = f["avg_err"]
        eir = f["EI_ratio"]

        print(f"  {vname:<16s} {avg:>5.1f}% {f['f_local']:.3f}{grade(fe):>2s} {f['gamma_local']:.3f}{grade(ge):>2s} {f['alpha_local']:.3f}{grade(ae):>2s} {f['G_local']:.3f}{grade(Ge):>2s} {f['lambda_local']:.3f}{grade(le):>2s} {eir:>7.3f}")

        if avg < best_avg:
            best_avg = avg
            best_name = vname

    print(f"\n  BEST: {best_name} (avg_err = {best_avg:.1f}%)")

    # ── E/I and error evolution ────────────────────────────────────
    print("\n" + "=" * 90)
    print("  E/I RATIO EVOLUTION")
    print("=" * 90)
    tshow = [500, 1000, 2000, 3000, 5000]
    header = f"  {'tick':>6s}"
    for v in all_results:
        header += f"  {v:>14s}"
    print(header)
    for t in tshow:
        line = f"  {t:>6d}"
        for v, h in all_results.items():
            rec = next((r for r in h if r["tick"] == t), None)
            line += f"  {rec['EI_ratio']:>14.4f}" if rec else f"  {'n/a':>14s}"
        print(line)

    print("\n" + "=" * 90)
    print("  AVG ERROR EVOLUTION (%)")
    print("=" * 90)
    print(header)
    for t in tshow:
        line = f"  {t:>6d}"
        for v, h in all_results.items():
            rec = next((r for r in h if r["tick"] == t), None)
            line += f"  {rec['avg_err']:>13.1f}%" if rec else f"  {'n/a':>14s}"
        print(line)

    # ── Save ───────────────────────────────────────────────────────
    results_dir = os.path.join(_here, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, f"exp_10_gravity_ximod_{ts}.json")
    save_data = {
        "experiment": "exp_10_gravity_ximod_variants",
        "date": ts,
        "grid": list(grid),
        "ticks": ticks,
        "targets": TARGETS,
        "variants": all_results,
        "best_variant": best_name,
        "best_avg_err": best_avg,
    }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {outpath}")


if __name__ == "__main__":
    main()
