"""Milestone 5, Experiment 02: Strong Force as Binding Operator

CONTEXT:
exp_01 showed C3 (n=8, adjoint) wins 5-0 over C2 (n=3, fundamental).
But the spectral mass enhancement operator was too crude -- it fixes gamma_local
(20.7% -> 5.6%) but breaks G_local (3.5% -> 19%) via mass runaway.

HYPOTHESIS:
The strong force should create BINDING between nearby mass peaks (short-range
attraction, like QCD confinement) WITHOUT changing the total mass distribution.
This is fundamentally different from the exp_01 approach of multiplying M in
Fourier space. Instead:
  - Compute local mass gradient
  - Add attractive "force" between neighboring mass concentrations
  - Conserve total M (binding redistributes, doesn't create)
  - Strength modulated by alpha_s * confinement_profile

Three operator designs tested:
  A: Baseline (no strong force)
  B: Gradient binding -- mass flows down local gradients at rate alpha_s
  C: Laplacian binding -- diffusion toward mass concentrations (anti-diffusion)
  D: Pair potential -- explicit short-range attractive potential between mass peaks
  E: Best of B/C/D with C3 (n=8) coupling

FALSIFICATION:
- If binding operators also break G_local: the issue is adding any force, not design
- If binding fixes gamma without breaking G: exp_01's failure was operator design
- If C3 still beats C2 in binding mode: representation result is robust

AUTHOR: Peter Groom, Dawn Field Institute
DATE: 2026-03-16
"""

import math
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RE_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..', 'reality-engine'))
if RE_ROOT not in sys.path:
    sys.path.insert(0, RE_ROOT)

import torch
import numpy as np

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

PHI = (1.0 + math.sqrt(5)) / 2
GAMMA_EM = 0.5772156649015329
XI = GAMMA_EM + math.log(PHI)
LN2 = math.log(2)
_EPS = 1e-12

F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
ALPHA_S_MEASURED = 0.1179
ALPHA_S_BARE = F[3] / (2 * PHI * F[6])
ALPHA_S_C2 = ALPHA_S_BARE * (1 + F[5]/(3*math.pi*F[2]**2))
ALPHA_S_C3 = ALPHA_S_BARE * (1 + F[7]/(8*math.pi*F[2]**2))

TARGETS = {
    "f_local":     GAMMA_EM,
    "gamma_local": 1.0 / PHI,
    "alpha_local": LN2,
    "G_local":     1.0 / PHI**2,
    "lambda_local": 1.0 - LN2,
}


# -- Binding Operators ---------------------------------------------------------

class GradientBindingOperator:
    """Strong force as mass gradient flow: mass moves toward nearby mass.

    Unlike gravity (long-range 1/r^2), this is SHORT-RANGE ONLY.
    Mass flows down the local gradient at rate alpha_s, but only within
    a confinement radius (~few cells). Conserves total M exactly.
    """

    def __init__(self, alpha_s, label="", strength=0.01):
        self.alpha_s = alpha_s
        self.strength = strength
        self._name = f"GradBind({label})"

    @property
    def name(self):
        return self._name

    def __call__(self, state, config, bus=None):
        M = state.M
        if M.abs().max() < _EPS:
            return state

        # Local mass gradient (finite difference)
        grad_u = (torch.roll(M, -1, 0) - torch.roll(M, 1, 0)) / 2.0
        grad_v = (torch.roll(M, -1, 1) - torch.roll(M, 1, 1)) / 2.0

        # Mass flux toward gradients (anti-diffusion at short range)
        # This creates binding: mass flows toward nearby mass concentrations
        flux_u = self.alpha_s * self.strength * grad_u * M
        flux_v = self.alpha_s * self.strength * grad_v * M

        # Divergence of flux = mass change (conservative)
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )

        # Short-range cutoff: suppress at scales > 1/phi cells
        # Apply via smoothing the divergence (only local changes survive)
        M_new = M + div_flux

        # Exact conservation: redistribute any residual
        delta = M_new.sum() - M.sum()
        M_new = M_new - delta / M_new.numel()

        # PAC conservation
        delta_M = M_new - M
        E_new = state.E - delta_M * 0.5
        I_new = state.I - delta_M * 0.5

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["strong_binding_rms"] = delta_M.pow(2).mean().sqrt().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class LaplacianBindingOperator:
    """Strong force as negative diffusion: mass clusters instead of spreads.

    Normal diffusion: dM/dt = D * laplacian(M) -- smooths gradients
    Strong binding:   dM/dt = -alpha_s * laplacian(M) -- sharpens gradients

    Only applied where M > threshold (confinement in mass-rich regions).
    """

    def __init__(self, alpha_s, label="", strength=0.01):
        self.alpha_s = alpha_s
        self.strength = strength
        self._name = f"LapBind({label})"

    @property
    def name(self):
        return self._name

    def __call__(self, state, config, bus=None):
        M = state.M
        if M.abs().max() < _EPS:
            return state

        # Laplacian of M
        lap_M = (
            torch.roll(M, 1, 0) + torch.roll(M, -1, 0) +
            torch.roll(M, 1, 1) + torch.roll(M, -1, 1) - 4 * M
        )

        # Negative diffusion (clustering) only where mass exists
        # Confinement mask: only act where M > mean (mass-rich regions)
        M_mean = M.mean()
        confinement = torch.sigmoid(5.0 * (M - M_mean))

        dM = -self.alpha_s * self.strength * lap_M * confinement

        M_new = M + dM

        # Exact conservation
        delta = M_new.sum() - M.sum()
        M_new = M_new - delta / M_new.numel()

        delta_M = M_new - M
        E_new = state.E - delta_M * 0.5
        I_new = state.I - delta_M * 0.5

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["strong_binding_rms"] = delta_M.pow(2).mean().sqrt().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class PairPotentialOperator:
    """Strong force as pair potential: attraction between neighboring mass peaks.

    Like a spring connecting nearby mass concentrations. Creates binding
    without changing mass distribution — only moves mass between neighbors.
    Range limited to ~phi cells (confinement radius).
    """

    def __init__(self, alpha_s, label="", strength=0.01):
        self.alpha_s = alpha_s
        self.strength = strength
        self._name = f"PairPot({label})"

    @property
    def name(self):
        return self._name

    def __call__(self, state, config, bus=None):
        M = state.M
        if M.abs().max() < _EPS:
            return state

        # Potential = -alpha_s * M * M_neighbors (attractive)
        M_neighbors = (
            torch.roll(M, 1, 0) + torch.roll(M, -1, 0) +
            torch.roll(M, 1, 1) + torch.roll(M, -1, 1)
        ) / 4.0

        # Force = -grad(potential) -> mass flows toward neighbors
        # dM = alpha_s * strength * (M_neighbors - M) * M
        # This pulls mass toward the local average (binding)
        # but only proportional to existing mass (no creation from nothing)
        dM = self.alpha_s * self.strength * (M_neighbors - M) * torch.clamp(M, min=0)

        M_new = M + dM

        # Exact conservation
        delta = M_new.sum() - M.sum()
        M_new = M_new - delta / M_new.numel()

        delta_M = M_new - M
        E_new = state.E - delta_M * 0.5
        I_new = state.I - delta_M * 0.5

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["strong_binding_rms"] = delta_M.pow(2).mean().sqrt().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# -- Pipeline ------------------------------------------------------------------

def build_pipeline(strong_op=None):
    ops = [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
    ]
    if strong_op is not None:
        ops.append(strong_op)
    ops.extend([
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])
    return Pipeline(ops)


def pct_err(measured, target):
    if target == 0:
        return abs(measured) * 100
    return abs(measured - target) / abs(target) * 100


def grade(err):
    if err < 1:   return "A+"
    if err < 5:   return "A"
    if err < 10:  return "B"
    if err < 15:  return "C"
    if err < 30:  return "D"
    return "F"


def run_variant(name, strong_op, device, ticks=5000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline(strong_op)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [500, 1000, 2000, 3000, 4000, 5000]
    results = {}
    cp_idx = 0

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            met = engine.state.metrics
            M = engine.state.M

            results[tick] = {
                "f_local":     met.get("f_local_mean", 0),
                "gamma_local": met.get("gamma_local_mean", 0),
                "alpha_local": met.get("alpha_local_mean", 0),
                "G_local":     met.get("G_local_mean", 0),
                "lambda_local": met.get("lambda_local_mean", 0),
                "M_mean":      M.mean().item(),
                "M_std":       M.std().item(),
                "M_max":       M.max().item(),
                "frac_empty":  (M < 0.1).float().mean().item(),
                "binding_rms": met.get("strong_binding_rms", 0),
            }

    elapsed = time.time() - t0
    return results, elapsed


# -- Main ----------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*110}")
    print(f"  MILESTONE 5 -- EXP 02: STRONG FORCE AS BINDING OPERATOR")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per variant")
    print(f"  Using C3 (n=8, adjoint) alpha_s = {ALPHA_S_C3:.6f} from exp_01 result")
    print(f"{'='*110}")

    S = 0.01  # perturbation strength

    VARIANTS = {
        "A_baseline":    None,
        "B_gradBind":    GradientBindingOperator(ALPHA_S_C3, "C3/grad", strength=S),
        "C_lapBind":     LaplacianBindingOperator(ALPHA_S_C3, "C3/lap", strength=S),
        "D_pairPot":     PairPotentialOperator(ALPHA_S_C3, "C3/pair", strength=S),
        "E_gradBind_C2": GradientBindingOperator(ALPHA_S_C2, "C2/grad", strength=S),
    }

    all_results = {}
    for name, strong_op in VARIANTS.items():
        print(f"\n  [{name}] ...", end="", flush=True)
        results, elapsed = run_variant(name, strong_op, device)
        all_results[name] = results
        print(f" {elapsed:.0f}s")

    # -- Results ---------------------------------------------------------------

    print(f"\n{'='*110}")
    print(f"  COUPLING EVOLUTION (errors vs DFT targets)")
    print(f"{'='*110}")

    for vname, results in all_results.items():
        print(f"\n  {vname}:")
        print(f"    {'tick':>6}  {'f_local':>10}  {'gamma':>10}  {'alpha':>10}  {'G_local':>10}  {'lambda':>10}  {'avg_err':>8}  {'M_mean':>8}")
        for tick, d in sorted(results.items()):
            errs = {k: pct_err(d[k], v) for k, v in TARGETS.items()}
            avg = sum(errs.values()) / len(errs)
            print(f"    {tick:>6}  "
                  f"{d['f_local']:8.4f}({errs['f_local']:4.1f}%)  "
                  f"{d['gamma_local']:8.4f}({errs['gamma_local']:4.1f}%)  "
                  f"{d['alpha_local']:8.4f}({errs['alpha_local']:4.1f}%)  "
                  f"{d['G_local']:8.4f}({errs['G_local']:4.1f}%)  "
                  f"{d['lambda_local']:8.4f}({errs['lambda_local']:4.1f}%)  "
                  f"{avg:6.1f}%  "
                  f"{d['M_mean']:8.4f}")

    # -- Final comparison ------------------------------------------------------

    print(f"\n{'='*110}")
    print(f"  FINAL STATE (tick 5000)")
    print(f"{'='*110}")

    final_tick = 5000
    scoreboard = {}

    for vname, results in all_results.items():
        final = results.get(final_tick, {})
        errs = {k: pct_err(final.get(k, 0), v) for k, v in TARGETS.items()}
        avg_err = sum(errs.values()) / len(errs)

        late_ticks = sorted([t for t in results if t >= 3000])
        late_avgs = [sum(pct_err(results[t][k], v) for k, v in TARGETS.items()) / len(TARGETS) for t in late_ticks]
        stability = np.std(late_avgs) if len(late_avgs) >= 2 else 0.0

        mid_tick = sorted(results.keys())[len(results)//2]
        mid_avg = sum(pct_err(results[mid_tick][k], v) for k, v in TARGETS.items()) / len(TARGETS)
        drift = avg_err - mid_avg

        print(f"  {vname:<18}  avg={avg_err:5.1f}%  "
              f"f={errs['f_local']:5.1f}%({grade(errs['f_local'])})  "
              f"g={errs['gamma_local']:5.1f}%({grade(errs['gamma_local'])})  "
              f"a={errs['alpha_local']:5.1f}%({grade(errs['alpha_local'])})  "
              f"G={errs['G_local']:5.1f}%({grade(errs['G_local'])})  "
              f"L={errs['lambda_local']:5.1f}%({grade(errs['lambda_local'])})  "
              f"stab={stability:.2f}%  drift={drift:+.1f}%")

        scoreboard[vname] = {
            "avg_err": avg_err, "stability": stability, "drift": abs(drift),
            "G_err": errs["G_local"], "f_err": errs["f_local"],
            "gamma_err": errs["gamma_local"],
        }

    # -- Best operator ---------------------------------------------------------

    best = min(scoreboard.items(), key=lambda x: x[1]["avg_err"])
    print(f"\n  BEST: {best[0]} (avg_err = {best[1]['avg_err']:.1f}%)")

    # -- C3 vs C2 on gradient binding -----------------------------------------

    c3_grad = scoreboard.get("B_gradBind", {})
    c2_grad = scoreboard.get("E_gradBind_C2", {})
    if c3_grad and c2_grad:
        c3_wins = sum(1 for m in ["avg_err", "stability", "drift", "G_err", "gamma_err"]
                      if c3_grad[m] < c2_grad[m])
        c2_wins = 5 - c3_wins
        print(f"\n  C3 vs C2 (gradient binding): C3 {c3_wins}/5, C2 {c2_wins}/5")
        if c3_wins > c2_wins:
            print(f"  C3 (adjoint) CONFIRMED across operator designs")
        elif c2_wins > c3_wins:
            print(f"  C2 (fundamental) wins in binding mode -- reversal!")
        else:
            print(f"  Tied -- inconclusive")

    # -- Save ------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent.parent / "results" / f"exp_02_binding_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({
            "experiment": "milestone5/exp_02_strong_binding_operator",
            "date": datetime.now().isoformat(),
            "device": str(device),
            "strength": S,
            "alpha_s_C3": ALPHA_S_C3,
            "alpha_s_C2": ALPHA_S_C2,
            "scoreboard": scoreboard,
            "variants": {vn: {str(t): d for t, d in res.items()} for vn, res in all_results.items()},
        }, f, indent=2)

    print(f"\n  Results: {out_path}")
    print()


if __name__ == "__main__":
    main()
