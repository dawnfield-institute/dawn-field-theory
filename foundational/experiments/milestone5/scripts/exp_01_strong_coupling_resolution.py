"""Milestone 5, Experiment 01: Strong Coupling Resolution via Simulator Discrimination

HYPOTHESIS:
The PAC cascade "sees" either the fundamental representation (n=3, color charges)
or the adjoint representation (n=8, gluon self-coupling) at the strong force boundary.
MAR exp_39 gives two leading candidates:
  C2 (n=3, fundamental): alpha_s = 0.1182 (0.29% error)
  C3 (n=8, adjoint):     alpha_s = 0.1172 (0.58% error)

The Reality Engine simulator can discriminate by testing which coupling value,
when used to modulate short-range mass interactions, produces more physically
consistent emergent behavior across the 13-metric scorecard.

METHOD:
Run 4 variants of the RE simulator (5K ticks each, checkpoints at 1K intervals):
  A: Baseline — current 16-operator pipeline, no strong force
  B: C2 modulation (n=3, fundamental) — short-range enhancement at alpha_s = 0.1182
  C: C3 modulation (n=8, adjoint) — short-range enhancement at alpha_s = 0.1172
  D: Bare modulation (no correction) — alpha_s = 0.0773 (high-energy value)

Strong force is implemented as spectral modulation: enhance high-k (local) modes
of the mass potential, suppress low-k (global) modes. This matches asymptotic
freedom — strong force is strong locally, weak globally.

DISCRIMINATION CRITERIA:
  1. Avg coupling error vs DFT targets (lower = better)
  2. Late-time stability (lower std of last 3 checkpoints)
  3. Drift (less drift from mid-run to end = better)
  4. G_local accuracy (gravity-strong coupling interplay)
  5. f_deviation (currently 16.2% — can strong force fix it?)

FALSIFICATION:
- If neither candidate improves scorecard vs baseline: strong force operator is wrong
- If both produce identical results: simulator can't discriminate (need longer runs)
- If a different coupling value works better: both C2 and C3 are wrong

DFT SOURCES:
- MAR exp_37: correction template F_a/(n*pi*F_b^2)
- MAR exp_38: gauge Fibonacci constraint, 1+3+8+1=13=F7
- MAR exp_39: strong coupling candidates C1-C4
- MAR exp_43: emergent coupling constants in RE v3

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

# Reality Engine on path
RE_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..', 'reality-engine'))
if RE_ROOT not in sys.path:
    sys.path.insert(0, RE_ROOT)

import torch
import numpy as np

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

# ── DFT Constants ─────────────────────────────────────────────────────────────

PHI = (1.0 + math.sqrt(5)) / 2
GAMMA_EM = 0.5772156649015329
XI = GAMMA_EM + math.log(PHI)
LN2 = math.log(2)
LN2_SQ = LN2 ** 2
_EPS = 1e-12

# Fibonacci
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Strong coupling from MAR exp_39
ALPHA_S_MEASURED = 0.1179
ALPHA_S_BARE = F[3] / (2 * PHI * F[6])                    # 0.077254
ALPHA_S_C2 = ALPHA_S_BARE * (1 + F[5]/(3*math.pi*F[2]**2))  # 0.1182
ALPHA_S_C3 = ALPHA_S_BARE * (1 + F[7]/(8*math.pi*F[2]**2))  # 0.1172

# DFT attractor targets
TARGETS = {
    "f_local":     GAMMA_EM,
    "gamma_local": 1.0 / PHI,
    "alpha_local": LN2,
    "G_local":     1.0 / PHI**2,
    "lambda_local": 1.0 - LN2,
}


# ── Strong Force Operator ─────────────────────────────────────────────────────

class StrongForceOperator:
    """Short-range mass interaction modulated by strong coupling alpha_s.

    Physics: the strong force creates short-range attraction between mass
    concentrations (confinement), while being asymptotically free at long range.
    Implemented as spectral modulation of the mass field's gravitational potential.

    The key DFT insight: the cascade-depth tiling filter already suppresses
    long-range gravity. The strong force does the OPPOSITE — it enhances
    short-range interactions beyond what gravity alone provides.
    """

    def __init__(self, alpha_s: float, label: str = "", strength: float = 0.1):
        self.alpha_s = alpha_s
        self.strength = strength
        self.label = label
        self._name = f"StrongForce({label})"

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        if M.abs().max() < _EPS:
            return state

        # Spectral decomposition of mass field
        M_fft = torch.fft.rfft2(M)
        nu, nv = M.shape
        ky = torch.fft.fftfreq(nu, device=M.device)
        kx = torch.fft.rfftfreq(nv, device=M.device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        K2 = KX**2 + KY**2
        K_mag = torch.sqrt(K2 + _EPS)

        # Strong coupling profile: asymptotic freedom
        # Strong at high k (short range), weak at low k (long range)
        # Transition scale ~ 1/phi (the natural PAC cascade boundary)
        k_transition = 1.0 / PHI
        # Smooth step: sigmoid centered at transition scale
        strong_profile = torch.sigmoid(10.0 * (K_mag - k_transition))

        # Modulate: enhance short-range mass potential by alpha_s
        # Small perturbation to stay in perturbative regime
        enhancement = 1.0 + self.alpha_s * strong_profile * self.strength

        M_fft_enhanced = M_fft * enhancement
        M_new = torch.fft.irfft2(M_fft_enhanced, s=M.shape)

        # PAC conservation: redistribute delta to E and I
        delta_M = M_new - M
        E_new = E - delta_M * 0.5
        I_new = I - delta_M * 0.5

        metrics = dict(state.metrics) if state.metrics else {}
        metrics["strong_force_alpha_s"] = self.alpha_s
        metrics["strong_force_delta_M_rms"] = delta_M.pow(2).mean().sqrt().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ── Pipeline Builder ──────────────────────────────────────────────────────────

def build_pipeline(strong_op=None):
    """Build the full 16-operator pipeline, optionally inserting strong force."""
    ops = [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
    ]
    # Insert strong force after gravity (short-range enhancement of gravity result)
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


# ── Measurement ───────────────────────────────────────────────────────────────

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


# ── Run Variant ───────────────────────────────────────────────────────────────

def run_variant(name, strong_op, device, ticks=5000):
    """Run one variant and collect coupling evolution at checkpoints."""
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
            E = engine.state.E
            I_field = engine.state.I

            results[tick] = {
                "f_local":     met.get("f_local_mean", 0),
                "gamma_local": met.get("gamma_local_mean", 0),
                "alpha_local": met.get("alpha_local_mean", 0),
                "G_local":     met.get("G_local_mean", 0),
                "lambda_local": met.get("lambda_local_mean", 0),
                "M_mean":      M.mean().item(),
                "M_std":       M.std().item(),
                "M_max":       M.max().item(),
                "xi_s_mean":   met.get("xi_s_mean", 0),
                "xi_mod_mean": met.get("xi_mod_mean", 0),
                "frac_empty":  (M < 0.1).float().mean().item(),
                "strong_delta_M": met.get("strong_force_delta_M_rms", 0),
            }

    elapsed = time.time() - t0
    return results, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*115}")
    print(f"  MILESTONE 5 — EXP 01: STRONG COUPLING RESOLUTION VIA SIMULATOR DISCRIMINATION")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per variant")
    print(f"{'='*115}")
    print()
    print(f"  DFT Strong Coupling Candidates (MAR exp_39):")
    print(f"    Bare:            alpha_s = {ALPHA_S_BARE:.6f}")
    print(f"    C2 (n=3, fund):  alpha_s = {ALPHA_S_C2:.6f}  ({pct_err(ALPHA_S_C2, ALPHA_S_MEASURED):.2f}% vs PDG)")
    print(f"    C3 (n=8, adj):   alpha_s = {ALPHA_S_C3:.6f}  ({pct_err(ALPHA_S_C3, ALPHA_S_MEASURED):.2f}% vs PDG)")
    print(f"    Measured (PDG):  alpha_s = {ALPHA_S_MEASURED}")

    VARIANTS = {
        "A_baseline":        None,
        "B_C2_s01":          StrongForceOperator(ALPHA_S_C2, "C2/0.01", strength=0.01),
        "C_C3_s01":          StrongForceOperator(ALPHA_S_C3, "C3/0.01", strength=0.01),
        "D_C2_s001":         StrongForceOperator(ALPHA_S_C2, "C2/0.001", strength=0.001),
        "E_C3_s001":         StrongForceOperator(ALPHA_S_C3, "C3/0.001", strength=0.001),
        "F_bare_s01":        StrongForceOperator(ALPHA_S_BARE, "bare/0.01", strength=0.01),
    }

    all_results = {}
    for name, strong_op in VARIANTS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        results, elapsed = run_variant(name, strong_op, device)
        all_results[name] = results
        print(f" {elapsed:.0f}s")

    # ── Results Table ─────────────────────────────────────────────────────────

    print(f"\n{'='*115}")
    print(f"  COUPLING EVOLUTION (errors vs DFT targets)")
    print(f"{'='*115}")

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

    # ── Discrimination ────────────────────────────────────────────────────────

    print(f"\n{'='*115}")
    print(f"  DISCRIMINATION: FINAL STATE (tick 5000)")
    print(f"{'='*115}")

    final_tick = 5000
    print(f"\n  {'Variant':<20}", end="")
    for name in TARGETS:
        print(f"  {name:>12}", end="")
    print(f"  {'avg_err':>8}  {'stability':>9}  {'drift':>7}")

    print(f"  {'─'*20}", end="")
    for _ in TARGETS:
        print(f"  {'─'*12}", end="")
    print(f"  {'─'*8}  {'─'*9}  {'─'*7}")

    scoreboard = {}
    for vname, results in all_results.items():
        final = results.get(final_tick, {})
        errs = {k: pct_err(final.get(k, 0), v) for k, v in TARGETS.items()}
        avg_err = sum(errs.values()) / len(errs)

        # Stability: std of error across last 3 checkpoints
        late_ticks = sorted([t for t in results if t >= 3000])
        late_avgs = []
        for t in late_ticks:
            d = results[t]
            e = sum(pct_err(d[k], v) for k, v in TARGETS.items()) / len(TARGETS)
            late_avgs.append(e)
        stability = np.std(late_avgs) if len(late_avgs) >= 2 else 0.0

        # Drift: mid to end
        mid_tick = sorted(results.keys())[len(results)//2]
        mid_d = results[mid_tick]
        mid_avg = sum(pct_err(mid_d[k], v) for k, v in TARGETS.items()) / len(TARGETS)
        drift = avg_err - mid_avg

        print(f"  {vname:<20}", end="")
        for name, target in TARGETS.items():
            err = errs[name]
            g = grade(err)
            print(f"  {err:7.1f}% ({g})", end="")
        print(f"  {avg_err:6.1f}%  {stability:8.2f}%  {drift:+6.1f}%")

        scoreboard[vname] = {
            "avg_err": avg_err,
            "stability": stability,
            "drift": abs(drift),
            "G_err": errs["G_local"],
            "f_err": errs["f_local"],
        }

    # ── Winner ────────────────────────────────────────────────────────────────

    # ── C2 vs C3 at matched strength ──────────────────────────────────────

    print(f"\n  SCOREBOARD (lower = better):")
    all_variant_names = list(scoreboard.keys())
    header = f"  {'Metric':<15}"
    for v in all_variant_names:
        header += f"  {v[:14]:>14}"
    print(header)
    print(f"  {'-'*15}" + "".join(f"  {'-'*14}" for _ in all_variant_names))

    metrics_to_compare = ["avg_err", "stability", "drift", "G_err", "f_err"]
    c2_wins = 0
    c3_wins = 0

    for metric in metrics_to_compare:
        vals = {v: scoreboard[v][metric] for v in scoreboard}
        row = f"  {metric:<15}"
        for v in all_variant_names:
            row += f"  {vals[v]:13.2f}%"
        print(row)

        # C2 vs C3 at strength=0.01 head-to-head
        c2_val = vals.get("B_C2_s01", float("inf"))
        c3_val = vals.get("C_C3_s01", float("inf"))
        if c2_val < c3_val:
            c2_wins += 1
        elif c3_val < c2_val:
            c3_wins += 1

    print(f"\n  HEAD-TO-HEAD (strength=0.01): C2 (n=3) {c2_wins}/5  vs  C3 (n=8) {c3_wins}/5")

    if c2_wins > c3_wins:
        print(f"\n  >>> VERDICT: C2 (n=3, FUNDAMENTAL) -- PAC cascade sees COLOR CHARGES")
        print(f"      alpha_s = F3/(2*phi*F6) * (1 + F5/(3*pi*F2^2)) = {ALPHA_S_C2:.6f}")
    elif c3_wins > c2_wins:
        print(f"\n  >>> VERDICT: C3 (n=8, ADJOINT) -- PAC cascade sees GLUON SELF-COUPLING")
        print(f"      alpha_s = F3/(2*phi*F6) * (1 + F7/(8*pi*F2^2)) = {ALPHA_S_C3:.6f}")
    else:
        print(f"\n  >>> VERDICT: INCONCLUSIVE -- tied {c2_wins}-{c3_wins}")
        print(f"      Need: longer runs (10K+), different strength tuning, or new metrics")

    # ── Save ──────────────────────────────────────────────────────────────────

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent.parent / "results" / f"exp_01_strong_coupling_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "experiment": "milestone5/exp_01_strong_coupling_resolution",
        "date": datetime.now().isoformat(),
        "device": str(device),
        "grid": "128x64",
        "ticks": 5000,
        "dft_constants": {
            "alpha_s_bare": ALPHA_S_BARE,
            "alpha_s_C2": ALPHA_S_C2,
            "alpha_s_C3": ALPHA_S_C3,
            "alpha_s_measured": ALPHA_S_MEASURED,
        },
        "scoreboard": scoreboard,
        "c2_wins": c2_wins,
        "c3_wins": c3_wins,
        "variants": {
            vname: {str(t): d for t, d in results.items()}
            for vname, results in all_results.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results: {out_path}")
    print()


if __name__ == "__main__":
    main()
