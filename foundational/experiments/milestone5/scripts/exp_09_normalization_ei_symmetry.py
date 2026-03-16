#!/usr/bin/env python3
"""
DFT Milestone 5 — Experiment 09: Normalization E/I Symmetry Fix

Exp_06 diagnosed that NormalizationOperator is the sole source of E/I drift:
  - delta_EI = +5.7e-3 per tick (drains I relative to E)
  - Actualization partially compensates (-5.4e-3) but 5.7% net imbalance
  - Over 5000 ticks: E/I ratio drifts from 1.7 to 4.0 (ideal = 1.0)

Root cause: tanh clamping + cross-injection + equal PAC correction are all
E/I-asymmetric when the fields have different magnitudes.

This experiment tests 4 normalization variants:
  A) Baseline (current normalization)
  B) Ratio-preserving: after tanh, restore original E/I ratio
  C) Symmetric scale: normalize E,I by the same factor (max of the two)
  D) No cross-injection: skip QBE cross-injection, send all excess to M

For each variant, run 5000 ticks and measure E/I ratio evolution + couplings.
"""

import json, math, os, sys, time
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__))
_ws   = os.path.join(_here, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_ws, 'reality-engine'))

import torch

# ── Reality Engine imports ──────────────────────────────────────────

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
TARGETS = {
    "f_local":     math.log(PHI),       # 0.4812
    "gamma_local": 1.0 / PHI,           # 0.6180
    "alpha_local": math.log(2),          # 0.6931
    "G_local":     1.0 / PHI**2,         # 0.3820
    "lambda_local": 1.0 - math.log(2),   # 0.3069
}
_EPS = 1e-12


# ── Modified Normalization Variants ────────────────────────────────

class NormVariantB(NormalizationOperator):
    """Ratio-preserving: after tanh clamp, restore original E/I ratio."""

    @property
    def name(self):
        return "normalization"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        if not config.enable_normalization:
            return state

        s = config.field_scale
        M_cap = s / 5.0

        # Hard M cap with Landauer reinjection (same as baseline)
        M_floored = torch.clamp(state.M, min=0.0)
        M_new = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_new
        reinjection = dM_removed * 0.5
        E_cur = state.E + reinjection
        I_cur = state.I + reinjection

        # Record E/I ratio BEFORE clamping
        E_abs = E_cur.abs()
        I_abs = I_cur.abs()
        total_abs = E_abs + I_abs + _EPS
        E_frac = E_abs / total_abs  # fraction of |E|+|I| that is |E|
        I_frac = I_abs / total_abs

        # Tanh clamp on TOTAL magnitude, then redistribute
        total_signed = E_cur + I_cur
        total_clamped = s * torch.tanh(total_signed / s)

        # Also clamp individual fields for safety
        E_clamped = s * torch.tanh(E_cur / s)
        I_clamped = s * torch.tanh(I_cur / s)

        # Compute how much was lost per field
        E_loss = E_cur - E_clamped
        I_loss = I_cur - I_clamped
        total_loss = E_loss + I_loss

        # Redistribute total loss preserving original E/I ratio
        # Instead of cross-injection, split the total loss by original fraction
        E_new = E_clamped + total_loss * E_frac * torch.sign(E_cur)
        I_new = I_clamped + total_loss * I_frac * torch.sign(I_cur)

        # Remainder to M
        pac_before = E_cur + I_cur + M_new
        pac_after = E_new + I_new + M_new
        residual = (pac_before - pac_after)
        M_new = M_new + residual
        M_new = torch.clamp(M_new, min=0.0)

        # Global PAC correction (equal split)
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()
        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac
        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        metrics["landauer_reinjection"] = dM_removed.sum().item()
        metrics["crystallisation"] = 0.0
        metrics["pac_correction"] = pac_residual
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class NormVariantC(NormalizationOperator):
    """Symmetric scale: normalize E,I by the same factor."""

    @property
    def name(self):
        return "normalization"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        if not config.enable_normalization:
            return state

        s = config.field_scale
        M_cap = s / 5.0

        # Hard M cap + Landauer (same)
        M_floored = torch.clamp(state.M, min=0.0)
        M_new = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_new
        reinjection = dM_removed * 0.5
        E_cur = state.E + reinjection
        I_cur = state.I + reinjection

        # Symmetric: find the max absolute value across BOTH fields
        # and apply the SAME scaling factor to both
        max_abs = torch.maximum(E_cur.abs(), I_cur.abs())
        # Only scale where fields exceed s
        scale = torch.where(max_abs > s, s / (max_abs + _EPS), torch.ones_like(max_abs))

        E_new = E_cur * scale
        I_new = I_cur * scale

        # Excess goes to M
        excess = (E_cur - E_new) + (I_cur - I_new)
        M_new = M_new + excess
        M_new = torch.clamp(M_new, min=0.0)

        # Global PAC correction
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()
        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac
        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        metrics["landauer_reinjection"] = dM_removed.sum().item()
        metrics["crystallisation"] = excess.sum().item()
        metrics["pac_correction"] = pac_residual
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class NormVariantD(NormalizationOperator):
    """No cross-injection: tanh clamp E,I independently, all excess to M."""

    @property
    def name(self):
        return "normalization"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        if not config.enable_normalization:
            return state

        s = config.field_scale
        M_cap = s / 5.0

        # Hard M cap + Landauer (same)
        M_floored = torch.clamp(state.M, min=0.0)
        M_new = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_new
        reinjection = dM_removed * 0.5
        E_cur = state.E + reinjection
        I_cur = state.I + reinjection

        # Tanh clamp independently (same as baseline)
        E_new = s * torch.tanh(E_cur / s)
        I_new = s * torch.tanh(I_cur / s)

        # ALL excess to M (no cross-injection at all)
        E_loss = E_cur - E_new
        I_loss = I_cur - I_new
        M_new = M_new + E_loss + I_loss
        M_new = torch.clamp(M_new, min=0.0)

        # Global PAC correction
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()
        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac
        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        metrics["landauer_reinjection"] = dM_removed.sum().item()
        metrics["crystallisation"] = (E_loss + I_loss).sum().item()
        metrics["pac_correction"] = pac_residual
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ── Measurement helpers ────────────────────────────────────────────

def measure_couplings(state):
    """Measure the 5 DFT coupling constants from field state."""
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


def coupling_errors(couplings):
    """Compute % error for each coupling."""
    errs = {}
    for k, v in couplings.items():
        t = TARGETS[k]
        errs[k] = abs(v - t) / t * 100
    errs["avg_err"] = sum(errs.values()) / len(TARGETS)
    return errs


def grade(err):
    if err < 2: return "A"
    if err < 5: return "A-"
    if err < 10: return "B"
    if err < 15: return "C"
    if err < 25: return "D"
    return "F"


# ── Build pipeline ─────────────────────────────────────────────────

def build_pipeline(norm_operator):
    """Full 16-operator pipeline with specified normalization variant."""
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        ActualizationOperator(),
        MemoryOperator(),
        PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(),
        ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        norm_operator,
        SECTrackingOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])


# ── Main experiment ────────────────────────────────────────────────

def run_variant(name, norm_operator, grid=(128, 64), ticks=5000, device="cuda"):
    """Run one normalization variant and collect time series."""
    cfg = SimulationConfig(nu=grid[0], nv=grid[1], device=device,
                           dt=0.001, enable_actualization=True,
                           actualization_threshold=0.05)
    pipe = build_pipeline(norm_operator)
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
            errs = coupling_errors(c)
            EI_ratio = st.E.pow(2).mean().item() / (st.I.pow(2).mean().item() + _EPS)
            diseq = (st.E.mean() - st.I.mean()).item()
            pac = (st.E + st.I + st.M).sum().item()

            record = {
                "tick": tick,
                "E_mean": st.E.mean().item(),
                "I_mean": st.I.mean().item(),
                "M_mean": st.M.mean().item(),
                "EI_ratio": EI_ratio,
                "diseq": diseq,
                "pac_total": pac,
                **c, **errs,
            }
            history.append(record)
            elapsed = time.time() - t0
            print(f"  [{name}] tick {tick:5d}  EI={EI_ratio:.4f}  avg_err={errs['avg_err']:.1f}%  [{elapsed:.0f}s]")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = (128, 64)
    ticks = 5000
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("  DFT Experiment 09: Normalization E/I Symmetry Fix")
    print("  Testing 4 normalization variants for E/I balance")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks")
    print("=" * 80)

    variants = {
        "A_baseline":    NormalizationOperator(),
        "B_ratio_pres":  NormVariantB(),
        "C_sym_scale":   NormVariantC(),
        "D_no_crossinj": NormVariantD(),
    }

    all_results = {}

    for vname, norm_op in variants.items():
        print(f"\n--- Variant {vname} ---")
        history = run_variant(vname, norm_op, grid=grid, ticks=ticks, device=device)
        all_results[vname] = history

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL STATE COMPARISON (tick {})".format(ticks))
    print("=" * 80)

    header = f"  {'Variant':<18s} {'avg_err':>8s} {'f_local':>10s} {'gamma':>10s} {'alpha':>10s} {'G_local':>10s} {'lambda':>10s} {'E/I ratio':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_avg = 999
    best_name = ""

    for vname, history in all_results.items():
        final = history[-1]
        f_err = final["f_local_err"] if "f_local_err" in final else final.get("f_local", 0)
        # Extract individual errors
        fe = abs(final["f_local"] - TARGETS["f_local"]) / TARGETS["f_local"] * 100
        ge = abs(final["gamma_local"] - TARGETS["gamma_local"]) / TARGETS["gamma_local"] * 100
        ae = abs(final["alpha_local"] - TARGETS["alpha_local"]) / TARGETS["alpha_local"] * 100
        Ge = abs(final["G_local"] - TARGETS["G_local"]) / TARGETS["G_local"] * 100
        le = abs(final["lambda_local"] - TARGETS["lambda_local"]) / TARGETS["lambda_local"] * 100
        avg = (fe + ge + ae + Ge + le) / 5

        eir = final["EI_ratio"]

        line = f"  {vname:<18s} {avg:>7.1f}% {final['f_local']:.4f}{grade(fe):>2s} {final['gamma_local']:.4f}{grade(ge):>2s} {final['alpha_local']:.4f}{grade(ae):>2s} {final['G_local']:.4f}{grade(Ge):>2s} {final['lambda_local']:.4f}{grade(le):>2s} {eir:>9.3f}"
        print(line)

        if avg < best_avg:
            best_avg = avg
            best_name = vname

    print(f"\n  BEST: {best_name} (avg_err = {best_avg:.1f}%)")

    # ── E/I ratio evolution comparison ─────────────────────────────
    print("\n" + "=" * 80)
    print("  E/I RATIO EVOLUTION")
    print("=" * 80)

    ticks_to_show = [500, 1000, 2000, 3000, 4000, 5000]
    header2 = f"  {'tick':>6s}"
    for vname in all_results:
        header2 += f"  {vname:>16s}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for t in ticks_to_show:
        line = f"  {t:>6d}"
        for vname, history in all_results.items():
            rec = next((h for h in history if h["tick"] == t), None)
            if rec:
                line += f"  {rec['EI_ratio']:>16.4f}"
            else:
                line += f"  {'n/a':>16s}"
        print(line)

    # ── Coupling convergence comparison ────────────────────────────
    print("\n" + "=" * 80)
    print("  AVG ERROR EVOLUTION (%)")
    print("=" * 80)
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for t in ticks_to_show:
        line = f"  {t:>6d}"
        for vname, history in all_results.items():
            rec = next((h for h in history if h["tick"] == t), None)
            if rec:
                line += f"  {rec['avg_err']:>15.1f}%"
            else:
                line += f"  {'n/a':>16s}"
        print(line)

    # ── Per-coupling best variant ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  PER-COUPLING WINNER (lowest error at tick {})".format(ticks))
    print("=" * 80)

    for coupling in TARGETS:
        best_v = ""
        best_e = 999
        for vname, history in all_results.items():
            final = history[-1]
            err = abs(final[coupling] - TARGETS[coupling]) / TARGETS[coupling] * 100
            if err < best_e:
                best_e = err
                best_v = vname
        print(f"  {coupling:<16s}: {best_v} ({best_e:.1f}%)")

    # ── Save results ───────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, f"exp_09_norm_symmetry_{ts}.json")

    save_data = {
        "experiment": "exp_09_normalization_ei_symmetry",
        "date": ts,
        "grid": list(grid),
        "ticks": ticks,
        "device": device,
        "targets": TARGETS,
        "variants": {k: v for k, v in all_results.items()},
        "best_variant": best_name,
        "best_avg_err": best_avg,
    }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {outpath}")


if __name__ == "__main__":
    main()
