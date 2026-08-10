"""Milestone 5, Experiment 06: Attractor Diagnostic — Why Are Couplings Offset?

HYPOTHESIS:
The systematic coupling offsets from exp_05 follow a clear pattern:
  f_local   overshoots by 17%  (0.564 vs 0.4812 target)
  gamma_local undershoots by 15%  (0.525 vs 0.6180 target)
  alpha_local undershoots by 8%   (0.641 vs 0.6931 target)
  G_local   overshoots by 12%  (0.427 vs 0.3820 target)
  lambda_local overshoots by 12%  (0.345 vs 0.3069 target)

The key insight: f_local overshoots AND gamma_local undershoots. Both are explained
by the SAME root cause: the I (information) field is systematically weak relative
to E (energy) and M (mass). Since f ~ E/I and gamma ~ I/E, a weak I drives f up
and gamma down simultaneously.

This experiment diagnoses the root cause by:
  1. Tracking E, I, M field averages and ratios over 5000 ticks
  2. Measuring per-operator contributions to E, I, M deltas
  3. Identifying which operators are draining I relative to E
  4. Computing entropy and PAC conservation at each checkpoint

FALSIFICATION:
- If E/I ratio stays near 1.0 throughout: the I deficit is not the cause
- If no single operator dominates the I drain: the problem is distributed
- If PAC total is not conserved: normalization is the suspect

DFT SOURCES:
- MAR exp_43: emergent coupling constants in RE v3
- Milestone 5 exp_05: RG flow measurement showing systematic offsets

AUTHOR: Peter Groom, Dawn Field Institute
DATE: 2026-03-16
"""

import copy
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Reality Engine on path
_here = os.path.dirname(os.path.abspath(__file__))
_ws = os.path.join(_here, '..', '..', '..', '..', '..')
RE_ROOT = os.path.normpath(os.path.join(_ws, 'reality-engine'))
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

# -- DFT Constants -------------------------------------------------------------

PHI = (1.0 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015329
XI = GAMMA_EM + LN_PHI
_EPS = 1e-12

TARGETS = {
    "f_local":      LN_PHI,
    "gamma_local":  1.0 / PHI,
    "alpha_local":  LN2,
    "G_local":      1.0 / PHI ** 2,
    "lambda_local": 1.0 - LN2,
}


# -- Helpers -------------------------------------------------------------------

def pct_err(measured, target):
    if abs(target) < _EPS:
        return abs(measured) * 100
    return abs(measured - target) / abs(target) * 100


def field_entropy(field_tensor):
    """Shannon entropy: -sum(p * log(p)) where p = |field| / sum(|field|)."""
    f = field_tensor.abs().flatten() + _EPS
    p = f / f.sum()
    return -(p * p.log()).sum().item()


def snapshot_state(state):
    """Deep copy a FieldState for per-operator auditing."""
    return state.replace(
        E=state.E.clone(),
        I=state.I.clone(),
        M=state.M.clone(),
        T=state.T.clone() if state.T is not None else None,
        metrics=dict(state.metrics) if state.metrics else {},
    )


def measure_fields(state):
    """Extract field statistics from a FieldState."""
    E = state.E
    I_f = state.I
    M = state.M
    T = state.T

    E_mean = E.mean().item()
    I_mean = I_f.mean().item()
    M_mean = M.mean().item()
    T_mean = T.mean().item() if T is not None else 0.0

    E_std = E.std().item()
    I_std = I_f.std().item()
    M_std = M.std().item()

    pac_total = E_mean + I_mean + M_mean
    ei_ratio = E_mean / (I_mean + _EPS)

    diseq = E - I_f
    diseq_mean = diseq.mean().item()
    diseq_std = diseq.std().item()

    entropy_E = field_entropy(E)
    entropy_I = field_entropy(I_f)
    entropy_M = field_entropy(M)

    return {
        "E_mean": E_mean, "I_mean": I_mean, "M_mean": M_mean, "T_mean": T_mean,
        "E_std": E_std, "I_std": I_std, "M_std": M_std,
        "EI_ratio": ei_ratio,
        "diseq_mean": diseq_mean, "diseq_std": diseq_std,
        "pac_total": pac_total,
        "entropy_E": entropy_E, "entropy_I": entropy_I, "entropy_M": entropy_M,
    }


def measure_couplings(state):
    """Extract coupling constants from state metrics."""
    met = state.metrics or {}
    return {
        "f_local":      met.get("f_local_mean", 0.0),
        "gamma_local":  met.get("gamma_local_mean", 0.0),
        "alpha_local":  met.get("alpha_local_mean", 0.0),
        "G_local":      met.get("G_local_mean", 0.0),
        "lambda_local": met.get("lambda_local_mean", 0.0),
    }


def build_operator_list():
    """Build the standard 16-operator list."""
    return [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ]


def operator_audit(state, config):
    """Run each operator individually on a snapshot and measure E, I, M deltas."""
    ops = build_operator_list()
    audit = []
    for op in ops:
        before = snapshot_state(state)
        try:
            after = op(before, config)
        except Exception as e:
            audit.append({
                "name": getattr(op, "name", op.__class__.__name__),
                "delta_E": 0.0, "delta_I": 0.0, "delta_M": 0.0,
                "delta_EI": 0.0, "error": str(e),
            })
            continue

        dE = (after.E.mean() - before.E.mean()).item()
        dI = (after.I.mean() - before.I.mean()).item()
        dM = (after.M.mean() - before.M.mean()).item()
        # delta_EI: how much this operator shifts E-I gap
        # Positive = widens gap (bad for I), Negative = narrows gap (good for I)
        dEI = dE - dI

        audit.append({
            "name": getattr(op, "name", op.__class__.__name__),
            "delta_E": dE, "delta_I": dI, "delta_M": dM,
            "delta_EI": dEI,
        })
    return audit


# -- Main ----------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 115)
    print("  MILESTONE 5 -- EXP 06: ATTRACTOR DIAGNOSTIC")
    print("  Why do coupling constants have systematic offsets from DFT targets?")
    print("  Device: %s | Grid: 128x64 | 5000 ticks" % device)
    print("=" * 115)
    print()
    print("  DFT Targets:")
    for name, val in TARGETS.items():
        print("    %-14s = %.6f" % (name, val))
    print()

    # -- Build engine ----------------------------------------------------------

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    ops = build_operator_list()
    pipeline = Pipeline(ops)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # -- Run simulation with checkpoints every 100 ticks ----------------------

    TOTAL_TICKS = 5000
    CHECKPOINT_INTERVAL = 100
    AUDIT_TICK = 3000

    time_series = []
    audit_result = None

    t0 = time.time()
    for tick in range(1, TOTAL_TICKS + 1):
        engine.tick()

        if tick % CHECKPOINT_INTERVAL == 0:
            fields = measure_fields(engine.state)
            couplings = measure_couplings(engine.state)
            entry = {"tick": tick}
            entry.update(fields)
            entry.update(couplings)
            time_series.append(entry)

            if tick % 500 == 0:
                elapsed = time.time() - t0
                print("  tick %5d  EI=%.4f  diseq=%.4f  pac=%.4f  [%.0fs]" % (
                    tick, fields["EI_ratio"], fields["diseq_mean"],
                    fields["pac_total"], elapsed))

        # Per-operator audit at tick 3000
        if tick == AUDIT_TICK:
            print()
            print("  [tick %d] Running per-operator audit..." % AUDIT_TICK)
            audit_result = operator_audit(engine.state, config)
            print("  [tick %d] Audit complete." % AUDIT_TICK)
            print()

    elapsed_total = time.time() - t0
    print()
    print("  Simulation complete: %.0fs" % elapsed_total)

    # -- Time Series Table: Field Evolution ------------------------------------

    print()
    print("=" * 115)
    print("  FIELD EVOLUTION (every 500 ticks)")
    print("=" * 115)
    print()
    print("  %5s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s" % (
        "tick", "E_mean", "I_mean", "M_mean", "E/I", "diseq",
        "pac_tot", "H(E)", "H(I)"))
    print("  %5s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s" % (
        "-----", "--------", "--------", "--------", "--------", "--------",
        "--------", "--------", "--------"))

    for entry in time_series:
        if entry["tick"] % 500 == 0:
            print("  %5d  %8.5f  %8.5f  %8.5f  %8.4f  %+8.5f  %8.5f  %8.2f  %8.2f" % (
                entry["tick"], entry["E_mean"], entry["I_mean"], entry["M_mean"],
                entry["EI_ratio"], entry["diseq_mean"], entry["pac_total"],
                entry["entropy_E"], entry["entropy_I"]))

    # -- Time Series Table: Coupling Evolution ---------------------------------

    print()
    print("=" * 115)
    print("  COUPLING EVOLUTION (every 500 ticks)")
    print("=" * 115)
    print()
    print("  %5s  %12s  %12s  %12s  %12s  %12s  %8s" % (
        "tick", "f_local", "gamma_local", "alpha_local", "G_local",
        "lambda_local", "avg_err"))
    print("  %5s  %12s  %12s  %12s  %12s  %12s  %8s" % (
        "-----", "------------", "------------", "------------", "------------",
        "------------", "--------"))

    for entry in time_series:
        if entry["tick"] % 500 == 0:
            errs = {k: pct_err(entry[k], v) for k, v in TARGETS.items()}
            avg_err = sum(errs.values()) / len(errs)
            print("  %5d  %6.4f(%4.1f%%)  %6.4f(%4.1f%%)  %6.4f(%4.1f%%)  %6.4f(%4.1f%%)  %6.4f(%4.1f%%)  %6.1f%%" % (
                entry["tick"],
                entry["f_local"], errs["f_local"],
                entry["gamma_local"], errs["gamma_local"],
                entry["alpha_local"], errs["alpha_local"],
                entry["G_local"], errs["G_local"],
                entry["lambda_local"], errs["lambda_local"],
                avg_err))

    # -- Time Series Table: Spatial Variance -----------------------------------

    print()
    print("=" * 115)
    print("  SPATIAL VARIANCE (every 500 ticks)")
    print("=" * 115)
    print()
    print("  %5s  %10s  %10s  %10s  %10s  %10s" % (
        "tick", "E_std", "I_std", "M_std", "diseq_std", "T_mean"))
    print("  %5s  %10s  %10s  %10s  %10s  %10s" % (
        "-----", "----------", "----------", "----------", "----------", "----------"))

    for entry in time_series:
        if entry["tick"] % 500 == 0:
            print("  %5d  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f" % (
                entry["tick"], entry["E_std"], entry["I_std"], entry["M_std"],
                entry["diseq_std"], entry["T_mean"]))

    # -- Per-Operator Audit at tick 3000 ---------------------------------------

    print()
    print("=" * 115)
    print("  PER-OPERATOR AUDIT (tick %d)" % AUDIT_TICK)
    print("  Each operator applied individually to a state snapshot.")
    print("  delta_EI = delta_E - delta_I (positive = drains I relative to E)")
    print("=" * 115)
    print()

    if audit_result:
        print("  %-30s  %12s  %12s  %12s  %12s  %s" % (
            "Operator", "delta_E", "delta_I", "delta_M", "delta_EI", "Verdict"))
        print("  %-30s  %12s  %12s  %12s  %12s  %s" % (
            "-" * 30, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 12))

        # Sort by |delta_EI| descending to show biggest offenders first
        sorted_audit = sorted(audit_result, key=lambda x: abs(x["delta_EI"]), reverse=True)

        for row in sorted_audit:
            # Verdict based on delta_EI
            dei = row["delta_EI"]
            if abs(dei) < 1e-8:
                verdict = "neutral"
            elif dei > 0:
                verdict = "DRAINS I (bad)"
            else:
                verdict = "BOOSTS I (good)"

            err_flag = " [ERROR: %s]" % row["error"] if "error" in row else ""

            print("  %-30s  %+12.6e  %+12.6e  %+12.6e  %+12.6e  %s%s" % (
                row["name"],
                row["delta_E"], row["delta_I"], row["delta_M"],
                row["delta_EI"], verdict, err_flag))

        # Summary: total bias
        total_dE = sum(r["delta_E"] for r in audit_result)
        total_dI = sum(r["delta_I"] for r in audit_result)
        total_dM = sum(r["delta_M"] for r in audit_result)
        total_dEI = total_dE - total_dI

        print()
        print("  %-30s  %+12.6e  %+12.6e  %+12.6e  %+12.6e" % (
            "TOTAL (all 16 ops)", total_dE, total_dI, total_dM, total_dEI))

        # Identify top 3 I drainers
        drainers = [r for r in sorted_audit if r["delta_EI"] > 1e-8]
        if drainers:
            print()
            print("  TOP I-DRAINING OPERATORS:")
            for i, r in enumerate(drainers[:5]):
                print("    %d. %-25s  delta_EI = %+.6e" % (i + 1, r["name"], r["delta_EI"]))
        else:
            print()
            print("  No operators show significant I-drain (all delta_EI < 1e-8)")

        # Identify top 3 I boosters
        boosters = [r for r in sorted_audit if r["delta_EI"] < -1e-8]
        if boosters:
            print()
            print("  TOP I-BOOSTING OPERATORS:")
            for i, r in enumerate(boosters[:5]):
                print("    %d. %-25s  delta_EI = %+.6e" % (i + 1, r["name"], r["delta_EI"]))

    else:
        print("  WARNING: Audit did not run (simulation may not have reached tick %d)" % AUDIT_TICK)

    # -- Diagnosis Summary -----------------------------------------------------

    print()
    print("=" * 115)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 115)
    print()

    if time_series:
        first = time_series[0]
        last = time_series[-1]
        mid = time_series[len(time_series) // 2]

        print("  E/I ratio evolution:")
        print("    Early (tick %d):  %.4f" % (first["tick"], first["EI_ratio"]))
        print("    Mid   (tick %d): %.4f" % (mid["tick"], mid["EI_ratio"]))
        print("    Final (tick %d): %.4f" % (last["tick"], last["EI_ratio"]))
        print("    Ideal:           1.0000")
        print()

        ei_drift = last["EI_ratio"] - first["EI_ratio"]
        print("  E/I drift (early to final): %+.4f" % ei_drift)
        if abs(last["EI_ratio"] - 1.0) > 0.05:
            print("  >>> CONFIRMED: E/I imbalance is present (%.1f%% off unity)" % (
                abs(last["EI_ratio"] - 1.0) * 100))
        else:
            print("  >>> E/I ratio is near unity -- imbalance hypothesis NOT supported")

        print()
        pac_drift = last["pac_total"] - first["pac_total"]
        print("  PAC conservation: %.6f -> %.6f (drift: %+.6f)" % (
            first["pac_total"], last["pac_total"], pac_drift))
        if abs(pac_drift) > 0.01:
            print("  >>> WARNING: PAC total is NOT conserved -- normalization may be redistributing")
        else:
            print("  >>> PAC total is approximately conserved")

        print()
        print("  Coupling errors at final tick:")
        for name, target in TARGETS.items():
            val = last[name]
            err = pct_err(val, target)
            direction = "overshoot" if val > target else "undershoot"
            print("    %-14s = %.4f  (target %.4f, %s %.1f%%)" % (
                name, val, target, direction, err))

    # -- Save Results ----------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent.parent / "results" / ("exp_06_attractor_%s.json" % timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "experiment": "milestone5/exp_06_attractor_diagnostic",
        "date": datetime.now().isoformat(),
        "device": str(device),
        "grid": "128x64",
        "ticks": TOTAL_TICKS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "audit_tick": AUDIT_TICK,
        "dft_targets": {k: v for k, v in TARGETS.items()},
        "time_series": time_series,
        "operator_audit": audit_result,
        "summary": {
            "final_EI_ratio": time_series[-1]["EI_ratio"] if time_series else None,
            "final_pac_total": time_series[-1]["pac_total"] if time_series else None,
            "final_diseq_mean": time_series[-1]["diseq_mean"] if time_series else None,
            "final_couplings": {k: time_series[-1][k] for k in TARGETS} if time_series else None,
            "final_errors": {
                k: pct_err(time_series[-1][k], v) for k, v in TARGETS.items()
            } if time_series else None,
        },
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print()
    print("  Results: %s" % out_path)
    print()


if __name__ == "__main__":
    main()
