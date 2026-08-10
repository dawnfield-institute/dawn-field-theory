#!/usr/bin/env python3
"""
DFT Milestone 5 -- Experiment 13: Symmetric De-Actualization

Exp_12 found that de-actualization (memory fading) reduces coupling drift by half.
But the dissolved mass returned 50/50 to E and I. This breaks the symmetry with
actualization, which splits potential via f_local = E^2/(E^2+I^2).

The PAC cycle should be FULLY symmetric:
  Actualization (in):  P -> f_local*P to E, (1-f_local)*P to I
  De-actualization (out): M -> f_local*dM from E, (1-f_local)*dM from I

This means: cells where E dominates (high f_local) get MORE of the dissolved
mass back as E. Cells where I dominates get more as I. The field remembers
its character even as it forgets its mass.

Variants:
  A) Baseline (no de-actualization)
  B) exp_12 winner: de-act eta=0.01, 50/50 return
  C) Symmetric: de-act eta=0.01, f_local split return
  D) Inverse: de-act eta=0.01, (1-f_local) split (E gets less, I gets more)
  E) Symmetric + tuned eta=0.02
  F) Symmetric + tuned eta=0.005 (gentler)
"""

import json, math, os, sys, time
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

# -- DFT targets --
PHI = (1 + math.sqrt(5)) / 2
GAMMA_EM = 0.5772156649015328
LN2 = math.log(2)

TARGETS = {
    "f_local_mean":      GAMMA_EM,
    "gamma_local_mean":  1.0 / PHI,
    "alpha_local_mean":  LN2,
    "G_local_mean":      1.0 / PHI**2,
    "lambda_local_mean": 1.0 - LN2,
}

LABELS = {
    "f_local_mean":      "f_local",
    "gamma_local_mean":  "gamma",
    "alpha_local_mean":  "alpha",
    "G_local_mean":      "G_local",
    "lambda_local_mean": "lambda",
}

_EPS = 1e-12


def pct_err(measured, target):
    if abs(target) < 1e-15:
        return abs(measured) * 100
    return abs(measured - target) / abs(target) * 100


def grade(err):
    if err < 2: return "A"
    if err < 5: return "A-"
    if err < 10: return "B"
    if err < 15: return "C"
    if err < 25: return "D"
    return "F"


class DeActMemoryOperator:
    """MemoryOperator + de-actualization with configurable return split.

    split_mode:
      "equal"   - 50/50 to E and I (exp_12 variant B)
      "f_local" - f_local to E, (1-f_local) to I (symmetric with actualization)
      "inverse" - (1-f_local) to E, f_local to I (asymmetric, biases toward I)
    """

    def __init__(self, eta=0.01, split_mode="equal"):
        self._inner = MemoryOperator()
        self.eta = eta
        self.split_mode = split_mode

    @property
    def name(self):
        return "memory"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        # Run standard memory operator
        state = self._inner(state, config, bus)

        E, I, M = state.E, state.I, state.M
        dt = config.dt

        # gamma_local = (E-I)^2 / (E^2 + I^2 + eps)
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Forgetting factor: high where balanced, zero where imbalanced
        forgetting = 1.0 - gamma_local

        # De-actualization: memory dissolves
        dM_deact = -self.eta * M * forgetting * dt
        M_new = torch.clamp(M + dM_deact, min=0.0)
        dM_actual = M_new - M  # negative

        # Return dissolved mass to E and I (PAC conserving)
        dissolved = -dM_actual  # positive amount to distribute

        if self.split_mode == "equal":
            E_return = dissolved * 0.5
            I_return = dissolved * 0.5
        elif self.split_mode == "f_local":
            # f_local = E^2/(E^2+I^2) -- same ratio actualization uses
            f_local = E.pow(2) / (E.pow(2) + I.pow(2) + _EPS)
            E_return = dissolved * f_local
            I_return = dissolved * (1.0 - f_local)
        elif self.split_mode == "inverse":
            # Inverse: bias return toward I (the weaker field)
            f_local = E.pow(2) / (E.pow(2) + I.pow(2) + _EPS)
            E_return = dissolved * (1.0 - f_local)
            I_return = dissolved * f_local
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        E_new = E + E_return
        I_new = I + I_return

        metrics = dict(state.metrics)
        metrics["deactualization_rate"] = dissolved.mean().item()
        metrics["deactualization_total"] = dissolved.sum().item()
        metrics["forgetting_mean"] = forgetting.mean().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


def build_pipeline(variant):
    if variant == "A":
        mem_op = MemoryOperator()
    elif variant == "B":
        mem_op = DeActMemoryOperator(eta=0.01, split_mode="equal")
    elif variant == "C":
        mem_op = DeActMemoryOperator(eta=0.01, split_mode="f_local")
    elif variant == "D":
        mem_op = DeActMemoryOperator(eta=0.01, split_mode="inverse")
    elif variant == "E":
        mem_op = DeActMemoryOperator(eta=0.02, split_mode="f_local")
    elif variant == "F":
        mem_op = DeActMemoryOperator(eta=0.005, split_mode="f_local")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        ActualizationOperator(),
        mem_op,
        PhiCascadeOperator(),
        GravitationalCollapseOperator(),
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


def run_variant(variant, device, grid, ticks, sample_ticks):
    pipe = build_pipeline(variant)
    cfg = SimulationConfig(nu=grid[0], nv=grid[1], device=device,
                           dt=0.001, enable_actualization=True,
                           actualization_threshold=0.05)
    torch.manual_seed(42)
    eng = Engine(config=cfg, pipeline=pipe)
    eng.initialize("big_bang", temperature=3.0)

    records = []
    t0 = time.time()

    for tick in range(1, ticks + 1):
        eng.tick()
        if tick in sample_ticks:
            st = eng.state
            m = st.metrics

            record = {"tick": tick, "variant": variant}
            for key in TARGETS:
                val = m.get(key, None)
                if val is not None:
                    record[key] = val
                    record[key + "_err"] = pct_err(val, TARGETS[key])
                else:
                    record[key] = float('nan')
                    record[key + "_err"] = float('nan')

            errs = [record[k + "_err"] for k in TARGETS
                    if not math.isnan(record.get(k + "_err", float('nan')))]
            record["avg_err"] = sum(errs) / len(errs) if errs else float('nan')
            record["max_err"] = max(errs) if errs else float('nan')

            record["M_mean"] = st.M.mean().item()
            record["M_max"] = st.M.max().item()
            record["E_mean"] = st.E.mean().item()
            record["I_mean"] = st.I.mean().item()
            record["pac_total"] = (st.E + st.I + st.M).sum().item()
            record["deact_rate"] = m.get("deactualization_rate", 0)
            record["forgetting"] = m.get("forgetting_mean", 0)

            records.append(record)

            if tick % 2000 == 0:
                elapsed = time.time() - t0
                print(f"    [{variant}] tick {tick:5d}  avg={record['avg_err']:.1f}%  "
                      f"f={record.get('f_local_mean',0):.4f}  "
                      f"G={record.get('G_local_mean',0):.4f}  "
                      f"M={record['M_mean']:.4f}  [{elapsed:.0f}s]")

    return records


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = (128, 64)
    ticks = 10000
    sample_ticks = set(range(500, ticks + 1, 500))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    variants = {
        "A": "Baseline (no de-actualization)",
        "B": "De-act eta=0.01, 50/50 return",
        "C": "De-act eta=0.01, f_local split (symmetric)",
        "D": "De-act eta=0.01, inverse split (I-biased)",
        "E": "De-act eta=0.02, f_local split (stronger)",
        "F": "De-act eta=0.005, f_local split (gentler)",
    }

    print("=" * 90)
    print("  DFT Experiment 13: Symmetric De-Actualization")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks")
    print("=" * 90)
    print()
    print("  Actualization splits P via f_local = E^2/(E^2+I^2)")
    print("  De-actualization should split dissolved M the same way (symmetry)")
    print()

    all_records = {}
    for v, desc in variants.items():
        print(f"  --- Variant {v}: {desc} ---")
        records = run_variant(v, device, grid, ticks, sample_ticks)
        all_records[v] = records
        print()

    # -- Comparison at key ticks --
    compare_ticks = [1000, 3000, 5000, 7000, 10000]

    print("=" * 90)
    print("  COMPARISON: avg_err (%) at key ticks")
    print("=" * 90)
    header = f"  {'tick':>6s}"
    for v in variants:
        header += f"  {v:>8s}"
    print(header)
    print("  " + "-" * (8 + 10 * len(variants)))

    for tick in compare_ticks:
        line = f"  {tick:>6d}"
        for v in variants:
            rec = next((r for r in all_records[v] if r["tick"] == tick), None)
            if rec:
                line += f"  {rec['avg_err']:>7.1f}%"
            else:
                line += f"  {'n/a':>8s}"
        print(line)

    # -- Detailed at tick 5000 and 10000 --
    for check_tick in [5000, 10000]:
        print(f"\n{'=' * 90}")
        print(f"  DETAILED at tick {check_tick}")
        print("=" * 90)

        header = f"  {'metric':>12s}"
        for v in variants:
            header += f"  {v:>10s}"
        print(header)
        print("  " + "-" * (14 + 12 * len(variants)))

        for key in TARGETS:
            label = LABELS[key]
            line = f"  {label:>12s}"
            for v in variants:
                rec = next((r for r in all_records[v] if r["tick"] == check_tick), None)
                if rec:
                    err = rec.get(key + "_err", float('nan'))
                    line += f"  {err:>7.1f}%{grade(err):<2s}"
                else:
                    line += f"  {'n/a':>10s}"
            print(line)

        for metric, fmt in [("avg_err", "{:>7.1f}%  "), ("max_err", "{:>7.1f}%  "),
                            ("M_mean", "{:>9.4f} "), ("M_max", "{:>9.2f} ")]:
            line = f"  {metric:>12s}"
            for v in variants:
                rec = next((r for r in all_records[v] if r["tick"] == check_tick), None)
                if rec:
                    val = rec.get(metric, float('nan'))
                    line += f"  {fmt.format(val)}"
                else:
                    line += f"  {'n/a':>10s}"
            print(line)

    # -- Drift analysis --
    print(f"\n{'=' * 90}")
    print("  DRIFT ANALYSIS: tick 3000 -> 10000")
    print("=" * 90)

    for v, desc in variants.items():
        r3k = next((r for r in all_records[v] if r["tick"] == 3000), None)
        r10k = next((r for r in all_records[v] if r["tick"] == 10000), None)
        if r3k and r10k:
            drift = r10k["avg_err"] - r3k["avg_err"]
            print(f"\n  {v} ({desc}):")
            print(f"    avg_err: {r3k['avg_err']:.1f}% -> {r10k['avg_err']:.1f}% (drift: {drift:+.1f}%)")
            for key in TARGETS:
                label = LABELS[key]
                d = r10k[key + "_err"] - r3k[key + "_err"]
                marker = " ***" if abs(d) > 5 else ""
                print(f"    {label:<8s}: {r3k[key+'_err']:.1f}% -> {r10k[key+'_err']:.1f}% ({d:+.1f}%){marker}")

    # -- Head-to-head: B vs C vs D --
    print(f"\n{'=' * 90}")
    print("  HEAD-TO-HEAD: Split Mode Comparison at tick 10000")
    print("=" * 90)

    for v in ["B", "C", "D"]:
        rec = next((r for r in all_records[v] if r["tick"] == 10000), None)
        if rec:
            print(f"\n  {v}: {variants[v]}")
            for key in TARGETS:
                label = LABELS[key]
                err = rec[key + "_err"]
                print(f"    {label:<8s}: {rec[key]:.4f} (target {TARGETS[key]:.4f}, err {err:.1f}% {grade(err)})")
            print(f"    avg_err: {rec['avg_err']:.1f}%  max_err: {rec['max_err']:.1f}%")

    # -- Save --
    results_dir = os.path.join(_here, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, f"exp_13_symmetric_deact_{ts}.json")

    best_v, best_err = None, float('inf')
    for v in variants:
        rec = next((r for r in all_records[v] if r["tick"] == 10000), None)
        if rec and rec["avg_err"] < best_err:
            best_err = rec["avg_err"]
            best_v = v

    save_data = {
        "experiment": "exp_13_symmetric_deactualization",
        "date": ts,
        "grid": list(grid),
        "ticks": ticks,
        "targets": {LABELS[k]: v for k, v in TARGETS.items()},
        "variants": variants,
        "best_variant_10k": best_v,
        "best_avg_err_10k": best_err,
        "all_records": all_records,
    }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {outpath}")
    print(f"  Best variant at 10K: {best_v} ({best_err:.1f}%)")


if __name__ == "__main__":
    main()
