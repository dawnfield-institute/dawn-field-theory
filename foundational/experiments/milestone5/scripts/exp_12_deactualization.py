#!/usr/bin/env python3
"""
DFT Milestone 5 — Experiment 12: De-Actualization (Memory Fading)

From infodynamics.md: "potential isn't only energy — it is ontological,
epistemic, and structural potential waiting to crystallize under recursive
field conditions."

From dawn-field-theory.md: M(x,t) is "Recursive memory of imbalance."

The PAC cycle should be COMPLETE:
  potential -> actualization -> memory -> potential (when conditions change)

Currently the MemoryOperator has:
  - Mass GENERATION: gamma_local * (E-I)^2  (imbalance -> memory)
  - Quantum pressure + diffusion (redistributes memory)
  - Hard cap at M_cap = field_scale/5  (prevents overflow)

What's MISSING: de-actualization — memory fading back to potential.
When gamma_local is LOW (E ≈ I, balanced, nothing to remember),
M should dissolve back into E + I.

De-actualization rate:
  dM_deact = -eta * M * (1 - gamma_local)

Where:
  - eta = de-actualization rate coefficient
  - M = mass to dissolve (more mass -> faster dissolution)
  - (1 - gamma_local) = "forgetting factor" — high when E ≈ I (balanced),
    zero when disequilibrium is maximal

The dissolved mass returns equally to E and I (PAC conserving):
  dE = -dM_deact / 2
  dI = -dM_deact / 2

This experiment tests:
  A) Baseline (current: hard cap, no de-actualization)
  B) De-actualization only (add fading, keep hard cap)
  C) De-actualization + remove hard cap (let fading be the natural brake)
  D) De-actualization + remove hard cap + higher eta (faster fading)
  E) De-actualization + remove hard cap + gamma_local^2 forgetting (gentler)
"""

import json, math, os, sys, time, copy
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

# ── DFT targets ──────────────────────────────────────────────────
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


# ── De-Actualization wrapper ─────────────────────────────────────
_EPS = 1e-12


class DeActualizationMemoryOperator:
    """MemoryOperator wrapper that adds de-actualization (memory fading).

    After the standard MemoryOperator runs, applies:
      dM_deact = -eta * M * forgetting_factor
      forgetting_factor = (1 - gamma_local)^power

    Dissolved mass returns equally to E and I (PAC conserving).
    Optionally removes the hard M_cap by setting it very high.
    """

    def __init__(self, eta=0.01, power=1.0, remove_cap=False):
        self._inner = MemoryOperator()
        self.eta = eta
        self.power = power
        self.remove_cap = remove_cap

    @property
    def name(self):
        return "memory"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        # Optionally disable hard cap by temporarily raising it
        if self.remove_cap:
            orig_scale = config.field_scale
            config.field_scale = 1e6  # effectively no cap (M_cap = 1e6/5 = 200000)

        # Run standard memory operator
        state = self._inner(state, config, bus)

        if self.remove_cap:
            config.field_scale = orig_scale

        # De-actualization: memory fading where disequilibrium is low
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        # gamma_local = (E-I)^2 / (E^2 + I^2 + eps) — same as MemoryOperator
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Forgetting factor: high where balanced (gamma_local low), zero where imbalanced
        forgetting = (1.0 - gamma_local).pow(self.power)

        # De-actualization rate: mass dissolves proportional to itself and forgetting
        dM_deact = -self.eta * M * forgetting * dt

        # Apply (M can't go below zero)
        M_new = torch.clamp(M + dM_deact, min=0.0)
        dM_actual = M_new - M  # negative (mass removed)

        # PAC conservation: dissolved mass returns equally to E and I
        returned = -dM_actual * 0.5
        E_new = E + returned
        I_new = I + returned

        # Track de-actualization metrics
        metrics = dict(state.metrics)
        metrics["deactualization_rate"] = (-dM_deact).mean().item()
        metrics["deactualization_total"] = (-dM_actual).sum().item()
        metrics["forgetting_mean"] = forgetting.mean().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


def build_pipeline(variant):
    """Build pipeline with variant-specific memory operator."""
    if variant == "A":
        mem_op = MemoryOperator()
    elif variant == "B":
        mem_op = DeActualizationMemoryOperator(eta=0.01, power=1.0, remove_cap=False)
    elif variant == "C":
        mem_op = DeActualizationMemoryOperator(eta=0.01, power=1.0, remove_cap=True)
    elif variant == "D":
        mem_op = DeActualizationMemoryOperator(eta=0.05, power=1.0, remove_cap=True)
    elif variant == "E":
        mem_op = DeActualizationMemoryOperator(eta=0.01, power=2.0, remove_cap=True)
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
    """Run one variant and return sampled records."""
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
            record["EI_ratio"] = st.E.mean().item() / max(st.I.mean().item(), 1e-12)
            record["pac_total"] = (st.E + st.I + st.M).sum().item()
            record["deact_rate"] = m.get("deactualization_rate", 0)
            record["forgetting"] = m.get("forgetting_mean", 0)

            records.append(record)

            if tick % 2000 == 0:
                elapsed = time.time() - t0
                print(f"    [{variant}] tick {tick:5d}  avg={record['avg_err']:.1f}%  "
                      f"M={record['M_mean']:.4f}  M_max={record['M_max']:.2f}  "
                      f"EI={record['EI_ratio']:.3f}  [{elapsed:.0f}s]")

    return records


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = (128, 64)
    ticks = 10000
    sample_ticks = set(range(500, ticks + 1, 500))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    variants = {
        "A": "Baseline (hard cap, no de-actualization)",
        "B": "De-actualization eta=0.01, keep cap",
        "C": "De-actualization eta=0.01, no cap",
        "D": "De-actualization eta=0.05, no cap (faster)",
        "E": "De-actualization eta=0.01, power=2, no cap (gentler)",
    }

    print("=" * 90)
    print("  DFT Experiment 12: De-Actualization (PAC Cycle Completion)")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks")
    print("=" * 90)
    print()
    print("  Theory: M is 'recursive memory of imbalance' (DFT §3).")
    print("  When imbalance resolves (gamma_local -> 0), memory should fade.")
    print("  De-actualization rate: dM = -eta * M * (1 - gamma_local)^p * dt")
    print("  Dissolved mass returns equally to E and I (PAC conserving).")
    print()

    all_records = {}
    for v, desc in variants.items():
        print(f"  --- Variant {v}: {desc} ---")
        records = run_variant(v, device, grid, ticks, sample_ticks)
        all_records[v] = records
        print()

    # ── Comparison at key ticks ──────────────────────────────────
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

    # ── Detailed comparison at tick 5000 and 10000 ───────────────
    for check_tick in [5000, 10000]:
        print(f"\n{'=' * 90}")
        print(f"  DETAILED COMPARISON at tick {check_tick}")
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

        # Summary metrics
        for metric, fmt in [("avg_err", "{:>7.1f}%  "), ("max_err", "{:>7.1f}%  "),
                            ("M_mean", "{:>9.4f} "), ("M_max", "{:>9.2f} "),
                            ("EI_ratio", "{:>9.3f} ")]:
            line = f"  {metric:>12s}"
            for v in variants:
                rec = next((r for r in all_records[v] if r["tick"] == check_tick), None)
                if rec:
                    val = rec.get(metric, float('nan'))
                    line += f"  {fmt.format(val)}"
                else:
                    line += f"  {'n/a':>10s}"
            print(line)

    # ── Late-time drift analysis ─────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  DRIFT ANALYSIS: error change from tick 3000 -> 10000")
    print("=" * 90)

    for v, desc in variants.items():
        r3k = next((r for r in all_records[v] if r["tick"] == 3000), None)
        r10k = next((r for r in all_records[v] if r["tick"] == 10000), None)
        if r3k and r10k:
            drift = r10k["avg_err"] - r3k["avg_err"]
            print(f"  {v}: avg_err {r3k['avg_err']:.1f}% -> {r10k['avg_err']:.1f}% "
                  f"(drift: {drift:+.1f}%)  M_max: {r3k['M_max']:.2f} -> {r10k['M_max']:.2f}")

            # Per-coupling drift
            for key in TARGETS:
                label = LABELS[key]
                d = r10k[key + "_err"] - r3k[key + "_err"]
                print(f"    {label:<8s}: {r3k[key+'_err']:.1f}% -> {r10k[key+'_err']:.1f}% ({d:+.1f}%)")
            print()

    # ── Mass dynamics ────────────────────────────────────────────
    print("=" * 90)
    print("  MASS DYNAMICS")
    print("=" * 90)

    for v in variants:
        print(f"\n  Variant {v}:")
        print(f"    {'tick':>6s} {'M_mean':>9s} {'M_max':>9s} {'deact':>10s} {'forget':>8s}")
        for r in all_records[v]:
            if r["tick"] % 2000 == 0:
                print(f"    {r['tick']:>6d} {r['M_mean']:>9.5f} {r['M_max']:>9.3f} "
                      f"{r['deact_rate']:>10.6f} {r['forgetting']:>8.4f}")

    # ── Save ─────────────────────────────────────────────────────
    results_dir = os.path.join(_here, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, f"exp_12_deactualization_{ts}.json")

    # Find best variant at tick 10000
    best_v = None
    best_err = float('inf')
    for v in variants:
        rec = next((r for r in all_records[v] if r["tick"] == 10000), None)
        if rec and rec["avg_err"] < best_err:
            best_err = rec["avg_err"]
            best_v = v

    save_data = {
        "experiment": "exp_12_deactualization",
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
