#!/usr/bin/env python3
"""
DFT Milestone 5 — Experiment 11: Coupling Trade-Off Analysis

The scorecard revealed a multi-attractor competition:
  - f_local converges by tick 1000 (1.5%) then drifts to 11.4% by tick 10000
  - G_local converges by tick 5000 (3.5%) then drifts to 10.3% by tick 10000
  - gamma/alpha/lambda slowly improve throughout

This experiment:
1. Runs 15K ticks with CORRECT metrics (from operator metrics, not raw fields)
2. Samples every 100 ticks for fine-grained coupling evolution
3. Finds the tick of minimum SIMULTANEOUS error (all 5 couplings)
4. Analyzes the coupling correlation structure (which couplings trade off?)
5. Tracks mass field statistics to understand the drift mechanism
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

# ── DFT targets (matching scorecard exactly) ───────────────────────
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015328

TARGETS = {
    "f_local_mean":      GAMMA_EM,       # 0.5772
    "gamma_local_mean":  1.0 / PHI,      # 0.6180
    "alpha_local_mean":  LN2,            # 0.6931
    "G_local_mean":      1.0 / PHI**2,   # 0.3820
    "lambda_local_mean": 1.0 - LN2,      # 0.3069
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = (128, 64)
    ticks = 15000
    sample_interval = 100
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 90)
    print("  DFT Experiment 11: Coupling Trade-Off Analysis")
    print(f"  Device: {device} | Grid: {grid[0]}x{grid[1]} | {ticks} ticks | sample every {sample_interval}")
    print("=" * 90)

    # Build standard pipeline
    pipe = Pipeline([
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
        NormalizationOperator(),
        SECTrackingOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])

    cfg = SimulationConfig(nu=grid[0], nv=grid[1], device=device,
                           dt=0.001, enable_actualization=True,
                           actualization_threshold=0.05)
    torch.manual_seed(42)
    eng = Engine(config=cfg, pipeline=pipe)
    eng.initialize("big_bang", temperature=3.0)

    history = []
    t0 = time.time()

    for tick in range(1, ticks + 1):
        eng.tick()
        if tick % sample_interval == 0:
            st = eng.state
            m = st.metrics

            # Read couplings from operator metrics (CORRECT method)
            record = {"tick": tick}
            for key in TARGETS:
                val = m.get(key, None)
                if val is not None:
                    record[key] = val
                    record[key + "_err"] = pct_err(val, TARGETS[key])
                else:
                    record[key] = float('nan')
                    record[key + "_err"] = float('nan')

            # Compute avg error (excluding NaN)
            errs = [record[k + "_err"] for k in TARGETS if not math.isnan(record.get(k + "_err", float('nan')))]
            record["avg_err"] = sum(errs) / len(errs) if errs else float('nan')

            # Field statistics
            record["E_mean"] = st.E.mean().item()
            record["I_mean"] = st.I.mean().item()
            record["M_mean"] = st.M.mean().item()
            record["M_max"] = st.M.max().item()
            record["pac_total"] = (st.E + st.I + st.M).sum().item()
            record["mass_gen_rate"] = m.get("mass_generation_rate", 0)
            record["actualization_count"] = m.get("actualization_count", 0)

            history.append(record)

            if tick % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  tick {tick:6d}  avg={record['avg_err']:.1f}%  "
                      f"f={record.get('f_local_mean', 0):.4f}  "
                      f"gamma={record.get('gamma_local_mean', 0):.4f}  "
                      f"G={record.get('G_local_mean', 0):.4f}  "
                      f"M_mean={record['M_mean']:.4f}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n  Simulation complete: {elapsed:.0f}s")

    # ── Find optimal tick ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  OPTIMAL TICK ANALYSIS")
    print("=" * 90)

    # Method 1: Minimum average error
    valid = [r for r in history if not math.isnan(r["avg_err"])]
    if valid:
        best_avg = min(valid, key=lambda r: r["avg_err"])
        print(f"\n  Min avg_err: tick {best_avg['tick']} ({best_avg['avg_err']:.2f}%)")
        for key in TARGETS:
            label = LABELS[key]
            print(f"    {label:<8s}: {best_avg[key]:.4f} ({best_avg[key + '_err']:.1f}% {grade(best_avg[key + '_err'])})")

    # Method 2: Minimum max-error (all couplings below threshold)
    def max_err(r):
        return max(r[k + "_err"] for k in TARGETS if not math.isnan(r.get(k + "_err", float('nan'))))

    valid_max = [r for r in valid if not any(math.isnan(r.get(k + "_err", float('nan'))) for k in TARGETS)]
    if valid_max:
        best_max = min(valid_max, key=max_err)
        print(f"\n  Min max_err: tick {best_max['tick']} (worst coupling: {max_err(best_max):.2f}%)")
        for key in TARGETS:
            label = LABELS[key]
            print(f"    {label:<8s}: {best_max[key]:.4f} ({best_max[key + '_err']:.1f}% {grade(best_max[key + '_err'])})")

    # Method 3: All below 10% threshold
    below_10 = [r for r in valid_max if max_err(r) < 10.0]
    if below_10:
        print(f"\n  Ticks where ALL couplings < 10% error: {len(below_10)}")
        first = below_10[0]
        last = below_10[-1]
        print(f"    Range: tick {first['tick']} - {last['tick']}")
    else:
        # Find best window
        below_15 = [r for r in valid_max if max_err(r) < 15.0]
        print(f"\n  No tick achieves all couplings < 10% error")
        print(f"  Ticks where all < 15%: {len(below_15)}")
        if below_15:
            print(f"    Range: tick {below_15[0]['tick']} - {below_15[-1]['tick']}")

    # ── Coupling evolution table (every 1000 ticks) ────────────────
    print("\n" + "=" * 90)
    print("  COUPLING EVOLUTION (% error)")
    print("=" * 90)
    print(f"  {'tick':>6s} {'f_local':>9s} {'gamma':>9s} {'alpha':>9s} {'G_local':>9s} {'lambda':>9s} {'avg':>8s} {'M_mean':>8s}")
    print("  " + "-" * 70)

    for r in history:
        if r["tick"] % 1000 == 0:
            parts = [f"  {r['tick']:>6d}"]
            for key in TARGETS:
                e = r.get(key + "_err", float('nan'))
                if not math.isnan(e):
                    parts.append(f"{e:>7.1f}%{grade(e)}")
                else:
                    parts.append(f"{'n/a':>9s}")
            parts.append(f"{r['avg_err']:>7.1f}%")
            parts.append(f"{r['M_mean']:>8.4f}")
            print(" ".join(parts))

    # ── Correlation analysis ───────────────────────────────────────
    print("\n" + "=" * 90)
    print("  COUPLING CORRELATION (do couplings trade off?)")
    print("=" * 90)

    # Simple: compute pairwise correlation of error time series
    keys = list(TARGETS.keys())
    n = len(history)
    if n > 10:
        series = {}
        for k in keys:
            series[k] = [r[k + "_err"] for r in history if not math.isnan(r.get(k + "_err", float('nan')))]

        print(f"\n  Pairwise correlation of error trajectories (n={n}):")
        print(f"  {'':>10s}", end="")
        for k in keys:
            print(f" {LABELS[k]:>8s}", end="")
        print()

        for k1 in keys:
            print(f"  {LABELS[k1]:>10s}", end="")
            for k2 in keys:
                s1, s2 = series[k1], series[k2]
                if len(s1) == len(s2) and len(s1) > 2:
                    m1 = sum(s1) / len(s1)
                    m2 = sum(s2) / len(s2)
                    cov = sum((a - m1) * (b - m2) for a, b in zip(s1, s2)) / len(s1)
                    std1 = (sum((a - m1)**2 for a in s1) / len(s1))**0.5
                    std2 = (sum((b - m2)**2 for b in s2) / len(s2))**0.5
                    if std1 > 0 and std2 > 0:
                        corr = cov / (std1 * std2)
                        print(f" {corr:>8.3f}", end="")
                    else:
                        print(f" {'n/a':>8s}", end="")
                else:
                    print(f" {'n/a':>8s}", end="")
            print()

    # ── Mass accumulation rate ─────────────────────────────────────
    print("\n" + "=" * 90)
    print("  MASS ACCUMULATION")
    print("=" * 90)
    print(f"  {'tick':>6s} {'M_mean':>9s} {'M_max':>9s} {'dM/dt':>10s} {'mass_gen':>10s} {'actualize':>10s}")
    print("  " + "-" * 60)

    prev_M = None
    for r in history:
        if r["tick"] % 1000 == 0:
            dM = (r["M_mean"] - prev_M) / (1000 * 0.001) if prev_M is not None else 0
            print(f"  {r['tick']:>6d} {r['M_mean']:>9.5f} {r['M_max']:>9.4f} {dM:>+10.6f} "
                  f"{r['mass_gen_rate']:>10.6f} {r['actualization_count']:>10.0f}")
            prev_M = r["M_mean"]

    # ── Save ───────────────────────────────────────────────────────
    results_dir = os.path.join(_here, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, f"exp_11_coupling_tradeoff_{ts}.json")
    save_data = {
        "experiment": "exp_11_coupling_tradeoff",
        "date": ts,
        "grid": list(grid),
        "ticks": ticks,
        "targets": {LABELS[k]: v for k, v in TARGETS.items()},
        "history": history,
        "best_avg_tick": best_avg["tick"] if valid else None,
        "best_avg_err": best_avg["avg_err"] if valid else None,
    }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {outpath}")


if __name__ == "__main__":
    main()
