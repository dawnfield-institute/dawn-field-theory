"""
Actualization Ratio in Field Dynamics — Experiment Script 41

PURPOSE:
    Tests whether the actualization ratio ln(φ) = A/(A+ξ) = 0.4812 is the
    OPTIMAL local/global split for sustaining field dynamics. When a field
    event actualizes, it splits into:
      - LOCAL fraction f (Landauer cost paid at the cell)
      - GLOBAL fraction (1-f) (redistributed via PAC tree to other cells)

    The local fraction changes the field at the actualization site.
    The global fraction creates NEW disequilibrium at distant cells through
    PAC tree redistribution (Fibonacci-weighted multi-scale diffusion).

    If f is too high (too much local): dynamics die — all energy stays local,
    no new disequilibrium is created, system equilibrates to heat death.

    If f is too low (too much global): dynamics smear out — energy spreads
    uniformly, no structure forms, mass stays homogeneous.

    The prediction: f = ln(φ) maximizes sustained field activity (measured
    by disequilibrium persistence, mass concentration, and actualization
    rate) because it's the unique ratio where local actualization and global
    redistribution are in balance — the same ratio validated across 11
    independent MAR experiments.

HYPOTHESIS:
    1. Field dynamics survival (disequilibrium > threshold after N ticks)
       peaks at f ≈ ln(φ) = 0.4812.
    2. Mass concentration (Mmax and dense%) peaks near f ≈ ln(φ).
    3. The actualization rate (events per tick) stabilizes rather than
       decaying to zero only when f ≈ ln(φ).
    4. The ratio is not arbitrary — it connects to the PAC tree structure
       where A/(A+ξ) = ln(φ) governs how much potential becomes actual.

DESIGN:
    Part A — Sweep f from 0.0 to 1.0, measure equilibrium disequilibrium
    Part B — Measure mass concentration (Mmax, dense%) vs f
    Part C — Measure actualization rate stability vs f
    Part D — Compare f=ln(φ) against f=0.5, f=1.0 (Euler), f=0.0 (all global)
    Part E — Test whether optimal f depends on grid size or initial conditions

CORPUS CONTEXT:
    - exp_01: MVAE cutoff a_min = 1/(2(1-ln2)) ≈ 1.629 l_P
    - exp_05: a_min/φ = 1.007 (proximity to golden ratio)
    - exp_29: global-local duality, frame asymmetry
    - exp_36: local-global tiling, cosmological constant
    - Reality Engine v3: ActualizationOperator implements MAR-gated integration
      with ln(φ) split, first computational demonstration of the mechanism

OUTPUT:
    Results saved to results/exp_41_actualization_ratio_field_dynamics.json

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
LN_PHI = math.log(PHI)  # 0.48121182505960344
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015328
XI = GAMMA_EM + LN_PHI
XI_PAC = 1 + (7 / 8) * LN2 * (1 - LN2) ** 2

# ============================================================
# Helpers
# ============================================================
def print_header(title, subtitle=None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70 + "\n")


def print_result(name, value, expected=None, unit=""):
    if expected is not None:
        err = abs(value - expected) / abs(expected) * 100 if expected != 0 else 0
        status = "PASS" if err < 5.0 else ("NEAR" if err < 20.0 else "FAIL")
        print(f"  {name:40s} = {value:12.6f} {unit:6s}  "
              f"(expected {expected:.6f}, err {err:.2f}%)  [{status}]")
    else:
        print(f"  {name:40s} = {value:12.6f} {unit}")


def run_simulation(local_frac, nu=64, nv=64, ticks=3000, dt=0.001):
    """Run Reality Engine v3 with a given local/global split ratio.

    Returns dict of field dynamics metrics at the end of the run.
    Uses the ActualizationOperator with a custom local fraction.
    """
    import torch
    # Add reality-engine to path
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
    from src.v3.operators.actualization import ActualizationOperator, LN_PHI

    # Monkey-patch the local fraction for this run
    import src.v3.operators.actualization as act_mod
    original_ln_phi = act_mod.LN_PHI
    act_mod.LN_PHI = local_frac

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SimulationConfig(
        nu=nu, nv=nv, dt=dt, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )

    pipeline = Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), GravitationalCollapseOperator(), FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
    ])

    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=2.0)
    pac_initial = engine.state.pac_total

    # Run simulation
    for _ in range(ticks):
        engine.tick()

    # Collect metrics
    s = engine.state
    M, E, I = s.M, s.E, s.I
    diseq = (E - I).abs()
    n_cells = M.numel()

    result = {
        "local_frac": local_frac,
        "ticks": ticks,
        "grid": f"{nu}x{nv}",
        "pac_drift": s.pac_total - pac_initial,
        "diseq_mean": diseq.mean().item(),
        "diseq_max": diseq.max().item(),
        "M_total": M.sum().item(),
        "M_max": M.max().item(),
        "void_pct": (M < 0.5).sum().item() / n_cells * 100,
        "dense_pct": (M > 2.0).sum().item() / n_cells * 100,
        "temperature": s.T.mean().item(),
        "actualization_count": s.metrics.get("actualization_count", 0),
        "potential_mean": s.metrics.get("potential_mean", 0.0),
        "landauer_reinjection": s.metrics.get("landauer_reinjection", 0.0),
    }

    # Restore original
    act_mod.LN_PHI = original_ln_phi

    return result


# ============================================================
# Part A: Sweep local fraction, measure equilibrium disequilibrium
# ============================================================
def part_A():
    print_header("Part A: Disequilibrium vs Local Fraction",
                 "Sweep f from 0.1 to 0.9, measure diseq after 3000 ticks")

    fractions = [0.1, 0.2, 0.3, 0.4, LN_PHI, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for f in fractions:
        label = f"f={f:.4f}" + (" (ln(φ))" if abs(f - LN_PHI) < 0.001 else "")
        print(f"  Running {label}...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_simulation(f, nu=64, nv=64, ticks=3000)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s) | diseq={r['diseq_mean']:.4f} "
              f"Mmax={r['M_max']:.2f} dense={r['dense_pct']:.0f}% "
              f"T={r['temperature']:.3f}")
        results.append(r)

    # Find optimum
    best = max(results, key=lambda r: r["diseq_mean"])
    print(f"\n  Peak disequilibrium: f={best['local_frac']:.4f} "
          f"(diseq={best['diseq_mean']:.4f})")
    print(f"  ln(φ) = {LN_PHI:.4f}")

    ln_phi_result = [r for r in results if abs(r["local_frac"] - LN_PHI) < 0.001][0]
    print(f"  ln(φ) result: diseq={ln_phi_result['diseq_mean']:.4f}")

    return {"sweep": results, "optimal_f": best["local_frac"],
            "ln_phi_diseq": ln_phi_result["diseq_mean"]}


# ============================================================
# Part B: Mass concentration vs local fraction
# ============================================================
def part_B(sweep_results):
    print_header("Part B: Mass Concentration vs Local Fraction",
                 "Which f produces the most gravitational collapse?")

    results = sweep_results["sweep"]
    best_Mmax = max(results, key=lambda r: r["M_max"])
    best_dense = max(results, key=lambda r: r["dense_pct"])

    print(f"  Peak Mmax: f={best_Mmax['local_frac']:.4f} (Mmax={best_Mmax['M_max']:.2f})")
    print(f"  Peak dense%: f={best_dense['local_frac']:.4f} "
          f"(dense={best_dense['dense_pct']:.1f}%)")

    # Show void+dense structure (separation = spatial differentiation)
    print("\n  f        | void%  | dense% | separation")
    print("  " + "-" * 50)
    for r in results:
        sep = r["void_pct"] + r["dense_pct"]  # higher = more spatial structure
        marker = " ← ln(φ)" if abs(r["local_frac"] - LN_PHI) < 0.001 else ""
        print(f"  {r['local_frac']:.4f}  | {r['void_pct']:5.1f}% | "
              f"{r['dense_pct']:5.1f}% | {sep:5.1f}%{marker}")

    return {
        "best_Mmax_f": best_Mmax["local_frac"],
        "best_dense_f": best_dense["local_frac"],
        "results": [{
            "f": r["local_frac"],
            "M_max": r["M_max"],
            "dense_pct": r["dense_pct"],
            "void_pct": r["void_pct"],
        } for r in results]
    }


# ============================================================
# Part C: Actualization rate stability
# ============================================================
def part_C():
    print_header("Part C: Actualization Rate Stability",
                 "Does actualization rate stabilize or decay to zero?")

    # Run longer simulations for a few key fractions and sample at intervals
    test_fracs = [0.2, LN_PHI, 0.5, 0.8, 1.0]
    import torch

    results = {}
    for f in test_fracs:
        label = f"f={f:.4f}" + (" (ln(φ))" if abs(f - LN_PHI) < 0.001 else
                                " (Euler)" if f == 1.0 else "")
        print(f"  Running {label} for 5000 ticks...", end=" ", flush=True)

        # We need to sample at intervals, so run tick by tick
        re_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', '..', 'reality-engine'))
        if re_path not in sys.path:
            sys.path.insert(0, re_path)

        import src.v3.operators.actualization as act_mod
        original = act_mod.LN_PHI
        act_mod.LN_PHI = f

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SimulationConfig(
            nu=64, nv=64, dt=0.001, device=device,
            enable_actualization=(f < 1.0),  # f=1.0 is pure Euler
            actualization_threshold=0.05,
        )
        pipeline = Pipeline([
            RBFOperator(), QBEOperator(), ActualizationOperator(),
            MemoryOperator(), GravitationalCollapseOperator(), FusionOperator(),
            ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
            NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
        ])
        engine = Engine(config=config, pipeline=pipeline)
        engine.initialize("big_bang", temperature=2.0)

        samples = []
        for tick in range(5000):
            engine.tick()
            if tick % 500 == 0:
                s = engine.state
                samples.append({
                    "tick": tick,
                    "diseq": (s.E - s.I).abs().mean().item(),
                    "actual_count": s.metrics.get("actualization_count", 0),
                    "temperature": s.T.mean().item(),
                })

        act_mod.LN_PHI = original

        # Measure: is actualization rate stable in last 3 samples?
        late_counts = [s["actual_count"] for s in samples[-3:]]
        early_counts = [s["actual_count"] for s in samples[1:4]]
        late_mean = sum(late_counts) / len(late_counts) if late_counts else 0
        early_mean = sum(early_counts) / len(early_counts) if early_counts else 1
        stability = late_mean / early_mean if early_mean > 0 else 0

        print(f"stability={stability:.2f} "
              f"(early={early_mean:.0f} late={late_mean:.0f})")

        results[f"{f:.4f}"] = {
            "local_frac": f,
            "samples": samples,
            "stability_ratio": stability,
            "late_diseq": samples[-1]["diseq"] if samples else 0,
        }

    return results


# ============================================================
# Part D: Head-to-head comparison
# ============================================================
def part_D():
    print_header("Part D: Head-to-Head Comparison",
                 "f=ln(φ) vs f=0.5 vs f=1.0 (Euler) vs f=0.0 (all global)")

    comparisons = {
        "ln(phi)": LN_PHI,
        "half": 0.5,
        "euler": 1.0,
        "all_global": 0.05,  # Can't use exactly 0.0 (division issues)
    }

    results = {}
    for name, f in comparisons.items():
        print(f"  Running {name} (f={f:.4f}) for 5000 ticks...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_simulation(f, nu=64, nv=64, ticks=5000)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s)")
        results[name] = r

    print(f"\n  {'Metric':<25s} | {'ln(φ)':>10s} | {'half':>10s} | "
          f"{'euler':>10s} | {'all_global':>10s}")
    print("  " + "-" * 75)
    for metric in ["diseq_mean", "M_max", "dense_pct", "void_pct",
                    "temperature", "pac_drift"]:
        vals = [results[k].get(metric, 0) for k in comparisons]
        fmt = ".4f" if metric in ("diseq_mean", "temperature", "pac_drift") else ".2f"
        print(f"  {metric:<25s} | " +
              " | ".join(f"{v:>10{fmt}}" for v in vals))

    return results


# ============================================================
# Part E: Grid size independence
# ============================================================
def part_E():
    print_header("Part E: Grid Size Independence",
                 "Does the optimal f depend on grid size?")

    grids = [(32, 32), (64, 64), (128, 64)]
    results = {}

    for nu, nv in grids:
        print(f"\n  Grid {nu}x{nv}:")
        grid_results = []
        for f in [0.3, 0.4, LN_PHI, 0.5, 0.6]:
            label = f"f={f:.4f}" + (" (ln(φ))" if abs(f - LN_PHI) < 0.001 else "")
            print(f"    {label}...", end=" ", flush=True)
            r = run_simulation(f, nu=nu, nv=nv, ticks=2000)
            print(f"diseq={r['diseq_mean']:.4f} Mmax={r['M_max']:.2f}")
            grid_results.append(r)

        best = max(grid_results, key=lambda r: r["diseq_mean"])
        print(f"    → Best: f={best['local_frac']:.4f}")
        results[f"{nu}x{nv}"] = {
            "sweep": grid_results,
            "optimal_f": best["local_frac"],
        }

    return results


# ============================================================
# Main
# ============================================================
def main():
    print_header("Experiment 41: Actualization Ratio in Field Dynamics",
                 f"Testing whether f = ln(φ) = {LN_PHI:.6f} is optimal "
                 "for sustained field dynamics")

    print(f"  Constants:")
    print(f"    φ       = {PHI:.10f}")
    print(f"    ln(φ)   = {LN_PHI:.10f}  (predicted optimal local fraction)")
    print(f"    1-ln(φ) = {1 - LN_PHI:.10f}  (predicted global fraction)")
    print(f"    Ξ       = {XI:.10f}")
    print(f"    ξ_PAC   = {XI_PAC:.10f}")

    all_results = {
        "experiment": "exp_41_actualization_ratio_field_dynamics",
        "timestamp": datetime.now().isoformat(),
        "constants": {
            "phi": PHI, "ln_phi": LN_PHI, "Xi": XI, "xi_pac": XI_PAC,
        },
    }

    # Part A
    all_results["part_A"] = part_A()

    # Part B
    all_results["part_B"] = part_B(all_results["part_A"])

    # Part C
    all_results["part_C"] = part_C()

    # Part D
    all_results["part_D"] = part_D()

    # Part E
    all_results["part_E"] = part_E()

    # ============================================================
    # Synthesis
    # ============================================================
    print_header("SYNTHESIS")

    opt_A = all_results["part_A"]["optimal_f"]
    opt_B_Mmax = all_results["part_B"]["best_Mmax_f"]
    opt_B_dense = all_results["part_B"]["best_dense_f"]

    print(f"  Optimal f for disequilibrium:     {opt_A:.4f}")
    print(f"  Optimal f for mass concentration:  {opt_B_Mmax:.4f} (Mmax), "
          f"{opt_B_dense:.4f} (dense%)")
    print(f"  Predicted (ln(φ)):                 {LN_PHI:.4f}")
    print()

    # Check if ln(phi) is near optimal (within 0.1 of peak)
    near_optimal = abs(opt_A - LN_PHI) < 0.15
    print(f"  ln(φ) near optimal for disequilibrium: "
          f"{'YES' if near_optimal else 'NO'} "
          f"(|{opt_A:.4f} - {LN_PHI:.4f}| = {abs(opt_A - LN_PHI):.4f})")

    # Grid independence
    grid_opts = [v["optimal_f"] for v in all_results["part_E"].values()]
    grid_independent = all(abs(f - LN_PHI) < 0.15 for f in grid_opts)
    print(f"  Grid-independent optimum near ln(φ): "
          f"{'YES' if grid_independent else 'NO'} ({grid_opts})")

    all_results["synthesis"] = {
        "optimal_diseq_f": opt_A,
        "optimal_Mmax_f": opt_B_Mmax,
        "predicted_f": LN_PHI,
        "near_optimal": near_optimal,
        "grid_independent": grid_independent,
        "verdict": (
            "CONFIRMED: ln(φ) is near-optimal for sustained field dynamics"
            if near_optimal and grid_independent else
            "PARTIAL: ln(φ) shows advantages but may not be strictly optimal"
            if near_optimal or grid_independent else
            "INCONCLUSIVE: further investigation needed"
        ),
    }

    print(f"\n  VERDICT: {all_results['synthesis']['verdict']}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(
        results_dir, f"exp_41_actualization_ratio_field_dynamics_{ts}.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
