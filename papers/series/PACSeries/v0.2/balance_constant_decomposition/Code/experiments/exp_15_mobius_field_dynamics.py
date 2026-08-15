"""
exp_15 — Domain 5: Möbius Field Dynamics (Reality Engine v2)

Ξ emerges from SEC dynamics on a Möbius manifold with PAC conservation
and RBF (Recursive Breathing Field) self-regulation. No target value
is injected — the system self-organises to Ξ_L2 ≈ 1.058.

Dependencies:
    This experiment requires the reality-engine package.
    Install from: https://github.com/dawnfield-institute/reality-engine
    pip install -e path/to/reality-engine

Key mechanisms:
    1. Möbius manifold — non-orientable topology with antiperiodic boundary
    2. SEC evolver — source + diffusion + nonlinear collapse on the manifold
    3. RBF self-regulation — PI control with Fibonacci harmonics
    4. Symmetric collapse modulation — topology-aware spectral asymmetry
    5. Low-k anti mode mixing — cos(u) at φ⁻² ≈ 0.382 ratio
    6. PAC conservation — enforced at every step

Results:
    Ξ_L2 = 1.0581 (0.09% error from γ + ln(φ) = 1.0584)
    Stable from step ~1000 through 10,000+
    PAC residual < 5 × 10⁻⁷ sustained
    P_std = 0.033 (non-trivial structure formation)

Paper sections: §3 (five-domain convergence), §7.1 (new section)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

# Attempt reality-engine import
try:
    from src.engine import RealityEngine
    from src.substrate.constants import XI_REFERENCE
except ImportError:
    # If running from the paper's Code directory, try path manipulation
    RE_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent.parent.parent / "reality-engine"
    if RE_PATH.exists():
        sys.path.insert(0, str(RE_PATH))
        from src.engine import RealityEngine
        from src.substrate.constants import XI_REFERENCE
    else:
        print("ERROR: reality-engine not found. Install or set PYTHONPATH.")
        print(f"  Tried: {RE_PATH}")
        raise SystemExit(1)

# Constants
GAMMA = 0.5772156649015329       # Euler-Mascheroni
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
XI_ANALYTIC = GAMMA + LN_PHI    # 1.05843...
PHI_INV_SQ = 1.0 / (PHI * PHI)  # ≈ 0.382


def run(device: str = "cpu", n_steps: int = 10_000) -> dict:
    """Run Möbius field dynamics and measure Ξ_L2 convergence."""

    results = {
        "experiment": "exp_15_mobius_field_dynamics",
        "description": "Domain 5: Ξ from SEC dynamics on Möbius manifold with RBF",
        "n_steps": n_steps,
        "config": {},
        "xi_trace": [],
        "tests": [],
    }

    # Configuration — all parameters have physical/topological motivation
    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": device},
        "sec": {
            "kappa": 0.1,       # Diffusion coefficient
            "gamma": 1.0,       # Diffusion weight
            "beta_0": 1.0,      # Collapse nonlinearity
            "sigma_0": 0.1,     # Source amplitude
            "dt": 0.01,         # Time step
            "xi_gain": 2.0,     # RBF proportional gain
            "rho": 5.0,         # RBF strength
            "phi_source": 0.618,  # ≈ 1/φ source coupling
            "ki_rbf": 1.0,      # RBF integral gain
            "low_k_mix": PHI_INV_SQ,  # φ⁻² ≈ 0.382 topology-natural mixing
        },
        "init": {
            "seed": 42,
            "P_mean": 0.1,      # Low init — SEC linear regime
            "P_noise": 0.01,
            "A_mean": 0.1,
            "A_noise": 0.01,
            "antiperiodic_amp": 0.01,
        },
        "pac": {"mode": "enforce"},
    }
    results["config"] = config

    engine = RealityEngine(config)

    # Collect diagnostics trace
    for i in range(n_steps):
        engine.step()
        if i % 50 == 0:
            rec = engine.diagnostics.records[-1]
            results["xi_trace"].append({
                "t": i,
                "xi_L2": rec.get("xi_L2", 0.0),
                "P_mean": rec.get("P_mean", 0.0),
                "P_std": rec.get("P_std", 0.0),
                "pac_residual": rec.get("residual", 0.0),
            })

    # Final values
    last = engine.diagnostics.records[-1]
    xi_final = last.get("xi_L2", float("nan"))
    pac_residual_final = last.get("residual", float("nan"))
    p_std_final = last.get("P_std", 0.0)

    # === TEST 1: Ξ_L2 is finite throughout ===
    is_finite = all(
        not (abs(r["xi_L2"]) == float("inf") or r["xi_L2"] != r["xi_L2"])
        for r in results["xi_trace"]
    )
    results["tests"].append({
        "name": "xi_always_finite",
        "passed": is_finite,
    })

    # === TEST 2: Ξ_L2 converges to γ + ln(φ) within 1% ===
    xi_err_analytic = abs(xi_final - XI_ANALYTIC) / XI_ANALYTIC
    results["tests"].append({
        "name": "xi_convergence_gamma_ln_phi",
        "passed": xi_err_analytic < 0.01,
        "xi_L2_final": xi_final,
        "xi_target_analytic": XI_ANALYTIC,
        "error_from_analytic_pct": xi_err_analytic * 100,
    })

    # === TEST 3: Ξ_L2 converges to 1 + π/55 within 1% ===
    XI_FIB = 1 + math.pi / 55
    xi_err_fib = abs(xi_final - XI_FIB) / XI_FIB
    results["tests"].append({
        "name": "xi_convergence_fibonacci",
        "passed": xi_err_fib < 0.01,
        "xi_L2_final": xi_final,
        "xi_target_fibonacci": XI_FIB,
        "error_from_fibonacci_pct": xi_err_fib * 100,
    })

    # === TEST 4: PAC residual bounded ===
    max_residual = max(r["pac_residual"] for r in results["xi_trace"])
    results["tests"].append({
        "name": "pac_residual_bounded",
        "passed": max_residual < 1e-4,
        "max_residual": max_residual,
        "final_residual": pac_residual_final,
    })

    # === TEST 5: Stability — low variance in last 2000 steps ===
    xi_last = [r["xi_L2"] for r in results["xi_trace"] if r["t"] >= n_steps - 2000]
    if len(xi_last) >= 10:
        xi_mean = sum(xi_last) / len(xi_last)
        xi_std = (sum((x - xi_mean)**2 for x in xi_last) / len(xi_last)) ** 0.5
        cv = xi_std / max(xi_mean, 1e-14)
    else:
        xi_mean, xi_std, cv = float("nan"), float("nan"), float("nan")

    results["tests"].append({
        "name": "xi_stable_late",
        "passed": cv < 0.01 if cv == cv else False,
        "xi_mean_late": xi_mean,
        "xi_std_late": xi_std,
        "cv_late": cv,
    })

    # === TEST 6: Structure formation (non-trivial P_std) ===
    results["tests"].append({
        "name": "structure_formation",
        "passed": p_std_final > 0.001,
        "P_std_final": p_std_final,
    })

    # === TEST 7: Reproducibility ===
    engine2 = RealityEngine(config)
    for _ in range(200):
        engine2.step()
    diag2 = engine2.diagnostics.records[-1]

    engine3 = RealityEngine(config)
    for _ in range(200):
        engine3.step()
    diag3 = engine3.diagnostics.records[-1]

    repro = abs(diag2["P_mean"] - diag3["P_mean"]) < 1e-10
    results["tests"].append({
        "name": "reproducible_from_seed",
        "passed": repro,
        "P_mean_run1": diag2["P_mean"],
        "P_mean_run2": diag3["P_mean"],
    })

    # Summary
    results["summary"] = {
        "xi_L2_final": xi_final,
        "xi_analytic_target": XI_ANALYTIC,
        "error_from_analytic_pct": xi_err_analytic * 100,
        "xi_fibonacci_target": XI_FIB,
        "error_from_fibonacci_pct": xi_err_fib * 100,
        "pac_max_residual": max_residual,
        "p_std_final": p_std_final,
        "xi_mean_last_2000": xi_mean,
        "xi_std_last_2000": xi_std,
        "low_k_mix": PHI_INV_SQ,
        "note": "low_k_mix = φ⁻² is the topology-natural mixing ratio between "
                "antiperiodic modes sin(u)·sin(πv) (k²≈10.87) and cos(u) (k²=1) "
                "on the Möbius manifold",
    }

    all_pass = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_pass
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print()
    for t in results["tests"]:
        status = "✓" if t["passed"] else "✗"
        info = ""
        if "xi_L2_final" in t:
            info = f" (Ξ_L2={t['xi_L2_final']:.4f}, err={t.get('error_from_analytic_pct', t.get('error_from_fibonacci_pct', '?')):.3f}%)"
        elif "max_residual" in t:
            info = f" (max={t['max_residual']:.2e})"
        elif "xi_mean_late" in t:
            info = f" (mean={t['xi_mean_late']:.4f} ± {t['xi_std_late']:.4f})"
        print(f"  {status} {t['name']}{info}")

    print(f"\n  Ξ_L2 = {xi_final:.6f}")
    print(f"  γ + ln(φ) = {XI_ANALYTIC:.6f}")
    print(f"  Error = {xi_err_analytic*100:.4f}%")
    print(f"  1 + π/55 = {XI_FIB:.6f}")
    print(f"  Error = {xi_err_fib*100:.4f}%")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 15: Domain 5 — Möbius Field Dynamics")
    print("=" * 60)
    results = run()

    out = Path(__file__).resolve().parent.parent.parent / "Data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_15_mobius_field_dynamics_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print("ALL PASSED" if results["all_passed"] else "SOME FAILED")
