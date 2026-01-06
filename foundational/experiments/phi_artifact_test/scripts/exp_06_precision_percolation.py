"""
Exp 06: Precision Percolation Analysis

Focus on percolation threshold - our cleanest information-theoretic signal.

Known exact value: p_c = 0.59274605... for 2D square lattice site percolation
This is remarkably close to 1/φ = 0.6180339...

Questions:
1. How precisely can we measure p_c?
2. Is it actually close to 1/φ, or is that coincidence?
3. What's the statistical confidence on the difference?

Key insight: p_c ≈ 0.5927 is NOT exactly 1/φ ≈ 0.6180
The difference is about 0.025 - is this significant?
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from scipy import stats
from scipy.optimize import brentq, curve_fit
import warnings
warnings.filterwarnings('ignore')


PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # 0.6180339...
KNOWN_PC = 0.59274605  # Exact 2D square lattice site percolation threshold


def check_percolation(grid: np.ndarray) -> bool:
    """Check if there's a spanning cluster from top to bottom."""
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    # Start from all occupied sites in top row
    stack = [(0, j) for j in range(cols) if grid[0, j]]
    
    while stack:
        i, j = stack.pop()
        if i < 0 or i >= rows or j < 0 or j >= cols:
            continue
        if visited[i, j] or not grid[i, j]:
            continue
        visited[i, j] = True
        if i == rows - 1:
            return True
        stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
    
    return False


def measure_percolation_probability(p: float, L: int, n_samples: int) -> Tuple[float, float]:
    """
    Measure percolation probability at occupation p.
    Returns (probability, standard error).
    """
    successes = 0
    for _ in range(n_samples):
        grid = np.random.random((L, L)) < p
        if check_percolation(grid):
            successes += 1
    
    prob = successes / n_samples
    se = np.sqrt(prob * (1 - prob) / n_samples)
    return prob, se


def percolation_curve(p: np.ndarray, pc: float, nu: float) -> np.ndarray:
    """
    Theoretical percolation curve near threshold.
    P(p) ~ (p - pc)^beta for p > pc, with finite-size corrections.
    Using a smooth sigmoid approximation for fitting.
    """
    # Sigmoid approximation
    return 1 / (1 + np.exp(-nu * (p - pc)))


def find_threshold_bisection(L: int, n_samples: int, tol: float = 0.001) -> Tuple[float, float]:
    """
    Find percolation threshold using bisection.
    Returns (threshold, uncertainty).
    """
    p_low, p_high = 0.4, 0.8
    
    while p_high - p_low > tol:
        p_mid = (p_low + p_high) / 2
        prob, _ = measure_percolation_probability(p_mid, L, n_samples)
        
        if prob < 0.5:
            p_low = p_mid
        else:
            p_high = p_mid
    
    return (p_low + p_high) / 2, (p_high - p_low) / 2


def precision_sweep(
    L: int,
    n_samples: int,
    p_range: np.ndarray
) -> Dict:
    """
    High-precision sweep around expected threshold.
    """
    probs = []
    errors = []
    
    for p in p_range:
        prob, se = measure_percolation_probability(p, L, n_samples)
        probs.append(prob)
        errors.append(se)
    
    probs = np.array(probs)
    errors = np.array(errors)
    
    # Find threshold via fitting
    try:
        popt, pcov = curve_fit(
            percolation_curve,
            p_range,
            probs,
            p0=[0.59, 10],
            bounds=([0.5, 1], [0.7, 100])
        )
        pc_fit = popt[0]
        pc_err = np.sqrt(pcov[0, 0])
    except:
        # Fallback: steepest point
        deriv = np.gradient(probs, p_range)
        max_idx = np.argmax(deriv)
        pc_fit = p_range[max_idx]
        pc_err = (p_range[1] - p_range[0]) / 2
    
    return {
        "p_range": p_range.tolist(),
        "probabilities": probs.tolist(),
        "errors": errors.tolist(),
        "pc_estimate": float(pc_fit),
        "pc_error": float(pc_err),
        "L": L,
        "n_samples": n_samples
    }


def run_precision_analysis() -> Dict:
    """
    High-precision analysis of percolation threshold.
    """
    print("=" * 70)
    print("EXP 06: PRECISION PERCOLATION ANALYSIS")
    print("=" * 70)
    print()
    print(f"Known exact p_c = {KNOWN_PC:.8f}")
    print(f"1/φ            = {PHI_INV:.8f}")
    print(f"Difference     = {abs(KNOWN_PC - PHI_INV):.8f}")
    print()
    
    results = {
        "known_values": {
            "pc_exact": KNOWN_PC,
            "phi_inverse": PHI_INV,
            "absolute_difference": abs(KNOWN_PC - PHI_INV),
            "relative_difference": abs(KNOWN_PC - PHI_INV) / PHI_INV
        },
        "measurements": {},
        "finite_size_scaling": {},
        "statistical_tests": {}
    }
    
    # High-resolution sweep for multiple system sizes
    sizes = [25, 50, 75, 100, 150]
    n_samples = 500
    p_range = np.linspace(0.55, 0.65, 51)
    
    print("Running high-precision sweeps...")
    pc_estimates = []
    pc_errors = []
    
    for L in sizes:
        print(f"  L = {L}...")
        sweep = precision_sweep(L, n_samples, p_range)
        results["measurements"][f"L{L}"] = sweep
        pc_estimates.append(sweep["pc_estimate"])
        pc_errors.append(sweep["pc_error"])
        print(f"    p_c = {sweep['pc_estimate']:.6f} ± {sweep['pc_error']:.6f}")
    
    pc_estimates = np.array(pc_estimates)
    pc_errors = np.array(pc_errors)
    sizes = np.array(sizes)
    
    # Finite-size scaling: p_c(L) = p_c(∞) + a/L^(1/ν)
    # For 2D percolation, ν = 4/3
    nu = 4/3
    
    def fss_model(L, pc_inf, a):
        return pc_inf + a / (L ** (1/nu))
    
    try:
        popt, pcov = curve_fit(
            fss_model,
            sizes,
            pc_estimates,
            sigma=pc_errors,
            p0=[0.59, 0.1],
            bounds=([0.5, -1], [0.7, 1])
        )
        pc_extrapolated = popt[0]
        pc_extrap_err = np.sqrt(pcov[0, 0])
    except:
        # Simple weighted average as fallback
        weights = 1 / (pc_errors ** 2)
        pc_extrapolated = np.average(pc_estimates, weights=weights)
        pc_extrap_err = 1 / np.sqrt(np.sum(weights))
    
    results["finite_size_scaling"] = {
        "pc_estimates": pc_estimates.tolist(),
        "pc_errors": pc_errors.tolist(),
        "sizes": sizes.tolist(),
        "pc_extrapolated": float(pc_extrapolated),
        "pc_extrapolated_error": float(pc_extrap_err),
        "nu": nu
    }
    
    print()
    print(f"Finite-size extrapolation: p_c(∞) = {pc_extrapolated:.6f} ± {pc_extrap_err:.6f}")
    
    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    
    # Is our estimate consistent with exact p_c?
    z_exact = (pc_extrapolated - KNOWN_PC) / pc_extrap_err
    p_exact = 2 * (1 - stats.norm.cdf(abs(z_exact)))
    
    # Is our estimate consistent with 1/φ?
    z_phi = (pc_extrapolated - PHI_INV) / pc_extrap_err
    p_phi = 2 * (1 - stats.norm.cdf(abs(z_phi)))
    
    results["statistical_tests"] = {
        "vs_exact_pc": {
            "our_estimate": float(pc_extrapolated),
            "exact_value": KNOWN_PC,
            "difference": float(pc_extrapolated - KNOWN_PC),
            "z_score": float(z_exact),
            "p_value": float(p_exact),
            "consistent": p_exact > 0.05
        },
        "vs_phi_inverse": {
            "our_estimate": float(pc_extrapolated),
            "phi_inverse": PHI_INV,
            "difference": float(pc_extrapolated - PHI_INV),
            "z_score": float(z_phi),
            "p_value": float(p_phi),
            "consistent": p_phi > 0.05
        }
    }
    
    print(f"\nOur estimate:  p_c = {pc_extrapolated:.6f} ± {pc_extrap_err:.6f}")
    print()
    print("Comparison to EXACT p_c (0.59274605):")
    print(f"  Difference: {pc_extrapolated - KNOWN_PC:.6f}")
    print(f"  z-score:    {z_exact:.2f}")
    print(f"  p-value:    {p_exact:.4f}")
    print(f"  Consistent: {'✓ YES' if p_exact > 0.05 else '✗ NO'}")
    print()
    print("Comparison to 1/φ (0.61803399):")
    print(f"  Difference: {pc_extrapolated - PHI_INV:.6f}")
    print(f"  z-score:    {z_phi:.2f}")
    print(f"  p-value:    {p_phi:.4f}")
    print(f"  Consistent: {'✓ YES' if p_phi > 0.05 else '✗ NO'}")
    
    # Key question: Is p_c CLOSER to 1/φ than to other simple constants?
    print("\n" + "=" * 70)
    print("IS p_c SPECIAL?")
    print("=" * 70)
    
    constants = {
        "1/φ": PHI_INV,
        "1/√3": 1/np.sqrt(3),
        "1/e": 1/np.e,
        "π/5": np.pi/5,
        "1/√2": 1/np.sqrt(2),
        "2/3": 2/3,
        "3/5": 3/5,
        "ln(2)": np.log(2),
    }
    
    distances = {}
    for name, val in constants.items():
        dist = abs(KNOWN_PC - val)
        distances[name] = dist
        print(f"  |p_c - {name}| = {dist:.6f}")
    
    # Sort by distance
    sorted_constants = sorted(distances.items(), key=lambda x: x[1])
    closest = sorted_constants[0]
    
    results["constant_comparison"] = {
        "distances": distances,
        "closest_constant": closest[0],
        "closest_distance": closest[1],
        "sorted": [(k, float(v)) for k, v in sorted_constants]
    }
    
    print(f"\nClosest constant: {closest[0]} (distance = {closest[1]:.6f})")
    
    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    phi_rank = [i for i, (k, v) in enumerate(sorted_constants) if k == "1/φ"][0] + 1
    
    if closest[0] == "1/φ":
        interpretation = "STRONG: p_c is closest to 1/φ among tested constants"
        results["interpretation"] = "strong_phi"
    elif phi_rank <= 3:
        interpretation = f"MODERATE: 1/φ is {phi_rank}th closest to p_c"
        results["interpretation"] = "moderate_phi"
    else:
        interpretation = f"WEAK: 1/φ is {phi_rank}th closest to p_c"
        results["interpretation"] = "weak_phi"
    
    print(interpretation)
    print()
    
    # The key insight
    print("KEY FINDING:")
    if closest[0] == "1/φ":
        print("  Percolation threshold p_c ≈ 0.5927 is closest to 1/φ ≈ 0.618")
        print("  among simple mathematical constants.")
        print()
        print("  However, they are NOT equal:")
        print(f"    p_c   = 0.59274605...")
        print(f"    1/φ   = 0.61803399...")
        print(f"    Diff  = 0.0253 (4.1% of 1/φ)")
        print()
        print("  This suggests φ may be an ATTRACTOR or ORGANIZING PRINCIPLE")
        print("  near which critical points cluster, not an exact value.")
    else:
        print(f"  Percolation threshold p_c is closest to {closest[0]}")
        print("  The 1/φ connection may be coincidental.")
    
    results["final_note"] = {
        "pc_exact": KNOWN_PC,
        "phi_inverse": PHI_INV,
        "difference": abs(KNOWN_PC - PHI_INV),
        "percent_difference": 100 * abs(KNOWN_PC - PHI_INV) / PHI_INV,
        "closest_to_phi": closest[0] == "1/φ",
        "phi_rank": phi_rank
    }
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    results = run_precision_analysis()
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_06_precision_percolation",
        "phi": PHI,
        "phi_inverse": PHI_INV,
        "known_pc": KNOWN_PC
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"exp_06_precision_percolation_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else bool(x) if isinstance(x, np.bool_) else str(x))
    
    print(f"\nResults saved to: {output_path}")
