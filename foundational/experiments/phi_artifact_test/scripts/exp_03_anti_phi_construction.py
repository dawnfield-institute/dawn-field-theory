"""
Exp 03: Anti-φ Construction Test

Question: Can PAC/SEC correctly FAIL to find φ in systems
explicitly designed to have non-golden structure?

If PAC is a confirmation bias machine, it will "find" φ everywhere.
If PAC has genuine specificity, it will correctly reject anti-φ systems.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path


PHI = (1 + np.sqrt(5)) / 2  # 1.618...
TOLERANCE = 0.05


def detect_phi_in_ratios(ratios: np.ndarray, tolerance: float = TOLERANCE) -> Dict:
    """Detect φ-like ratios in a set of values."""
    if len(ratios) == 0:
        return {"phi_detected": False, "phi_rate": 0, "mean_ratio": 0}
    
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) == 0:
        return {"phi_detected": False, "phi_rate": 0, "mean_ratio": 0}
    
    phi_matches = np.abs(ratios - PHI) < (PHI * tolerance)
    phi_inv_matches = np.abs(ratios - 1/PHI) < ((1/PHI) * tolerance)
    combined_matches = phi_matches | phi_inv_matches
    
    return {
        "phi_detected": np.any(combined_matches),
        "phi_rate": float(np.mean(combined_matches)),
        "mean_ratio": float(np.mean(ratios)),
        "closest_to_phi": float(np.min(np.abs(ratios - PHI)))
    }


# ============================================================
# ANTI-φ CONSTRUCTIONS
# ============================================================

def construct_sqrt2_system(n: int = 500) -> Dict:
    """
    System based on √2 ratios (irrational, but NOT golden).
    
    √2 ≈ 1.414, distinctly different from φ ≈ 1.618
    """
    sqrt2 = np.sqrt(2)
    
    # Pell-like sequence: converges to 1 + √2 ≈ 2.414
    seq = [1.0, 2.0]
    for i in range(n - 2):
        seq.append(2 * seq[-1] + seq[-2])
    
    # Ratios should converge to 1 + √2
    seq = np.array(seq)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "sqrt2_pell",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": 1 + sqrt2,
        "expected_symbol": "1+√2"
    }


def construct_e_system(n: int = 500) -> Dict:
    """
    System based on e ratios.
    
    e ≈ 2.718, very different from φ ≈ 1.618
    """
    # Exponential growth with base e
    t = np.linspace(0, 10, n)
    seq = np.exp(t)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "exponential_e",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": np.exp(10/(n-1)),  # Constant ratio
        "expected_symbol": "e^δ"
    }


def construct_pi_system(n: int = 500) -> Dict:
    """
    System with π-based structure.
    
    π ≈ 3.14159, very different from φ
    """
    # Sequence where ratios approach π
    seq = [1.0]
    for i in range(n - 1):
        seq.append(seq[-1] * np.pi ** (1/10))  # Slow approach
    
    seq = np.array(seq)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "pi_based",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": np.pi ** (1/10),
        "expected_symbol": "π^0.1"
    }


def construct_integer_ratio_system(n: int = 500) -> Dict:
    """
    System with rational ratios (2/1, 3/2, 4/3, etc.).
    
    These are rational and should NOT match irrational φ.
    """
    # Harmonic-like: 1, 2, 3, 4, 5...
    seq = np.arange(1, n + 1, dtype=float)
    ratios = seq[1:] / seq[:-1]  # n/(n-1) → 1
    
    return {
        "name": "harmonic_integers",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": 1.0,  # Converges to 1
        "expected_symbol": "1"
    }


def construct_silver_ratio_system(n: int = 500) -> Dict:
    """
    Silver ratio: δ_s = 1 + √2 ≈ 2.414
    
    The "silver mean" - related to Pell numbers, NOT Fibonacci.
    """
    silver = 1 + np.sqrt(2)
    
    # Generate silver ratio sequence
    seq = [1.0, silver]
    for i in range(n - 2):
        seq.append(seq[-1] * silver)
    
    seq = np.array(seq)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "silver_ratio",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": silver,
        "expected_symbol": "δ_s (silver)"
    }


def construct_plastic_ratio_system(n: int = 500) -> Dict:
    """
    Plastic ratio: ρ ≈ 1.3247
    
    The unique real solution to x³ = x + 1
    Another "metallic mean" but NOT golden.
    """
    plastic = 1.324717957244746  # Plastic constant
    
    # Padovan-like sequence: P(n) = P(n-2) + P(n-3)
    seq = [1, 1, 1]
    for i in range(n - 3):
        seq.append(seq[-2] + seq[-3])
    
    seq = np.array(seq, dtype=float)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "plastic_ratio",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": plastic,
        "expected_symbol": "ρ (plastic)"
    }


def construct_anti_correlated_to_phi(n: int = 500) -> Dict:
    """
    Sequence explicitly designed to avoid φ.
    
    When ratio approaches φ, we perturb away from it.
    """
    seq = [1.0, 2.0]  # Start with ratio = 2
    
    for i in range(n - 2):
        # Natural next value
        candidate = seq[-1] * 1.5  # Try ratio of 1.5
        
        # Check if this would create a ratio near φ
        ratio = candidate / seq[-1]
        if abs(ratio - PHI) < 0.2:
            # Perturb away from φ
            if ratio > PHI:
                candidate = seq[-1] * (PHI + 0.3)  # Push higher
            else:
                candidate = seq[-1] * (PHI - 0.3)  # Push lower
        
        seq.append(candidate)
    
    seq = np.array(seq)
    ratios = seq[1:] / seq[:-1]
    
    return {
        "name": "anti_phi_explicit",
        "sequence": seq,
        "ratios": ratios,
        "expected_ratio": None,  # Intentionally varies
        "expected_symbol": "anti-φ"
    }


# ============================================================
# PAC-STYLE ANALYSIS
# ============================================================

def pac_analysis(sequence: np.ndarray) -> Dict:
    """Apply PAC-style φ detection to a sequence."""
    
    # Consecutive ratios
    consecutive = sequence[1:] / sequence[:-1]
    consecutive = consecutive[np.isfinite(consecutive) & (consecutive > 0.1) & (consecutive < 10)]
    
    # Skip-1 ratios (Fibonacci-like)
    if len(sequence) >= 3:
        skip1 = sequence[2:] / sequence[:-2]
        skip1 = skip1[np.isfinite(skip1) & (skip1 > 0.1) & (skip1 < 10)]
    else:
        skip1 = np.array([])
    
    return {
        "consecutive": detect_phi_in_ratios(consecutive),
        "skip1": detect_phi_in_ratios(skip1),
        "any_phi_detected": (
            detect_phi_in_ratios(consecutive)["phi_detected"] or
            (len(skip1) > 0 and detect_phi_in_ratios(skip1)["phi_detected"])
        )
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_anti_phi_test() -> Dict:
    """
    Test PAC's specificity: can it correctly reject anti-φ systems?
    """
    print("=" * 60)
    print("EXP 03: ANTI-φ CONSTRUCTION TEST")
    print("=" * 60)
    print("Testing if PAC correctly FAILS to find φ in non-golden systems...")
    print()
    
    n = 500
    
    # Construct anti-φ systems
    anti_phi_systems = [
        construct_sqrt2_system(n),
        construct_e_system(n),
        construct_pi_system(n),
        construct_integer_ratio_system(n),
        construct_silver_ratio_system(n),
        construct_plastic_ratio_system(n),
        construct_anti_correlated_to_phi(n),
    ]
    
    # Also include true φ system as positive control
    fib = [1, 1]
    for i in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib, dtype=float)
    
    positive_control = {
        "name": "fibonacci_control",
        "sequence": fib,
        "ratios": fib[1:] / fib[:-1],
        "expected_ratio": PHI,
        "expected_symbol": "φ (golden)"
    }
    
    results = {
        "anti_phi_systems": {},
        "positive_control": {},
        "specificity_analysis": {}
    }
    
    # Analyze anti-φ systems
    print("Analyzing anti-φ constructions:")
    false_positives = 0
    for system in anti_phi_systems:
        analysis = pac_analysis(system["sequence"])
        
        results["anti_phi_systems"][system["name"]] = {
            "expected_ratio": float(system["expected_ratio"]) if system["expected_ratio"] else None,
            "expected_symbol": system["expected_symbol"],
            "pac_analysis": analysis,
            "phi_incorrectly_detected": analysis["any_phi_detected"]
        }
        
        if analysis["any_phi_detected"]:
            false_positives += 1
            status = "⚠️ FALSE POSITIVE"
        else:
            status = "✓ Correctly rejected"
        
        actual_ratio = float(np.mean(system["ratios"][-10:]))
        print(f"  {system['name']}: {status}")
        print(f"      Expected: {system['expected_symbol']} ≈ {system['expected_ratio']}")
        print(f"      Actual mean ratio: {actual_ratio:.4f}")
    
    # Analyze positive control
    print("\nPositive control (Fibonacci):")
    control_analysis = pac_analysis(positive_control["sequence"])
    results["positive_control"] = {
        "expected_ratio": PHI,
        "pac_analysis": control_analysis,
        "phi_correctly_detected": control_analysis["any_phi_detected"]
    }
    
    if control_analysis["any_phi_detected"]:
        print(f"  fibonacci_control: ✓ φ correctly detected")
    else:
        print(f"  fibonacci_control: ⚠️ FALSE NEGATIVE - φ not detected!")
    
    # Calculate specificity
    n_anti_phi = len(anti_phi_systems)
    true_negatives = n_anti_phi - false_positives
    specificity = true_negatives / n_anti_phi
    
    # Sensitivity from control
    sensitivity = 1.0 if control_analysis["any_phi_detected"] else 0.0
    
    results["specificity_analysis"] = {
        "n_anti_phi_systems": n_anti_phi,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "artifact_hypothesis_supported": false_positives > n_anti_phi / 2
    }
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Anti-φ systems tested: {n_anti_phi}")
    print(f"False positives (φ found where it shouldn't be): {false_positives}")
    print(f"True negatives (correctly rejected): {true_negatives}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Sensitivity (from control): {sensitivity:.2%}")
    print()
    
    if specificity >= 0.8:
        print("✓ ARTIFACT HYPOTHESIS REJECTED")
        print(f"  PAC shows {specificity:.0%} specificity - it correctly rejects anti-φ systems")
        print("  This suggests φ detection is genuine, not an artifact")
    elif specificity >= 0.5:
        print("⚠️ INCONCLUSIVE")
        print(f"  PAC shows {specificity:.0%} specificity - mixed results")
        print("  More investigation needed")
    else:
        print("⚠️ ARTIFACT HYPOTHESIS SUPPORTED")
        print(f"  PAC shows only {specificity:.0%} specificity - too many false positives")
        print("  This suggests φ detection may be an artifact")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    results = run_anti_phi_test()
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_03_anti_phi_construction",
        "hypothesis": "PAC finds φ even in systems designed to exclude it",
        "phi": PHI,
        "tolerance": TOLERANCE
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"exp_03_anti_phi_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
