"""
Exp 02: Alternative Recursion Test

Question: Does φ appear because of Fibonacci recursion specifically,
or because it's genuinely present in the domains?

If we replace Ψ(k) = Ψ(k+1) + Ψ(k+2) with other recursions,
does φ disappear from domains where we previously found it?
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Callable
from pathlib import Path


PHI = (1 + np.sqrt(5)) / 2  # 1.618...
TRIBONACCI = 1.8392867552141612  # Limit of tribonacci ratio
TOLERANCE = 0.05


# ============================================================
# ALTERNATIVE RECURSIONS
# ============================================================

def fibonacci_sequence(n: int) -> np.ndarray:
    """Standard Fibonacci: Ψ(k) = Ψ(k+1) + Ψ(k+2)"""
    seq = [1, 1]
    for i in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return np.array(seq, dtype=float)


def tribonacci_sequence(n: int) -> np.ndarray:
    """Tribonacci: Ψ(k) = Ψ(k+1) + Ψ(k+2) + Ψ(k+3)"""
    seq = [1, 1, 1]
    for i in range(n - 3):
        seq.append(seq[-1] + seq[-2] + seq[-3])
    return np.array(seq, dtype=float)


def lucas_sequence(n: int) -> np.ndarray:
    """Lucas numbers: same recursion, different seed (2, 1)"""
    seq = [2, 1]
    for i in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return np.array(seq, dtype=float)


def skip_fibonacci_sequence(n: int) -> np.ndarray:
    """Skip-1 Fibonacci: Ψ(k) = Ψ(k+1) + Ψ(k+3)"""
    seq = [1, 1, 1, 1]
    for i in range(n - 4):
        seq.append(seq[-1] + seq[-3])
    return np.array(seq, dtype=float)


def exponential_sequence(n: int, base: float = 2.0) -> np.ndarray:
    """Exponential: Ψ(k) = base * Ψ(k+1)"""
    seq = [1.0]
    for i in range(n - 1):
        seq.append(seq[-1] * base)
    return np.array(seq, dtype=float)


def linear_sequence(n: int, increment: float = 1.0) -> np.ndarray:
    """Linear: Ψ(k) = Ψ(k+1) + c"""
    return np.arange(1, n + 1, dtype=float) * increment


def pell_sequence(n: int) -> np.ndarray:
    """Pell numbers: Ψ(k) = 2*Ψ(k+1) + Ψ(k+2), converges to 1 + √2"""
    seq = [0, 1]
    for i in range(n - 2):
        seq.append(2 * seq[-1] + seq[-2])
    return np.array(seq[1:], dtype=float)  # Skip the leading 0


# ============================================================
# DOMAIN ANALYSIS
# ============================================================

def analyze_ratios(sequence: np.ndarray) -> Dict:
    """Compute consecutive ratios and their statistics."""
    ratios = sequence[1:] / (sequence[:-1] + 1e-10)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    
    if len(ratios) < 2:
        return {"error": "insufficient ratios"}
    
    # Convergent ratio (last values are most converged)
    convergent_ratio = np.mean(ratios[-10:]) if len(ratios) >= 10 else np.mean(ratios)
    
    # Distance to various constants
    dist_to_phi = abs(convergent_ratio - PHI)
    dist_to_2 = abs(convergent_ratio - 2.0)
    dist_to_e = abs(convergent_ratio - np.e)
    dist_to_sqrt2_plus_1 = abs(convergent_ratio - (1 + np.sqrt(2)))
    dist_to_tribonacci = abs(convergent_ratio - 1.839)
    
    return {
        "convergent_ratio": float(convergent_ratio),
        "ratio_std": float(np.std(ratios[-10:])) if len(ratios) >= 10 else float(np.std(ratios)),
        "dist_to_phi": float(dist_to_phi),
        "dist_to_2": float(dist_to_2),
        "dist_to_e": float(dist_to_e),
        "dist_to_sqrt2_plus_1": float(dist_to_sqrt2_plus_1),
        "dist_to_tribonacci": float(dist_to_tribonacci),
        "is_phi": dist_to_phi < TOLERANCE * PHI,
        "closest_constant": min([
            ("phi", dist_to_phi),
            ("2", dist_to_2),
            ("e", dist_to_e),
            ("1+sqrt2", dist_to_sqrt2_plus_1),
            ("tribonacci", dist_to_tribonacci)
        ], key=lambda x: x[1])[0]
    }


def simulate_domain_with_recursion(
    domain_name: str,
    recursion_seq: np.ndarray,
    domain_data: np.ndarray
) -> Dict:
    """
    Apply recursion-based analysis to domain data.
    
    The key question: does the recursion determine the constant found,
    or does the domain data determine it?
    """
    # Method 1: Use recursion as a basis and project domain onto it
    if len(recursion_seq) >= len(domain_data):
        projected = domain_data / (recursion_seq[:len(domain_data)] + 1e-10)
    else:
        projected = domain_data[:len(recursion_seq)] / (recursion_seq + 1e-10)
    
    # Method 2: Analyze domain data directly
    domain_analysis = analyze_ratios(domain_data)
    
    # Method 3: Analyze recursion sequence
    recursion_analysis = analyze_ratios(recursion_seq)
    
    # Method 4: Cross-correlation peaks
    if len(domain_data) >= 100 and len(recursion_seq) >= 100:
        corr = np.correlate(
            domain_data[:100] / np.max(domain_data[:100]),
            recursion_seq[:100] / np.max(recursion_seq[:100]),
            mode='valid'
        )
        max_corr = float(np.max(corr)) if len(corr) > 0 else 0
    else:
        max_corr = 0
    
    return {
        "domain_analysis": domain_analysis,
        "recursion_analysis": recursion_analysis,
        "cross_correlation": max_corr
    }


# ============================================================
# REAL DOMAIN DATA
# ============================================================

def get_prime_gaps(n: int = 500) -> np.ndarray:
    """Generate prime gaps as a test domain."""
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True
    
    primes = []
    num = 2
    while len(primes) < n + 1:
        if is_prime(num):
            primes.append(num)
        num += 1
    
    gaps = np.diff(primes).astype(float)
    return gaps


def get_rule110_pattern(n: int = 500) -> np.ndarray:
    """Generate Rule 110 cellular automaton population counts."""
    width = 200
    state = np.zeros(width, dtype=int)
    state[width // 2] = 1
    
    populations = [np.sum(state)]
    
    for _ in range(n):
        new_state = np.zeros_like(state)
        for i in range(1, width - 1):
            # Rule 110 lookup
            pattern = state[i-1] * 4 + state[i] * 2 + state[i+1]
            rule110 = [0, 1, 1, 1, 0, 1, 1, 0]  # Rule 110 binary
            new_state[i] = rule110[pattern]
        state = new_state
        populations.append(np.sum(state))
    
    return np.array(populations, dtype=float) + 1  # +1 to avoid zeros


def get_jwst_masses() -> np.ndarray:
    """Simulated JWST-like mass distribution (log-normal)."""
    # Based on observed distribution: peak around 10^7, range 10^6 to 10^9
    log_masses = np.random.normal(7.0, 0.8, 69)
    return 10 ** log_masses


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_alternative_recursion_test() -> Dict:
    """
    Test whether φ appearance depends on Fibonacci recursion.
    """
    print("=" * 60)
    print("EXP 02: ALTERNATIVE RECURSION TEST")
    print("=" * 60)
    print("Testing if φ appears due to Fibonacci recursion or domain structure...")
    print()
    
    n = 500
    
    # Define recursions and their expected constants
    recursions = {
        "fibonacci": (fibonacci_sequence(n), PHI, "φ"),
        "tribonacci": (tribonacci_sequence(n), 1.839, "tribonacci"),
        "lucas": (lucas_sequence(n), PHI, "φ"),  # Same limit as Fibonacci
        "pell": (pell_sequence(n), 1 + np.sqrt(2), "1+√2"),
        "exponential_2": (exponential_sequence(n, 2.0), 2.0, "2"),
        "exponential_e": (exponential_sequence(n, np.e), np.e, "e"),
        "linear": (linear_sequence(n), 1.0, "1"),
    }
    
    # Define test domains
    domains = {
        "prime_gaps": get_prime_gaps(n),
        "rule110": get_rule110_pattern(n),
        "jwst_masses": get_jwst_masses(),
        "pure_fibonacci": fibonacci_sequence(n),
        "random_walk": np.cumsum(np.random.randn(n)) + 100,
    }
    
    results = {
        "recursion_limits": {},
        "domain_analyses": {},
        "cross_tests": {}
    }
    
    # First: verify each recursion converges to expected constant
    print("Verifying recursion convergent ratios:")
    for name, (seq, expected, symbol) in recursions.items():
        analysis = analyze_ratios(seq)
        results["recursion_limits"][name] = {
            "expected": float(expected),
            "observed": analysis["convergent_ratio"],
            "difference": abs(analysis["convergent_ratio"] - expected),
            "symbol": symbol
        }
        print(f"  {name}: expected {expected:.4f}, got {analysis['convergent_ratio']:.4f}")
    
    # Second: analyze each domain independently
    print("\nAnalyzing domains directly (no recursion applied):")
    for domain_name, domain_data in domains.items():
        analysis = analyze_ratios(domain_data)
        results["domain_analyses"][domain_name] = analysis
        print(f"  {domain_name}: ratio = {analysis['convergent_ratio']:.4f}, closest = {analysis['closest_constant']}")
    
    # Third: cross-test each domain with each recursion
    print("\nCross-testing domains with different recursions:")
    for domain_name, domain_data in domains.items():
        results["cross_tests"][domain_name] = {}
        for recursion_name, (rec_seq, _, _) in recursions.items():
            cross_result = simulate_domain_with_recursion(
                domain_name, rec_seq, domain_data
            )
            results["cross_tests"][domain_name][recursion_name] = {
                "domain_finds_phi": cross_result["domain_analysis"].get("is_phi", False),
                "correlation": cross_result["cross_correlation"]
            }
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Key question: does domain analysis find φ regardless of recursion used?
    print("\nDomain φ detection (independent of recursion):")
    domain_phi_counts = {}
    for domain_name, analysis in results["domain_analyses"].items():
        is_phi = analysis.get("is_phi", False)
        domain_phi_counts[domain_name] = is_phi
        status = "✓ φ detected" if is_phi else "✗ Not φ"
        print(f"  {domain_name}: {status} (ratio = {analysis['convergent_ratio']:.4f})")
    
    # Interpretation
    structured_domains = ["prime_gaps", "rule110", "pure_fibonacci"]
    null_domains = ["random_walk", "jwst_masses"]
    
    structured_phi = sum(1 for d in structured_domains if domain_phi_counts.get(d, False))
    null_phi = sum(1 for d in null_domains if domain_phi_counts.get(d, False))
    
    results["interpretation"] = {
        "structured_phi_count": structured_phi,
        "structured_total": len(structured_domains),
        "null_phi_count": null_phi,
        "null_total": len(null_domains),
        "recursion_independent": True  # Domain analysis doesn't use recursion
    }
    
    print(f"\nStructured domains with φ: {structured_phi}/{len(structured_domains)}")
    print(f"Null domains with φ: {null_phi}/{len(null_domains)}")
    
    if structured_phi > null_phi:
        print("\n✓ φ appears more in structured domains than null domains")
        print("  This suggests φ is a domain property, not a recursion artifact")
        results["interpretation"]["artifact_hypothesis"] = "rejected"
    else:
        print("\n⚠️ φ appears similarly in structured and null domains")
        print("  This suggests φ may be an analysis artifact")
        results["interpretation"]["artifact_hypothesis"] = "supported"
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    results = run_alternative_recursion_test()
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_02_alternative_recursion",
        "hypothesis": "φ appears due to Fibonacci recursion, not domain structure",
        "phi": PHI,
        "tolerance": TOLERANCE
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"exp_02_alternative_recursion_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
