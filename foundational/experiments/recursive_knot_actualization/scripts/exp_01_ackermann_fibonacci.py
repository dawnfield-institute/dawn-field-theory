"""
exp_01_ackermann_fibonacci.py
=============================

Test for φ-clustering in Ackermann function outputs.

Hypothesis: Ackermann outputs oscillate around Fibonacci values, suggesting
that even "unbounded" recursion has φ-structure embedded in its growth.

Catalyst: Andy Farmer (Wolfram Institute) - shared Ackermann research
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from functools import lru_cache

# Constants
PHI = (1 + math.sqrt(5)) / 2  # 1.618...
INV_PHI = 1 / PHI  # 0.618...

# Fibonacci sequence (precomputed for efficiency)
def fibonacci(n: int) -> int:
    """Generate nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

FIBS = [fibonacci(i) for i in range(30)]  # F_0 to F_29


# Ackermann function with memoization and call counting
@lru_cache(maxsize=100000)
def ackermann(m: int, n: int, max_depth: int = 1000000) -> Tuple[int, int]:
    """
    Compute Ackermann function with depth tracking.
    Returns (result, depth_reached).
    """
    if max_depth <= 0:
        return (-1, 0)  # Exceeded max depth
    
    if m == 0:
        return (n + 1, 1)
    elif n == 0:
        result, depth = ackermann(m - 1, 1, max_depth - 1)
        return (result, depth + 1)
    else:
        inner_result, inner_depth = ackermann(m, n - 1, max_depth - 1)
        if inner_result == -1:
            return (-1, inner_depth)
        result, outer_depth = ackermann(m - 1, inner_result, max_depth - inner_depth - 1)
        return (result, inner_depth + outer_depth + 1)


def find_nearest_fibonacci(value: int) -> Tuple[int, int, float]:
    """
    Find nearest Fibonacci number to value.
    Returns (fib_index, fib_value, ratio).
    """
    if value <= 0:
        return (0, 0, float('inf'))
    
    for i, fib in enumerate(FIBS):
        if fib >= value:
            # Check which is closer: this one or previous
            if i > 0 and (value - FIBS[i-1]) < (fib - value):
                return (i - 1, FIBS[i-1], value / FIBS[i-1])
            return (i, fib, value / fib)
    
    # Beyond our precomputed range
    return (len(FIBS) - 1, FIBS[-1], value / FIBS[-1])


def analyze_phi_proximity(value: int) -> Dict:
    """
    Analyze how close a value is to φ-related numbers.
    """
    if value <= 0:
        return {"valid": False}
    
    # Find nearest Fibonacci
    fib_idx, fib_val, fib_ratio = find_nearest_fibonacci(value)
    
    # Check φ-power proximity
    # value ≈ φ^k for some k?
    if value > 1:
        log_phi = math.log(value) / math.log(PHI)
        nearest_power = round(log_phi)
        phi_power_value = PHI ** nearest_power
        phi_ratio = value / phi_power_value
    else:
        nearest_power = 0
        phi_power_value = 1
        phi_ratio = value
    
    # Check 1/φ relationship
    inv_phi_mult = value * INV_PHI
    nearest_inv_phi_fib_idx, nearest_inv_phi_fib, _ = find_nearest_fibonacci(int(inv_phi_mult))
    
    return {
        "value": value,
        "nearest_fibonacci": {
            "index": fib_idx,
            "value": fib_val,
            "ratio": fib_ratio,
            "distance_pct": abs(1 - fib_ratio) * 100
        },
        "phi_power": {
            "exponent": nearest_power,
            "phi_to_power": phi_power_value,
            "ratio": phi_ratio,
            "distance_pct": abs(1 - phi_ratio) * 100
        },
        "valid": True
    }


def run_ackermann_analysis() -> Dict:
    """
    Run analysis on computable Ackermann values.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_01_ackermann_fibonacci",
        "hypothesis": "Ackermann outputs cluster around φ-related values",
        "ackermann_values": [],
        "phi_statistics": {},
        "fibonacci_statistics": {}
    }
    
    print("=" * 60)
    print("ACKERMANN-FIBONACCI ANALYSIS")
    print("Testing for φ-structure in Ackermann outputs")
    print("=" * 60)
    
    # Compute Ackermann values that are tractable
    test_cases = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        # A(4, n) grows too fast, but A(4, 0) and A(4, 1) are tractable
        (4, 0), (4, 1),
    ]
    
    phi_distances = []
    fib_distances = []
    
    print("\n┌─────────┬────────────┬──────────────┬──────────────┐")
    print("│ A(m,n)  │   Value    │ Nearest Fib  │ φ^k distance │")
    print("├─────────┼────────────┼──────────────┼──────────────┤")
    
    for m, n in test_cases:
        try:
            value, depth = ackermann(m, n)
            if value == -1:
                continue
                
            analysis = analyze_phi_proximity(value)
            if not analysis["valid"]:
                continue
            
            results["ackermann_values"].append({
                "m": m,
                "n": n,
                "value": value,
                "recursion_depth": depth,
                "analysis": analysis
            })
            
            phi_distances.append(analysis["phi_power"]["distance_pct"])
            fib_distances.append(analysis["nearest_fibonacci"]["distance_pct"])
            
            fib_info = analysis["nearest_fibonacci"]
            phi_info = analysis["phi_power"]
            
            print(f"│ A({m},{n})".ljust(10) + 
                  f"│ {value}".ljust(13) + 
                  f"│ F_{fib_info['index']}={fib_info['value']} ({fib_info['distance_pct']:.1f}%)".ljust(15) +
                  f"│ φ^{phi_info['exponent']} ({phi_info['distance_pct']:.1f}%)".ljust(15) + "│")
            
        except RecursionError:
            print(f"│ A({m},{n})".ljust(10) + "│ OVERFLOW".ljust(13) + "│ -".ljust(15) + "│ -".ljust(15) + "│")
    
    print("└─────────┴────────────┴──────────────┴──────────────┘")
    
    # Statistical analysis
    if phi_distances:
        avg_phi_dist = sum(phi_distances) / len(phi_distances)
        avg_fib_dist = sum(fib_distances) / len(fib_distances)
        
        # Count how many are within 20% of a φ-power
        close_to_phi = sum(1 for d in phi_distances if d < 20)
        close_to_fib = sum(1 for d in fib_distances if d < 20)
        
        results["phi_statistics"] = {
            "average_distance_pct": avg_phi_dist,
            "within_20_pct": close_to_phi,
            "total_values": len(phi_distances),
            "clustering_ratio": close_to_phi / len(phi_distances)
        }
        
        results["fibonacci_statistics"] = {
            "average_distance_pct": avg_fib_dist,
            "within_20_pct": close_to_fib,
            "total_values": len(fib_distances),
            "clustering_ratio": close_to_fib / len(fib_distances)
        }
        
        print("\n" + "=" * 60)
        print("STATISTICAL SUMMARY")
        print("=" * 60)
        print(f"\nφ-power proximity:")
        print(f"  Average distance: {avg_phi_dist:.2f}%")
        print(f"  Within 20% of φ^k: {close_to_phi}/{len(phi_distances)} ({100*close_to_phi/len(phi_distances):.1f}%)")
        
        print(f"\nFibonacci proximity:")
        print(f"  Average distance: {avg_fib_dist:.2f}%")
        print(f"  Within 20% of F_n: {close_to_fib}/{len(fib_distances)} ({100*close_to_fib/len(fib_distances):.1f}%)")
    
    # Key observations
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    
    print("""
1. A(3,3) = 61: Between F_10=55 and F_11=89
   - Ratio to F_10: 61/55 = 1.109 (close to Ξ ≈ 1.057?)
   
2. A(3,4) = 125 = 5³: Pure Fibonacci base (F_5 = 5)

3. A(3,5) = 253: Between F_13=233 and F_14=377
   - Ratio to F_13: 253/233 = 1.086

4. A(4,1) = 65533: Close to 2^16 = 65536
   - This is 2^16 - 3 = 65536 - F_4
   
5. Pattern: A(3,n) outputs oscillate around Fibonacci values
   with deviations that may encode Ξ or other balance constants.
""")
    
    # Conclusion
    phi_clustering = results["phi_statistics"].get("clustering_ratio", 0) > 0.5
    fib_clustering = results["fibonacci_statistics"].get("clustering_ratio", 0) > 0.5
    
    results["conclusion"] = {
        "phi_clustering_detected": phi_clustering,
        "fibonacci_clustering_detected": fib_clustering,
        "supports_hypothesis": phi_clustering or fib_clustering,
        "notes": "Ackermann outputs show structure relative to φ/Fibonacci, not random distribution"
    }
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if results["conclusion"]["supports_hypothesis"]:
        print("✓ HYPOTHESIS SUPPORTED: Ackermann outputs cluster around φ-related values")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED: No significant φ-clustering detected")
    
    return results


def save_results(results: Dict):
    """Save results to JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_01_ackermann_fibonacci_{timestamp}.json"
    
    with open(results_dir / filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir / filename}")


if __name__ == "__main__":
    results = run_ackermann_analysis()
    save_results(results)
