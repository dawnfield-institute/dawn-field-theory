"""
Experiment 18: RESOLVING THE CONTRADICTION
==========================================

Two facts that seem incompatible:

FACT 1: Inverse Fibonacci f(4) = f(5) + f(6) holds with increasing accuracy
  - N=100k: 9.5% error
  - N=2M: 0.46% error
  - Converging to exact equality

FACT 2: Math says inverse Fibonacci implies r = f(k+1)/f(k) → 1/φ
  But empirically, f(5)/f(4) CROSSED 1/φ at N=500k and keeps rising!

RESOLUTION HYPOTHESIS:
The inverse Fibonacci holds LOCALLY at k=4 only, not globally.
The equilibrium r = 1/φ requires recursion at ALL k.

Let's test: does f(k) = f(k+1) + f(k+2) hold at OTHER k values?
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from growth_engine import sieve_of_eratosthenes, big_omega


PHI = (1 + np.sqrt(5)) / 2
ONE_OVER_PHI = 1 / PHI


def test_recursion_at_all_k(limit=5000000):
    """
    Test: At which k does f(k) = f(k+1) + f(k+2) hold?
    """
    print("=" * 70)
    print(f"TEST 1: INVERSE FIBONACCI AT ALL k VALUES (N = {limit:,})")
    print("=" * 70)
    
    print("\nComputing Ω distribution...")
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    omega_counts = defaultdict(int)
    for n in range(4, limit):
        if n not in prime_set:
            omega_counts[big_omega(n)] += 1
    
    total = sum(omega_counts.values())
    
    print(f"\nInverse Fibonacci: f(k) = f(k+1) + f(k+2)")
    print(f"Ratio should be 1.0 if exact\n")
    
    print(f"{'k':>4} | {'f(k)':>12} | {'f(k+1)+f(k+2)':>15} | {'Ratio':>10} | {'Error %':>10}")
    print("-" * 65)
    
    ratios = {}
    for k in range(2, 15):
        fk = omega_counts[k]
        fk1 = omega_counts[k+1]
        fk2 = omega_counts[k+2]
        
        sum_next = fk1 + fk2
        ratio = fk / sum_next if sum_next > 0 else 0
        error_pct = abs(ratio - 1.0) * 100
        
        ratios[k] = ratio
        
        marker = "***" if error_pct < 2 else ""
        print(f"{k:>4} | {fk:>12,} | {sum_next:>15,} | {ratio:>10.4f} | {error_pct:>9.2f}% {marker}")
    
    return ratios, omega_counts, total


def test_what_recursion_holds(omega_counts, total):
    """
    If not inverse Fibonacci globally, what recursion DOES hold?
    
    Test: f(k) = α(k)*f(k+1) + β(k)*f(k+2)
    Find α(k), β(k) for each k
    """
    print("\n" + "=" * 70)
    print("TEST 2: WHAT RECURSION ACTUALLY HOLDS?")
    print("=" * 70)
    
    print(f"\nFitting f(k) = α*f(k+1) + β*f(k+2) at each k\n")
    
    # For each k, solve: f(k) = α*f(k+1) + β*f(k+2)
    # This is underdetermined (2 unknowns, 1 equation)
    # So let's try specific cases:
    
    # Case 1: α = β (symmetric)
    print("--- Case 1: α = β (symmetric recursion) ---")
    print(f"{'k':>4} | {'α = β':>10}")
    print("-" * 20)
    
    for k in range(2, 12):
        fk = omega_counts[k]
        fk1 = omega_counts[k+1]
        fk2 = omega_counts[k+2]
        
        # f(k) = α*(f(k+1) + f(k+2))
        alpha = fk / (fk1 + fk2) if (fk1 + fk2) > 0 else 0
        print(f"{k:>4} | {alpha:>10.4f}")
    
    # Case 2: β = 1 (fix β, solve for α)
    print("\n--- Case 2: β = 1 (standard recursion with different α) ---")
    print(f"{'k':>4} | {'α':>10} | {'Note':>20}")
    print("-" * 40)
    
    for k in range(2, 12):
        fk = omega_counts[k]
        fk1 = omega_counts[k+1]
        fk2 = omega_counts[k+2]
        
        # f(k) = α*f(k+1) + f(k+2)
        # α = (f(k) - f(k+2)) / f(k+1)
        alpha = (fk - fk2) / fk1 if fk1 > 0 else 0
        
        note = ""
        if abs(alpha - 1.0) < 0.05:
            note = "≈ Fibonacci"
        elif abs(alpha - ONE_OVER_PHI) < 0.05:
            note = "≈ 1/φ"
        elif abs(alpha - PHI) < 0.05:
            note = "≈ φ"
        
        print(f"{k:>4} | {alpha:>10.4f} | {note:>20}")


def test_ratio_chain(omega_counts, total):
    """
    The ratio r(k) = f(k+1)/f(k) tells us the LOCAL slope.
    Let's see the full chain.
    """
    print("\n" + "=" * 70)
    print("TEST 3: RATIO CHAIN ANALYSIS")
    print("=" * 70)
    
    print(f"\n1/φ = {ONE_OVER_PHI:.6f}")
    print(f"\n{'k':>4} | {'r(k)':>12} | {'r(k)/r(k-1)':>12} | {'vs 1/φ':>12}")
    print("-" * 55)
    
    prev_r = None
    for k in range(2, 14):
        fk = omega_counts[k]
        fk1 = omega_counts[k+1]
        
        r = fk1 / fk if fk > 0 else 0
        
        ratio_of_ratios = r / prev_r if prev_r and prev_r > 0 else None
        
        if ratio_of_ratios:
            print(f"{k:>4} | {r:>12.6f} | {ratio_of_ratios:>12.6f} | {r - ONE_OVER_PHI:>+12.6f}")
        else:
            print(f"{k:>4} | {r:>12.6f} | {'--':>12} | {r - ONE_OVER_PHI:>+12.6f}")
        
        prev_r = r


def test_why_k4_special(omega_counts, total):
    """
    Why does inverse Fibonacci hold at k=4 specifically?
    
    Look at the STRUCTURE of numbers with Ω = 4 vs Ω = 5 vs Ω = 6
    """
    print("\n" + "=" * 70)
    print("TEST 4: WHY k=4 IS SPECIAL")
    print("=" * 70)
    
    # Numbers with Ω = 4: 2^4=16, 2³×3=24, 2²×3²=36, 2²×5=20, etc.
    # Numbers with Ω = 5: 2^5=32, 2⁴×3=48, etc.
    # Numbers with Ω = 6: 2^6=64, etc.
    
    # Hypothesis: k=4 is where the distribution is most "balanced"
    
    f3 = omega_counts[3] / total
    f4 = omega_counts[4] / total
    f5 = omega_counts[5] / total
    f6 = omega_counts[6] / total
    
    print(f"\nDistribution around k=4:")
    print(f"  f(3) = {f3:.5f}")
    print(f"  f(4) = {f4:.5f}")
    print(f"  f(5) = {f5:.5f}")
    print(f"  f(6) = {f6:.5f}")
    
    print(f"\nSymmetry check:")
    print(f"  f(3)/f(5) = {f3/f5:.4f}")
    print(f"  f(4)/f(6) = {f4/f6:.4f}")
    
    # The ratio f(k)/f(k+1) at each point
    print(f"\nSlope ratios:")
    print(f"  f(3)/f(4) = {f3/f4:.4f} (φ = {PHI:.4f})")
    print(f"  f(4)/f(5) = {f4/f5:.4f}")
    print(f"  f(5)/f(6) = {f5/f6:.4f}")
    
    # At k=4, f(4)/f(5) ≈ φ might be the key
    print(f"\n  f(4)/f(5) - φ = {f4/f5 - PHI:+.4f}")
    print(f"  f(5)/f(4) - 1/φ = {f5/f4 - ONE_OVER_PHI:+.4f}")


def test_erdos_kac_prediction(limit, omega_counts, total):
    """
    Erdős-Kac says: Ω(n) ~ Normal(log log n, √(log log n))
    
    What does EK predict for f(k+1)/f(k)?
    """
    print("\n" + "=" * 70)
    print("TEST 5: ERDŐS-KAC PREDICTION")
    print("=" * 70)
    
    import math
    from scipy import stats
    
    mu = math.log(math.log(limit))
    sigma = math.sqrt(mu)
    
    print(f"\nN = {limit:,}")
    print(f"Erdős-Kac μ = log(log(N)) = {mu:.4f}")
    print(f"Erdős-Kac σ = √μ = {sigma:.4f}")
    
    # For normal distribution, the ratio at any point is:
    # PDF(k+1)/PDF(k) = exp(-(2k+1-2μ)/(2σ²))
    
    print(f"\n{'k':>4} | {'EK ratio':>12} | {'Actual ratio':>12} | {'Difference':>12}")
    print("-" * 55)
    
    for k in range(2, 12):
        ek_ratio = math.exp(-(2*k + 1 - 2*mu) / (2*sigma**2))
        actual = (omega_counts[k+1] / omega_counts[k]) if omega_counts[k] > 0 else 0
        diff = actual - ek_ratio
        print(f"{k:>4} | {ek_ratio:>12.5f} | {actual:>12.5f} | {diff:>+12.5f}")
    
    # Where does EK predict ratio = 1?
    # exp(-(2k+1-2μ)/(2σ²)) = 1
    # -(2k+1-2μ)/(2σ²) = 0
    # 2k+1 = 2μ
    # k = μ - 0.5
    k_unity = mu - 0.5
    
    print(f"\nEK predicts f(k+1)/f(k) = 1 at k = μ - 0.5 = {k_unity:.2f}")
    print(f"Peak is at k ≈ {round(k_unity)}")
    
    # EK ratio crosses 1/φ where:
    # exp(-(2k+1-2μ)/(2σ²)) = 1/φ = 0.618
    # -(2k+1-2μ)/(2σ²) = ln(1/φ) = -ln(φ) ≈ -0.481
    # 2k+1-2μ = 2σ² * 0.481
    # k = μ - 0.5 + σ² * ln(φ)
    k_phi = mu - 0.5 + sigma**2 * math.log(PHI)
    
    print(f"EK predicts f(k+1)/f(k) = 1/φ at k ≈ {k_phi:.2f}")


def the_resolution():
    """
    Explain the resolution to the contradiction.
    """
    print("\n" + "=" * 70)
    print("RESOLUTION OF THE CONTRADICTION")
    print("=" * 70)
    
    print("""
WHY INVERSE FIBONACCI AT k=4 DOESN'T IMPLY f(5)/f(4) → 1/φ:

The mathematical derivation assumed:
  f(k) = f(k+1) + f(k+2) for ALL k

But empirically:
  - k=4: ratio = 1.00 ± 0.5%  ✓ (holds!)
  - k=3: ratio = 0.76         ✗ (doesn't hold)
  - k=5: ratio = 1.18         ✗ (doesn't hold)

The equilibrium argument requires the recursion to propagate.
Since it ONLY holds at k=4, we can't conclude r = 1/φ everywhere.

WHAT'S ACTUALLY HAPPENING:

1. At k=4 (the transition point from peak):
   f(4) = f(5) + f(6)   [EXACT at large N]
   This means: 1 = f(5)/f(4) + f(6)/f(4)
              1 = r(4) + r(4)*r(5)
   
2. But r(4) ≠ r(5)! The ratios are different at each k.
   - r(4) = f(5)/f(4) ≈ 0.63 (above 1/φ)
   - r(5) = f(6)/f(5) ≈ 0.55 (below 1/φ)

3. The equation 1 = r(4)*(1 + r(5)) IS satisfied:
   1 ≈ 0.63 * (1 + 0.55) = 0.63 * 1.55 = 0.98 ≈ 1 ✓

4. But this doesn't require r(4) = 1/φ!
   It only requires r(4)*(1 + r(5)) = 1

THE REAL CONSTRAINT:
At the transition point k=4:
    r(4) = 1/(1 + r(5))

If r(5) = 0.55, then r(4) = 1/1.55 = 0.645
Actual r(4) ≈ 0.63, which is close!

CONCLUSION:
The inverse Fibonacci at k=4 constrains r(4) and r(5) to satisfy
    r(4)*(1 + r(5)) = 1
But this is NOT the same as r = 1/φ.

The 1/φ crossing at N=500k was COINCIDENTAL, not fundamental.
""")


def save_results(results, filename):
    """Save results to JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("EXPERIMENT 18: RESOLVING THE CONTRADICTION")
    print("=" * 70)
    
    limit = 5000000
    
    # Test 1: Recursion at all k
    ratios, omega_counts, total = test_recursion_at_all_k(limit)
    
    # Test 2: What recursion holds
    test_what_recursion_holds(omega_counts, total)
    
    # Test 3: Ratio chain
    test_ratio_chain(omega_counts, total)
    
    # Test 4: Why k=4
    test_why_k4_special(omega_counts, total)
    
    # Test 5: Erdős-Kac
    test_erdos_kac_prediction(limit, omega_counts, total)
    
    # Resolution
    the_resolution()
    
    # Verify the actual constraint
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    f4 = omega_counts[4]
    f5 = omega_counts[5]
    f6 = omega_counts[6]
    
    r4 = f5 / f4
    r5 = f6 / f5
    
    print(f"\nAt N = {limit:,}:")
    print(f"  r(4) = f(5)/f(4) = {r4:.6f}")
    print(f"  r(5) = f(6)/f(5) = {r5:.6f}")
    print(f"  r(4) * (1 + r(5)) = {r4 * (1 + r5):.6f} (should be ≈ 1)")
    print(f"  1 / (1 + r(5)) = {1 / (1 + r5):.6f} (predicted r(4))")
    print(f"  Error: {abs(r4 - 1/(1+r5)):.6f}")
    
    # Save
    results = {
        'ratios': ratios,
        'r4': r4,
        'r5': r5,
        'product': r4 * (1 + r5),
        'predicted_r4': 1 / (1 + r5)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"exp_18_contradiction_{timestamp}.json")


if __name__ == "__main__":
    main()
