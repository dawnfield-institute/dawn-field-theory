"""
Experiment 14: Casimir Effect - Baseline Analysis
=================================================
Part V: Test SEC framework on Casimir effect predictions

The Casimir effect: Two uncharged conducting plates in vacuum 
experience an attractive force due to quantum vacuum fluctuations.

Standard formula: F/A = -π²ℏc/(240 d⁴)

Key question: Does the π²/240 factor have Fibonacci structure?

Note: 240 = 2⁴ × 15 = 16 × 15
      π²/240 ≈ 0.0411
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

def casimir_standard():
    """Standard Casimir force coefficient."""
    # F/A = -π²ℏc/(240 d⁴)
    # The dimensionless factor is π²/240
    return np.pi**2 / 240

def analyze_240():
    """Analyze the factor 240 for structure."""
    print("=" * 60)
    print("EXPERIMENT 14: CASIMIR EFFECT - BASELINE ANALYSIS")
    print("=" * 60)
    
    results = {
        'experiment': 'exp_14_casimir_baseline',
        'timestamp': datetime.now().isoformat()
    }
    
    # Standard Casimir coefficient
    casimir_coef = casimir_standard()
    print(f"\nStandard Casimir coefficient: π²/240 = {casimir_coef:.6f}")
    
    # Analyze 240
    print("\n" + "=" * 60)
    print("ANALYSIS OF 240")
    print("=" * 60)
    
    print("\nPrime factorization: 240 = 2⁴ × 3 × 5 = 16 × 15")
    print(f"  2⁴ = 16")
    print(f"  3 × 5 = 15")
    print(f"  Check: 16 × 15 = {16 * 15}")
    
    # Fibonacci connection?
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    print("\nFibonacci sequence:", fib[:10])
    
    # Check if 240 relates to Fibonacci
    print("\nFibonacci products near 240:")
    for i in range(len(fib)):
        for j in range(i, len(fib)):
            prod = fib[i] * fib[j]
            if 200 < prod < 280:
                print(f"  F_{i+1} × F_{j+1} = {fib[i]} × {fib[j]} = {prod}")
    
    # 3 × 5 × 16 = F_4 × F_5 × 16
    print("\nKey observation:")
    print(f"  240 = F_4 × F_5 × 2⁴ = 3 × 5 × 16")
    print(f"  240 = F_4 × F_5 × F_3⁴ = 3 × 5 × 2⁴")
    
    results['factor_240'] = {
        'value': 240,
        'factorization': '2^4 × 3 × 5',
        'fibonacci_form': 'F_3^4 × F_4 × F_5'
    }
    
    # Analyze π² / 240 as ratio
    print("\n" + "=" * 60)
    print("FIBONACCI APPROXIMATIONS TO π²/240")
    print("=" * 60)
    
    target = np.pi**2 / 240
    print(f"\nTarget: π²/240 = {target:.8f}")
    
    # Try Fibonacci-based approximations
    approximations = []
    
    for i in range(1, 12):
        for j in range(1, 12):
            ratio = fib[i-1] / (fib[j-1] * 240)
            error = abs(ratio - target) / target * 100
            if error < 10:
                approximations.append({
                    'formula': f"F_{i} / (F_{j} × 240)",
                    'value': ratio,
                    'error_pct': error
                })
    
    # Also try F_i / F_j forms
    for i in range(1, 12):
        for j in range(1, 12):
            ratio = fib[i-1] / fib[j-1]
            # What multiplier brings this to π²/240?
            if ratio > 0:
                multiplier = target / ratio
                if 0.001 < multiplier < 1000:
                    # Check if multiplier is simple
                    for k in [1, 2, 4, 8, 10, 16, 24, 30, 48, 60, 120, 240]:
                        test_val = ratio / k
                        error = abs(test_val - target) / target * 100
                        if error < 5:
                            approximations.append({
                                'formula': f"F_{i} / (F_{j} × {k})",
                                'value': test_val,
                                'error_pct': error
                            })
    
    # Sort by error
    approximations.sort(key=lambda x: x['error_pct'])
    
    print("\nBest Fibonacci approximations:")
    seen = set()
    count = 0
    for approx in approximations:
        key = f"{approx['value']:.8f}"
        if key not in seen:
            seen.add(key)
            print(f"  {approx['formula']}: {approx['value']:.8f} (error: {approx['error_pct']:.3f}%)")
            count += 1
            if count >= 5:
                break
    
    results['approximations'] = approximations[:10]
    
    # Golden ratio analysis
    print("\n" + "=" * 60)
    print("GOLDEN RATIO CONNECTION")
    print("=" * 60)
    
    phi = (1 + np.sqrt(5)) / 2
    
    # π² ≈ 9.8696...
    # 240 / π² ≈ 24.32...
    ratio_240_pi2 = 240 / np.pi**2
    print(f"\n240 / π² = {ratio_240_pi2:.6f}")
    print(f"φ⁷ = {phi**7:.6f}")
    print(f"Ratio: {ratio_240_pi2 / phi**7:.6f}")
    
    # Check various phi powers
    print("\nφ-power analysis:")
    for n in range(1, 15):
        phin = phi ** n
        ratio = 240 / phin
        pi2_approx = ratio
        if 8 < pi2_approx < 12:  # Near π²
            error = abs(pi2_approx - np.pi**2) / np.pi**2 * 100
            print(f"  240/φ^{n} = {pi2_approx:.6f} (π² error: {error:.2f}%)")
    
    # 240 / φ^7 ≈ 8.87 (close to π² = 9.87, about 10% off)
    # But 240 / 24 = 10, and 24 ≈ φ^7/1.19
    
    results['phi_analysis'] = {
        '240_over_pi2': ratio_240_pi2,
        'phi_7': phi**7,
        'ratio': ratio_240_pi2 / phi**7
    }
    
    # Dimensional analysis
    print("\n" + "=" * 60)
    print("SEC INTERPRETATION")
    print("=" * 60)
    
    print("""
The Casimir effect from SEC perspective:
  
  The vacuum between plates is a constrained information field.
  Boundary conditions create information gradients (∇I).
  The force emerges from entropy minimization.
  
  SEC: ∂S/∂t = α∇I - β∇H
  
  At equilibrium between plates:
  - ∇I is set by boundary conditions (plate separation)
  - ∇H is vacuum entropy gradient
  - Balance gives the Casimir force
  
  Question: Does π²/240 emerge from SEC with Fibonacci parameters?
""")
    
    # The 240 factor may relate to mode counting
    # In 3D, modes go as n³, with 240 = 2×3×5×8 having structure
    print("Mode counting interpretation:")
    print(f"  240 = (2×3) × (5×8) = 6 × 40")
    print(f"  240 = 4! × 10 = 24 × 10")
    print(f"  240 = 5! × 2 = 120 × 2")
    print(f"  4! = number of permutations in 4D")
    print(f"  This suggests dimensional combinatorics")
    
    results['sec_interpretation'] = {
        'factorial_form': '5! × 2 = 240',
        'combinatorial': '4! × 10 = 240',
        'dimensional_hint': 'Mode counting in bounded vacuum'
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_14_casimir_baseline_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    analyze_240()
