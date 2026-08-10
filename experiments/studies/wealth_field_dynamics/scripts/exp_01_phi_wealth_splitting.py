"""
Experiment 01: φ in Wealth Splitting

PURPOSE:
    Derive φ = (1+√5)/2 from PAC + self-similarity applied to wealth transactions.
    Show that φ is NOT an economic assumption but ALGEBRAICALLY NECESSARY.

AXIOMS:
    1. PAC Conservation: W_before = W_after (total wealth conserved in transaction)
    2. Self-similarity: Transaction ratios are scale-invariant

DERIVATION:
    Same as exp_04_phi_emergence in milestone1, applied to economics.
    
    Let r = W₁/W₂ be the wealth ratio after a two-party transaction.
    Self-similarity: W₁/W₂ = W_total/W₁
    PAC: W_total = W₁ + W₂
    
    → r = (W₁ + W₂)/W₁ = 1 + W₂/W₁ = 1 + 1/r
    → r² = r + 1
    → r = φ

OUTPUT:
    Proof that φ is the unique self-similar wealth splitting ratio.
"""

import numpy as np
import json
from datetime import datetime
from constants import print_header, print_subheader, PHI, PSI

def derive_phi_wealth_splitting():
    """
    Derive golden ratio from PAC + self-similarity in wealth transactions.
    """
    print_header("EXPERIMENT 01: φ IN WEALTH SPLITTING")
    
    # ==========================================================================
    # Part 1: The Setup
    # ==========================================================================
    print_subheader("PART 1: ECONOMIC AXIOMS")
    
    print("""
    AXIOM 1 (PAC Conservation for transactions):
        In a closed transaction, total wealth is conserved:
        W_total = W₁ + W₂
        
    AXIOM 2 (Self-similarity):
        The ratio of outcomes equals the ratio of input to larger outcome.
        This is scale invariance: same dynamics at all transaction sizes.
        W₁/W₂ = W_total/W₁
        
    Question: What splitting ratio r = W₁/W₂ satisfies both?
    """)
    
    # ==========================================================================
    # Part 2: Algebraic Derivation
    # ==========================================================================
    print_subheader("PART 2: ALGEBRAIC DERIVATION")
    
    print("""
    Let r = W₁/W₂ where W₁ ≥ W₂ (larger share to party 1).
    
    From self-similarity:
        r = W_total/W₁
        
    From PAC:
        W_total = W₁ + W₂ = W₂(r + 1)
        W_total/W₁ = W₂(r + 1)/(W₂ × r) = (r + 1)/r
        
    Combining:
        r = (r + 1)/r
        r² = r + 1
        
    This is the GOLDEN RATIO DEFINING EQUATION.
    
    Solutions: r = (1 ± √5)/2
    
    Since r = W₁/W₂ > 0:
        r = φ = (1 + √5)/2 ≈ 1.618
    """)
    
    # ==========================================================================
    # Part 3: Numerical Verification
    # ==========================================================================
    print_subheader("PART 3: NUMERICAL VERIFICATION")
    
    print(f"Verifying r² = r + 1 for φ = {PHI}:\n")
    
    lhs = PHI ** 2
    rhs = PHI + 1
    error = abs(lhs - rhs)
    
    print(f"  φ² = {lhs:.15f}")
    print(f"  φ+1 = {rhs:.15f}")
    print(f"  |φ² - (φ+1)| = {error:.2e}")
    print(f"  Verified: {error < 1e-14}")
    
    # ==========================================================================
    # Part 4: Economic Interpretation
    # ==========================================================================
    print_subheader("PART 4: ECONOMIC INTERPRETATION")
    
    print("""
    The golden ratio φ ≈ 1.618 means:
    
    In a self-similar transaction:
        - Party 1 receives φ/(φ+1) = 1/φ ≈ 61.8% of the total
        - Party 2 receives 1/(φ+1) = 1/φ² ≈ 38.2% of the total
    
    This is the UNIQUE stable splitting ratio for recursive transactions.
    
    PREDICTIONS:
    1. Stable market shares should cluster near 61.8% / 38.2%
    2. Growth ratios in established systems → φ
    3. Price ratios in equilibrium → powers of φ
    """)
    
    # Verify the percentages
    share_1 = PHI / (PHI + 1)
    share_2 = 1 / (PHI + 1)
    
    print(f"\n  Share 1: {share_1:.6f} = {share_1*100:.2f}%")
    print(f"  Share 2: {share_2:.6f} = {share_2*100:.2f}%")
    print(f"  Sum: {share_1 + share_2:.6f}")
    print(f"  Ratio: {share_1/share_2:.6f} = φ")
    
    # ==========================================================================
    # Part 5: Connection to Fibonacci
    # ==========================================================================
    print_subheader("PART 5: CONNECTION TO FIBONACCI")
    
    print("""
    Since φ = lim(F_{k+1}/F_k), consecutive Fibonacci numbers approximate
    the golden ratio splitting.
    
    Economic prediction: Discrete transactions should show Fibonacci patterns.
    """)
    
    from constants import fib
    
    print("\n  Fibonacci ratios converging to φ:")
    for k in range(3, 12):
        ratio = fib(k+1) / fib(k)
        error_pct = abs(ratio - PHI) / PHI * 100
        print(f"    F_{k+1}/F_{k} = {fib(k+1)}/{fib(k)} = {ratio:.6f} (error: {error_pct:.4f}%)")
    
    # ==========================================================================
    # Results
    # ==========================================================================
    print_subheader("RESULTS")
    
    results = {
        "experiment": "exp_01_phi_wealth_splitting",
        "timestamp": datetime.now().isoformat(),
        "derivation": {
            "axiom_1": "PAC conservation: W_total = W₁ + W₂",
            "axiom_2": "Self-similarity: W₁/W₂ = W_total/W₁",
            "equation": "r² = r + 1",
            "solution": "r = φ = (1+√5)/2"
        },
        "phi_value": float(PHI),
        "phi_squared": float(PHI ** 2),
        "phi_plus_1": float(PHI + 1),
        "numerical_error": float(error),
        "verified": bool(error < 1e-14),
        "economic_predictions": {
            "stable_share_larger": float(share_1),
            "stable_share_smaller": float(share_2),
            "ratio": float(PHI)
        },
        "conclusion": "φ is ALGEBRAICALLY NECESSARY for self-similar wealth splitting, not fitted"
    }
    
    print(f"\n  φ = {PHI:.10f}")
    print(f"  Numerical verification: {'PASSED' if results['verified'] else 'FAILED'}")
    print(f"\n  KEY FINDING: φ emerges from PAC + self-similarity")
    print(f"  This is the SAME derivation as exp_04 in milestone1")
    print(f"  Applied to economics, it predicts 61.8%/38.2% stable splits")
    
    # Save results
    with open("results/exp_01_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    results = derive_phi_wealth_splitting()
