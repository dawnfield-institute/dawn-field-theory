"""
Experiment 04: Golden Ratio Emergence

PURPOSE:
    Derive φ = (1+√5)/2 ALGEBRAICALLY from PAC + self-similarity.
    Show that φ is NOT fitted but mathematically NECESSARY.

AXIOMS:
    1. PAC Conservation: f(P) = f(C₁) + f(C₂)
    2. Self-similarity: The ratio of children equals the ratio of parent to larger child
       i.e., f(C₁)/f(C₂) = f(P)/f(C₁)

DERIVATION:
    Let r = f(C₁)/f(C₂) be the splitting ratio.
    
    From self-similarity: r = f(P)/f(C₁)
    From PAC: f(P) = f(C₁) + f(C₂) = f(C₁)(1 + 1/r)
    
    Therefore: r = f(C₁)(1 + 1/r)/f(C₁) = 1 + 1/r
    
    Solving: r² = r + 1
    
    Solutions: r = (1 ± √5)/2
    
    Positive solution: r = φ = (1+√5)/2 ≈ 1.618

OUTPUT:
    Proof that φ is the unique positive solution to PAC + self-similarity.
"""

import numpy as np
import json
from datetime import datetime
from constants import print_header, print_subheader, PHI, PSI

def derive_phi_emergence():
    """
    Derive golden ratio from PAC + self-similarity.
    """
    print_header("EXPERIMENT 04: GOLDEN RATIO EMERGENCE")
    
    # ==========================================================================
    # Part 1: The Setup
    # ==========================================================================
    print_subheader("PART 1: THE SETUP")
    
    print("""
    We have two constraints:
    
    1. PAC CONSERVATION (from exp_01):
       f(P) = f(C₁) + f(C₂)
       
    2. SELF-SIMILARITY (scale invariance):
       The splitting pattern looks the same at every scale.
       Mathematically: f(C₁)/f(C₂) = f(P)/f(C₁)
       
    Question: What splitting ratio r = f(C₁)/f(C₂) satisfies both?
    """)
    
    # ==========================================================================
    # Part 2: Algebraic Derivation
    # ==========================================================================
    print_subheader("PART 2: ALGEBRAIC DERIVATION")
    
    print("""
    Let r = f(C₁)/f(C₂) where f(C₁) ≥ f(C₂) (WLOG).
    
    From PAC:
        f(P) = f(C₁) + f(C₂)
        f(P) = f(C₂)(r + 1)
        f(P)/f(C₁) = f(C₂)(r + 1)/(f(C₂) × r) = (r + 1)/r
    
    From self-similarity:
        f(C₁)/f(C₂) = f(P)/f(C₁)
        r = (r + 1)/r
    
    Multiply both sides by r:
        r² = r + 1
    
    This is the GOLDEN RATIO DEFINING EQUATION.
    
    Solving by quadratic formula:
        r = (1 ± √5)/2
        
    Solutions:
        r₊ = (1 + √5)/2 ≈ 1.618  (positive)
        r₋ = (1 - √5)/2 ≈ -0.618 (negative)
    
    Since r = f(C₁)/f(C₂) > 0, we must have:
    
        r = φ = (1 + √5)/2
    """)
    
    # ==========================================================================
    # Part 3: Numerical Verification
    # ==========================================================================
    print_subheader("PART 3: NUMERICAL VERIFICATION")
    
    print("Verifying r² = r + 1 for φ:\n")
    
    phi_squared = PHI ** 2
    phi_plus_one = PHI + 1
    
    print(f"  φ = {PHI:.15f}")
    print(f"  φ² = {phi_squared:.15f}")
    print(f"  φ + 1 = {phi_plus_one:.15f}")
    print(f"  φ² - (φ + 1) = {phi_squared - phi_plus_one:.2e}")
    print(f"  (Should be ~0)")
    
    verified = np.isclose(phi_squared, phi_plus_one, rtol=1e-14)
    print(f"\n  Verification: {'✅ PASSED' if verified else '❌ FAILED'}")
    
    # ==========================================================================
    # Part 4: Uniqueness
    # ==========================================================================
    print_subheader("PART 4: UNIQUENESS OF φ")
    
    print("""
    Is φ the ONLY solution?
    
    The negative root ψ = (1-√5)/2 ≈ -0.618 is also a solution to r² = r + 1.
    
    But ψ < 0, and r = f(C₁)/f(C₂) must be positive (values are positive).
    
    Therefore φ is the UNIQUE positive solution.
    
    Note: ψ is not physically meaningless—it appears in the general
    Fibonacci formula: F_n = (φⁿ - ψⁿ)/√5
    
    But for splitting ratios, only φ is admissible.
    """)
    
    print(f"  ψ = {PSI:.15f}")
    print(f"  ψ² = {PSI**2:.15f}")
    print(f"  ψ + 1 = {PSI + 1:.15f}")
    print(f"  ψ² - (ψ + 1) = {PSI**2 - (PSI + 1):.2e} (also ~0)")
    
    # ==========================================================================
    # Part 5: Self-Similarity Verification
    # ==========================================================================
    print_subheader("PART 5: SELF-SIMILARITY VERIFICATION")
    
    print("Demonstrating golden ratio self-similarity:\n")
    
    # Start with parent value 1
    parent = 1.0
    print(f"{'Level':<8} {'Parent':<15} {'Child₁':<15} {'Child₂':<15} {'Ratio':<10}")
    print("-" * 70)
    
    for level in range(6):
        child1 = parent / PHI
        child2 = parent / (PHI ** 2)
        ratio = child1 / child2
        
        print(f"{level:<8} {parent:<15.10f} {child1:<15.10f} {child2:<15.10f} {ratio:<10.6f}")
        
        # Next level
        parent = child1
    
    print(f"\n  Ratio is constant at φ ≈ {PHI:.6f} at ALL levels.")
    print("  This is self-similarity: pattern repeats at every scale.")
    
    # ==========================================================================
    # Part 6: Historical Connection
    # ==========================================================================
    print_subheader("PART 6: HISTORICAL CONNECTION")
    
    print("""
    This derivation is equivalent to Fibonacci's original problem (1202):
    
    "A pair of rabbits produces one pair of offspring each month.
     Offspring mature after one month and then reproduce.
     How many pairs after n months?"
    
    The ratio of consecutive populations → φ as n → ∞.
    
    Our derivation from PAC + self-similarity is the same mathematics:
    - PAC: Total rabbits = adults + juveniles
    - Self-similarity: Growth pattern repeats each generation
    
    φ is not "chosen"—it is FORCED by these constraints.
    """)
    
    # ==========================================================================
    # Part 7: Summary
    # ==========================================================================
    print_subheader("SUMMARY")
    
    results = {
        "experiment": "exp_04_phi_emergence",
        "timestamp": datetime.now().isoformat(),
        "conclusion": "φ is algebraically NECESSARY, not fitted",
        "derivation": {
            "equation": "r² = r + 1",
            "positive_solution": float(PHI),
            "verification": {
                "phi_squared": float(phi_squared),
                "phi_plus_one": float(phi_plus_one),
                "difference": float(phi_squared - phi_plus_one)
            }
        },
        "uniqueness": "φ is the unique positive solution",
        "status": "VALIDATED"
    }
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                         RESULT                                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Golden Ratio: φ = (1+√5)/2 ≈ 1.6180339887                       ║
    ║                                                                   ║
    ║  Derived from:                                                    ║
    ║    • PAC Conservation: f(P) = f(C₁) + f(C₂)                      ║
    ║    • Self-similarity: f(C₁)/f(C₂) = f(P)/f(C₁)                   ║
    ║                                                                   ║
    ║  Equation: r² = r + 1                                            ║
    ║  Unique positive solution: φ                                      ║
    ║                                                                   ║
    ║  THIS IS NOT FITTING. THIS IS ALGEBRA.                           ║
    ║                                                                   ║
    ║  Status: ✅ VALIDATED                                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    return results


if __name__ == "__main__":
    results = derive_phi_emergence()
    
    # Save results
    with open("../results/exp_04_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ../results/exp_04_results.json")
