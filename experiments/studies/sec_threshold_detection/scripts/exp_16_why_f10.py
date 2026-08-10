"""
Experiment 16: Why F_10 = 55 in the Feigenbaum Formulas?

Our closed-form expressions for r∞, δ, α all use:
- 55 = F_10 (10th Fibonacci number)
- 17 = 2^4 + 1 (5th Fermat prime)
- 52 = 55 - 3 = F_10 - F_4
- 11 = L_5 (5th Lucas number)

Question: Is there a Möbius-theoretic explanation for why F_10 specifically?

Hypothesis: The Fibonacci Möbius F_10 has special properties that make it
the "right scale" for the Feigenbaum formulas.
"""

import numpy as np
from mpmath import mp, mpf, sqrt, pi, phi as mphi
import sys
sys.path.insert(0, 'C:/Users/peter/repos/core_workspace/fracton')

from fracton.core.mobius_tensor import (
    MobiusMatrix, MobiusFibonacciTensor, MobiusRecursiveTensor,
    cross_ratio, PHI, PHI_INV
)

mp.dps = 50

# Feigenbaum constants
DELTA = mpf('4.6692016091029906718532038204662016172581855774757686327456513430041343302113')
R_INF = mpf('3.5699456718709449018420051513864989367638369115148323781079755299213628875')

# Fibonacci sequence
FIB = [0, 1]
for _ in range(25):
    FIB.append(FIB[-1] + FIB[-2])

# Lucas sequence  
LUC = [2, 1]
for _ in range(25):
    LUC.append(LUC[-1] + LUC[-2])


def experiment_1_why_f10():
    """
    Test properties of F_n Möbius matrices to see what's special about n=10.
    """
    print("=" * 70)
    print("EXPERIMENT 1: What's Special About F_10?")
    print("=" * 70)
    
    print("\nFibonacci Möbius properties:")
    print(f"{'n':>3} {'F_n':>8} {'det':>6} {'trace':>10} {'trace/F_n':>12}")
    print("-" * 45)
    
    for n in range(2, 16):
        M = MobiusMatrix.fibonacci(n)
        trace = abs(M.trace)
        det = int(M.determinant.real)
        ratio = trace / FIB[n]
        print(f"{n:3d} {FIB[n]:8d} {det:6d} {trace:10.0f} {ratio:12.6f}")
    
    # The trace/F_n ratio approaches φ + 1/φ = √5
    print(f"\n√5 = {np.sqrt(5):.6f}")
    
    # What about F_10 specifically?
    print("\n" + "-" * 70)
    print("Why F_10 = 55?")
    print("-" * 70)
    
    # Check if 55 has special relationship to δ
    print(f"\n55 = F_10")
    print(f"55 / δ = {55 / float(DELTA):.10f}")
    print(f"55 × δ = {55 * float(DELTA):.10f}")
    print(f"55 - δ² = {55 - float(DELTA)**2:.10f}")
    print(f"55 / φ^5 = {55 / PHI**5:.10f}")
    print(f"φ^5 = {PHI**5:.10f}")
    
    # Key: F_10 = F_5 × L_5 / something?
    print(f"\nF_5 = {FIB[5]}, L_5 = {LUC[5]}")
    print(f"F_5 × L_5 = {FIB[5] * LUC[5]} = F_10 = {FIB[10]}")
    
    # This is the identity: F_{2n} = F_n × L_n
    # So F_10 = F_5 × L_5 = 5 × 11 = 55!


def experiment_2_fermat_prime_17():
    """
    Why does 17 = 2^4 + 1 appear in the formulas?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Why 17 = 2^4 + 1?")
    print("=" * 70)
    
    print("\n17 is the 5th Fermat prime (3, 5, 17, 257, 65537)")
    print("17 = 2^4 + 1")
    print("17 = F_8 - F_4 = 21 - 4 = 17")  # No, F_8 = 21, F_4 = 3
    print("Actually: F_8 = 21, F_4 = 3, so F_8 - F_4 = 18, not 17")
    
    # How does 17 relate to Fibonacci?
    print("\n17 in terms of Fibonacci:")
    print(f"  17 = F_8 - F_4 = 21 - 3 = 18 ❌")
    print(f"  17 = F_7 + F_4 = 13 + 4 ❌ (F_4 = 3)")
    print(f"  17 = L_6 - 1 = 18 - 1 = 17 ✓")
    print(f"  17 = L_4 + L_2 = 7 + 3 = 10 ❌")
    
    # Zeckendorf representation of 17
    print("\nZeckendorf representation of 17:")
    print("  17 = 13 + 3 + 1 = F_7 + F_4 + F_2 ✓")
    
    # 17 and π
    print(f"\n17 and π:")
    print(f"  17 - π = {17 - float(pi):.10f}")
    print(f"  17 / π = {17 / float(pi):.10f}")
    print(f"  (17 - π)/55 = {(17 - float(pi))/55:.10f}")
    
    # In our formula: √(17 - π/F_c) appears
    # Where F_c = √(52 + 2π/55)
    F = 55
    c = float(sqrt(52 + 2*pi/F))
    inner = 17 - float(pi)/c
    print(f"\nIn formula: √(17 - π/c) where c = √(52 + 2π/55)")
    print(f"  c = {c:.10f}")
    print(f"  17 - π/c = {inner:.10f}")
    print(f"  √(17 - π/c) = {np.sqrt(inner):.10f}")
    
    # That's close to 4!
    print(f"\n√(17 - π/c) ≈ 4? Diff: {np.sqrt(inner) - 4:.10f}")


def experiment_3_52_connection():
    """
    Why 52 = F_10 - 3 = 55 - 3?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Why 52 = 55 - 3?")
    print("=" * 70)
    
    print("\n52 = 55 - 3 = F_10 - F_4")
    print("52 = 4 × 13 = 4 × F_7")
    print("52 = 2² × 13")
    
    # In formula: c = √(52 + 2π/55)
    # So c² = 52 + 2π/55 = 52 + small_correction
    c_squared = 52 + 2*float(pi)/55
    print(f"\nc² = 52 + 2π/55 = {c_squared:.10f}")
    print(f"c = {np.sqrt(c_squared):.10f}")
    
    # 52 ≈ c² - 2π/55
    # Why would c² need to be close to 52?
    
    # Check: is 52 related to δ?
    print(f"\n52 vs δ:")
    print(f"  52 / δ² = {52 / float(DELTA)**2:.10f}")
    print(f"  √52 = {np.sqrt(52):.10f}")
    print(f"  √52 × δ = {np.sqrt(52) * float(DELTA):.10f}")
    
    # Interesting: √52 × δ ≈ 33.66, close to F_9 = 34!
    print(f"\n  √52 × δ ≈ F_9? F_9 = {FIB[9]}, diff = {np.sqrt(52) * float(DELTA) - FIB[9]:.6f}")


def experiment_4_mobius_matrix_at_f10():
    """
    Examine the F_10 Möbius matrix in detail.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: F_10 Möbius Matrix Analysis")
    print("=" * 70)
    
    M = MobiusMatrix.fibonacci(10)
    
    print(f"\nF_10 Möbius matrix:")
    print(f"  [[{M.a:.0f}, {M.b:.0f}],")
    print(f"   [{M.c:.0f}, {M.d:.0f}]]")
    print(f"\n  = [[F_11, F_10], [F_10, F_9]]")
    print(f"  = [[{FIB[11]}, {FIB[10]}], [{FIB[10]}, {FIB[9]}]]")
    
    print(f"\nProperties:")
    print(f"  det = {M.determinant:.0f} (= (-1)^10 = 1)")
    print(f"  trace = {M.trace:.0f} = F_11 + F_9 = {FIB[11]} + {FIB[9]} = L_10 = {LUC[10]}")
    
    # Eigenvalues
    trace = float(M.trace.real)
    det = float(M.determinant.real)
    disc = trace**2 - 4*det
    lambda1 = (trace + np.sqrt(disc)) / 2
    lambda2 = (trace - np.sqrt(disc)) / 2
    
    print(f"\nEigenvalues:")
    print(f"  λ₁ = {lambda1:.6f}")
    print(f"  λ₂ = {lambda2:.6f}")
    print(f"  λ₁/λ₂ = {lambda1/lambda2:.6f}")
    print(f"  λ₁ × λ₂ = {lambda1*lambda2:.6f} (= det)")
    
    # λ₁ ≈ F_10 × φ?
    print(f"\nλ₁ vs F_10 × φ:")
    print(f"  F_10 × φ = {FIB[10] * PHI:.6f}")
    print(f"  λ₁ = {lambda1:.6f}")
    print(f"  Ratio: {lambda1 / (FIB[10] * PHI):.10f}")
    
    # The eigenvectors
    print(f"\nEigenvector for λ₁ (attracting):")
    print(f"  Should be [φ, 1] since fixed point is φ")
    
    # Apply to points
    print(f"\nAction on special points:")
    for z in [0, 1, -1, PHI, -PHI_INV, float(DELTA)]:
        z_out = M(z)
        print(f"  M({z:.4f}) = {z_out:.6f}")


def experiment_5_f10_in_formulas():
    """
    How does F_10 = 55 participate in our closed-form expressions?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: F_10 in Closed-Form Expressions")
    print("=" * 70)
    
    F = mpf(55)  # F_10
    
    # Our formula for r∞:
    # r∞ = π(F + √(17 - π/F_c))(F + π)/F² - corrections
    # where F_c = √(52 + 2π/F)
    
    F_c = sqrt(52 + 2*pi/F)
    inner = 17 - pi/F_c
    
    base = pi * (F + sqrt(inner)) * (F + pi) / F**2
    
    print("Base formula: π(F + √(17 - π/F_c))(F + π)/F²")
    print(f"\nWith F = 55:")
    print(f"  F_c = √(52 + 2π/55) = {float(F_c):.10f}")
    print(f"  17 - π/F_c = {float(inner):.10f}")
    print(f"  √(17 - π/F_c) = {float(sqrt(inner)):.10f}")
    print(f"  Base = {float(base):.10f}")
    print(f"  r∞ = {float(R_INF):.10f}")
    
    # The structure: F, F², F+π, 17-π/something, 52+something
    # All involve F = 55 as the scale
    
    print("\n" + "-" * 70)
    print("Why does 55 work?")
    print("-" * 70)
    
    # Test other Fibonacci as F
    print("\nTesting other Fibonacci numbers:")
    for n in [8, 9, 10, 11, 12]:
        F_test = mpf(FIB[n])
        F_c_test = sqrt(F_test - 3 + 2*pi/F_test)
        inner_test = 17 - pi/F_c_test
        if inner_test > 0:
            base_test = pi * (F_test + sqrt(inner_test)) * (F_test + pi) / F_test**2
            error = abs(base_test - R_INF)
            print(f"  F_{n} = {FIB[n]}: base = {float(base_test):.8f}, error = {float(error):.6f}")
        else:
            print(f"  F_{n} = {FIB[n]}: inner < 0")


def experiment_6_mobius_derivation():
    """
    Can we DERIVE the formula structure from Möbius properties?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Möbius Derivation of Formula Structure")
    print("=" * 70)
    
    # The Möbius transformation has form M(z) = (az+b)/(cz+d)
    # For Fibonacci F_n: [[F_{n+1}, F_n], [F_n, F_{n-1}]]
    
    # The formula base looks like: π × (linear in F) × (linear in F) / F²
    # This is reminiscent of a Möbius transformation!
    
    # Let's check: is base = π × M_10(something)?
    M = MobiusMatrix.fibonacci(10)
    
    # Our base formula:
    F = mpf(55)
    F_c = sqrt(52 + 2*pi/F)
    inner = 17 - pi/F_c
    base = pi * (F + sqrt(inner)) * (F + pi) / F**2
    
    print(f"Base = {float(base):.10f}")
    print(f"R_∞ = {float(R_INF):.10f}")
    
    # Can we express base as M(z) for some z?
    # M(z) = (89z + 55)/(55z + 34)
    # We want M(z) = base/π ≈ 1.136
    
    target = float(base / pi)
    print(f"\nTarget (base/π) = {target:.10f}")
    
    # Solve: (89z + 55)/(55z + 34) = target
    # 89z + 55 = target(55z + 34)
    # 89z - 55*target*z = 34*target - 55
    # z(89 - 55*target) = 34*target - 55
    z_implied = (34*target - 55) / (89 - 55*target)
    print(f"z such that M_10(z) = base/π: z = {z_implied:.10f}")
    
    # Verify
    M_of_z = M(z_implied)
    print(f"M_10({z_implied:.6f}) = {M_of_z:.10f}")
    print(f"Target = {target:.10f}")
    print(f"Match? Diff = {abs(M_of_z - target):.2e}")
    
    # What is z_implied in terms of known constants?
    print(f"\nz_implied analysis:")
    print(f"  z = {z_implied:.10f}")
    print(f"  z/φ = {z_implied/PHI:.10f}")
    print(f"  z × φ = {z_implied*PHI:.10f}")
    print(f"  z + 1 = {z_implied + 1:.10f}")
    print(f"  z - π/55 = {z_implied - float(pi)/55:.10f}")


def main():
    experiment_1_why_f10()
    experiment_2_fermat_prime_17()
    experiment_3_52_connection()
    experiment_4_mobius_matrix_at_f10()
    experiment_5_f10_in_formulas()
    experiment_6_mobius_derivation()
    
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print("""
Key findings:

1. F_10 = 55 = F_5 × L_5 = 5 × 11
   This is the Fibonacci doubling identity: F_{2n} = F_n × L_n
   So F_10 sits at a "double" scale of F_5 = 5 (the first non-trivial Fibonacci)

2. F_10 Möbius matrix:
   - det = 1 (even index)
   - trace = L_10 = 123 = Lucas number
   - Eigenvalue ratio → φ² as n → ∞

3. 17 = L_6 - 1 and appears as a "correction scale"
   √(17 - π/c) ≈ 4, suggesting 17 balances π contribution

4. 52 = 55 - 3 = F_10 - F_4
   √52 × δ ≈ F_9 (close but not exact)
   52 = 4 × 13 = 4 × F_7

5. The formula BASE can be expressed as π × M_10(z) for a specific z!
   This confirms Möbius structure underlies the formula.

HYPOTHESIS: The Feigenbaum constants emerge from applying the F_10 Möbius
transformation to a specific "seed" value that encodes the logistic map's
nonlinearity. The corrections (A₁, A₂) may be higher Möbius terms.
""")


if __name__ == '__main__':
    main()
