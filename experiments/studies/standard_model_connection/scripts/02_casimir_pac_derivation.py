#!/usr/bin/env python3
"""
02_casimir_pac_derivation.py - Casimir Effect from PAC Mode Counting

REFINED VERSION: Focus on what PAC can actually say about Casimir.

KEY INSIGHT: The Casimir effect uses zeta function regularization.
  - Standard: sum_n n gives zeta(-1) = -1/12 after regularization
  - This magical -1/12 is what gives pi^2/720 coefficient

PAC APPROACH:
  - PAC tree gives NATURAL UV regularization via k^-2 weighting
  - Question: Does PAC's regularization reproduce -1/12?
  - If yes: PAC explains WHY zeta regularization works
  - If no: Casimir constrains PAC mode counting

NEW TEST: Check if Ramanujan summation (which gives -1/12) has
Fibonacci structure.

Author: Dawn Field Institute
Date: December 2025
Status: Experimental (REFINED)
"""

import numpy as np
from scipy import constants
import json
from datetime import datetime

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

HBAR = constants.hbar      # 1.054571817e-34 J·s
C = constants.c            # 299792458 m/s
PI = np.pi

# Casimir coefficient (exact)
CASIMIR_COEFF = PI**2 / 240  # ≈ 0.0411

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# STANDARD CASIMIR CALCULATION
# =============================================================================

def standard_casimir_pressure(d):
    """
    Standard QED Casimir pressure between parallel plates.
    
    P = -π²ħc / (240 d⁴)
    
    Args:
        d: plate separation in meters
    
    Returns:
        pressure in Pascals (N/m²)
    """
    return -PI**2 * HBAR * C / (240 * d**4)

def standard_casimir_energy_density(d):
    """
    Casimir energy per unit area.
    
    E/A = -π²ħc / (720 d³)
    """
    return -PI**2 * HBAR * C / (720 * d**3)

# =============================================================================
# PAC MODE COUNTING APPROACH
# =============================================================================

def pac_mode_spectrum(k_max, depth=20):
    """
    Generate PAC tree mode spectrum.
    
    On a PAC tree, total energy at wavenumber k scales as:
      E(k) ~ k⁻² (topological)
    
    This means mode density goes as:
      ρ(k) ~ k⁻²
    
    Args:
        k_max: maximum wavenumber (UV cutoff)
        depth: tree depth
    
    Returns:
        k_values, mode_weights
    """
    # Binary tree: k = 2^level
    levels = np.arange(depth)
    k_values = 2.0 ** levels
    
    # PAC k⁻² spectrum: weight at each level
    # More weight at low k, less at high k
    mode_weights = k_values ** (-2)
    
    # Normalize so total = k_max modes
    mode_weights = mode_weights / mode_weights.sum() * k_max
    
    return k_values, mode_weights

def pac_mode_count_in_cavity(d, k_max):
    """
    Count allowed PAC modes in a cavity of size d.
    
    Boundary conditions: k_n = nπ/d for n = 1, 2, 3, ...
    
    The key insight: PAC gives weight k⁻² to each mode,
    rather than uniform weight.
    
    Standard sum: Σ_n (nπ/d) → divergent, needs regularization
    PAC sum: Σ_n (nπ/d)⁻¹ → convergent!
    """
    # Allowed wavenumbers in cavity
    n_max = int(k_max * d / PI)
    n_values = np.arange(1, n_max + 1)
    k_allowed = n_values * PI / d
    
    # Standard mode energy: ½ħω = ½ħck
    standard_energy = 0.5 * HBAR * C * k_allowed
    
    # PAC-weighted mode energy: weight each mode by k⁻²
    # This is equivalent to: each mode contributes ½ħck × (k₀/k)²
    # where k₀ is some reference scale
    k0 = PI / d  # fundamental mode
    pac_weight = (k0 / k_allowed) ** 2
    pac_energy = standard_energy * pac_weight
    
    return {
        'n_modes': len(n_values),
        'k_allowed': k_allowed,
        'standard_energy_sum': np.sum(standard_energy),
        'pac_energy_sum': np.sum(pac_energy)
    }

def derive_casimir_from_pac(d_values):
    """
    Attempt to derive Casimir force from PAC mode counting.
    
    The challenge: Casimir requires comparing infinite sums between
    cavity and free space, with careful regularization.
    
    PAC approach: The k⁻² weighting provides natural UV regularization.
    """
    print("=" * 70)
    print("PAC CASIMIR DERIVATION")
    print("=" * 70)
    
    results = []
    
    # UV cutoff (should drop out of final answer)
    k_max = 1e10  # 1/m, roughly atomic scale
    
    print(f"\nUsing UV cutoff k_max = {k_max:.0e} m⁻¹")
    print(f"\nPlate separation | QED Pressure | PAC Analysis")
    print("-" * 55)
    
    for d in d_values:
        # Standard QED result
        P_qed = standard_casimir_pressure(d)
        
        # PAC mode counting
        pac_result = pac_mode_count_in_cavity(d, k_max)
        
        # Energy difference between cavity and free space
        # Free space: continuous spectrum, but PAC-weighted
        # Cavity: discrete spectrum, PAC-weighted
        
        # In standard derivation:
        # ΔE = ½ħc [Σ_cavity k_n - ∫ dk k]
        # The difference gives finite result after regularization
        
        # In PAC: the k⁻² weighting changes the sum
        # Σ k × k⁻² = Σ k⁻¹ which is harmonic series
        
        # Actually, let's compute the standard way but check scaling
        results.append({
            'd_m': d,
            'P_qed_Pa': P_qed,
            'n_modes': pac_result['n_modes'],
            'standard_sum': pac_result['standard_energy_sum'],
            'pac_sum': pac_result['pac_energy_sum']
        })
        
        print(f"{d:15.2e} | {P_qed:12.4e} | {pac_result['n_modes']:5d} modes")
    
    return results

# =============================================================================
# DIMENSIONAL ANALYSIS WITH PAC
# =============================================================================

def pac_dimensional_analysis():
    """
    Derive Casimir scaling from PAC dimensional analysis.
    
    Standard argument:
      [E/A] = energy/area = J/m² = kg/s²
      Available quantities: ħ (J·s), c (m/s), d (m)
      Only combination: ħc/d³
      
    With PAC:
      Additional quantity: φ (dimensionless)
      Could have: (ħc/d³) × f(φ)
    """
    print("\n" + "=" * 70)
    print("DIMENSIONAL ANALYSIS")
    print("=" * 70)
    
    print("""
STANDARD DIMENSIONAL ANALYSIS:

  Available quantities: ħ [J·s], c [m/s], d [m]
  
  Energy per area: [E/A] = [J/m²] = [kg·m²/s²]/[m²] = [kg/s²]
  
  Combination: ħᵃ cᵇ dᶜ with [J·s]ᵃ [m/s]ᵇ [m]ᶜ = [kg/s²]
  
  From J = kg·m²/s²:
    [kg]¹ [m]⁰ [s]⁻² = [kg·m²·s⁻²]ᵃ [s]ᵃ [m·s⁻¹]ᵇ [m]ᶜ
                     = [kg]ᵃ [m]²ᵃ⁺ᵇ⁺ᶜ [s]⁻²ᵃ⁺ᵃ⁻ᵇ
                     = [kg]ᵃ [m]²ᵃ⁺ᵇ⁺ᶜ [s]⁻ᵃ⁻ᵇ
  
  Matching:
    kg: a = 1
    s:  -a - b = -2 → b = 1
    m:  2a + b + c = 0 → c = -3
  
  Therefore: E/A ∝ ħc/d³
  
  The coefficient π²/720 comes from the mode sum.
    """)
    
    print("""
PAC MODIFICATION:

  If PAC k⁻² spectrum changes the mode sum, we might get a different
  coefficient, but the d⁻³ scaling is fixed by dimensional analysis.
  
  The question is: does PAC predict the coefficient π²/720?
  
  Standard sum: Σₙ n → needs regularization (zeta function: ζ(-1) = -1/12)
  PAC sum: Σₙ n⁻¹ → diverges logarithmically
  
  This suggests PAC doesn't directly give Casimir. BUT:
  
  The PAC k⁻² is for TOTAL energy at wavenumber k, not mode density.
  Mode density is n(k) = dk/dε where ε is energy per mode.
  
  If E_total(k) ~ k⁻², and there are N(k) ~ k modes at level k,
  then E_per_mode ~ k⁻³.
  
  This is FASTER falloff than standard QED (where all modes contribute equally).
    """)
    
    # Check: does Σ n⁻³ give π² coefficient?
    n_max = 100000
    n = np.arange(1, n_max + 1, dtype=float)
    
    # Standard Casimir uses ζ(-3) via regularization
    # But direct sum of n gives... well, it diverges
    
    # Sum of n⁻³ converges to ζ(3) ≈ 1.202
    zeta_3 = np.sum(n ** (-3))
    print(f"\n  Sum n^-3 (n=1 to {n_max}) = {zeta_3:.6f}")
    print(f"  zeta(3) = 1.202056903...")
    
    # Sum of n⁻² converges to ζ(2) = π²/6
    zeta_2 = np.sum(n ** (-2))
    print(f"  Sum n^-2 = {zeta_2:.6f}")
    print(f"  zeta(2) = pi^2/6 = {PI**2/6:.6f}")
    
    # The Casimir coefficient π²/720 = π²/(6×120) = ζ(2)/120
    print(f"\n  Casimir coefficient: π²/720 = {PI**2/720:.6f}")
    print(f"  This is ζ(2)/120 = {zeta_2/120:.6f}")

def test_pac_casimir_coefficient():
    """
    Test if PAC tree structure predicts Casimir coefficient.
    
    On a PAC tree with Fibonacci structure:
      - F_7 = 13 total "slots"
      - Mode sum might involve F_n ratios
    """
    print("\n" + "=" * 70)
    print("PAC COEFFICIENT TEST")
    print("=" * 70)
    
    # The coefficient π²/720 ≈ 0.01370
    casimir_coeff = PI**2 / 720
    
    # Fibonacci ratios to test
    from fractions import Fraction
    
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    print(f"\nTarget: π²/720 = {casimir_coeff:.6f}")
    print(f"\nTesting Fibonacci ratio combinations:")
    print("-" * 40)
    
    best_match = None
    best_error = 1.0
    
    for i in range(len(fib)):
        for j in range(len(fib)):
            for k in range(len(fib)):
                if fib[i] * fib[j] > 0:
                    # Try F_i / (F_j × F_k × something)
                    for factor in [1, 6, 12, 24, 60, 120, 720]:
                        ratio = fib[i] / (fib[j] * fib[k] * factor)
                        error = abs(ratio - casimir_coeff) / casimir_coeff
                        if error < best_error:
                            best_error = error
                            best_match = (i+1, j+1, k+1, factor, ratio)
    
    if best_match:
        i, j, k, factor, ratio = best_match
        print(f"  Best match: F_{i}/(F_{j}×F_{k}×{factor}) = {ratio:.6f}")
        print(f"  Error: {best_error*100:.2f}%")
    
    # Also try φ-based expressions
    print(f"\n  φ/(360) = {PHI/360:.6f} (error: {abs(PHI/360 - casimir_coeff)/casimir_coeff*100:.1f}%)")
    print(f"  1/(φ⁵×18) = {1/(PHI**5 * 18):.6f} (error: {abs(1/(PHI**5*18) - casimir_coeff)/casimir_coeff*100:.1f}%)")
    print(f"  π²/(6×F_8) = {PI**2/(6*21):.6f} (error: {abs(PI**2/(6*21) - casimir_coeff)/casimir_coeff*100:.1f}%)")

# =============================================================================
# EXPERIMENTAL COMPARISON
# =============================================================================

def compare_to_experiment():
    """
    Compare Casimir predictions to experimental measurements.
    
    Key experiments:
    - Lamoreaux (1997): First precision measurement, ~5% accuracy
    - Decca et al. (2007): ~0.5% accuracy at 160-750 nm
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTAL COMPARISON")
    print("=" * 70)
    
    print("""
EXPERIMENTAL MEASUREMENTS:

1. Lamoreaux (1997)
   - Torsion pendulum with sphere-plate geometry
   - Distance: 0.6-6 μm
   - Accuracy: ~5%
   - Result: Confirmed QED prediction

2. Decca et al. (2007)
   - MEMS-based torsion oscillator
   - Distance: 160-750 nm
   - Accuracy: ~0.5%
   - Result: Confirmed QED at 0.5% level

3. Modern tests (2020s)
   - Testing for non-Newtonian gravity
   - Distance: <100 nm
   - Looking for deviations from QED

QED PREDICTION vs PAC:

The standard QED Casimir effect is extremely well-verified.
PAC would need to either:
  (a) Exactly reproduce QED result (validates mode counting)
  (b) Predict small corrections (testable)
  (c) Fail to reproduce (falsifies PAC mode counting)
    """)
    
    # Table of QED predictions at experimental distances
    distances = [100e-9, 200e-9, 500e-9, 1e-6, 2e-6, 5e-6]  # meters
    
    print("\nQED Predictions at experimental distances:")
    print("-" * 50)
    print(f"{'d (nm)':>10} | {'Pressure (Pa)':>15} | {'F/A (mN/m²)':>15}")
    print("-" * 50)
    
    for d in distances:
        P = standard_casimir_pressure(d)
        print(f"{d*1e9:10.0f} | {P:15.4e} | {P*1e3:15.4e}")

# =============================================================================
# REFINED: ZETA FUNCTION AND FIBONACCI
# =============================================================================

def test_zeta_fibonacci_connection():
    """
    REFINED TEST: The -1/12 in Casimir comes from zeta(-1).
    
    Does zeta function at negative integers have Fibonacci structure?
    
    Key values:
      zeta(-1) = -1/12  (Ramanujan's 1+2+3+... = -1/12)
      zeta(-3) = 1/120
      zeta(-5) = -1/252
      zeta(2) = pi^2/6
      zeta(4) = pi^4/90
    
    The Casimir coefficient pi^2/720 = zeta(2)/120 = (pi^2/6)/120
    And 120 = zeta(-3)^(-1)
    
    So: Casimir coefficient = zeta(2) * zeta(-3) = (pi^2/6) * (1/120)
    """
    print("\n" + "=" * 70)
    print("REFINED: ZETA FUNCTION AND FIBONACCI")
    print("=" * 70)
    
    # Zeta function at negative odd integers (Bernoulli numbers)
    # zeta(1-2n) = -B_{2n}/(2n) where B_n are Bernoulli numbers
    
    # Bernoulli numbers
    B = {
        0: 1,
        1: -1/2,  # or +1/2 by some conventions
        2: 1/6,
        4: -1/30,
        6: 1/42,
        8: -1/30,
        10: 5/66,
        12: -691/2730,
    }
    
    # Zeta at negative integers
    # zeta(-1) = -B_2/2 = -(1/6)/2 = -1/12
    # zeta(-3) = -B_4/4 = -(-1/30)/4 = 1/120
    # zeta(-5) = -B_6/6 = -(1/42)/6 = -1/252
    
    print("\nZeta function at negative integers:")
    print("-" * 50)
    
    zeta_neg = {
        -1: -1/12,
        -3: 1/120,
        -5: -1/252,
        -7: 1/240,
        -9: -1/132,
    }
    
    for n, val in zeta_neg.items():
        print(f"  zeta({n}) = {val:.6f} = {val.as_integer_ratio()[0]}/{val.as_integer_ratio()[1]}")
    
    # Check denominators for Fibonacci structure
    print("\nDenominators:")
    print("  12 = 3 × 4 = F_4 × 4")
    print("  120 = 5! = 8 × 15 = F_6 × 15")
    print("  252 = 4 × 63 = 4 × 7 × 9")
    print("  240 = 16 × 15 = 2^4 × 15")
    
    # More interesting: Casimir coefficient structure
    print("\n" + "-" * 50)
    print("CASIMIR COEFFICIENT DECOMPOSITION:")
    print("-" * 50)
    
    print(f"\n  pi^2/720 = {PI**2/720:.8f}")
    print(f"  = (pi^2/6) × (1/120)")
    print(f"  = zeta(2) × zeta(-3)")
    print(f"  = {PI**2/6:.6f} × {1/120:.6f}")
    
    # Does 720 have Fibonacci structure?
    print(f"\n  720 = 6! = 720")
    print(f"      = 8 × 90 = F_6 × 90")
    print(f"      = 5 × 144 = F_5 × F_12")
    print(f"      = 3 × 240 = F_4 × 240")
    print(f"      = 13 × 55.38... (not integer)")
    
    # Key finding: 720 = F_5 × F_12 = 5 × 144
    print(f"\n  KEY: 720 = F_5 × F_12 = 5 × 144")
    print(f"  So: Casimir coefficient = pi^2 / (F_5 × F_12)")
    
    # Verify
    casimir_coeff = PI**2 / 720
    fib_prediction = PI**2 / (5 * 144)
    print(f"\n  Verification: pi^2/(F_5 × F_12) = {fib_prediction:.8f}")
    print(f"  Standard:     pi^2/720        = {casimir_coeff:.8f}")
    print(f"  Match: {np.isclose(casimir_coeff, fib_prediction)}")
    
    # Can we predict the pi^2 from PAC?
    print("\n" + "-" * 50)
    print("THE pi^2 FACTOR:")
    print("-" * 50)
    print("""
  The pi^2 comes from zeta(2) = sum(1/n^2) = pi^2/6
  
  Euler's proof: pi^2/6 = 1 + 1/4 + 1/9 + 1/16 + ...
  
  This is purely about the GEOMETRY of mode counting.
  
  On a PAC tree:
  - Modes at level k have energy ~ k
  - Total modes at level k = 2^k (binary tree)
  - Sum: 2^0 × 1 + 2^1 × 2 + 2^2 × 3 + ... (standard cascade)
  
  PAC weighting k^-2 would give:
  - Sum: 2^0/1 + 2^1/4 + 2^2/9 + ... = sum(2^k/k^2)
  - This is NOT zeta(2); it diverges!
  
  So PAC k^-2 doesn't directly give Casimir coefficient.
  The 720 = F_5 × F_12 decomposition is interesting but may be coincidence.
    """)
    
    return {
        'casimir_coeff': casimir_coeff,
        '720_decomposition': '5 × 144 = F_5 × F_12',
        'status': 'Interesting but inconclusive'
    }

def test_pac_natural_cutoff():
    """
    REFINED: Can PAC provide NATURAL UV cutoff for Casimir?
    
    Standard Casimir needs regularization because sum diverges.
    PAC k^-2 spectrum provides natural suppression of high modes.
    
    Does this give PHYSICAL UV cutoff?
    """
    print("\n" + "=" * 70)
    print("REFINED: PAC AS NATURAL CUTOFF")
    print("=" * 70)
    
    print("""
HYPOTHESIS: PAC provides natural UV regularization.

In QFT, we sum modes: E = sum_k (1/2) hbar omega_k
For Casimir: omega_k = c k_n where k_n = n pi/d
So: E = (hbar c / 2) sum_n (n pi/d) = (hbar c pi / 2d) sum_n n

This diverges! Standard fix: zeta regularization gives -1/12.

PAC ALTERNATIVE:
If each mode is WEIGHTED by k^-2, then:
E_PAC = (hbar c / 2) sum_n (n pi/d) × (d/n pi)^2
      = (hbar c / 2) sum_n (d / n pi)
      = (hbar c d / 2 pi) sum_n (1/n)
      = (hbar c d / 2 pi) × (divergent harmonic series)

This STILL diverges, just more slowly (ln N vs N^2).

CONCLUSION: PAC k^-2 doesn't fix Casimir divergence.
The regularization must come from somewhere else.
    """)
    
    # Numerical demonstration
    N_max = 1000
    n = np.arange(1, N_max + 1, dtype=float)
    
    # Standard sum (diverges as N^2)
    standard_sum = np.cumsum(n)
    
    # PAC-weighted sum (diverges as ln N)
    pac_sum = np.cumsum(1/n)  # harmonic series
    
    print(f"\nNumerical comparison (N = {N_max}):")
    print(f"  Standard sum(n):   {standard_sum[-1]:.2e} (grows as N^2)")
    print(f"  PAC sum(1/n):      {pac_sum[-1]:.4f} (grows as ln N)")
    print(f"  ln({N_max}):       {np.log(N_max):.4f}")
    
    print("\n  Both diverge, but PAC is MUCH slower.")
    print("  PAC doesn't eliminate need for regularization,")
    print("  but it might make the physics more tractable.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PAC CASIMIR EFFECT ANALYSIS (REFINED)")
    print("=" * 70)
    print(f"\nDate: {datetime.now().isoformat()}")
    print(f"\nPhysical constants:")
    print(f"  hbar = {HBAR:.6e} J*s")
    print(f"  c = {C:.0f} m/s")
    print(f"  Casimir coefficient pi^2/240 = {CASIMIR_COEFF:.6f}")
    
    # Run analysis
    d_values = [1e-7, 2e-7, 5e-7, 1e-6, 2e-6]  # 100nm to 2μm
    results = derive_casimir_from_pac(d_values)
    
    pac_dimensional_analysis()
    test_pac_casimir_coefficient()
    compare_to_experiment()
    
    # New refined tests
    zeta_results = test_zeta_fibonacci_connection()
    test_pac_natural_cutoff()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (REFINED)")
    print("=" * 70)
    
    print("""
CONCLUSIONS:

1. FIBONACCI IN CASIMIR COEFFICIENT
   - Casimir: E/A = -pi^2 hbar c / (720 d^3)
   - 720 = F_5 × F_12 = 5 × 144 (Fibonacci decomposition!)
   - This might connect to PAC mode counting on a binary tree
   
2. ZETA FUNCTION STRUCTURE  
   - Casimir uses zeta(-1) = -1/12 (regularized sum)
   - zeta(-3) = 1/120, which appears in 720 = 6 × 120
   - The zeta values at negative integers have rational structure
   
3. PAC AS REGULARIZATION
   - PAC k^-2 weighting DOES NOT eliminate Casimir divergence
   - It changes sum(n) -> sum(1/n), which still diverges
   - But divergence is logarithmic, not quadratic

4. TESTABLE PREDICTION
   - If 720 = F_5 × F_12 is meaningful, other QFT coefficients
     should have similar Fibonacci decompositions
   - Check: pi^4/90 (zeta(4)), pi^6/945 (zeta(6)), etc.

STATUS: PARTIAL CONNECTION
The 720 = F_5 × F_12 decomposition is intriguing but needs
deeper theoretical justification. PAC alone doesn't explain
why zeta regularization gives -1/12.
    """)
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"../results/casimir_analysis_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"\nCould not save results: {e}")
    
    return results

if __name__ == "__main__":
    main()
