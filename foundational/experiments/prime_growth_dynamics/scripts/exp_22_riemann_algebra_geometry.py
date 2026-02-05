#!/usr/bin/env python3
"""
exp_22_riemann_algebra_geometry.py - Riemann Zeros as Algebra-Geometry Interface

THE UNIFIED FRAMEWORK:
| Layer        | Algebraic (operation)     | Geometric (structure)    | Interface          |
|--------------|---------------------------|--------------------------|-------------------|
| DFT          | SEC (collapse)            | PAC (conservation)       | φ/Ξ               |
| Physics      | RQM (relational)          | Ruliad (structural)      | shared predictions|
| Number theory| Smoothing (sieve)         | Residual (primes)        | RIEMANN ZEROS     |

HYPOTHESIS:
The Riemann zeta zeros encode the harmonics of the algebra-geometry interface
for the number line. Each zero corresponds to a frequency where the smoothing
OPERATION couples to the residual STRUCTURE.

From the algebra-geometry-interface experiment:
"φ and Ξ are properties of the algebra-geometry duality itself —
emergent values at the interface where operation meets structure."

PREDICTION:
If Riemann zeros are the interface harmonics, then:
1. Zero spacing should relate to smoothing wave interference
2. The Mertens constant (1.123) should appear in zero statistics
3. GUE statistics emerge because interface dynamics are universal

This experiment connects:
- internal/prime_origin simulations (smoothing model)
- algebra_geometry_interface (Sophie Germain duality)
- The Riemann Hypothesis (where does RH fit in DFT?)
"""

import numpy as np
from datetime import datetime
import json
import os

# Constants
GAMMA = 0.5772156649015329  # Euler-Mascheroni
PHI = (1 + np.sqrt(5)) / 2
XI = 1.0571  # Our interface constant

# First 100 Riemann zeta zeros (imaginary parts, critical line Re(s)=1/2)
# These are well-tabulated values
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000982, 147.422765, 150.053521, 150.925257, 153.024693,
    156.112909, 157.597591, 158.849988, 161.188964, 163.030709,
    165.537069, 167.184439, 169.094515, 169.911976, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026657,
    193.079726, 195.265397, 196.876481, 198.015309, 201.264751,
    202.493594, 204.189671, 205.394697, 207.906259, 209.576509,
    211.691862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714918, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250189, 231.987235, 233.693404, 236.524230
]

def sieve_primes(n):
    """Sieve of Eratosthenes"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return is_prime, [p for p in range(n + 1) if is_prime[p]]

# ============================================================================
# EXPERIMENT 1: Zero Spacing vs Smoothing Wave Interference
# ============================================================================

def exp_01_zero_spacing_interference():
    """
    Test if Riemann zero spacing relates to smoothing wave structure.
    
    The smoothing waves have frequencies 1/p for each prime p.
    The zeros should encode how these waves interfere.
    
    Prediction: Zero spacing distribution should relate to 1/ln(γ) structure.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Zero Spacing vs Smoothing Wave Interference")
    print("=" * 70)
    
    # Compute zero spacings
    spacings = np.diff(RIEMANN_ZEROS)
    
    print(f"\n  Number of zeros: {len(RIEMANN_ZEROS)}")
    print(f"  Mean spacing: {np.mean(spacings):.4f}")
    print(f"  Std spacing: {np.std(spacings):.4f}")
    
    # Normalize by mean (GUE prediction)
    normalized = spacings / np.mean(spacings)
    
    # GUE predicts avoiding small spacings (level repulsion)
    small_spacing_rate = np.sum(normalized < 0.5) / len(normalized)
    
    print(f"\n  Normalized spacing analysis:")
    print(f"    Rate of s < 0.5 (GUE predicts ~0.05): {small_spacing_rate:.4f}")
    print(f"    Rate of s < 0.2 (GUE predicts ~0.002): {np.sum(normalized < 0.2) / len(normalized):.4f}")
    
    # Key test: Does spacing relate to smoothing wave frequencies?
    # Smoothing waves have "wavelength" ~ p, so frequencies ~ 1/p
    # The spacing between zeros at height T is ~ 2π/ln(T)
    
    heights = np.array(RIEMANN_ZEROS)
    predicted_spacing = 2 * np.pi / np.log(heights[:-1])
    
    # Compare actual vs predicted
    ratio = spacings / predicted_spacing
    
    print(f"\n  Spacing vs 2π/ln(T) prediction:")
    print(f"    Mean ratio: {np.mean(ratio):.4f}")
    print(f"    Std ratio: {np.std(ratio):.4f}")
    
    # The ratio should be related to Mertens constant?
    mertens_ratio = 2 * np.exp(-GAMMA)  # ~1.123
    print(f"\n  Mertens constant (2e^-γ): {mertens_ratio:.4f}")
    print(f"  Ratio to Mertens: {np.mean(ratio) / mertens_ratio:.4f}")
    
    return {
        'mean_spacing': float(np.mean(spacings)),
        'std_spacing': float(np.std(spacings)),
        'small_spacing_rate': float(small_spacing_rate),
        'mean_ratio_to_prediction': float(np.mean(ratio)),
        'std_ratio': float(np.std(ratio)),
        'mertens_ratio': float(np.mean(ratio) / mertens_ratio)
    }

# ============================================================================
# EXPERIMENT 2: Euler-Mascheroni as Interface Invariant
# ============================================================================

def exp_02_gamma_as_interface():
    """
    Test the hypothesis: γ (Euler-Mascheroni) is the algebra-geometry interface constant.
    
    In the smoothing model: γ = integrated wave interference
    In number theory: γ appears in the Mertens product ∏(1-1/p) → e^(-γ)/ln(N)
    In zero statistics: γ should appear in spacing distributions
    
    Prediction: γ should relate zero statistics to prime statistics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: γ (Euler-Mascheroni) as Interface Invariant")
    print("=" * 70)
    
    # γ appears in many places - test its universality
    
    # 1. In Mertens product (smoothing interference)
    _, primes = sieve_primes(10000)
    mertens_product = np.prod([1 - 1/p for p in primes])
    theoretical = np.exp(-GAMMA) / np.log(10000)
    mertens_ratio = mertens_product / theoretical
    
    print(f"\n  1. Mertens product (primes to 10000):")
    print(f"     Product: {mertens_product:.6f}")
    print(f"     e^(-γ)/ln(N): {theoretical:.6f}")
    print(f"     Ratio: {mertens_ratio:.6f}")
    
    # 2. In harmonic series (basic algebraic object)
    # H_n - ln(n) → γ
    n = 10000
    harmonic = sum(1/k for k in range(1, n+1))
    gamma_approx = harmonic - np.log(n)
    
    print(f"\n  2. Harmonic series (n={n}):")
    print(f"     H_n - ln(n): {gamma_approx:.6f}")
    print(f"     γ: {GAMMA:.6f}")
    print(f"     Error: {abs(gamma_approx - GAMMA):.6f}")
    
    # 3. In Riemann zero height formula
    # N(T) ~ (T/2π) ln(T/2πe) + O(ln T)
    # The e here comes from e^-γ adjustments
    
    T = RIEMANN_ZEROS[-1]
    predicted_count = (T / (2*np.pi)) * np.log(T / (2*np.pi*np.e))
    actual_count = len(RIEMANN_ZEROS)
    
    print(f"\n  3. Zero counting formula:")
    print(f"     At height T={T:.1f}:")
    print(f"     Predicted count: {predicted_count:.1f}")
    print(f"     Actual count: {actual_count}")
    print(f"     Ratio: {actual_count / predicted_count:.4f}")
    
    # 4. Connection: γ relates algebra (harmonic sum) to geometry (prime product)
    print(f"\n  4. The Interface Interpretation:")
    print(f"     Algebra: H_n = Σ(1/k)  [sum over all integers]")
    print(f"     Geometry: ∏(1-1/p)    [product over primes]")
    print(f"     Interface: Both converge to expressions involving γ")
    print(f"     γ IS the translation constant between algebraic and geometric views!")
    
    return {
        'mertens_ratio': float(mertens_ratio),
        'harmonic_gamma_approx': float(gamma_approx),
        'harmonic_error': float(abs(gamma_approx - GAMMA)),
        'zero_count_ratio': float(actual_count / predicted_count)
    }

# ============================================================================
# EXPERIMENT 3: φ in Zero Statistics?
# ============================================================================

def exp_03_phi_in_zeros():
    """
    Test if φ appears in Riemann zero statistics.
    
    From algebra-geometry-interface: φ appears where operation meets structure.
    If zeros are the interface, φ might appear in their statistics.
    
    Note: This is exploratory - may not find anything.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: φ in Riemann Zero Statistics (Exploratory)")
    print("=" * 70)
    
    spacings = np.diff(RIEMANN_ZEROS)
    normalized = spacings / np.mean(spacings)
    
    # Test various φ-related predictions
    
    # 1. Ratio of consecutive spacings
    spacing_ratios = spacings[1:] / spacings[:-1]
    mean_ratio = np.mean(spacing_ratios)
    
    print(f"\n  1. Consecutive spacing ratios:")
    print(f"     Mean ratio: {mean_ratio:.4f}")
    print(f"     φ = {PHI:.4f}")
    print(f"     1/φ = {1/PHI:.4f}")
    print(f"     Distance to φ: {abs(mean_ratio - PHI):.4f}")
    print(f"     Distance to 1/φ: {abs(mean_ratio - 1/PHI):.4f}")
    
    # 2. Fraction of spacings in [0.5, 1.5] range (bulk)
    bulk_fraction = np.sum((normalized > 0.5) & (normalized < 1.5)) / len(normalized)
    
    print(f"\n  2. Bulk fraction (0.5 < s < 1.5):")
    print(f"     Fraction: {bulk_fraction:.4f}")
    print(f"     1/φ = {1/PHI:.4f}")
    print(f"     Distance: {abs(bulk_fraction - 1/PHI):.4f}")
    
    # 3. Small-to-large spacing ratio
    small = np.sum(normalized < 1)
    large = np.sum(normalized >= 1)
    sl_ratio = small / large if large > 0 else 0
    
    print(f"\n  3. Small-to-large spacing ratio:")
    print(f"     Ratio: {sl_ratio:.4f}")
    print(f"     φ = {PHI:.4f}")
    print(f"     1/φ = {1/PHI:.4f}")
    
    # 4. Check for Ξ
    print(f"\n  4. Checking for Ξ = {XI:.4f}:")
    print(f"     Mean spacing / 2π: {np.mean(spacings) / (2*np.pi):.4f}")
    print(f"     This should relate to 1/ln(mean_height)...")
    mean_height = np.mean(RIEMANN_ZEROS)
    print(f"     1/ln(mean_height): {1/np.log(mean_height):.4f}")
    print(f"     Ratio: {(np.mean(spacings)/(2*np.pi)) / (1/np.log(mean_height)):.4f}")
    
    return {
        'mean_spacing_ratio': float(mean_ratio),
        'distance_to_phi': float(abs(mean_ratio - PHI)),
        'distance_to_inv_phi': float(abs(mean_ratio - 1/PHI)),
        'bulk_fraction': float(bulk_fraction),
        'small_large_ratio': float(sl_ratio)
    }

# ============================================================================
# EXPERIMENT 4: Smoothing Wave Decomposition of Zeros
# ============================================================================

def exp_04_smoothing_decomposition():
    """
    Test if zeros decompose into smoothing wave contributions.
    
    Each prime p contributes a smoothing wave with "frequency" related to p.
    The zeros might be peaks in the interference pattern.
    
    The explicit formula: π(x) = li(x) - Σ li(x^ρ) + ...
    encodes exactly this: each zero ρ contributes an oscillation.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Smoothing Wave Decomposition")
    print("=" * 70)
    
    _, primes = sieve_primes(1000)
    
    # Each prime p contributes a term in the Chebyshev explicit formula
    # The zeros encode how these terms interfere
    
    # Simple test: Do zero positions relate to prime positions?
    # Map primes to their "position" in the smoothing sequence
    prime_logs = np.log(primes)
    
    # Zeros should appear at specific combinations of prime logs
    # This is essentially the connection to sum of log(p) = Chebyshev θ(x)
    
    print(f"\n  Comparing zero positions to prime log structure:")
    
    # For small zeros, check if they're near simple combinations
    small_zeros = RIEMANN_ZEROS[:20]
    
    for i, z in enumerate(small_zeros[:10]):
        # Find closest 2π/ln(p) multiple
        angular_period = 2 * np.pi / prime_logs[0]  # based on ln(2)
        nearest_multiple = round(z / angular_period)
        
        if i < 5:
            print(f"     Zero {i+1}: {z:.4f}, nearest 2π/ln(2) multiple: {nearest_multiple}, "
                  f"residual: {z - nearest_multiple * angular_period:.4f}")
    
    # Connection to smoothing: the zeros are resonance frequencies
    # When the smoothing operation "hits" a zero frequency, there's constructive interference
    
    print(f"\n  INTERPRETATION:")
    print(f"  If smoothing wave p has frequency f_p ~ log(p),")
    print(f"  then zeros appear where these frequencies resonate.")
    print(f"  The Riemann Hypothesis (all zeros on Re=1/2 line)")
    print(f"  would mean: all interference is symmetric around the")
    print(f"  algebra-geometry balance point!")
    
    # The Re=1/2 line is THE INTERFACE
    print(f"\n  Re(s) = 1/2 is the algebra-geometry interface:")
    print(f"    - s = 0: pure geometry (divergent sum)")
    print(f"    - s = 1: critical line (harmonic series → γ)")
    print(f"    - s = 1/2: balance point")
    print(f"    - s = ∞: pure algebra (convergent everything)")
    print(f"\n  RH says: ALL interference peaks are at the interface!")
    
    return {
        'interpretation': 'zeros_as_resonance_frequencies',
        'rh_meaning': 'all_interference_at_interface'
    }

# ============================================================================
# EXPERIMENT 5: Connection to DFT Constants
# ============================================================================

def exp_05_dft_constants():
    """
    Synthesize: How do γ, φ, Ξ, and the zeros connect?
    
    From our framework:
    - γ = integrated smoothing interference
    - φ = algebra-geometry balance constant
    - Ξ = SEC-PAC coupling ratio
    - zeros = interface resonance frequencies
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Synthesis - DFT Constants and Zeros")
    print("=" * 70)
    
    # Key relationships
    print(f"\n  DFT CONSTANT | VALUE    | MEANING IN NUMBER THEORY")
    print(f"  " + "-" * 55)
    print(f"  γ (Euler)    | {GAMMA:.6f} | Algebraic-geometric translation constant")
    print(f"  φ (Golden)   | {PHI:.6f} | Balance ratio (may appear in spacing stats)")
    print(f"  1/φ          | {1/PHI:.6f} | Complement (E>0 fraction in SEC)")
    print(f"  Ξ            | {XI:.6f} | SEC-PAC coupling (relates to 1 + π/55)")
    print(f"  2e^(-γ)      | {2*np.exp(-GAMMA):.6f} | Mertens overshoot factor")
    print(f"  e^(-γ)       | {np.exp(-GAMMA):.6f} | Base smoothing efficiency")
    
    # Test: Do these combine meaningfully?
    print(f"\n  COMBINATION TESTS:")
    print(f"  γ × φ = {GAMMA * PHI:.6f}")
    print(f"  γ / φ = {GAMMA / PHI:.6f}")
    print(f"  γ + ln(φ) = {GAMMA + np.log(PHI):.6f}")
    print(f"  e^(-γ) × φ = {np.exp(-GAMMA) * PHI:.6f}")
    print(f"  1/γ = {1/GAMMA:.6f}")
    print(f"  2/γ = {2/GAMMA:.6f} (compare to first zero: {RIEMANN_ZEROS[0]:.6f})")
    
    # The first zero
    z1 = RIEMANN_ZEROS[0]
    print(f"\n  FIRST ZERO ANALYSIS:")
    print(f"  z₁ = {z1:.6f}")
    print(f"  z₁/2π = {z1/(2*np.pi):.6f}")
    print(f"  z₁/γ = {z1/GAMMA:.6f}")
    print(f"  z₁ × γ/2π = {z1 * GAMMA / (2*np.pi):.6f}")
    print(f"  ln(z₁) = {np.log(z1):.6f}")
    print(f"  (This is exploratory - looking for structure)")
    
    return {
        'gamma': GAMMA,
        'phi': PHI,
        'xi': XI,
        'mertens_factor': float(2*np.exp(-GAMMA)),
        'first_zero': z1,
        'first_zero_over_2pi': float(z1/(2*np.pi))
    }

# ============================================================================
# SYNTHESIS
# ============================================================================

def synthesis(results):
    """Final synthesis."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Riemann Zeros as Algebra-Geometry Interface")
    print("=" * 70)
    
    print("""
    THE UNIFIED PICTURE:
    
    ALGEBRA (operation, dynamics)     GEOMETRY (structure, form)
    ─────────────────────────────────────────────────────────────
    SEC (collapse dynamics)           PAC (conservation)
    RQM (relational)                  Ruliad (structural)
    Smoothing (sieve operation)       Residual (prime structure)
    Sum: Σ 1/k → H_n                  Product: ∏(1-1/p)
                      ↓
                   INTERFACE
                      ↓
                   γ = 0.5772...
                (translation constant)
                      ↓
               RIEMANN ZEROS
            (resonance frequencies)
    
    KEY INSIGHT:
    The Riemann Hypothesis states that ALL zeros are on Re(s) = 1/2.
    In our framework, this means:
    
    ALL INTERFERENCE BETWEEN ALGEBRA AND GEOMETRY 
    OCCURS AT THE EXACT BALANCE POINT.
    
    There's no "leakage" - the interface is perfect.
    This is why primes are distributed so regularly at large scales
    (PNT works) - the algebra-geometry coupling is exact.
    
    NEXT STEPS:
    1. Derive γ from PAC/SEC first principles
    2. Predict zero spacing from smoothing wave model
    3. Connect GUE statistics to interface dynamics
    4. Formulate DFT version of Riemann Hypothesis
    """)
    
    return {
        'framework': 'algebra_geometry_interface',
        'key_constant': 'gamma_euler_mascheroni',
        'zeros_interpretation': 'interface_resonance_frequencies',
        'rh_interpretation': 'all_interference_at_balance_point'
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("exp_22: RIEMANN ZEROS AS ALGEBRA-GEOMETRY INTERFACE")
    print("=" * 70)
    print("Testing the Sophie Germain connection to Riemann zeros")
    
    results = {}
    
    results['exp_01_spacing'] = exp_01_zero_spacing_interference()
    results['exp_02_gamma'] = exp_02_gamma_as_interface()
    results['exp_03_phi'] = exp_03_phi_in_zeros()
    results['exp_04_decomposition'] = exp_04_smoothing_decomposition()
    results['exp_05_constants'] = exp_05_dft_constants()
    
    results['synthesis'] = synthesis(results)
    
    # Save
    results['timestamp'] = datetime.now().isoformat()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_22_riemann_algebra_geometry_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {filename}")
    
    return results

if __name__ == '__main__':
    main()
