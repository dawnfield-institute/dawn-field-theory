"""
Script 13: Deriving Ξ_mean from First Principles

GOAL: Show that Ξ_mean ≈ 1.028 emerges from pure geometry, not just
GAIA simulation measurements.

APPROACH:
1. Ξ = ratio of Möbius to Circle spectral sums
2. Ξ_min ≈ 1.0015 (minimum asymmetry for information)
3. Ξ_PAC ≈ 1.0571 (maximum stable asymmetry)
4. Where does Ξ_mean ≈ 1.028 come from?

HYPOTHESIS:
Ξ_mean is the GEOMETRIC MEAN of Ξ_min and Ξ_PAC:
    Ξ_mean = √(Ξ_min × Ξ_PAC)

Or it's determined by φ and π in some fundamental way.

Let's test multiple derivation paths.
"""

import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime

# Known constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
XI_MIN = 1.0015              # Minimum asymmetry (from theory)
XI_PAC = 1.0571              # Maximum stable asymmetry (from PAC)
XI_MEAN_OBSERVED = 1.028     # From GAIA simulations

def test_geometric_mean():
    """Test if Ξ_mean = √(Ξ_min × Ξ_PAC)"""
    xi_geom = np.sqrt(XI_MIN * XI_PAC)
    error = abs(xi_geom - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    return {
        'formula': '√(Ξ_min × Ξ_PAC)',
        'value': xi_geom,
        'observed': XI_MEAN_OBSERVED,
        'error_percent': error
    }

def test_arithmetic_mean():
    """Test if Ξ_mean = (Ξ_min + Ξ_PAC)/2"""
    xi_arith = (XI_MIN + XI_PAC) / 2
    error = abs(xi_arith - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    return {
        'formula': '(Ξ_min + Ξ_PAC)/2',
        'value': xi_arith,
        'observed': XI_MEAN_OBSERVED,
        'error_percent': error
    }

def test_harmonic_mean():
    """Test if Ξ_mean = 2/(1/Ξ_min + 1/Ξ_PAC)"""
    xi_harm = 2 / (1/XI_MIN + 1/XI_PAC)
    error = abs(xi_harm - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    return {
        'formula': '2/(1/Ξ_min + 1/Ξ_PAC)',
        'value': xi_harm,
        'observed': XI_MEAN_OBSERVED,
        'error_percent': error
    }

def test_golden_ratio_relations():
    """Test various φ-based derivations"""
    results = {}
    
    # Test: Ξ_mean = 1 + 1/φ³
    xi_phi3 = 1 + 1/PHI**3
    results['1 + 1/φ³'] = {
        'value': xi_phi3,
        'error_percent': abs(xi_phi3 - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # Test: Ξ_mean = 1 + 1/φ⁴  
    xi_phi4 = 1 + 1/PHI**4
    results['1 + 1/φ⁴'] = {
        'value': xi_phi4,
        'error_percent': abs(xi_phi4 - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # Test: Ξ_mean = φ/φ² = 1/φ (no, this is < 1)
    
    # Test: Ξ_mean = 1 + (φ-1)/φ³
    xi_phi_combo = 1 + (PHI - 1)/PHI**3
    results['1 + (φ-1)/φ³'] = {
        'value': xi_phi_combo,
        'error_percent': abs(xi_phi_combo - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # Test: Ξ_mean = 1 + 1/(φ² × π)
    xi_phi_pi = 1 + 1/(PHI**2 * np.pi)
    results['1 + 1/(φ²π)'] = {
        'value': xi_phi_pi,
        'error_percent': abs(xi_phi_pi - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # Test: Ξ_mean = 1 + π/(φ⁵)
    xi_pi_phi5 = 1 + np.pi/PHI**5
    results['1 + π/φ⁵'] = {
        'value': xi_pi_phi5,
        'error_percent': abs(xi_pi_phi5 - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # Test: From Möbius holonomy - Ξ related to half-twist
    # A half-twist is π radians, full return is 4π
    # Ξ might encode the "twist efficiency"
    xi_twist = 1 + np.pi/(4*np.pi + PHI**4)  
    results['1 + π/(4π + φ⁴)'] = {
        'value': xi_twist,
        'error_percent': abs(xi_twist - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    return results

def test_spectral_derivation():
    """
    Derive Ξ from Möbius vs Circle spectral sums directly.
    
    Circle eigenvalues: λₙ = n²
    Möbius eigenvalues: λₙ = (n + 1/2)²
    
    Ξ(N) = Σ(n+1/2)² / Σn²  for n=1..N
    """
    results = {}
    
    for N in [3, 5, 7, 10, 13, 21, 34, 55, 89]:  # Fibonacci values
        circle_sum = sum(n**2 for n in range(1, N+1))
        mobius_sum = sum((n + 0.5)**2 for n in range(1, N+1))
        xi_N = mobius_sum / circle_sum
        
        results[N] = {
            'xi': xi_N,
            'error_from_mean': abs(xi_N - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
        }
    
    return results

def test_fibonacci_weighted_spectral():
    """
    What if we weight the spectral sum by Fibonacci?
    
    This would connect the spectral definition to the recursion structure.
    """
    def fib(n):
        if n <= 0: return 0
        elif n <= 2: return 1
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b
    
    results = {}
    
    # Weight each eigenvalue by its Fibonacci index
    for N in [7, 10, 13]:
        circle_weighted = sum(fib(n) * n**2 for n in range(1, N+1))
        mobius_weighted = sum(fib(n) * (n + 0.5)**2 for n in range(1, N+1))
        
        if circle_weighted > 0:
            xi_fib = mobius_weighted / circle_weighted
            results[f'N={N}'] = {
                'xi_fibonacci_weighted': xi_fib,
                'error_from_mean': abs(xi_fib - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
            }
    
    return results

def test_half_way_point():
    """
    What if Ξ_mean is exactly half way between 1 and Ξ_PAC on a φ-scale?
    
    The "center" of the asymmetry range.
    """
    # Linear half-way
    xi_linear_half = 1 + (XI_PAC - 1) / 2
    
    # Logarithmic half-way  
    xi_log_half = np.exp((np.log(1) + np.log(XI_PAC)) / 2)
    
    # φ-scaled half-way: at distance 1/φ from 1 toward XI_PAC
    xi_phi_scale = 1 + (XI_PAC - 1) / PHI
    
    # At 1/φ² of the way
    xi_phi2_scale = 1 + (XI_PAC - 1) / PHI**2
    
    return {
        'linear_half': {
            'value': xi_linear_half,
            'error': abs(xi_linear_half - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
        },
        'log_half': {
            'value': xi_log_half,
            'error': abs(xi_log_half - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
        },
        'phi_scale (1/φ)': {
            'value': xi_phi_scale,
            'error': abs(xi_phi_scale - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
        },
        'phi2_scale (1/φ²)': {
            'value': xi_phi2_scale,
            'error': abs(xi_phi2_scale - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
        }
    }

def test_mobius_holonomy_connection():
    """
    The Möbius strip requires 4π rotation for full holonomy return.
    
    The half-twist is π radians = 0.5 turns.
    
    If Ξ encodes "how much twist per recursion level":
        Ξ - 1 = fractional twist beyond identity
        
    At equilibrium, the system should be at some optimal twist fraction.
    """
    results = {}
    
    # If twist is distributed across F_7 = 13 levels:
    # Total twist = π (half turn)
    # Per-level twist = π/13
    xi_per_level = 1 + np.pi / 13
    results['1 + π/13 (per F₇ level)'] = {
        'value': xi_per_level,
        'error': abs(xi_per_level - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # If twist is π/F_10 (where F_10 = 55, the balance depth from PACSeries):
    xi_f10 = 1 + np.pi / 55
    results['1 + π/55 (F₁₀ balance)'] = {
        'value': xi_f10,
        'error': abs(xi_f10 - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # The PACSeries found r = 11/(8π) ≈ 0.4377
    # What's 1 + r/some_constant?
    r_mas = 11 / (8 * np.pi)
    xi_r_based = 1 + r_mas / 15.5  # Just testing
    results['1 + r_MAS/15.5'] = {
        'value': xi_r_based,
        'error': abs(xi_r_based - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # What about: Ξ_mean - 1 = (Ξ_PAC - 1) × (1 - 1/φ²)?
    # This would mean equilibrium is at 1/φ² = 38.2% of the way from 1 to Ξ_PAC
    xi_38 = 1 + (XI_PAC - 1) * (1 - 1/PHI**2)
    results['1 + (Ξ_PAC-1)×(1-1/φ²)'] = {
        'value': xi_38,
        'error': abs(xi_38 - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    # The golden angle: 2π/φ² ≈ 137.5°
    # What if Ξ_mean encodes this?
    golden_angle = 2 * np.pi / PHI**2
    xi_golden_angle = 1 + golden_angle / (4 * np.pi)  # Fraction of full holonomy
    results['1 + (2π/φ²)/(4π)'] = {
        'value': xi_golden_angle,
        'error': abs(xi_golden_angle - XI_MEAN_OBSERVED) / XI_MEAN_OBSERVED * 100
    }
    
    return results


def main():
    print("=" * 70)
    print("SCRIPT 13: DERIVING Ξ_mean FROM FIRST PRINCIPLES")
    print("=" * 70)
    
    print(f"\nTarget: Ξ_mean = {XI_MEAN_OBSERVED} (observed from GAIA)")
    print(f"Range: [{XI_MIN}, {XI_PAC}]")
    
    all_results = {}
    
    # Test 1: Mean types
    print("\n" + "=" * 60)
    print("TEST 1: Standard Mean Types")
    print("=" * 60)
    
    geom = test_geometric_mean()
    arith = test_arithmetic_mean()
    harm = test_harmonic_mean()
    
    print(f"\n  Geometric mean: {geom['value']:.6f} (error: {geom['error_percent']:.2f}%)")
    print(f"  Arithmetic mean: {arith['value']:.6f} (error: {arith['error_percent']:.2f}%)")
    print(f"  Harmonic mean: {harm['value']:.6f} (error: {harm['error_percent']:.2f}%)")
    
    all_results['means'] = {'geometric': geom, 'arithmetic': arith, 'harmonic': harm}
    
    # Test 2: Golden ratio relations
    print("\n" + "=" * 60)
    print("TEST 2: Golden Ratio Relations")
    print("=" * 60)
    
    phi_results = test_golden_ratio_relations()
    
    for formula, data in phi_results.items():
        marker = " ← CLOSE!" if data['error_percent'] < 5 else ""
        print(f"  {formula:20s} = {data['value']:.6f} (error: {data['error_percent']:.2f}%){marker}")
    
    all_results['golden_ratio'] = phi_results
    
    # Test 3: Spectral derivation
    print("\n" + "=" * 60)
    print("TEST 3: Direct Spectral Ξ(N)")
    print("=" * 60)
    
    spectral = test_spectral_derivation()
    
    print("\n  N   | Ξ(N)     | Error from 1.028")
    print("  " + "-" * 35)
    for N, data in spectral.items():
        marker = " ← CLOSE!" if data['error_from_mean'] < 5 else ""
        print(f"  {N:3d} | {data['xi']:.6f} | {data['error_from_mean']:.2f}%{marker}")
    
    all_results['spectral'] = spectral
    
    # Test 4: Fibonacci-weighted spectral
    print("\n" + "=" * 60)
    print("TEST 4: Fibonacci-Weighted Spectral")
    print("=" * 60)
    
    fib_spectral = test_fibonacci_weighted_spectral()
    for key, data in fib_spectral.items():
        print(f"  {key}: Ξ = {data['xi_fibonacci_weighted']:.6f} (error: {data['error_from_mean']:.2f}%)")
    
    all_results['fibonacci_weighted'] = fib_spectral
    
    # Test 5: Half-way points
    print("\n" + "=" * 60)
    print("TEST 5: Midpoint on Different Scales")
    print("=" * 60)
    
    halfway = test_half_way_point()
    for name, data in halfway.items():
        marker = " ← CLOSE!" if data['error'] < 5 else ""
        print(f"  {name:20s}: {data['value']:.6f} (error: {data['error']:.2f}%){marker}")
    
    all_results['halfway'] = halfway
    
    # Test 6: Möbius holonomy
    print("\n" + "=" * 60)
    print("TEST 6: Möbius Holonomy Connection")
    print("=" * 60)
    
    holonomy = test_mobius_holonomy_connection()
    for formula, data in holonomy.items():
        marker = " ← CLOSE!" if data['error'] < 5 else ""
        print(f"  {formula:30s}: {data['value']:.6f} (error: {data['error']:.2f}%){marker}")
    
    all_results['holonomy'] = holonomy
    
    # Find best matches
    print("\n" + "=" * 60)
    print("BEST MATCHES (error < 5%)")
    print("=" * 60)
    
    best_matches = []
    
    # Collect all results with errors
    for formula, data in phi_results.items():
        if data['error_percent'] < 5:
            best_matches.append((formula, data['value'], data['error_percent']))
    
    for name, data in halfway.items():
        if data['error'] < 5:
            best_matches.append((name, data['value'], data['error']))
            
    for formula, data in holonomy.items():
        if data['error'] < 5:
            best_matches.append((formula, data['value'], data['error']))
    
    best_matches.sort(key=lambda x: x[2])
    
    print("\n  Formula                          | Value    | Error")
    print("  " + "-" * 55)
    for formula, value, error in best_matches[:10]:
        print(f"  {formula:35s} | {value:.6f} | {error:.3f}%")
    
    # Synthesis
    print("\n" + "=" * 60)
    print("SYNTHESIS")
    print("=" * 60)
    
    if best_matches:
        best = best_matches[0]
        print(f"""
    Best derivation: {best[0]}
    Value: {best[1]:.6f}
    Error: {best[2]:.3f}%
    
    This suggests Ξ_mean is geometrically determined by:
    {best[0]}
    """)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'target': XI_MEAN_OBSERVED,
        'best_matches': [{'formula': f, 'value': v, 'error': e} for f, v, e in best_matches],
        'all_results': {k: str(v) for k, v in all_results.items()}
    }
    
    output_path = f"../results/13_xi_mean_derivation_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
