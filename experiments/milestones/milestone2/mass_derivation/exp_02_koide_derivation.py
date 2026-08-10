#!/usr/bin/env python3
"""
Experiment 02: Koide Individual Mass Derivation

Part VII: Mass Ratio Derivation

Koide formula: Q = (me + mμ + mτ) / (√me + √mμ + √mτ)² = 2/3

This is ONE constraint on THREE masses. We need TWO more to solve uniquely.

Hypothesis: The additional constraints are also Fibonacci-based.

Approach:
1. Parameterize masses in terms of a base mass and ratios
2. Apply Koide constraint (Q = 2/3 = F₃/F₄)
3. Search for second Fibonacci constraint that predicts correct ratios
4. Validate against measured values
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple
from scipy.optimize import fsolve


# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
XI = 1 + np.pi / 55

# Fibonacci
def fib(n: int) -> int:
    if n <= 1:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

FIB = [fib(i) for i in range(25)]

# Measured lepton masses (MeV)
ME = 0.51099895  # electron
MMU = 105.6583755  # muon
MTAU = 1776.86  # tau

# Measured ratios
R_MU_E = MMU / ME  # 206.768
R_TAU_E = MTAU / ME  # 3477.23
R_TAU_MU = MTAU / MMU  # 16.817


def koide_Q(m1: float, m2: float, m3: float) -> float:
    """Compute Koide Q value."""
    numerator = m1 + m2 + m3
    denominator = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return numerator / denominator


def verify_koide():
    """Verify Koide formula with measured masses."""
    Q = koide_Q(ME, MMU, MTAU)
    Q_fib = 2/3
    error = abs(Q - Q_fib) / Q_fib * 100
    
    print("=" * 70)
    print("Koide Formula Verification")
    print("=" * 70)
    print(f"Q = (me + mμ + mτ) / (√me + √mμ + √mτ)²")
    print(f"Measured: Q = {Q:.8f}")
    print(f"Fibonacci: 2/3 = {Q_fib:.8f}")
    print(f"Error: {error:.6f}%")
    
    return Q


def test_second_constraint():
    """
    Search for a second Fibonacci-based constraint.
    
    If we parameterize: mμ = r₁ × me, mτ = r₂ × me
    
    Then Koide gives us ONE equation in r₁, r₂.
    We need a SECOND equation.
    
    Candidates:
    1. r₂/r₁ = F_i/F_j (ratio of ratios is Fibonacci)
    2. r₁ × r₂ = F_i × F_j (product is Fibonacci product)
    3. r₁^a × r₂^b = F_i (some power relation)
    4. sqrt(r₁) + sqrt(r₂) = F_i × φ^n (sum of roots)
    """
    
    print("\n" + "=" * 70)
    print("Testing Second Constraint Candidates")
    print("=" * 70)
    
    r1 = R_MU_E  # 206.768
    r2 = R_TAU_E  # 3477.23
    
    results = []
    
    # Test 1: ratio of ratios
    print("\n--- Test 1: r₂/r₁ = τ/μ ---")
    ratio = r2 / r1
    print(f"τ/μ = {ratio:.6f}")
    
    for i in range(2, 18):
        for j in range(2, 18):
            if FIB[j] == 0:
                continue
            fib_ratio = FIB[i] / FIB[j]
            error = abs(ratio - fib_ratio) / ratio * 100
            if error < 5:
                print(f"  F_{i}/F_{j} = {fib_ratio:.4f} ({error:.3f}% error)")
                results.append({
                    'type': 'ratio_of_ratios',
                    'formula': f'F_{i}/F_{j}',
                    'value': fib_ratio,
                    'error_pct': error
                })
    
    # Test 2: product r₁ × r₂
    print("\n--- Test 2: r₁ × r₂ ---")
    prod = r1 * r2
    print(f"μ/e × τ/e = {prod:.1f}")
    
    for i in range(5, 20):
        for j in range(i, 20):
            fib_prod = FIB[i] * FIB[j]
            error = abs(prod - fib_prod) / prod * 100
            if error < 5:
                print(f"  F_{i} × F_{j} = {fib_prod} ({error:.3f}% error)")
                results.append({
                    'type': 'product',
                    'formula': f'F_{i} × F_{j}',
                    'value': fib_prod,
                    'error_pct': error
                })
    
    # Test 3: sqrt sum
    print("\n--- Test 3: √r₁ + √r₂ ---")
    sqrt_sum = np.sqrt(r1) + np.sqrt(r2)
    print(f"√(μ/e) + √(τ/e) = {sqrt_sum:.4f}")
    
    for i in range(3, 15):
        for n in range(-3, 5):
            val = FIB[i] * (PHI ** n)
            error = abs(sqrt_sum - val) / sqrt_sum * 100
            if error < 3:
                print(f"  F_{i} × φ^{n} = {val:.4f} ({error:.3f}% error)")
                results.append({
                    'type': 'sqrt_sum',
                    'formula': f'F_{i} × φ^{n}',
                    'value': val,
                    'error_pct': error
                })
    
    # Test 4: geometric mean
    print("\n--- Test 4: √(r₁ × r₂) ---")
    geom = np.sqrt(r1 * r2)
    print(f"√(μ/e × τ/e) = {geom:.4f}")
    
    for i in range(5, 18):
        for n in range(-3, 5):
            val = FIB[i] * (PHI ** n)
            error = abs(geom - val) / geom * 100
            if error < 3:
                print(f"  F_{i} × φ^{n} = {val:.4f} ({error:.3f}% error)")
                results.append({
                    'type': 'geom_mean',
                    'formula': f'F_{i} × φ^{n}',
                    'value': val,
                    'error_pct': error
                })
    
    return results


def koide_parameterization():
    """
    Explore Koide-style parameterization.
    
    Koide's original insight: masses can be written as
    m_i = M × (1 + ε × cos(θ_i))²
    
    where θ_1, θ_2, θ_3 are evenly spaced by 2π/3.
    
    This automatically satisfies Q = 2/3!
    The question is: what determines ε and M?
    """
    
    print("\n" + "=" * 70)
    print("Koide Parameterization Analysis")
    print("=" * 70)
    
    # From Koide's form, solve for ε
    # Using geometric approach
    
    # The phase angles for leptons
    # Koide proposed: θ = 0, 2π/3, 4π/3 (equilateral in phase space)
    
    # Fit ε and M to measured masses
    def residuals(params):
        M, eps = params
        theta = np.array([0, 2*np.pi/3, 4*np.pi/3])
        m_pred = M * (1 + eps * np.cos(theta))**2
        m_meas = np.array([ME, MMU, MTAU])
        return [m_pred[1]/m_pred[0] - R_MU_E, m_pred[2]/m_pred[0] - R_TAU_E]
    
    # This is underdetermined - need to add phase offset
    # Actually Koide adds a phase δ: θ_i = 2πi/3 + δ
    
    def residuals_with_phase(params):
        M, eps, delta = params
        theta = np.array([delta, 2*np.pi/3 + delta, 4*np.pi/3 + delta])
        m_pred = M * (1 + eps * np.cos(theta))**2
        m_meas = np.array([ME, MMU, MTAU])
        return [
            m_pred[0] - ME,
            m_pred[1] - MMU,
            m_pred[2] - MTAU
        ]
    
    from scipy.optimize import fsolve
    
    # Initial guess
    M0 = 300  # MeV scale
    eps0 = 0.8
    delta0 = 0.2
    
    solution = fsolve(residuals_with_phase, [M0, eps0, delta0], full_output=True)
    M, eps, delta = solution[0]
    
    print(f"\nFitted parameters:")
    print(f"  M = {M:.4f} MeV (base mass scale)")
    print(f"  ε = {eps:.6f} (amplitude)")
    print(f"  δ = {delta:.6f} rad = {np.degrees(delta):.2f}° (phase offset)")
    
    # Check for Fibonacci structure in these parameters
    print(f"\nFibonacci analysis of parameters:")
    print(f"  ε ≈ {eps:.4f}")
    print(f"    1/φ = {PHI_INV:.4f} (error: {abs(eps - PHI_INV)/eps*100:.2f}%)")
    print(f"    √(2/3) = {np.sqrt(2/3):.4f} (error: {abs(eps - np.sqrt(2/3))/eps*100:.2f}%)")
    
    print(f"\n  δ/π = {delta/np.pi:.6f}")
    print(f"    2/9 = {2/9:.6f} (error: {abs(delta/np.pi - 2/9)/(delta/np.pi)*100:.2f}%)")
    
    return {'M': M, 'eps': eps, 'delta': delta}


def main():
    print("=" * 70)
    print("Experiment 02: Koide Individual Mass Derivation")
    print("=" * 70)
    
    # Step 1: Verify Koide
    Q = verify_koide()
    
    # Step 2: Test second constraints
    constraints = test_second_constraint()
    
    # Step 3: Analyze Koide parameterization
    params = koide_parameterization()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Koide Q = {Q:.8f} ≈ 2/3 = F₃/F₄")
    
    if constraints:
        best = min(constraints, key=lambda x: x['error_pct'])
        print(f"\n2. Best second constraint: {best['type']}")
        print(f"   Formula: {best['formula']} = {best['value']:.4f}")
        print(f"   Error: {best['error_pct']:.4f}%")
    
    print(f"\n3. Koide parameterization:")
    print(f"   ε ≈ {params['eps']:.4f} (compare 1/φ = {PHI_INV:.4f})")
    print(f"   δ/π ≈ {params['delta']/np.pi:.4f} (compare 2/9 = {2/9:.4f})")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_02_koide_derivation',
        'koide_Q': Q,
        'koide_Q_fibonacci': 2/3,
        'second_constraints': constraints,
        'koide_params': params,
        'measured_ratios': {
            'mu/e': R_MU_E,
            'tau/e': R_TAU_E,
            'tau/mu': R_TAU_MU
        }
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'exp_02_koide_derivation_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to results/exp_02_koide_derivation_{timestamp}.json")
    
    return output


if __name__ == '__main__':
    main()
