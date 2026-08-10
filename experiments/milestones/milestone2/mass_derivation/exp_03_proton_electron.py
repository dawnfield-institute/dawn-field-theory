#!/usr/bin/env python3
"""
Experiment 03: Proton-Electron Mass Ratio

Part VII: Mass Ratio Derivation

mp/me = 1836.15267343...

This is the most fundamental baryon/lepton ratio.
Can we derive it from Fibonacci structure?

Key observations:
- 1836 is between F₁₇ = 1597 and F₁₈ = 2584
- 1836/1597 = 1.150 ≈ Ξ^2.6
- 1836 ≈ 6 × 306 = 6 × (F₁₃ + F₁₁) = 6 × (233 + 89) = 6 × 322... no
- 1836 ≈ 3 × 612 = 3 × (F₁₅ + 2) = 3 × 612... no

Need to think differently. Maybe it's not pure Fibonacci,
but involves the Koide structure or gauge sector?
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
XI = 1 + np.pi / 55
PI = np.pi

# Fibonacci
def fib(n: int) -> int:
    if n <= 1:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

FIB = [fib(i) for i in range(30)]

# Measured ratio
MP_ME = 1836.15267343


def analyze_nearby_fibonacci():
    """Find Fibonacci numbers near the ratio."""
    
    print("=" * 70)
    print("Analysis: Nearby Fibonacci Numbers")
    print("=" * 70)
    
    print(f"\nTarget: mp/me = {MP_ME:.4f}")
    print(f"\nNearby Fibonacci numbers:")
    
    for i in range(14, 22):
        ratio = MP_ME / FIB[i]
        print(f"  F_{i} = {FIB[i]:6d}  →  mp/me / F_{i} = {ratio:.6f}")
    
    # Key observation
    print(f"\nKey observation:")
    print(f"  mp/me / F₁₇ = {MP_ME / FIB[17]:.6f}")
    print(f"  Compare to Ξ = {XI:.6f}")
    print(f"  Compare to φ = {PHI:.6f}")


def test_pure_formulas():
    """Test various pure Fibonacci formulas."""
    
    print("\n" + "=" * 70)
    print("Testing Pure Formulas")
    print("=" * 70)
    
    candidates = []
    
    # F_i × φ^n
    print("\n--- F_i × φ^n ---")
    for i in range(12, 20):
        for n in range(-5, 10):
            val = FIB[i] * (PHI ** n)
            error = abs(MP_ME - val) / MP_ME * 100
            if error < 1:
                print(f"  F_{i} × φ^{n} = {val:.4f} ({error:.4f}%)")
                candidates.append({
                    'formula': f'F_{i} × φ^{n}',
                    'value': val,
                    'error_pct': error
                })
    
    # F_i × Ξ^n
    print("\n--- F_i × Ξ^n ---")
    for i in range(15, 20):
        for n in range(1, 8):
            val = FIB[i] * (XI ** n)
            error = abs(MP_ME - val) / MP_ME * 100
            if error < 1:
                print(f"  F_{i} × Ξ^{n} = {val:.4f} ({error:.4f}%)")
                candidates.append({
                    'formula': f'F_{i} × Ξ^{n}',
                    'value': val,
                    'error_pct': error
                })
    
    # F_i × F_j / F_k
    print("\n--- F_i × F_j / F_k ---")
    for i in range(10, 18):
        for j in range(5, 15):
            for k in range(3, 12):
                if FIB[k] == 0:
                    continue
                val = FIB[i] * FIB[j] / FIB[k]
                error = abs(MP_ME - val) / MP_ME * 100
                if error < 0.5:
                    print(f"  F_{i} × F_{j} / F_{k} = {val:.4f} ({error:.4f}%)")
                    candidates.append({
                        'formula': f'F_{i} × F_{j} / F_{k}',
                        'value': val,
                        'error_pct': error
                    })
    
    # F_i² / F_j
    print("\n--- F_i² / F_j ---")
    for i in range(10, 20):
        for j in range(3, 15):
            if FIB[j] == 0:
                continue
            val = (FIB[i] ** 2) / FIB[j]
            error = abs(MP_ME - val) / MP_ME * 100
            if error < 0.5:
                print(f"  F_{i}² / F_{j} = {val:.4f} ({error:.4f}%)")
                candidates.append({
                    'formula': f'F_{i}² / F_{j}',
                    'value': val,
                    'error_pct': error
                })
    
    return candidates


def test_alpha_connection():
    """
    Test if mp/me connects to fine structure constant α.
    
    Some historical attempts:
    - mp/me ≈ α⁻² × something
    - 1/α ≈ 137.036, α⁻² ≈ 18778
    """
    
    print("\n" + "=" * 70)
    print("Testing α Connection")
    print("=" * 70)
    
    # Fine structure constant
    ALPHA = 1 / 137.035999084
    ALPHA_INV = 1 / ALPHA
    
    print(f"\n1/α = {ALPHA_INV:.4f}")
    print(f"mp/me = {MP_ME:.4f}")
    print(f"\nRatios:")
    print(f"  (mp/me) / (1/α) = {MP_ME / ALPHA_INV:.6f}")
    print(f"  (mp/me) × α = {MP_ME * ALPHA:.6f}")
    print(f"  (mp/me)² / (1/α) = {MP_ME**2 / ALPHA_INV:.4f}")
    
    # Check if ratio is Fibonacci-like
    ratio = MP_ME / ALPHA_INV
    print(f"\nIs {ratio:.4f} Fibonacci-structured?")
    print(f"  F₇ = 13, 13.4/13 = {ratio/13:.4f}")
    print(f"  F₆ = 8, 13.4/8 = {ratio/8:.4f}")
    
    # Another approach: mp/me = (something) / α
    target = MP_ME * ALPHA
    print(f"\nmp/me × α = {target:.6f}")
    print(f"  F₇ = 13 → {13:.4f}")
    print(f"  This is very close to F₇!")


def test_depth_structure():
    """
    Test if mp/me has depth structure like gravity (F₁₈₃).
    
    Hypothesis: different particles "live" at different Fibonacci depths.
    """
    
    print("\n" + "=" * 70)
    print("Testing Depth Structure")
    print("=" * 70)
    
    print(f"\nIf electron is at depth d_e and proton at depth d_p:")
    print(f"Then mp/me ∝ F(d_p) / F(d_e)")
    
    print(f"\nFor mp/me = {MP_ME:.4f}:")
    
    # Find which depth pairs work
    for de in range(1, 10):
        for dp in range(de + 5, de + 20):
            ratio = FIB[dp] / FIB[de] if FIB[de] > 0 else 0
            if ratio == 0:
                continue
            error = abs(MP_ME - ratio) / MP_ME * 100
            if error < 5:
                print(f"  d_e = {de}, d_p = {dp}: F_{dp}/F_{de} = {ratio:.4f} ({error:.2f}%)")


def test_koide_extended():
    """
    Test if proton/electron ratio relates to Koide structure.
    
    Koide Q = 2/3 for leptons.
    Does a similar structure apply to baryon/lepton?
    """
    
    print("\n" + "=" * 70)
    print("Testing Koide Extension")
    print("=" * 70)
    
    # From exp_02, we know mμ/me ≈ 206.768
    # mp/me ≈ 1836.15
    
    # Ratio of ratios
    ratio = MP_ME / 206.768
    print(f"\n(mp/me) / (mμ/me) = mp/mμ = {ratio:.4f}")
    print(f"  F₆/F₃ = 8/2 = 4... no")
    print(f"  F₆ + F₄ = 8 + 3 = 11... close to F₇/F₄ = 13/3 = 4.33")
    print(f"  3³ = 27, ratio/27 = {ratio/27:.4f}")
    
    # Check: is mp related to mτ?
    MTAU_ME = 3477.23
    print(f"\nmτ/me = {MTAU_ME:.4f}")
    print(f"mp/mτ = {MP_ME / MTAU_ME:.4f}")
    print(f"Compare: 1/φ² = {1/PHI**2:.4f}")


def main():
    print("=" * 70)
    print("Experiment 03: Proton-Electron Mass Ratio")
    print("=" * 70)
    
    print(f"\nTarget: mp/me = {MP_ME:.8f}")
    
    # Analysis
    analyze_nearby_fibonacci()
    candidates = test_pure_formulas()
    test_alpha_connection()
    test_depth_structure()
    test_koide_extended()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if candidates:
        best = min(candidates, key=lambda x: x['error_pct'])
        print(f"\nBest formula found:")
        print(f"  {best['formula']} = {best['value']:.6f}")
        print(f"  Error: {best['error_pct']:.4f}%")
        print(f"\nTop 5 candidates:")
        for c in sorted(candidates, key=lambda x: x['error_pct'])[:5]:
            print(f"  {c['formula']:25s} = {c['value']:.4f} ({c['error_pct']:.4f}%)")
    else:
        print("\nNo pure Fibonacci formula found within 1%")
        print("This ratio may require mixed structure or different approach")
    
    # Key insight
    print(f"\n--- Key Insight ---")
    print(f"mp/me × α = {MP_ME * (1/137.036):.4f} ≈ 13.4 ≈ F₇ + correction")
    print(f"This suggests: mp/me = F₇ / α × (1 + correction)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_03_proton_electron',
        'target': MP_ME,
        'candidates': candidates,
        'alpha_connection': {
            'mp_me_times_alpha': MP_ME / 137.036,
            'nearest_fib': 13,
            'interpretation': 'mp/me ≈ F_7 / α'
        }
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'exp_03_proton_electron_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to results/exp_03_proton_electron_{timestamp}.json")
    
    return output


if __name__ == '__main__':
    main()
