#!/usr/bin/env python3
"""
Experiment 05: Tighten Mass Formulas

Part VII: Mass Ratio Derivation

The base Fibonacci formulas work but need tightening:
- μ/e = F_3 × F_6 × F_7 = 208 vs 206.768 (0.6% error)
- τ/e = F_4 × F_7 × F_11 = 3471 vs 3477.23 (0.18% error)

Try systematic corrections:
1. Ξ = 1 + π/55 ≈ 1.0571 (from Navier-Stokes)
2. φ powers (0.618^n, 1.618^n)
3. π-based corrections
4. F_i/F_j ratios as multipliers
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


# Constants
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # 0.618...
XI = 1 + np.pi / 55  # ≈ 1.0571
PI = np.pi
E = np.e

# Fibonacci
def fib(n: int) -> int:
    if n <= 1:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

FIB = [fib(i) for i in range(25)]

# Measured values (CODATA 2018)
MEASURED = {
    'mu/e': 206.7682830,
    'tau/e': 3477.23,
    'tau/mu': 16.8170,
    'p/e': 1836.15267343,
}

# Base formulas from exp_01
BASE = {
    'mu/e': FIB[3] * FIB[6] * FIB[7],  # 2 × 8 × 13 = 208
    'tau/e': FIB[4] * FIB[7] * FIB[11],  # 3 × 13 × 89 = 3471
    'tau/mu': FIB[9] / FIB[3],  # 34/2 = 17
    'p/e': FIB[4] * FIB[11] * (PHI ** 4),  # 3 × 89 × 6.854
}


def test_corrections():
    """Test systematic corrections to base formulas."""
    print("=" * 70)
    print("Testing Ξ, φ, and π corrections")
    print("=" * 70)
    
    # Correction factors to try
    corrections = {
        'none': 1.0,
        'Ξ': XI,
        '1/Ξ': 1/XI,
        'φ^-1': INV_PHI,
        'φ^-2': INV_PHI ** 2,
        'φ^-3': INV_PHI ** 3,
        '1 - 1/φ²': 1 - 1/(PHI**2),
        'F_3/F_4': FIB[3]/FIB[4],  # 2/3
        'F_4/F_5': FIB[4]/FIB[5],  # 3/5
        'F_5/F_6': FIB[5]/FIB[6],  # 5/8
        '1 - F_3/F_7': 1 - FIB[3]/FIB[7],  # 1 - 2/13
        '1 + 1/F_7': 1 + 1/FIB[7],  # 1 + 1/13
        'φ/Ξ': PHI/XI,
        '1 - 1/F_11': 1 - 1/FIB[11],  # 1 - 1/89
    }
    
    results = {}
    
    for ratio_name, measured in MEASURED.items():
        base = BASE[ratio_name]
        print(f"\n{'='*60}")
        print(f"{ratio_name}: measured = {measured:.6f}, base = {base:.4f}")
        print(f"Base error: {abs(measured - base) / measured * 100:.4f}%")
        print(f"{'='*60}")
        
        best_error = abs(measured - base) / measured * 100
        best_formula = 'base'
        best_value = base
        
        all_results = []
        
        for corr_name, corr in corrections.items():
            val = base * corr
            error = abs(measured - val) / measured * 100
            all_results.append({
                'correction': corr_name,
                'value': val,
                'error_pct': error
            })
            if error < best_error:
                best_error = error
                best_formula = corr_name
                best_value = val
        
        # Sort by error
        all_results.sort(key=lambda x: x['error_pct'])
        
        print(f"\nTop 5 corrections:")
        for r in all_results[:5]:
            print(f"  {r['correction']:20s}: {r['value']:.6f} ({r['error_pct']:.4f}%)")
        
        results[ratio_name] = {
            'measured': measured,
            'base': base,
            'base_error': abs(measured - base) / measured * 100,
            'best_correction': best_formula,
            'best_value': best_value,
            'best_error': best_error,
            'all_corrections': all_results[:10]
        }
    
    return results


def test_double_corrections():
    """Test combinations of two corrections."""
    print("\n" + "=" * 70)
    print("Testing Double Corrections (A × B)")
    print("=" * 70)
    
    # Simple corrections to combine
    simple_corr = {
        'Ξ': XI,
        '1/Ξ': 1/XI,
        'φ^-1': INV_PHI,
        'φ^-2': INV_PHI ** 2,
        'F_3/F_4': FIB[3]/FIB[4],
        'F_4/F_5': FIB[4]/FIB[5],
        'F_5/F_6': FIB[5]/FIB[6],
        '1+1/F_7': 1 + 1/FIB[7],
        '1-1/F_7': 1 - 1/FIB[7],
    }
    
    results = {}
    
    for ratio_name, measured in MEASURED.items():
        base = BASE[ratio_name]
        base_error = abs(measured - base) / measured * 100
        
        print(f"\n{ratio_name}: base error = {base_error:.4f}%")
        
        all_results = []
        
        for (name1, c1), (name2, c2) in product(simple_corr.items(), repeat=2):
            val = base * c1 * c2
            error = abs(measured - val) / measured * 100
            if error < base_error:  # Only keep improvements
                all_results.append({
                    'correction': f'{name1} × {name2}',
                    'value': val,
                    'error_pct': error
                })
        
        all_results.sort(key=lambda x: x['error_pct'])
        
        if all_results:
            print(f"Top 3 double corrections:")
            for r in all_results[:3]:
                print(f"  {r['correction']:30s}: {r['value']:.6f} ({r['error_pct']:.4f}%)")
        else:
            print("  No improvement from double corrections")
        
        results[ratio_name] = all_results[:5] if all_results else []
    
    return results


def test_exact_formula_search():
    """
    Search for exact formulas using Fibonacci products and ratios.
    
    Form: (F_a × F_b × ... / F_c × F_d × ...) × φ^n × Ξ^m
    """
    print("\n" + "=" * 70)
    print("Exact Formula Search")
    print("=" * 70)
    
    results = {}
    
    for ratio_name, measured in MEASURED.items():
        print(f"\n{'='*60}")
        print(f"Searching for: {ratio_name} = {measured:.6f}")
        print(f"{'='*60}")
        
        best = []
        
        # Form: F_i × F_j × F_k × correction
        for i in range(2, 14):
            for j in range(i, 14):
                for k in range(j, 14):
                    base_prod = FIB[i] * FIB[j] * FIB[k]
                    
                    # Try various corrections
                    tests = [
                        ('none', base_prod),
                        ('× φ^-1', base_prod * INV_PHI),
                        ('× φ^-2', base_prod * INV_PHI**2),
                        ('× φ^1', base_prod * PHI),
                        ('× Ξ^-1', base_prod / XI),
                        ('× (1-1/F_7)', base_prod * (1 - 1/FIB[7])),
                        ('× (1+1/F_7)', base_prod * (1 + 1/FIB[7])),
                        ('× F_4/F_5', base_prod * FIB[4]/FIB[5]),
                        ('× F_5/F_6', base_prod * FIB[5]/FIB[6]),
                    ]
                    
                    for corr_name, val in tests:
                        if val == 0:
                            continue
                        error = abs(measured - val) / measured * 100
                        if error < 0.3:  # Only sub-0.3% matches
                            formula = f"F_{i}×F_{j}×F_{k} {corr_name}"
                            best.append({
                                'formula': formula,
                                'value': val,
                                'error_pct': error
                            })
        
        # Form: F_i × F_j × F_k / F_l
        for i in range(2, 14):
            for j in range(i, 14):
                for k in range(j, 14):
                    for l in range(2, 10):
                        if FIB[l] == 0:
                            continue
                        val = FIB[i] * FIB[j] * FIB[k] / FIB[l]
                        error = abs(measured - val) / measured * 100
                        if error < 0.3:
                            formula = f"F_{i}×F_{j}×F_{k}/F_{l}"
                            best.append({
                                'formula': formula,
                                'value': val,
                                'error_pct': error
                            })
        
        # Sort and deduplicate
        best.sort(key=lambda x: x['error_pct'])
        seen = set()
        unique_best = []
        for r in best:
            key = f"{r['value']:.4f}"
            if key not in seen:
                seen.add(key)
                unique_best.append(r)
        
        if unique_best:
            print(f"\nBest formulas (< 0.3% error):")
            for r in unique_best[:8]:
                print(f"  {r['formula']:35s} = {r['value']:.6f} ({r['error_pct']:.4f}%)")
        else:
            print(f"  No sub-0.3% formulas found")
        
        results[ratio_name] = unique_best[:10]
    
    return results


def derive_from_koide():
    """
    Use Koide Q = 2/3 as constraint to derive masses.
    
    Q = (me + mμ + mτ) / (√me + √mμ + √mτ)² = 2/3
    
    If μ/e = 208 × correction and we solve for τ...
    """
    print("\n" + "=" * 70)
    print("Koide-Constrained Derivation")
    print("=" * 70)
    
    # Known: me = 1 (normalized)
    me = 1.0
    
    # Known: Q = 2/3 = F_3/F_4
    Q = 2/3
    
    # Test: if μ/e has a certain value, what must τ/e be for Koide to hold?
    
    test_mu_e = [
        ('208', 208),
        ('208 × (1-1/13)', 208 * (1 - 1/13)),
        ('F_3×F_6×F_7 × φ^-1', FIB[3]*FIB[6]*FIB[7] * INV_PHI),
        ('206.7683 (exact)', 206.7683),
    ]
    
    print(f"\nSolving for τ/e given μ/e and Koide Q = 2/3:")
    header = f"{'μ/e formula':<30s} {'μ/e':>10s} {'τ/e derived':>12s} {'τ/e actual':>12s} {'error':>8s}"
    print(header)
    print("-" * 75)
    
    tau_e_actual = 3477.23
    
    for name, mu_e in test_mu_e:
        # Koide formula: Q = (1 + μ + τ) / (1 + √μ + √τ)²
        # Solve for τ given Q and μ
        # This is quadratic in √τ
        
        sqrt_mu = np.sqrt(mu_e)
        
        # Q(1 + √μ + √τ)² = 1 + μ + τ
        # Let x = √τ
        # Q(1 + √μ + x)² = 1 + μ + x²
        # Q(1 + √μ)² + 2Q(1+√μ)x + Qx² = 1 + μ + x²
        # (Q - 1)x² + 2Q(1+√μ)x + Q(1+√μ)² - 1 - μ = 0
        
        a = Q - 1
        b = 2 * Q * (1 + sqrt_mu)
        c = Q * (1 + sqrt_mu)**2 - 1 - mu_e
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            print(f"{name:<30s} {mu_e:>10.4f} {'no solution':>12s}")
            continue
        
        sqrt_tau = (-b + np.sqrt(discriminant)) / (2*a)  # Take positive root
        tau_e = sqrt_tau ** 2
        
        error = abs(tau_e - tau_e_actual) / tau_e_actual * 100
        
        print(f"{name:<30s} {mu_e:>10.4f} {tau_e:>12.4f} {tau_e_actual:>12.4f} {error:>7.3f}%")
    
    return {}


def final_summary():
    """Print final tightened formulas."""
    print("\n" + "=" * 70)
    print("FINAL TIGHTENED FORMULAS")
    print("=" * 70)
    
    formulas = {}
    
    # μ/e
    mu_e_measured = 206.7682830
    mu_e_base = FIB[3] * FIB[6] * FIB[7]  # 208
    mu_e_tight = mu_e_base * (1 - 1/FIB[7])  # 208 × (1 - 1/13) = 192
    # That makes it worse. Try:
    mu_e_tight2 = mu_e_base * (FIB[5]/FIB[6]) * (1 + 1/FIB[7])  # 208 × 5/8 × 14/13
    # = 208 × 0.625 × 1.077 = 140 (wrong direction)
    
    # The issue: 208 > 206.77, so we need a small reduction
    # 206.77/208 = 0.9942
    # Close to 1 - 1/F_8 = 1 - 1/21 = 0.9524 (too much)
    # Close to 1 - 1/F_9 = 1 - 1/34 = 0.9706 (still too much)
    # Close to 1 - 1/F_10 = 1 - 1/55 = 0.9818 (too much)
    # Close to 1 - 1/F_11 = 1 - 1/89 = 0.9888 (close!)
    # 208 × 0.9888 = 205.67 (0.53% error - slight improvement)
    
    # Try: (1 - 1/(F_7 × F_5)) = 1 - 1/65 = 0.9846
    # 208 × 0.9846 = 204.8 (worse)
    
    # Maybe the base is wrong? Let's check F_3 × F_5 × F_7 = 2 × 5 × 13 = 130
    # Or F_4 × F_5 × F_7 = 3 × 5 × 13 = 195
    # 195 × (1 + 1/F_8) = 195 × (22/21) = 204.3 (nope)
    
    # F_4 × F_6 × F_6 = 3 × 8 × 8 = 192
    # F_3 × F_6 × F_7 = 2 × 8 × 13 = 208 <- our current
    # F_2 × F_7 × F_8 = 1 × 13 × 21 = 273 (too big)
    
    # Try ratio formulas:
    # F_7 × F_8 / F_3 = 13 × 21 / 2 = 136.5
    # F_8 × F_9 / F_5 = 21 × 34 / 5 = 142.8
    # F_9 × F_10 / F_6 = 34 × 55 / 8 = 233.75
    # F_8 × F_10 / F_6 = 21 × 55 / 8 = 144.375
    # F_9 × F_9 / F_6 = 34 × 34 / 8 = 144.5
    # F_9 × F_10 / F_7 = 34 × 55 / 13 = 143.85
    
    # Hmm, 208 is actually quite close. The 0.6% may be intrinsic.
    # Let's check if 206.768 itself has meaning:
    # 206.768 / φ = 127.8 (not obvious Fibonacci)
    # 206.768 × φ = 334.5 (not obvious)
    # √206.768 = 14.38 ≈ F_7 + 1.38 ≈ F_7 + 1/φ² = 13 + 0.382 = 13.382
    # Hmm! √(μ/e) ≈ F_7 + 1/φ²
    
    sqrt_mu_e = np.sqrt(mu_e_measured)
    print(f"\nμ/e = {mu_e_measured:.6f}")
    print(f"√(μ/e) = {sqrt_mu_e:.4f}")
    print(f"F_7 + 1/φ² = {FIB[7] + 1/PHI**2:.4f}")
    print(f"  Difference: {abs(sqrt_mu_e - (FIB[7] + 1/PHI**2)):.4f} ({abs(sqrt_mu_e - (FIB[7] + 1/PHI**2))/sqrt_mu_e*100:.3f}%)")
    
    # So: μ/e = (F_7 + 1/φ²)² = (13.382)² = 179.1 (not right)
    
    # What about: √(μ/e) = F_7 + φ^-1 = 13 + 0.618 = 13.618
    # 13.618² = 185.4 (not right)
    
    # √(μ/e) = F_7 + 1 + 1/φ = 14.618 -> 213.7 (too big)
    
    # Actually: μ/e ≈ F_3 × F_6 × F_7 × (F_5/F_6)^(1/φ)
    # = 208 × (5/8)^0.618 = 208 × 0.739 = 153.7 (wrong)
    
    # Let's try: μ/e = F_4 × F_5 × F_7 × Ξ
    # = 3 × 5 × 13 × 1.0571 = 206.14 (0.30% error!)
    val = FIB[4] * FIB[5] * FIB[7] * XI
    err = abs(mu_e_measured - val) / mu_e_measured * 100
    print(f"\nTrying: F_4 × F_5 × F_7 × Ξ = {val:.4f} ({err:.3f}%)")
    
    formulas['mu/e'] = {
        'formula': 'F_4 × F_5 × F_7 × Ξ',
        'symbolic': '3 × 5 × 13 × (1 + π/55)',
        'value': val,
        'measured': mu_e_measured,
        'error_pct': err
    }
    
    # τ/e
    tau_e_measured = 3477.23
    tau_e_base = FIB[4] * FIB[7] * FIB[11]  # 3471
    err_base = abs(tau_e_measured - tau_e_base) / tau_e_measured * 100
    print(f"\nτ/e = {tau_e_measured:.4f}")
    print(f"F_4 × F_7 × F_11 = {tau_e_base} ({err_base:.3f}%)")
    
    # 3477.23 / 3471 = 1.00179
    # Close to 1 + 1/F_9 = 1 + 1/34 = 1.0294 (too much)
    # Close to 1 + 1/F_10 = 1 + 1/55 = 1.0182 (very close!)
    val = tau_e_base * (1 + 1/FIB[10])
    err = abs(tau_e_measured - val) / tau_e_measured * 100
    print(f"F_4 × F_7 × F_11 × (1 + 1/F_10) = {val:.4f} ({err:.3f}%)")
    
    # Try: F_4 × F_7 × F_11 × Ξ / Ξ + tiny
    # 3471 × 1.00179 = 3477.2 (that's what we need)
    # 1.00179 ≈ ?
    # (F_10 + 1)/F_10 = 56/55 = 1.0182 (0.03% off target)
    
    # F_4 × F_7 × F_11 + F_5 = 3471 + 5 = 3476 (very close!)
    val = FIB[4] * FIB[7] * FIB[11] + FIB[5]
    err = abs(tau_e_measured - val) / tau_e_measured * 100
    print(f"F_4 × F_7 × F_11 + F_5 = {val:.4f} ({err:.3f}%)")
    
    formulas['tau/e'] = {
        'formula': 'F_4 × F_7 × F_11 + F_5',
        'symbolic': '3 × 13 × 89 + 5',
        'value': val,
        'measured': tau_e_measured,
        'error_pct': err
    }
    
    # p/e
    p_e_measured = 1836.15267343
    # From exp_03: mp/me × α ≈ F_7 = 13
    # So mp/me ≈ F_7 / α ≈ 13 / 0.00729735 = 1782 (3% error)
    
    # Try: F_11 × F_9 / F_5 = 89 × 34 / 5 = 605.2 (too small)
    # F_11 × F_10 / F_7 = 89 × 55 / 13 = 376.5
    # F_12 × F_8 = 144 × 21 = 3024 (too big)
    # F_11 × F_8 = 89 × 21 = 1869 (1.8% error!)
    val = FIB[11] * FIB[8]
    err = abs(p_e_measured - val) / p_e_measured * 100
    print(f"\np/e = {p_e_measured:.4f}")
    print(f"F_11 × F_8 = {val} ({err:.3f}%)")
    
    # 1836.15 / 1869 = 0.9824
    # = 1 - 1/F_9 + tiny = 1 - 0.0294 = 0.9706 (not quite)
    # Try F_11 × F_8 × (1 - 1/F_10) = 1869 × 0.9818 = 1835.0 (0.06%!)
    val = FIB[11] * FIB[8] * (1 - 1/FIB[10])
    err = abs(p_e_measured - val) / p_e_measured * 100
    print(f"F_11 × F_8 × (1 - 1/F_10) = {val:.4f} ({err:.4f}%)")
    
    # Even better: F_11 × F_8 - F_4 = 1869 - 3 = 1866 (1.6% error)
    # F_11 × F_8 × (F_10-1)/F_10 = 1869 × 54/55 = 1835.0
    
    formulas['p/e'] = {
        'formula': 'F_11 × F_8 × (1 - 1/F_10)',
        'symbolic': '89 × 21 × (1 - 1/55) = 89 × 21 × 54/55',
        'value': val,
        'measured': p_e_measured,
        'error_pct': err
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY OF TIGHTENED FORMULAS")
    print("=" * 70)
    
    for name, data in formulas.items():
        print(f"\n{name}:")
        print(f"  Formula: {data['formula']}")
        print(f"  = {data['symbolic']}")
        print(f"  = {data['value']:.6f}")
        print(f"  Measured: {data['measured']:.6f}")
        print(f"  Error: {data['error_pct']:.4f}%")
    
    return formulas


def main():
    print("=" * 70)
    print("Experiment 05: Tighten Mass Formulas")
    print("=" * 70)
    
    results = {}
    
    # Single corrections
    results['single'] = test_corrections()
    
    # Double corrections
    results['double'] = test_double_corrections()
    
    # Exact formula search
    results['exact_search'] = test_exact_formula_search()
    
    # Koide derivation
    results['koide'] = derive_from_koide()
    
    # Final summary
    results['final'] = final_summary()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_05_tighten_mass',
        'results': results
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'exp_05_tighten_mass_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to results/exp_05_tighten_mass_{timestamp}.json")
    
    return output


if __name__ == '__main__':
    main()
