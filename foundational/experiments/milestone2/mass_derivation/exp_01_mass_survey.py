#!/usr/bin/env python3
"""
Experiment 01: Mass Ratio Survey

Part VII: Mass Ratio Derivation

Survey all known particle mass ratios and test for Fibonacci structure.
This is exploratory - looking for patterns before attempting derivation.

Methodology:
- Catalog major mass ratios
- Test against Fibonacci numbers, products, ratios
- Test against φ powers and Ξ corrections
- Identify candidates for deeper derivation
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations, product


# Fibonacci sequence
def fib(n: int) -> int:
    if n <= 1:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# First 25 Fibonacci numbers
FIB = [fib(i) for i in range(25)]
FIB_SET = set(FIB[1:])  # exclude 0

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
XI = 1 + np.pi / 55

# Measured mass ratios (CODATA 2022)
MASS_RATIOS = {
    # Lepton ratios
    'mu/e': 206.7682830,
    'tau/e': 3477.23,
    'tau/mu': 16.8170,
    
    # Baryon ratios
    'p/e': 1836.15267343,
    'n/e': 1838.68366173,
    'n/p': 1.00137841931,
    
    # Quark ratios (approximate, PDG)
    'c/u': 554,  # ~1.27 GeV / 2.3 MeV
    's/d': 19.5,  # ~95 MeV / 4.8 MeV
    'b/s': 44.2,  # ~4.18 GeV / 95 MeV
    't/b': 41.4,  # ~173 GeV / 4.18 GeV
    
    # Boson ratios
    'W/e': 157327,  # 80.4 GeV / 511 keV
    'Z/W': 1.134,
    'H/W': 1.553,
}


def test_pure_fibonacci(ratio: float, tolerance: float = 0.05) -> List[Dict]:
    """Test if ratio is close to a Fibonacci number."""
    matches = []
    for i, f in enumerate(FIB[1:], 1):
        error = abs(ratio - f) / ratio
        if error < tolerance:
            matches.append({
                'type': 'pure_fib',
                'formula': f'F_{i}',
                'value': f,
                'error_pct': error * 100
            })
    return matches


def test_fibonacci_products(ratio: float, max_terms: int = 3, tolerance: float = 0.05) -> List[Dict]:
    """Test if ratio is close to product of Fibonacci numbers."""
    matches = []
    
    # 2-term products
    for i in range(2, 15):
        for j in range(i, 15):
            prod = FIB[i] * FIB[j]
            if prod == 0:
                continue
            error = abs(ratio - prod) / ratio
            if error < tolerance:
                matches.append({
                    'type': 'fib_product',
                    'formula': f'F_{i} × F_{j}',
                    'value': prod,
                    'error_pct': error * 100
                })
    
    # 3-term products
    if max_terms >= 3:
        for i in range(2, 12):
            for j in range(i, 12):
                for k in range(j, 12):
                    prod = FIB[i] * FIB[j] * FIB[k]
                    if prod == 0:
                        continue
                    error = abs(ratio - prod) / ratio
                    if error < tolerance:
                        matches.append({
                            'type': 'fib_product_3',
                            'formula': f'F_{i} × F_{j} × F_{k}',
                            'value': prod,
                            'error_pct': error * 100
                        })
    
    return matches


def test_fibonacci_ratios(ratio: float, tolerance: float = 0.05) -> List[Dict]:
    """Test if ratio is close to ratio of Fibonacci numbers."""
    matches = []
    
    for i in range(2, 20):
        for j in range(2, 20):
            if FIB[j] == 0:
                continue
            fib_ratio = FIB[i] / FIB[j]
            error = abs(ratio - fib_ratio) / ratio
            if error < tolerance:
                matches.append({
                    'type': 'fib_ratio',
                    'formula': f'F_{i} / F_{j}',
                    'value': fib_ratio,
                    'error_pct': error * 100
                })
    
    return matches


def test_phi_powers(ratio: float, tolerance: float = 0.05) -> List[Dict]:
    """Test if ratio is close to a power of φ."""
    matches = []
    
    for n in range(-20, 30):
        phi_power = PHI ** n
        error = abs(ratio - phi_power) / ratio
        if error < tolerance:
            matches.append({
                'type': 'phi_power',
                'formula': f'φ^{n}',
                'value': phi_power,
                'error_pct': error * 100
            })
    
    return matches


def test_xi_corrections(ratio: float, tolerance: float = 0.05) -> List[Dict]:
    """Test if ratio is close to Fibonacci × Ξ correction."""
    matches = []
    
    for i in range(2, 20):
        for power in range(1, 4):
            corrected = FIB[i] * (XI ** power)
            error = abs(ratio - corrected) / ratio
            if error < tolerance:
                matches.append({
                    'type': 'xi_corrected',
                    'formula': f'F_{i} × Ξ^{power}',
                    'value': corrected,
                    'error_pct': error * 100
                })
    
    return matches


def test_mixed_formulas(ratio: float, tolerance: float = 0.02) -> List[Dict]:
    """Test more complex formulas mixing Fibonacci, φ, and π."""
    matches = []
    
    # F_i × φ^n
    for i in range(2, 18):
        for n in range(-5, 10):
            val = FIB[i] * (PHI ** n)
            error = abs(ratio - val) / ratio
            if error < tolerance:
                matches.append({
                    'type': 'fib_phi',
                    'formula': f'F_{i} × φ^{n}',
                    'value': val,
                    'error_pct': error * 100
                })
    
    # F_i × F_j × φ^n
    for i in range(2, 12):
        for j in range(2, 12):
            for n in range(-3, 5):
                val = FIB[i] * FIB[j] * (PHI ** n)
                error = abs(ratio - val) / ratio
                if error < tolerance:
                    matches.append({
                        'type': 'fib_fib_phi',
                        'formula': f'F_{i} × F_{j} × φ^{n}',
                        'value': val,
                        'error_pct': error * 100
                    })
    
    # F_i² × something
    for i in range(2, 15):
        for n in range(-3, 5):
            val = (FIB[i] ** 2) * (PHI ** n)
            error = abs(ratio - val) / ratio
            if error < tolerance:
                matches.append({
                    'type': 'fib_squared_phi',
                    'formula': f'F_{i}² × φ^{n}',
                    'value': val,
                    'error_pct': error * 100
                })
    
    return matches


def survey_ratio(name: str, ratio: float) -> Dict:
    """Run full survey on a single ratio."""
    all_matches = []
    
    all_matches.extend(test_pure_fibonacci(ratio))
    all_matches.extend(test_fibonacci_products(ratio))
    all_matches.extend(test_fibonacci_ratios(ratio))
    all_matches.extend(test_phi_powers(ratio))
    all_matches.extend(test_xi_corrections(ratio))
    all_matches.extend(test_mixed_formulas(ratio))
    
    # Sort by error
    all_matches.sort(key=lambda x: x['error_pct'])
    
    return {
        'name': name,
        'measured': ratio,
        'n_matches': len(all_matches),
        'best_matches': all_matches[:10],  # Top 10
        'best_error_pct': all_matches[0]['error_pct'] if all_matches else None
    }


def main():
    print("=" * 70)
    print("Experiment 01: Mass Ratio Survey")
    print("=" * 70)
    
    results = {}
    
    for name, ratio in MASS_RATIOS.items():
        print(f"\n--- {name} = {ratio} ---")
        survey = survey_ratio(name, ratio)
        results[name] = survey
        
        if survey['best_matches']:
            print(f"  Best matches ({survey['n_matches']} total):")
            for match in survey['best_matches'][:5]:
                print(f"    {match['formula']:20s} = {match['value']:.4f} ({match['error_pct']:.3f}% error)")
        else:
            print("  No matches within tolerance")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Fibonacci Representations")
    print("=" * 70)
    
    ranked = sorted(results.values(), key=lambda x: x['best_error_pct'] if x['best_error_pct'] else 999)
    
    print(f"\n{'Ratio':<12} {'Measured':<15} {'Best Formula':<25} {'Error %':<10}")
    print("-" * 65)
    
    for r in ranked:
        if r['best_matches']:
            best = r['best_matches'][0]
            print(f"{r['name']:<12} {r['measured']:<15.4f} {best['formula']:<25} {best['error_pct']:.4f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_01_mass_survey',
        'results': results,
        'summary': {
            'total_ratios': len(MASS_RATIOS),
            'ratios_with_matches': sum(1 for r in results.values() if r['best_matches']),
            'best_overall': ranked[0] if ranked else None
        }
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'exp_01_mass_survey_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to results/exp_01_mass_survey_{timestamp}.json")
    
    return output


if __name__ == '__main__':
    main()
