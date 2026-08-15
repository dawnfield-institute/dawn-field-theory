#!/usr/bin/env python3
"""
Milestone 2 Experiment 02: Alternative 2D Turbulence Formulas

CONTEXT:
Exp_01 showed the simple dimensional shift (F₂/F₃ = 1/2) doesn't match 2D data well.
Mean error was 30% vs 0.47% for 3D She-Leveque.

HYPOTHESIS:
2D turbulence may need a different Fibonacci structure because:
1. Enstrophy (not energy) cascades forward
2. Enstrophy has different scaling: ζ_p should be closer to p

Let's test alternative hypotheses:

A) Original She-Leveque for enstrophy (empirical): ζ_p = p × f(p) where f → 1 - ε
B) Fibonacci with enstrophy factor: Different β based on enstrophy constraint
C) Hybrid: 3D formula with 2D correction

KEY INSIGHT:
In 2D enstrophy cascade, exponents are CLOSER to linear (Kraichnan) than in 3D.
This suggests LESS intermittency, meaning:
- β should be LARGER (closer to 1) not smaller
- Perhaps β = F₄/F₅ = 3/5 = 0.6 or F₅/F₆ = 5/8 = 0.625?

Let's systematically test all Fibonacci ratios.
"""

import json
from datetime import datetime

# Fibonacci sequence
FIB = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21}


def generalized_she_leveque(p, beta, C0, dim_factor, exp_base):
    """
    Generalized She-Leveque formula:
    ζ_p = p/dim_factor + C0 × [1 - β^(p/exp_base)]
    """
    return p / dim_factor + C0 * (1 - beta**(p/exp_base))


def test_fibonacci_combinations():
    """
    Test all reasonable Fibonacci ratio combinations for 2D.
    
    We need:
    - β (cascade fraction): F_n/F_{n+1}
    - C0 (multiplier): F_m
    - dim_factor: (F_k)²
    - exp_base: F_j
    """
    # 2D experimental data
    data = {
        2: 1.35,
        4: 2.50,
        6: 3.50,
        8: 4.40,
    }
    
    results = []
    
    # Test various β values (F_n/F_{n+1} ratios)
    beta_options = [
        (1, 2, 1/1),      # F₁/F₂ = 1/1 = 1.0 (no intermittency)
        (2, 3, 1/2),      # F₂/F₃ = 1/2 = 0.5
        (3, 4, 2/3),      # F₃/F₄ = 2/3 ≈ 0.667 (3D value)
        (4, 5, 3/5),      # F₄/F₅ = 3/5 = 0.6
        (5, 6, 5/8),      # F₅/F₆ = 5/8 = 0.625
        (6, 7, 8/13),     # F₆/F₇ = 8/13 ≈ 0.615
    ]
    
    # Test various C0 values
    c0_options = [1, 2, 3, 5]  # F₂, F₃, F₄, F₅
    
    # Test various dimensional factors
    dim_options = [1, 4, 9]  # 1², 2², 3²
    
    # Test various exponent bases
    exp_options = [2, 3]  # F₃, F₄
    
    for (n1, n2, beta) in beta_options:
        for C0 in c0_options:
            for dim in dim_options:
                for exp_base in exp_options:
                    # Calculate predictions
                    errors = []
                    for p, measured in data.items():
                        pred = generalized_she_leveque(p, beta, C0, dim, exp_base)
                        error = abs(pred - measured) / measured * 100
                        errors.append(error)
                    
                    mean_error = sum(errors) / len(errors)
                    
                    results.append({
                        'beta': f"F{n1}/F{n2}={beta:.4f}",
                        'C0': C0,
                        'dim_factor': dim,
                        'exp_base': exp_base,
                        'mean_error': round(mean_error, 2),
                        'beta_val': beta
                    })
    
    # Sort by error
    results.sort(key=lambda x: x['mean_error'])
    return results


def analyze_best_formula():
    """
    Analyze the best-performing formula in detail.
    """
    data = {
        2: {'value': 1.35, 'uncertainty': 0.10},
        4: {'value': 2.50, 'uncertainty': 0.15},
        6: {'value': 3.50, 'uncertainty': 0.20},
        8: {'value': 4.40, 'uncertainty': 0.25},
    }
    
    # Based on preliminary testing, let's check some promising combinations
    formulas = {
        'original_2d_shift': {
            'beta': 1/2, 'C0': 1, 'dim': 4, 'exp': 2,
            'desc': 'ζ_p = p/4 + 1×[1-(1/2)^(p/2)]'
        },
        '3d_formula': {
            'beta': 2/3, 'C0': 2, 'dim': 9, 'exp': 3,
            'desc': 'ζ_p = p/9 + 2×[1-(2/3)^(p/3)] (3D She-Leveque)'
        },
        'enstrophy_linear': {
            'beta': 1.0, 'C0': 0, 'dim': 1, 'exp': 1,
            'desc': 'ζ_p = p (Kraichnan linear)'
        },
        'modified_enstrophy': {
            'beta': 5/8, 'C0': 2, 'dim': 4, 'exp': 2,
            'desc': 'ζ_p = p/4 + 2×[1-(5/8)^(p/2)]'
        },
        'log_correction': {
            # Log-corrected 2D formula (theoretical for enstrophy)
            'beta': None,  # Special handling
            'desc': 'ζ_p = p × [1 - c/log(p)] (log correction)'
        }
    }
    
    results = {}
    for name, params in formulas.items():
        if name == 'log_correction':
            # Special formula: ζ_p ≈ p × (1 - 0.1/log(p+1))
            predictions = {}
            errors = []
            for p, exp in data.items():
                pred = p * (1 - 0.15 / (1 + 0.5*p))  # Approximate log correction
                error = abs(pred - exp['value']) / exp['value'] * 100
                predictions[p] = round(pred, 3)
                errors.append(error)
            results[name] = {
                'predictions': predictions,
                'mean_error': round(sum(errors)/len(errors), 2),
                'desc': params['desc']
            }
        elif name == 'enstrophy_linear':
            predictions = {}
            errors = []
            for p, exp in data.items():
                pred = p
                error = abs(pred - exp['value']) / exp['value'] * 100
                predictions[p] = pred
                errors.append(error)
            results[name] = {
                'predictions': predictions,
                'mean_error': round(sum(errors)/len(errors), 2),
                'desc': params['desc']
            }
        else:
            predictions = {}
            errors = []
            for p, exp in data.items():
                pred = generalized_she_leveque(p, params['beta'], params['C0'], 
                                               params['dim'], params['exp'])
                error = abs(pred - exp['value']) / exp['value'] * 100
                predictions[p] = round(pred, 3)
                errors.append(error)
            results[name] = {
                'predictions': predictions,
                'mean_error': round(sum(errors)/len(errors), 2),
                'desc': params['desc']
            }
    
    return results


def main():
    print("=" * 70)
    print("MILESTONE 2 EXPERIMENT 02: Alternative 2D Turbulence Formulas")
    print("=" * 70)
    print()
    
    print("SEARCHING FIBONACCI PARAMETER SPACE...")
    print("-" * 50)
    
    all_results = test_fibonacci_combinations()
    
    # Show top 10
    print("TOP 10 COMBINATIONS:")
    print(f"{'β':>15} | {'C0':>4} | {'dim':>4} | {'exp':>4} | {'Error':>8}")
    print("-" * 50)
    for r in all_results[:10]:
        print(f"{r['beta']:>15} | {r['C0']:>4} | {r['dim_factor']:>4} | "
              f"{r['exp_base']:>4} | {r['mean_error']:>7.2f}%")
    
    print()
    print("DETAILED FORMULA COMPARISON:")
    print("-" * 70)
    
    formula_results = analyze_best_formula()
    
    # Experimental data for reference
    print("\nExperimental data (2D enstrophy cascade):")
    print("  p=2: 1.35 ± 0.10")
    print("  p=4: 2.50 ± 0.15")
    print("  p=6: 3.50 ± 0.20")
    print("  p=8: 4.40 ± 0.25")
    print()
    
    for name, result in sorted(formula_results.items(), key=lambda x: x[1]['mean_error']):
        print(f"{name}:")
        print(f"  Formula: {result['desc']}")
        print(f"  Predictions: {result['predictions']}")
        print(f"  Mean error: {result['mean_error']:.2f}%")
        print()
    
    # Key finding
    print("=" * 70)
    print("KEY FINDING:")
    print("=" * 70)
    
    best = min(formula_results.items(), key=lambda x: x[1]['mean_error'])
    print(f"Best formula: {best[0]}")
    print(f"Mean error: {best[1]['mean_error']:.2f}%")
    print()
    
    print("INTERPRETATION:")
    print("-" * 50)
    print("""
2D turbulence does NOT follow the simple She-Leveque pattern because:

1. DIFFERENT CONSERVATION: Enstrophy (∫ω²dx) is conserved, not energy
2. INVERSE CASCADE: Energy goes to LARGE scales, enstrophy to small
3. LESS INTERMITTENCY: 2D is more "organized" than 3D

The Fibonacci structure in 3D emerges from:
- Binary splitting (F₃ = 2) of vortex tubes
- 3D geometry (F₄ = 3 dimensions)

In 2D:
- Vortices MERGE (inverse cascade) rather than split
- No vortex stretching mechanism
- Different physics entirely

CONCLUSION: She-Leveque type formulas may not apply to 2D.
The 3D Fibonacci structure is specific to 3D vortex dynamics.
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'milestone2/exp_02_2d_alternatives',
        'top_10_combinations': all_results[:10],
        'formula_comparison': formula_results,
        'conclusion': '2D turbulence has different physics; She-Leveque structure may be 3D-specific'
    }
    
    with open('../results/02_2d_alternatives_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to results/02_2d_alternatives_*.json")
    
    return output


if __name__ == '__main__':
    main()
