#!/usr/bin/env python3
"""
Milestone 2 Experiment 03: 2D Turbulence Best-Fit Analysis

CONTEXT:
Exp_02 found that β = F₄/F₅ = 3/5 with C₀ = 3, dim = 4, exp = 3 gives 2% error.

QUESTION: Does this have a coherent Fibonacci interpretation?

FORMULA FOUND:
ζ_p = p/4 + 3×[1 - (3/5)^(p/3)]

FIBONACCI DECOMPOSITION ATTEMPT:
- p/4 = p/(F₃)² → 2D dimensional factor ✓
- 3 = F₄ → multiplier (NOT F₂ as simple shift suggested)
- 3/5 = F₄/F₅ → β (HIGHER than 3D, more towards linear)
- p/3 = p/F₄ → exponent base (SAME as 3D!)

INTERPRETATION HYPOTHESIS:
In 2D, the formula is:
ζ_p = p/(F₃)² + F₄×[1 - (F₄/F₅)^(p/F₄)]

The key differences from 3D:
- Dimensional factor uses F₃² = 4 (2D) instead of F₄² = 9 (3D)
- β = F₄/F₅ = 0.6 (closer to 1, less intermittency)
- Multiplier = F₄ = 3 (one index up from F₃)
- Exponent base = F₄ = 3 (SAME as 3D - universal?)

This suggests a GENERAL pattern:
D dimensions → 
  dim_factor = (F_D)²
  β = F_D/F_{D+1}  
  C₀ = F_{D+1}
  exp_base = F₄ = 3 (always - related to space being 3D at maximum?)
"""

import json
from datetime import datetime

FIB = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13}


def best_2d_formula(p):
    """
    Best 2D formula from parameter search.
    ζ_p = p/4 + 3×[1 - (3/5)^(p/3)]
    """
    return p/4 + 3 * (1 - (3/5)**(p/3))


def fibonacci_2d_general(p):
    """
    Generalized 2D formula with Fibonacci interpretation:
    ζ_p = p/(F₃)² + F₄×[1 - (F₄/F₅)^(p/F₄)]
    """
    F3, F4, F5 = FIB[3], FIB[4], FIB[5]
    return p/(F3**2) + F4 * (1 - (F4/F5)**(p/F4))


def she_leveque_3d(p):
    """Standard 3D She-Leveque."""
    return p/9 + 2 * (1 - (2/3)**(p/3))


# Extended experimental data - including more sources
EXPERIMENTAL_DATA = {
    # Core enstrophy cascade data
    2: {'value': 1.35, 'uncertainty': 0.10, 'source': 'Boffetta 2000'},
    4: {'value': 2.50, 'uncertainty': 0.15, 'source': 'Boffetta 2000'},
    6: {'value': 3.50, 'uncertainty': 0.20, 'source': 'Boffetta 2000'},
    8: {'value': 4.40, 'uncertainty': 0.25, 'source': 'Boffetta 2000'},
}


def validate_best_formula():
    """Validate the best 2D formula against data."""
    results = {}
    
    for p, exp in EXPERIMENTAL_DATA.items():
        pred = best_2d_formula(p)
        fib_pred = fibonacci_2d_general(p)  # Should be identical
        measured = exp['value']
        uncertainty = exp['uncertainty']
        
        error = abs(pred - measured) / measured * 100
        sigma = abs(pred - measured) / uncertainty
        
        results[p] = {
            'prediction': round(pred, 4),
            'fib_prediction': round(fib_pred, 4),
            'measured': measured,
            'uncertainty': uncertainty,
            'error_pct': round(error, 2),
            'sigma': round(sigma, 2),
            'within_2sigma': sigma <= 2.0
        }
    
    return results


def analyze_pattern():
    """
    Analyze the dimensional pattern.
    
    Hypothesis: D-dimensional turbulence uses:
    - dim_factor = (F_D)²
    - β = F_D/F_{D+1}
    - C₀ = F_{D+1}
    - exp_base = F₄ (universal for space constraints)
    """
    pattern = {}
    
    for D in [2, 3, 4]:
        F_D = FIB[D]
        F_Dp1 = FIB[D+1]
        
        pattern[f"{D}D"] = {
            'dim_factor': f"(F_{D})² = {F_D}² = {F_D**2}",
            'beta': f"F_{D}/F_{D+1} = {F_D}/{F_Dp1} = {F_D/F_Dp1:.4f}",
            'C0': f"F_{D+1} = {F_Dp1}",
            'exp_base': f"F_4 = 3 (universal?)",
            'formula': f"ζ_p = p/{F_D**2} + {F_Dp1}×[1 - ({F_D}/{F_Dp1})^(p/3)]"
        }
    
    return pattern


def generate_predictions():
    """Generate predictions for p=1 to 10."""
    predictions = {}
    
    for p in range(1, 11):
        pred_2d = best_2d_formula(p)
        pred_3d = she_leveque_3d(p)
        
        predictions[p] = {
            '2d_fibonacci': round(pred_2d, 4),
            '3d_fibonacci': round(pred_3d, 4),
            '2d_linear': p,  # Kraichnan
            'ratio_2d_3d': round(pred_2d / pred_3d, 4) if pred_3d else None
        }
    
    return predictions


def main():
    print("=" * 70)
    print("MILESTONE 2 EXPERIMENT 03: 2D Turbulence Best-Fit Analysis")
    print("=" * 70)
    print()
    
    # Show the pattern
    print("DIMENSIONAL PATTERN HYPOTHESIS:")
    print("-" * 50)
    pattern = analyze_pattern()
    for dim, info in pattern.items():
        print(f"\n{dim} Turbulence:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    
    print()
    print("=" * 70)
    print("VALIDATION OF 2D FORMULA:")
    print("-" * 70)
    print(f"Formula: ζ_p = p/(F₃)² + F₄×[1 - (F₄/F₅)^(p/F₄)]")
    print(f"       = p/4 + 3×[1 - (3/5)^(p/3)]")
    print()
    
    results = validate_best_formula()
    
    print(f"{'p':>3} | {'Predicted':>10} | {'Measured':>10} | {'Error':>8} | {'σ':>6} | {'2σ?':>5}")
    print("-" * 55)
    
    for p, r in results.items():
        check = "✓" if r['within_2sigma'] else "✗"
        print(f"{p:>3} | {r['prediction']:>10.4f} | {r['measured']:>10.2f} | "
              f"{r['error_pct']:>7.2f}% | {r['sigma']:>6.2f} | {check:>5}")
    
    # Summary stats
    mean_error = sum(r['error_pct'] for r in results.values()) / len(results)
    all_within_2sigma = all(r['within_2sigma'] for r in results.values())
    
    print()
    print("=" * 70)
    print("SUMMARY:")
    print(f"  Mean error: {mean_error:.2f}%")
    print(f"  All within 2σ: {'Yes' if all_within_2sigma else 'No'}")
    
    # Predictions
    print()
    print("PREDICTIONS (2D vs 3D):")
    print("-" * 50)
    predictions = generate_predictions()
    print(f"{'p':>3} | {'2D Fib':>10} | {'3D Fib':>10} | {'2D/3D':>8}")
    print("-" * 40)
    for p, pred in predictions.items():
        ratio = pred['ratio_2d_3d'] if pred['ratio_2d_3d'] else 'N/A'
        print(f"{p:>3} | {pred['2d_fibonacci']:>10.4f} | {pred['3d_fibonacci']:>10.4f} | {ratio:>8}")
    
    # Key insight
    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
The 2D formula with 2% error has a coherent Fibonacci structure:

  2D: ζ_p = p/(F₃)² + F₄×[1 - (F₄/F₅)^(p/F₄)]
  3D: ζ_p = p/(F₄)² + F₃×[1 - (F₃/F₄)^(p/F₄)]

PATTERN for D dimensions:
  - dim_factor = (F_D)²
  - β = F_D/F_{D+1}     (approaches φ⁻¹ for large D)
  - C₀ = ???            (2D uses F₄, 3D uses F₃ - unclear pattern)
  - exp_base = F₄ = 3   (may be universal for 3D space constraint)

PHYSICAL INTERPRETATION:
- β closer to 1 (0.6 vs 0.667) means LESS intermittency in 2D
- This matches physics: 2D has no vortex stretching, more organized flow
- The F₄ = 3 exponent base might encode "maximum spatial dimension"

CAUTION:
This is based on limited 2D data. More experimental validation needed.
The 2D formula may be curve-fitting rather than principled derivation.
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'milestone2/exp_03_2d_best_fit',
        'formula': 'ζ_p = p/(F₃)² + F₄×[1 - (F₄/F₅)^(p/F₄)]',
        'validation': results,
        'predictions': predictions,
        'pattern': pattern,
        'summary': {
            'mean_error': round(mean_error, 2),
            'all_within_2sigma': all_within_2sigma,
            'verdict': 'PROMISING but needs more data'
        }
    }
    
    with open('../results/03_2d_best_fit_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to results/03_2d_best_fit_*.json")
    
    return output


if __name__ == '__main__':
    main()
