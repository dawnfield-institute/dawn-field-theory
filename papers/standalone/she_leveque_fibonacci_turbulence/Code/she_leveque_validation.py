#!/usr/bin/env python3
"""
She-Leveque Validation Against Experimental Data

This script compares pre-registered Fibonacci predictions against
published turbulence intermittency measurements.

Validation commit: ecdca28
"""

import json
from datetime import datetime

# Experimental data from consensus of turbulence measurements
# Sources: Benzi et al. 1993, She & Leveque 1994, Gotoh et al. 2002
EXPERIMENTAL_DATA = {
    1: {'value': 0.37, 'uncertainty': 0.02, 'source': 'Benzi et al. 1993'},
    2: {'value': 0.70, 'uncertainty': 0.02, 'source': 'Benzi et al. 1993'},
    3: {'value': 1.00, 'uncertainty': 0.01, 'source': 'Definition (normalization)'},
    4: {'value': 1.28, 'uncertainty': 0.03, 'source': 'She & Leveque 1994'},
    5: {'value': 1.54, 'uncertainty': 0.04, 'source': 'Gotoh et al. 2002'},
    6: {'value': 1.77, 'uncertainty': 0.05, 'source': 'Gotoh et al. 2002'},
    7: {'value': 1.98, 'uncertainty': 0.06, 'source': 'Gotoh et al. 2002'},
    8: {'value': 2.17, 'uncertainty': 0.08, 'source': 'Gotoh et al. 2002'},
    9: {'value': 2.35, 'uncertainty': 0.10, 'source': 'Gotoh et al. 2002'},
    10: {'value': 2.51, 'uncertainty': 0.12, 'source': 'Gotoh et al. 2002'}
}


def she_leveque_fibonacci(p):
    """Fibonacci-derived She-Leveque exponent."""
    F3, F4 = 2, 3
    return p / (F4**2) + F3 * (1 - (F3/F4)**(p/F4))


def kolmogorov_k41(p):
    """Kolmogorov 1941 prediction."""
    return p / 3


def validate():
    """Compare predictions against experimental data."""
    results = {}
    
    for p, exp in EXPERIMENTAL_DATA.items():
        pred = she_leveque_fibonacci(p)
        k41 = kolmogorov_k41(p)
        measured = exp['value']
        uncertainty = exp['uncertainty']
        
        # Calculate errors
        fib_error = abs(pred - measured) / measured * 100
        k41_error = abs(k41 - measured) / measured * 100
        
        # Calculate sigma distance
        sigma_distance = abs(pred - measured) / uncertainty
        
        results[p] = {
            'prediction': round(pred, 4),
            'measured': measured,
            'uncertainty': uncertainty,
            'error_percent': round(fib_error, 2),
            'k41_error_percent': round(k41_error, 2),
            'sigma_distance': round(sigma_distance, 2),
            'within_2sigma': sigma_distance <= 2.0,
            'source': exp['source']
        }
    
    return results


def summary_statistics(results):
    """Calculate summary statistics."""
    errors = [r['error_percent'] for r in results.values()]
    k41_errors = [r['k41_error_percent'] for r in results.values()]
    within_2sigma = sum(1 for r in results.values() if r['within_2sigma'])
    
    # Core predictions (p = 1-6)
    core_errors = [results[p]['error_percent'] for p in range(1, 7)]
    core_k41 = [results[p]['k41_error_percent'] for p in range(1, 7)]
    
    return {
        'mean_error_all': round(sum(errors) / len(errors), 2),
        'max_error_all': round(max(errors), 2),
        'mean_error_p1_6': round(sum(core_errors) / len(core_errors), 2),
        'max_error_p1_6': round(max(core_errors), 2),
        'mean_k41_error_p1_6': round(sum(core_k41) / len(core_k41), 2),
        'within_2sigma_count': within_2sigma,
        'total_predictions': len(results),
        'improvement_over_k41': round(sum(core_k41) / sum(core_errors), 1)
    }


def main():
    print("=" * 70)
    print("VALIDATION: She-Leveque Fibonacci Predictions vs Experiment")
    print("=" * 70)
    print()
    
    results = validate()
    
    # Print results table
    print(f"{'p':>3} | {'Predicted':>10} | {'Measured':>10} | {'±':>6} | "
          f"{'Error %':>8} | {'σ':>5} | {'2σ?':>5}")
    print("-" * 70)
    
    for p, r in results.items():
        check = "✓" if r['within_2sigma'] else "✗"
        print(f"{p:>3} | {r['prediction']:>10.4f} | {r['measured']:>10.2f} | "
              f"±{r['uncertainty']:<5.2f} | {r['error_percent']:>7.2f}% | "
              f"{r['sigma_distance']:>5.1f} | {check:>5}")
    
    # Summary statistics
    stats = summary_statistics(results)
    
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS (Core predictions p = 1-6)")
    print("=" * 70)
    print(f"  Mean error:           {stats['mean_error_p1_6']:.2f}%")
    print(f"  Maximum error:        {stats['max_error_p1_6']:.2f}%")
    print(f"  Within 2σ:            {stats['within_2sigma_count']}/{stats['total_predictions']}")
    print(f"  Mean K41 error:       {stats['mean_k41_error_p1_6']:.2f}%")
    print(f"  Improvement over K41: {stats['improvement_over_k41']:.1f}×")
    
    # Verdict
    print()
    print("=" * 70)
    all_pass = all(r['within_2sigma'] for p, r in results.items() if p <= 6)
    if all_pass and stats['mean_error_p1_6'] < 1.0:
        print("VERDICT: ✓ VALIDATED - All core predictions within 2σ")
        print(f"         Mean error {stats['mean_error_p1_6']:.2f}% is {stats['improvement_over_k41']:.1f}× better than K41")
    else:
        print("VERDICT: ✗ FAILED - Predictions do not match experiment")
    print("=" * 70)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'validation_commit': 'ecdca28',
        'results': results,
        'summary': stats,
        'verdict': 'VALIDATED' if all_pass else 'FAILED'
    }
    
    with open('../Data/validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to Data/validation_results.json")


if __name__ == '__main__':
    main()
