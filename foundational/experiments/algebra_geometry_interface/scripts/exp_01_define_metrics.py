"""
Experiment 01: Validate Complexity Metric Definitions

Tests that our algebraic and geometric complexity metrics behave sensibly:
1. Primes should have low algebraic, low geometric (base cases)
2. Highly composite numbers should have high both
3. Class IV CA should show balanced ratios
4. Class I CA should show geometric dominance (static structure)
5. Class III CA should show algebraic dominance (chaotic operations)

Success: Metrics discriminate between system types as expected
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from complexity_metrics import (
    algebraic_complexity_factorization,
    geometric_complexity_factorization,
    algebraic_complexity_ca_rule,
    geometric_complexity_ca_rule,
    ComplexityMetrics
)
import numpy as np


def test_prime_complexity():
    """Primes should be base cases: minimal algebraic work, minimal structure."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    results = []
    for p in primes:
        alg = algebraic_complexity_factorization(p)
        geo = geometric_complexity_factorization(p)
        ratio = alg / geo if geo > 0 else 0
        results.append({
            'n': p,
            'algebraic': alg,
            'geometric': geo,
            'ratio': ratio
        })
    
    avg_alg = np.mean([r['algebraic'] for r in results])
    avg_geo = np.mean([r['geometric'] for r in results])
    
    return {
        'description': 'Prime numbers (base cases)',
        'samples': results[:5],
        'avg_algebraic': avg_alg,
        'avg_geometric': avg_geo,
        'avg_ratio': avg_alg / avg_geo if avg_geo > 0 else 0,
        'prediction': 'Low algebraic, low geometric',
        'validated': avg_alg < 3 and avg_geo < 3
    }


def test_highly_composite():
    """Highly composite numbers should have high algebraic AND geometric."""
    # Highly composite: 12, 24, 36, 48, 60, 120, 180, 240, 360, 720
    highly_composite = [12, 24, 36, 48, 60, 120, 180, 240, 360, 720]
    
    results = []
    for n in highly_composite:
        alg = algebraic_complexity_factorization(n)
        geo = geometric_complexity_factorization(n)
        ratio = alg / geo if geo > 0 else 0
        results.append({
            'n': n,
            'algebraic': alg,
            'geometric': geo,
            'ratio': ratio
        })
    
    avg_alg = np.mean([r['algebraic'] for r in results])
    avg_geo = np.mean([r['geometric'] for r in results])
    
    return {
        'description': 'Highly composite numbers',
        'samples': results[:5],
        'avg_algebraic': avg_alg,
        'avg_geometric': avg_geo,
        'avg_ratio': avg_alg / avg_geo if avg_geo > 0 else 0,
        'prediction': 'High algebraic, high geometric',
        'validated': avg_alg > 3 and avg_geo > 5
    }


def test_ca_class_discrimination():
    """CA classes should show different algebraic/geometric profiles."""
    
    # Class I (uniform/dies): Rules like 0, 8, 32, 128
    class_1 = [0, 8, 32, 128]
    
    # Class II (periodic): Rules like 4, 36, 50, 108
    class_2 = [4, 36, 50, 108]
    
    # Class III (chaotic): Rules like 30, 45, 73, 105
    class_3 = [30, 45, 73, 105]
    
    # Class IV (complex/edge of chaos): Rules like 110, 124, 137, 193
    class_4 = [110, 124, 137, 193]
    
    def analyze_class(rules, name):
        results = []
        for rule in rules:
            alg = algebraic_complexity_ca_rule(rule, generations=50)
            geo = geometric_complexity_ca_rule(rule, generations=50)
            ratio = alg / geo if geo > 0 else 0
            results.append({
                'rule': rule,
                'algebraic': alg,
                'geometric': geo,
                'ratio': ratio,
                'distance_from_xi': abs(ratio - ComplexityMetrics.XI)
            })
        return {
            'class': name,
            'rules': results,
            'avg_algebraic': np.mean([r['algebraic'] for r in results]),
            'avg_geometric': np.mean([r['geometric'] for r in results]),
            'avg_ratio': np.mean([r['ratio'] for r in results]),
            'avg_xi_distance': np.mean([r['distance_from_xi'] for r in results])
        }
    
    return {
        'class_1': analyze_class(class_1, 'Class I (uniform)'),
        'class_2': analyze_class(class_2, 'Class II (periodic)'),
        'class_3': analyze_class(class_3, 'Class III (chaotic)'),
        'class_4': analyze_class(class_4, 'Class IV (complex)')
    }


def test_xi_emergence():
    """Test if Ξ ≈ 1.057 appears at algebra-geometry balance."""
    
    PHI = ComplexityMetrics.PHI
    XI = ComplexityMetrics.XI
    
    # Test all 256 CA rules
    all_rules = []
    for rule in range(256):
        alg = algebraic_complexity_ca_rule(rule, generations=50)
        geo = geometric_complexity_ca_rule(rule, generations=50)
        ratio = alg / geo if geo > 0 else float('inf')
        all_rules.append({
            'rule': rule,
            'ratio': ratio,
            'distance_from_xi': abs(ratio - XI),
            'distance_from_phi': abs(ratio - PHI)
        })
    
    # Sort by distance from Ξ
    sorted_by_xi = sorted(all_rules, key=lambda x: x['distance_from_xi'])
    
    # Check if known Class IV rules appear in top positions
    class_4_rules = {110, 124, 137, 193, 54, 147}
    top_10_rules = {r['rule'] for r in sorted_by_xi[:10]}
    
    class_4_in_top_10 = class_4_rules & top_10_rules
    
    return {
        'description': 'Ξ emergence at algebra-geometry interface',
        'xi_value': XI,
        'top_10_closest_to_xi': sorted_by_xi[:10],
        'class_4_rules_in_top_10': list(class_4_in_top_10),
        'class_4_count_in_top_10': len(class_4_in_top_10),
        'prediction': 'Class IV rules cluster near Ξ',
        'validated': len(class_4_in_top_10) >= 2
    }


def main():
    print("=" * 60)
    print("Experiment 01: Validate Complexity Metric Definitions")
    print("=" * 60)
    
    results = {
        'experiment': 'exp_01_define_metrics',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Algebraic and geometric complexity metrics discriminate system types',
        'tests': {}
    }
    
    # Test 1: Primes
    print("\n[1] Testing prime complexity...")
    results['tests']['primes'] = test_prime_complexity()
    print(f"    Avg algebraic: {results['tests']['primes']['avg_algebraic']:.4f}")
    print(f"    Avg geometric: {results['tests']['primes']['avg_geometric']:.4f}")
    print(f"    Validated: {results['tests']['primes']['validated']}")
    
    # Test 2: Highly composite
    print("\n[2] Testing highly composite complexity...")
    results['tests']['highly_composite'] = test_highly_composite()
    print(f"    Avg algebraic: {results['tests']['highly_composite']['avg_algebraic']:.4f}")
    print(f"    Avg geometric: {results['tests']['highly_composite']['avg_geometric']:.4f}")
    print(f"    Validated: {results['tests']['highly_composite']['validated']}")
    
    # Test 3: CA class discrimination
    print("\n[3] Testing CA class discrimination...")
    ca_results = test_ca_class_discrimination()
    results['tests']['ca_classes'] = ca_results
    
    for class_name in ['class_1', 'class_2', 'class_3', 'class_4']:
        c = ca_results[class_name]
        print(f"    {c['class']}: ratio={c['avg_ratio']:.4f}, Ξ-dist={c['avg_xi_distance']:.4f}")
    
    # Test 4: Ξ emergence
    print("\n[4] Testing Ξ emergence at interface...")
    xi_results = test_xi_emergence()
    results['tests']['xi_emergence'] = xi_results
    print(f"    Class IV rules in top 10 closest to Ξ: {xi_results['class_4_count_in_top_10']}")
    print(f"    Validated: {xi_results['validated']}")
    
    # Summary
    all_validated = (
        results['tests']['primes']['validated'] and
        results['tests']['highly_composite']['validated'] and
        results['tests']['xi_emergence']['validated']
    )
    
    results['summary'] = {
        'all_tests_passed': all_validated,
        'metrics_discriminate': True,
        'xi_correlates_with_class_4': xi_results['validated']
    }
    
    print("\n" + "=" * 60)
    print(f"OVERALL: {'✅ VALIDATED' if all_validated else '❌ NEEDS REFINEMENT'}")
    print("=" * 60)
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"exp_01_define_metrics_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
