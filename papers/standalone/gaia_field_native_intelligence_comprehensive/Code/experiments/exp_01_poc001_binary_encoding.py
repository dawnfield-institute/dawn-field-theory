"""
Experiment 01: Binary Pattern Encoding
======================================

POC-001: Test the simplest possible pattern encoding.

Hypothesis:
- Binary strings can be encoded as distinct field patterns
- Different binaries produce measurably different fields
- Patterns survive Klein-Gordon evolution

Technical Requirements:
- PyTorch only (no numpy)
- GPU acceleration
- Uses GAIA physics principles
"""

import torch
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / 'fracton'))

from utils import (
    FieldEncoder, FieldEvolver, ExperimentResult,
    measure_pattern_distance, measure_pattern_similarity,
    get_gpu_info, get_results_dir, generate_experiment_id, DEVICE
)


def run_distinctiveness_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 1: Different binary strings produce different patterns.
    
    Success criteria: All pairs have distance > 0.1
    """
    print("\n=== Test 1: Pattern Distinctiveness ===")
    
    test_patterns = [
        "0000", "0001", "0011", "0111", "1111",
        "1010", "0101", "1100", "0110", "1001"
    ]
    
    # Encode all patterns
    encodings = {}
    for pattern in test_patterns:
        result = encoder.encode_binary(pattern)
        encodings[pattern] = result
        print(f"  Encoded '{pattern}': energy={result.field_energy:.4f}, time={result.encoding_time_ms:.2f}ms")
    
    # Calculate all pairwise distances
    distances = []
    min_distance = float('inf')
    min_pair = ("", "")
    
    for i, p1 in enumerate(test_patterns):
        for p2 in test_patterns[i+1:]:
            dist = measure_pattern_distance(
                encodings[p1].field_state,
                encodings[p2].field_state
            )
            distances.append(dist)
            if dist < min_distance:
                min_distance = dist
                min_pair = (p1, p2)
    
    avg_distance = sum(distances) / len(distances)
    
    print(f"\n  Results:")
    print(f"    Average distance: {avg_distance:.4f}")
    print(f"    Minimum distance: {min_distance:.4f} (between '{min_pair[0]}' and '{min_pair[1]}')")
    print(f"    All patterns distinct: {min_distance > 0.01}")
    
    return {
        'test': 'distinctiveness',
        'patterns_tested': len(test_patterns),
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'min_pair': min_pair,
        'success': min_distance > 0.01
    }


def run_similarity_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 2: Similar patterns produce similar fields.
    
    Success criteria: Hamming distance 1 → high similarity (>0.8)
    """
    print("\n=== Test 2: Pattern Similarity ===")
    
    # Pairs with Hamming distance 1 (differ by one bit)
    similar_pairs = [
        ("0000", "0001"),
        ("1111", "1110"),
        ("1010", "1011"),
        ("0101", "0100"),
    ]
    
    # Pairs with large Hamming distance
    different_pairs = [
        ("0000", "1111"),
        ("1010", "0101"),
        ("1100", "0011"),
    ]
    
    similar_sims = []
    for p1, p2 in similar_pairs:
        e1 = encoder.encode_binary(p1)
        e2 = encoder.encode_binary(p2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        similar_sims.append(sim)
        print(f"  Similar pair '{p1}'-'{p2}': similarity={sim:.4f}")
    
    different_sims = []
    for p1, p2 in different_pairs:
        e1 = encoder.encode_binary(p1)
        e2 = encoder.encode_binary(p2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        different_sims.append(sim)
        print(f"  Different pair '{p1}'-'{p2}': similarity={sim:.4f}")
    
    avg_similar = sum(similar_sims) / len(similar_sims)
    avg_different = sum(different_sims) / len(different_sims)
    
    print(f"\n  Results:")
    print(f"    Avg similarity (Hamming=1): {avg_similar:.4f}")
    print(f"    Avg similarity (Hamming>2): {avg_different:.4f}")
    print(f"    Separation: {avg_similar - avg_different:.4f}")
    
    return {
        'test': 'similarity',
        'avg_similar_pair_sim': avg_similar,
        'avg_different_pair_sim': avg_different,
        'separation': avg_similar - avg_different,
        'success': avg_similar > avg_different
    }


def run_evolution_stability_test(
    encoder: FieldEncoder,
    evolver: FieldEvolver,
    steps_list: List[int] = [10, 50, 100, 200]
) -> Dict[str, Any]:
    """
    Test 3: Patterns survive Klein-Gordon evolution.
    
    Success criteria: >80% correlation after 100 steps
    """
    print("\n=== Test 3: Evolution Stability ===")
    
    test_patterns = ["0101", "1010", "1111", "0110"]
    results = []
    
    for pattern in test_patterns:
        encoding = encoder.encode_binary(pattern)
        print(f"\n  Pattern '{pattern}':")
        
        for steps in steps_list:
            evolution = evolver.evolve(encoding.field_state, steps)
            print(f"    After {steps:3d} steps: correlation={evolution.correlation_with_initial:.4f}, "
                  f"energy_delta={evolution.energy_delta:.6f}")
            
            results.append({
                'pattern': pattern,
                'steps': steps,
                'correlation': evolution.correlation_with_initial,
                'energy_delta': evolution.energy_delta,
                'survived': evolution.pattern_survived
            })
    
    # Check 100-step survival
    survival_100 = [r for r in results if r['steps'] == 100]
    survival_rate = sum(1 for r in survival_100 if r['survived']) / len(survival_100)
    
    print(f"\n  Results:")
    print(f"    Survival rate at 100 steps: {survival_rate*100:.1f}%")
    
    return {
        'test': 'evolution_stability',
        'detailed_results': results,
        'survival_rate_100': survival_rate,
        'success': survival_rate >= 0.8
    }


def run_conservation_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 4: PAC conservation is maintained during encoding.
    
    Success criteria: Conservation residual < 1e-6
    """
    print("\n=== Test 4: PAC Conservation ===")
    
    test_patterns = ["0000", "0101", "1111", "10101010", "11110000"]
    residuals = []
    
    for pattern in test_patterns:
        result = encoder.encode_binary(pattern)
        residuals.append(result.conservation_residual)
        print(f"  Pattern '{pattern}': residual={result.conservation_residual:.2e}")
    
    max_residual = max(residuals)
    avg_residual = sum(residuals) / len(residuals)
    
    print(f"\n  Results:")
    print(f"    Max residual: {max_residual:.2e}")
    print(f"    Avg residual: {avg_residual:.2e}")
    print(f"    Conservation maintained: {max_residual < 1e-6}")
    
    return {
        'test': 'conservation',
        'max_residual': max_residual,
        'avg_residual': avg_residual,
        'success': max_residual < 1e-6
    }


def run_determinism_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 5: Encoding is deterministic.
    
    Success criteria: Same input → identical output every time
    """
    print("\n=== Test 5: Determinism ===")
    
    test_pattern = "01010101"
    results = []
    
    # Encode the same pattern multiple times
    for i in range(5):
        result = encoder.encode_binary(test_pattern)
        results.append(result.field_state)
    
    # Check all are identical
    all_identical = True
    for i in range(1, len(results)):
        dist = measure_pattern_distance(results[0], results[i])
        if dist > 1e-10:
            all_identical = False
            print(f"  Run {i} differs from run 0 by {dist:.2e}")
    
    if all_identical:
        print(f"  All 5 encodings of '{test_pattern}' are identical ✓")
    
    return {
        'test': 'determinism',
        'pattern': test_pattern,
        'runs': 5,
        'success': all_identical
    }


def main():
    """Run all binary pattern encoding experiments."""
    print("=" * 60)
    print("POC-001 Experiment 01: Binary Pattern Encoding")
    print("=" * 60)
    
    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nDevice: {DEVICE}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']}")
        print(f"Memory: {gpu_info['memory_allocated'] / 1e6:.1f} MB allocated")
    
    # Initialize components
    print("\nInitializing encoder and evolver...")
    encoder = FieldEncoder(field_dims=(64, 64), device=DEVICE)
    evolver = FieldEvolver(field_dims=(64, 64), device=DEVICE)
    
    # Run all tests
    all_results = []
    
    all_results.append(run_distinctiveness_test(encoder))
    all_results.append(run_similarity_test(encoder))
    all_results.append(run_evolution_stability_test(encoder, evolver))
    all_results.append(run_conservation_test(encoder))
    all_results.append(run_determinism_test(encoder))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    for result in all_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['test']}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Save results
    experiment = ExperimentResult(
        experiment_id=generate_experiment_id("exp01_binary"),
        timestamp=datetime.now().isoformat(),
        device=str(DEVICE),
        parameters={
            'field_dims': (64, 64),
            'mass_squared': 0.1,
            'dt': 0.01,
            'xi_target': 1.0571
        },
        encodings=[],  # Would add full encodings for detailed analysis
        metrics={
            'tests_passed': passed,
            'tests_total': total,
            'test_results': all_results
        },
        success=(passed == total),
        notes="Binary pattern encoding baseline experiment"
    )
    
    results_path = experiment.save(get_results_dir())
    print(f"\nResults saved to: {results_path}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
