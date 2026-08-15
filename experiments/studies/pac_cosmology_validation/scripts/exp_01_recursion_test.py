"""
Experiment 01: PAC Recursion Load-Bearing Test

Tests whether φ = 1.618... is structurally necessary for PAC.

HYPOTHESIS: The PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2) only holds
for φ = (1+√5)/2. Any other value should produce residual errors.

If the framework "works" with arbitrary φ, then φ is NOT load-bearing
and PAC has no structural content beyond curve fitting.

METHODOLOGY:
1. Test PAC recursion with true φ
2. Test PAC recursion with alternative φ values
3. Verify true φ satisfies all identities (φ²=φ+1, 1/φ+1/φ²=1)
4. Verify alternatives FAIL these identities

SUCCESS CRITERION: Only φ = 1.618... passes all tests.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_constraints import (
    verify_pac_recursion,
    verify_fibonacci_identity,
    verify_sum_identity,
    test_phi_is_loadbearing,
    check_hierarchy_consistency
)
from core.constants import PHI, PHI_SQUARED, M_GALAXY_SCALE


def run_experiment():
    """Run Experiment 01: PAC Recursion Load-Bearing Test."""
    
    print("=" * 70)
    print("EXPERIMENT 01: PAC RECURSION LOAD-BEARING TEST")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "experiment": "exp_01_recursion_test",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "PAC recursion only holds for φ = (1+√5)/2"
    }
    
    # =================================================================
    # Test 1: Verify true φ satisfies PAC recursion
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 1: Does true φ satisfy PAC recursion?")
    print("-" * 50)
    
    true_phi_result = verify_pac_recursion(PHI, depth=15)
    
    print(f"\nφ = {PHI:.10f}")
    print(f"Recursion error: {true_phi_result.recursion_error:.2e}")
    print(f"Max level error: {true_phi_result.max_level_error:.2e}")
    print(f"Valid: {true_phi_result.is_valid}")
    print(f"\nResult: {true_phi_result.interpretation}")
    
    results["test_1_true_phi"] = {
        "phi": PHI,
        "recursion_error": true_phi_result.recursion_error,
        "is_valid": true_phi_result.is_valid
    }
    
    # =================================================================
    # Test 2: Verify identities
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 2: Verify φ identities")
    print("-" * 50)
    
    fib_ok, fib_err = verify_fibonacci_identity(PHI)
    sum_ok, sum_err = verify_sum_identity(PHI)
    
    print(f"\nφ² = φ + 1:")
    print(f"  φ² = {PHI**2:.10f}")
    print(f"  φ + 1 = {PHI + 1:.10f}")
    print(f"  Error: {fib_err:.2e}")
    print(f"  Valid: {fib_ok}")
    
    print(f"\n1/φ + 1/φ² = 1:")
    print(f"  1/φ = {1/PHI:.10f}")
    print(f"  1/φ² = {1/PHI**2:.10f}")
    print(f"  Sum = {1/PHI + 1/PHI**2:.10f}")
    print(f"  Error: {sum_err:.2e}")
    print(f"  Valid: {sum_ok}")
    
    results["test_2_identities"] = {
        "fibonacci_valid": fib_ok,
        "fibonacci_error": fib_err,
        "sum_valid": sum_ok,
        "sum_error": sum_err
    }
    
    # =================================================================
    # Test 3: Alternative φ values FAIL
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 3: Do alternative φ values break PAC recursion?")
    print("-" * 50)
    
    alternatives = [1.2, 1.4, 1.5, 1.55, 1.6, 1.62, 1.65, 1.7, 2.0]
    alt_results = []
    
    print(f"\n{'φ':<12} {'Recur.Err':<15} {'Fib.Err':<15} {'Sum.Err':<15} {'All Valid?':<12}")
    print("-" * 69)
    
    for phi_alt in alternatives:
        rec = verify_pac_recursion(phi_alt, depth=15)
        fib_ok_alt, fib_err_alt = verify_fibonacci_identity(phi_alt)
        sum_ok_alt, sum_err_alt = verify_sum_identity(phi_alt)
        
        all_valid = rec.is_valid and fib_ok_alt and sum_ok_alt
        valid_str = "✓" if all_valid else "✗"
        
        print(f"{phi_alt:<12.4f} {rec.recursion_error:<15.2e} {fib_err_alt:<15.2e} {sum_err_alt:<15.2e} {valid_str:<12}")
        
        alt_results.append({
            "phi": phi_alt,
            "recursion_error": rec.recursion_error,
            "fibonacci_error": fib_err_alt,
            "sum_error": sum_err_alt,
            "all_valid": all_valid
        })
    
    # Check that all alternatives fail
    all_alts_fail = all(not r["all_valid"] for r in alt_results)
    
    print(f"\nAll alternatives fail: {all_alts_fail}")
    
    results["test_3_alternatives"] = {
        "tested": len(alternatives),
        "all_fail": all_alts_fail,
        "details": alt_results
    }
    
    # =================================================================
    # Test 4: Full load-bearing test
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 4: Complete load-bearing assessment")
    print("-" * 50)
    
    loadbearing = test_phi_is_loadbearing()
    
    print(f"\nTrue φ works: {loadbearing['true_phi_works']}")
    print(f"All others fail: {loadbearing['all_others_fail']}")
    print(f"\n{loadbearing['interpretation']}")
    
    results["test_4_loadbearing"] = {
        "is_loadbearing": loadbearing["is_loadbearing"],
        "true_phi_works": loadbearing["true_phi_works"],
        "all_others_fail": loadbearing["all_others_fail"]
    }
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 01 SUMMARY")
    print("=" * 70)
    
    all_tests_pass = (
        true_phi_result.is_valid and
        fib_ok and sum_ok and
        all_alts_fail and
        loadbearing["is_loadbearing"]
    )
    
    results["success"] = all_tests_pass
    results["conclusion"] = (
        "φ = 1.618... is STRUCTURALLY NECESSARY for PAC recursion"
        if all_tests_pass else
        "WARNING: φ may not be uniquely load-bearing"
    )
    
    print(f"\n{'Test':<40} {'Result':<10}")
    print("-" * 50)
    print(f"{'True φ satisfies recursion':<40} {'✓' if true_phi_result.is_valid else '✗':<10}")
    print(f"{'φ² = φ + 1 verified':<40} {'✓' if fib_ok else '✗':<10}")
    print(f"{'1/φ + 1/φ² = 1 verified':<40} {'✓' if sum_ok else '✗':<10}")
    print(f"{'All alternative φ fail':<40} {'✓' if all_alts_fail else '✗':<10}")
    print(f"{'φ is load-bearing':<40} {'✓' if loadbearing['is_loadbearing'] else '✗':<10}")
    
    print(f"\n{'Overall Success':<40} {'✓' if all_tests_pass else '✗':<10}")
    print(f"\nConclusion: {results['conclusion']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_01_recursion_test_{timestamp}.json"
    
    # Custom JSON encoder for numpy types
    def json_serializer(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_dir / filename, "w") as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    print(f"\nResults saved to: results/{filename}")
    
    return results


if __name__ == "__main__":
    run_experiment()
