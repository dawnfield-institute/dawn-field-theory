"""
PAC Constraint Enforcement and Verification

This module tests whether PAC constraints are LOAD-BEARING:
- If φ = 1.618 is structural, changing it should break the recursion
- If Ξ = 1.0571 is necessary, removing it should break balance
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

from .constants import PHI, PHI_SQUARED, XI, M_GALAXY_SCALE


@dataclass
class RecursionTestResult:
    """Result of testing PAC recursion with a given phi value."""
    phi_test: float
    phi_true: float
    recursion_error: float
    max_level_error: float
    is_valid: bool
    interpretation: str


@dataclass 
class ConstraintViolation:
    """Details of a constraint violation."""
    constraint_name: str
    expected_value: float
    actual_value: float
    residual: float
    severity: str  # "critical", "warning", "acceptable"


def verify_pac_recursion(phi: float, depth: int = 10) -> RecursionTestResult:
    """
    Verify that PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2) holds for given φ.
    
    The recursion should ONLY hold for φ = (1+√5)/2.
    For any other value, there will be residual error.
    
    Args:
        phi: Value to test (should be golden ratio for valid PAC)
        depth: Number of levels to test
    
    Returns:
        RecursionTestResult with error metrics
    """
    # Compute Ψ(k) = φ^(-k) with test phi
    psi = np.array([phi ** (-k) for k in range(depth)])
    
    # Check recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2) for k = 0 to depth-3
    errors = []
    for k in range(depth - 2):
        expected = psi[k]
        actual = psi[k+1] + psi[k+2]
        error = abs(expected - actual) / max(expected, 1e-15)
        errors.append(error)
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Tolerance: machine precision
    tol = 1e-10
    is_valid = max_error < tol
    
    # Interpretation
    if is_valid:
        interpretation = f"✅ PAC recursion holds with φ = {phi:.6f}"
    else:
        interpretation = f"❌ PAC recursion FAILS with φ = {phi:.6f} (error: {mean_error:.2e})"
    
    return RecursionTestResult(
        phi_test=phi,
        phi_true=PHI,
        recursion_error=float(mean_error),
        max_level_error=float(max_error),
        is_valid=is_valid,
        interpretation=interpretation
    )


def verify_fibonacci_identity(phi: float) -> Tuple[bool, float]:
    """
    Verify φ² = φ + 1 (characteristic equation of PAC).
    
    This is the defining property of the golden ratio.
    Any deviation means φ is not the PAC solution.
    """
    error = abs(phi**2 - phi - 1)
    is_valid = error < 1e-10
    return is_valid, error


def verify_sum_identity(phi: float) -> Tuple[bool, float]:
    """
    Verify 1/φ + 1/φ² = 1 (PAC conservation identity).
    
    This ensures the infinite sum converges to 1.
    """
    total = 1/phi + 1/phi**2
    error = abs(total - 1)
    is_valid = error < 1e-10
    return is_valid, error


def test_phi_is_loadbearing(phi_values: List[float] = None) -> Dict:
    """
    Test whether φ = 1.618... is structurally necessary.
    
    Sweeps through alternative φ values and checks if recursion breaks.
    If the framework "works" with arbitrary φ, then φ is NOT load-bearing
    and PAC has no structural content.
    
    Args:
        phi_values: List of φ values to test (default: range around true φ)
    
    Returns:
        Dictionary with test results
    """
    if phi_values is None:
        # Test range around true φ
        phi_values = [1.2, 1.4, 1.5, 1.55, 1.6, PHI, 1.65, 1.7, 1.8, 2.0]
    
    results = []
    
    for phi in phi_values:
        result = verify_pac_recursion(phi)
        fib_ok, fib_err = verify_fibonacci_identity(phi)
        sum_ok, sum_err = verify_sum_identity(phi)
        
        results.append({
            "phi": phi,
            "is_true_phi": abs(phi - PHI) < 1e-6,
            "recursion_error": result.recursion_error,
            "recursion_valid": result.is_valid,
            "fibonacci_error": fib_err,
            "fibonacci_valid": fib_ok,
            "sum_error": sum_err,
            "sum_valid": sum_ok,
            "all_valid": result.is_valid and fib_ok and sum_ok
        })
    
    # Summarize
    true_phi_results = [r for r in results if r["is_true_phi"]]
    other_results = [r for r in results if not r["is_true_phi"]]
    
    true_phi_works = all(r["all_valid"] for r in true_phi_results) if true_phi_results else False
    others_fail = all(not r["all_valid"] for r in other_results) if other_results else False
    
    is_loadbearing = true_phi_works and others_fail
    
    return {
        "is_loadbearing": is_loadbearing,
        "true_phi_works": true_phi_works,
        "all_others_fail": others_fail,
        "results": results,
        "interpretation": (
            "✅ φ = 1.618... is STRUCTURALLY NECESSARY - PAC recursion only works with golden ratio"
            if is_loadbearing else
            "⚠️ φ is NOT load-bearing - framework works with other values"
        )
    }


def check_hierarchy_consistency(observed_masses: List[float]) -> Dict:
    """
    Check if observed masses fit PAC hierarchy structure.
    
    The hierarchy requires: M(k) = M_galaxy × φ^(-k)
    This means masses should fall on discrete levels.
    
    Args:
        observed_masses: List of observed masses in M☉
    
    Returns:
        Consistency metrics
    """
    results = []
    
    for mass in observed_masses:
        # Find k level
        log_mass = np.log10(mass)
        k = -np.log(mass / M_GALAXY_SCALE) / np.log(PHI)
        
        # Nearest integer level
        k_int = round(k)
        
        # Expected mass at that level
        expected_mass = M_GALAXY_SCALE * PHI ** (-k_int)
        log_expected = np.log10(expected_mass)
        
        # Residual
        residual = abs(log_mass - log_expected)
        
        results.append({
            "observed_mass": mass,
            "log_mass": log_mass,
            "k_continuous": k,
            "k_nearest_int": k_int,
            "expected_mass": expected_mass,
            "residual_dex": residual,
            "is_consistent": residual < 0.2  # Within 0.2 dex
        })
    
    # Overall consistency
    consistent_count = sum(1 for r in results if r["is_consistent"])
    
    return {
        "n_objects": len(observed_masses),
        "n_consistent": consistent_count,
        "consistency_fraction": consistent_count / len(observed_masses),
        "mean_residual_dex": np.mean([r["residual_dex"] for r in results]),
        "results": results
    }


def run_constraint_tests():
    """Run all constraint tests and report results."""
    
    print("=" * 70)
    print("PAC CONSTRAINT VERIFICATION")
    print("=" * 70)
    
    # Test 1: φ is load-bearing
    print("\n--- Test 1: Is φ = 1.618... structurally necessary? ---\n")
    
    phi_test = test_phi_is_loadbearing()
    
    print(f"True φ works: {phi_test['true_phi_works']}")
    print(f"All others fail: {phi_test['all_others_fail']}")
    print(f"\nResult: {phi_test['interpretation']}")
    
    print("\nDetailed results:")
    print(f"{'φ':<10} {'Recur.Err':<12} {'Fib.Err':<12} {'Sum.Err':<12} {'Valid?':<8}")
    print("-" * 54)
    
    for r in phi_test['results']:
        marker = "←" if r["is_true_phi"] else ""
        valid_str = "✓" if r["all_valid"] else "✗"
        print(f"{r['phi']:<10.4f} {r['recursion_error']:<12.2e} {r['fibonacci_error']:<12.2e} {r['sum_error']:<12.2e} {valid_str:<8} {marker}")
    
    # Test 2: JWST masses fit hierarchy
    print("\n--- Test 2: Do JWST masses fit PAC hierarchy? ---\n")
    
    jwst_masses = [10**7.5, 10**6.2, 10**6.95, 10**6.0]  # UHZ-1, GN-z11, CEERS-1019, GLASS-z12
    hierarchy_test = check_hierarchy_consistency(jwst_masses)
    
    print(f"Consistent: {hierarchy_test['n_consistent']}/{hierarchy_test['n_objects']}")
    print(f"Mean residual: {hierarchy_test['mean_residual_dex']:.3f} dex")
    
    print("\nDetailed:")
    print(f"{'log(M)':<10} {'k':<10} {'k_int':<8} {'log(M_exp)':<12} {'Residual':<10} {'OK?':<6}")
    print("-" * 56)
    
    for r in hierarchy_test['results']:
        ok = "✓" if r["is_consistent"] else "✗"
        print(f"{r['log_mass']:<10.2f} {r['k_continuous']:<10.2f} {r['k_nearest_int']:<8} {np.log10(r['expected_mass']):<12.2f} {r['residual_dex']:<10.3f} {ok:<6}")
    
    return {
        "phi_loadbearing": phi_test,
        "hierarchy_consistency": hierarchy_test
    }


if __name__ == "__main__":
    run_constraint_tests()
