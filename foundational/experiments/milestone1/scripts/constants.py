"""
Milestone 1: Shared Constants and Utilities

All physical constants and Fibonacci utilities used across experiments.
CODATA 2018 values where applicable.
"""

import numpy as np
from typing import List, Tuple, Dict
from functools import lru_cache

# =============================================================================
# MATHEMATICAL CONSTANTS (derived, not fitted)
# =============================================================================

# Golden ratio - THE solution to r² = r + 1
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...
PSI = (1 - np.sqrt(5)) / 2  # -0.6180339887... (conjugate)

# Balance operator Xi - emerges in MED, pre-field, vCPU
# Defined as 1 + π/F₁₀ where F₁₀ = 55 is 10th Fibonacci number
XI = 1 + np.pi / 55  # 1.0571081...

# Fibonacci sequence (first 200 terms computed lazily)
@lru_cache(maxsize=500)
def fib(n: int) -> int:
    """Compute nth Fibonacci number (0-indexed: F₀=0, F₁=1, F₂=1, F₃=2...)"""
    if n < 0:
        raise ValueError("Fibonacci index must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n-1) + fib(n-2)

# Commonly used Fibonacci numbers
F = {i: fib(i) for i in range(20)}
# F[1]=1, F[2]=1, F[3]=2, F[4]=3, F[5]=5, F[6]=8, F[7]=13, F[8]=21, F[9]=34, F[10]=55...

# Key Fibonacci values for physics
F3 = 2   # Binary splitting
F4 = 3   # Spatial dimensions, SU(2)
F5 = 5   # Pentagon symmetry
F6 = 8   # SU(3) gluons, cube structure
F7 = 13  # Gauge closure
F8 = 21  # Extended structure
F9 = 34  # 
F10 = 55 # EM recursion depth, edge-of-chaos

# Aliases for compatibility
F_3, F_4, F_5, F_6, F_7, F_8, F_9, F_10 = F3, F4, F5, F6, F7, F8, F9, F10

# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================

# Fine structure constant
ALPHA_MEASURED = 7.2973525693e-3  # CODATA 2018
ALPHA_UNCERTAINTY = 0.0000000011e-3

# Weinberg angle
SIN2_THETA_W_MEASURED = 0.23121  # at M_Z scale
SIN2_THETA_W_UNCERTAINTY = 0.00004

# Speed of light (exact by definition)
C_EXACT = 299792458  # m/s

# Elementary charge
E_CHARGE_MEASURED = 1.602176634e-19  # C (exact by definition since 2019)

# Planck constant
H_PLANCK = 6.62607015e-34  # J·s (exact by definition)
HBAR = H_PLANCK / (2 * np.pi)

# Gravitational constant
G_NEWTON = 6.67430e-11  # m³/(kg·s²)
G_UNCERTAINTY = 0.00015e-11

# Planck mass
M_PLANCK_KG = np.sqrt(HBAR * C_EXACT / G_NEWTON)  # kg
M_PLANCK_GEV = 1.22089e19  # GeV

# Lepton masses (MeV/c²)
M_ELECTRON = 0.51099895000
M_MUON = 105.6583755
M_TAU = 1776.86

# =============================================================================
# PAC/SEC DERIVED CONSTANTS
# =============================================================================

# Balance operator (phenomenological - see FALSIFICATION_REGISTRY.md)
XI_FORMULA = 1 + np.pi / 55  # 1.0571...
XI_EMPIRICAL = 1.0571  # From Navier-Stokes symbolic engine

# =============================================================================
# KEY DERIVED FORMULAS
# =============================================================================

def alpha_pac() -> float:
    """
    Fine structure constant from PAC/Fibonacci.
    
    α = (F₃/(F₄·φ·F₁₀)) × (1 - F₁₀/(4π·F₇²))
    
    Returns:
        Predicted α value
    """
    term1 = F_3 / (F_4 * PHI * F_10)
    correction = 1 - F_10 / (4 * np.pi * F_7**2)
    return term1 * correction

def sin2_theta_w_pac() -> float:
    """
    Weinberg angle from Fibonacci ratio.
    
    sin²θ_W = F₄/F₇ = 3/13
    
    Returns:
        Predicted sin²θ_W value
    """
    return F_4 / F_7

def koide_q() -> float:
    """
    Koide ratio for charged leptons.
    
    Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²
    
    Returns:
        Measured Koide Q value
    """
    numerator = M_ELECTRON + M_MUON + M_TAU
    sqrt_sum = np.sqrt(M_ELECTRON) + np.sqrt(M_MUON) + np.sqrt(M_TAU)
    return numerator / sqrt_sum**2

def koide_pac() -> float:
    """
    Koide ratio from Fibonacci.
    
    Q = F₃/F₄ = 2/3
    
    Returns:
        Predicted Koide Q value
    """
    return F_3 / F_4

# =============================================================================
# FIBONACCI UTILITIES
# =============================================================================

def is_fibonacci(n: int) -> bool:
    """Check if n is a Fibonacci number."""
    if n < 0:
        return False
    # n is Fibonacci iff 5n² ± 4 is a perfect square
    test1 = 5 * n * n + 4
    test2 = 5 * n * n - 4
    return is_perfect_square(test1) or is_perfect_square(test2)

def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = int(np.sqrt(n))
    return root * root == n

def fibonacci_index(n: int) -> int:
    """Find index k such that F_k = n, or -1 if not Fibonacci."""
    if not is_fibonacci(n):
        return -1
    k = 0
    while fib(k) < n:
        k += 1
    return k if fib(k) == n else -1

def nearest_fibonacci(n: int) -> Tuple[int, int]:
    """Find nearest Fibonacci number and its index."""
    k = 0
    while fib(k) < n:
        k += 1
    if k == 0:
        return fib(0), 0
    lower, upper = fib(k-1), fib(k)
    if n - lower <= upper - n:
        return lower, k-1
    return upper, k

def fib_large(n: int) -> float:
    """Compute F_n for large n using Binet's formula."""
    return (PHI**n - PSI**n) / np.sqrt(5)

def fib_approx(n: int) -> float:
    """Approximate F_n ≈ φⁿ/√5 for large n."""
    return PHI**n / np.sqrt(5)

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def percent_error(predicted: float, measured: float) -> float:
    """Calculate percent error."""
    return abs(predicted - measured) / measured * 100

def sigma_deviation(predicted: float, measured: float, uncertainty: float) -> float:
    """Calculate number of standard deviations from measured value."""
    return abs(predicted - measured) / uncertainty

def validate_result(name: str, predicted: float, measured: float, 
                   uncertainty: float = None, threshold: float = 1.0) -> Dict:
    """
    Validate a predicted value against measurement.
    
    Args:
        name: Name of quantity
        predicted: PAC prediction
        measured: Measured value
        uncertainty: Measurement uncertainty (optional)
        threshold: Percent error threshold for pass/fail
        
    Returns:
        Dict with validation results
    """
    error_pct = percent_error(predicted, measured)
    passed = error_pct < threshold
    
    result = {
        "name": name,
        "predicted": predicted,
        "measured": measured,
        "error_percent": error_pct,
        "passed": passed,
        "threshold": threshold
    }
    
    if uncertainty is not None:
        sigma = sigma_deviation(predicted, measured, uncertainty)
        result["uncertainty"] = uncertainty
        result["sigma"] = sigma
    
    return result

# =============================================================================
# PRINTING UTILITIES
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_subheader(title: str, width: int = 70):
    """Print subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width + "\n")

def print_result(result_or_name, passed=None):
    """Print validation result. Accepts dict or (name, bool)."""
    if isinstance(result_or_name, dict):
        result = result_or_name
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{result['name']}: {status}")
        print(f"  Predicted: {result['predicted']:.10g}")
        print(f"  Measured:  {result['measured']:.10g}")
        print(f"  Error:     {result['error_percent']:.6f}%")
        if "sigma" in result:
            print(f"  Deviation: {result['sigma']:.2f}σ")
        print()
    else:
        # Simple (name, bool) form
        name = result_or_name
        status = "✅ VALIDATED" if passed else "❌ FAILED"
        print(f"\n{name}: {status}")

# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print_header("MILESTONE 1: CONSTANTS VERIFICATION")
    
    print("Golden Ratio:")
    print(f"  φ = {PHI:.15f}")
    print(f"  1/φ = {1/PHI:.15f}")
    print(f"  φ - 1 = {PHI - 1:.15f}")
    print(f"  φ² - φ - 1 = {PHI**2 - PHI - 1:.2e} (should be ~0)")
    
    print("\nFibonacci Sequence (F₁ to F₁₀):")
    for i in range(1, 11):
        print(f"  F_{i} = {fib(i)}")
    
    print("\nKey Physics Values:")
    print(f"  α (PAC):    {alpha_pac():.10f}")
    print(f"  α (CODATA): {ALPHA_MEASURED:.10f}")
    print(f"  Error:      {percent_error(alpha_pac(), ALPHA_MEASURED):.6f}%")
    
    print(f"\n  sin²θ_W (PAC):    {sin2_theta_w_pac():.10f}")
    print(f"  sin²θ_W (measured): {SIN2_THETA_W_MEASURED:.10f}")
    print(f"  Error:              {percent_error(sin2_theta_w_pac(), SIN2_THETA_W_MEASURED):.4f}%")
    
    print(f"\n  Koide Q (measured): {koide_q():.10f}")
    print(f"  Koide Q (PAC):      {koide_pac():.10f}")
    print(f"  Error:              {percent_error(koide_pac(), koide_q()):.6f}%")
    
    print("\nLarge Fibonacci (gravity scale):")
    print(f"  F₁₈₃ ≈ {fib_approx(183):.3e}")
    print(f"  M_P² ≈ {M_PLANCK_GEV**2:.3e} GeV²")
