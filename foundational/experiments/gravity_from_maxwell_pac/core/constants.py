#!/usr/bin/env python3
"""
constants.py - Shared constants for gravity from Maxwell derivation

Combines physical constants with Fibonacci structure.
"""

import numpy as np
from functools import lru_cache

# =============================================================================
# FIBONACCI
# =============================================================================

@lru_cache(maxsize=500)
def fib(n: int) -> int:
    """Fibonacci number F_n."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)

# Key Fibonacci numbers
F3 = fib(3)   # 2
F4 = fib(4)   # 3
F7 = fib(7)   # 13
F10 = fib(10) # 55

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...

# Balance operator
# DERIVATION (2026-01-19): Ξ - 1 = π/55 proven from PAC collapse dynamics
#   within = 2√(r(1-r)) - 1 = -0.0283 per level (r = 1/φ)
#   cross = +0.0854 per level, net = π/55 per level
#   Trace: oscillation_attractor_dynamics/scripts/exp_24_comprehensive_validation.py
XI = 1 + np.pi / 55  # ≈ 1.0571 (DERIVED, not curve-fit)

# =============================================================================
# GRAVITY DEPTH
# =============================================================================

# The gravity Fibonacci index
GRAVITY_DEPTH = 183  # = F7² + F7 + 1

def verify_183():
    """Verify 183 = F₇² + F₇ + 1."""
    return F7**2 + F7 + 1 == 183

# F₁₈₃ using Binet's formula (too large for exact integer)
def log10_fib(k: int) -> float:
    """Log₁₀ of F_k using Binet approximation."""
    return k * np.log10(PHI) - 0.5 * np.log10(5)

LOG10_F183 = log10_fib(183)  # ≈ 38.1

# =============================================================================
# PHYSICAL CONSTANTS (SI)
# =============================================================================

# Electromagnetic
C = 299792458  # m/s (exact)
EPSILON_0 = 8.8541878128e-12  # F/m
MU_0 = 1.25663706212e-6  # H/m
ALPHA_EM = 7.2973525693e-3  # fine structure constant

# Gravitational
G = 6.67430e-11  # m³/(kg·s²)
HBAR = 1.054571817e-34  # J·s

# Masses
M_PLANCK = np.sqrt(HBAR * C / G)  # ≈ 2.176e-8 kg
M_PROTON = 1.67262192e-27  # kg
M_ELECTRON = 9.1093837e-31  # kg

# Derived
PLANCK_LENGTH = np.sqrt(HBAR * G / C**3)  # ≈ 1.616e-35 m
PLANCK_TIME = PLANCK_LENGTH / C

# =============================================================================
# HIERARCHY RATIOS
# =============================================================================

# EM/gravity hierarchy
MASS_HIERARCHY = (M_PLANCK / M_PROTON)**2  # ≈ 1.7e38
COUPLING_HIERARCHY = ALPHA_EM / (G * M_PROTON**2 / (HBAR * C))  # dimensionless

# =============================================================================
# MED BOUNDS
# =============================================================================

MED_MAX_DEPTH = 2
MED_MAX_NODES = 3

# =============================================================================
# UTILITY
# =============================================================================

def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[{status}] {name}")
    if details:
        print(f"        {details}")
