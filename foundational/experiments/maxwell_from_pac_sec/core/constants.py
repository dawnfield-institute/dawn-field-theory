#!/usr/bin/env python3
"""
PAC/SEC Constants for Maxwell Derivation

This module defines fundamental constants from Dawn Field Theory
used in deriving Maxwell's equations from information dynamics.

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 15, 2026
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0, hbar, e, pi
from scipy.constants import physical_constants, alpha as alpha_em

# =============================================================================
# Golden Ratio and Fibonacci
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
PHI_INV = 1 / PHI           # 1/φ ≈ 0.618033988749895

# Fibonacci sequence (F_0 through F_14)
FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

def fibonacci(n: int) -> int:
    """Get nth Fibonacci number (0-indexed: F_0=0, F_1=1, F_2=1, ...)"""
    if n < len(FIB):
        return FIB[n]
    a, b = FIB[-2], FIB[-1]
    for _ in range(n - len(FIB) + 1):
        a, b = b, a + b
    return b

# Key Fibonacci numbers
F_2 = FIB[2]   # 1 - U(1)_EM
F_4 = FIB[4]   # 3 - SU(2)_L
F_6 = FIB[6]   # 8 - SU(3)_c
F_7 = FIB[7]   # 13 - Total gauge content
F_10 = FIB[10] # 55 - Xi balance depth

# =============================================================================
# PAC Balance Operators
# =============================================================================

# The balance operator Ξ (Xi)
XI = 1 + pi / 55  # ≈ 1.05712856964...

# Xi bounds
XI_MIN = 1.0015   # Ground state (minimal inflation)
XI_PAC = XI       # PAC collapse point
XI_MEAN = np.sqrt(XI_PAC * XI_MIN)  # Geometric mean ≈ 1.0289

# =============================================================================
# SEC Parameters
# =============================================================================

# SEC equation: ∂S/∂t = α∇I - β∇H
# Coupled: ∂I/∂t = γ∇S, ∂H/∂t = -δ∇S
# Wave speed: c² = αγ + βδ

# Hypothesis 1: Symmetric (α=β, γ=δ)
ALPHA_SYM = c / np.sqrt(2)
GAMMA_SYM = c / np.sqrt(2)

# Hypothesis 2: Xi-balanced (α/β = Ξ)
def xi_balanced_params():
    """SEC parameters with α/β = Ξ"""
    # c² = αγ + βδ = βΞγ + βδ = β(Ξγ + δ)
    # If γ = δ: c² = βγ(Ξ + 1)
    # Let v₀ = √(c²/(Ξ+1))
    v0 = np.sqrt(c**2 / (XI + 1))
    return {
        'alpha': XI * v0,
        'beta': v0,
        'gamma': v0,
        'delta': v0
    }

# Hypothesis 3: Golden ratio (α/γ = φ)
def phi_structured_params():
    """SEC parameters with α/γ = φ"""
    # More complex relationship
    # Placeholder for derivation
    return {
        'alpha': c * np.sqrt(PHI / (1 + PHI)),
        'beta': c * np.sqrt(1 / (PHI * (1 + PHI))),
        'gamma': c * np.sqrt(1 / (PHI * (1 + PHI))),
        'delta': c * np.sqrt(PHI / (1 + PHI))
    }

# =============================================================================
# Electromagnetic Constants (for comparison)
# =============================================================================

# Speed of light
C = c  # 299,792,458 m/s

# Fine structure constant
ALPHA_EM = alpha_em  # ≈ 1/137.036

# Vacuum permittivity and permeability
EPSILON_0 = epsilon_0  # ≈ 8.854e-12 F/m
MU_0 = mu_0           # = 4π×10⁻⁷ H/m (exact by definition)

# Elementary charge
E_CHARGE = e  # ≈ 1.602e-19 C

# =============================================================================
# Fibonacci Gauge Predictions
# =============================================================================

# Weinberg angle from Fibonacci
SIN2_THETA_W_PAC = F_4 / F_7  # 3/13 ≈ 0.2308
SIN2_THETA_W_MEASURED = 0.23121

# Fine structure constant from Fibonacci (with correction)
# α ≈ F₃/(F₄·φ·F₁₀)·(1 - F₁₀/4πF₇²)
ALPHA_PAC_APPROX = (FIB[3] / (FIB[4] * PHI * FIB[10])) * (1 - FIB[10]/(4*pi*FIB[7]**2))

# Strong coupling from Fibonacci
ALPHA_S_PAC = FIB[4] / (2 * PHI * FIB[6])  # ≈ 0.116

# =============================================================================
# MED Bounds
# =============================================================================

MED_MAX_DEPTH = 2
MED_MAX_NODES = 3

# =============================================================================
# Universal Frequency
# =============================================================================

F_UNIVERSAL = 0.03  # Hz - from balance-seeking oscillation

# =============================================================================
# Planck Units (for reference)
# =============================================================================

L_PLANCK = physical_constants['Planck length'][0]  # ~1.616e-35 m
T_PLANCK = physical_constants['Planck time'][0]    # ~5.391e-44 s
M_PLANCK = physical_constants['Planck mass'][0]    # ~2.176e-8 kg

# =============================================================================
# Utility Functions
# =============================================================================

def verify_wave_speed(alpha, beta, gamma, delta):
    """Verify that SEC parameters give correct wave speed."""
    c_sec = np.sqrt(alpha * gamma + beta * delta)
    error = abs(c_sec - c) / c
    return c_sec, error

def phi_power(n: int) -> float:
    """Return φ^n"""
    return PHI ** n

def fib_ratio(i: int, j: int) -> float:
    """Return F_i / F_j"""
    return fibonacci(i) / fibonacci(j)

# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PAC/SEC CONSTANTS MODULE")
    print("=" * 60)
    
    print(f"\nGolden ratio φ = {PHI:.10f}")
    print(f"Balance operator Ξ = {XI:.10f}")
    print(f"Ξ - 1 = π/55 = {XI - 1:.10f}")
    
    print(f"\nKey Fibonacci numbers:")
    print(f"  F_7 = {F_7} (gauge crystallization)")
    print(f"  F_10 = {F_10} (Xi balance depth)")
    
    print(f"\nWeinberg angle:")
    print(f"  PAC: sin²θ_W = {F_4}/{F_7} = {SIN2_THETA_W_PAC:.6f}")
    print(f"  Measured: sin²θ_W = {SIN2_THETA_W_MEASURED:.6f}")
    print(f"  Error: {100*abs(SIN2_THETA_W_PAC - SIN2_THETA_W_MEASURED)/SIN2_THETA_W_MEASURED:.3f}%")
    
    print(f"\nFine structure constant:")
    print(f"  PAC approx: α = {ALPHA_PAC_APPROX:.8f}")
    print(f"  Measured: α = {ALPHA_EM:.8f}")
    print(f"  Error: {100*abs(ALPHA_PAC_APPROX - ALPHA_EM)/ALPHA_EM:.4f}%")
    
    print(f"\nSEC wave speed verification:")
    for name, params in [('Symmetric', {'alpha': ALPHA_SYM, 'beta': ALPHA_SYM, 
                                        'gamma': GAMMA_SYM, 'delta': GAMMA_SYM}),
                        ('Xi-balanced', xi_balanced_params()),
                        ('Phi-structured', phi_structured_params())]:
        c_sec, err = verify_wave_speed(**params)
        print(f"  {name}: c_SEC = {c_sec:.6e} m/s (error: {100*err:.6f}%)")
