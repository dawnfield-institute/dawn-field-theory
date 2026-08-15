#!/usr/bin/env python3
"""
24_pac_bell_corrected.py - Corrected Bell Test for PAC Structure
=================================================================

The previous test had measurement issues. This version properly implements:
1. Fibonacci-weighted Bell states
2. Correct quantum measurement operators
3. Proper CHSH computation

Key insight: PAC conservation Ψ(k) = Ψ(k+1) + Ψ(k+2) creates entanglement
where the LEFT and RIGHT branches are correlated with weights F_{k-1} : F_{k-2}
"""

import numpy as np
from typing import Tuple

print("=" * 78)
print("PAC BELL TEST - CORRECTED QUANTUM MODEL")
print("=" * 78)

phi = (1 + np.sqrt(5)) / 2

def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# ============================================================================
# PROPER QUANTUM MECHANICS FOR ENTANGLED STATES
# ============================================================================

class FibonacciBellState:
    """
    A Bell-like state with Fibonacci-weighted coefficients.
    
    Standard Bell singlet: |ψ⟩ = (|01⟩ - |10⟩)/√2
    Fibonacci Bell state:  |ψ⟩ = (F_L|01⟩ - F_R|10⟩)/N
    
    where F_L = F_{k-1}, F_R = F_{k-2}, and N = √(F_L² + F_R²)
    """
    
    def __init__(self, k: int = 7):
        self.k = k
        self.F_L = fib(k - 1)
        self.F_R = fib(k - 2)
        self.norm = np.sqrt(self.F_L**2 + self.F_R**2)
        
        # Coefficients: |ψ⟩ = α|01⟩ + β|10⟩
        self.alpha = self.F_L / self.norm   # coefficient of |01⟩
        self.beta = -self.F_R / self.norm   # coefficient of |10⟩ (negative for singlet-like)
        
    def correlation(self, theta_a: float, theta_b: float) -> float:
        """
        Compute E(a,b) = ⟨ψ|σ_a ⊗ σ_b|ψ⟩
        
        For spin measurements at angles θ_a and θ_b from z-axis.
        
        For the state |ψ⟩ = α|01⟩ + β|10⟩:
        E(a,b) = -2αβ[cos(θ_a)cos(θ_b) + sin(θ_a)sin(θ_b)]
               = -2αβ cos(θ_a - θ_b)
        
        Wait - that's for |01⟩ + |10⟩. For our anticorrelated state,
        we need the proper quantum calculation.
        """
        # For state |ψ⟩ = α|01⟩ + β|10⟩ where |0⟩=spin up, |1⟩=spin down
        # The correlation function is:
        # E(θ_a, θ_b) = ⟨σ_a · σ_b⟩
        
        # For singlet-like state (α=1/√2, β=-1/√2): E = -cos(θ_a - θ_b)
        # For our Fibonacci state:
        
        # General formula for |ψ⟩ = α|01⟩ + β|10⟩:
        # E(a,b) = |α|²[-cos(θ_a)cos(θ_b) + sin(θ_a)sin(θ_b)cos(φ_a-φ_b)]
        #        + |β|²[cos(θ_a)cos(θ_b) - sin(θ_a)sin(θ_b)cos(φ_a-φ_b)]
        #        + 2Re[α*β] sin(θ_a)sin(θ_b)cos(φ_a-φ_b)
        
        # For measurements in x-z plane (φ=0), this simplifies
        # After careful calculation for |ψ⟩ = α|01⟩ - β|10⟩:
        
        # Actually, let's use the density matrix approach
        # ρ = |ψ⟩⟨ψ| and E(a,b) = Tr[ρ (σ_a ⊗ σ_b)]
        
        alpha, beta = self.alpha, self.beta
        
        # For measurements in x-z plane at angles θ from z:
        # σ_θ = cos(θ)σ_z + sin(θ)σ_x
        
        # The correlation for state α|01⟩ + β|10⟩ is:
        # E(θ_a, θ_b) = -|α|²cos(θ_a-θ_b) - |β|²cos(θ_a-θ_b) + 2αβcos(θ_a+θ_b)
        # Wait, that's not right either...
        
        # Let me just compute it directly.
        # State: |ψ⟩ = α|0⟩_A|1⟩_B + β|1⟩_A|0⟩_B
        
        # Measurement operators:
        # A(θ) = cos(θ)|0⟩⟨0| - cos(θ)|1⟩⟨1| + sin(θ)|0⟩⟨1| + sin(θ)|1⟩⟨0|
        #      = cos(θ)σ_z + sin(θ)σ_x
        
        # E(θ_a, θ_b) = ⟨ψ|(A(θ_a)⊗B(θ_b))|ψ⟩
        
        ca, sa = np.cos(theta_a), np.sin(theta_a)
        cb, sb = np.cos(theta_b), np.sin(theta_b)
        
        # Expand |ψ⟩ = α|01⟩ + β|10⟩
        # (A⊗B)|01⟩ = (ca|0⟩ + sa|1⟩) ⊗ (-cb|1⟩ + sb|0⟩) 
        #           = -ca·cb|01⟩ + ca·sb|00⟩ - sa·cb|11⟩ + sa·sb|10⟩
        # (A⊗B)|10⟩ = (-ca|1⟩ + sa|0⟩) ⊗ (cb|0⟩ + sb|1⟩)
        #           = -ca·cb|10⟩ - ca·sb|11⟩ + sa·cb|00⟩ + sa·sb|01⟩
        
        # E = α*⟨01|(A⊗B)(α|01⟩ + β|10⟩) + β*⟨10|(A⊗B)(α|01⟩ + β|10⟩)
        
        # ⟨01|(A⊗B)|01⟩ = -ca·cb
        # ⟨01|(A⊗B)|10⟩ = sa·sb  
        # ⟨10|(A⊗B)|01⟩ = sa·sb
        # ⟨10|(A⊗B)|10⟩ = -ca·cb
        
        # E = α*(α*(-ca·cb) + β*(sa·sb)) + β*(α*(sa·sb) + β*(-ca·cb))
        # E = -α²·ca·cb + αβ·sa·sb + αβ·sa·sb - β²·ca·cb
        # E = -(α² + β²)·ca·cb + 2αβ·sa·sb
        # E = -(α² + β²)cos(θ_a)cos(θ_b) + 2αβ sin(θ_a)sin(θ_b)
        
        E = -(alpha**2 + beta**2) * ca * cb + 2 * alpha * beta * sa * sb
        
        # Since α² + β² = 1 (normalized):
        # E = -cos(θ_a)cos(θ_b) + 2αβ sin(θ_a)sin(θ_b)
        
        return E
    
    def CHSH(self, a: float, a_prime: float, b: float, b_prime: float) -> float:
        """Compute CHSH value analytically."""
        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)
        
        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
        return S, (E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)


# ============================================================================
# TEST: Compare Standard Bell vs Fibonacci Bell
# ============================================================================

print("\n" + "=" * 78)
print("COMPARING STANDARD BELL STATE VS FIBONACCI BELL STATES")
print("=" * 78)

# Standard Bell singlet: α = 1/√2, β = -1/√2
# E(θ_a, θ_b) = -cos(θ_a)cos(θ_b) + 2*(1/√2)*(-1/√2)*sin(θ_a)sin(θ_b)
#             = -cos(θ_a)cos(θ_b) - sin(θ_a)sin(θ_b)
#             = -cos(θ_a - θ_b)  ← This is the famous result!

print("\n1. STANDARD BELL SINGLET |ψ⟩ = (|01⟩ - |10⟩)/√2")
print("-" * 50)

class StandardBell:
    def __init__(self):
        self.alpha = 1/np.sqrt(2)
        self.beta = -1/np.sqrt(2)
    
    def correlation(self, theta_a, theta_b):
        return -np.cos(theta_a - theta_b)
    
    def CHSH(self, a, a_prime, b, b_prime):
        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)
        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
        return S, (E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)

# Optimal angles for Bell state
a = 0
a_prime = np.pi/2
b = np.pi/4
b_prime = 3*np.pi/4

standard = StandardBell()
S_std, correlations = standard.CHSH(a, a_prime, b, b_prime)

print(f"Optimal angles: a=0, a'=π/2, b=π/4, b'=3π/4")
print(f"S = {S_std:.4f}")
print(f"Maximum possible: 2√2 = {2*np.sqrt(2):.4f}")
print(f"VIOLATES CLASSICAL BOUND (S > 2): {'YES ✓' if abs(S_std) > 2 else 'NO'}")

# ============================================================================
# FIBONACCI BELL STATES
# ============================================================================

print("\n\n2. FIBONACCI BELL STATES |ψ⟩ = (F_L|01⟩ - F_R|10⟩)/N")
print("-" * 50)

def find_optimal_CHSH(state, n_angles=50):
    """Search for angles that maximize |S|."""
    max_S = 0
    best_angles = None
    
    for a in np.linspace(0, np.pi, n_angles):
        for a_prime in np.linspace(0, np.pi, n_angles):
            for b in np.linspace(0, np.pi, n_angles):
                for b_prime in np.linspace(0, np.pi, n_angles):
                    S, _ = state.CHSH(a, a_prime, b, b_prime)
                    if abs(S) > abs(max_S):
                        max_S = S
                        best_angles = (a, a_prime, b, b_prime)
    
    return max_S, best_angles

print(f"\n{'k':>3} {'F_k':>5} {'F_L/F_R':>8} {'2αβ':>8} {'S_opt':>8} {'S_std':>8} {'Violation?':>12}")
print("-" * 70)

for k in range(4, 15):
    fib_bell = FibonacciBellState(k)
    
    # The key parameter is 2αβ
    two_alpha_beta = 2 * fib_bell.alpha * fib_bell.beta
    
    # At standard Bell angles
    S_at_std, _ = fib_bell.CHSH(a, a_prime, b, b_prime)
    
    # Search for optimal
    S_opt, best = find_optimal_CHSH(fib_bell, n_angles=30)
    
    F_k = fib(k)
    ratio = fib_bell.F_L / fib_bell.F_R
    
    violation = "YES ✓" if abs(S_opt) > 2 else "no"
    
    print(f"{k:3d} {F_k:5d} {ratio:8.4f} {two_alpha_beta:8.4f} {S_opt:8.4f} {S_at_std:8.4f} {violation:>12}")

# ============================================================================
# ANALYSIS: WHY FIBONACCI REDUCES VIOLATION
# ============================================================================

print("\n\n" + "=" * 78)
print("ANALYSIS: THE FIBONACCI VIOLATION FACTOR")
print("=" * 78)

print("""
For state |ψ⟩ = α|01⟩ + β|10⟩, the correlation is:
E(θ_a, θ_b) = -cos(θ_a)cos(θ_b) + 2αβ sin(θ_a)sin(θ_b)

For the standard Bell state: 2αβ = 2×(1/√2)×(-1/√2) = -1
This gives E = -cos(θ_a - θ_b), leading to S_max = 2√2

For Fibonacci Bell at index k:
  α = F_{k-1}/N,  β = -F_{k-2}/N,  where N = √(F_{k-1}² + F_{k-2}²)
  
  2αβ = -2×F_{k-1}×F_{k-2} / (F_{k-1}² + F_{k-2}²)
""")

# Calculate the "entanglement parameter" for different k
print("\nEntanglement parameter 2αβ as k → ∞:")
print("-" * 50)

for k in range(5, 20):
    F_L, F_R = fib(k-1), fib(k-2)
    two_alpha_beta = -2 * F_L * F_R / (F_L**2 + F_R**2)
    print(f"  k={k:2d}: 2αβ = {two_alpha_beta:.6f}")

# Analytical limit
# As k→∞, F_{k-1}/F_{k-2} → φ
# So 2αβ → -2φ/(φ² + 1) = -2φ/(φ² + 1)
# Since φ² = φ + 1: 2αβ → -2φ/(φ + 2)
limit = -2 * phi / (phi + 2)
print(f"\n  Limit as k→∞: 2αβ → -2φ/(φ+2) = {limit:.6f}")
print(f"  Compare to Bell singlet: 2αβ = -1.000000")
print(f"  Ratio: {abs(limit):.4f} = {abs(limit)*100:.1f}% of maximum entanglement")

# ============================================================================
# THE MAXIMUM CHSH FOR FIBONACCI STATES
# ============================================================================

print("\n\n" + "=" * 78)
print("MAXIMUM CHSH FOR FIBONACCI BELL STATES")
print("=" * 78)

# For the correlation E = -cos(θ_a)cos(θ_b) + c×sin(θ_a)sin(θ_b)
# where c = 2αβ, the maximum CHSH is:
# S_max = 2√(1 + c²)  for c ∈ [-1, 0]

# Wait, that's not quite right. Let me derive it properly.

# E(θ_a, θ_b) = -cos(θ_a)cos(θ_b) + c sin(θ_a)sin(θ_b)
# This can be rewritten using cos(θ_a - θ_b) = cos(θ_a)cos(θ_b) + sin(θ_a)sin(θ_b)
#                   and cos(θ_a + θ_b) = cos(θ_a)cos(θ_b) - sin(θ_a)sin(θ_b)

# E = -cos(θ_a)cos(θ_b) + c sin(θ_a)sin(θ_b)
# E = -(1/2)[cos(θ_a-θ_b) + cos(θ_a+θ_b)] + (c/2)[cos(θ_a-θ_b) - cos(θ_a+θ_b)]
# E = [(c-1)/2]cos(θ_a-θ_b) - [(c+1)/2]cos(θ_a+θ_b)

# For c = -1 (Bell singlet): E = -cos(θ_a - θ_b)  ✓

print("""
For correlation E = -cos(θ_a)cos(θ_b) + c×sin(θ_a)sin(θ_b), where c = 2αβ:

The CHSH can be maximized by noting that:
E = [(c-1)/2]cos(θ_a - θ_b) - [(c+1)/2]cos(θ_a + θ_b)

For standard Bell (c=-1): E = -cos(θ_a - θ_b), giving S_max = 2√2

For Fibonacci Bell (c = -2φ/(φ+2) ≈ -0.894):
The optimal angles and maximum S depend on c.
""")

# Compute S_max numerically for the limiting Fibonacci state
class LimitFibonacciBell:
    def __init__(self):
        self.c = -2 * phi / (phi + 2)
    
    def correlation(self, theta_a, theta_b):
        return -np.cos(theta_a)*np.cos(theta_b) + self.c * np.sin(theta_a)*np.sin(theta_b)
    
    def CHSH(self, a, a_prime, b, b_prime):
        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)
        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
        return S, (E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)

limit_state = LimitFibonacciBell()
S_limit, best = find_optimal_CHSH(limit_state, n_angles=50)

print(f"\nLimiting Fibonacci Bell State (k → ∞):")
print(f"  Entanglement parameter: c = 2αβ = {limit_state.c:.6f}")
print(f"  Maximum |S| found: {abs(S_limit):.4f}")
print(f"  Compare to quantum max: 2√2 = {2*np.sqrt(2):.4f}")
print(f"  Compare to classical bound: 2")

if abs(S_limit) > 2:
    print(f"\n  🔔 FIBONACCI BELL STATE VIOLATES CLASSICAL BOUND!")
    print(f"     Violation: {abs(S_limit) - 2:.4f}")
    print(f"     This is {(abs(S_limit) - 2)/(2*np.sqrt(2) - 2) * 100:.1f}% of max quantum violation")
else:
    print(f"\n  📊 Fibonacci Bell state does NOT violate classical bound")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 78)
print("SUMMARY: PAC BELL CORRELATIONS")
print("=" * 78)

SEC_CHSH = 1.002  # From quantum_validation

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BELL VIOLATION COMPARISON                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Framework              |  Max |S|   |  Status                               ║
║  ───────────────────────┼────────────┼────────────────────────────────────── ║
║  SEC only (local)       |  {SEC_CHSH:.3f}      |  Well below classical bound         ║
║  Fibonacci Bell (k→∞)   |  {abs(S_limit):.3f}      |  {"EXCEEDS classical! ✓" if abs(S_limit) > 2 else "Below classical bound"}            ║
║  Standard Bell          |  {2*np.sqrt(2):.3f}      |  Maximum quantum violation          ║
║                                                                              ║
║  BOUNDS:                                                                     ║
║  Classical (local HV):  |S| ≤ 2.000                                          ║
║  Quantum maximum:       |S| ≤ 2.828                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

INTERPRETATION:
───────────────
""")

if abs(S_limit) > 2:
    print(f"""The PAC conservation law Ψ(k) = Ψ(k+1) + Ψ(k+2) creates an entangled state
where the LEFT (F_{{k-1}}) and RIGHT (F_{{k-2}}) branches are correlated.

This "Fibonacci entanglement" with ratio {phi:.4f} (golden ratio) produces:
- Bell violation of {abs(S_limit):.4f} > 2.0 (classical bound)
- Violation is {(abs(S_limit)-2)/(2*np.sqrt(2)-2)*100:.1f}% of maximum quantum violation
- The "missing" violation comes from the asymmetric weighting

THIS CONFIRMS THE PAC-SEC ARCHITECTURE:
- SEC handles LOCAL dynamics (Born rule, interference) - no Bell violation
- PAC handles NON-LOCAL structure (entanglement) - Bell violation from tree

The golden ratio φ appears as the natural entanglement weight,
just as it appears in coupling constants and mass ratios!
""")
else:
    print(f"""The Fibonacci weighting reduces entanglement below the Bell violation threshold.
This suggests PAC structure alone doesn't create quantum nonlocality,
but rather a special kind of classical correlation at the boundary.

The ratio of 2αβ = {limit:.4f} vs -1 (Bell singlet) shows that
Fibonacci correlations are {abs(limit)*100:.1f}% as strong as maximum entanglement.
""")

print("\n" + "=" * 78)
print("TEST COMPLETE")
print("=" * 78)
