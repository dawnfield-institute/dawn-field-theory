"""
Milestone 3: Shared constants for all experiments.

All constants are defined here to avoid duplication across experiment scripts.
These are the same constants used across milestones 1, 2, and the energy_equivalence session.
"""

import math

# -- Fundamental mathematical constants --
PHI = (1 + math.sqrt(5)) / 2         # Golden ratio: 1.6180339887...
INV_PHI = 1 / PHI                     # 1/φ: 0.6180339887...
LN_PHI = math.log(PHI)                # ln(φ): 0.4812118250...
GAMMA_EM = 0.5772156649015329         # Euler-Mascheroni constant

# -- Framework constants --
XI_BALANCE = GAMMA_EM + LN_PHI        # Ξ = γ + ln(φ) ≈ 1.0584
PI_OVER_55 = math.pi / 55             # π/55 ≈ 0.05712

# -- Thermodynamic constants --
KT_DEFAULT = 1.0                      # Default thermal energy (natural units)
LANDAUER_MIN = KT_DEFAULT * math.log(2)  # Minimum erasure cost: kT·ln(2)

# -- Fibonacci sequence (precomputed first 20) --
def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: F(0)=0, F(1)=1, F(2)=1, ...)."""
    if n < 0:
        raise ValueError(f"Negative index: {n}")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Precompute F(0) through F(20)
FIB = [fibonacci(i) for i in range(21)]

# -- Standard Model reference values (PDG 2024) --
ALPHA_EM_PDG = 0.0072973525693       # Fine structure constant
ALPHA_EM_ERR = 0.0000000000011       # Uncertainty
SIN2_THETA_W_PDG = 0.23122           # Weinberg angle (MS-bar, M_Z)
SIN2_THETA_W_ERR = 0.00003           # Uncertainty
