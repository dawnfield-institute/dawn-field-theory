"""
Milestone 1 Shared Constants (imported for consistency)

Using the same constants as the main milestone1 experiment suite.
"""

import numpy as np
from functools import lru_cache

# =============================================================================
# MATHEMATICAL CONSTANTS (derived, not fitted)
# =============================================================================

# Golden ratio - THE solution to r² = r + 1
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...
PSI = (1 - np.sqrt(5)) / 2  # -0.6180339887... (conjugate)

# Balance operator Xi
# DERIVED: Ξ = 1 + π/55 where 55 = F₁₀
XI = 1 + np.pi / 55  # 1.0571081...

# Fibonacci sequence
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

# Key Fibonacci numbers
F = {i: fib(i) for i in range(20)}
F3, F4, F5, F6, F7, F10 = 2, 3, 5, 8, 13, 55

# =============================================================================
# PRINTING UTILITIES
# =============================================================================

def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

def print_subheader(text: str):
    print("\n" + "-" * 50)
    print(text)
    print("-" * 50)
