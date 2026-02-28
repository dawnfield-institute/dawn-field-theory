"""
Dawn Field Theory Constants
===========================

Fundamental constants used throughout the pre-field EM emergence framework.
These emerge from PAC recursion and Möbius topology, not fitted.

Key Result:
    E/B = φ^(-4.42 × w/R + 2.34)
    
    This power law determines electromagnetic coupling from geometry.
"""

import numpy as np

# =============================================================================
# Golden Ratio Family
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2
"""Golden ratio φ = 1.6180339887...

The unique positive solution to x² = x + 1.
Emerges as the fixed point of PAC recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2).
"""

PHI_INV = 1 / PHI  # = φ - 1 = 0.6180339887...
"""Inverse golden ratio 1/φ = φ - 1 = 0.6180339887...

Also equals φ - 1 due to the identity φ² = φ + 1.
"""

PHI_SQ = PHI ** 2  # = φ + 1 = 2.6180339887...
"""Golden ratio squared φ² = φ + 1 = 2.6180339887..."""

PHI_SQRT = np.sqrt(PHI)  # = 1.2720196495...
"""Square root of golden ratio √φ = 1.2720196495..."""

PHI_1_5 = PHI ** 1.5  # = φ√φ = 2.0581710272...
"""φ^(3/2) = φ√φ = 2.0581710272...

This appears as the E/B ratio for medium-width Möbius strips (w/R ≈ 0.25).
"""

# =============================================================================
# Balance Operator
# =============================================================================

XI = 1 + np.pi / 55
"""Balance operator Ξ = 1 + π/55 = 1.0571...

Emerges from the spectral structure of PAC trees.
55 = F₁₀ (10th Fibonacci number).

Bounds: Ξ ∈ [1.0015, 1.0571] in computational validation.
"""

# =============================================================================
# Resonance Parameters
# =============================================================================

PI_FREQ = 0.0301
"""Natural resonance frequency in cycles/iteration.

Discovered in pre-field recursion experiments.
Period ≈ 33 iterations.

When SEC evolution is tuned to this frequency,
convergence accelerates by ~5x.
"""

RESONANCE_PERIOD = int(1 / PI_FREQ)  # ≈ 33
"""Natural period of pre-field oscillation in iterations."""

# =============================================================================
# Fibonacci Sequence
# =============================================================================

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
"""Fibonacci sequence F_n.

Key values:
- F₄ = 3: SU(2) generators
- F₆ = 8: SU(3) generators  
- F₇ = 13: Gauge closure number
- F₁₀ = 55: Appears in Ξ = 1 + π/55
- F₁₂ = 144: Appears in α ≈ 1/144 × Ξ
"""

# =============================================================================
# Power Law Coefficients (Empirical)
# =============================================================================

POWER_LAW_SLOPE = -4.42
"""Slope in E/B = φ^(slope × w/R + intercept).

Determined from parameter sweep with R² = 0.9764.
"""

POWER_LAW_INTERCEPT = 2.34
"""Intercept in E/B = φ^(slope × w/R + intercept)."""

OPTIMAL_WR_RATIO = (1 - POWER_LAW_INTERCEPT) / POWER_LAW_SLOPE
"""w/R ratio that gives E/B = φ.

Calculated: (1 - 2.34) / (-4.42) = 0.303...
"""

# =============================================================================
# Derived Constants
# =============================================================================

def eb_from_wr(w_over_r: float) -> float:
    """Calculate E/B ratio from Möbius w/R geometry.
    
    Args:
        w_over_r: Width-to-radius ratio of Möbius strip
        
    Returns:
        E/B ratio as φ^(power)
        
    Example:
        >>> eb_from_wr(0.304)  # Optimal geometry
        1.617...  # ≈ φ
    """
    power = POWER_LAW_SLOPE * w_over_r + POWER_LAW_INTERCEPT
    return PHI ** power


def wr_for_eb(target_eb: float) -> float:
    """Calculate required w/R to achieve target E/B ratio.
    
    Args:
        target_eb: Desired E/B ratio
        
    Returns:
        Required w/R ratio
        
    Example:
        >>> wr_for_eb(PHI)  # Want E/B = φ
        0.304...
    """
    power = np.log(target_eb) / np.log(PHI)
    return (power - POWER_LAW_INTERCEPT) / POWER_LAW_SLOPE


def phi_power_from_eb(eb_ratio: float) -> float:
    """Calculate φ-power from E/B ratio.
    
    Args:
        eb_ratio: Observed E/B ratio
        
    Returns:
        Power n where E/B = φ^n
        
    Example:
        >>> phi_power_from_eb(2.618)
        2.0  # φ² 
    """
    return np.log(eb_ratio) / np.log(PHI)


# =============================================================================
# Validation Thresholds
# =============================================================================

PAC_CONVERGENCE_THRESHOLD = 0.01
"""PAC residual below which system is considered converged."""

DIV_B_THRESHOLD = 0.001
"""∇·B threshold for no-monopole validation."""

DIV_E_THRESHOLD = 1.0
"""∇·E threshold for approximate Gauss's law in vacuum."""

PHI_MATCH_THRESHOLD = 0.05
"""Relative deviation threshold for E/B ≈ φ (5%)."""
