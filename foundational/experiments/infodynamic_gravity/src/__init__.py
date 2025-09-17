"""
Infodynamic Gravity - Core Implementation

A scale-dependent infodynamic gravity theory implementation that unifies
galaxy and cosmic web physics through information-theoretic principles.

Key Components:
- InfoGravityField: Main gravity field implementation
- scale_dependent_arithmetic: Scale transition mathematics  
- SECDynamics: Structured Entropy Collapse dynamics
- GalaxySimulator: Multi-scale simulation engine

Usage:
    from src.infodynamic_gravity import InfoGravityField, InfoGravityConfig
    from src.scale_dependent_arithmetic import get_scale_dependent_parameters
    from src.galaxy_simulator import GalaxySimulator, GalaxyConfig
    from src.sec_dynamics import SECDynamics, SECConfig
"""

from .infodynamic_gravity import InfoGravityField, InfoGravityConfig
from .scale_dependent_arithmetic import (
    get_scale_dependent_parameters,
    calculate_characteristic_length,
    scale_transition_function,
    ScaleRegimes
)
from .galaxy_simulator import GalaxySimulator, GalaxyConfig
from .sec_dynamics import SECDynamics, SECConfig

__version__ = "1.0.0"
__author__ = "Dawn Field Institute"

__all__ = [
    "InfoGravityField",
    "InfoGravityConfig", 
    "get_scale_dependent_parameters",
    "calculate_characteristic_length",
    "scale_transition_function",
    "ScaleRegimes",
    "GalaxySimulator",
    "GalaxyConfig",
    "SECDynamics",
    "SECConfig"
]
