"""
Pre-Field EM Emergence Core Module
==================================

Core implementations for pre-field dynamics, SEC evolution,
and 3D electromagnetic field extraction.

Classes:
    MobiusField: Pre-field state on Möbius manifold
    SECOperator: Symbolic Entropy Collapse evolution
    EMProjector: 3D projection and EM field extraction
    MaxwellValidator: Maxwell equation compliance checking

Constants:
    PHI: Golden ratio (1.618...)
    XI: Balance operator (1.0571)
    PI_FREQ: Natural resonance frequency (0.0301 Hz)
"""

from .constants import PHI, PHI_INV, PHI_SQ, XI, PI_FREQ, FIB
from .mobius_field import MobiusField
from .sec_operator import SECOperator
from .projector import EMProjector, MaxwellValidator

__version__ = "1.0.0"
__author__ = "Peter Lorne Groom, Claude (Anthropic)"

__all__ = [
    # Constants
    "PHI", "PHI_INV", "PHI_SQ", "XI", "PI_FREQ", "FIB",
    # Classes
    "MobiusField", "SECOperator", "EMProjector", "MaxwellValidator",
]
