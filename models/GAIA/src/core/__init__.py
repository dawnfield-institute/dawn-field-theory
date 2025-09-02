"""
GAIA v2.0 - Core Module
Main components of the GAIA architecture
"""

from .data_structures import FieldState, CollapseEvent, SymbolicStructure, GAIAState
from .field_engine import FieldEngine
from .collapse_core import CollapseCore
from .gaia import GAIA

__all__ = [
    'GAIA',
    'GAIAState', 
    'FieldEngine',
    'FieldState',
    'CollapseEvent',
    'CollapseCore',
    'SymbolicStructure'
]
