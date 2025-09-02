"""
GAIA v2.0 - Shared Data Structures
Common data structures used across GAIA modules

TORCH ONLY - NO NUMPY
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FieldState:
    """Current state of entropy and information fields"""
    energy_field: torch.Tensor
    information_field: torch.Tensor
    entropy_tensor: torch.Tensor
    field_pressure: float
    collapse_likelihood: float
    timestamp: float


@dataclass
class CollapseEvent:
    """Represents a field collapse event"""
    location: Tuple[int, ...]
    entropy_delta: float
    field_pressure_pre: float
    field_pressure_post: float
    collapse_type: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class SymbolicStructure:
    """Represents a crystallized symbolic structure from collapse"""
    structure_id: str
    collapse_location: Tuple[int, ...]
    symbolic_content: torch.Tensor
    entropy_signature: float
    thermodynamic_cost: float
    creation_timestamp: float
    ancestry_trace: List[str]
    geometric_properties: Dict[str, float]


@dataclass
class GAIAState:
    """Complete state of GAIA system"""
    field_state: FieldState
    symbolic_structures: List[SymbolicStructure]
    timestep: int
    total_collapses: int
    cognitive_load: float
