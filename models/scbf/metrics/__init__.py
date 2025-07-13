"""
SCBF Core Metrics Module
========================

Implements core symbolic collapse and bifractal field metrics based on 
Dawn Field Theory literature and TinyCIMM-Euler experimental results.

**STRICT NO SYNTHETIC DATA POLICY**
- All functions require real experimental data
- Missing/invalid data raises clear exceptions
- Zero tolerance for fallback/placeholder values
"""

from .entropy_collapse import compute_symbolic_entropy_collapse
from .activation_ancestry import compute_activation_ancestry
from .phase_alignment import compute_collapse_phase_alignment
from .semantic_attractors import compute_semantic_attractor_density
from .bifractal_lineage import compute_bifractal_lineage

__all__ = [
    'compute_symbolic_entropy_collapse',
    'compute_activation_ancestry', 
    'compute_collapse_phase_alignment',
    'compute_semantic_attractor_density',
    'compute_bifractal_lineage'
]
