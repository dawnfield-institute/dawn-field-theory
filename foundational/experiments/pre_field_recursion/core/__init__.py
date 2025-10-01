"""
Core modules for Pre-Field Recursion experiments

v2.2: Resonance-aware convergence with FFT frequency detection
"""

# Legacy modules (keeping for backwards compatibility)
try:
    from .mobius_topology import MobiusTopology, TopologyAnalyzer
    from .sec_field import SECFieldSimulator
    from .boundary_conditions import AntiPeriodicBoundary
    legacy_available = True
except ImportError:
    legacy_available = False

# v2.0 modules
from .formal_definitions import PreFieldState, RecursionOperator, create_initial_state
from .transition_dynamics import PreFieldTransition

# v2.1 modules
from .adaptive_recursion import AdaptiveRecursionOperator

# v2.2 modules
from .resonance_detector import ResonanceDetector, visualize_resonance_analysis

# Export all
__all__ = [
    # v2.2 Framework (latest)
    'ResonanceDetector',
    'visualize_resonance_analysis',
    
    # v2.1 Framework
    'AdaptiveRecursionOperator',
    
    # v2.0 Framework
    'PreFieldState',
    'RecursionOperator',
    'PreFieldTransition',
    'create_initial_state',
]

# Add legacy if available
if legacy_available:
    __all__.extend([
        'MobiusTopology',
        'TopologyAnalyzer',
        'SECFieldSimulator',
        'AntiPeriodicBoundary'
    ])

__version__ = '2.2.0'