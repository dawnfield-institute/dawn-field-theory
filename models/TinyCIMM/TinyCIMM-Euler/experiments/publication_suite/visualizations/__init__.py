"""
TinyCIMM-Euler Publication Suite: Visualization Modules
======================================================

This package contains publication-quality visualization modules for TinyCIMM-Euler experiments.
Each module provides comprehensive visualization capabilities for different aspects of the
experimental analysis.

Modules:
- entropy_collapse_overlay: Entropy collapse and role assignment visualization
- activation_overlays: Field topology and structural emergence visualization
- neuron_trace_analysis: Neuron specialization and role dynamics analysis
- convergence_timeline: Symbolic convergence and abstraction progression

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
"""

from .entropy_collapse_overlay import EntropyCollapseOverlay
from .activation_overlays import ActivationOverlayGenerator
from .neuron_trace_analysis import NeuronTraceAnalyzer
from .convergence_timeline import ConvergenceTimelineGenerator

__all__ = [
    'EntropyCollapseOverlay',
    'ActivationOverlayGenerator', 
    'NeuronTraceAnalyzer',
    'ConvergenceTimelineGenerator'
]

__version__ = "1.0.0"
__author__ = "Dawn Field Theory Research Team"
