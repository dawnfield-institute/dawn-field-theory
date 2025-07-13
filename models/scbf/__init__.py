"""
SCBF - Symbolic Collapse Bifractal Framework
===========================================

A comprehensive framework for analyzing neural network learning dynamics
through symbolic entropy collapse and bifractal analysis.

Main Components:
- metrics: Core SCBF analysis metrics
- loggers: Experiment tracking and logging
- visualization: Plotting and analysis dashboards  
- utils: Shared utilities and helpers
- scbf_runner: Modular experiment runner

Usage:
    from scbf import run_scbf_analysis_step
    from scbf.loggers import create_experiment_logger
    from scbf.visualization import plot_complete_scbf_dashboard
"""

# Core metrics
from .metrics import (
    compute_symbolic_entropy_collapse,
    compute_activation_ancestry,
    compute_collapse_phase_alignment,
    compute_semantic_attractor_density,
    compute_bifractal_lineage
)

# Experiment runner and integration
from .scbf_runner import (
    run_scbf_analysis_step,
    extract_model_activations,
    register_experiment,
    run_experiment,
    list_experiments,
    analyze_experiment
)

# Logging
from .loggers import (
    create_experiment_logger,
    finalize_experiment,
    list_all_experiments,
    load_experiment_results
)

# Visualization
from .visualization import (
    plot_complete_scbf_dashboard,
    save_all_plots
)

# Utilities
from .utils import (
    validate_scbf_input,
    create_mock_activations,
    format_experiment_summary
)

__version__ = "1.0.0"

__all__ = [
    # Core metrics
    'compute_symbolic_entropy_collapse',
    'compute_activation_ancestry', 
    'compute_collapse_phase_alignment',
    'compute_semantic_attractor_density',
    'compute_bifractal_lineage',
    
    # Experiment runner
    'run_scbf_analysis_step',
    'extract_model_activations',
    'register_experiment',
    'run_experiment',
    'list_experiments',
    'analyze_experiment',
    
    # Logging
    'create_experiment_logger',
    'finalize_experiment',
    'list_all_experiments',
    'load_experiment_results',
    
    # Visualization
    'plot_complete_scbf_dashboard',
    'save_all_plots',
    
    # Utilities
    'validate_scbf_input',
    'create_mock_activations',
    'format_experiment_summary'
]
