"""
SCBF Utilities Module
====================

Shared utility functions and helpers for SCBF framework.
"""

from .scbf_utils import (
    normalize_activations,
    compute_activation_hash,
    safe_entropy,
    sliding_window_analysis,
    detect_outliers,
    compute_fractal_dimension,
    compute_correlation_dimension,
    validate_scbf_input,
    create_mock_activations,
    format_experiment_summary
)

__all__ = [
    'normalize_activations',
    'compute_activation_hash',
    'safe_entropy',
    'sliding_window_analysis',
    'detect_outliers',
    'compute_fractal_dimension',
    'compute_correlation_dimension',
    'validate_scbf_input',
    'create_mock_activations',
    'format_experiment_summary'
]
