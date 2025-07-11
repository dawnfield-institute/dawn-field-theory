"""
TinyCIMM-Euler Publication Suite: Analysis Modules
==================================================

This package contains comprehensive analysis modules for TinyCIMM-Euler experiments.
Provides detailed SCBF analysis, convergence metrics, and publication-quality summaries.

Modules:
- scbf_summary_generator: SCBF and activation trace analysis with comprehensive metrics

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
"""

from .scbf_summary_generator import SCBFSummaryGenerator

__all__ = ['SCBFSummaryGenerator']

__version__ = "1.0.0"
__author__ = "Dawn Field Theory Research Team"
