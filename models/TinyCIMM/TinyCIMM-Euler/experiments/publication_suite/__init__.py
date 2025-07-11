"""
TinyCIMM-Euler Publication Suite
================================

Complete publication-ready experiment framework for TinyCIMM-Euler mathematical reasoning research.
This suite provides comprehensive tools for conducting, analyzing, and documenting reproducible
experiments with publication-quality outputs.

Main Components:
- run_publication_experiments: Main CLI orchestrator
- visualizations: Publication-quality visualization modules
- analysis: Comprehensive SCBF and convergence analysis
- logging: Structured logging and report generation
- config: Experiment configuration templates

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
"""

from .run_publication_experiments import PublicationExperimentSuite, PublicationExperimentRunner

__all__ = ['PublicationExperimentSuite', 'PublicationExperimentRunner']

__version__ = "1.0.0"
__author__ = "Dawn Field Theory Research Team"
