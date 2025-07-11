"""
TinyCIMM-Euler Publication Suite: Logging Modules
=================================================

This package contains structured logging capabilities for TinyCIMM-Euler experiments.
Provides multi-format logging, experiment tracking, and automated report generation.

Modules:
- publication_logger: Comprehensive structured logging system

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
"""

from .publication_logger import PublicationLogger, create_experiment_logger

__all__ = ['PublicationLogger', 'create_experiment_logger']

__version__ = "1.0.0"
__author__ = "Dawn Field Theory Research Team"
