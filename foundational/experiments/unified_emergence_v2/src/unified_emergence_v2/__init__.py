"""
Unified Emergence Framework v2

A clean, maintainable framework for analyzing emergence patterns across multiple domains.
"""

from .application.framework import UnifiedEmergenceFramework
from .domain.models import EmergenceSignature, EmergenceResults, ValidationConfig
from .domain.protocols import DomainAdapter

__version__ = "2.0.0"
__all__ = [
    "UnifiedEmergenceFramework",
    "EmergenceSignature", 
    "EmergenceResults",
    "ValidationConfig",
    "DomainAdapter"
]
