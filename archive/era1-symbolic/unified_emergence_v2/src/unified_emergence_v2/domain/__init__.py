"""
Domain layer exports for the Unified Emergence Framework v2.
"""

from .models import (
    EmergenceSignature,
    CorrelationMatrix, 
    ValidationMetrics,
    ValidationConfig,
    EmergenceResults
)

from .protocols import (
    DomainAdapter,
    TestRunner,
    ResultsRepository,
    PatternAnalyzer
)

__all__ = [
    # Models
    'EmergenceSignature',
    'CorrelationMatrix',
    'ValidationMetrics', 
    'ValidationConfig',
    'EmergenceResults',
    
    # Protocols
    'DomainAdapter',
    'TestRunner', 
    'ResultsRepository',
    'PatternAnalyzer'
]
