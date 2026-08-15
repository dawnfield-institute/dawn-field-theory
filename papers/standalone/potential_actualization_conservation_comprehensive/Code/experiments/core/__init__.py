"""
Information Amplification Core Module
"""

from .measurement import InformationMeasurement
from .compression_engine import CompressionEngine
from .text_generator import TextGenerator
from .baseline_generator import BaselineGenerator
from .sec_field_engine import AuthenticSECField
from .quantum_validator import QuantumValidator

__all__ = [
    'InformationMeasurement',
    'CompressionEngine', 
    'TextGenerator',
    'BaselineGenerator',
    'AuthenticSECField',
    'QuantumValidator'
]

__version__ = "1.1.0"
