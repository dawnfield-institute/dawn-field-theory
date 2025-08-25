"""
Information Amplification Core Module
"""

from .measurement import InformationMeasurement
from .compression_engine import CompressionEngine
from .text_generator import TextGenerator
from .sec_weight_interpreter import SECWeightInterpreter

# Keep original classes for backwards compatibility
from .amplification_test import (
    InformationAmplificationTest,
    OptimalCompressor,
    InformationMeter,
    EnvironmentProfiler,
    CompressionResult,
    AmplificationMeasurement
)

__all__ = [
    'InformationMeasurement',
    'CompressionEngine', 
    'TextGenerator',
    'SECWeightInterpreter',
    # Legacy exports
    'InformationAmplificationTest',
    'OptimalCompressor', 
    'InformationMeter',
    'EnvironmentProfiler',
    'CompressionResult',
    'AmplificationMeasurement'
]

__version__ = "1.1.0"
