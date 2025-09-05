"""
Fracton Core Package

The core package provides the fundamental runtime engine for the Fracton
computational modeling language, including recursive execution, entropy
dispatch, bifractal tracing, and memory field management.
"""

from .recursive_engine import RecursiveExecutor, ExecutionContext, get_default_executor
from .entropy_dispatch import EntropyDispatcher, DispatchConditions, get_default_dispatcher
from .bifractal_trace import BifractalTrace, TraceAnalysis
from .memory_field import MemoryField, FieldController, get_default_controller

__version__ = "0.1.0"
__all__ = [
    "RecursiveExecutor",
    "ExecutionContext", 
    "get_default_executor",
    "EntropyDispatcher",
    "DispatchConditions",
    "get_default_dispatcher",
    "BifractalTrace",
    "TraceAnalysis",
    "MemoryField",
    "FieldController",
    "get_default_controller"
]
