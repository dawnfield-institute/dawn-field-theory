"""
Test configuration and shared fixtures for Fracton test suite.

This module provides common test utilities, fixtures, and configuration
aligned with Dawn Field Theory's foundational principles.
"""

import pytest
import time
import numpy as np
from typing import Dict, Any, List, Callable
from unittest.mock import Mock, patch

# Import Fracton modules for testing
import sys
import os
fracton_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, fracton_root)

# Import core modules directly
from core.recursive_engine import RecursiveExecutor
from core.memory_field import MemoryField  
from core.entropy_dispatch import EntropyDispatcher
from core.bifractal_trace import BifractalTrace

# Create a minimal Context class for testing
class Context:
    def __init__(self, entropy_level=0.5, depth=0, metadata=None):
        self.entropy_level = entropy_level
        self.depth = depth
        self.metadata = metadata or {}

# Create fracton module-like namespace for decorators
class FractonNamespace:
    def recursive(self, func):
        """Mock recursive decorator"""
        func._is_recursive = True
        return func
    
    def entropy_gate(self, low, high):
        """Mock entropy gate decorator"""
        def decorator(func):
            func._entropy_gate = (low, high)
            return func
        return decorator
    
    def recurse(self, func, memory, context):
        """Mock recurse function"""
        return func(memory, context)

fracton = FractonNamespace()


# Test Configuration Constants
class TestConfig:
    """Test configuration aligned with foundational theory requirements."""
    
    # SEC (Symbolic Entropy Collapse) parameters
    SEC_LOW_ENTROPY = 0.1      # Low entropy for crystallized states
    SEC_HIGH_ENTROPY = 0.9     # High entropy for exploratory states
    SEC_TRANSITION = 0.5       # Entropy threshold for transitions
    
    # MED (Macro Emergence Dynamics) parameters
    MED_MACRO_SCALE = 100      # Large-scale field size
    MED_MICRO_SCALE = 10       # Small-scale field size
    MED_EMERGENCE_THRESHOLD = 0.7  # Emergence detection threshold
    
    # Bifractal time parameters
    BIFRACTAL_ANCESTRY_DEPTH = 5   # Backward tracing depth
    BIFRACTAL_FUTURE_HORIZON = 3   # Forward emergence horizon
    
    # Recursive balance field parameters
    BALANCE_FIELD_CAPACITY = 1000  # Default field capacity
    BALANCE_ADAPTATION_RATE = 0.1  # Dynamic adaptation rate
    
    # Performance testing parameters
    PERFORMANCE_MAX_DEPTH = 50     # Maximum recursion depth for testing
    PERFORMANCE_TIMEOUT = 5.0      # Test timeout in seconds
    
    # Entropy regulation parameters
    LANDAUER_COST_THRESHOLD = 0.01  # Minimum erasure cost
    ENTROPY_REGULATION_EPSILON = 1e-6  # Numerical precision


# Foundational Theory Fixtures

@pytest.fixture
def sec_low_entropy_context():
    """Context for SEC low-entropy (crystallized) state testing."""
    return Context(
        entropy=TestConfig.SEC_LOW_ENTROPY,
        depth=0,
        experiment="sec_low_entropy",
        timestamp=time.time()
    )


@pytest.fixture
def sec_high_entropy_context():
    """Context for SEC high-entropy (exploratory) state testing."""
    return Context(
        entropy=TestConfig.SEC_HIGH_ENTROPY,
        depth=0,
        experiment="sec_high_entropy",
        timestamp=time.time()
    )


@pytest.fixture
def sec_transition_context():
    """Context for SEC entropy transition testing."""
    return Context(
        entropy=TestConfig.SEC_TRANSITION,
        depth=0,
        experiment="sec_transition",
        timestamp=time.time()
    )


@pytest.fixture
def med_macro_field():
    """Large-scale memory field for MED macro emergence testing."""
    return MemoryField(
        capacity=TestConfig.MED_MACRO_SCALE,
        entropy=TestConfig.SEC_HIGH_ENTROPY,
        field_id="med_macro"
    )


@pytest.fixture
def med_micro_field():
    """Small-scale memory field for MED micro emergence testing."""
    return MemoryField(
        capacity=TestConfig.MED_MICRO_SCALE,
        entropy=TestConfig.SEC_LOW_ENTROPY,
        field_id="med_micro"
    )


@pytest.fixture
def bifractal_trace():
    """Bifractal trace for temporal symmetry testing."""
    return BifractalTrace(
        ancestry_depth=TestConfig.BIFRACTAL_ANCESTRY_DEPTH,
        future_horizon=TestConfig.BIFRACTAL_FUTURE_HORIZON
    )


@pytest.fixture
def recursive_executor():
    """Recursive executor for engine testing."""
    return RecursiveExecutor(
        max_depth=TestConfig.PERFORMANCE_MAX_DEPTH,
        entropy_regulation=True
    )


@pytest.fixture
def entropy_dispatcher():
    """Entropy dispatcher for routing testing."""
    return EntropyDispatcher()


@pytest.fixture
def balanced_memory_field():
    """Memory field configured for recursive balance testing."""
    return MemoryField(
        capacity=TestConfig.BALANCE_FIELD_CAPACITY,
        entropy=TestConfig.SEC_TRANSITION,
        adaptation_rate=TestConfig.BALANCE_ADAPTATION_RATE,
        field_id="balanced_field"
    )


# Test Helper Functions

def assert_sec_compliance(entropy_before: float, entropy_after: float, 
                         operation_type: str = "collapse"):
    """
    Assert that an operation complies with SEC dynamics.
    
    SEC: ∇_micro → Ψ_macro (micro states lead to macro emergence)
    For collapse: entropy should decrease (micro → macro crystallization)
    For exploration: entropy should increase (macro → micro expansion)
    """
    if operation_type == "collapse":
        assert entropy_after < entropy_before, \
            f"SEC collapse should decrease entropy: {entropy_before} → {entropy_after}"
    elif operation_type == "exploration":
        assert entropy_after > entropy_before, \
            f"SEC exploration should increase entropy: {entropy_before} → {entropy_after}"
    else:
        raise ValueError(f"Unknown SEC operation type: {operation_type}")


def assert_med_compliance(macro_state: Dict[str, Any], micro_state: Dict[str, Any]):
    """
    Assert that states comply with MED patterns.
    
    MED: ∇_macro → Ψ_micro (macro fields constrain micro actualization)
    Macro structure should influence micro organization.
    """
    # Check that macro state has larger scale/scope
    macro_scale = macro_state.get('scale', 0)
    micro_scale = micro_state.get('scale', 0)
    assert macro_scale > micro_scale, \
        f"MED requires macro scale > micro scale: {macro_scale} vs {micro_scale}"
    
    # Check that macro entropy influences micro entropy
    macro_entropy = macro_state.get('entropy', 0.5)
    micro_entropy = micro_state.get('entropy', 0.5)
    # Micro entropy should be constrained by macro structure
    assert abs(micro_entropy - macro_entropy) < 0.3, \
        f"MED requires macro-micro entropy coupling: {macro_entropy} vs {micro_entropy}"


def assert_bifractal_symmetry(trace: BifractalTrace, operation_id: str):
    """
    Assert that trace exhibits bifractal temporal symmetry.
    
    Bifractal time: backward ancestry ↔ forward emergence
    Past operations should be balanced by future potential.
    """
    ancestry = trace.get_ancestry(operation_id)
    emergence = trace.get_emergence_potential(operation_id)
    
    # Check that both directions are populated
    assert len(ancestry) > 0, "Bifractal trace requires backward ancestry"
    assert len(emergence) > 0, "Bifractal trace requires forward emergence"
    
    # Check temporal symmetry (balance between past and future)
    ancestry_depth = len(ancestry)
    emergence_depth = len(emergence)
    symmetry_ratio = min(ancestry_depth, emergence_depth) / max(ancestry_depth, emergence_depth)
    assert symmetry_ratio > 0.5, \
        f"Bifractal symmetry requires balanced temporal depth: {ancestry_depth} vs {emergence_depth}"


def assert_recursive_balance(field: MemoryField, operation_count: int):
    """
    Assert that field maintains recursive balance.
    
    Recursive balance: dynamic adaptation to maintain energy-information equilibrium
    Field should adapt its properties based on operation history.
    """
    # Check that field has adapted based on operations
    assert field.get_operation_count() == operation_count, \
        f"Field should track operation count: expected {operation_count}, got {field.get_operation_count()}"
    
    # Check that entropy has evolved (not static)
    initial_entropy = field.get_initial_entropy()
    current_entropy = field.get_entropy()
    if operation_count > 0:
        assert current_entropy != initial_entropy, \
            "Recursive balance requires entropy evolution under operations"
    
    # Check that field maintains stability (doesn't collapse or explode)
    assert 0.0 <= current_entropy <= 1.0, \
        f"Field entropy must remain in valid range: {current_entropy}"


def create_sec_test_function(entropy_gate_low: float = 0.3, 
                           entropy_gate_high: float = 0.8) -> Callable:
    """Create a test function with SEC-compliant entropy gates."""
    
    @fracton.recursive
    @fracton.entropy_gate(entropy_gate_low, entropy_gate_high)
    def sec_test_function(memory: MemoryField, context: Context) -> Any:
        """Test function that demonstrates SEC compliance."""
        if context.depth >= 3:
            return {"result": "crystallized", "entropy": memory.get_entropy()}
        
        # Recursive call with entropy evolution
        new_context = context.deeper(1).with_entropy(context.entropy * 0.9)
        result = fracton.recurse(sec_test_function, memory, new_context)
        
        return {"result": "exploring", "child": result, "entropy": memory.get_entropy()}
    
    return sec_test_function


def create_med_test_system():
    """Create a test system demonstrating MED patterns."""
    
    @fracton.recursive
    def macro_field_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
        """Process macro-scale field structures."""
        macro_data = memory.get("macro_structures", [])
        
        # Macro processing influences micro organization
        micro_constraints = {
            "boundary_conditions": len(macro_data),
            "field_pressure": context.entropy,
            "scale": TestConfig.MED_MACRO_SCALE
        }
        
        memory.set("micro_constraints", micro_constraints)
        return {"macro_result": macro_data, "constraints": micro_constraints}
    
    @fracton.recursive  
    def micro_field_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
        """Process micro-scale actualizations under macro constraints."""
        constraints = memory.get("micro_constraints", {})
        
        # Micro behavior constrained by macro structure
        micro_result = {
            "actualization": "constrained_by_macro",
            "boundary_respect": constraints.get("boundary_conditions", 0),
            "scale": TestConfig.MED_MICRO_SCALE,
            "entropy": context.entropy * constraints.get("field_pressure", 1.0)
        }
        
        return micro_result
    
    return macro_field_processor, micro_field_processor


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor performance during testing."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < TestConfig.PERFORMANCE_TIMEOUT, \
            f"Operation exceeded timeout: {duration:.3f}s > {TestConfig.PERFORMANCE_TIMEOUT}s"
        
        return result
    return wrapper


# Test Data Generators

def generate_entropy_sequence(length: int, trend: str = "collapse") -> List[float]:
    """Generate entropy sequence for testing SEC dynamics."""
    if trend == "collapse":
        # Decreasing entropy (crystallization)
        return [0.9 - (i * 0.8 / length) for i in range(length)]
    elif trend == "exploration":
        # Increasing entropy (exploration)
        return [0.1 + (i * 0.8 / length) for i in range(length)]
    elif trend == "oscillation":
        # Oscillating entropy (complex dynamics)
        return [0.5 + 0.3 * np.sin(i * np.pi / 4) for i in range(length)]
    else:
        raise ValueError(f"Unknown entropy trend: {trend}")


def generate_field_hierarchy(levels: int = 3) -> List[Dict[str, Any]]:
    """Generate hierarchical field structures for MED testing."""
    hierarchy = []
    for level in range(levels):
        scale = TestConfig.MED_MACRO_SCALE // (2 ** level)
        field_data = {
            "level": level,
            "scale": scale,
            "entropy": TestConfig.SEC_TRANSITION + (level * 0.1),
            "structures": [f"structure_{i}" for i in range(scale // 10)]
        }
        hierarchy.append(field_data)
    return hierarchy


# Mock and Patch Utilities

@pytest.fixture
def mock_entropy_calculator():
    """Mock entropy calculator for controlled testing."""
    mock = Mock()
    mock.calculate.return_value = TestConfig.SEC_TRANSITION
    mock.track_evolution.return_value = generate_entropy_sequence(5, "collapse")
    return mock


@pytest.fixture
def mock_field_dynamics():
    """Mock field dynamics for isolation testing."""
    mock = Mock()
    mock.apply_pressure.return_value = {"pressure": 0.5, "stability": True}
    mock.detect_emergence.return_value = {"emerged": True, "pattern": "bifractal"}
    return mock


# Test Markers

# Use these markers to categorize tests
# pytest -m "sec_compliance" to run only SEC tests
# pytest -m "performance" to run only performance tests

sec_compliance = pytest.mark.sec_compliance
med_compliance = pytest.mark.med_compliance  
bifractal_symmetry = pytest.mark.bifractal_symmetry
recursive_balance = pytest.mark.recursive_balance
performance = pytest.mark.performance
integration = pytest.mark.integration
foundational_theory = pytest.mark.foundational_theory
