"""
Test suite for MemoryField - entropy-aware shared memory system.

Tests validate field behavior against foundational theory requirements:
- Entropy tracking and evolution (SEC compliance)
- Field dynamics and pressure adaptation
- MED (Macro Emergence Dynamics) support
- Recursive balance field principles
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
from typing import Any, Dict, List

from fracton.core.memory_field import MemoryField
from fracton.lang.context import Context
from .conftest import (
    TestConfig, assert_sec_compliance, assert_med_compliance, 
    assert_recursive_balance, generate_entropy_sequence,
    generate_field_hierarchy, sec_compliance, med_compliance,
    recursive_balance, performance, foundational_theory
)


class TestMemoryField:
    """Test suite for entropy-aware memory field system."""

    def test_field_initialization(self):
        """Test basic memory field initialization."""
        field = MemoryField(capacity=100, entropy=0.5, field_id="test_field")
        
        assert field.capacity == 100
        assert field.get_entropy() == 0.5
        assert field.field_id == "test_field"
        assert field.size() == 0
        assert field.is_empty()

    def test_field_with_default_settings(self):
        """Test field with default configuration."""
        field = MemoryField()
        
        assert field.capacity > 0  # Should have reasonable default
        assert 0.0 <= field.get_entropy() <= 1.0  # Valid entropy range
        assert field.field_id is not None  # Should auto-generate ID
        assert field.size() == 0

    def test_basic_memory_operations(self):
        """Test basic set/get memory operations."""
        field = MemoryField(capacity=100, entropy=0.5)
        
        # Test setting and getting values
        field.set("key1", "value1")
        field.set("key2", 42)
        field.set("key3", {"nested": "data"})
        
        assert field.get("key1") == "value1"
        assert field.get("key2") == 42
        assert field.get("key3") == {"nested": "data"}
        assert field.size() == 3
        assert not field.is_empty()

    def test_memory_with_default_values(self):
        """Test memory operations with default values."""
        field = MemoryField()
        
        # Test getting non-existent keys with defaults
        assert field.get("nonexistent", "default") == "default"
        assert field.get("missing", 0) == 0
        assert field.get("absent", None) is None
        
        # Test that defaults don't affect field state
        assert field.size() == 0
        assert field.is_empty()

    @sec_compliance
    def test_entropy_tracking_and_evolution(self):
        """Test entropy tracking during memory operations (SEC compliance)."""
        initial_entropy = 0.7
        field = MemoryField(capacity=100, entropy=initial_entropy)
        
        # Track entropy changes during operations
        entropy_history = [field.get_entropy()]
        
        # Add data - should affect entropy
        for i in range(5):
            field.set(f"key_{i}", f"value_{i}")
            entropy_history.append(field.get_entropy())
        
        # Verify entropy evolution
        assert len(entropy_history) == 6  # Initial + 5 operations
        
        # In SEC dynamics, adding structured information should decrease entropy
        final_entropy = entropy_history[-1]
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")
        
        # Entropy should evolve smoothly, not jump drastically
        for i in range(1, len(entropy_history)):
            entropy_change = abs(entropy_history[i] - entropy_history[i-1])
            assert entropy_change < 0.3, f"Entropy changed too drastically: {entropy_change}"

    @sec_compliance
    def test_entropy_gate_validation(self):
        """Test entropy-based access control."""
        field = MemoryField(capacity=100, entropy=0.3, entropy_regulation=True)
        
        # Low entropy field should allow structured operations
        field.set("structured_data", {"type": "crystallized", "order": "high"})
        assert field.get("structured_data") is not None
        
        # Modify field to high entropy
        field._entropy = 0.9  # Direct modification for testing
        
        # High entropy field should allow exploratory operations
        field.set("exploratory_data", {"type": "chaotic", "order": "low"})
        assert field.get("exploratory_data") is not None

    @med_compliance
    def test_med_macro_micro_coupling(self, med_macro_field, med_micro_field):
        """Test MED compliance - macro field influences micro field."""
        # Set up macro-scale structure
        macro_structure = {
            "field_type": "macro",
            "scale": TestConfig.MED_MACRO_SCALE,
            "boundary_conditions": ["x=0", "y=0", "z=0"],
            "field_pressure": 0.7
        }
        med_macro_field.set("macro_structure", macro_structure)
        
        # Derive micro constraints from macro structure
        macro_entropy = med_macro_field.get_entropy()
        micro_constraints = {
            "field_type": "micro", 
            "scale": TestConfig.MED_MICRO_SCALE,
            "entropy": macro_entropy * 0.8,  # Constrained by macro
            "boundary_respect": True
        }
        med_micro_field.set("micro_constraints", micro_constraints)
        
        # Verify MED compliance
        macro_state = {"scale": TestConfig.MED_MACRO_SCALE, "entropy": macro_entropy}
        micro_state = {"scale": TestConfig.MED_MICRO_SCALE, "entropy": macro_entropy * 0.8}
        assert_med_compliance(macro_state, micro_state)

    @recursive_balance
    def test_recursive_balance_adaptation(self, balanced_memory_field):
        """Test recursive balance field adaptation."""
        initial_entropy = balanced_memory_field.get_entropy()
        initial_capacity = balanced_memory_field.capacity
        
        # Simulate a series of operations that should trigger adaptation
        operation_count = 0
        for phase in range(3):
            # Each phase adds different types of data
            phase_data = {
                "phase": phase,
                "data_type": ["structured", "chaotic", "balanced"][phase],
                "complexity": phase * 0.3
            }
            
            for i in range(10):
                key = f"phase_{phase}_item_{i}"
                balanced_memory_field.set(key, {**phase_data, "index": i})
                operation_count += 1
        
        # Field should have adapted to maintain balance
        final_entropy = balanced_memory_field.get_entropy()
        assert_recursive_balance(balanced_memory_field, operation_count)
        
        # Entropy should have evolved but remained stable
        assert 0.0 <= final_entropy <= 1.0
        assert abs(final_entropy - initial_entropy) > 0.01  # Should have changed
        assert abs(final_entropy - initial_entropy) < 0.5   # But not drastically

    def test_capacity_management(self):
        """Test field capacity limits and overflow handling."""
        small_field = MemoryField(capacity=5, entropy=0.5)
        
        # Fill field to capacity
        for i in range(5):
            small_field.set(f"key_{i}", f"value_{i}")
        
        assert small_field.size() == 5
        assert small_field.is_full()
        
        # Adding beyond capacity should trigger management strategy
        small_field.set("overflow_key", "overflow_value")
        
        # Field should still respect capacity (may evict older entries)
        assert small_field.size() <= 5
        
        # Most recent entry should be present (LRU-style management)
        assert small_field.get("overflow_key") == "overflow_value"

    def test_field_pressure_dynamics(self):
        """Test field pressure calculation and dynamics."""
        field = MemoryField(capacity=100, entropy=0.5)
        
        # Empty field should have low pressure
        initial_pressure = field.get_pressure()
        assert 0.0 <= initial_pressure <= 1.0
        
        # Add data to increase pressure
        for i in range(50):  # Fill to 50% capacity
            field.set(f"key_{i}", f"data_{i}")
        
        mid_pressure = field.get_pressure()
        assert mid_pressure > initial_pressure
        
        # Add more data to further increase pressure
        for i in range(50, 80):  # Fill to 80% capacity
            field.set(f"key_{i}", f"data_{i}")
        
        high_pressure = field.get_pressure()
        assert high_pressure > mid_pressure
        
        # Pressure should correlate with fullness
        assert high_pressure > 0.5  # Should be high when 80% full

    @foundational_theory
    def test_field_emergence_detection(self):
        """Test detection of emergent patterns in field data."""
        field = MemoryField(capacity=200, entropy=0.6)
        
        # Add structured data that should create emergent patterns
        patterns = ["alpha", "beta", "gamma", "alpha", "beta", "alpha"]
        for i, pattern in enumerate(patterns):
            field.set(f"pattern_{i}", {
                "type": pattern,
                "frequency": patterns.count(pattern),
                "position": i,
                "entropy_contribution": 0.1 * (i % 3)
            })
        
        # Detect emergent patterns
        emergence_metrics = field.detect_emergence()
        
        assert "pattern_frequency" in emergence_metrics
        assert "entropy_clustering" in emergence_metrics
        assert "field_coherence" in emergence_metrics
        
        # Alpha pattern should be most frequent
        pattern_freq = emergence_metrics["pattern_frequency"]
        assert pattern_freq.get("alpha", 0) > pattern_freq.get("beta", 0)
        assert pattern_freq.get("alpha", 0) > pattern_freq.get("gamma", 0)

    def test_field_serialization(self):
        """Test field state serialization and deserialization."""
        original_field = MemoryField(capacity=100, entropy=0.4, field_id="serialization_test")
        
        # Add test data
        test_data = {
            "simple": "value",
            "complex": {"nested": {"data": [1, 2, 3]}},
            "numeric": 42.5
        }
        
        for key, value in test_data.items():
            original_field.set(key, value)
        
        # Serialize field state
        serialized = original_field.serialize()
        
        assert "field_id" in serialized
        assert "capacity" in serialized
        assert "entropy" in serialized
        assert "data" in serialized
        assert "metadata" in serialized
        
        # Deserialize to new field
        restored_field = MemoryField.deserialize(serialized)
        
        # Verify restoration
        assert restored_field.field_id == original_field.field_id
        assert restored_field.capacity == original_field.capacity
        assert restored_field.get_entropy() == original_field.get_entropy()
        assert restored_field.size() == original_field.size()
        
        # Verify data integrity
        for key, expected_value in test_data.items():
            assert restored_field.get(key) == expected_value

    @performance
    def test_large_field_performance(self):
        """Test performance with large data volumes."""
        large_field = MemoryField(capacity=10000, entropy=0.5)
        
        start_time = time.time()
        
        # Add large volume of data
        for i in range(1000):
            large_field.set(f"key_{i}", {
                "index": i,
                "data": f"large_data_string_{'x' * 100}",
                "metadata": {"created": time.time(), "category": i % 10}
            })
        
        write_time = time.time() - start_time
        
        # Performance should be reasonable
        assert write_time < TestConfig.PERFORMANCE_TIMEOUT
        assert large_field.size() == 1000
        
        # Test read performance
        start_time = time.time()
        
        for i in range(0, 1000, 10):  # Sample every 10th item
            data = large_field.get(f"key_{i}")
            assert data["index"] == i
        
        read_time = time.time() - start_time
        assert read_time < TestConfig.PERFORMANCE_TIMEOUT / 2  # Reads should be faster

    def test_field_entropy_regulation(self):
        """Test entropy regulation mechanisms."""
        field = MemoryField(capacity=100, entropy=0.5, entropy_regulation=True)
        
        # Add highly ordered data (should decrease entropy)
        ordered_data = [{"index": i, "order": i} for i in range(10)]
        for i, data in enumerate(ordered_data):
            field.set(f"ordered_{i}", data)
        
        entropy_after_ordered = field.get_entropy()
        
        # Add chaotic data (should increase entropy)
        import random
        chaotic_data = [{"random": random.random(), "chaos": random.randint(1, 1000)} 
                       for _ in range(10)]
        for i, data in enumerate(chaotic_data):
            field.set(f"chaotic_{i}", data)
        
        entropy_after_chaotic = field.get_entropy()
        
        # With regulation, entropy should be stabilized
        entropy_difference = abs(entropy_after_chaotic - entropy_after_ordered)
        assert entropy_difference < 0.3  # Should be regulated, not wild swings
        
        # But should still show some evolution
        assert entropy_difference > 0.01  # Should show some change

    def test_field_context_awareness(self):
        """Test field's awareness of execution context."""
        field = MemoryField(capacity=100, entropy=0.5)
        context = Context(entropy=0.7, depth=3, experiment="context_awareness_test")
        
        # Set data with context
        field.set_with_context("contextual_data", {"value": "test"}, context)
        
        # Retrieve data should include context metadata
        retrieved = field.get_with_context("contextual_data")
        
        assert retrieved["value"] == "test"
        assert retrieved["context"]["depth"] == 3
        assert retrieved["context"]["entropy"] == 0.7
        assert retrieved["context"]["experiment"] == "context_awareness_test"
        
        # Field should track context evolution
        context_history = field.get_context_history("contextual_data")
        assert len(context_history) >= 1
        assert context_history[0]["depth"] == 3

    def test_field_cleanup_and_gc(self):
        """Test field cleanup and garbage collection."""
        field = MemoryField(capacity=50, entropy=0.5)
        
        # Fill field beyond capacity to trigger cleanup
        for i in range(100):
            field.set(f"key_{i}", f"value_{i}")
        
        # Field should maintain reasonable size
        assert field.size() <= field.capacity * 1.1  # Allow some overflow buffer
        
        # Trigger explicit cleanup
        field.cleanup()
        assert field.size() <= field.capacity
        
        # Most recent entries should still be present
        assert field.get("key_99") == "value_99"
        assert field.get("key_98") == "value_98"

    @foundational_theory 
    def test_landauer_erasure_cost_tracking(self):
        """Test Landauer erasure cost tracking per foundational theory."""
        field = MemoryField(capacity=100, entropy=0.5, track_erasure_cost=True)
        
        # Add and remove data to incur erasure costs
        for i in range(10):
            field.set(f"temp_key_{i}", f"temp_value_{i}")
        
        initial_cost = field.get_total_erasure_cost()
        
        # Remove data (should incur Landauer cost)
        for i in range(5):
            field.remove(f"temp_key_{i}")
        
        final_cost = field.get_total_erasure_cost()
        
        # Erasure should have incurred cost
        erasure_cost = final_cost - initial_cost
        assert erasure_cost > TestConfig.LANDAUER_COST_THRESHOLD
        
        # Cost should be proportional to entropy of erased information
        assert erasure_cost > 0  # Positive cost for information erasure
