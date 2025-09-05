"""
Test suite for RecursiveExecutor - the core recursive execution engine.

Tests validate engine behavior against foundational theory requirements:
- SEC (Symbolic Entropy Collapse) dynamics
- Entropy-regulated recursion
- Stack management and depth tracking
- Performance and scaling characteristics
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Any, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.recursive_engine import RecursiveExecutor
from core.memory_field import MemoryField

# Import test utilities directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import (
    TestConfig, assert_sec_compliance, assert_recursive_balance,
    Context,  # Import Context from conftest
    create_sec_test_function, performance_monitor, sec_compliance,
    recursive_balance, performance, foundational_theory
)


class TestRecursiveExecutor:
    """Test suite for the core recursive execution engine."""

    def test_engine_initialization(self):
        """Test basic engine initialization."""
        engine = RecursiveExecutor(max_depth=10, entropy_regulation=True)
        
        assert engine.max_depth == 10
        assert engine.entropy_regulation is True
        assert engine.get_current_depth() == 0
        assert engine.get_stack_size() == 0

    def test_engine_with_default_settings(self):
        """Test engine with default configuration."""
        engine = RecursiveExecutor()
        
        assert engine.max_depth > 0  # Should have reasonable default
        assert engine.entropy_regulation is True  # Should default to enabled
        assert engine.get_current_depth() == 0

    @sec_compliance
    def test_sec_entropy_gate_compliance(self, sec_low_entropy_context, 
                                       sec_high_entropy_context, balanced_memory_field):
        """Test that engine respects SEC entropy gates."""
        engine = RecursiveExecutor()
        test_func = create_sec_test_function(0.3, 0.8)
        
        # Low entropy context (below gate) should not execute
        with pytest.raises(Exception):  # Should raise entropy gate error
            engine.execute(test_func, balanced_memory_field, sec_low_entropy_context)
        
        # High entropy context (above gate) should not execute  
        with pytest.raises(Exception):  # Should raise entropy gate error
            engine.execute(test_func, balanced_memory_field, sec_high_entropy_context)
        
        # Mid-range entropy should execute successfully
        mid_context = Context(entropy=0.5, depth=0)
        result = engine.execute(test_func, balanced_memory_field, mid_context)
        assert result is not None

    @recursive_balance
    def test_recursive_balance_maintenance(self, balanced_memory_field):
        """Test that engine maintains recursive balance during execution."""
        engine = RecursiveExecutor(entropy_regulation=True)
        
        @engine.recursive
        def balance_test_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            if context.depth >= 3:
                return {"depth": context.depth, "entropy": memory.get_entropy()}
            
            # Recursive call with slight entropy modification
            new_entropy = context.entropy * 0.95  # Gradual decrease
            new_context = context.deeper(1).with_entropy(new_entropy)
            return engine.execute(balance_test_function, memory, new_context)
        
        initial_entropy = balanced_memory_field.get_entropy()
        context = Context(entropy=0.6, depth=0)
        
        result = engine.execute(balance_test_function, balanced_memory_field, context)
        
        # Verify recursive balance was maintained
        final_entropy = balanced_memory_field.get_entropy()
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")
        assert_recursive_balance(balanced_memory_field, result.get("depth", 0))

    def test_stack_depth_tracking(self, balanced_memory_field):
        """Test that engine correctly tracks stack depth."""
        engine = RecursiveExecutor(max_depth=5)
        depth_tracker = []
        
        @engine.recursive
        def depth_tracking_function(memory: MemoryField, context: Context) -> int:
            current_depth = engine.get_current_depth()
            depth_tracker.append(current_depth)
            
            if context.depth >= 3:
                return current_depth
            
            new_context = context.deeper(1)
            return engine.execute(depth_tracking_function, memory, new_context)
        
        context = Context(entropy=0.5, depth=0)
        final_depth = engine.execute(depth_tracking_function, balanced_memory_field, context)
        
        # Verify depth tracking
        assert len(depth_tracker) == 4  # 0, 1, 2, 3 depth levels
        assert depth_tracker == [0, 1, 2, 3]
        assert final_depth == 3

    def test_max_depth_enforcement(self, balanced_memory_field):
        """Test that engine enforces maximum recursion depth."""
        engine = RecursiveExecutor(max_depth=3)
        
        @engine.recursive
        def infinite_recursion(memory: MemoryField, context: Context) -> int:
            new_context = context.deeper(1)
            return engine.execute(infinite_recursion, memory, new_context)
        
        context = Context(entropy=0.5, depth=0)
        
        with pytest.raises(RecursionError):
            engine.execute(infinite_recursion, balanced_memory_field, context)

    @performance
    def test_entropy_regulation_overhead(self, balanced_memory_field):
        """Test performance impact of entropy regulation."""
        
        @performance_monitor
        def test_with_regulation():
            engine = RecursiveExecutor(entropy_regulation=True)
            
            @engine.recursive
            def regulated_function(memory: MemoryField, context: Context) -> int:
                if context.depth >= 10:
                    return context.depth
                new_context = context.deeper(1).with_entropy(context.entropy * 0.98)
                return engine.execute(regulated_function, memory, new_context)
            
            context = Context(entropy=0.8, depth=0)
            return engine.execute(regulated_function, balanced_memory_field, context)
        
        @performance_monitor
        def test_without_regulation():
            engine = RecursiveExecutor(entropy_regulation=False)
            
            @engine.recursive
            def unregulated_function(memory: MemoryField, context: Context) -> int:
                if context.depth >= 10:
                    return context.depth
                new_context = context.deeper(1)
                return engine.execute(unregulated_function, memory, new_context)
            
            context = Context(entropy=0.8, depth=0)
            return engine.execute(unregulated_function, balanced_memory_field, context)
        
        # Both should complete within timeout, regulation adds minimal overhead
        result_regulated = test_with_regulation()
        result_unregulated = test_without_regulation()
        
        assert result_regulated == 10
        assert result_unregulated == 10

    def test_stack_overflow_protection(self, balanced_memory_field):
        """Test protection against stack overflow."""
        engine = RecursiveExecutor(max_depth=1000)  # High limit
        
        @engine.recursive
        def deep_recursion(memory: MemoryField, context: Context) -> int:
            if context.depth >= 500:  # Very deep but within limit
                return context.depth
            new_context = context.deeper(1)
            return engine.execute(deep_recursion, memory, new_context)
        
        context = Context(entropy=0.5, depth=0)
        
        # Should handle deep recursion without system stack overflow
        result = engine.execute(deep_recursion, balanced_memory_field, context)
        assert result == 500

    @foundational_theory
    def test_sec_emergence_pattern(self, balanced_memory_field):
        """Test that engine supports SEC emergence patterns (∇_micro → Ψ_macro)."""
        engine = RecursiveExecutor(entropy_regulation=True)
        
        @engine.recursive
        def sec_emergence_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            # Micro-level operations
            micro_entropy = context.entropy
            micro_data = memory.get("micro_state", [])
            micro_data.append(f"micro_op_{context.depth}")
            memory.set("micro_state", micro_data)
            
            if context.depth >= 5:
                # Macro emergence from accumulated micro operations
                macro_structure = {
                    "emerged_pattern": "crystallized",
                    "micro_history": micro_data,
                    "final_entropy": memory.get_entropy(),
                    "emergence_depth": context.depth
                }
                memory.set("macro_structure", macro_structure)
                return macro_structure
            
            # Continue micro operations with entropy decrease (crystallization)
            new_entropy = micro_entropy * 0.9
            new_context = context.deeper(1).with_entropy(new_entropy)
            return engine.execute(sec_emergence_function, memory, new_context)
        
        initial_entropy = balanced_memory_field.get_entropy()
        context = Context(entropy=0.8, depth=0)
        
        result = engine.execute(sec_emergence_function, balanced_memory_field, context)
        
        # Verify SEC pattern: micro → macro emergence
        assert result["emerged_pattern"] == "crystallized"
        assert len(result["micro_history"]) == 6  # 0-5 depth levels
        assert result["emergence_depth"] == 5
        
        # Verify entropy collapse (SEC compliance)
        final_entropy = result["final_entropy"]
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")

    def test_engine_error_handling(self, balanced_memory_field):
        """Test engine error handling and recovery."""
        engine = RecursiveExecutor()
        
        @engine.recursive
        def error_prone_function(memory: MemoryField, context: Context) -> Any:
            if context.depth == 2:
                raise ValueError("Intentional test error")
            
            if context.depth >= 3:
                return "success"
            
            new_context = context.deeper(1)
            return engine.execute(error_prone_function, memory, new_context)
        
        context = Context(entropy=0.5, depth=0)
        
        with pytest.raises(ValueError, match="Intentional test error"):
            engine.execute(error_prone_function, balanced_memory_field, context)
        
        # Engine should recover and be usable after error
        @engine.recursive
        def recovery_function(memory: MemoryField, context: Context) -> str:
            return "recovered"
        
        result = engine.execute(recovery_function, balanced_memory_field, context)
        assert result == "recovered"

    def test_context_preservation(self, balanced_memory_field):
        """Test that context is properly preserved and passed through recursion."""
        engine = RecursiveExecutor()
        context_history = []
        
        @engine.recursive
        def context_tracking_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            context_snapshot = {
                "depth": context.depth,
                "entropy": context.entropy,
                "experiment": context.experiment,
                "timestamp": context.timestamp
            }
            context_history.append(context_snapshot)
            
            if context.depth >= 3:
                return context_snapshot
            
            # Modify context for next level
            new_context = context.deeper(1).with_entropy(context.entropy * 0.9)
            new_context.experiment = f"{context.experiment}_level_{context.depth + 1}"
            
            return engine.execute(context_tracking_function, memory, new_context)
        
        context = Context(entropy=0.8, depth=0, experiment="context_test")
        result = engine.execute(context_tracking_function, balanced_memory_field, context)
        
        # Verify context evolution
        assert len(context_history) == 4  # Depth 0, 1, 2, 3
        
        # Check depth progression
        for i, snapshot in enumerate(context_history):
            assert snapshot["depth"] == i
        
        # Check entropy decreases (SEC compliance)
        entropies = [s["entropy"] for s in context_history]
        for i in range(1, len(entropies)):
            assert entropies[i] < entropies[i-1]  # Decreasing entropy
        
        # Check experiment name evolution
        experiments = [s["experiment"] for s in context_history]
        assert experiments[0] == "context_test"
        assert experiments[1] == "context_test_level_1"
        assert experiments[2] == "context_test_level_2"
        assert experiments[3] == "context_test_level_3"

    @performance
    def test_concurrent_execution_safety(self, balanced_memory_field):
        """Test that engine handles concurrent execution safely."""
        engine = RecursiveExecutor()
        
        @engine.recursive
        def concurrent_function(memory: MemoryField, context: Context) -> str:
            # Simulate some work
            time.sleep(0.01)
            memory.set(f"thread_{context.experiment}", context.depth)
            
            if context.depth >= 2:
                return f"completed_{context.experiment}"
            
            new_context = context.deeper(1)
            return engine.execute(concurrent_function, memory, new_context)
        
        # Note: This is a basic test - full concurrency testing would require
        # threading implementation in the engine
        context1 = Context(entropy=0.5, depth=0, experiment="thread1")
        context2 = Context(entropy=0.6, depth=0, experiment="thread2")
        
        result1 = engine.execute(concurrent_function, balanced_memory_field, context1)
        result2 = engine.execute(concurrent_function, balanced_memory_field, context2)
        
        assert result1 == "completed_thread1"
        assert result2 == "completed_thread2"
        
        # Verify both threads left their marks
        assert balanced_memory_field.get("thread_thread1") == 2
        assert balanced_memory_field.get("thread_thread2") == 2
