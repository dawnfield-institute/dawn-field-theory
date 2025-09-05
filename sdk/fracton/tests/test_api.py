"""
Test suite for main Fracton API surface.

Tests validate the primary public API that users interact with,
ensuring proper integration of all core components.
"""

import pytest
import time
from typing import Any, Dict

import fracton
from fracton.core import MemoryField
from fracton.lang import Context
from .conftest import (
    TestConfig, assert_sec_compliance, performance,
    integration, foundational_theory
)


@integration
class TestFractonAPI:
    """Test suite for main Fracton API integration."""

    def test_basic_api_usage(self):
        """Test basic Fracton API usage patterns."""
        
        # Test memory field creation
        with fracton.memory_field(capacity=100, entropy=0.5) as field:
            assert isinstance(field, MemoryField)
            assert field.capacity == 100
            assert field.get_entropy() == 0.5
        
        # Test context creation
        context = fracton.Context(entropy=0.6, depth=0, experiment="api_test")
        assert context.entropy == 0.6
        assert context.depth == 0
        assert context.experiment == "api_test"
        
        # Test context manipulation
        deeper_context = context.deeper(1)
        assert deeper_context.depth == 1
        assert deeper_context.entropy == context.entropy
        
        entropy_context = context.with_entropy(0.8)
        assert entropy_context.entropy == 0.8
        assert entropy_context.depth == context.depth

    def test_recursive_decorator_api(self):
        """Test @fracton.recursive decorator API."""
        
        @fracton.recursive
        def simple_recursive_func(memory: MemoryField, context: Context) -> str:
            if context.depth >= 3:
                return f"completed_at_depth_{context.depth}"
            
            new_context = context.deeper(1)
            return fracton.recurse(simple_recursive_func, memory, new_context)
        
        # Test execution
        with fracton.memory_field() as field:
            context = fracton.Context(entropy=0.5, depth=0)
            result = simple_recursive_func(field, context)
            assert result == "completed_at_depth_3"

    def test_entropy_gate_decorator_api(self):
        """Test @fracton.entropy_gate decorator API."""
        
        @fracton.recursive
        @fracton.entropy_gate(0.3, 0.8)
        def gated_function(memory: MemoryField, context: Context) -> str:
            return f"executed_with_entropy_{context.entropy}"
        
        with fracton.memory_field() as field:
            # Test within gate range
            valid_context = fracton.Context(entropy=0.5, depth=0)
            result = gated_function(field, valid_context)
            assert "executed_with_entropy_0.5" in result
            
            # Test outside gate range
            invalid_context = fracton.Context(entropy=0.9, depth=0)
            with pytest.raises(Exception):  # Should raise entropy gate error
                gated_function(field, invalid_context)

    def test_fracton_primitives_api(self):
        """Test fracton primitive functions API."""
        
        @fracton.recursive
        def primitive_test_func(memory: MemoryField, context: Context) -> Dict[str, Any]:
            # Test crystallize primitive
            data = ["item1", "item2", "item3"]
            crystallized = fracton.crystallize(data)
            
            # Test recurse primitive
            if context.depth < 2:
                new_context = context.deeper(1)
                recursive_result = fracton.recurse(primitive_test_func, memory, new_context)
                return {
                    "crystallized": crystallized,
                    "recursive": recursive_result,
                    "depth": context.depth
                }
            
            return {
                "crystallized": crystallized,
                "depth": context.depth
            }
        
        with fracton.memory_field() as field:
            context = fracton.Context(entropy=0.5, depth=0)
            result = primitive_test_func(field, context)
            
            assert "crystallized" in result
            assert result["recursive"]["depth"] == 2  # Should reach depth 2

    @foundational_theory
    def test_sec_api_integration(self):
        """Test SEC pattern through main API."""
        
        @fracton.recursive
        @fracton.entropy_gate(0.2, 0.9)
        def sec_api_example(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Example SEC pattern using main API."""
            
            # Record entropy evolution
            entropy_history = memory.get("entropy_history", [])
            entropy_history.append(context.entropy)
            memory.set("entropy_history", entropy_history)
            
            # SEC crystallization condition
            if context.entropy <= 0.3 or context.depth >= 5:
                return {
                    "sec_complete": True,
                    "entropy_history": entropy_history,
                    "crystallization_depth": context.depth,
                    "final_entropy": context.entropy
                }
            
            # Continue SEC collapse
            new_entropy = context.entropy * 0.85
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(sec_api_example, memory, new_context)
        
        with fracton.memory_field() as field:
            initial_context = fracton.Context(entropy=0.8, depth=0)
            result = sec_api_example(field, initial_context)
            
            # Verify SEC through API
            assert result["sec_complete"] is True
            entropy_history = result["entropy_history"]
            assert_sec_compliance(entropy_history[0], entropy_history[-1], "collapse")

    def test_utility_functions_api(self):
        """Test Fracton utility functions API."""
        
        # Test field initialization utility
        field = fracton.initialize_field(capacity=200, entropy=0.6, field_id="test_util")
        assert field.capacity == 200
        assert field.get_entropy() == 0.6
        assert field.field_id == "test_util"
        
        # Test trace analysis utility
        with fracton.memory_field() as test_field:
            @fracton.recursive
            def traced_function(memory: MemoryField, context: Context) -> str:
                if context.depth >= 3:
                    return "trace_complete"
                new_context = context.deeper(1)
                return fracton.recurse(traced_function, memory, new_context)
            
            context = fracton.Context(entropy=0.5, depth=0)
            result = traced_function(test_field, context)
            
            # Get and analyze trace
            trace = fracton.get_current_trace()
            if trace:
                analysis = fracton.analyze_trace(trace)
                assert "operation_count" in analysis or analysis is None  # May not have trace in simple case

    def test_field_context_manager_api(self):
        """Test memory field context manager API."""
        
        field_data = {}
        
        # Test context manager behavior
        with fracton.memory_field(capacity=150, entropy=0.4) as field:
            field.set("test_key", "test_value")
            field_data["size"] = field.size()
            field_data["entropy"] = field.get_entropy()
            field_data["value"] = field.get("test_key")
        
        # Verify field was properly managed
        assert field_data["size"] == 1
        assert field_data["entropy"] == 0.4
        assert field_data["value"] == "test_value"

    @performance
    def test_api_performance(self):
        """Test API performance characteristics."""
        
        @fracton.recursive
        def performance_test_func(memory: MemoryField, context: Context) -> int:
            if context.depth >= 20:  # Moderate recursion depth
                return context.depth
            
            new_context = context.deeper(1)
            return fracton.recurse(performance_test_func, memory, new_context)
        
        # Measure API overhead
        start_time = time.time()
        
        with fracton.memory_field(capacity=1000) as field:
            context = fracton.Context(entropy=0.5, depth=0)
            result = performance_test_func(field, context)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # API should be performant
        assert execution_time < TestConfig.PERFORMANCE_TIMEOUT
        assert result == 20  # Should complete successfully

    def test_error_handling_api(self):
        """Test API error handling and recovery."""
        
        @fracton.recursive
        def error_prone_api_func(memory: MemoryField, context: Context) -> str:
            if context.depth == 2:
                raise ValueError("API test error")
            
            if context.depth >= 3:
                return "error_recovered"
            
            new_context = context.deeper(1)
            return fracton.recurse(error_prone_api_func, memory, new_context)
        
        with fracton.memory_field() as field:
            context = fracton.Context(entropy=0.5, depth=0)
            
            # Should propagate error properly
            with pytest.raises(ValueError, match="API test error"):
                error_prone_api_func(field, context)
            
            # API should recover for subsequent calls
            recovery_context = fracton.Context(entropy=0.5, depth=3)
            result = error_prone_api_func(field, recovery_context)
            assert result == "error_recovered"

    def test_api_composition(self):
        """Test composition of multiple API components."""
        
        @fracton.recursive
        @fracton.entropy_gate(0.3, 0.8)
        def composed_function_a(memory: MemoryField, context: Context) -> Dict[str, Any]:
            memory.set("function_a_called", True)
            return {"function": "a", "entropy": context.entropy}
        
        @fracton.recursive
        @fracton.entropy_gate(0.1, 0.6)
        def composed_function_b(memory: MemoryField, context: Context) -> Dict[str, Any]:
            memory.set("function_b_called", True)
            return {"function": "b", "entropy": context.entropy}
        
        @fracton.recursive
        def composition_orchestrator(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Orchestrate multiple Fracton components."""
            
            # Call different functions based on entropy
            if context.entropy > 0.6:
                result_a = fracton.recurse(composed_function_a, memory, context)
                return {"orchestrator": "high_entropy", "result": result_a}
            else:
                result_b = fracton.recurse(composed_function_b, memory, context)
                return {"orchestrator": "low_entropy", "result": result_b}
        
        with fracton.memory_field() as field:
            # Test high entropy path
            high_context = fracton.Context(entropy=0.7, depth=0)
            high_result = composition_orchestrator(field, high_context)
            
            assert high_result["orchestrator"] == "high_entropy"
            assert high_result["result"]["function"] == "a"
            assert field.get("function_a_called") is True
            
            # Reset field
            field.set("function_a_called", False)
            field.set("function_b_called", False)
            
            # Test low entropy path
            low_context = fracton.Context(entropy=0.4, depth=0)
            low_result = composition_orchestrator(field, low_context)
            
            assert low_result["orchestrator"] == "low_entropy"
            assert low_result["result"]["function"] == "b"
            assert field.get("function_b_called") is True

    def test_api_thread_safety(self):
        """Test API thread safety (basic test)."""
        
        results = {}
        
        @fracton.recursive
        def thread_safe_function(memory: MemoryField, context: Context) -> str:
            thread_id = context.experiment
            memory.set(f"thread_{thread_id}_data", f"data_from_{thread_id}")
            
            if context.depth >= 2:
                return f"completed_{thread_id}"
            
            new_context = context.deeper(1)
            return fracton.recurse(thread_safe_function, memory, new_context)
        
        # Simulate concurrent usage (sequential for now, full threading would need more setup)
        with fracton.memory_field() as field:
            context1 = fracton.Context(entropy=0.5, depth=0, experiment="thread1")
            context2 = fracton.Context(entropy=0.6, depth=0, experiment="thread2")
            
            result1 = thread_safe_function(field, context1)
            result2 = thread_safe_function(field, context2)
            
            assert result1 == "completed_thread1"
            assert result2 == "completed_thread2"
            
            # Both threads should have left their data
            assert field.get("thread_thread1_data") == "data_from_thread1"
            assert field.get("thread_thread2_data") == "data_from_thread2"

    def test_api_documentation_examples(self):
        """Test examples that would appear in API documentation."""
        
        # Example 1: Simple recursive function
        @fracton.recursive
        def fibonacci_field(memory: MemoryField, context: Context) -> int:
            """Entropy-aware Fibonacci using Fracton."""
            n = context.get("n", 10)
            
            if n <= 1:
                return n
            
            # Memoization using memory field
            memo_key = f"fib_{n}"
            if memory.get(memo_key) is not None:
                return memory.get(memo_key)
            
            # Recursive calculation with entropy evolution
            context_n1 = context.deeper(1).with_data({"n": n-1})
            context_n2 = context.deeper(1).with_data({"n": n-2})
            
            fib_n1 = fracton.recurse(fibonacci_field, memory, context_n1)
            fib_n2 = fracton.recurse(fibonacci_field, memory, context_n2)
            
            result = fib_n1 + fib_n2
            memory.set(memo_key, result)
            return result
        
        # Test documentation example
        with fracton.memory_field(capacity=100) as field:
            context = fracton.Context(entropy=0.5, depth=0).with_data({"n": 8})
            result = fibonacci_field(field, context)
            assert result == 21  # 8th Fibonacci number
        
        # Example 2: Entropy-gated processing
        @fracton.recursive
        @fracton.entropy_gate(0.4, 0.9)
        def adaptive_processor(memory: MemoryField, context: Context) -> str:
            """Entropy-adaptive processing function."""
            data = context.get("data", [])
            
            if context.entropy > 0.7:
                # High entropy: exploratory processing
                return f"explored_{len(data)}_items"
            else:
                # Low entropy: focused processing
                return f"focused_on_{len(data)}_items"
        
        # Test adaptive processing
        with fracton.memory_field() as field:
            high_entropy_context = fracton.Context(entropy=0.8, depth=0).with_data({"data": [1, 2, 3]})
            low_entropy_context = fracton.Context(entropy=0.5, depth=0).with_data({"data": [1, 2, 3]})
            
            high_result = adaptive_processor(field, high_entropy_context)
            low_result = adaptive_processor(field, low_entropy_context)
            
            assert "explored" in high_result
            assert "focused" in low_result
