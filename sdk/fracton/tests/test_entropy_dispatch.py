"""
Test suite for EntropyDispatcher - context-aware function routing.

Tests validate dispatcher behavior against foundational theory requirements:
- SEC (Symbolic Entropy Collapse) routing patterns
- Context-sensitive function selection
- Entropy-driven dispatch logic
- Performance and routing efficiency
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Any, Dict, Callable, List

from fracton.core.entropy_dispatch import EntropyDispatcher
from fracton.lang.context import Context
from fracton.core.memory_field import MemoryField
from .conftest import (
    TestConfig, assert_sec_compliance, create_sec_test_function,
    sec_compliance, performance, foundational_theory
)


class TestEntropyDispatcher:
    """Test suite for entropy-aware function dispatch system."""

    def test_dispatcher_initialization(self):
        """Test basic dispatcher initialization."""
        dispatcher = EntropyDispatcher()
        
        assert dispatcher.get_route_count() == 0
        assert dispatcher.is_empty()
        assert dispatcher.default_strategy == "entropy_weighted"

    def test_dispatcher_with_custom_strategy(self):
        """Test dispatcher with custom routing strategy."""
        dispatcher = EntropyDispatcher(default_strategy="highest_entropy")
        
        assert dispatcher.default_strategy == "highest_entropy"
        assert dispatcher.get_route_count() == 0

    def test_function_registration(self):
        """Test function registration with entropy conditions."""
        dispatcher = EntropyDispatcher()
        
        def low_entropy_func(memory, context):
            return "low_entropy_result"
        
        def high_entropy_func(memory, context):
            return "high_entropy_result"
        
        # Register functions with entropy ranges
        dispatcher.register(low_entropy_func, entropy_min=0.0, entropy_max=0.5)
        dispatcher.register(high_entropy_func, entropy_min=0.5, entropy_max=1.0)
        
        assert dispatcher.get_route_count() == 2
        assert not dispatcher.is_empty()

    @sec_compliance
    def test_sec_entropy_based_routing(self, balanced_memory_field):
        """Test SEC-compliant entropy-based function routing."""
        dispatcher = EntropyDispatcher()
        
        # Define functions for different entropy regimes
        def crystallization_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Function for low entropy (crystallized) states."""
            return {
                "regime": "crystallization",
                "entropy": context.entropy,
                "operation": "structure_formation"
            }
        
        def exploration_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Function for high entropy (exploratory) states."""
            return {
                "regime": "exploration", 
                "entropy": context.entropy,
                "operation": "space_search"
            }
        
        def transition_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Function for mid-range entropy (transition) states."""
            return {
                "regime": "transition",
                "entropy": context.entropy,
                "operation": "phase_change"
            }
        
        # Register functions with appropriate entropy ranges
        dispatcher.register(crystallization_function, entropy_min=0.0, entropy_max=0.3)
        dispatcher.register(transition_function, entropy_min=0.3, entropy_max=0.7)
        dispatcher.register(exploration_function, entropy_min=0.7, entropy_max=1.0)
        
        # Test routing based on entropy
        low_context = Context(entropy=0.2, depth=0)
        mid_context = Context(entropy=0.5, depth=0)
        high_context = Context(entropy=0.8, depth=0)
        
        low_result = dispatcher.dispatch(balanced_memory_field, low_context)
        mid_result = dispatcher.dispatch(balanced_memory_field, mid_context)
        high_result = dispatcher.dispatch(balanced_memory_field, high_context)
        
        # Verify correct routing
        assert low_result["regime"] == "crystallization"
        assert mid_result["regime"] == "transition"
        assert high_result["regime"] == "exploration"
        
        # Verify SEC compliance patterns
        assert low_result["operation"] == "structure_formation"  # Low entropy → structure
        assert high_result["operation"] == "space_search"       # High entropy → exploration

    def test_context_aware_routing(self, balanced_memory_field):
        """Test routing based on multiple context factors."""
        dispatcher = EntropyDispatcher()
        
        def shallow_function(memory: MemoryField, context: Context) -> str:
            return f"shallow_depth_{context.depth}"
        
        def deep_function(memory: MemoryField, context: Context) -> str:
            return f"deep_depth_{context.depth}"
        
        def experimental_function(memory: MemoryField, context: Context) -> str:
            return f"experimental_{context.experiment}"
        
        # Register with complex conditions
        dispatcher.register(shallow_function, 
                          entropy_min=0.0, entropy_max=1.0,
                          depth_max=3)
        dispatcher.register(deep_function,
                          entropy_min=0.0, entropy_max=1.0, 
                          depth_min=3)
        dispatcher.register(experimental_function,
                          entropy_min=0.0, entropy_max=1.0,
                          experiment_pattern="special_*")
        
        # Test different contexts
        shallow_context = Context(entropy=0.5, depth=1, experiment="normal_test")
        deep_context = Context(entropy=0.5, depth=5, experiment="normal_test")
        special_context = Context(entropy=0.5, depth=2, experiment="special_experiment")
        
        shallow_result = dispatcher.dispatch(balanced_memory_field, shallow_context)
        deep_result = dispatcher.dispatch(balanced_memory_field, deep_context)
        special_result = dispatcher.dispatch(balanced_memory_field, special_context)
        
        assert shallow_result == "shallow_depth_1"
        assert deep_result == "deep_depth_5"
        assert special_result == "experimental_special_experiment"

    def test_priority_based_routing(self, balanced_memory_field):
        """Test function routing with priority levels."""
        dispatcher = EntropyDispatcher()
        
        def high_priority_func(memory: MemoryField, context: Context) -> str:
            return "high_priority"
        
        def medium_priority_func(memory: MemoryField, context: Context) -> str:
            return "medium_priority"
        
        def low_priority_func(memory: MemoryField, context: Context) -> str:
            return "low_priority"
        
        # Register with overlapping conditions but different priorities
        dispatcher.register(low_priority_func, entropy_min=0.0, entropy_max=1.0, priority=1)
        dispatcher.register(medium_priority_func, entropy_min=0.0, entropy_max=1.0, priority=5)
        dispatcher.register(high_priority_func, entropy_min=0.0, entropy_max=1.0, priority=10)
        
        # Should route to highest priority function
        context = Context(entropy=0.5, depth=0)
        result = dispatcher.dispatch(balanced_memory_field, context)
        
        assert result == "high_priority"

    def test_fallback_routing(self, balanced_memory_field):
        """Test fallback routing when no specific match found."""
        dispatcher = EntropyDispatcher()
        
        def specific_function(memory: MemoryField, context: Context) -> str:
            return "specific_match"
        
        def fallback_function(memory: MemoryField, context: Context) -> str:
            return "fallback_match"
        
        # Register specific function with narrow conditions
        dispatcher.register(specific_function, entropy_min=0.8, entropy_max=0.9)
        
        # Register fallback function
        dispatcher.set_fallback(fallback_function)
        
        # Test specific match
        specific_context = Context(entropy=0.85, depth=0)
        specific_result = dispatcher.dispatch(balanced_memory_field, specific_context)
        assert specific_result == "specific_match"
        
        # Test fallback match
        fallback_context = Context(entropy=0.5, depth=0)
        fallback_result = dispatcher.dispatch(balanced_memory_field, fallback_context)
        assert fallback_result == "fallback_match"

    @foundational_theory
    def test_sec_collapse_routing_pattern(self, balanced_memory_field):
        """Test SEC collapse patterns through routing."""
        dispatcher = EntropyDispatcher()
        entropy_history = []
        
        def high_entropy_explorer(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """High entropy exploration function."""
            entropy_history.append(context.entropy)
            # Slightly reduce entropy (moving toward collapse)
            new_entropy = context.entropy * 0.95
            memory.set("last_entropy", new_entropy)
            return {"phase": "exploration", "entropy_trend": "decreasing"}
        
        def mid_entropy_transition(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Mid entropy transition function."""
            entropy_history.append(context.entropy)
            # Moderate entropy reduction (transition phase)
            new_entropy = context.entropy * 0.85
            memory.set("last_entropy", new_entropy)
            return {"phase": "transition", "entropy_trend": "collapsing"}
        
        def low_entropy_crystallizer(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Low entropy crystallization function."""
            entropy_history.append(context.entropy)
            # Finalize collapse
            memory.set("last_entropy", 0.1)
            return {"phase": "crystallization", "entropy_trend": "collapsed"}
        
        # Register SEC-aligned functions
        dispatcher.register(high_entropy_explorer, entropy_min=0.7, entropy_max=1.0)
        dispatcher.register(mid_entropy_transition, entropy_min=0.3, entropy_max=0.7)
        dispatcher.register(low_entropy_crystallizer, entropy_min=0.0, entropy_max=0.3)
        
        # Simulate SEC collapse sequence
        entropies = [0.9, 0.6, 0.2]  # Decreasing entropy sequence
        results = []
        
        for entropy in entropies:
            context = Context(entropy=entropy, depth=0)
            result = dispatcher.dispatch(balanced_memory_field, context)
            results.append(result)
        
        # Verify SEC pattern: exploration → transition → crystallization
        assert results[0]["phase"] == "exploration"
        assert results[1]["phase"] == "transition"
        assert results[2]["phase"] == "crystallization"
        
        # Verify entropy trends
        for result in results:
            assert result["entropy_trend"] in ["decreasing", "collapsing", "collapsed"]
        
        # Verify overall collapse compliance
        initial_entropy = entropy_history[0]
        final_entropy = balanced_memory_field.get("last_entropy")
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")

    @performance
    def test_routing_performance(self, balanced_memory_field):
        """Test dispatcher performance with many registered functions."""
        dispatcher = EntropyDispatcher()
        
        # Register many functions with different conditions
        for i in range(100):
            entropy_min = i / 100.0
            entropy_max = (i + 1) / 100.0
            
            def indexed_function(memory, context, index=i):
                return f"function_{index}"
            
            dispatcher.register(indexed_function, 
                              entropy_min=entropy_min, 
                              entropy_max=entropy_max)
        
        # Measure routing performance
        start_time = time.time()
        
        # Perform many dispatch operations
        for i in range(50):
            entropy = i / 50.0
            context = Context(entropy=entropy, depth=0)
            result = dispatcher.dispatch(balanced_memory_field, context)
            assert result.startswith("function_")
        
        end_time = time.time()
        dispatch_time = end_time - start_time
        
        # Should complete within performance threshold
        assert dispatch_time < TestConfig.PERFORMANCE_TIMEOUT
        
        # Average dispatch time should be reasonable
        avg_dispatch_time = dispatch_time / 50
        assert avg_dispatch_time < 0.01  # Less than 10ms per dispatch

    def test_dispatcher_error_handling(self, balanced_memory_field):
        """Test error handling in dispatch operations."""
        dispatcher = EntropyDispatcher()
        
        def error_function(memory: MemoryField, context: Context):
            raise ValueError("Intentional test error")
        
        def safe_function(memory: MemoryField, context: Context):
            return "safe_result"
        
        # Register both functions
        dispatcher.register(error_function, entropy_min=0.0, entropy_max=0.5)
        dispatcher.register(safe_function, entropy_min=0.5, entropy_max=1.0)
        
        # Test error handling
        error_context = Context(entropy=0.3, depth=0)
        
        with pytest.raises(ValueError, match="Intentional test error"):
            dispatcher.dispatch(balanced_memory_field, error_context)
        
        # Test that dispatcher recovers
        safe_context = Context(entropy=0.7, depth=0)
        result = dispatcher.dispatch(balanced_memory_field, safe_context)
        assert result == "safe_result"

    def test_routing_statistics(self, balanced_memory_field):
        """Test dispatcher statistics and analytics."""
        dispatcher = EntropyDispatcher(track_statistics=True)
        
        def function_a(memory: MemoryField, context: Context):
            return "result_a"
        
        def function_b(memory: MemoryField, context: Context):
            return "result_b"
        
        dispatcher.register(function_a, entropy_min=0.0, entropy_max=0.5)
        dispatcher.register(function_b, entropy_min=0.5, entropy_max=1.0)
        
        # Perform multiple dispatches
        contexts = [
            Context(entropy=0.3, depth=0),  # Should route to function_a
            Context(entropy=0.7, depth=0),  # Should route to function_b
            Context(entropy=0.2, depth=0),  # Should route to function_a
            Context(entropy=0.8, depth=0),  # Should route to function_b
            Context(entropy=0.1, depth=0),  # Should route to function_a
        ]
        
        for context in contexts:
            dispatcher.dispatch(balanced_memory_field, context)
        
        # Check statistics
        stats = dispatcher.get_statistics()
        
        assert stats["total_dispatches"] == 5
        assert stats["function_usage"]["function_a"] == 3
        assert stats["function_usage"]["function_b"] == 2
        assert "average_dispatch_time" in stats
        assert stats["entropy_distribution"]["low"] == 3   # entropy < 0.5
        assert stats["entropy_distribution"]["high"] == 2  # entropy >= 0.5

    def test_dynamic_function_modification(self, balanced_memory_field):
        """Test dynamic modification of registered functions."""
        dispatcher = EntropyDispatcher()
        
        def original_function(memory: MemoryField, context: Context):
            return "original_result"
        
        def updated_function(memory: MemoryField, context: Context):
            return "updated_result"
        
        # Register original function
        function_id = dispatcher.register(original_function, entropy_min=0.0, entropy_max=1.0)
        
        # Test original behavior
        context = Context(entropy=0.5, depth=0)
        result = dispatcher.dispatch(balanced_memory_field, context)
        assert result == "original_result"
        
        # Update function
        dispatcher.update_function(function_id, updated_function)
        
        # Test updated behavior
        result = dispatcher.dispatch(balanced_memory_field, context)
        assert result == "updated_result"
        
        # Remove function
        dispatcher.unregister(function_id)
        
        # Should now have no matching functions (fallback or error)
        with pytest.raises(Exception):  # No matching function error
            dispatcher.dispatch(balanced_memory_field, context)

    def test_conditional_routing_chains(self, balanced_memory_field):
        """Test complex conditional routing chains."""
        dispatcher = EntropyDispatcher()
        
        def chain_start(memory: MemoryField, context: Context) -> str:
            memory.set("chain_step", 1)
            return "chain_started"
        
        def chain_middle(memory: MemoryField, context: Context) -> str:
            step = memory.get("chain_step", 0)
            memory.set("chain_step", step + 1)
            return "chain_continued"
        
        def chain_end(memory: MemoryField, context: Context) -> str:
            step = memory.get("chain_step", 0)
            memory.set("chain_step", step + 1)
            return "chain_completed"
        
        # Register chain functions with memory-dependent conditions
        dispatcher.register_conditional(
            chain_start,
            condition=lambda memory, context: memory.get("chain_step", 0) == 0
        )
        dispatcher.register_conditional(
            chain_middle,
            condition=lambda memory, context: memory.get("chain_step", 0) == 1
        )
        dispatcher.register_conditional(
            chain_end,
            condition=lambda memory, context: memory.get("chain_step", 0) == 2
        )
        
        # Execute chain
        context = Context(entropy=0.5, depth=0)
        
        result1 = dispatcher.dispatch(balanced_memory_field, context)
        assert result1 == "chain_started"
        assert balanced_memory_field.get("chain_step") == 1
        
        result2 = dispatcher.dispatch(balanced_memory_field, context)
        assert result2 == "chain_continued"
        assert balanced_memory_field.get("chain_step") == 2
        
        result3 = dispatcher.dispatch(balanced_memory_field, context)
        assert result3 == "chain_completed"
        assert balanced_memory_field.get("chain_step") == 3
