"""
Test suite for BifractalTrace - operation recording and pattern analysis.

Tests validate trace behavior against foundational theory requirements:
- Bifractal time symmetry (backward ancestry ↔ forward emergence)
- Operation lineage tracking and analysis
- Pattern detection in recursive structures
- Temporal dynamics and causality chains
"""

import pytest
import time
import uuid
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Tuple

from fracton.core.bifractal_trace import BifractalTrace
from fracton.lang.context import Context
from fracton.core.memory_field import MemoryField
from .conftest import (
    TestConfig, assert_bifractal_symmetry, generate_entropy_sequence,
    bifractal_symmetry, performance, foundational_theory
)


class TestBifractalTrace:
    """Test suite for bifractal tracing and pattern analysis."""

    def test_trace_initialization(self):
        """Test basic trace initialization."""
        trace = BifractalTrace(
            ancestry_depth=TestConfig.BIFRACTAL_ANCESTRY_DEPTH,
            future_horizon=TestConfig.BIFRACTAL_FUTURE_HORIZON
        )
        
        assert trace.ancestry_depth == TestConfig.BIFRACTAL_ANCESTRY_DEPTH
        assert trace.future_horizon == TestConfig.BIFRACTAL_FUTURE_HORIZON
        assert trace.get_operation_count() == 0
        assert trace.is_empty()

    def test_trace_with_default_settings(self):
        """Test trace with default configuration."""
        trace = BifractalTrace()
        
        assert trace.ancestry_depth > 0  # Should have reasonable default
        assert trace.future_horizon > 0  # Should have reasonable default
        assert trace.get_operation_count() == 0

    def test_basic_operation_recording(self):
        """Test basic operation recording and retrieval."""
        trace = BifractalTrace()
        
        # Record a simple operation
        op_id = trace.record_operation(
            operation_type="test_operation",
            context=Context(entropy=0.5, depth=0),
            input_data={"test": "input"},
            output_data={"test": "output"}
        )
        
        assert op_id is not None
        assert trace.get_operation_count() == 1
        assert not trace.is_empty()
        
        # Retrieve operation
        operation = trace.get_operation(op_id)
        assert operation["operation_type"] == "test_operation"
        assert operation["input_data"] == {"test": "input"}
        assert operation["output_data"] == {"test": "output"}
        assert operation["context"]["entropy"] == 0.5

    @bifractal_symmetry
    def test_bifractal_temporal_symmetry(self):
        """Test bifractal temporal symmetry (backward ancestry ↔ forward emergence)."""
        trace = BifractalTrace(ancestry_depth=5, future_horizon=5)
        
        # Create a chain of operations to establish ancestry/emergence
        operation_chain = []
        
        for i in range(10):
            context = Context(entropy=0.5 - (i * 0.04), depth=i)  # Decreasing entropy
            
            op_id = trace.record_operation(
                operation_type=f"chain_operation_{i}",
                context=context,
                input_data={"step": i, "previous": operation_chain[-1] if operation_chain else None},
                output_data={"result": f"step_{i}_result"}
            )
            
            operation_chain.append(op_id)
            
            # Link to previous operation if exists
            if len(operation_chain) > 1:
                trace.link_operations(operation_chain[-2], op_id, "sequential")
        
        # Test bifractal symmetry for middle operation
        middle_op = operation_chain[5]  # Operation 5 (middle of chain)
        
        # Verify symmetry
        assert_bifractal_symmetry(trace, middle_op)
        
        # Test ancestry retrieval
        ancestry = trace.get_ancestry(middle_op)
        assert len(ancestry) <= 5  # Should respect ancestry_depth
        assert all(op in operation_chain[:5] for op in ancestry)  # Should be from earlier operations
        
        # Test emergence potential
        emergence = trace.get_emergence_potential(middle_op)
        assert len(emergence) <= 5  # Should respect future_horizon
        assert all(op in operation_chain[6:] for op in emergence)  # Should be from later operations

    def test_operation_lineage_tracking(self):
        """Test tracking of operation lineage and dependencies."""
        trace = BifractalTrace()
        
        # Create parent operation
        parent_context = Context(entropy=0.8, depth=0)
        parent_id = trace.record_operation(
            operation_type="parent_operation",
            context=parent_context,
            input_data={"type": "parent"},
            output_data={"children_count": 3}
        )
        
        # Create child operations
        child_ids = []
        for i in range(3):
            child_context = Context(entropy=0.7 - (i * 0.1), depth=1)
            child_id = trace.record_operation(
                operation_type=f"child_operation_{i}",
                context=child_context,
                input_data={"parent": parent_id, "index": i},
                output_data={"child_result": f"result_{i}"},
                parent_operation=parent_id
            )
            child_ids.append(child_id)
        
        # Verify lineage relationships
        children = trace.get_children(parent_id)
        assert len(children) == 3
        assert all(child_id in children for child_id in child_ids)
        
        # Verify parent relationships
        for child_id in child_ids:
            parent = trace.get_parent(child_id)
            assert parent == parent_id
        
        # Test lineage depth
        lineage = trace.get_full_lineage(child_ids[0])
        assert parent_id in lineage
        assert child_ids[0] in lineage

    @foundational_theory
    def test_sec_pattern_detection(self):
        """Test detection of SEC (Symbolic Entropy Collapse) patterns in traces."""
        trace = BifractalTrace()
        
        # Simulate SEC pattern: high entropy → gradual collapse → crystallization
        sec_sequence = generate_entropy_sequence(10, "collapse")
        operation_ids = []
        
        for i, entropy in enumerate(sec_sequence):
            context = Context(entropy=entropy, depth=i)
            
            op_id = trace.record_operation(
                operation_type="sec_operation",
                context=context,
                input_data={"entropy": entropy, "step": i},
                output_data={"crystallization_progress": (1.0 - entropy)}
            )
            operation_ids.append(op_id)
        
        # Analyze SEC patterns
        patterns = trace.analyze_sec_patterns()
        
        assert "collapse_events" in patterns
        assert "entropy_trend" in patterns
        assert "crystallization_points" in patterns
        
        # Should detect collapse pattern
        assert patterns["entropy_trend"] == "decreasing"
        assert len(patterns["collapse_events"]) > 0
        
        # Should identify crystallization points (low entropy operations)
        crystallization_points = patterns["crystallization_points"]
        assert len(crystallization_points) > 0
        
        # Final operations should be most crystallized
        final_ops = operation_ids[-3:]  # Last 3 operations
        assert any(op_id in crystallization_points for op_id in final_ops)

    def test_pattern_emergence_detection(self):
        """Test detection of emergent patterns in operation traces."""
        trace = BifractalTrace()
        
        # Create repeating pattern structure
        pattern_types = ["alpha", "beta", "gamma", "alpha", "beta", "alpha"]
        operation_ids = []
        
        for i, pattern_type in enumerate(pattern_types):
            context = Context(entropy=0.5 + (0.1 * (i % 3)), depth=i)
            
            op_id = trace.record_operation(
                operation_type=pattern_type,
                context=context,
                input_data={"pattern": pattern_type, "iteration": i // 3},
                output_data={"pattern_strength": (i % 3) + 1}
            )
            operation_ids.append(op_id)
        
        # Analyze emergent patterns
        emergence_analysis = trace.analyze_emergence_patterns()
        
        assert "pattern_frequency" in emergence_analysis
        assert "pattern_cycles" in emergence_analysis
        assert "emergence_strength" in emergence_analysis
        
        # Should detect alpha as most frequent pattern
        pattern_freq = emergence_analysis["pattern_frequency"]
        assert pattern_freq["alpha"] > pattern_freq["beta"]
        assert pattern_freq["alpha"] > pattern_freq["gamma"]
        
        # Should detect cyclical pattern
        cycles = emergence_analysis["pattern_cycles"]
        assert len(cycles) > 0
        assert cycles[0]["pattern"] in ["alpha", "beta", "gamma"]

    def test_trace_branching_and_merging(self):
        """Test trace handling of branching and merging operation flows."""
        trace = BifractalTrace()
        
        # Create trunk operation
        trunk_context = Context(entropy=0.7, depth=0)
        trunk_id = trace.record_operation(
            operation_type="trunk",
            context=trunk_context,
            input_data={"flow": "main"},
            output_data={"branches": 2}
        )
        
        # Create two branches
        branch_ids = []
        for branch in range(2):
            branch_context = Context(entropy=0.6 - (branch * 0.1), depth=1)
            branch_id = trace.record_operation(
                operation_type=f"branch_{branch}",
                context=branch_context,
                input_data={"trunk": trunk_id, "branch": branch},
                output_data={"branch_result": f"branch_{branch}_data"},
                parent_operation=trunk_id
            )
            branch_ids.append(branch_id)
        
        # Create merge operation
        merge_context = Context(entropy=0.4, depth=2)
        merge_id = trace.record_operation(
            operation_type="merge",
            context=merge_context,
            input_data={"branches": branch_ids},
            output_data={"merged_result": "combined_data"}
        )
        
        # Link branches to merge
        for branch_id in branch_ids:
            trace.link_operations(branch_id, merge_id, "convergence")
        
        # Verify branching structure
        trunk_children = trace.get_children(trunk_id)
        assert len(trunk_children) == 2
        assert all(branch_id in trunk_children for branch_id in branch_ids)
        
        # Verify merging structure
        merge_predecessors = trace.get_predecessors(merge_id)
        assert len(merge_predecessors) == 2
        assert all(branch_id in merge_predecessors for branch_id in branch_ids)
        
        # Analyze flow topology
        topology = trace.analyze_flow_topology()
        assert "branch_points" in topology
        assert "merge_points" in topology
        assert trunk_id in topology["branch_points"]
        assert merge_id in topology["merge_points"]

    @performance
    def test_large_trace_performance(self):
        """Test performance with large operation traces."""
        trace = BifractalTrace()
        
        start_time = time.time()
        
        # Record large number of operations
        operation_ids = []
        for i in range(1000):
            context = Context(entropy=0.5 + 0.3 * (i % 10) / 10, depth=i % 50)
            
            op_id = trace.record_operation(
                operation_type=f"operation_{i % 10}",
                context=context,
                input_data={"index": i, "batch": i // 100},
                output_data={"result": f"output_{i}"}
            )
            operation_ids.append(op_id)
        
        record_time = time.time() - start_time
        
        # Recording should be efficient
        assert record_time < TestConfig.PERFORMANCE_TIMEOUT
        assert trace.get_operation_count() == 1000
        
        # Test retrieval performance
        start_time = time.time()
        
        for i in range(0, 1000, 10):  # Sample every 10th operation
            operation = trace.get_operation(operation_ids[i])
            assert operation["input_data"]["index"] == i
        
        retrieval_time = time.time() - start_time
        assert retrieval_time < TestConfig.PERFORMANCE_TIMEOUT / 2

    def test_trace_serialization(self):
        """Test trace serialization and deserialization."""
        original_trace = BifractalTrace(ancestry_depth=3, future_horizon=3)
        
        # Add test operations
        operation_ids = []
        for i in range(5):
            context = Context(entropy=0.6 - (i * 0.1), depth=i)
            op_id = original_trace.record_operation(
                operation_type=f"test_op_{i}",
                context=context,
                input_data={"step": i},
                output_data={"result": f"output_{i}"}
            )
            operation_ids.append(op_id)
        
        # Link operations
        for i in range(1, len(operation_ids)):
            original_trace.link_operations(operation_ids[i-1], operation_ids[i], "sequential")
        
        # Serialize trace
        serialized = original_trace.serialize()
        
        assert "ancestry_depth" in serialized
        assert "future_horizon" in serialized
        assert "operations" in serialized
        assert "links" in serialized
        assert "metadata" in serialized
        
        # Deserialize trace
        restored_trace = BifractalTrace.deserialize(serialized)
        
        # Verify restoration
        assert restored_trace.ancestry_depth == original_trace.ancestry_depth
        assert restored_trace.future_horizon == original_trace.future_horizon
        assert restored_trace.get_operation_count() == original_trace.get_operation_count()
        
        # Verify operations integrity
        for op_id in operation_ids:
            original_op = original_trace.get_operation(op_id)
            restored_op = restored_trace.get_operation(op_id)
            assert original_op["operation_type"] == restored_op["operation_type"]
            assert original_op["input_data"] == restored_op["input_data"]
            assert original_op["output_data"] == restored_op["output_data"]

    def test_trace_visualization(self):
        """Test trace visualization and text output."""
        trace = BifractalTrace()
        
        # Create a small trace for visualization
        operations = [
            ("start", 0.8, 0, {"init": True}),
            ("process", 0.6, 1, {"data": "processing"}),
            ("analyze", 0.4, 2, {"analysis": "complete"}),
            ("finalize", 0.2, 3, {"final": True})
        ]
        
        operation_ids = []
        for op_type, entropy, depth, data in operations:
            context = Context(entropy=entropy, depth=depth)
            op_id = trace.record_operation(
                operation_type=op_type,
                context=context,
                input_data=data,
                output_data={"processed": True}
            )
            operation_ids.append(op_id)
        
        # Link operations sequentially
        for i in range(1, len(operation_ids)):
            trace.link_operations(operation_ids[i-1], operation_ids[i], "sequential")
        
        # Generate text visualization
        text_viz = trace.visualize_text()
        
        assert "start" in text_viz
        assert "process" in text_viz
        assert "analyze" in text_viz
        assert "finalize" in text_viz
        assert "entropy" in text_viz.lower()
        
        # Generate pattern analysis visualization
        pattern_viz = trace.visualize_patterns()
        
        assert "patterns" in pattern_viz.lower()
        assert "entropy" in pattern_viz.lower()

    @foundational_theory
    def test_recursive_causality_tracking(self):
        """Test tracking of recursive causality chains."""
        trace = BifractalTrace()
        
        # Create recursive operation chain
        def create_recursive_operation(level: int, max_level: int, parent_id: str = None) -> str:
            context = Context(entropy=0.8 - (level * 0.1), depth=level)
            
            op_id = trace.record_operation(
                operation_type="recursive_operation",
                context=context,
                input_data={"level": level, "max_level": max_level},
                output_data={"recursive_result": f"level_{level}"},
                parent_operation=parent_id
            )
            
            if level < max_level:
                # Create recursive call
                child_id = create_recursive_operation(level + 1, max_level, op_id)
                trace.link_operations(op_id, child_id, "recursive_call")
            
            return op_id
        
        # Create recursive chain
        root_id = create_recursive_operation(0, 5)
        
        # Analyze recursive patterns
        recursive_analysis = trace.analyze_recursive_patterns()
        
        assert "recursion_depth" in recursive_analysis
        assert "recursive_chains" in recursive_analysis
        assert "causality_loops" in recursive_analysis
        
        # Should detect recursion depth of 5
        assert recursive_analysis["recursion_depth"] == 5
        
        # Should identify recursive chain starting from root
        chains = recursive_analysis["recursive_chains"]
        assert len(chains) >= 1
        assert any(chain["root"] == root_id for chain in chains)

    def test_entropy_correlation_analysis(self):
        """Test analysis of entropy correlations in operation traces."""
        trace = BifractalTrace()
        
        # Create operations with correlated entropy patterns
        entropy_patterns = [
            0.9, 0.8, 0.7, 0.6, 0.5,  # Decreasing pattern
            0.6, 0.7, 0.8, 0.9,       # Increasing pattern
            0.5, 0.3, 0.7, 0.4, 0.8   # Oscillating pattern
        ]
        
        operation_ids = []
        for i, entropy in enumerate(entropy_patterns):
            context = Context(entropy=entropy, depth=i)
            
            op_id = trace.record_operation(
                operation_type="entropy_test",
                context=context,
                input_data={"entropy": entropy, "pattern_index": i},
                output_data={"correlation_test": True}
            )
            operation_ids.append(op_id)
        
        # Analyze entropy correlations
        correlation_analysis = trace.analyze_entropy_correlations()
        
        assert "entropy_trend" in correlation_analysis
        assert "correlation_strength" in correlation_analysis
        assert "pattern_detection" in correlation_analysis
        
        # Should detect mixed trends (decreasing, then increasing, then oscillating)
        patterns = correlation_analysis["pattern_detection"]
        assert len(patterns) >= 2  # Should detect multiple patterns
        
        # Should calculate correlation metrics
        strength = correlation_analysis["correlation_strength"]
        assert 0.0 <= strength <= 1.0  # Valid correlation strength range
