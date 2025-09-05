"""
Test suite for SEC (Symbolic Entropy Collapse) compliance.

Tests validate that Fracton correctly implements SEC dynamics according to
foundational theory: ∇_micro → Ψ_macro (micro states lead to macro emergence)
"""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock

import fracton
from fracton.core import RecursiveExecutor, MemoryField, BifractalTrace
from fracton.lang import Context
from .conftest import (
    TestConfig, assert_sec_compliance, generate_entropy_sequence,
    sec_compliance, foundational_theory
)


@sec_compliance
@foundational_theory
class TestSECCompliance:
    """Test suite for Symbolic Entropy Collapse compliance."""

    def test_sec_basic_collapse_dynamics(self, balanced_memory_field):
        """Test basic SEC collapse dynamics: high entropy → low entropy crystallization."""
        executor = RecursiveExecutor(entropy_regulation=True)
        
        @fracton.recursive
        @fracton.entropy_gate(0.3, 0.9)
        def sec_collapse_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Function demonstrating SEC collapse pattern."""
            current_entropy = context.entropy
            
            # Record current state
            states = memory.get("collapse_states", [])
            states.append({
                "depth": context.depth,
                "entropy": current_entropy,
                "timestamp": context.timestamp
            })
            memory.set("collapse_states", states)
            
            # Base case: crystallization achieved
            if current_entropy <= 0.3 or context.depth >= 5:
                return {
                    "collapsed": True,
                    "final_entropy": current_entropy,
                    "collapse_depth": context.depth,
                    "states": states
                }
            
            # Recursive collapse: reduce entropy
            new_entropy = current_entropy * 0.8  # 20% reduction per level
            new_context = context.deeper(1).with_entropy(new_entropy)
            
            return fracton.recurse(sec_collapse_function, memory, new_context)
        
        # Start with high entropy (exploratory state)
        initial_context = Context(entropy=0.9, depth=0)
        result = executor.execute(sec_collapse_function, balanced_memory_field, initial_context)
        
        # Verify SEC compliance
        assert result["collapsed"] is True
        assert result["final_entropy"] < 0.9  # Should have decreased
        
        # Verify entropy collapse pattern
        states = result["states"]
        entropies = [state["entropy"] for state in states]
        
        # Should show monotonic decrease (SEC collapse)
        for i in range(1, len(entropies)):
            assert entropies[i] < entropies[i-1], f"Entropy should decrease: {entropies[i-1]} → {entropies[i]}"
        
        # Verify SEC compliance with helper
        assert_sec_compliance(entropies[0], entropies[-1], "collapse")

    def test_sec_micro_to_macro_emergence(self, balanced_memory_field):
        """Test SEC micro → macro emergence pattern."""
        executor = RecursiveExecutor()
        trace = BifractalTrace()
        
        @fracton.recursive
        def micro_accumulation_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Accumulate micro-level information."""
            # Record micro operation
            micro_op_id = trace.record_operation(
                operation_type="micro_accumulation",
                context=context,
                input_data={"entropy": context.entropy, "depth": context.depth},
                output_data={"micro_data": f"micro_{context.depth}"}
            )
            
            # Accumulate micro information
            micro_data = memory.get("micro_accumulations", [])
            micro_data.append({
                "operation_id": micro_op_id,
                "entropy": context.entropy,
                "depth": context.depth,
                "data": f"micro_info_{context.depth}"
            })
            memory.set("micro_accumulations", micro_data)
            
            # Continue until sufficient micro accumulation
            if context.depth >= 4:
                # Macro emergence from accumulated micro states
                macro_structure = {
                    "emerged_from_micro": True,
                    "micro_count": len(micro_data),
                    "macro_entropy": memory.get_entropy(),
                    "emergence_pattern": "crystallized",
                    "sec_verified": True
                }
                
                # Record macro emergence
                macro_op_id = trace.record_operation(
                    operation_type="macro_emergence",
                    context=context,
                    input_data={"micro_accumulations": micro_data},
                    output_data=macro_structure
                )
                
                memory.set("macro_structure", macro_structure)
                return macro_structure
            
            # Continue micro accumulation with entropy decrease
            new_entropy = context.entropy * 0.9
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(micro_accumulation_function, memory, new_context)
        
        # Execute SEC pattern
        initial_context = Context(entropy=0.8, depth=0)
        result = executor.execute(micro_accumulation_function, balanced_memory_field, initial_context)
        
        # Verify micro → macro emergence
        assert result["emerged_from_micro"] is True
        assert result["micro_count"] == 5  # Should have accumulated 5 micro states
        assert result["emergence_pattern"] == "crystallized"
        assert result["sec_verified"] is True
        
        # Verify that macro emergence has lower entropy than initial state
        initial_entropy = 0.8
        macro_entropy = result["macro_entropy"]
        assert_sec_compliance(initial_entropy, macro_entropy, "collapse")
        
        # Verify trace shows SEC pattern
        patterns = trace.analyze_sec_patterns()
        assert patterns["entropy_trend"] == "decreasing"
        assert len(patterns["collapse_events"]) > 0

    def test_sec_symbolic_crystallization(self, balanced_memory_field):
        """Test SEC symbolic crystallization process."""
        executor = RecursiveExecutor()
        
        @fracton.recursive
        @fracton.entropy_gate(0.2, 0.9)
        def symbolic_crystallization(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Crystallize symbolic structures through entropy reduction."""
            
            # Get current symbol space
            symbols = memory.get("symbol_space", set())
            
            # Add new symbolic elements based on entropy level
            if context.entropy > 0.7:
                # High entropy: exploratory symbol generation
                new_symbols = {f"explore_{context.depth}_{i}" for i in range(5)}
                symbols.update(new_symbols)
                crystallization_state = "exploration"
                
            elif context.entropy > 0.4:
                # Mid entropy: symbol organization
                symbols = {s for s in symbols if "explore" in s}  # Filter symbols
                crystallization_state = "organization"
                
            else:
                # Low entropy: symbol crystallization
                core_symbols = {s for s in symbols if "_0" in s or "_1" in s}  # Keep core
                symbols = core_symbols
                crystallization_state = "crystallized"
            
            memory.set("symbol_space", symbols)
            
            # Check for crystallization completion
            if crystallization_state == "crystallized" or context.depth >= 6:
                return {
                    "crystallized": True,
                    "final_symbols": list(symbols),
                    "symbol_count": len(symbols),
                    "crystallization_state": crystallization_state,
                    "final_entropy": context.entropy,
                    "sec_pattern": "symbolic_collapse"
                }
            
            # Continue crystallization with entropy reduction
            new_entropy = context.entropy * 0.85
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(symbolic_crystallization, memory, new_context)
        
        # Execute symbolic crystallization
        initial_context = Context(entropy=0.85, depth=0)
        result = executor.execute(symbolic_crystallization, balanced_memory_field, initial_context)
        
        # Verify symbolic crystallization
        assert result["crystallized"] is True
        assert result["crystallization_state"] in ["crystallized", "organization"]
        assert result["sec_pattern"] == "symbolic_collapse"
        
        # Should have fewer symbols after crystallization (compression)
        assert result["symbol_count"] <= 10  # Should be compressed
        
        # Verify entropy collapse
        initial_entropy = 0.85
        final_entropy = result["final_entropy"]
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")

    def test_sec_entropy_phase_transitions(self, balanced_memory_field):
        """Test SEC entropy phase transitions."""
        executor = RecursiveExecutor()
        phase_history = []
        
        @fracton.recursive
        def phase_transition_function(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Track entropy phase transitions during SEC."""
            
            current_entropy = context.entropy
            
            # Determine current phase
            if current_entropy > 0.7:
                phase = "high_entropy_exploration"
                behavior = "chaotic_search"
            elif current_entropy > 0.4:
                phase = "medium_entropy_transition"
                behavior = "pattern_formation"
            else:
                phase = "low_entropy_crystallization"
                behavior = "structure_stabilization"
            
            phase_data = {
                "depth": context.depth,
                "entropy": current_entropy,
                "phase": phase,
                "behavior": behavior,
                "timestamp": context.timestamp
            }
            phase_history.append(phase_data)
            
            # Store phase transition
            transitions = memory.get("phase_transitions", [])
            transitions.append(phase_data)
            memory.set("phase_transitions", transitions)
            
            # Complete when reaching crystallization or max depth
            if phase == "low_entropy_crystallization" or context.depth >= 8:
                return {
                    "phase_complete": True,
                    "final_phase": phase,
                    "transition_count": len(transitions),
                    "phase_history": phase_history,
                    "sec_verified": True
                }
            
            # Continue with entropy reduction
            new_entropy = current_entropy * 0.88
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(phase_transition_function, memory, new_context)
        
        # Execute phase transition sequence
        initial_context = Context(entropy=0.9, depth=0)
        result = executor.execute(phase_transition_function, balanced_memory_field, initial_context)
        
        # Verify phase transitions
        assert result["phase_complete"] is True
        assert result["sec_verified"] is True
        
        # Verify phase progression
        phases = [p["phase"] for p in result["phase_history"]]
        
        # Should progress through phases in SEC order
        assert "high_entropy_exploration" in phases
        assert phases.index("high_entropy_exploration") < phases.index("medium_entropy_transition") if "medium_entropy_transition" in phases else True
        
        # Verify entropy decreases across phases
        entropies = [p["entropy"] for p in result["phase_history"]]
        for i in range(1, len(entropies)):
            assert entropies[i] <= entropies[i-1] * 1.01  # Allow slight numerical tolerance

    def test_sec_collapse_event_detection(self):
        """Test detection of discrete SEC collapse events."""
        field = MemoryField(capacity=200, entropy=0.8)
        executor = RecursiveExecutor()
        trace = BifractalTrace()
        
        collapse_events = []
        
        @fracton.recursive
        def collapse_event_detector(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Detect discrete collapse events during SEC."""
            
            previous_entropy = memory.get("previous_entropy", context.entropy)
            current_entropy = context.entropy
            
            # Detect collapse event (significant entropy drop)
            entropy_drop = previous_entropy - current_entropy
            if entropy_drop > 0.15:  # Significant drop threshold
                collapse_event = {
                    "event_type": "collapse",
                    "depth": context.depth,
                    "entropy_before": previous_entropy,
                    "entropy_after": current_entropy,
                    "entropy_drop": entropy_drop,
                    "timestamp": context.timestamp
                }
                
                collapse_events.append(collapse_event)
                
                # Record collapse event in trace
                trace.record_operation(
                    operation_type="collapse_event",
                    context=context,
                    input_data={"entropy_before": previous_entropy},
                    output_data=collapse_event
                )
            
            memory.set("previous_entropy", current_entropy)
            
            if context.depth >= 6 or current_entropy <= 0.2:
                return {
                    "collapse_detection_complete": True,
                    "collapse_events": collapse_events,
                    "total_events": len(collapse_events),
                    "final_entropy": current_entropy
                }
            
            # Create entropy drop for next iteration
            new_entropy = current_entropy * 0.7  # Larger drop to trigger events
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(collapse_event_detector, memory, new_context)
        
        # Execute collapse detection
        initial_context = Context(entropy=0.9, depth=0)
        result = executor.execute(collapse_event_detector, field, initial_context)
        
        # Verify collapse event detection
        assert result["collapse_detection_complete"] is True
        assert result["total_events"] > 0  # Should detect discrete collapse events
        
        # Verify collapse events have proper structure
        events = result["collapse_events"]
        for event in events:
            assert event["event_type"] == "collapse"
            assert event["entropy_drop"] > 0.15  # Should meet threshold
            assert event["entropy_after"] < event["entropy_before"]  # Entropy should decrease
        
        # Verify overall SEC compliance
        initial_entropy = 0.9
        final_entropy = result["final_entropy"]
        assert_sec_compliance(initial_entropy, final_entropy, "collapse")

    def test_sec_information_crystallization_rate(self, balanced_memory_field):
        """Test SEC information crystallization rate analysis."""
        executor = RecursiveExecutor()
        
        crystallization_data = []
        
        @fracton.recursive
        def crystallization_rate_analyzer(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Analyze rate of information crystallization."""
            
            # Measure information content before processing
            info_before = memory.size()
            entropy_before = context.entropy
            
            # Add structured information (crystallization)
            structured_info = {
                f"crystal_{context.depth}_{i}": {
                    "structure": "ordered",
                    "entropy_level": context.entropy,
                    "crystallization_step": i
                }
                for i in range(int(5 * (1.0 - context.entropy)))  # More structures at low entropy
            }
            
            for key, value in structured_info.items():
                memory.set(key, value)
            
            # Measure information content after processing
            info_after = memory.size()
            entropy_after = memory.get_entropy()
            
            # Calculate crystallization rate
            info_gain = info_after - info_before
            entropy_change = entropy_before - entropy_after
            crystallization_rate = info_gain / max(entropy_change, 0.001)  # Avoid division by zero
            
            crystallization_point = {
                "depth": context.depth,
                "info_before": info_before,
                "info_after": info_after,
                "info_gain": info_gain,
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_change": entropy_change,
                "crystallization_rate": crystallization_rate
            }
            
            crystallization_data.append(crystallization_point)
            
            if context.depth >= 5 or entropy_after <= 0.2:
                return {
                    "crystallization_complete": True,
                    "crystallization_data": crystallization_data,
                    "final_rate": crystallization_rate,
                    "total_info_gain": info_after - crystallization_data[0]["info_before"],
                    "total_entropy_drop": crystallization_data[0]["entropy_before"] - entropy_after
                }
            
            # Continue with entropy reduction
            new_entropy = context.entropy * 0.85
            new_context = context.deeper(1).with_entropy(new_entropy)
            return fracton.recurse(crystallization_rate_analyzer, memory, new_context)
        
        # Execute crystallization rate analysis
        initial_context = Context(entropy=0.8, depth=0)
        result = executor.execute(crystallization_rate_analyzer, balanced_memory_field, initial_context)
        
        # Verify crystallization analysis
        assert result["crystallization_complete"] is True
        assert result["total_info_gain"] > 0  # Should have added structured information
        assert result["total_entropy_drop"] > 0  # Should have reduced entropy
        
        # Verify crystallization rate progression
        rates = [point["crystallization_rate"] for point in result["crystallization_data"]]
        
        # Later stages should have higher crystallization rates (more efficient)
        assert rates[-1] >= rates[0]  # Rate should increase or stay stable

    def test_sec_reversibility_and_exploration(self, balanced_memory_field):
        """Test SEC reversibility and exploration phase behavior."""
        executor = RecursiveExecutor()
        
        @fracton.recursive
        @fracton.entropy_gate(0.1, 0.95)
        def sec_reversibility_test(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Test SEC reversibility (exploration after collapse)."""
            
            phase_type = memory.get("phase_type", "collapse")
            entropy_history = memory.get("entropy_history", [])
            entropy_history.append(context.entropy)
            memory.set("entropy_history", entropy_history)
            
            if phase_type == "collapse":
                # Collapse phase: reduce entropy
                if context.entropy <= 0.3:
                    # Switch to exploration phase
                    memory.set("phase_type", "exploration")
                    memory.set("collapse_complete", True)
                    new_entropy = context.entropy * 1.5  # Increase entropy for exploration
                else:
                    new_entropy = context.entropy * 0.8  # Continue collapse
                    
            else:  # exploration phase
                # Exploration phase: increase entropy
                if context.entropy >= 0.7 or context.depth >= 10:
                    return {
                        "reversibility_test_complete": True,
                        "entropy_history": entropy_history,
                        "phase_transitions": ["collapse", "exploration"],
                        "collapse_achieved": memory.get("collapse_complete", False),
                        "final_entropy": context.entropy,
                        "sec_reversibility_verified": True
                    }
                new_entropy = context.entropy * 1.2  # Continue exploration
            
            # Continue with next phase
            new_context = context.deeper(1).with_entropy(min(new_entropy, 0.95))
            return fracton.recurse(sec_reversibility_test, memory, new_context)
        
        # Execute reversibility test
        initial_context = Context(entropy=0.9, depth=0)
        result = executor.execute(sec_reversibility_test, balanced_memory_field, initial_context)
        
        # Verify reversibility
        assert result["reversibility_test_complete"] is True
        assert result["collapse_achieved"] is True
        assert result["sec_reversibility_verified"] is True
        
        # Verify entropy went down then up (collapse then exploration)
        entropy_history = result["entropy_history"]
        
        # Find minimum entropy point (collapse bottom)
        min_entropy_idx = entropy_history.index(min(entropy_history))
        
        # Verify collapse phase (decreasing entropy)
        collapse_phase = entropy_history[:min_entropy_idx + 1]
        if len(collapse_phase) > 1:
            for i in range(1, len(collapse_phase)):
                assert collapse_phase[i] <= collapse_phase[i-1] * 1.01  # Allow tolerance
        
        # Verify exploration phase (increasing entropy)
        exploration_phase = entropy_history[min_entropy_idx:]
        if len(exploration_phase) > 1:
            assert exploration_phase[-1] > exploration_phase[0]  # Should end higher
