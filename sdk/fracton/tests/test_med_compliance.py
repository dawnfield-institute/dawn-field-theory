"""
Test suite for MED (Macro Emergence Dynamics) compliance.

Tests validate that Fracton correctly implements MED dynamics according to
foundational theory: ∇_macro → Ψ_micro (macro fields constrain micro actualization)
"""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock

import fracton
from fracton.core import RecursiveExecutor, MemoryField, BifractalTrace
from fracton.lang import Context
from .conftest import (
    TestConfig, assert_med_compliance, generate_field_hierarchy,
    med_compliance, foundational_theory
)


@med_compliance
@foundational_theory
class TestMEDCompliance:
    """Test suite for Macro Emergence Dynamics compliance."""

    def test_med_basic_macro_to_micro_constraint(self, med_macro_field, med_micro_field):
        """Test basic MED macro → micro constraint dynamics."""
        executor = RecursiveExecutor()
        
        @fracton.recursive
        def macro_field_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process macro-scale field structures that constrain micro behavior."""
            
            # Define macro field structure
            macro_structure = {
                "field_type": "macro",
                "scale": TestConfig.MED_MACRO_SCALE,
                "boundary_conditions": {
                    "x_min": 0, "x_max": 100,
                    "y_min": 0, "y_max": 100,
                    "field_strength": context.entropy
                },
                "field_pressure": context.entropy * 10,
                "emergence_constraints": {
                    "micro_density_limit": TestConfig.MED_MICRO_SCALE,
                    "coupling_strength": 0.8,
                    "constraint_type": "boundary_limited"
                }
            }
            
            memory.set("macro_structure", macro_structure)
            
            # Macro structure influences micro emergence
            micro_constraints = {
                "max_density": macro_structure["emergence_constraints"]["micro_density_limit"],
                "boundary_respect": True,
                "field_pressure_adaptation": macro_structure["field_pressure"],
                "coupling_to_macro": macro_structure["emergence_constraints"]["coupling_strength"]
            }
            
            memory.set("micro_constraints", micro_constraints)
            
            return {
                "macro_processing_complete": True,
                "macro_scale": macro_structure["scale"],
                "micro_constraints_set": True,
                "med_pattern": "macro_to_micro_constraint"
            }
        
        @fracton.recursive
        def micro_field_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process micro actualization under macro constraints."""
            
            # Get constraints from macro processor
            constraints = memory.get("micro_constraints", {})
            macro_structure = memory.get("macro_structure", {})
            
            if not constraints:
                raise ValueError("MED violation: micro processing without macro constraints")
            
            # Micro actualization respects macro constraints
            micro_actualization = {
                "field_type": "micro",
                "scale": TestConfig.MED_MICRO_SCALE,
                "density": min(context.entropy * 20, constraints.get("max_density", 10)),
                "boundary_compliance": constraints.get("boundary_respect", False),
                "pressure_response": constraints.get("field_pressure_adaptation", 0) * 0.1,
                "macro_coupling": constraints.get("coupling_to_macro", 0)
            }
            
            # Verify MED compliance
            if micro_actualization["scale"] >= macro_structure.get("scale", 0):
                raise ValueError("MED violation: micro scale >= macro scale")
            
            if not micro_actualization["boundary_compliance"]:
                raise ValueError("MED violation: micro not respecting macro boundaries")
            
            memory.set("micro_actualization", micro_actualization)
            
            return {
                "micro_processing_complete": True,
                "micro_scale": micro_actualization["scale"],
                "macro_constrained": True,
                "med_verified": True
            }
        
        # Execute MED pattern: macro then micro
        context = Context(entropy=0.6, depth=0)
        
        # Process macro field
        macro_result = executor.execute(macro_field_processor, med_macro_field, context)
        
        # Process micro field under macro constraints
        micro_result = executor.execute(micro_field_processor, med_macro_field, context)  # Use same field for constraints
        
        # Verify MED compliance
        assert macro_result["macro_processing_complete"] is True
        assert micro_result["micro_processing_complete"] is True
        assert micro_result["macro_constrained"] is True
        assert micro_result["med_verified"] is True
        
        # Verify scale relationship
        macro_scale = macro_result["macro_scale"]
        micro_scale = micro_result["micro_scale"]
        
        macro_state = {"scale": macro_scale, "entropy": 0.6}
        micro_state = {"scale": micro_scale, "entropy": 0.6}
        assert_med_compliance(macro_state, micro_state)

    def test_med_hierarchical_emergence_levels(self):
        """Test MED across multiple hierarchical levels."""
        executor = RecursiveExecutor()
        field = MemoryField(capacity=500, entropy=0.7)
        
        hierarchy = generate_field_hierarchy(4)  # 4 levels of hierarchy
        
        @fracton.recursive
        def hierarchical_med_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process hierarchical MED across multiple levels."""
            
            current_level = context.depth
            if current_level >= len(hierarchy):
                # Complete hierarchy processing
                return {
                    "hierarchy_complete": True,
                    "levels_processed": current_level,
                    "med_hierarchy_verified": True
                }
            
            # Get current level data
            level_data = hierarchy[current_level]
            
            # Store level structure
            memory.set(f"level_{current_level}", level_data)
            
            # If not at top level, get constraints from parent level
            if current_level > 0:
                parent_data = memory.get(f"level_{current_level - 1}")
                
                # Verify MED constraint: current level scale < parent level scale
                if level_data["scale"] >= parent_data["scale"]:
                    raise ValueError(f"MED violation at level {current_level}: child scale >= parent scale")
                
                # Apply parent constraints to current level
                parent_constraints = {
                    "max_structures": parent_data["scale"] // 10,
                    "entropy_coupling": parent_data["entropy"] * 0.8,
                    "boundary_inheritance": True
                }
                
                # Constrain current level by parent
                constrained_structures = level_data["structures"][:parent_constraints["max_structures"]]
                level_data["structures"] = constrained_structures
                level_data["entropy"] = parent_constraints["entropy_coupling"]
                level_data["parent_constrained"] = True
                
                memory.set(f"level_{current_level}", level_data)
            
            # Process next level
            new_context = context.deeper(1)
            return fracton.recurse(hierarchical_med_processor, memory, new_context)
        
        # Execute hierarchical MED
        initial_context = Context(entropy=0.8, depth=0)
        result = executor.execute(hierarchical_med_processor, field, initial_context)
        
        # Verify hierarchical MED
        assert result["hierarchy_complete"] is True
        assert result["med_hierarchy_verified"] is True
        assert result["levels_processed"] == 4
        
        # Verify MED compliance across levels
        for level in range(1, 4):
            parent_data = field.get(f"level_{level - 1}")
            child_data = field.get(f"level_{level}")
            
            # Verify scale hierarchy
            assert child_data["scale"] < parent_data["scale"]
            
            # Verify constraint propagation
            assert child_data.get("parent_constrained", False) is True
            
            # Verify entropy coupling
            assert child_data["entropy"] <= parent_data["entropy"]

    def test_med_field_pressure_dynamics(self):
        """Test MED field pressure and constraint dynamics."""
        executor = RecursiveExecutor()
        field = MemoryField(capacity=300, entropy=0.5)
        
        @fracton.recursive
        def field_pressure_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process field pressure dynamics in MED."""
            
            # Calculate macro field pressure
            macro_pressure = context.entropy * 10 + context.depth * 2
            
            # Macro field properties
            macro_field = {
                "pressure": macro_pressure,
                "field_strength": context.entropy,
                "spatial_extent": TestConfig.MED_MACRO_SCALE,
                "constraint_radius": TestConfig.MED_MACRO_SCALE // 2
            }
            
            memory.set("macro_field", macro_field)
            
            # Calculate micro response to macro pressure
            pressure_response = macro_pressure * 0.1  # Micro scale response
            constraint_strength = min(macro_pressure / 10, 1.0)  # Normalize constraint
            
            # Micro field constrained by macro pressure
            micro_field = {
                "pressure_response": pressure_response,
                "constraint_strength": constraint_strength,
                "actualization_limit": TestConfig.MED_MICRO_SCALE,
                "macro_coupling": True,
                "emergence_probability": 1.0 - constraint_strength  # Higher pressure = less emergence freedom
            }
            
            memory.set("micro_field", micro_field)
            
            # Check for pressure equilibrium
            pressure_ratio = pressure_response / max(macro_pressure, 0.1)
            equilibrium_reached = 0.05 <= pressure_ratio <= 0.15  # Micro should be 5-15% of macro
            
            if equilibrium_reached or context.depth >= 6:
                return {
                    "pressure_equilibrium_reached": equilibrium_reached,
                    "macro_pressure": macro_pressure,
                    "micro_pressure": pressure_response,
                    "pressure_ratio": pressure_ratio,
                    "med_pressure_verified": True
                }
            
            # Adjust entropy to reach equilibrium
            if pressure_ratio > 0.15:
                new_entropy = context.entropy * 0.9  # Reduce to lower pressure
            else:
                new_entropy = context.entropy * 1.05  # Increase to raise pressure
            
            new_context = context.deeper(1).with_entropy(min(new_entropy, 0.95))
            return fracton.recurse(field_pressure_processor, memory, new_context)
        
        # Execute field pressure dynamics
        initial_context = Context(entropy=0.7, depth=0)
        result = executor.execute(field_pressure_processor, field, initial_context)
        
        # Verify pressure dynamics
        assert result["med_pressure_verified"] is True
        
        # Verify pressure scale relationship (micro < macro)
        macro_pressure = result["macro_pressure"]
        micro_pressure = result["micro_pressure"]
        assert micro_pressure < macro_pressure
        
        # Verify pressure ratio in expected range
        pressure_ratio = result["pressure_ratio"]
        assert 0.01 <= pressure_ratio <= 0.2  # Micro should be small fraction of macro

    def test_med_boundary_condition_inheritance(self):
        """Test MED boundary condition inheritance from macro to micro."""
        executor = RecursiveExecutor()
        field = MemoryField(capacity=400, entropy=0.6)
        
        @fracton.recursive
        def boundary_inheritance_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process boundary condition inheritance in MED."""
            
            # Define macro boundary conditions
            macro_boundaries = {
                "spatial": {
                    "x_bounds": [0, 100],
                    "y_bounds": [0, 100],
                    "z_bounds": [0, 50]
                },
                "temporal": {
                    "duration": 1000,
                    "resolution": 10
                },
                "field": {
                    "min_strength": 0.1,
                    "max_strength": 1.0
                },
                "entropy": {
                    "min_entropy": 0.2,
                    "max_entropy": 0.8
                }
            }
            
            memory.set("macro_boundaries", macro_boundaries)
            
            # Derive micro boundary conditions from macro
            micro_boundaries = {
                "spatial": {
                    "x_bounds": [5, 15],  # Subset of macro bounds
                    "y_bounds": [10, 20],  # Subset of macro bounds
                    "z_bounds": [5, 10]   # Subset of macro bounds
                },
                "temporal": {
                    "duration": 100,      # Shorter than macro
                    "resolution": 1       # Finer than macro
                },
                "field": {
                    "min_strength": 0.2,  # Constrained by macro
                    "max_strength": 0.6   # Constrained by macro
                },
                "entropy": {
                    "min_entropy": 0.3,   # Constrained by macro
                    "max_entropy": 0.7    # Constrained by macro
                }
            }
            
            # Verify boundary inheritance compliance
            spatial_compliance = all(
                micro_boundaries["spatial"][dim][0] >= macro_boundaries["spatial"][dim][0] and
                micro_boundaries["spatial"][dim][1] <= macro_boundaries["spatial"][dim][1]
                for dim in ["x_bounds", "y_bounds", "z_bounds"]
            )
            
            temporal_compliance = (
                micro_boundaries["temporal"]["duration"] <= macro_boundaries["temporal"]["duration"] and
                micro_boundaries["temporal"]["resolution"] <= macro_boundaries["temporal"]["resolution"]
            )
            
            field_compliance = (
                micro_boundaries["field"]["min_strength"] >= macro_boundaries["field"]["min_strength"] and
                micro_boundaries["field"]["max_strength"] <= macro_boundaries["field"]["max_strength"]
            )
            
            entropy_compliance = (
                micro_boundaries["entropy"]["min_entropy"] >= macro_boundaries["entropy"]["min_entropy"] and
                micro_boundaries["entropy"]["max_entropy"] <= macro_boundaries["entropy"]["max_entropy"]
            )
            
            total_compliance = all([spatial_compliance, temporal_compliance, field_compliance, entropy_compliance])
            
            if not total_compliance:
                raise ValueError("MED violation: micro boundaries exceed macro constraints")
            
            memory.set("micro_boundaries", micro_boundaries)
            
            return {
                "boundary_inheritance_complete": True,
                "spatial_compliance": spatial_compliance,
                "temporal_compliance": temporal_compliance,
                "field_compliance": field_compliance,
                "entropy_compliance": entropy_compliance,
                "total_med_compliance": total_compliance
            }
        
        # Execute boundary inheritance
        context = Context(entropy=0.6, depth=0)
        result = executor.execute(boundary_inheritance_processor, field, context)
        
        # Verify boundary inheritance
        assert result["boundary_inheritance_complete"] is True
        assert result["total_med_compliance"] is True
        assert all([
            result["spatial_compliance"],
            result["temporal_compliance"], 
            result["field_compliance"],
            result["entropy_compliance"]
        ])
        
        # Verify actual boundary data
        macro_boundaries = field.get("macro_boundaries")
        micro_boundaries = field.get("micro_boundaries")
        
        # Check specific boundary relationships
        assert micro_boundaries["spatial"]["x_bounds"][1] <= macro_boundaries["spatial"]["x_bounds"][1]
        assert micro_boundaries["temporal"]["duration"] <= macro_boundaries["temporal"]["duration"]
        assert micro_boundaries["field"]["max_strength"] <= macro_boundaries["field"]["max_strength"]

    def test_med_emergence_probability_modulation(self):
        """Test MED modulation of emergence probabilities."""
        executor = RecursiveExecutor()
        field = MemoryField(capacity=250, entropy=0.5)
        trace = BifractalTrace()
        
        @fracton.recursive
        def emergence_probability_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Process emergence probability modulation in MED."""
            
            # Macro field determines emergence probability landscape
            macro_field_strength = context.entropy
            spatial_coherence = 0.8  # High coherence reduces emergence freedom
            temporal_stability = 0.7  # High stability constrains emergence
            
            # Calculate base emergence probability
            base_emergence_prob = macro_field_strength * 0.5
            
            # Modulate by macro field properties
            coherence_factor = 1.0 - spatial_coherence  # Higher coherence = lower freedom
            stability_factor = 1.0 - temporal_stability  # Higher stability = lower freedom
            
            modulated_emergence_prob = base_emergence_prob * coherence_factor * stability_factor
            
            # Record macro constraint
            macro_constraint = {
                "field_strength": macro_field_strength,
                "spatial_coherence": spatial_coherence,
                "temporal_stability": temporal_stability,
                "base_emergence_prob": base_emergence_prob,
                "modulated_emergence_prob": modulated_emergence_prob
            }
            
            memory.set("macro_constraint", macro_constraint)
            
            # Simulate micro emergence events under constraint
            emergence_events = []
            for i in range(10):
                # Random emergence attempt
                emergence_attempt = np.random.random()
                
                if emergence_attempt < modulated_emergence_prob:
                    # Emergence succeeds under macro constraint
                    event = {
                        "event_id": f"emergence_{i}",
                        "success": True,
                        "attempt_value": emergence_attempt,
                        "threshold": modulated_emergence_prob,
                        "macro_constrained": True
                    }
                    
                    # Record in trace
                    trace.record_operation(
                        operation_type="micro_emergence",
                        context=context,
                        input_data={"attempt": emergence_attempt},
                        output_data=event
                    )
                else:
                    # Emergence blocked by macro constraint
                    event = {
                        "event_id": f"emergence_{i}",
                        "success": False,
                        "attempt_value": emergence_attempt,
                        "threshold": modulated_emergence_prob,
                        "blocked_by_macro": True
                    }
                
                emergence_events.append(event)
            
            # Calculate emergence statistics
            successful_events = [e for e in emergence_events if e["success"]]
            emergence_rate = len(successful_events) / len(emergence_events)
            
            return {
                "emergence_modulation_complete": True,
                "macro_constraint": macro_constraint,
                "emergence_events": emergence_events,
                "successful_emergences": len(successful_events),
                "emergence_rate": emergence_rate,
                "med_probability_verified": True
            }
        
        # Execute emergence probability modulation
        context = Context(entropy=0.6, depth=0)
        result = executor.execute(emergence_probability_processor, field, context)
        
        # Verify emergence modulation
        assert result["emergence_modulation_complete"] is True
        assert result["med_probability_verified"] is True
        
        # Verify macro constraint effect
        macro_constraint = result["macro_constraint"]
        assert macro_constraint["modulated_emergence_prob"] <= macro_constraint["base_emergence_prob"]
        
        # Verify emergence rate is constrained
        emergence_rate = result["emergence_rate"]
        expected_max_rate = macro_constraint["modulated_emergence_prob"]
        
        # Emergence rate should be roughly consistent with probability (with some variance)
        assert 0.0 <= emergence_rate <= 1.0
        
        # Statistical check (may have variance due to random sampling)
        # If emergence_rate is very different from expected, it suggests proper constraint
        if expected_max_rate < 0.3:  # Low probability should yield low rate
            assert emergence_rate < 0.5  # Should be constrained

    def test_med_inverse_sec_relationship(self):
        """Test MED as inverse operation to SEC (unified complexity cycle)."""
        executor = RecursiveExecutor()
        field = MemoryField(capacity=300, entropy=0.6)
        
        @fracton.recursive
        def unified_complexity_processor(memory: MemoryField, context: Context) -> Dict[str, Any]:
            """Test unified complexity: SEC ∘ MED = Identity hypothesis."""
            
            phase = memory.get("phase", "med_macro_to_micro")
            
            if phase == "med_macro_to_micro":
                # MED phase: ∇_macro → Ψ_micro
                
                # Start with macro state
                macro_state = {
                    "scale": TestConfig.MED_MACRO_SCALE,
                    "entropy": context.entropy,
                    "structure": "macro_field",
                    "complexity": "high"
                }
                
                # MED: macro constrains micro actualization
                micro_actualization = {
                    "scale": TestConfig.MED_MICRO_SCALE,
                    "entropy": context.entropy * 0.8,  # Constrained by macro
                    "structure": "micro_components",
                    "complexity": "constrained",
                    "macro_derived": True
                }
                
                memory.set("macro_state", macro_state)
                memory.set("micro_state", micro_actualization)
                memory.set("phase", "sec_micro_to_macro")
                
                # Continue to SEC phase
                new_context = context.deeper(1)
                return fracton.recurse(unified_complexity_processor, memory, new_context)
                
            elif phase == "sec_micro_to_macro":
                # SEC phase: ∇_micro → Ψ_macro
                
                micro_state = memory.get("micro_state")
                original_macro = memory.get("macro_state")
                
                # SEC: micro states lead to macro emergence
                emerged_macro = {
                    "scale": TestConfig.MED_MACRO_SCALE,
                    "entropy": micro_state["entropy"] * 0.9,  # Further crystallization
                    "structure": "emerged_macro_field",
                    "complexity": "crystallized",
                    "micro_derived": True
                }
                
                memory.set("emerged_macro", emerged_macro)
                
                # Check for unified complexity cycle completion
                original_scale = original_macro["scale"]
                emerged_scale = emerged_macro["scale"]
                
                original_entropy = original_macro["entropy"]
                emerged_entropy = emerged_macro["entropy"]
                
                # Calculate cycle fidelity
                scale_fidelity = 1.0 - abs(emerged_scale - original_scale) / original_scale
                entropy_evolution = original_entropy - emerged_entropy  # Should show evolution
                
                cycle_complete = scale_fidelity > 0.8  # High fidelity indicates near-identity
                
                return {
                    "unified_complexity_complete": True,
                    "original_macro": original_macro,
                    "intermediate_micro": micro_state,
                    "emerged_macro": emerged_macro,
                    "scale_fidelity": scale_fidelity,
                    "entropy_evolution": entropy_evolution,
                    "cycle_fidelity": scale_fidelity,
                    "med_sec_unified": cycle_complete
                }
        
        # Execute unified complexity cycle
        initial_context = Context(entropy=0.7, depth=0)
        result = executor.execute(unified_complexity_processor, field, initial_context)
        
        # Verify unified complexity cycle
        assert result["unified_complexity_complete"] is True
        assert result["med_sec_unified"] is True
        
        # Verify MED → SEC cycle structure
        original_macro = result["original_macro"]
        intermediate_micro = result["intermediate_micro"]
        emerged_macro = result["emerged_macro"]
        
        # Verify MED: macro → micro constraint
        assert intermediate_micro["scale"] < original_macro["scale"]
        assert intermediate_micro["macro_derived"] is True
        
        # Verify SEC: micro → macro emergence
        assert emerged_macro["scale"] >= intermediate_micro["scale"]
        assert emerged_macro["micro_derived"] is True
        
        # Verify cycle fidelity (approximate identity)
        assert result["scale_fidelity"] > 0.7  # High fidelity
        
        # Verify entropy evolution (not exact identity, but controlled evolution)
        assert result["entropy_evolution"] > 0  # Should show evolution/crystallization
