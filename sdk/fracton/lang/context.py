"""
Fracton Context Management - Execution context creation and manipulation

This module provides utilities for creating and managing execution contexts
in the Fracton runtime, including context factories and manipulation functions.
"""

import time
import uuid
from typing import Any, Dict, Optional
from ..core.recursive_engine import ExecutionContext


def Context(entropy: float = 0.5, depth: int = 0, **kwargs) -> ExecutionContext:
    """
    Create a new execution context with specified properties.
    
    This is an alias for ExecutionContext that provides a more convenient
    interface for context creation in user code.
    
    Args:
        entropy: Initial entropy level (0.0 - 1.0)
        depth: Initial recursion depth
        **kwargs: Additional metadata to include in context
        
    Returns:
        New ExecutionContext instance
        
    Example:
        context = fracton.Context(
            entropy=0.7,
            depth=0,
            operation="data_analysis",
            timestamp=time.time()
        )
    """
    return ExecutionContext(
        entropy=entropy,
        depth=depth,
        metadata=kwargs
    )


def create_context(entropy: float = 0.5, depth: int = 0,
                  field_state: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  trace_id: Optional[str] = None) -> ExecutionContext:
    """
    Create a new execution context with comprehensive configuration.
    
    This function provides full control over all context properties for
    advanced use cases requiring specific context configuration.
    
    Args:
        entropy: Initial entropy level (0.0 - 1.0)
        depth: Initial recursion depth
        field_state: Initial field state dictionary
        metadata: Initial metadata dictionary
        trace_id: Optional trace identifier
        
    Returns:
        New ExecutionContext instance
        
    Example:
        context = fracton.create_context(
            entropy=0.8,
            depth=0,
            field_state={"memory_size": 1000, "last_operation": "initialize"},
            metadata={"experiment": "entropy_dynamics", "run_id": "exp_001"},
            trace_id="trace_12345"
        )
    """
    return ExecutionContext(
        entropy=entropy,
        depth=depth,
        trace_id=trace_id or str(uuid.uuid4()),
        field_state=field_state or {},
        metadata=metadata or {}
    )


def context_from_template(template_name: str, **overrides) -> ExecutionContext:
    """
    Create context from predefined template with optional overrides.
    
    Templates provide common context configurations for typical use cases,
    reducing boilerplate code and ensuring consistency.
    
    Args:
        template_name: Name of the context template to use
        **overrides: Properties to override in the template
        
    Returns:
        New ExecutionContext based on template
        
    Available templates:
        - "low_entropy": Low entropy stable processing (entropy=0.2)
        - "medium_entropy": Balanced processing (entropy=0.5)
        - "high_entropy": High entropy exploratory processing (entropy=0.8)
        - "debug": Debug context with comprehensive metadata
        - "performance": Performance-optimized context
        - "experimental": High-entropy experimental context
        
    Example:
        # Create a debug context
        debug_context = fracton.context_from_template("debug", 
                                                     experiment_id="test_001")
        
        # Create high-entropy exploration context
        explore_context = fracton.context_from_template("high_entropy",
                                                       depth=5)
    """
    templates = {
        "low_entropy": {
            "entropy": 0.2,
            "depth": 0,
            "metadata": {
                "mode": "stable",
                "processing_type": "crystallization",
                "optimization_level": "high"
            }
        },
        
        "medium_entropy": {
            "entropy": 0.5,
            "depth": 0,
            "metadata": {
                "mode": "balanced",
                "processing_type": "standard",
                "optimization_level": "medium"
            }
        },
        
        "high_entropy": {
            "entropy": 0.8,
            "depth": 0,
            "metadata": {
                "mode": "exploratory",
                "processing_type": "divergent",
                "optimization_level": "adaptive"
            }
        },
        
        "debug": {
            "entropy": 0.5,
            "depth": 0,
            "metadata": {
                "mode": "debug",
                "enable_verbose_tracing": True,
                "capture_intermediate_results": True,
                "timestamp": time.time()
            }
        },
        
        "performance": {
            "entropy": 0.3,
            "depth": 0,
            "metadata": {
                "mode": "performance",
                "enable_memoization": True,
                "tail_recursion_optimization": True,
                "minimize_tracing": True
            }
        },
        
        "experimental": {
            "entropy": 0.9,
            "depth": 0,
            "metadata": {
                "mode": "experimental",
                "allow_high_entropy": True,
                "enable_pattern_discovery": True,
                "capture_emergence": True
            }
        }
    }
    
    if template_name not in templates:
        available = ", ".join(templates.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    # Get template configuration
    template_config = templates[template_name].copy()
    
    # Apply overrides
    for key, value in overrides.items():
        if key == "metadata":
            # Merge metadata instead of replacing
            template_config["metadata"].update(value)
        else:
            template_config[key] = value
    
    return create_context(**template_config)


def adaptive_context(base_entropy: float = 0.5, 
                    adaptation_rate: float = 0.1,
                    entropy_bounds: tuple = (0.1, 0.9)) -> ExecutionContext:
    """
    Create an adaptive context that can modify its entropy dynamically.
    
    Adaptive contexts adjust their entropy based on execution feedback,
    allowing for dynamic optimization during recursive operations.
    
    Args:
        base_entropy: Starting entropy level
        adaptation_rate: Rate of entropy adjustment (0.0 - 1.0)
        entropy_bounds: (min_entropy, max_entropy) bounds for adaptation
        
    Returns:
        ExecutionContext with adaptive capabilities
        
    Example:
        adaptive_ctx = fracton.adaptive_context(
            base_entropy=0.6,
            adaptation_rate=0.2,
            entropy_bounds=(0.2, 0.8)
        )
    """
    min_entropy, max_entropy = entropy_bounds
    
    if not (0.0 <= min_entropy <= max_entropy <= 1.0):
        raise ValueError("Invalid entropy bounds")
    
    if not (0.0 <= adaptation_rate <= 1.0):
        raise ValueError("Adaptation rate must be between 0.0 and 1.0")
    
    clamped_entropy = max(min_entropy, min(max_entropy, base_entropy))
    
    return ExecutionContext(
        entropy=clamped_entropy,
        depth=0,
        metadata={
            "adaptive": True,
            "adaptation_rate": adaptation_rate,
            "entropy_bounds": entropy_bounds,
            "base_entropy": base_entropy,
            "creation_time": time.time()
        }
    )


def context_pipeline(*contexts: ExecutionContext) -> ExecutionContext:
    """
    Create a context pipeline that evolves through multiple stages.
    
    Context pipelines provide a way to define multi-stage processing where
    each stage has different entropy and execution characteristics.
    
    Args:
        *contexts: Sequence of contexts representing pipeline stages
        
    Returns:
        ExecutionContext configured for pipeline execution
        
    Example:
        # Create a three-stage pipeline
        stage1 = fracton.Context(entropy=0.8)  # Exploration
        stage2 = fracton.Context(entropy=0.5)  # Processing  
        stage3 = fracton.Context(entropy=0.2)  # Crystallization
        
        pipeline_ctx = fracton.context_pipeline(stage1, stage2, stage3)
    """
    if not contexts:
        return ExecutionContext()
    
    if len(contexts) == 1:
        return contexts[0]
    
    # Use first context as base
    base_context = contexts[0]
    
    # Store pipeline stages in metadata
    pipeline_stages = []
    for i, ctx in enumerate(contexts):
        pipeline_stages.append({
            "stage": i,
            "entropy": ctx.entropy,
            "depth": ctx.depth,
            "metadata": ctx.metadata.copy()
        })
    
    return ExecutionContext(
        entropy=base_context.entropy,
        depth=base_context.depth,
        trace_id=base_context.trace_id,
        field_state=base_context.field_state.copy(),
        metadata={
            **base_context.metadata,
            "pipeline": True,
            "pipeline_stages": pipeline_stages,
            "current_stage": 0,
            "total_stages": len(contexts)
        }
    )


def entropy_schedule(entropy_function: callable, 
                    max_depth: int = 100) -> ExecutionContext:
    """
    Create a context with scheduled entropy evolution.
    
    Entropy schedules define how entropy should change as recursion depth
    increases, allowing for sophisticated entropy management strategies.
    
    Args:
        entropy_function: Function that takes depth and returns entropy (0.0-1.0)
        max_depth: Maximum depth for the schedule
        
    Returns:
        ExecutionContext with entropy scheduling
        
    Example:
        # Decreasing entropy schedule (starts high, becomes more ordered)
        def decreasing_entropy(depth):
            return max(0.1, 0.9 - (depth * 0.05))
        
        scheduled_ctx = fracton.entropy_schedule(decreasing_entropy, max_depth=50)
        
        # Oscillating entropy schedule
        import math
        def oscillating_entropy(depth):
            return 0.5 + 0.3 * math.sin(depth * 0.5)
        
        oscillating_ctx = fracton.entropy_schedule(oscillating_entropy)
    """
    try:
        # Validate entropy function
        test_entropy = entropy_function(0)
        if not (0.0 <= test_entropy <= 1.0):
            raise ValueError("Entropy function must return values between 0.0 and 1.0")
    except Exception as e:
        raise ValueError(f"Invalid entropy function: {e}")
    
    initial_entropy = entropy_function(0)
    
    return ExecutionContext(
        entropy=initial_entropy,
        depth=0,
        metadata={
            "entropy_scheduled": True,
            "entropy_function": entropy_function,
            "max_scheduled_depth": max_depth,
            "schedule_type": "function"
        }
    )


def context_with_constraints(entropy: float = 0.5,
                           max_recursion_depth: Optional[int] = None,
                           execution_time_limit: Optional[float] = None,
                           memory_limit: Optional[int] = None,
                           **metadata) -> ExecutionContext:
    """
    Create a context with execution constraints.
    
    Constrained contexts enforce limits on recursion depth, execution time,
    and memory usage to prevent runaway computation.
    
    Args:
        entropy: Initial entropy level
        max_recursion_depth: Maximum allowed recursion depth
        execution_time_limit: Maximum execution time in seconds
        memory_limit: Maximum memory usage in MB
        **metadata: Additional context metadata
        
    Returns:
        ExecutionContext with constraint enforcement
        
    Example:
        constrained_ctx = fracton.context_with_constraints(
            entropy=0.7,
            max_recursion_depth=100,
            execution_time_limit=30.0,  # 30 seconds
            memory_limit=500,  # 500 MB
            experiment="bounded_computation"
        )
    """
    constraints = {}
    
    if max_recursion_depth is not None:
        if max_recursion_depth <= 0:
            raise ValueError("max_recursion_depth must be positive")
        constraints["max_recursion_depth"] = max_recursion_depth
    
    if execution_time_limit is not None:
        if execution_time_limit <= 0:
            raise ValueError("execution_time_limit must be positive")
        constraints["execution_time_limit"] = execution_time_limit
        constraints["start_time"] = time.time()
    
    if memory_limit is not None:
        if memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        constraints["memory_limit"] = memory_limit
    
    return ExecutionContext(
        entropy=entropy,
        depth=0,
        metadata={
            **metadata,
            "constrained": True,
            "constraints": constraints
        }
    )


def merge_context_metadata(*contexts: ExecutionContext) -> Dict[str, Any]:
    """
    Merge metadata from multiple contexts with conflict resolution.
    
    Args:
        *contexts: Contexts whose metadata should be merged
        
    Returns:
        Merged metadata dictionary
        
    Example:
        ctx1 = fracton.Context(experiment="test1", run=1)
        ctx2 = fracton.Context(experiment="test2", phase="analysis")
        
        merged_metadata = fracton.merge_context_metadata(ctx1, ctx2)
        # Result: {"experiment": "test2", "run": 1, "phase": "analysis"}
    """
    merged = {}
    
    for context in contexts:
        for key, value in context.metadata.items():
            if key in merged:
                # Handle conflicts by creating a list
                if not isinstance(merged[key], list):
                    merged[key] = [merged[key]]
                if value not in merged[key]:
                    merged[key].append(value)
            else:
                merged[key] = value
    
    return merged


def context_diff(context1: ExecutionContext, context2: ExecutionContext) -> Dict[str, Any]:
    """
    Calculate differences between two execution contexts.
    
    Args:
        context1: First context for comparison
        context2: Second context for comparison
        
    Returns:
        Dictionary describing differences between contexts
        
    Example:
        diff = fracton.context_diff(old_context, new_context)
        print(f"Entropy changed by: {diff['entropy_delta']}")
        print(f"Depth changed by: {diff['depth_delta']}")
    """
    return {
        "entropy_delta": context2.entropy - context1.entropy,
        "depth_delta": context2.depth - context1.depth,
        "trace_id_changed": context1.trace_id != context2.trace_id,
        "metadata_added": set(context2.metadata.keys()) - set(context1.metadata.keys()),
        "metadata_removed": set(context1.metadata.keys()) - set(context2.metadata.keys()),
        "metadata_changed": {
            key: (context1.metadata.get(key), context2.metadata.get(key))
            for key in set(context1.metadata.keys()) & set(context2.metadata.keys())
            if context1.metadata.get(key) != context2.metadata.get(key)
        },
        "field_state_changed": context1.field_state != context2.field_state
    }
