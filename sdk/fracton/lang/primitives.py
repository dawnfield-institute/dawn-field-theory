"""
Fracton Language Primitives - Core runtime functions

This module provides the fundamental runtime functions for the Fracton language,
including recursive calls, crystallization, branching, and context management.
"""

import uuid
from typing import Any, Callable, Dict, List, Optional, Union
from ..core.recursive_engine import ExecutionContext, get_default_executor, Continuation
from ..core.bifractal_trace import BifractalTrace
from ..core.memory_field import MemoryField


def recurse(func: Callable, memory: MemoryField, context: ExecutionContext, 
           trace: Optional[BifractalTrace] = None, max_depth: Optional[int] = None) -> Any:
    """
    Initiate a recursive call with proper tracing and context management.
    
    This is the primary function for making recursive calls within the Fracton
    runtime. It handles entropy checking, trace recording, and execution context
    propagation with built-in recursion safety.
    
    Args:
        func: The function to call recursively
        memory: Shared memory field
        context: Execution context for the call
        trace: Optional bifractal trace for recording the call
        max_depth: Override default maximum recursion depth
        
    Returns:
        Result of the recursive function call
        
    Raises:
        StackOverflowError: If recursion depth exceeds limits
        EntropyGateError: If entropy conditions are not met
        
    Example:
        @fracton.recursive
        def fibonacci(memory, context):
            if context.depth <= 1:
                return 1
            
            a = fracton.recurse(fibonacci, memory, context.deeper(1))
            b = fracton.recurse(fibonacci, memory, context.deeper(2))
            return a + b
    """
    # Safety check: prevent infinite recursion
    effective_max_depth = max_depth or 100  # Conservative default
    if context.depth >= effective_max_depth:
        from ..core.recursive_engine import StackOverflowError
        raise StackOverflowError(context.depth, effective_max_depth)
    
    # Additional safety: check if function is marked as recursive
    if not hasattr(func, '_fracton_recursive') or not func._fracton_recursive:
        raise ValueError(f"Function {func.__name__} must be decorated with @recursive to use recurse()")
    
    # Check if we're inside a trampoline execution
    import threading
    current_thread = threading.current_thread()
    in_trampoline = getattr(current_thread, '_fracton_in_trampoline', False)
    
    if in_trampoline:
        # We're inside a trampoline, return a continuation
        if trace:
            entry_id = trace.record_call(func, context)
        return Continuation(func, memory, context)
    else:
        # We're called directly, execute through the executor with trampoline
        executor = get_default_executor()
        return executor.execute(func, memory, context)


def crystallize(data: Any, memory: Optional[MemoryField] = None, 
                context: Optional[ExecutionContext] = None,
                patterns: Optional[List] = None, 
                entropy_threshold: float = 0.3,
                trace: Optional[BifractalTrace] = None) -> Any:
    """
    Crystallize data into stable structures based on entropy patterns.
    
    This function analyzes data and reduces its entropy by identifying and
    reinforcing stable patterns, creating more organized structures.
    
    Args:
        data: Data to crystallize
        memory: Optional memory field to store crystallized data
        context: Optional execution context for entropy awareness
        patterns: Optional list of patterns to reinforce
        entropy_threshold: Entropy level below which crystallization occurs
        trace: Optional bifractal trace for recording crystallization
        
    Returns:
        If memory is provided, returns the storage ID. Otherwise returns crystallized data.
        
    Example:
        chaotic_data = {"values": [1, 5, 2, 8, 3, 7, 4, 6]}
        stable_data = fracton.crystallize(chaotic_data)
        # Result: {"values": [1, 2, 3, 4, 5, 6, 7, 8]}  # Sorted
    """
    # Use context entropy if available, otherwise calculate from data
    if context and hasattr(context, 'entropy'):
        current_entropy = context.entropy
    else:
        current_entropy = _calculate_data_entropy(data)
    
    # Record in trace if provided
    if trace:
        trace_id = trace.record_operation(
            operation_type="crystallization",
            context=context,
            input_data={"data": data, "entropy_threshold": entropy_threshold}
        )
    
    if current_entropy <= entropy_threshold:
        # Already stable enough
        crystallized_data = data
    else:
        # Apply crystallization based on data type
        if isinstance(data, dict):
            crystallized_data = _crystallize_dict(data, patterns)
        elif isinstance(data, list):
            crystallized_data = _crystallize_list(data, patterns)
        elif isinstance(data, str):
            crystallized_data = _crystallize_string(data, patterns)
        else:
            # For other types, try to find inherent order
            crystallized_data = _crystallize_generic(data, patterns)
    
    # Update trace with result if provided
    if trace:
        trace.record_operation(
            operation_type="crystallization_complete",
            context=context,
            output_data={"crystallized_data": crystallized_data},
            parent_operation=trace_id
        )
    
    # Store in memory if provided
    if memory:
        crystal_id = f"crystal_{uuid.uuid4().hex[:8]}"
        memory.set(crystal_id, crystallized_data)
        return crystal_id
    else:
        return crystallized_data


def branch(condition: Union[bool, Callable, List], 
          if_true: Union[Callable, List, MemoryField] = None, 
          if_false: Union[Callable, ExecutionContext, List] = None,
          memory: MemoryField = None, context: Union[ExecutionContext, List] = None) -> Any:
    """
    Entropy-aware conditional branching for recursive operations.
    
    This function provides conditional execution that considers entropy levels
    and execution context for optimal path selection.
    
    Args:
        condition: Boolean condition, function that returns boolean, or list of functions
        if_true: Function to call if condition is true, or MemoryField for multi-branch
        if_false: Function to call if condition is false, or contexts for multi-branch
        memory: Shared memory field (traditional mode)
        context: Execution context or list of contexts for multi-branch
        
    Returns:
        Result of the selected branch function, or list of results for multi-branch
        
    Example:
        # Simple branch
        result = fracton.branch(
            context.entropy > 0.5,
            high_entropy_path,
            low_entropy_path,
            memory,
            context
        )
        
        # Multi-branch over contexts: branch(functions_list, memory, contexts_list)
        results = fracton.branch(
            [func1, func2],
            memory,
            [context1, context2]
        )
    """
    # Handle multi-branch case: branch([func1, func2], memory, [ctx1, ctx2])
    if isinstance(condition, list) and if_false is not None and isinstance(if_false, list):
        functions = condition
        memory_field = if_true  # Second arg is memory
        contexts = if_false     # Third arg is contexts list
        results = []
        for func, ctx in zip(functions, contexts):
            # For non-recursive functions, call directly
            if hasattr(func, '_fracton_recursive') and func._fracton_recursive:
                result = recurse(func, memory_field, ctx)
            else:
                result = func(memory_field, ctx)
            results.append(result)
        return results
    
    # Handle traditional branch case: branch(condition, if_true, if_false, memory, context)
    # Evaluate condition if it's a callable
    if callable(condition):
        condition_result = condition(memory, context)
    else:
        condition_result = condition
    
    # Select and execute branch
    selected_func = if_true if condition_result else if_false
    
    # For non-recursive functions, call directly
    if hasattr(selected_func, '_fracton_recursive') and selected_func._fracton_recursive:
        return recurse(selected_func, memory, context)
    else:
        return selected_func(memory, context)


def merge_contexts(*contexts: ExecutionContext) -> ExecutionContext:
    """
    Merge multiple execution contexts into a single context.
    
    This function combines entropy levels, metadata, and field states from
    multiple contexts to create a unified execution environment.
    
    Args:
        *contexts: Variable number of ExecutionContext objects to merge
        
    Returns:
        New ExecutionContext with merged properties
        
    Example:
        context1 = ExecutionContext(entropy=0.6, depth=5)
        context2 = ExecutionContext(entropy=0.8, depth=3)
        merged = fracton.merge_contexts(context1, context2)
        # merged.entropy = 0.7 (average)
        # merged.depth = 5 (maximum)
    """
    if not contexts:
        return ExecutionContext()
    
    if len(contexts) == 1:
        return contexts[0]
    
    # Calculate merged properties
    entropies = [c.entropy for c in contexts]
    depths = [c.depth for c in contexts]
    
    # Average entropy, max depth
    merged_entropy = sum(entropies) / len(entropies)
    merged_depth = max(depths)
    
    # Merge metadata (later contexts override earlier ones)
    merged_metadata = {}
    for context in contexts:
        merged_metadata.update(context.metadata)
    
    # Merge field states
    merged_field_state = {}
    for context in contexts:
        merged_field_state.update(context.field_state)
    
    return ExecutionContext(
        entropy=merged_entropy,
        depth=merged_depth,
        metadata=merged_metadata,
        field_state=merged_field_state
    )


def regulate_entropy(memory: MemoryField, target_entropy: float, 
                    adjustment_factor: float = 0.1) -> float:
    """
    Adjust memory field entropy toward a target value.
    
    This function modifies the memory field to bring its entropy closer to
    the target value through controlled transformations.
    
    Args:
        memory: Memory field to adjust
        target_entropy: Desired entropy level (0.0 - 1.0)
        adjustment_factor: Rate of adjustment (0.0 - 1.0)
        
    Returns:
        New entropy level after adjustment
        
    Example:
        current_entropy = memory.get_entropy()  # 0.9 (too high)
        new_entropy = fracton.regulate_entropy(memory, 0.6)
        # new_entropy ≈ 0.87 (moved toward target)
    """
    current_entropy = memory.get_entropy()
    
    if abs(current_entropy - target_entropy) < 0.01:
        return current_entropy  # Already close enough
    
    # Calculate adjustment direction and magnitude
    entropy_diff = target_entropy - current_entropy
    adjustment = entropy_diff * adjustment_factor
    
    # Apply entropy regulation through memory transformations
    if entropy_diff > 0:
        # Need to increase entropy (add randomness)
        _increase_memory_entropy(memory, adjustment)
    else:
        # Need to decrease entropy (add order)
        _decrease_memory_entropy(memory, abs(adjustment))
    
    return memory.get_entropy()


def detect_patterns(memory: MemoryField, min_confidence: float = 0.7) -> List[Dict]:
    """
    Detect patterns in memory field contents.
    
    This function analyzes the memory field to identify recurring patterns,
    relationships, and structures that can inform future operations.
    
    Args:
        memory: Memory field to analyze
        min_confidence: Minimum confidence level for pattern detection
        
    Returns:
        List of detected patterns with metadata
        
    Example:
        patterns = fracton.detect_patterns(memory, min_confidence=0.8)
        for pattern in patterns:
            print(f"Pattern: {pattern['type']}, Confidence: {pattern['confidence']}")
    """
    patterns = []
    content = dict(memory.items())
    
    if not content:
        return patterns
    
    # Detect value type patterns
    type_pattern = _detect_type_patterns(content, min_confidence)
    if type_pattern:
        patterns.append(type_pattern)
    
    # Detect sequence patterns
    sequence_pattern = _detect_sequence_patterns(content, min_confidence)
    if sequence_pattern:
        patterns.append(sequence_pattern)
    
    # Detect hierarchical patterns
    hierarchy_pattern = _detect_hierarchy_patterns(content, min_confidence)
    if hierarchy_pattern:
        patterns.append(hierarchy_pattern)
    
    return patterns


# Helper functions for internal operations

def _calculate_data_entropy(data: Any) -> float:
    """Calculate entropy of arbitrary data structure."""
    import math
    from collections import Counter
    
    if isinstance(data, (list, tuple)):
        if not data:
            return 0.0
        
        # Count frequencies of items
        counter = Counter(str(item) for item in data)
        total = len(data)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize to 0-1 range
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    elif isinstance(data, dict):
        if not data:
            return 0.0
        
        # Calculate entropy based on value diversity
        values = list(data.values())
        return _calculate_data_entropy(values)
    
    elif isinstance(data, str):
        if not data:
            return 0.0
        
        # Character frequency entropy
        counter = Counter(data.lower())
        total = len(data)
        
        entropy = 0.0
        for count in counter.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    else:
        # For other types, use string representation
        return _calculate_data_entropy(str(data))


def _crystallize_dict(data: dict, patterns: Optional[List]) -> dict:
    """Crystallize dictionary data by organizing keys and values."""
    # Sort keys for consistent ordering
    crystallized = {}
    for key in sorted(data.keys(), key=str):
        value = data[key]
        
        # Recursively crystallize nested structures
        if isinstance(value, dict):
            crystallized[key] = _crystallize_dict(value, patterns)
        elif isinstance(value, list):
            crystallized[key] = _crystallize_list(value, patterns)
        else:
            crystallized[key] = value
    
    return crystallized


def _crystallize_list(data: list, patterns: Optional[List]) -> list:
    """Crystallize list data by organizing elements."""
    if not data:
        return data
    
    # Try to sort if all elements are comparable
    try:
        return sorted(data, key=str)
    except:
        # If sorting fails, group by type
        grouped = {}
        for item in data:
            type_name = type(item).__name__
            if type_name not in grouped:
                grouped[type_name] = []
            grouped[type_name].append(item)
        
        # Return flattened grouped data
        result = []
        for type_name in sorted(grouped.keys()):
            result.extend(grouped[type_name])
        
        return result


def _crystallize_string(data: str, patterns: Optional[List]) -> str:
    """Crystallize string data by normalizing formatting."""
    # Basic string crystallization: normalize whitespace and case
    return ' '.join(data.split()).strip()


def _crystallize_generic(data: Any, patterns: Optional[List]) -> Any:
    """Crystallize generic data types."""
    # For unknown types, return as-is
    return data


def _increase_memory_entropy(memory: MemoryField, adjustment: float) -> None:
    """Increase entropy in memory field by adding randomness."""
    import random
    
    # Add some random key-value pairs
    num_additions = max(1, int(adjustment * 10))
    for i in range(num_additions):
        random_key = f"entropy_boost_{random.randint(1000, 9999)}"
        random_value = random.choice([
            random.random(),
            random.randint(1, 100),
            f"random_string_{random.randint(1, 1000)}"
        ])
        memory.set(random_key, random_value)


def _decrease_memory_entropy(memory: MemoryField, adjustment: float) -> None:
    """Decrease entropy in memory field by adding order."""
    # Remove random elements to increase order
    keys = memory.keys()
    if keys:
        num_removals = max(1, int(adjustment * len(keys)))
        keys_to_remove = keys[:num_removals]
        for key in keys_to_remove:
            memory.delete(key)


def _detect_type_patterns(content: dict, min_confidence: float) -> Optional[Dict]:
    """Detect patterns in data types."""
    if not content:
        return None
    
    from collections import Counter
    
    # Count value types
    type_counter = Counter()
    for value in content.values():
        type_counter[type(value).__name__] += 1
    
    total_values = len(content)
    most_common_type, count = type_counter.most_common(1)[0]
    confidence = count / total_values
    
    if confidence >= min_confidence:
        return {
            'type': 'value_type_pattern',
            'dominant_type': most_common_type,
            'confidence': confidence,
            'frequency': count,
            'total': total_values
        }
    
    return None


def _detect_sequence_patterns(content: dict, min_confidence: float) -> Optional[Dict]:
    """Detect sequential patterns in keys or values."""
    keys = list(content.keys())
    
    # Check for numeric sequence in keys
    numeric_keys = []
    for key in keys:
        try:
            if isinstance(key, (int, float)):
                numeric_keys.append(key)
            elif isinstance(key, str) and key.isdigit():
                numeric_keys.append(int(key))
        except:
            pass
    
    if len(numeric_keys) >= 3:
        numeric_keys.sort()
        # Check if it's an arithmetic sequence
        differences = [numeric_keys[i+1] - numeric_keys[i] for i in range(len(numeric_keys)-1)]
        
        if len(set(differences)) == 1:  # All differences are the same
            confidence = len(numeric_keys) / len(keys)
            if confidence >= min_confidence:
                return {
                    'type': 'arithmetic_sequence',
                    'difference': differences[0],
                    'confidence': confidence,
                    'length': len(numeric_keys)
                }
    
    return None


def _detect_hierarchy_patterns(content: dict, min_confidence: float) -> Optional[Dict]:
    """Detect hierarchical patterns in nested structures."""
    nested_count = 0
    total_count = len(content)
    
    for value in content.values():
        if isinstance(value, (dict, list)):
            nested_count += 1
    
    if total_count == 0:
        return None
    
    confidence = nested_count / total_count
    
    if confidence >= min_confidence:
        return {
            'type': 'hierarchical_pattern',
            'nested_ratio': confidence,
            'confidence': confidence,
            'nested_count': nested_count,
            'total_count': total_count
        }
    
    return None
