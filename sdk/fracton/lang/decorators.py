"""
Fracton Decorators - High-level function decoration for Fracton language features

This module provides the core decorators that transform regular Python functions
into Fracton-aware recursive functions with entropy gating and tool expression.
"""

import functools
from typing import Any, Callable, Dict, Optional, Union
from ..core import get_default_executor, get_default_dispatcher, DispatchConditions


def recursive(func: Callable = None, *, 
             max_depth: Optional[int] = None,
             enable_tracing: bool = True) -> Callable:
    """
    Mark a function as recursively callable within the Fracton runtime.
    
    This decorator transforms a regular function into a Fracton-aware recursive
    function that can be called through the recursive execution engine.
    
    Args:
        func: The function to decorate
        max_depth: Maximum recursion depth (overrides executor default)
        enable_tracing: Whether to enable bifractal tracing for this function
        
    Returns:
        Decorated function that can be used with fracton.recurse()
        
    Example:
        @fracton.recursive
        def fibonacci(memory, context):
            if context.depth <= 1:
                return 1
            
            a = fracton.recurse(fibonacci, memory, context.deeper(1))
            b = fracton.recurse(fibonacci, memory, context.deeper(2))
            return a + b
    """
    def decorator(f: Callable) -> Callable:
        # Mark function as Fracton-aware
        f._fracton_recursive = True
        f._fracton_max_depth = max_depth
        f._fracton_enable_tracing = enable_tracing
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # If called directly (not through fracton.recurse), execute normally
            return f(*args, **kwargs)
        
        # Copy fracton metadata to wrapper
        wrapper._fracton_recursive = True
        wrapper._fracton_max_depth = max_depth
        wrapper._fracton_enable_tracing = enable_tracing
        wrapper._fracton_original = f
        
        return wrapper
    
    if func is None:
        # Called with arguments: @recursive(max_depth=100)
        return decorator
    else:
        # Called without arguments: @recursive
        return decorator(func)


def entropy_gate(min_threshold: float, max_threshold: float = 1.0, 
                auto_adjust: bool = False) -> Callable:
    """
    Set entropy thresholds for function execution.
    
    Functions decorated with this will only execute when the execution context
    entropy falls within the specified range.
    
    Args:
        min_threshold: Minimum entropy required for execution (0.0 - 1.0)
        max_threshold: Maximum entropy allowed for execution (0.0 - 1.0)
        auto_adjust: Whether to automatically adjust thresholds based on performance
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If thresholds are invalid
        
    Example:
        @fracton.recursive
        @fracton.entropy_gate(0.5, 0.9)
        def high_entropy_operation(memory, context):
            # Only executes when 0.5 <= context.entropy <= 0.9
            return process_complex_data(memory)
    """
    if not (0.0 <= min_threshold <= 1.0) or not (0.0 <= max_threshold <= 1.0):
        raise ValueError("Entropy thresholds must be between 0.0 and 1.0")
    
    if min_threshold > max_threshold:
        raise ValueError("min_threshold cannot be greater than max_threshold")
    
    def decorator(func: Callable) -> Callable:
        # Register entropy gate with the default executor
        executor = get_default_executor()
        executor.register_entropy_gate(func, min_threshold, max_threshold)
        
        # Register with dispatcher for context-aware selection
        dispatcher = get_default_dispatcher()
        conditions = DispatchConditions(
            min_entropy=min_threshold,
            max_entropy=max_threshold
        )
        dispatcher.register_function(func, conditions)
        
        # Store entropy gate metadata
        func._fracton_entropy_gate = (min_threshold, max_threshold)
        func._fracton_auto_adjust = auto_adjust
        
        return func
    
    return decorator


def tool_binding(tool_name: str, *, 
                context_sensitive: bool = True,
                require_permissions: Optional[list] = None) -> Callable:
    """
    Mark a function as a tool expression interface.
    
    Functions decorated with this become tool expression handlers that can
    access external systems through the Fracton tool framework.
    
    Args:
        tool_name: Name of the tool this function binds to
        context_sensitive: Whether tool access depends on execution context
        require_permissions: List of required permissions for tool access
        
    Returns:
        Decorator function
        
    Example:
        @fracton.tool_binding("github")
        @fracton.recursive
        def github_operations(memory, context):
            if context.entropy > 0.7:
                return fracton.express_tool("github", "create_issue", context.data)
            else:
                return fracton.express_tool("github", "list_issues")
    """
    def decorator(func: Callable) -> Callable:
        # Store tool binding metadata
        func._fracton_tool_binding = tool_name
        func._fracton_context_sensitive = context_sensitive
        func._fracton_require_permissions = require_permissions or []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Tool binding functions are called normally
            # Tool expression happens inside the function via fracton.express_tool
            return func(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._fracton_tool_binding = tool_name
        wrapper._fracton_context_sensitive = context_sensitive
        wrapper._fracton_require_permissions = require_permissions or []
        
        return wrapper
    
    return decorator


def tail_recursive(func: Callable) -> Callable:
    """
    Mark a function as tail-recursive for optimization.
    
    Tail-recursive functions can be optimized to reuse stack frames and
    avoid stack overflow for deep recursion.
    
    Args:
        func: The function to mark as tail-recursive
        
    Returns:
        Decorated function with tail recursion optimization
        
    Example:
        @fracton.recursive
        @fracton.tail_recursive
        def countdown(memory, context):
            if context.depth <= 0:
                return "done"
            return fracton.recurse(countdown, memory, context.deeper(-1))
    """
    # Register with executor for tail recursion optimization
    executor = get_default_executor()
    executor.register_tail_recursive(func)
    
    # Store tail recursive metadata
    func._fracton_tail_recursive = True
    
    return func


def fallback(fallback_func: Callable) -> Callable:
    """
    Specify a fallback function for error recovery.
    
    If the decorated function fails, the fallback function will be called
    with the same arguments for graceful degradation.
    
    Args:
        fallback_func: Function to call if the main function fails
        
    Returns:
        Decorator function
        
    Example:
        def simple_fallback(memory, context):
            return "fallback_result"
        
        @fracton.recursive
        @fracton.fallback(simple_fallback)
        def complex_operation(memory, context):
            # Complex logic that might fail
            return risky_computation(memory, context)
    """
    def decorator(func: Callable) -> Callable:
        func._fracton_fallback = fallback_func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try fallback function
                try:
                    return fallback_func(*args, **kwargs)
                except Exception:
                    # Re-raise original exception if fallback also fails
                    raise e
        
        # Copy metadata
        wrapper._fracton_fallback = fallback_func
        
        return wrapper
    
    return decorator


def memoize(max_cache_size: int = 1000, 
           entropy_sensitive: bool = True) -> Callable:
    """
    Add memoization to a recursive function.
    
    Caches function results based on arguments and optionally entropy level
    to avoid redundant computation in recursive calls.
    
    Args:
        max_cache_size: Maximum number of cached results
        entropy_sensitive: Whether cache depends on entropy level
        
    Returns:
        Decorator function
        
    Example:
        @fracton.recursive
        @fracton.memoize(max_cache_size=500)
        def expensive_computation(memory, context):
            # Expensive recursive computation
            return complex_calculation(memory, context)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(args), str(sorted(kwargs.items()))]
            
            # Include entropy in key if entropy_sensitive
            if entropy_sensitive and len(args) >= 2:
                # Assume second argument is context with entropy
                try:
                    context = args[1]
                    if hasattr(context, 'entropy'):
                        entropy_bucket = round(context.entropy, 1)  # Round to nearest 0.1
                        key_parts.append(str(entropy_bucket))
                except:
                    pass
            
            cache_key = "|".join(key_parts)
            
            # Check cache
            if cache_key in cache:
                return cache[cache_key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = result
            cache_order.append(cache_key)
            
            # Limit cache size
            if len(cache) > max_cache_size:
                # Remove oldest entry
                oldest_key = cache_order.pop(0)
                if oldest_key in cache:
                    del cache[oldest_key]
            
            return result
        
        # Store memoization metadata
        wrapper._fracton_memoize = True
        wrapper._fracton_cache_size = max_cache_size
        wrapper._fracton_entropy_sensitive = entropy_sensitive
        wrapper._fracton_cache = cache  # For inspection/debugging
        
        return wrapper
    
    return decorator


def profile(include_entropy: bool = True, 
           include_memory: bool = False) -> Callable:
    """
    Add profiling to a recursive function.
    
    Collects performance metrics for recursive function calls including
    execution time, entropy evolution, and optionally memory usage.
    
    Args:
        include_entropy: Whether to track entropy evolution
        include_memory: Whether to track memory usage (expensive)
        
    Returns:
        Decorator function
        
    Example:
        @fracton.recursive
        @fracton.profile(include_memory=True)
        def performance_critical_function(memory, context):
            return intensive_computation(memory, context)
    """
    def decorator(func: Callable) -> Callable:
        import time
        import sys
        
        profile_data = {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'entropy_history': [],
            'memory_history': []
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Track memory if requested
            if include_memory:
                start_memory = sys.getsizeof(args) + sys.getsizeof(kwargs)
            
            # Track entropy if requested and available
            if include_entropy and len(args) >= 2:
                try:
                    context = args[1]
                    if hasattr(context, 'entropy'):
                        profile_data['entropy_history'].append(
                            (start_time, context.entropy)
                        )
                except:
                    pass
            
            try:
                result = func(*args, **kwargs)
                
                # Update timing statistics
                execution_time = time.time() - start_time
                profile_data['call_count'] += 1
                profile_data['total_time'] += execution_time
                profile_data['min_time'] = min(profile_data['min_time'], execution_time)
                profile_data['max_time'] = max(profile_data['max_time'], execution_time)
                
                # Track memory if requested
                if include_memory:
                    end_memory = sys.getsizeof(result) if result is not None else 0
                    memory_delta = end_memory - start_memory
                    profile_data['memory_history'].append(
                        (start_time, memory_delta)
                    )
                
                return result
                
            except Exception as e:
                # Still update timing for failed calls
                execution_time = time.time() - start_time
                profile_data['call_count'] += 1
                profile_data['total_time'] += execution_time
                raise
        
        # Store profiling metadata and data
        wrapper._fracton_profile = True
        wrapper._fracton_profile_data = profile_data
        wrapper._fracton_include_entropy = include_entropy
        wrapper._fracton_include_memory = include_memory
        
        return wrapper
    
    return decorator
