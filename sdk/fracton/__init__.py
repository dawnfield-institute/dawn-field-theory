"""
Fracton: Infodynamics Computational Modeling Language

Fracton is a domain-specific computational modeling language designed for 
infodynamics research and recursive field-aware systems. It provides a 
unified substrate for modeling emergent intelligence, entropy dynamics, 
and bifractal computation patterns.

Main Features:
- Recursive execution as first-class primitive
- Entropy-driven function dispatch
- Bifractal tracing for operation analysis
- Field-aware memory management
- Tool expression framework

Example Usage:
    import fracton
    
    @fracton.recursive
    @fracton.entropy_gate(0.5)
    def fibonacci_field(memory, context):
        if context.depth <= 1:
            return 1
        
        a = fracton.recurse(fibonacci_field, memory, context.deeper(1))
        b = fracton.recurse(fibonacci_field, memory, context.deeper(2))
        return a + b
    
    with fracton.memory_field() as field:
        context = fracton.Context(depth=10, entropy=0.8)
        result = fibonacci_field(field, context)
"""

# Core runtime components
from .core import (
    RecursiveExecutor, ExecutionContext, 
    EntropyDispatcher, DispatchConditions,
    BifractalTrace, TraceAnalysis,
    MemoryField, FieldController,
    memory_field
)

# Language constructs
from .lang import (
    recursive, entropy_gate, tool_binding, tail_recursive,
    recurse, crystallize, branch, merge_contexts,
    Context, create_context
)

# Version info
__version__ = "0.1.0"
__author__ = "Dawn Field Institute"
__email__ = "info@dawnfield.org"

# Main API exports
__all__ = [
    # Core execution
    "RecursiveExecutor",
    "ExecutionContext", 
    "Context",
    "create_context",
    
    # Decorators
    "recursive",
    "entropy_gate",
    "tool_binding", 
    "tail_recursive",
    
    # Runtime functions
    "recurse",
    "crystallize",
    "branch",
    "merge_contexts",
    
    # Memory management
    "MemoryField",
    "FieldController",
    "memory_field",
    
    # Dispatch and analysis
    "EntropyDispatcher",
    "DispatchConditions",
    "BifractalTrace",
    "TraceAnalysis",
    
    # Utilities
    "initialize_field",
    "analyze_trace",
    "visualize_trace",
    "express_tool"
]


def initialize_field(capacity: int = 1000, entropy: float = 0.5) -> MemoryField:
    """
    Initialize a new memory field with specified capacity and entropy.
    
    Args:
        capacity: Maximum number of items in the field
        entropy: Initial entropy level (0.0 - 1.0)
        
    Returns:
        New MemoryField instance
        
    Example:
        field = fracton.initialize_field(capacity=2000, entropy=0.7)
        field.set("data", my_data)
    """
    from .core.memory_field import get_default_controller
    controller = get_default_controller()
    return controller.create_field(capacity, entropy)


def analyze_trace(trace: BifractalTrace) -> TraceAnalysis:
    """
    Analyze a bifractal trace for patterns and performance insights.
    
    Args:
        trace: BifractalTrace to analyze
        
    Returns:
        TraceAnalysis with comprehensive insights
        
    Example:
        analysis = fracton.analyze_trace(my_trace)
        print(f"Max depth: {analysis.max_depth}")
        print(f"Total calls: {analysis.total_calls}")
    """
    return trace.analyze_patterns()


def visualize_trace(trace: BifractalTrace, format: str = "text", **kwargs) -> str:
    """
    Generate visualization of a bifractal trace.
    
    Args:
        trace: BifractalTrace to visualize
        format: Visualization format ("text", "graph", "timeline")
        **kwargs: Additional visualization options
        
    Returns:
        Visualization string or data structure
        
    Example:
        text_viz = fracton.visualize_trace(my_trace, "text", max_depth=20)
        print(text_viz)
    """
    if format == "text":
        return trace.visualize_text(kwargs.get("max_depth", 50))
    elif format == "graph":
        return trace.visualize_graph()
    elif format == "timeline":
        return trace.get_entropy_timeline()
    else:
        raise ValueError(f"Unsupported visualization format: {format}")


def express_tool(tool_name: str, operation: str = None, *args, **kwargs):
    """
    Express (access) an external tool through the Fracton tool framework.
    
    This function provides context-aware access to external systems based
    on the current execution context and entropy levels.
    
    Args:
        tool_name: Name of the tool to access
        operation: Specific operation to perform
        *args: Operation arguments
        **kwargs: Operation keyword arguments
        
    Returns:
        Result of tool operation
        
    Example:
        # Basic tool expression
        result = fracton.express_tool("github", "create_issue", 
                                    title="Bug report", 
                                    body="Description of the bug")
        
        # Context-sensitive expression
        if context.entropy > 0.8:
            data = fracton.express_tool("database", "experimental_query", query)
        else:
            data = fracton.express_tool("database", "stable_query", query)
    """
    # Tool expression will be implemented in the tools package
    # For now, provide a placeholder that can be extended
    from .tools import get_tool_registry
    
    registry = get_tool_registry()
    tool = registry.get_tool(tool_name)
    
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    if operation:
        return tool.execute(operation, *args, **kwargs)
    else:
        return tool.execute(*args, **kwargs)


# Alias for common Context usage
Context = ExecutionContext


def set_global_config(**kwargs):
    """
    Set global configuration options for Fracton runtime.
    
    Available options:
        max_recursion_depth: Global maximum recursion depth
        enable_tracing: Enable/disable bifractal tracing globally
        default_entropy: Default entropy level for new contexts
        cache_size: Size of various internal caches
        
    Example:
        fracton.set_global_config(
            max_recursion_depth=2000,
            enable_tracing=True,
            default_entropy=0.6
        )
    """
    from .core.recursive_engine import get_default_executor
    from .core.entropy_dispatch import get_default_dispatcher
    
    executor = get_default_executor()
    dispatcher = get_default_dispatcher()
    
    if 'max_recursion_depth' in kwargs:
        executor.max_depth = kwargs['max_recursion_depth']
    
    if 'enable_tracing' in kwargs:
        # This would need to be implemented in the executor
        pass
    
    # Add other configuration options as needed


# Utility functions for tests and convenience
def memory_field(capacity: int = 1000, entropy: float = 0.5, **kwargs) -> MemoryField:
    """Create a new memory field instance."""
    return MemoryField(capacity=capacity, entropy=entropy, **kwargs)

def initialize_field(capacity: int = 1000, entropy: float = 0.5, 
                    field_id: str = None, **kwargs) -> MemoryField:
    """Initialize a memory field with specific parameters."""
    return MemoryField(capacity=capacity, entropy=entropy, 
                      field_id=field_id, **kwargs)

def get_runtime_stats() -> dict:
    """
    Get runtime statistics for the Fracton system.
    
    Returns:
        Dictionary with execution statistics, memory usage, and performance metrics
        
    Example:
        stats = fracton.get_runtime_stats()
        print(f"Total recursive calls: {stats['total_calls']}")
        print(f"Average entropy: {stats['avg_entropy']}")
    """
    from .core.recursive_engine import get_default_executor
    from .core.entropy_dispatch import get_default_dispatcher
    from .core.memory_field import get_default_controller
    
    executor = get_default_executor()
    dispatcher = get_default_dispatcher()
    controller = get_default_controller()
    
    exec_stats = executor.get_execution_stats()
    dispatch_stats = dispatcher.get_dispatch_stats()
    
    return {
        'execution': exec_stats.__dict__ if hasattr(exec_stats, '__dict__') else {},
        'dispatch': dispatch_stats,
        'memory_fields': len(controller.get_all_fields()),
        'version': __version__
    }
