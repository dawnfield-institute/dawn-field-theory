# Fracton Language Specification

## Table of Contents
1. [Language Overview](#language-overview)
2. [Core Concepts](#core-concepts)
3. [Syntax Reference](#syntax-reference)
4. [Built-in Functions](#built-in-functions)
5. [Memory Model](#memory-model)
6. [Tool Expression](#tool-expression)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

## Language Overview

Fracton is a recursive, entropy-aware computational modeling language designed for infodynamics research. It treats recursion as the fundamental computational primitive and uses entropy dynamics to control execution flow.

### Key Characteristics
- **Recursive-First**: All computation flows through recursive function calls
- **Entropy-Driven**: Execution controlled by entropy thresholds and field dynamics
- **Bifractal**: Forward and reverse traces maintained for all operations
- **Context-Aware**: Execution context includes entropy, depth, and field state
- **Tool-Expressive**: External systems accessed through contextual expression

## Core Concepts

### 1. Recursive Functions

All Fracton functions are potentially recursive and receive two parameters:
- `memory`: Shared memory field containing persistent state
- `context`: Execution context with entropy, depth, and metadata

```python
@fracton.recursive
def my_function(memory, context):
    # Function body
    return result
```

### 2. Entropy Gates

Functions can specify minimum entropy thresholds for execution:

```python
@fracton.entropy_gate(min_threshold=0.5, max_threshold=0.9)
def entropy_sensitive_function(memory, context):
    # Only executes when 0.5 <= context.entropy <= 0.9
    pass
```

### 3. Context Management

The execution context carries metadata through recursive calls:

```python
class Context:
    entropy: float          # Current entropy level (0.0 - 1.0)
    depth: int             # Recursion depth
    trace_id: str          # Unique identifier for trace
    field_state: dict      # Field-specific metadata
    parent_context: Context # Reference to calling context
```

### 4. Memory Fields

Shared memory structures that maintain state across recursive calls:

```python
with fracton.memory_field(capacity=1000) as field:
    # Operations within this field share memory
    result = recursive_function(field, context)
```

## Syntax Reference

### Function Decorators

#### @fracton.recursive
Marks a function as recursively callable within the Fracton runtime.

```python
@fracton.recursive
def process_data(memory, context):
    return memory.transform(context.entropy)
```

#### @fracton.entropy_gate(min_threshold, max_threshold=1.0)
Sets entropy thresholds for function execution.

```python
@fracton.entropy_gate(0.7)  # Only execute if entropy >= 0.7
def high_entropy_operation(memory, context):
    pass
```

#### @fracton.tool_binding(tool_name)
Marks a function as a tool expression interface.

```python
@fracton.tool_binding("github")
def github_operations(memory, context):
    return fracton.express_tool("github", context)
```

### Control Flow

#### fracton.recurse(function, memory, context)
Initiates a recursive call with proper tracing.

```python
@fracton.recursive
def fibonacci(memory, context):
    if context.depth <= 1:
        return 1
    
    a = fracton.recurse(fibonacci, memory, context.deeper(1))
    b = fracton.recurse(fibonacci, memory, context.deeper(2))
    return a + b
```

#### fracton.crystallize(data, patterns=None)
Crystallizes data into stable structures based on entropy patterns.

```python
result = fracton.crystallize(
    processed_data, 
    patterns=context.discovered_patterns
)
```

#### fracton.branch(condition, if_true, if_false, memory, context)
Entropy-aware conditional branching.

```python
result = fracton.branch(
    context.entropy > 0.5,
    high_entropy_path,
    low_entropy_path,
    memory,
    context
)
```

### Memory Operations

#### memory.get(key, default=None)
Retrieves value from shared memory field.

```python
symbols = memory.get("symbols", [])
```

#### memory.set(key, value)
Stores value in shared memory field.

```python
memory.set("processed_symbols", crystallized_symbols)
```

#### memory.transform(entropy_level)
Applies entropy-based transformation to memory contents.

```python
transformed = memory.transform(context.entropy)
```

#### memory.snapshot()
Creates a snapshot of current memory state for rollback.

```python
snapshot = memory.snapshot()
# ... operations ...
memory.restore(snapshot)  # Rollback if needed
```

### Context Operations

#### context.deeper(steps=1)
Creates a new context with increased depth.

```python
child_context = context.deeper(2)  # Increase depth by 2
```

#### context.with_entropy(new_entropy)
Creates a new context with modified entropy.

```python
high_entropy_context = context.with_entropy(0.9)
```

#### context.with_metadata(**kwargs)
Adds metadata to context.

```python
annotated_context = context.with_metadata(
    operation="pattern_analysis",
    timestamp=time.now()
)
```

## Built-in Functions

### Core Primitives

#### fracton.initialize_field(capacity=1000, entropy=0.5)
Creates a new memory field with specified capacity and initial entropy.

#### fracton.merge_fields(*fields)
Merges multiple memory fields into a single field.

#### fracton.analyze_trace(trace)
Analyzes a bifractal trace for patterns and performance metrics.

#### fracton.visualize_trace(trace, format="graph")
Generates visualization of recursive execution trace.

### Entropy Functions

#### fracton.calculate_entropy(data)
Calculates entropy of given data structure.

#### fracton.entropy_gradient(field, direction="forward")
Calculates entropy gradient across memory field.

#### fracton.regulate_entropy(field, target_entropy)
Adjusts field entropy toward target value.

### Pattern Recognition

#### fracton.detect_patterns(memory, min_confidence=0.7)
Identifies patterns in memory contents.

#### fracton.validate_patterns(patterns, test_data)
Validates discovered patterns against test data.

## Memory Model

### Field Structure

Memory fields are hierarchical structures that maintain:
- **Content**: Actual data being processed
- **Metadata**: Information about data relationships and entropy
- **Traces**: Forward and reverse operation histories
- **Snapshots**: Point-in-time states for rollback

### Memory Isolation

Each memory field provides isolation boundaries:
```python
with fracton.memory_field() as field1:
    with fracton.memory_field() as field2:
        # field1 and field2 are isolated
        # Operations in field2 don't affect field1
        pass
```

### Cross-Field Communication

Fields can communicate through controlled interfaces:
```python
field1.send_message(field2, data, entropy_threshold=0.6)
response = field1.receive_message(timeout=1.0)
```

## Tool Expression

### Tool Registry

Tools are registered with the Fracton runtime:
```python
fracton.register_tool("database", DatabaseConnector())
fracton.register_tool("github", GitHubInterface())
```

### Context-Aware Tool Access

Tools are accessed based on current execution context:
```python
@fracton.tool_binding("database")
def data_operations(memory, context):
    if context.entropy > 0.8:
        return fracton.express_tool("database", "high_entropy_query")
    else:
        return fracton.express_tool("database", "stable_query")
```

### Tool Chaining

Tools can be chained through recursive calls:
```python
@fracton.recursive
def data_pipeline(memory, context):
    # Fetch data
    raw_data = fracton.express_tool("database", "fetch")
    
    # Process recursively
    processed = fracton.recurse(process_data, memory, context)
    
    # Store results
    fracton.express_tool("storage", "save", processed)
```

## Error Handling

### Entropy-Based Error Recovery

Fracton provides entropy-aware error handling:
```python
try:
    result = fracton.recurse(risky_operation, memory, context)
except fracton.EntropyError as e:
    # Adjust entropy and retry
    adjusted_context = context.with_entropy(e.suggested_entropy)
    result = fracton.recurse(risky_operation, memory, adjusted_context)
```

### Trace-Based Debugging

Errors include full bifractal traces for debugging:
```python
try:
    result = complex_recursive_operation(memory, context)
except fracton.RecursionError as e:
    # Analyze the trace to understand the failure
    fracton.visualize_trace(e.trace)
    print(f"Failed at depth {e.failure_depth}")
```

### Graceful Degradation

Functions can specify fallback behaviors:
```python
@fracton.recursive
@fracton.fallback(simple_fallback_function)
def complex_operation(memory, context):
    # Complex logic that might fail
    pass
```

## Performance Considerations

### Tail Recursion Optimization

Fracton optimizes tail-recursive calls:
```python
@fracton.recursive
@fracton.tail_recursive
def countdown(memory, context):
    if context.depth <= 0:
        return "done"
    return fracton.recurse(countdown, memory, context.deeper(-1))
```

### Memory Management

- Use `memory.compact()` to reduce memory footprint
- Set appropriate field capacities to avoid memory exhaustion
- Use `memory.snapshot()` sparingly as it consumes additional memory

### Entropy Calculation

- Entropy calculations can be expensive for large data structures
- Cache entropy values when possible using `context.cache_entropy()`
- Use entropy approximation for performance-critical paths

### Trace Management

- Bifractal traces grow with recursion depth
- Use `fracton.prune_trace()` to remove unnecessary trace elements
- Set trace depth limits for production systems

---

This specification provides the foundation for implementing and using the Fracton language. For implementation details, see the architecture documentation.
