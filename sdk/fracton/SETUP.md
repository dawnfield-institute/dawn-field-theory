# Fracton Development Setup

This document provides setup instructions for developing with the Fracton computational modeling language.

## Installation

Since Fracton is currently in development, install it in development mode:

```bash
cd sdk/fracton
pip install -e .
```

## Quick Start

```python
import fracton

# Create a simple recursive function
@fracton.recursive
@fracton.entropy_gate(0.3, 0.8)
def fibonacci(memory, context):
    if context.depth <= 1:
        return 1
    
    a = fracton.recurse(fibonacci, memory, context.deeper(1))
    b = fracton.recurse(fibonacci, memory, context.deeper(2))
    return a + b

# Run with memory field and context
with fracton.memory_field() as field:
    context = fracton.Context(entropy=0.6, depth=10)
    result = fibonacci(field, context)
    print(f"Result: {result}")
```

## Running Examples

```python
# Run all examples
from fracton.examples import run_all_examples
run_all_examples()

# Run specific example
from fracton.examples import run_fibonacci_example
run_fibonacci_example()

# Run GAIA integration example
from fracton.examples.gaia_integration import run_gaia_fracton_integration
run_gaia_fracton_integration()
```

## Development Status

### Completed ✅
- Core recursive execution engine
- Memory field management
- Entropy dispatch system
- Bifractal trace recording
- Basic language decorators and primitives
- Context management
- Comprehensive examples
- GAIA integration demonstration

### In Progress 🚧
- Tool expression framework (placeholder)
- DSL compiler (basic implementation)
- Performance optimizations
- Advanced visualization

### TODO 📋
- Complete tool bindings
- Production-ready DSL compiler
- GPU acceleration
- Distributed memory fields
- Web-based visualization
- Comprehensive test suite

## Architecture Overview

```
fracton/
├── core/                 # Core runtime engine
│   ├── recursive_engine.py
│   ├── memory_field.py
│   ├── entropy_dispatch.py
│   └── bifractal_trace.py
├── lang/                 # Language constructs
│   ├── decorators.py
│   ├── primitives.py
│   ├── context.py
│   └── compiler.py
├── tools/                # Tool expression (placeholder)
├── examples/             # Comprehensive examples
└── __init__.py          # Main API
```

## Key Concepts

### Recursive Functions
Functions decorated with `@fracton.recursive` can be called through the recursive engine:

```python
@fracton.recursive
def my_function(memory, context):
    # Function receives shared memory and execution context
    return fracton.recurse(other_function, memory, context.deeper())
```

### Entropy Gates
Control when functions execute based on entropy levels:

```python
@fracton.entropy_gate(0.5, 0.9)  # Only execute when 0.5 <= entropy <= 0.9
def high_entropy_function(memory, context):
    pass
```

### Memory Fields
Shared memory with entropy tracking:

```python
with fracton.memory_field(capacity=1000, entropy=0.5) as field:
    field.set("key", value)
    data = field.get("key")
    current_entropy = field.get_entropy()
```

### Execution Context
Carries metadata through recursive calls:

```python
context = fracton.Context(
    entropy=0.7,
    depth=0,
    experiment="test_run",
    timestamp=time.time()
)

# Create derived contexts
deeper_context = context.deeper(1)
modified_context = context.with_entropy(0.5)
```

### Bifractal Tracing
Automatic recording of recursive operations:

```python
trace = fracton.BifractalTrace()
# Tracing happens automatically during execution
analysis = trace.analyze_patterns()
visualization = trace.visualize_text()
```

## Integration with GAIA

Fracton is designed as the computational substrate for GAIA:

```python
# GAIA's symbolic processing using Fracton
@fracton.recursive
@fracton.entropy_gate(0.4, 0.9)
def gaia_symbolic_processor(memory, context):
    symbols = memory.get("symbols", [])
    
    if context.entropy > 0.7:
        # High entropy: exploratory processing
        return explore_symbolic_space(symbols, memory, context)
    else:
        # Low entropy: crystallize symbols
        return fracton.crystallize(symbols)
```

## Performance Considerations

- Use entropy gates to control execution flow
- Leverage tail recursion optimization with `@fracton.tail_recursive`
- Monitor memory field size and entropy
- Use memoization for expensive computations
- Profile recursive functions with `@fracton.profile`

## Error Handling

Fracton provides entropy-aware error handling:

```python
try:
    result = fracton.recurse(risky_function, memory, context)
except fracton.EntropyGateError as e:
    # Adjust entropy and retry
    adjusted_context = context.with_entropy(e.suggested_entropy)
    result = fracton.recurse(risky_function, memory, adjusted_context)
```

## Contributing

1. Follow the phased roadmap in ROADMAP.md
2. Maintain comprehensive test coverage
3. Document all public APIs
4. Include examples for new features
5. Consider integration with GAIA use cases

## Research Applications

Fracton is designed for:
- Recursive cognition modeling
- Entropy dynamics simulation
- Complex systems analysis
- Emergent intelligence research
- Bifractal computation patterns
- Field-aware algorithms

## Support

For questions and support:
- Check examples in `fracton/examples/`
- Review architecture docs in `ARCHITECTURE.md`
- Follow development roadmap in `ROADMAP.md`
- See language specification in `SPEC.md`
