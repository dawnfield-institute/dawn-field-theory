# Fracton Test Suite

This directory contains comprehensive test coverage for the Fracton computational modeling language, designed around Dawn Field Theory's foundational principles:

## Test Organization

### Core Engine Tests
- `test_recursive_engine.py` - Recursive execution, entropy gates, stack management
- `test_memory_field.py` - Field dynamics, entropy tracking, MED compliance  
- `test_entropy_dispatch.py` - Context-aware routing, SEC patterns
- `test_bifractal_trace.py` - Operation recording, pattern analysis

### Language Tests
- `test_decorators.py` - @fracton.recursive, @fracton.entropy_gate behavior
- `test_primitives.py` - fracton.recurse(), context primitives
- `test_context.py` - Execution context, depth tracking, entropy evolution
- `test_compiler.py` - DSL compilation (when implemented)

### Integration Tests
- `test_api.py` - Main Fracton API surface
- `test_examples.py` - Example algorithms validation
- `test_field_dynamics.py` - Multi-component field behavior
- `test_sec_compliance.py` - Symbolic Entropy Collapse validation
- `test_med_compliance.py` - Macro Emergence Dynamics validation

### Performance Tests
- `test_performance.py` - Benchmarks and scaling
- `test_memory_limits.py` - Resource management
- `test_concurrency.py` - Thread safety (future)

## Theoretical Compliance

Tests validate compliance with:
- **SEC (Symbolic Entropy Collapse)**: `∇_micro → Ψ_macro` dynamics
- **MED (Macro Emergence Dynamics)**: `∇_macro → Ψ_micro` patterns  
- **Bifractal Time**: Backward ancestry / forward emergence
- **Recursive Balance Fields**: Dynamic potential adaptation
- **Entropy-Regulated Modeling**: Landauer cost awareness

## Test-Driven Development

Run tests continuously during development:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_recursive_engine.py -v

# Run tests with coverage
python -m pytest tests/ --cov=fracton --cov-report=html

# Run only foundational theory compliance tests
python -m pytest tests/test_sec_compliance.py tests/test_med_compliance.py -v
```

## Test Categories

### Unit Tests (95% coverage target)
- Individual class/function behavior
- Boundary conditions and edge cases
- Error handling and validation

### Integration Tests
- Component interaction
- End-to-end workflows
- API surface validation

### Theoretical Compliance Tests
- SEC dynamics verification
- MED pattern detection
- Bifractal ancestry tracking
- Entropy regulation validation

### Performance Tests
- Recursive depth scaling
- Memory field efficiency
- Entropy computation overhead
- Trace recording impact

## Foundation Theory Validation

Each test module includes theoretical validation:

```python
def test_sec_dynamics():
    """Validate ∇_micro → Ψ_macro collapse patterns"""
    
def test_med_emergence():
    """Validate ∇_macro → Ψ_micro emergence patterns"""
    
def test_bifractal_time():
    """Validate backward/forward temporal symmetry"""
    
def test_recursive_balance():
    """Validate dynamic potential adaptation"""
```

## Continuous Validation

Tests ensure Fracton remains aligned with foundational principles as it evolves, providing confidence that the computational substrate correctly implements the theoretical framework.
