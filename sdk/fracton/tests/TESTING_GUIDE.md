# Fracton Test-Driven Development Guide

This guide explains how to use Fracton's comprehensive test suite for test-driven development, ensuring compliance with Dawn Field Theory's foundational principles.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
python tests/run_tests.py

# Run only foundational theory compliance tests
python tests/run_tests.py --theory-only

# Run with coverage
python tests/run_tests.py --coverage
```

## Test Categories

### 1. Core Component Tests
**Files**: `test_recursive_engine.py`, `test_memory_field.py`, `test_entropy_dispatch.py`, `test_bifractal_trace.py`

**Purpose**: Validate individual core components against their specifications.

```bash
# Run core tests only
pytest tests/test_recursive_engine.py tests/test_memory_field.py -v
```

**Key Validations**:
- Recursive execution with entropy regulation
- Memory field dynamics and entropy tracking
- Context-aware function dispatch
- Bifractal trace recording and analysis

### 2. Foundational Theory Compliance
**Files**: `test_sec_compliance.py`, `test_med_compliance.py`

**Purpose**: Ensure Fracton correctly implements Dawn Field Theory principles.

```bash
# Test SEC compliance
pytest tests/test_sec_compliance.py -m sec_compliance -v

# Test MED compliance  
pytest tests/test_med_compliance.py -m med_compliance -v
```

**SEC (Symbolic Entropy Collapse)**: `∇_micro → Ψ_macro`
- Entropy collapse dynamics
- Micro-to-macro emergence patterns
- Symbolic crystallization processes
- Phase transition validation

**MED (Macro Emergence Dynamics)**: `∇_macro → Ψ_micro`
- Macro-to-micro constraint propagation
- Hierarchical field dynamics
- Boundary condition inheritance
- Emergence probability modulation

### 3. Integration Tests
**Files**: `test_api.py`, `test_examples.py`, `test_field_dynamics.py`

**Purpose**: Validate component interaction and end-to-end workflows.

```bash
# Run integration tests
pytest tests/ -m integration -v
```

### 4. Performance Tests
**Files**: `test_performance.py`, `test_memory_limits.py`, `test_concurrency.py`

**Purpose**: Ensure Fracton meets performance requirements.

```bash
# Run performance benchmarks
pytest tests/ -m performance -v
```

## Test-Driven Development Workflow

### Step 1: Write Theory-Compliant Tests First

Before implementing new functionality, write tests that validate compliance with foundational theory:

```python
# Example: Adding new SEC feature
@sec_compliance
def test_new_sec_feature(self, balanced_memory_field):
    """Test new SEC feature compliance."""
    
    # Define expected SEC behavior
    initial_entropy = 0.8
    expected_final_entropy = 0.3
    
    # Test implementation (this will fail initially)
    result = new_sec_feature(balanced_memory_field, initial_entropy)
    
    # Validate SEC compliance
    assert_sec_compliance(initial_entropy, result.final_entropy, "collapse")
```

### Step 2: Implement to Pass Theory Tests

Implement functionality to satisfy foundational theory requirements:

```python
def new_sec_feature(memory_field, initial_entropy):
    """Implement new SEC feature."""
    
    # Implementation must satisfy SEC dynamics
    # ∇_micro → Ψ_macro pattern
    
    context = Context(entropy=initial_entropy, depth=0)
    
    # Ensure entropy decreases (SEC collapse)
    while context.entropy > 0.3:
        # Micro operations leading to macro crystallization
        context = context.with_entropy(context.entropy * 0.9)
    
    return SecResult(final_entropy=context.entropy)
```

### Step 3: Validate Integration

Run integration tests to ensure new feature works with existing components:

```bash
pytest tests/test_api.py -v
```

### Step 4: Performance Validation

Ensure new feature meets performance requirements:

```bash
pytest tests/test_performance.py -v
```

## Custom Test Assertions

Fracton provides specialized assertions for foundational theory compliance:

### SEC Compliance

```python
from tests.conftest import assert_sec_compliance

# Validate entropy collapse
assert_sec_compliance(entropy_before, entropy_after, "collapse")

# Validate entropy exploration
assert_sec_compliance(entropy_before, entropy_after, "exploration")
```

### MED Compliance

```python
from tests.conftest import assert_med_compliance

# Validate macro-micro scale relationship
macro_state = {"scale": 100, "entropy": 0.6}
micro_state = {"scale": 10, "entropy": 0.5}
assert_med_compliance(macro_state, micro_state)
```

### Bifractal Symmetry

```python
from tests.conftest import assert_bifractal_symmetry

# Validate temporal symmetry
assert_bifractal_symmetry(trace, operation_id)
```

### Recursive Balance

```python
from tests.conftest import assert_recursive_balance

# Validate field balance maintenance
assert_recursive_balance(memory_field, operation_count)
```

## Test Data Generators

Use provided generators for consistent test data:

```python
from tests.conftest import generate_entropy_sequence, generate_field_hierarchy

# Generate SEC collapse sequence
entropy_sequence = generate_entropy_sequence(10, "collapse")

# Generate MED hierarchy
field_hierarchy = generate_field_hierarchy(4)
```

## Continuous Validation

### Pre-commit Hooks

Set up pre-commit hooks to run theory compliance tests:

```bash
# .git/hooks/pre-commit
#!/bin/bash
python tests/run_tests.py --theory-only --quiet
if [ $? -ne 0 ]; then
    echo "❌ Foundational theory compliance failed!"
    exit 1
fi
```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
# .github/workflows/fracton-tests.yml
name: Fracton Test Suite

on: [push, pull_request]

jobs:
  theory-compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Theory Compliance Tests
        run: python tests/run_tests.py --theory-only
      
  full-test-suite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Full Test Suite
        run: python tests/run_tests.py --coverage
```

## Debugging Failed Tests

### Theory Compliance Failures

When foundational theory tests fail:

1. **Check Entropy Evolution**: Ensure entropy changes follow SEC/MED patterns
2. **Validate Scale Relationships**: Verify macro/micro scale hierarchies
3. **Review Temporal Dynamics**: Check bifractal symmetry requirements

```python
# Debug entropy evolution
print(f"Entropy history: {entropy_history}")
print(f"Expected trend: {'decreasing' if sec_test else 'constrained'}")

# Debug scale relationships
print(f"Macro scale: {macro_scale}, Micro scale: {micro_scale}")
print(f"Scale ratio: {micro_scale / macro_scale}")
```

### Performance Test Failures

When performance tests fail:

1. **Profile Execution**: Use performance markers to identify bottlenecks
2. **Check Scaling**: Verify algorithmic complexity matches expectations
3. **Memory Usage**: Monitor memory field efficiency

```python
# Performance debugging
import cProfile
cProfile.run('slow_function()', 'profile_output')

# Memory debugging
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Implementation
    pass
```

## Best Practices

### 1. Theory-First Testing

Always write foundational theory compliance tests before implementation:

```python
# ✅ Good: Theory compliance test first
def test_new_feature_sec_compliance(self):
    # Test SEC pattern compliance
    pass

def implement_new_feature(self):
    # Implementation follows
    pass

# ❌ Bad: Implementation without theory validation
def implement_new_feature_without_tests(self):
    # Implementation without validation
    pass
```

### 2. Isolation and Integration

Test components in isolation, then integration:

```python
# Unit test: isolated component
def test_memory_field_isolation(self):
    field = MemoryField(capacity=100)
    # Test field alone
    
# Integration test: component interaction
def test_memory_field_with_executor(self):
    field = MemoryField(capacity=100)
    executor = RecursiveExecutor()
    # Test interaction
```

### 3. Comprehensive Coverage

Aim for high test coverage with meaningful validation:

```bash
# Check coverage
pytest tests/ --cov=fracton --cov-report=html
open htmlcov/index.html
```

### 4. Performance Regression Prevention

Include performance tests in regular test runs:

```python
@performance
def test_recursive_execution_scaling(self):
    """Ensure recursive execution scales appropriately."""
    
    for depth in [10, 20, 50]:
        start_time = time.time()
        # Execute at depth
        duration = time.time() - start_time
        
        # Verify scaling
        assert duration < depth * 0.01  # Linear scaling expectation
```

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure Fracton is installed in development mode
   ```bash
   cd sdk/fracton
   pip install -e .
   ```

2. **Theory Test Failures**: Review foundational requirements in `foundational/` directory

3. **Performance Issues**: Check system resources and test configuration

### Getting Help

1. Review foundational theory documents in `foundational/docs/`
2. Check existing test examples in `tests/`
3. Examine ARCHITECTURE.md and SPEC.md for requirements
4. Run `python tests/run_tests.py --help` for options

## Extending the Test Suite

### Adding New Test Categories

1. Create new test file: `test_new_category.py`
2. Add markers in `pytest.ini`
3. Update `run_tests.py` to include new category
4. Document in this guide

### Custom Test Fixtures

Add domain-specific fixtures to `conftest.py`:

```python
@pytest.fixture
def custom_field_configuration():
    """Custom field for specific test scenarios."""
    return MemoryField(
        capacity=500,
        entropy=0.6,
        custom_property="test_value"
    )
```

---

**Remember**: The goal is not just passing tests, but ensuring Fracton correctly implements the theoretical framework that makes it a true computational substrate for consciousness research and recursive intelligence modeling. 🧬✨
