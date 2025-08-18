# Navier-Stokes Symbolic Engine - Test Suite

Comprehensive test suite for the Navier-Stokes Symbolic Engine with organized structure.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_imports.py      # Import and module availability
│   ├── test_patterns.py     # Pattern generation and trees
│   ├── test_entropy.py      # Entropy hashing and navigation
│   ├── test_thermodynamics.py # Thermodynamic validation
│   └── test_engine.py       # Main engine interface
├── integration/             # Integration and workflow tests
│   └── test_workflow.py     # End-to-end testing
├── performance/             # Performance and benchmarking
│   └── test_benchmarks.py   # Performance measurements
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Running Tests

### Via Unified Framework
```bash
# Run all tests
python unified_experimental_framework.py --unit-tests

# Run specific test types
python unified_experimental_framework.py --unit-tests --organized imports
python unified_experimental_framework.py --unit-tests --organized unit
python unified_experimental_framework.py --unit-tests --organized integration
```

### Direct pytest
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run specific test files
pytest tests/unit/test_imports.py
pytest tests/unit/test_engine.py

# Run with markers
pytest -m performance
pytest -m stress
```

## Test Categories

### Unit Tests
- **test_imports.py**: Validates all modules can be imported
- **test_patterns.py**: Tests pattern generation (Poiseuille, Couette, turbulent)
- **test_entropy.py**: Tests entropy hashing and navigation paths
- **test_thermodynamics.py**: Tests thermodynamic validation (Landauer bounds)
- **test_engine.py**: Tests main engine interface and simulation execution

### Integration Tests
- **test_workflow.py**: End-to-end workflow testing with component integration

### Performance Tests
- **test_benchmarks.py**: Performance benchmarks and stress testing

## Test Philosophy

- **Comprehensive Coverage**: All components tested systematically
- **Fast Feedback**: Quick unit tests for rapid development cycles
- **Realistic Integration**: End-to-end workflows with real scenarios
- **Performance Aware**: Quantified benchmarks and stress testing
- **Well Documented**: Clear test structure and comprehensive documentation
