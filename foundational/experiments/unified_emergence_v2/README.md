# Unified Emergence Framework v2

A clean, maintainable framework for analyzing emergence patterns across multiple domains using Clean Architecture principles.

## Overview

The Unified Emergence Framework v2 is a complete reimplementation of the emergence validation system with:

- **Clean Architecture**: Clear separation of concerns with domain, application, and infrastructure layers
- **Protocol-Based Design**: Dependency injection and interface-driven development
- **Comprehensive Testing**: Unit, integration, and performance tests
- **Domain Adapters**: Pluggable adapters for gravity, MED, Navier-Stokes, TinyCIMM, and Hodge domains
- **Unified Data Flow**: Consistent data structures and processing pipeline
- **Production Ready**: Error handling, logging, monitoring, and scalability

## Quick Start

### Installation

1. Navigate to the framework directory:
```bash
cd foundational/experiments/unified_emergence_v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run a quick validation test:
```bash
python run_validation.py --quick-test
```

Run validation on specific domains:
```bash
python run_validation.py --domains gravity med --field-sizes 32 64 --runs 2
```

Run with custom configuration:
```bash
python run_validation.py --domains gravity med navier --runs 3 --parallel --timeout 600 --output-dir results
```

### Programmatic Usage

```python
from unified_emergence_v2 import UnifiedEmergenceFramework

# Initialize framework
framework = UnifiedEmergenceFramework()

# Configure validation
config = {
    'domains': ['gravity', 'med', 'navier'],
    'field_sizes': [32, 64],
    'runs_per_domain': 2,
    'parallel_execution': True,
    'timeout_seconds': 300
}

# Run validation
results = framework.run_phase1_validation(config)

# Check results
if results.success:
    print(f"Overall score: {results.metrics.get_overall_score():.3f}")
    print(f"Phase 1 ready: {results.is_phase1_ready()}")
else:
    print("Validation failed:", results.error_messages)
```

## Architecture

The framework follows Clean Architecture principles with clear separation of concerns:

- **Domain Layer**: Core business logic (EmergenceSignature, ValidationMetrics)
- **Application Layer**: Use cases and orchestration (EmergenceOrchestrator)
- **Infrastructure Layer**: External systems and technical concerns (TestRunner, ResultsRepository)

See [Architecture Documentation](docs/architecture/) for detailed design decisions.

## Key Features

### 🎯 **From v1 Spike Learnings**
- ✅ **Unified Logging**: Consistent logging format across all modules
- ✅ **Standardized Data Flow**: Single EmergenceResults container for all outputs
- ✅ **Protocol-Based Interfaces**: Easy to extend with new domains
- ✅ **Dependency Injection**: Testable and maintainable code
- ✅ **Configuration-Driven**: No hardcoded dependencies

### 🚀 **Production Ready**
- **Comprehensive Error Handling**: Graceful failure modes
- **Performance Optimized**: Efficient cross-domain correlation algorithms
- **Extensible**: Easy to add new physics domains
- **Well Tested**: Unit and integration tests for all components

## Directory Structure

```
unified_emergence_v2/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── setup.py                 # Package configuration
├── docs/                    # Documentation
│   ├── architecture/        # Architecture documentation
│   ├── api/                # API documentation
│   └── examples/           # Usage examples
├── src/                    # Source code
│   ├── domain/             # Domain layer (core business logic)
│   ├── application/        # Application layer (use cases)
│   ├── infrastructure/     # Infrastructure layer (external systems)
│   └── adapters/          # Domain-specific adapters
├── tests/                  # Comprehensive test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data and fixtures
└── examples/              # Working examples and demos
```

## Supported Domains

- **Gravity Dynamics**: Orbital mechanics and gravitational field emergence
- **MED (Macro Emergence Dynamics)**: Complex system emergence patterns
- **Navier-Stokes**: Turbulence and fluid dynamics emergence
- **TinyCIMM**: Information architecture emergence
- **Hodge Mapping**: Symbolic entropy collapse patterns
- **SEC Fields**: Symbolic entropy collapse field dynamics

## Development Status

- **Phase 1**: Core Infrastructure ✅ (Completed)
- **Phase 2**: Domain Adapters 🚧 (In Progress)
- **Phase 3**: Integration & Testing 📋 (Planned)
- **Phase 4**: Migration from v1 📋 (Planned)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd unified_emergence_v2

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Review the [Architecture Documentation](docs/architecture/)
2. Check the [API Documentation](docs/api/)
3. Follow the established patterns in existing adapters
4. Add comprehensive tests for new features
5. Update documentation for any API changes

## Migration from v1

See [Migration Guide](docs/migration_guide.md) for detailed instructions on migrating from the v1 spike implementation.

## Performance

The v2 architecture is designed for performance:

- **Lazy Loading**: Domain adapters only load when needed
- **Parallel Processing**: Cross-domain correlations calculated in parallel
- **Memory Efficient**: Streaming processing for large datasets
- **Caching**: Intelligent caching of computed patterns

## License

[Add license information]

## Support

For questions or issues:
- Review the [Architecture Documentation](docs/architecture/)
- Check existing [Issues](issues/)
- Create a new issue with detailed information
