# Domain Adapters Architecture

## Overview

Domain Adapters are the bridge between the unified framework and domain-specific physics simulations. They implement the `DomainAdapter` protocol to provide a standardized interface while preserving domain-specific logic.

## DomainAdapter Protocol

```python
from typing import Protocol, Dict, List, Any
from ..domain.emergence_signature import EmergenceSignature

class DomainAdapter(Protocol):
    """Interface that all domain implementations must satisfy."""
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """Extract emergence patterns from domain-specific results."""
        ...
    
    def validate_constraints(self, results: Dict[str, Any]) -> List[str]:
        """Validate domain-specific physical/mathematical constraints."""
        ...
    
    @property
    def domain_name(self) -> str:
        """Return the domain identifier."""
        ...
```

## Supported Domains

### Gravity Domain Adapter

**Purpose**: Extract orbital dynamics and gravitational field emergence patterns

**Key Patterns**:
- Orbital stability signatures
- Energy conservation patterns
- Angular momentum conservation
- Gravitational field coherence

**Constraints Validated**:
- Conservation of energy (< 5% deviation)
- Conservation of angular momentum (< 1% deviation)
- Orbital stability bounds
- General relativity constraints

```python
class GravityDomainAdapter:
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        signatures = []
        
        for run_data in domain_results.get('runs', []):
            for field_size, data in run_data.items():
                if 'orbital_stability' in data:
                    signatures.append(EmergenceSignature(
                        domain='gravity',
                        pattern_type='orbital_dynamics',
                        features=[
                            data['orbital_stability'],
                            data['energy_conservation'],
                            data['angular_momentum_conservation'],
                            1.0 - data['orbital_eccentricity']  # Circular orbit preference
                        ],
                        confidence=data['orbital_stability'],
                        emergence_strength=self._calculate_emergence_strength(data),
                        metadata={
                            'field_size': field_size,
                            'trajectory_points': data.get('trajectory_points', 0),
                            'mean_radius': data.get('mean_orbital_radius_au', 0)
                        }
                    ))
        
        return signatures
```

### MED (Macro Emergence Dynamics) Domain Adapter

**Purpose**: Extract complex system emergence patterns and parameter optimization results

**Key Patterns**:
- Complexity bound satisfaction
- Parameter optimization convergence
- Scale-adaptive behavior
- Information cascade patterns

**Constraints Validated**:
- Complexity bounds satisfaction (must be >= 0.95)
- Parameter convergence criteria
- Physical scaling laws
- Information conservation

```python
class MEDDomainAdapter:
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        signatures = []
        
        # Extract complexity bound patterns
        bound_satisfaction = domain_results.get('complexity_bound_satisfaction', 0.0)
        best_score = domain_results.get('best_score', 0.0)
        
        signatures.append(EmergenceSignature(
            domain='med',
            pattern_type='complexity_emergence',
            features=[
                bound_satisfaction,
                best_score,
                self._calculate_parameter_diversity(domain_results),
                self._calculate_convergence_stability(domain_results)
            ],
            confidence=bound_satisfaction,
            emergence_strength=best_score,
            metadata={
                'best_parameters': domain_results.get('best_parameters', {}),
                'parameter_count': len(domain_results.get('parameter_analysis', {}))
            }
        ))
        
        return signatures
```

### Navier-Stokes Domain Adapter

**Purpose**: Extract turbulence and fluid dynamics emergence patterns

**Key Patterns**:
- Turbulence cascade signatures
- Vorticity concentration patterns
- Reynolds number scaling
- Energy dissipation patterns

**Constraints Validated**:
- Reynolds number regime consistency
- Energy cascade conservation
- Turbulence intensity bounds
- Computational stability

```python
class NavierDomainAdapter:
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        signatures = []
        
        for run_data in domain_results.get('runs', []):
            reynolds = run_data.get('reynolds_number', 0)
            accuracy = run_data.get('turbulence_detection_accuracy', 0)
            
            signatures.append(EmergenceSignature(
                domain='navier',
                pattern_type='turbulence_emergence',
                features=[
                    accuracy,
                    self._normalize_reynolds(reynolds),
                    run_data.get('vorticity_strength', 0),
                    run_data.get('energy_cascade_efficiency', 0)
                ],
                confidence=accuracy,
                emergence_strength=self._calculate_turbulence_strength(run_data),
                metadata={
                    'reynolds_number': reynolds,
                    'grid_resolution': run_data.get('grid_size', 0),
                    'time_steps': run_data.get('time_steps', 0)
                }
            ))
        
        return signatures
```

### TinyCIMM Domain Adapter

**Purpose**: Extract information architecture emergence patterns

**Key Patterns**:
- Entropy navigation effectiveness
- Information architecture emergence
- Compression efficiency patterns
- Memory organization signatures

**Constraints Validated**:
- Information conservation laws
- Entropy bounds
- Compression ratio limits
- Architecture coherence

```python
class TinyCIMMDomainAdapter:
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        signatures = []
        
        for run_data in domain_results.get('runs', []):
            architecture = run_data.get('architecture', 'unknown')
            score = run_data.get('score', 0)
            
            signatures.append(EmergenceSignature(
                domain='tinycimm',
                pattern_type='information_architecture',
                features=[
                    score,
                    run_data.get('entropy_navigation', 0),
                    run_data.get('compression_efficiency', 0),
                    self._calculate_architecture_complexity(architecture)
                ],
                confidence=min(score, 1.0),
                emergence_strength=score,
                metadata={
                    'architecture': architecture,
                    'field_size': run_data.get('field_size', 0),
                    'parameter_variant': run_data.get('navier_param', 0)
                }
            ))
        
        return signatures
```

### Hodge Domain Adapter

**Purpose**: Extract symbolic entropy collapse and topological patterns

**Key Patterns**:
- Symbolic cycle detection
- Entropy collapse signatures
- Prime modulation effects
- Topological invariant preservation

**Constraints Validated**:
- Topological consistency
- Symbolic entropy bounds
- Prime number theory constraints
- Cycle detection accuracy

```python
class HodgeDomainAdapter:
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        signatures = []
        
        total_cycles = domain_results.get('total_cycles_detected', 0)
        detection_rate = domain_results.get('cycle_detection_rate', 0)
        
        for run_data in domain_results.get('runs', []):
            field_size = run_data.get('field_size', 0)
            prime = run_data.get('prime', 0)
            
            signatures.append(EmergenceSignature(
                domain='hodge',
                pattern_type='symbolic_entropy_collapse',
                features=[
                    detection_rate,
                    self._normalize_field_size(field_size),
                    self._prime_modulation_strength(prime),
                    run_data.get('entropy_collapse_rate', 0)
                ],
                confidence=detection_rate,
                emergence_strength=total_cycles / max(len(domain_results.get('runs', [])), 1),
                metadata={
                    'field_size': field_size,
                    'prime_modulus': prime,
                    'cycles_detected': run_data.get('cycles_detected', 0)
                }
            ))
        
        return signatures
```

## Adapter Implementation Guidelines

### 1. Pattern Extraction Best Practices

```python
def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
    """
    Best practices for pattern extraction:
    1. Always validate input data structure
    2. Handle missing data gracefully
    3. Normalize features to [0, 1] range
    4. Calculate meaningful confidence scores
    5. Include rich metadata for debugging
    """
    signatures = []
    
    # Validate input structure
    if not isinstance(domain_results, dict):
        self.logger.warning(f"Invalid domain_results type: {type(domain_results)}")
        return signatures
    
    # Handle missing runs gracefully
    runs = domain_results.get('runs', [])
    if not runs:
        self.logger.info(f"No runs found in {self.domain_name} results")
        return signatures
    
    for run_data in runs:
        try:
            signature = self._extract_single_pattern(run_data)
            if signature and signature.confidence >= self.min_confidence:
                signatures.append(signature)
        except Exception as e:
            self.logger.error(f"Failed to extract pattern from {self.domain_name}: {e}")
            continue
    
    return signatures
```

### 2. Constraint Validation Patterns

```python
def validate_constraints(self, results: Dict[str, Any]) -> List[str]:
    """
    Constraint validation should:
    1. Check domain-specific physical laws
    2. Validate computational stability
    3. Verify data quality bounds
    4. Return descriptive error messages
    """
    violations = []
    
    # Example: Energy conservation check
    energy_conservation = results.get('energy_conservation', 1.0)
    if abs(1.0 - energy_conservation) > 0.05:  # 5% tolerance
        violations.append(
            f"Energy conservation violation: {energy_conservation:.3f} "
            f"(expected ~1.0, tolerance ±0.05)"
        )
    
    # Example: Data quality check
    if results.get('trajectory_points', 0) < 100:
        violations.append(
            f"Insufficient trajectory points: {results.get('trajectory_points', 0)} "
            f"(minimum required: 100)"
        )
    
    return violations
```

### 3. Feature Normalization

```python
def _normalize_features(self, raw_features: List[float]) -> List[float]:
    """
    Normalize features to [0, 1] range for consistent correlation analysis.
    """
    normalized = []
    
    for feature in raw_features:
        # Handle different feature types
        if isinstance(feature, (int, float)):
            # Clamp to [0, 1] range
            normalized.append(max(0.0, min(1.0, float(feature))))
        else:
            self.logger.warning(f"Non-numeric feature encountered: {feature}")
            normalized.append(0.0)
    
    return normalized
```

### 4. Error Handling

```python
class DomainAdapterBase:
    """Base class providing common functionality for domain adapters."""
    
    def __init__(self, test_runner: TestRunner):
        self.test_runner = test_runner
        self.logger = LoggingConfig.setup_logger(f"{__name__}.{self.domain_name}")
        self.min_confidence = 0.1  # Minimum confidence threshold
    
    def safe_extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """Wrapper for safe pattern extraction with error handling."""
        try:
            return self.extract_patterns(domain_results)
        except Exception as e:
            self.logger.error(f"Pattern extraction failed for {self.domain_name}: {e}")
            return []
    
    def safe_validate_constraints(self, results: Dict[str, Any]) -> List[str]:
        """Wrapper for safe constraint validation with error handling."""
        try:
            return self.validate_constraints(results)
        except Exception as e:
            self.logger.error(f"Constraint validation failed for {self.domain_name}: {e}")
            return [f"Constraint validation error: {str(e)}"]
```

## Testing Domain Adapters

### Unit Testing

```python
class TestGravityDomainAdapter:
    def test_extract_patterns_valid_data(self):
        adapter = GravityDomainAdapter(mock_test_runner)
        
        test_data = {
            'runs': [{
                'field_size_32': {
                    'orbital_stability': 0.926,
                    'energy_conservation': 0.848,
                    'angular_momentum_conservation': 0.999,
                    'orbital_eccentricity': 0.08
                }
            }]
        }
        
        signatures = adapter.extract_patterns(test_data)
        
        assert len(signatures) == 1
        assert signatures[0].domain == 'gravity'
        assert signatures[0].pattern_type == 'orbital_dynamics'
        assert signatures[0].confidence == 0.926
    
    def test_validate_constraints_energy_violation(self):
        adapter = GravityDomainAdapter(mock_test_runner)
        
        test_data = {
            'energy_conservation': 0.5  # Too low
        }
        
        violations = adapter.validate_constraints(test_data)
        
        assert len(violations) > 0
        assert 'Energy conservation violation' in violations[0]
```

### Integration Testing

```python
class TestDomainAdapterIntegration:
    def test_full_workflow_with_real_data(self):
        """Test adapter with real domain simulation data."""
        adapter = GravityDomainAdapter(real_test_runner)
        
        # Run actual gravity simulation
        domain_results = adapter.test_runner.run_domain_tests({
            'field_sizes': [32],
            'runs': 1
        })
        
        # Extract patterns
        signatures = adapter.extract_patterns(domain_results)
        
        # Validate constraints
        violations = adapter.validate_constraints(domain_results)
        
        # Verify results
        assert len(signatures) > 0
        assert all(sig.confidence > 0 for sig in signatures)
        assert len(violations) == 0  # Should pass constraints
```

This domain adapter architecture provides a clean, extensible way to integrate different physics domains while maintaining consistent interfaces and error handling patterns.
