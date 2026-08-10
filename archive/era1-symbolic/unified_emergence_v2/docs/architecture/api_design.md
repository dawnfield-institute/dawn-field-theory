# API Design and Interfaces

## Overview

The Unified Emergence Framework v2 provides a clean, consistent API that abstracts the complexity of cross-domain emergence validation while providing flexibility for advanced use cases.

## Public API

### Primary Interface

```python
from unified_emergence_v2 import UnifiedEmergenceFramework

class UnifiedEmergenceFramework:
    """Main framework entry point with clean, simple interface."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize framework with optional configuration file."""
        
    def run_phase1_validation(self, config: Optional[Dict[str, Any]] = None) -> EmergenceResults:
        """
        Run Phase 1 emergence validation.
        
        Args:
            config: Optional configuration dictionary. If None, uses defaults.
            
        Returns:
            EmergenceResults containing all validation results and metrics.
            
        Example:
            framework = UnifiedEmergenceFramework()
            results = framework.run_phase1_validation({
                'domains': ['gravity', 'med', 'navier'],
                'field_sizes': [32, 64],
                'runs': 2
            })
        """
        
    def run_custom_validation(self, config: ValidationConfig) -> EmergenceResults:
        """
        Run custom validation with full configuration control.
        
        Args:
            config: Complete ValidationConfig object.
            
        Returns:
            EmergenceResults containing all validation results and metrics.
        """
        
    def analyze_existing_results(self, session_id: str) -> EmergenceResults:
        """
        Load and analyze previously saved results.
        
        Args:
            session_id: Session identifier from previous run.
            
        Returns:
            Loaded EmergenceResults.
            
        Raises:
            ResultsNotFoundError: If session_id not found.
        """
        
    def list_available_domains(self) -> List[str]:
        """Return list of available domain adapters."""
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for validation runs."""
        
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration without running tests.
        
        Args:
            config: Configuration dictionary to validate.
            
        Returns:
            List of validation errors (empty if valid).
        """
```

### Quick Start Examples

```python
# 1. Simplest usage - run with all defaults
framework = UnifiedEmergenceFramework()
results = framework.run_phase1_validation()

# 2. Custom domain selection
results = framework.run_phase1_validation({
    'domains': ['gravity', 'med'],
    'field_sizes': [32],
    'runs': 1
})

# 3. High-performance configuration
results = framework.run_phase1_validation({
    'domains': ['gravity', 'med', 'navier', 'tinycimm', 'hodge'],
    'field_sizes': [32, 64, 128],
    'runs': 3,
    'parallel_execution': True,
    'max_workers': 8
})

# 4. Analysis-focused run
results = framework.run_phase1_validation({
    'save_intermediate_results': True,
    'export_visualizations': True,
    'output_directory': 'detailed_analysis'
})
```

## Configuration API

### ValidationConfig Class

```python
@dataclass
class ValidationConfig:
    """Complete configuration for emergence validation runs."""
    
    # Required fields
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Test execution parameters
    domains: List[str] = field(default_factory=lambda: ['gravity', 'med', 'navier', 'tinycimm', 'hodge'])
    field_sizes: List[int] = field(default_factory=lambda: [32, 64])
    runs_per_domain: int = 1
    param_sweep: bool = False
    
    # Validation thresholds
    sec_classification_threshold: float = 0.9
    pattern_assembly_threshold: float = 0.8
    emergence_consistency_threshold: float = 0.85
    correlation_threshold: float = 0.7
    min_pattern_confidence: float = 0.1
    
    # Performance parameters
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    memory_limit_mb: int = 2048
    
    # Output configuration
    save_intermediate_results: bool = True
    export_visualizations: bool = False
    output_directory: str = "results"
    output_format: str = "json"  # json, pickle, hdf5
    compression: bool = True
    
    # Advanced configuration
    enable_enhanced_patterns: bool = True  # Use macro_emergence_dynamics
    debug_mode: bool = False
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Validate domains
        valid_domains = ['gravity', 'med', 'navier', 'tinycimm', 'hodge', 'sec']
        for domain in self.domains:
            if domain not in valid_domains:
                errors.append(f"Invalid domain: {domain}. Valid options: {valid_domains}")
        
        # Validate thresholds
        for threshold_name, threshold_value in [
            ('sec_classification_threshold', self.sec_classification_threshold),
            ('pattern_assembly_threshold', self.pattern_assembly_threshold),
            ('emergence_consistency_threshold', self.emergence_consistency_threshold),
            ('correlation_threshold', self.correlation_threshold),
            ('min_pattern_confidence', self.min_pattern_confidence)
        ]:
            if not 0.0 <= threshold_value <= 1.0:
                errors.append(f"{threshold_name} must be between 0.0 and 1.0, got {threshold_value}")
        
        # Validate field sizes
        for size in self.field_sizes:
            if size < 8 or size > 512 or (size & (size - 1)) != 0:
                errors.append(f"Field size {size} must be power of 2 between 8 and 512")
        
        return errors
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ValidationConfig':
        """Load configuration from file (JSON or YAML)."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                data = yaml.safe_load(f)
            else:
                import json
                data = json.load(f)
        
        return cls(**data)
    
    def to_file(self, config_path: str):
        """Save configuration to file."""
        data = asdict(self)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                yaml.safe_dump(data, f, indent=2)
            else:
                import json
                json.dump(data, f, indent=2)
```

### Configuration File Examples

```yaml
# config.yaml - Basic configuration
domains:
  - gravity
  - med
  - navier
field_sizes:
  - 32
  - 64
runs_per_domain: 2
parallel_execution: true
max_workers: 4

# Thresholds
sec_classification_threshold: 0.9
pattern_assembly_threshold: 0.8
emergence_consistency_threshold: 0.85

# Output
output_directory: "production_results"
save_intermediate_results: true
export_visualizations: true
```

```json
{
  "domains": ["gravity", "med", "navier", "tinycimm", "hodge"],
  "field_sizes": [16, 32, 64, 128],
  "runs_per_domain": 5,
  "param_sweep": true,
  "parallel_execution": true,
  "max_workers": 8,
  "timeout_seconds": 600,
  "sec_classification_threshold": 0.95,
  "pattern_assembly_threshold": 0.85,
  "emergence_consistency_threshold": 0.9,
  "output_directory": "high_performance_run",
  "enable_enhanced_patterns": true,
  "debug_mode": false,
  "log_level": "INFO"
}
```

## Results API

### EmergenceResults Interface

```python
class EmergenceResults:
    """Results container with rich access methods."""
    
    def is_phase1_complete(self) -> bool:
        """Check if Phase 1 validation criteria are met."""
        return self.metrics.is_phase1_complete(self.configuration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results for quick analysis."""
        return self.to_summary_dict()
    
    def get_domain_results(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get raw results for specific domain."""
        return self.raw_domain_results.get(domain)
    
    def get_domain_patterns(self, domain: str) -> List[EmergenceSignature]:
        """Get emergence signatures for specific domain."""
        return [sig for sig in self.signatures if sig.domain == domain]
    
    def get_correlation(self, domain1: str, domain2: str) -> Optional[float]:
        """Get correlation coefficient between two domains."""
        domains = self.correlation_matrix.domains
        try:
            i = domains.index(domain1)
            j = domains.index(domain2)
            return self.correlation_matrix.correlation_values[i][j]
        except (ValueError, IndexError):
            return None
    
    def get_top_patterns(self, n: int = 10, by: str = 'confidence') -> List[EmergenceSignature]:
        """
        Get top N patterns sorted by specified criteria.
        
        Args:
            n: Number of patterns to return
            by: Sort criteria ('confidence', 'emergence_strength')
        """
        if by == 'confidence':
            sorted_patterns = sorted(self.signatures, key=lambda s: s.confidence, reverse=True)
        elif by == 'emergence_strength':
            sorted_patterns = sorted(self.signatures, key=lambda s: s.emergence_strength, reverse=True)
        else:
            raise ValueError(f"Invalid sort criteria: {by}")
        
        return sorted_patterns[:n]
    
    def export(self, 
               output_path: str, 
               format: str = 'json', 
               include_raw_data: bool = False) -> str:
        """
        Export results to file.
        
        Args:
            output_path: Path to save results
            format: Export format ('json', 'pickle', 'hdf5')
            include_raw_data: Whether to include raw domain results
            
        Returns:
            Path to saved file
        """
        
    def create_report(self, output_path: str, format: str = 'markdown') -> str:
        """
        Generate human-readable report.
        
        Args:
            output_path: Path to save report
            format: Report format ('markdown', 'html', 'pdf')
            
        Returns:
            Path to saved report
        """
        
    def visualize(self, output_dir: str) -> List[str]:
        """
        Generate visualization plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plots
        """
```

### Pattern Query API

```python
class PatternQuery:
    """Fluent interface for querying emergence patterns."""
    
    def __init__(self, signatures: List[EmergenceSignature]):
        self.signatures = signatures
    
    def from_domain(self, domain: str) -> 'PatternQuery':
        """Filter patterns from specific domain."""
        filtered = [s for s in self.signatures if s.domain == domain]
        return PatternQuery(filtered)
    
    def with_type(self, pattern_type: str) -> 'PatternQuery':
        """Filter patterns by type."""
        filtered = [s for s in self.signatures if s.pattern_type == pattern_type]
        return PatternQuery(filtered)
    
    def min_confidence(self, threshold: float) -> 'PatternQuery':
        """Filter patterns with minimum confidence."""
        filtered = [s for s in self.signatures if s.confidence >= threshold]
        return PatternQuery(filtered)
    
    def min_emergence_strength(self, threshold: float) -> 'PatternQuery':
        """Filter patterns with minimum emergence strength."""
        filtered = [s for s in self.signatures if s.emergence_strength >= threshold]
        return PatternQuery(filtered)
    
    def top(self, n: int, by: str = 'confidence') -> List[EmergenceSignature]:
        """Get top N patterns."""
        if by == 'confidence':
            sorted_patterns = sorted(self.signatures, key=lambda s: s.confidence, reverse=True)
        elif by == 'emergence_strength':
            sorted_patterns = sorted(self.signatures, key=lambda s: s.emergence_strength, reverse=True)
        else:
            raise ValueError(f"Invalid sort criteria: {by}")
        
        return sorted_patterns[:n]
    
    def count(self) -> int:
        """Count matching patterns."""
        return len(self.signatures)
    
    def mean_confidence(self) -> float:
        """Calculate mean confidence of matching patterns."""
        if not self.signatures:
            return 0.0
        return sum(s.confidence for s in self.signatures) / len(self.signatures)

# Usage examples
query = PatternQuery(results.signatures)

# Find high-confidence gravity patterns
gravity_patterns = query.from_domain('gravity').min_confidence(0.8).top(5)

# Find strong emergence patterns across all domains
strong_patterns = query.min_emergence_strength(0.7).top(10, by='emergence_strength')

# Count patterns by domain
for domain in ['gravity', 'med', 'navier']:
    count = query.from_domain(domain).count()
    print(f"{domain}: {count} patterns")
```

## Advanced API

### Custom Domain Integration

```python
from unified_emergence_v2.domain import DomainAdapter
from unified_emergence_v2.infrastructure import TestRunner

class CustomDomainAdapter:
    """Example of custom domain adapter implementation."""
    
    def __init__(self, test_runner: TestRunner):
        self.test_runner = test_runner
        self.logger = LoggingConfig.setup_logger(f"{__name__}.CustomDomain")
    
    @property
    def domain_name(self) -> str:
        return "custom_physics"
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """Extract patterns from custom physics simulation."""
        # Implementation specific to custom physics domain
        pass
    
    def validate_constraints(self, results: Dict[str, Any]) -> List[str]:
        """Validate custom physics constraints."""
        # Implementation specific to custom physics domain
        pass

# Register custom domain
framework = UnifiedEmergenceFramework()
framework.register_domain_adapter(CustomDomainAdapter(custom_test_runner))

# Use in validation
results = framework.run_phase1_validation({
    'domains': ['gravity', 'med', 'custom_physics']
})
```

### Async API

```python
import asyncio
from unified_emergence_v2.async_api import AsyncEmergenceFramework

class AsyncEmergenceFramework:
    """Async version of the framework for high-performance applications."""
    
    async def run_phase1_validation_async(self, config: Optional[Dict[str, Any]] = None) -> EmergenceResults:
        """Async version of phase1 validation."""
        
    async def run_multiple_validations_async(self, configs: List[Dict[str, Any]]) -> List[EmergenceResults]:
        """Run multiple validations in parallel."""
        
    async def stream_validation_progress(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Stream validation progress updates."""

# Usage
async def main():
    framework = AsyncEmergenceFramework()
    
    # Single async validation
    results = await framework.run_phase1_validation_async()
    
    # Multiple parallel validations
    configs = [
        {'domains': ['gravity', 'med'], 'field_sizes': [32]},
        {'domains': ['navier', 'tinycimm'], 'field_sizes': [64]},
        {'domains': ['hodge'], 'field_sizes': [32, 64]}
    ]
    
    all_results = await framework.run_multiple_validations_async(configs)
    
    # Stream progress
    async for progress in framework.stream_validation_progress(configs[0]):
        print(f"Progress: {progress['percentage']:.1f}% - {progress['stage']}")

asyncio.run(main())
```

### Plugin System

```python
from unified_emergence_v2.plugins import Plugin

class VisualizationPlugin(Plugin):
    """Plugin for advanced visualizations."""
    
    def __init__(self):
        super().__init__("visualization", "1.0.0")
    
    def process_results(self, results: EmergenceResults) -> Dict[str, Any]:
        """Process results and add visualization data."""
        # Generate advanced plots and visualizations
        return {"visualizations": ["plot1.png", "plot2.png"]}

# Register and use plugin
framework = UnifiedEmergenceFramework()
framework.register_plugin(VisualizationPlugin())

results = framework.run_phase1_validation()
# Plugin automatically processes results
```

This API design provides multiple levels of abstraction, from simple one-liner usage to advanced customization capabilities, while maintaining consistency and type safety throughout.
