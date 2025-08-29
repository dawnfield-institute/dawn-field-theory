"""
Domain models for the Unified Emergence Framework v2.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime


@dataclass(frozen=True)
class EmergenceSignature:
    """
    Represents an emergence pattern extracted from a specific domain.
    
    This is the core data structure that captures patterns of emergence
    across different domains (gravity, MED, Navier-Stokes, etc.).
    """
    
    domain: str  # Domain name (e.g., 'gravity', 'med', 'navier')
    pattern_type: str  # Type of pattern (e.g., 'orbital_dynamics', 'complexity_bound')
    features: List[float]  # Normalized feature vector representing the pattern
    confidence: float  # Confidence in the pattern extraction (0.0 to 1.0)
    emergence_strength: float  # Strength of emergence detected (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional domain-specific metadata
    
    # Auto-generated fields
    feature_hash: str = field(init=False)
    extraction_timestamp: str = field(init=False)
    
    def __post_init__(self):
        # Validate confidence and emergence_strength
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not (0.0 <= self.emergence_strength <= 1.0):
            raise ValueError(f"Emergence strength must be between 0.0 and 1.0, got {self.emergence_strength}")
        
        # Generate feature hash for pattern identification
        feature_str = json.dumps(self.features, sort_keys=True)
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        object.__setattr__(self, 'feature_hash', feature_hash)
        
        # Set extraction timestamp
        timestamp = datetime.now().isoformat()
        object.__setattr__(self, 'extraction_timestamp', timestamp)


@dataclass
class CorrelationMatrix:
    """
    Represents correlations between patterns across domains.
    """
    
    domains: List[str]
    correlation_values: List[List[float]]
    mean_correlation: float
    correlation_consistency: float
    calculation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_correlation(self, domain1: str, domain2: str) -> Optional[float]:
        """Get correlation between two specific domains."""
        try:
            idx1 = self.domains.index(domain1)
            idx2 = self.domains.index(domain2)
            return self.correlation_values[idx1][idx2]
        except (ValueError, IndexError):
            return None
    
    def get_domain_correlations(self, domain: str) -> Dict[str, float]:
        """Get all correlations for a specific domain."""
        try:
            idx = self.domains.index(domain)
            return {
                other_domain: self.correlation_values[idx][i]
                for i, other_domain in enumerate(self.domains)
            }
        except ValueError:
            return {}


@dataclass
class ValidationMetrics:
    """
    Comprehensive metrics for emergence validation.
    """
    
    # Core metrics from v1
    sec_classification_accuracy: float
    pattern_assembly_success_rate: float
    emergence_consistency_score: float
    phase1_readiness_score: float
    
    # Pattern statistics
    total_patterns_extracted: int
    patterns_per_domain: Dict[str, int]
    
    # Cross-domain metrics
    cross_domain_correlations: float
    correlation_consistency: float
    
    # Performance metrics
    processing_time_seconds: float
    memory_usage_mb: float
    
    # Quality metrics
    average_pattern_confidence: float
    average_emergence_strength: float
    pattern_diversity_score: float
    
    calculation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_overall_score(self) -> float:
        """Calculate overall validation score."""
        core_score = (
            self.sec_classification_accuracy + 
            self.pattern_assembly_success_rate + 
            self.emergence_consistency_score + 
            self.phase1_readiness_score
        ) / 4.0
        
        quality_score = (
            self.average_pattern_confidence + 
            self.average_emergence_strength + 
            self.pattern_diversity_score
        ) / 3.0
        
        # Weight core metrics more heavily
        return (core_score * 0.7) + (quality_score * 0.3)


@dataclass
class ValidationConfig:
    """
    Configuration for emergence validation runs.
    """
    
    # Session identification
    session_id: str = field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Domain configuration
    domains: List[str] = field(default_factory=lambda: ['gravity', 'med', 'navier', 'tinycimm', 'hodge'])
    field_sizes: List[int] = field(default_factory=lambda: [32, 64])
    runs_per_domain: int = 2
    
    # Execution configuration
    parallel_execution: bool = True
    max_workers: Optional[int] = None
    timeout_seconds: int = 300
    
    # Parameter sweep configuration
    enable_parameter_sweep: bool = False
    sweep_field_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    sweep_runs_per_config: int = 5
    sweep_domains: Optional[List[str]] = None  # If None, uses domains list
    
    # Statistical analysis configuration
    enable_statistical_analysis: bool = False
    statistical_confidence_level: float = 0.95
    min_runs_for_statistics: int = 3
    bootstrap_samples: int = 1000
    
    # Advanced analysis configuration
    enable_convergence_analysis: bool = False
    convergence_window: int = 5
    convergence_threshold: float = 0.01
    enable_outlier_detection: bool = False
    outlier_threshold: float = 2.0  # Standard deviations
    
    # Performance profiling
    enable_performance_profiling: bool = False
    profile_memory_usage: bool = False
    profile_execution_time: bool = True
    
    # Output configuration
    output_directory: Optional[str] = None
    save_intermediate_results: bool = True
    save_raw_domain_results: bool = False
    
    # Validation thresholds
    sec_classification_threshold: float = 0.7
    pattern_assembly_threshold: float = 0.6
    emergence_consistency_threshold: float = 0.5
    phase1_readiness_threshold: float = 0.8
    
    # Pattern extraction parameters
    min_pattern_confidence: float = 0.5
    min_emergence_strength: float = 0.3
    max_patterns_per_domain: int = 10
    
    # Correlation parameters
    correlation_method: str = 'pearson'  # 'pearson', 'spearman', 'cosine'
    min_correlation_significance: float = 0.05
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate domains
        valid_domains = {'gravity', 'med', 'navier', 'tinycimm', 'hodge'}
        invalid_domains = set(self.domains) - valid_domains
        if invalid_domains:
            errors.append(f"Invalid domains: {invalid_domains}")
        
        # Validate field sizes (should be powers of 2)
        for size in self.field_sizes:
            if size <= 0 or (size & (size - 1)) != 0:
                errors.append(f"Field size {size} is not a positive power of 2")
        
        # Validate thresholds
        thresholds = [
            ('sec_classification_threshold', self.sec_classification_threshold),
            ('pattern_assembly_threshold', self.pattern_assembly_threshold),
            ('emergence_consistency_threshold', self.emergence_consistency_threshold),
            ('phase1_readiness_threshold', self.phase1_readiness_threshold),
            ('min_pattern_confidence', self.min_pattern_confidence),
            ('min_emergence_strength', self.min_emergence_strength),
        ]
        
        for name, value in thresholds:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Validate execution parameters
        if self.runs_per_domain <= 0:
            errors.append("runs_per_domain must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.max_workers is not None and self.max_workers <= 0:
            errors.append("max_workers must be positive or None")
        
        return errors


@dataclass
class EmergenceResults:
    """
    Complete results from an emergence validation run.
    """
    
    # Session information
    session_id: str
    timestamp: str
    configuration: ValidationConfig
    
    # Core results
    signatures: List[EmergenceSignature]
    metrics: ValidationMetrics
    correlation_matrix: CorrelationMatrix
    
    # Raw data (optional)
    raw_domain_results: Dict[str, Any]
    
    # Execution information
    processing_log: List[str]
    success: bool
    error_messages: List[str]
    warnings: List[str]
    
    # Performance data
    execution_time_seconds: float = 0.0
    peak_memory_usage_mb: float = 0.0
    
    def get_signatures_by_domain(self, domain: str) -> List[EmergenceSignature]:
        """Get all signatures for a specific domain."""
        return [sig for sig in self.signatures if sig.domain == domain]
    
    def get_domain_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for each domain."""
        summary = {}
        
        for domain in self.configuration.domains:
            domain_signatures = self.get_signatures_by_domain(domain)
            
            if domain_signatures:
                confidences = [sig.confidence for sig in domain_signatures]
                strengths = [sig.emergence_strength for sig in domain_signatures]
                
                summary[domain] = {
                    'pattern_count': len(domain_signatures),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'avg_emergence_strength': sum(strengths) / len(strengths),
                    'max_confidence': max(confidences),
                    'pattern_types': list(set(sig.pattern_type for sig in domain_signatures))
                }
            else:
                summary[domain] = {
                    'pattern_count': 0,
                    'avg_confidence': 0.0,
                    'avg_emergence_strength': 0.0,
                    'max_confidence': 0.0,
                    'pattern_types': []
                }
        
        return summary
    
    def is_phase1_ready(self) -> bool:
        """Check if results meet Phase 1 readiness criteria."""
        if not self.success:
            return False
        
        config = self.configuration
        metrics = self.metrics
        
        return (
            metrics.sec_classification_accuracy >= config.sec_classification_threshold and
            metrics.pattern_assembly_success_rate >= config.pattern_assembly_threshold and
            metrics.emergence_consistency_score >= config.emergence_consistency_threshold and
            metrics.phase1_readiness_score >= config.phase1_readiness_threshold
        )


@dataclass
class StatisticalSummary:
    """Statistical analysis of multiple validation runs."""
    
    # Basic statistics
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    median: float
    
    # Confidence intervals
    confidence_level: float
    confidence_interval: Tuple[float, float]
    
    # Distribution properties
    skewness: float
    kurtosis: float
    is_normal: bool  # Shapiro-Wilk test result
    
    # Sample properties
    sample_size: int
    outliers_count: int
    outlier_indices: List[int]


@dataclass
class ParameterSweepResult:
    """Results from a parameter sweep configuration."""
    
    # Configuration
    field_size: int
    domain: str
    run_index: int
    
    # Results
    metrics: ValidationMetrics
    signatures: List[EmergenceSignature]
    execution_time: float
    memory_usage: float
    
    # Analysis flags
    is_outlier: bool = False
    convergence_iteration: Optional[int] = None


@dataclass
class SweepAnalysis:
    """Comprehensive analysis of parameter sweep results."""
    
    # Sweep configuration
    sweep_parameters: Dict[str, Any]
    total_runs: int
    successful_runs: int
    
    # Statistical summaries for each metric
    overall_score_stats: StatisticalSummary
    sec_classification_stats: StatisticalSummary
    pattern_assembly_stats: StatisticalSummary
    emergence_consistency_stats: StatisticalSummary
    phase1_readiness_stats: StatisticalSummary
    
    # Performance statistics
    execution_time_stats: StatisticalSummary
    memory_usage_stats: StatisticalSummary
    
    # Pattern analysis
    total_patterns_stats: StatisticalSummary
    pattern_confidence_stats: StatisticalSummary
    emergence_strength_stats: StatisticalSummary
    
    # Domain-specific analysis
    domain_performance: Dict[str, StatisticalSummary]
    field_size_performance: Dict[int, StatisticalSummary]
    
    # Convergence analysis
    convergence_analysis: Optional[Dict[str, Any]] = None
    
    # Correlations
    parameter_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    phase1_success_rate: float = 0.0
