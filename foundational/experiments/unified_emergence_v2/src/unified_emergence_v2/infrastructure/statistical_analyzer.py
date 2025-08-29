"""
Statistical analysis module for parameter sweeps and multi-run analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from dataclasses import asdict

from ..domain.models import (
    ParameterSweepResult, SweepAnalysis, StatisticalSummary, 
    ValidationConfig, ValidationMetrics
)

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Comprehensive statistical analyzer for parameter sweep results."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize statistical analyzer with configuration."""
        self.config = config
        self.confidence_level = config.statistical_confidence_level
        self.min_runs = config.min_runs_for_statistics
        self.bootstrap_samples = config.bootstrap_samples
        self.outlier_threshold = config.outlier_threshold
        
    def analyze_sweep_results(self, results: List[ParameterSweepResult]) -> SweepAnalysis:
        """Perform comprehensive analysis of parameter sweep results."""
        if len(results) < self.min_runs:
            raise ValueError(f"Insufficient data for statistical analysis. Need at least {self.min_runs} runs, got {len(results)}")
        
        logger.info(f"Analyzing {len(results)} sweep results")
        
        # Extract metric values
        metrics_data = self._extract_metrics_data(results)
        
        # Detect outliers
        outlier_indices = self._detect_outliers(metrics_data['overall_score'])
        self._mark_outliers(results, outlier_indices)
        
        # Calculate statistical summaries
        overall_score_stats = self._calculate_statistical_summary(metrics_data['overall_score'])
        sec_classification_stats = self._calculate_statistical_summary(metrics_data['sec_classification'])
        pattern_assembly_stats = self._calculate_statistical_summary(metrics_data['pattern_assembly'])
        emergence_consistency_stats = self._calculate_statistical_summary(metrics_data['emergence_consistency'])
        phase1_readiness_stats = self._calculate_statistical_summary(metrics_data['phase1_readiness'])
        
        # Performance statistics
        execution_time_stats = self._calculate_statistical_summary(metrics_data['execution_time'])
        memory_usage_stats = self._calculate_statistical_summary(metrics_data['memory_usage'])
        
        # Pattern analysis
        total_patterns_stats = self._calculate_statistical_summary(metrics_data['total_patterns'])
        pattern_confidence_stats = self._calculate_statistical_summary(metrics_data['pattern_confidence'])
        emergence_strength_stats = self._calculate_statistical_summary(metrics_data['emergence_strength'])
        
        # Domain and field size analysis
        domain_performance = self._analyze_by_domain(results)
        field_size_performance = self._analyze_by_field_size(results)
        
        # Convergence analysis
        convergence_analysis = None
        if self.config.enable_convergence_analysis:
            convergence_analysis = self._analyze_convergence(results)
        
        # Parameter correlations
        parameter_correlations = self._calculate_parameter_correlations(results)
        
        # Optimal parameters
        optimal_parameters = self._find_optimal_parameters(results)
        
        # Phase 1 success rate
        phase1_success_rate = self._calculate_phase1_success_rate(results)
        
        # Create sweep analysis
        sweep_analysis = SweepAnalysis(
            sweep_parameters=self._get_sweep_parameters(),
            total_runs=len(results),
            successful_runs=len([r for r in results if r.metrics.get_overall_score() > 0]),
            overall_score_stats=overall_score_stats,
            sec_classification_stats=sec_classification_stats,
            pattern_assembly_stats=pattern_assembly_stats,
            emergence_consistency_stats=emergence_consistency_stats,
            phase1_readiness_stats=phase1_readiness_stats,
            execution_time_stats=execution_time_stats,
            memory_usage_stats=memory_usage_stats,
            total_patterns_stats=total_patterns_stats,
            pattern_confidence_stats=pattern_confidence_stats,
            emergence_strength_stats=emergence_strength_stats,
            domain_performance=domain_performance,
            field_size_performance=field_size_performance,
            convergence_analysis=convergence_analysis,
            parameter_correlations=parameter_correlations,
            optimal_parameters=optimal_parameters,
            phase1_success_rate=phase1_success_rate
        )
        
        logger.info(f"Statistical analysis completed. Phase 1 success rate: {phase1_success_rate:.2%}")
        
        return sweep_analysis
    
    def _extract_metrics_data(self, results: List[ParameterSweepResult]) -> Dict[str, List[float]]:
        """Extract metric values for statistical analysis."""
        data = {
            'overall_score': [],
            'sec_classification': [],
            'pattern_assembly': [],
            'emergence_consistency': [],
            'phase1_readiness': [],
            'execution_time': [],
            'memory_usage': [],
            'total_patterns': [],
            'pattern_confidence': [],
            'emergence_strength': []
        }
        
        for result in results:
            metrics = result.metrics
            data['overall_score'].append(metrics.get_overall_score())
            data['sec_classification'].append(metrics.sec_classification_accuracy)
            data['pattern_assembly'].append(metrics.pattern_assembly_success_rate)
            data['emergence_consistency'].append(metrics.emergence_consistency_score)
            data['phase1_readiness'].append(metrics.phase1_readiness_score)
            data['execution_time'].append(result.execution_time)
            data['memory_usage'].append(result.memory_usage)
            data['total_patterns'].append(metrics.total_patterns_extracted)
            data['pattern_confidence'].append(metrics.average_pattern_confidence)
            data['emergence_strength'].append(metrics.average_emergence_strength)
        
        return data
    
    def _calculate_statistical_summary(self, values: List[float]) -> StatisticalSummary:
        """Calculate comprehensive statistical summary for a metric."""
        values_array = np.array(values)
        
        # Basic statistics
        mean = float(np.mean(values_array))
        std_dev = float(np.std(values_array, ddof=1))
        min_value = float(np.min(values_array))
        max_value = float(np.max(values_array))
        median = float(np.median(values_array))
        
        # Confidence interval
        confidence_interval = self._calculate_confidence_interval(values_array)
        
        # Distribution properties
        skewness = float(stats.skew(values_array))
        kurtosis = float(stats.kurtosis(values_array))
        
        # Normality test (only if we have enough samples)
        is_normal = True
        if len(values) >= 3:
            _, p_value = stats.shapiro(values_array)
            is_normal = p_value > 0.05
        
        # Outlier detection
        outlier_indices = self._detect_outliers(values)
        
        return StatisticalSummary(
            mean=mean,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            median=median,
            confidence_level=self.confidence_level,
            confidence_interval=confidence_interval,
            skewness=skewness,
            kurtosis=kurtosis,
            is_normal=is_normal,
            sample_size=len(values),
            outliers_count=len(outlier_indices),
            outlier_indices=outlier_indices
        )
    
    def _calculate_confidence_interval(self, values: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap or t-distribution."""
        if len(values) < 3:
            return (float(np.min(values)), float(np.max(values)))
        
        try:
            # Use t-distribution for small samples, bootstrap for larger samples
            if len(values) < 30:
                # t-distribution
                confidence_level = self.confidence_level
                alpha = 1 - confidence_level
                dof = len(values) - 1
                t_critical = stats.t.ppf(1 - alpha/2, dof)
                margin_error = t_critical * (np.std(values, ddof=1) / np.sqrt(len(values)))
                mean = np.mean(values)
                return (float(mean - margin_error), float(mean + margin_error))
            else:
                # Bootstrap
                bootstrap_means = []
                for _ in range(self.bootstrap_samples):
                    bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                alpha = 1 - self.confidence_level
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                ci_lower = np.percentile(bootstrap_means, lower_percentile)
                ci_upper = np.percentile(bootstrap_means, upper_percentile)
                
                return (float(ci_lower), float(ci_upper))
                
        except Exception as e:
            logger.warning(f"Error calculating confidence interval: {e}")
            return (float(np.min(values)), float(np.max(values)))
    
    def _detect_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using modified z-score method."""
        if len(values) < 3:
            return []
        
        values_array = np.array(values)
        median = np.median(values_array)
        mad = np.median(np.abs(values_array - median))
        
        if mad == 0:
            return []
        
        modified_z_scores = 0.6745 * (values_array - median) / mad
        outlier_indices = np.where(np.abs(modified_z_scores) > self.outlier_threshold)[0]
        
        return outlier_indices.tolist()
    
    def _mark_outliers(self, results: List[ParameterSweepResult], outlier_indices: List[int]):
        """Mark outlier results."""
        for idx in outlier_indices:
            if 0 <= idx < len(results):
                results[idx].is_outlier = True
    
    def _analyze_by_domain(self, results: List[ParameterSweepResult]) -> Dict[str, StatisticalSummary]:
        """Analyze performance by domain."""
        domain_scores = {}
        
        for result in results:
            domain = result.domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(result.metrics.get_overall_score())
        
        domain_analysis = {}
        for domain, scores in domain_scores.items():
            if len(scores) >= 1:
                domain_analysis[domain] = self._calculate_statistical_summary(scores)
        
        return domain_analysis
    
    def _analyze_by_field_size(self, results: List[ParameterSweepResult]) -> Dict[int, StatisticalSummary]:
        """Analyze performance by field size."""
        field_size_scores = {}
        
        for result in results:
            field_size = result.field_size
            if field_size not in field_size_scores:
                field_size_scores[field_size] = []
            field_size_scores[field_size].append(result.metrics.get_overall_score())
        
        field_size_analysis = {}
        for field_size, scores in field_size_scores.items():
            if len(scores) >= 1:
                field_size_analysis[field_size] = self._calculate_statistical_summary(scores)
        
        return field_size_analysis
    
    def _analyze_convergence(self, results: List[ParameterSweepResult]) -> Dict[str, Any]:
        """Analyze convergence properties of the results."""
        # Group by configuration
        config_groups = {}
        for result in results:
            key = f"{result.domain}_{result.field_size}"
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(result)
        
        convergence_data = {}
        
        for config_key, config_results in config_groups.items():
            if len(config_results) >= self.config.convergence_window:
                scores = [r.metrics.get_overall_score() for r in config_results]
                
                # Calculate running mean and check for convergence
                running_means = []
                for i in range(self.config.convergence_window, len(scores) + 1):
                    window_mean = np.mean(scores[i-self.config.convergence_window:i])
                    running_means.append(window_mean)
                
                # Find convergence point
                convergence_iteration = None
                if len(running_means) > 1:
                    for i in range(1, len(running_means)):
                        if abs(running_means[i] - running_means[i-1]) < self.config.convergence_threshold:
                            convergence_iteration = i + self.config.convergence_window - 1
                            break
                
                convergence_data[config_key] = {
                    'scores': scores,
                    'running_means': running_means,
                    'convergence_iteration': convergence_iteration,
                    'final_mean': running_means[-1] if running_means else None,
                    'converged': convergence_iteration is not None
                }
        
        return convergence_data
    
    def _calculate_parameter_correlations(self, results: List[ParameterSweepResult]) -> Dict[str, float]:
        """Calculate correlations between parameters and performance."""
        correlations = {}
        
        # Field size correlation
        field_sizes = [r.field_size for r in results]
        overall_scores = [r.metrics.get_overall_score() for r in results]
        
        if len(set(field_sizes)) > 1:  # Only if there's variation
            correlation, p_value = stats.pearsonr(field_sizes, overall_scores)
            if p_value < 0.05:  # Significant correlation
                correlations['field_size_vs_performance'] = correlation
        
        # Execution time correlation
        execution_times = [r.execution_time for r in results]
        correlation, p_value = stats.pearsonr(execution_times, overall_scores)
        if p_value < 0.05:
            correlations['execution_time_vs_performance'] = correlation
        
        # Memory usage correlation
        memory_usage = [r.memory_usage for r in results]
        correlation, p_value = stats.pearsonr(memory_usage, overall_scores)
        if p_value < 0.05:
            correlations['memory_usage_vs_performance'] = correlation
        
        return correlations
    
    def _find_optimal_parameters(self, results: List[ParameterSweepResult]) -> Dict[str, Any]:
        """Find optimal parameter combinations."""
        # Find best performing result
        best_result = max(results, key=lambda r: r.metrics.get_overall_score())
        
        optimal_params = {
            'best_field_size': best_result.field_size,
            'best_domain': best_result.domain,
            'best_overall_score': best_result.metrics.get_overall_score(),
            'best_phase1_ready': best_result.metrics.phase1_readiness_score >= self.config.phase1_readiness_threshold
        }
        
        # Find most reliable parameters (best mean performance)
        domain_means = {}
        field_size_means = {}
        
        for result in results:
            # Domain performance
            if result.domain not in domain_means:
                domain_means[result.domain] = []
            domain_means[result.domain].append(result.metrics.get_overall_score())
            
            # Field size performance
            if result.field_size not in field_size_means:
                field_size_means[result.field_size] = []
            field_size_means[result.field_size].append(result.metrics.get_overall_score())
        
        # Calculate means
        domain_avg = {domain: np.mean(scores) for domain, scores in domain_means.items()}
        field_size_avg = {size: np.mean(scores) for size, scores in field_size_means.items()}
        
        optimal_params['most_reliable_domain'] = max(domain_avg, key=domain_avg.get)
        optimal_params['most_reliable_field_size'] = max(field_size_avg, key=field_size_avg.get)
        optimal_params['domain_mean_scores'] = domain_avg
        optimal_params['field_size_mean_scores'] = field_size_avg
        
        return optimal_params
    
    def _calculate_phase1_success_rate(self, results: List[ParameterSweepResult]) -> float:
        """Calculate Phase 1 readiness success rate."""
        phase1_ready_count = sum(
            1 for r in results 
            if r.metrics.phase1_readiness_score >= self.config.phase1_readiness_threshold
        )
        return phase1_ready_count / len(results) if results else 0.0
    
    def _get_sweep_parameters(self) -> Dict[str, Any]:
        """Get sweep parameter configuration."""
        return {
            'field_sizes': self.config.sweep_field_sizes,
            'domains': self.config.sweep_domains or self.config.domains,
            'runs_per_config': self.config.sweep_runs_per_config,
            'confidence_level': self.confidence_level,
            'outlier_threshold': self.outlier_threshold,
            'bootstrap_samples': self.bootstrap_samples
        }
