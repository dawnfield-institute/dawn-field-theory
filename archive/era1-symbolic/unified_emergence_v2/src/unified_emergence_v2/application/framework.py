"""
Main framework implementation for the Unified Emergence Framework v2.
"""

import logging
import time
import psutil
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..domain.models import (
    ValidationConfig, EmergenceResults, EmergenceSignature,
    SweepAnalysis, ParameterSweepResult, StatisticalSummary
)
from ..domain.protocols import DomainAdapter, TestRunner, ResultsRepository
from ..application.pattern_analyzer import PatternAnalyzerImpl
from ..infrastructure.test_runner import TestRunnerImpl
from ..infrastructure.results_repository import ResultsRepositoryImpl
from ..adapters import (
    GravityDomainAdapter,
    MEDDomainAdapter, 
    NavierDomainAdapter,
    TinyCIMMDomainAdapter,
    HodgeDomainAdapter
)


logger = logging.getLogger(__name__)


class UnifiedEmergenceFramework:
    """
    Main framework for unified emergence analysis across multiple domains.
    
    This class orchestrates the entire emergence validation pipeline using
    Clean Architecture principles with dependency injection.
    """
    
    def __init__(self, 
                 test_runner: Optional[TestRunner] = None,
                 results_repository: Optional[ResultsRepository] = None,
                 pattern_analyzer: Optional[PatternAnalyzerImpl] = None):
        """
        Initialize the framework with optional dependency injection.
        
        Args:
            test_runner: Custom test runner (default: TestRunnerImpl)
            results_repository: Custom results repository (default: ResultsRepositoryImpl)
            pattern_analyzer: Custom pattern analyzer (default: PatternAnalyzerImpl)
        """
        
        # Dependency injection with sensible defaults
        self.test_runner = test_runner or TestRunnerImpl()
        self.results_repository = results_repository or ResultsRepositoryImpl()
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzerImpl()
        
        # Initialize domain adapters
        self.domain_adapters: Dict[str, DomainAdapter] = {
            'gravity': GravityDomainAdapter(self.test_runner),
            'med': MEDDomainAdapter(self.test_runner),
            'navier': NavierDomainAdapter(self.test_runner),
            'tinycimm': TinyCIMMDomainAdapter(self.test_runner),
            'hodge': HodgeDomainAdapter(self.test_runner)
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup unified logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('unified_emergence_v2.log')
            ]
        )
    
    def run_phase1_validation(self, config: Dict[str, Any]) -> EmergenceResults:
        """
        Run Phase 1 validation with the given configuration.
        
        Args:
            config: Validation configuration dictionary
            
        Returns:
            Complete validation results
        """
        # Convert config dict to ValidationConfig object
        validation_config = self._create_validation_config(config)
        
        # Validate configuration
        config_errors = validation_config.validate()
        if config_errors:
            return self._create_error_results(validation_config, config_errors)
        
        logger.info(f"Starting Phase 1 validation for session {validation_config.session_id}")
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Execute domain tests
            domain_results = self._execute_domain_tests(validation_config)
            
            # Extract patterns from all domains
            all_signatures = self._extract_patterns_from_results(domain_results, validation_config)
            
            # Calculate correlations
            correlation_matrix = self.pattern_analyzer.calculate_correlations(all_signatures)
            
            # Calculate comprehensive metrics
            metrics = self.pattern_analyzer.calculate_metrics(all_signatures, correlation_matrix)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = final_memory - initial_memory
            
            metrics.processing_time_seconds = execution_time
            metrics.memory_usage_mb = peak_memory
            
            # Create results
            results = EmergenceResults(
                session_id=validation_config.session_id,
                timestamp=datetime.now().isoformat(),
                configuration=validation_config,
                signatures=all_signatures,
                metrics=metrics,
                correlation_matrix=correlation_matrix,
                raw_domain_results=domain_results if validation_config.save_raw_domain_results else {},
                processing_log=self._get_processing_log(),
                success=True,
                error_messages=[],
                warnings=[],
                execution_time_seconds=execution_time,
                peak_memory_usage_mb=peak_memory
            )
            
            # Save results if configured
            if validation_config.save_intermediate_results and validation_config.output_directory:
                self.results_repository.save_results(results)
            
            logger.info(f"Phase 1 validation completed successfully in {execution_time:.2f}s")
            logger.info(f"Overall score: {metrics.get_overall_score():.3f}")
            logger.info(f"Phase 1 ready: {results.is_phase1_ready()}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Phase 1 validation failed: {e}", exc_info=True)
            
            return EmergenceResults(
                session_id=validation_config.session_id,
                timestamp=datetime.now().isoformat(),
                configuration=validation_config,
                signatures=[],
                metrics=self.pattern_analyzer._empty_metrics(),
                correlation_matrix=self.pattern_analyzer.calculate_correlations([]),
                raw_domain_results={},
                processing_log=self._get_processing_log(),
                success=False,
                error_messages=[str(e)],
                warnings=[],
                execution_time_seconds=execution_time,
                peak_memory_usage_mb=0.0
            )
    
    def _create_validation_config(self, config: Dict[str, Any]) -> ValidationConfig:
        """Create ValidationConfig from dictionary, using defaults for missing values."""
        return ValidationConfig(
            session_id=config.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            domains=config.get('domains', ['gravity', 'med', 'navier', 'tinycimm', 'hodge']),
            field_sizes=config.get('field_sizes', [32, 64]),
            runs_per_domain=config.get('runs_per_domain', 2),
            parallel_execution=config.get('parallel_execution', True),
            max_workers=config.get('max_workers', None),
            timeout_seconds=config.get('timeout_seconds', 300),
            output_directory=config.get('output_directory', 'results'),  # Default to local results folder
            save_intermediate_results=config.get('save_intermediate_results', True),
            save_raw_domain_results=config.get('save_raw_domain_results', False),
            sec_classification_threshold=config.get('sec_classification_threshold', 0.7),
            pattern_assembly_threshold=config.get('pattern_assembly_threshold', 0.6),
            emergence_consistency_threshold=config.get('emergence_consistency_threshold', 0.5),
            phase1_readiness_threshold=config.get('phase1_readiness_threshold', 0.8),
            min_pattern_confidence=config.get('min_pattern_confidence', 0.3),
            min_emergence_strength=config.get('min_emergence_strength', 0.2),
            max_patterns_per_domain=config.get('max_patterns_per_domain', 10),
            correlation_method=config.get('correlation_method', 'pearson'),
            min_correlation_significance=config.get('min_correlation_significance', 0.05)
        )
    
    def _execute_domain_tests(self, config: ValidationConfig) -> Dict[str, Any]:
        """Execute tests for all configured domains."""
        domain_results = {}
        
        if config.parallel_execution:
            domain_results = self._execute_domains_parallel(config)
        else:
            domain_results = self._execute_domains_sequential(config)
        
        return domain_results
    
    def _execute_domains_parallel(self, config: ValidationConfig) -> Dict[str, Any]:
        """Execute domain tests in parallel."""
        domain_results = {}
        max_workers = config.max_workers or min(len(config.domains), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all domain test jobs
            future_to_domain = {}
            for domain in config.domains:
                if domain in self.domain_adapters:
                    future = executor.submit(self._run_single_domain, domain, config)
                    future_to_domain[future] = domain
                else:
                    logger.warning(f"No adapter found for domain: {domain}")
            
            # Collect results as they complete
            for future in as_completed(future_to_domain, timeout=config.timeout_seconds):
                domain = future_to_domain[future]
                try:
                    result = future.result()
                    domain_results[domain] = result
                    logger.info(f"Completed domain: {domain}")
                except Exception as e:
                    logger.error(f"Domain {domain} failed: {e}")
                    domain_results[domain] = {'error': str(e)}
        
        return domain_results
    
    def _execute_domains_sequential(self, config: ValidationConfig) -> Dict[str, Any]:
        """Execute domain tests sequentially."""
        domain_results = {}
        
        for domain in config.domains:
            if domain in self.domain_adapters:
                try:
                    result = self._run_single_domain(domain, config)
                    domain_results[domain] = result
                    logger.info(f"Completed domain: {domain}")
                except Exception as e:
                    logger.error(f"Domain {domain} failed: {e}")
                    domain_results[domain] = {'error': str(e)}
            else:
                logger.warning(f"No adapter found for domain: {domain}")
        
        return domain_results
    
    def _run_single_domain(self, domain: str, config: ValidationConfig) -> Dict[str, Any]:
        """Run tests for a single domain."""
        domain_config = {
            'field_sizes': config.field_sizes,
            'runs_per_domain': config.runs_per_domain,
            'timeout_seconds': config.timeout_seconds
        }
        
        return self.test_runner.run_domain_tests(domain, domain_config)
    
    def _extract_patterns_from_results(self, domain_results: Dict[str, Any], 
                                     config: ValidationConfig) -> List[EmergenceSignature]:
        """Extract patterns from all domain results."""
        all_signatures = []
        
        for domain, results in domain_results.items():
            if 'error' in results:
                logger.warning(f"Skipping pattern extraction for failed domain: {domain}")
                continue
            
            if domain in self.domain_adapters:
                try:
                    adapter = self.domain_adapters[domain]
                    signatures = adapter.extract_patterns(results)
                    
                    # Filter signatures by quality thresholds
                    filtered_signatures = [
                        sig for sig in signatures
                        if (sig.confidence >= config.min_pattern_confidence and
                            sig.emergence_strength >= config.min_emergence_strength)
                    ]
                    
                    # Limit number of patterns per domain
                    if len(filtered_signatures) > config.max_patterns_per_domain:
                        # Sort by confidence * emergence_strength and take top N
                        filtered_signatures.sort(
                            key=lambda s: s.confidence * s.emergence_strength, 
                            reverse=True
                        )
                        filtered_signatures = filtered_signatures[:config.max_patterns_per_domain]
                    
                    all_signatures.extend(filtered_signatures)
                    logger.info(f"Extracted {len(filtered_signatures)} patterns from {domain}")
                    
                except Exception as e:
                    logger.error(f"Pattern extraction failed for {domain}: {e}")
        
        return all_signatures
    
    def _create_error_results(self, config: ValidationConfig, errors: List[str]) -> EmergenceResults:
        """Create error results for configuration validation failures."""
        return EmergenceResults(
            session_id=config.session_id,
            timestamp=datetime.now().isoformat(),
            configuration=config,
            signatures=[],
            metrics=self.pattern_analyzer._empty_metrics(),
            correlation_matrix=self.pattern_analyzer.calculate_correlations([]),
            raw_domain_results={},
            processing_log=[],
            success=False,
            error_messages=errors,
            warnings=[],
            execution_time_seconds=0.0,
            peak_memory_usage_mb=0.0
        )
    
    def _get_processing_log(self) -> List[str]:
        """Get processing log entries."""
        # Placeholder - in real implementation, this would capture log entries
        return [f"Processing completed at {datetime.now().isoformat()}"]
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return any errors.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages
        """
        validation_config = self._create_validation_config(config)
        return validation_config.validate()
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domain adapters."""
        return list(self.domain_adapters.keys())
    
    def get_domain_adapter(self, domain: str) -> Optional[DomainAdapter]:
        """Get domain adapter for specific domain."""
        return self.domain_adapters.get(domain)
    
    def run_parameter_sweep(self, config: Dict[str, Any]) -> 'SweepAnalysis':
        """
        Run comprehensive parameter sweep analysis.
        
        Args:
            config: Configuration including sweep parameters
            
        Returns:
            SweepAnalysis with comprehensive statistical results
        """
        from ..domain.models import SweepAnalysis, ParameterSweepResult
        
        validation_config = self._create_validation_config(config)
        validation_config.enable_parameter_sweep = True
        validation_config.enable_statistical_analysis = True
        
        logger.info(f"Starting parameter sweep for session {validation_config.session_id}")
        
        # Generate all parameter combinations
        sweep_configs = self._generate_sweep_configurations(validation_config)
        logger.info(f"Generated {len(sweep_configs)} parameter combinations")
        
        # Execute all configurations
        sweep_results = []
        for i, sweep_config in enumerate(sweep_configs):
            logger.info(f"Running sweep configuration {i+1}/{len(sweep_configs)}")
            
            try:
                result = self.run_phase1_validation(sweep_config)
                sweep_result = ParameterSweepResult(
                    field_size=sweep_config['field_sizes'][0],
                    domain=sweep_config['domains'][0] if len(sweep_config['domains']) == 1 else 'mixed',
                    run_index=i,
                    metrics=result.metrics,
                    signatures=result.signatures,
                    execution_time=result.execution_time_seconds,
                    memory_usage=result.peak_memory_usage_mb
                )
                sweep_results.append(sweep_result)
                
            except Exception as e:
                logger.error(f"Sweep configuration {i+1} failed: {e}")
        
        # Perform statistical analysis
        sweep_analysis = self._analyze_sweep_results(sweep_results, validation_config)
        
        # Save sweep results
        if validation_config.save_intermediate_results:
            self._save_sweep_analysis(sweep_analysis, validation_config)
        
        logger.info(f"Parameter sweep completed. Success rate: {sweep_analysis.phase1_success_rate:.2%}")
        
        return sweep_analysis
    
    def _generate_sweep_configurations(self, config: ValidationConfig) -> List[Dict[str, Any]]:
        """Generate all parameter sweep configurations."""
        import itertools
        
        # Parameters to sweep
        field_sizes = config.sweep_field_sizes
        domains = config.sweep_domains or config.domains
        
        # Generate all combinations
        configurations = []
        
        for field_size in field_sizes:
            for domain in domains:
                for run_idx in range(config.sweep_runs_per_config):
                    sweep_config = {
                        'session_id': f"{config.session_id}_sweep_{len(configurations):03d}",
                        'domains': [domain],
                        'field_sizes': [field_size],
                        'runs_per_domain': 1,
                        'parallel_execution': config.parallel_execution,
                        'timeout_seconds': config.timeout_seconds,
                        'output_directory': config.output_directory,
                        'save_intermediate_results': False,  # Don't save individual runs
                        'save_raw_domain_results': config.save_raw_domain_results,
                        'sec_classification_threshold': config.sec_classification_threshold,
                        'pattern_assembly_threshold': config.pattern_assembly_threshold,
                        'emergence_consistency_threshold': config.emergence_consistency_threshold,
                        'phase1_readiness_threshold': config.phase1_readiness_threshold
                    }
                    configurations.append(sweep_config)
        
        return configurations
    
    def _analyze_sweep_results(self, results: List['ParameterSweepResult'], config: ValidationConfig) -> 'SweepAnalysis':
        """Perform comprehensive statistical analysis of sweep results."""
        from ..infrastructure.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(config)
        return analyzer.analyze_sweep_results(results)
    
    def _save_sweep_analysis(self, analysis: 'SweepAnalysis', config: ValidationConfig):
        """Save parameter sweep analysis results."""
        if self.results_repository:
            self.results_repository.save_sweep_analysis(analysis, config)
