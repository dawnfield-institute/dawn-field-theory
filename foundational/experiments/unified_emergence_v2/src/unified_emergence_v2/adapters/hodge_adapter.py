"""
Hodge theory domain adapter for the Unified Emergence Framework v2.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..domain.models import EmergenceSignature
from ..domain.protocols import DomainAdapter, TestRunner

logger = logging.getLogger(__name__)


class HodgeDomainAdapter:
    """
    Domain adapter for Hodge theory emergence pattern analysis.
    
    This adapter extracts emergence patterns from Hodge decomposition and
    differential form analysis, focusing on topological emergence metrics.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize Hodge theory domain adapter.
        
        Args:
            test_runner: Test runner for executing Hodge theory validations
        """
        self.test_runner = test_runner
        self._domain_name = 'hodge'
    
    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self._domain_name
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from Hodge theory validation results.
        
        Args:
            domain_results: Raw results from Hodge theory validations
            
        Returns:
            List of emergence signatures representing topological emergence patterns
        """
        signatures = []
        
        if 'error' in domain_results:
            logger.warning(f"Skipping pattern extraction due to error: {domain_results['error']}")
            return signatures
        
        try:
            runs = domain_results.get('runs', [])
            
            for run_data in runs:
                for field_key, metrics in run_data.items():
                    if field_key.startswith('field_size_'):
                        field_size = int(field_key.split('_')[-1])
                        
                        # Create signature from standalone Hodge theory simulation data
                        signature = self._create_hodge_signature(metrics, field_size)
                        if signature:
                            signatures.append(signature)
            
            logger.info(f"Extracted {len(signatures)} Hodge theory patterns")
            
        except Exception as e:
            logger.error(f"Error extracting Hodge theory patterns: {e}")
        
        return signatures
    
    def _create_hodge_signature(self, metrics: Dict[str, Any], field_size: int) -> EmergenceSignature:
        """Create emergence signature for Hodge theory patterns."""
        try:
            # Extract key Hodge theory metrics from standalone simulator
            form_coherence = float(metrics.get('form_coherence', 0.0))
            boundary_consistency = float(metrics.get('boundary_consistency', 0.0))
            cohomology_rank = int(metrics.get('cohomology_rank', 0))
            differential_stability = float(metrics.get('differential_stability', 0.0))
            topological_invariant = float(metrics.get('topological_invariant', 0.0))
            
            # Normalize cohomology rank to [0,1] range
            normalized_rank = min(1.0, cohomology_rank / 10.0)
            
            # Create feature vector [coherence, consistency, rank, stability, invariant]
            features = [
                form_coherence,
                boundary_consistency,
                normalized_rank,
                differential_stability,
                topological_invariant
            ]
            
            # Calculate confidence based on mathematical consistency
            confidence = np.mean([form_coherence, boundary_consistency, differential_stability])
            
            # Calculate emergence strength based on topological complexity
            # Higher coherence and stability with non-trivial topology indicate emergence
            coherence_factor = (form_coherence + boundary_consistency) / 2.0
            complexity_factor = normalized_rank * topological_invariant
            emergence_strength = (coherence_factor * 0.6 + complexity_factor * 0.4)
            
            # Create metadata
            metadata = {
                'field_size': field_size,
                'cohomology_rank': cohomology_rank,
                'form_coherence': form_coherence,
                'topological_invariant': topological_invariant
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='topological_emergence',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating Hodge signature: {e}")
            return None
    
    def _create_topological_signature(self, total_cycles: int, detection_rate: float, 
                                    runs: List[Dict]) -> EmergenceSignature:
        """Create emergence signature for overall topological emergence."""
        try:
            # Calculate aggregate metrics from runs
            field_sizes = [run.get('field_size', 32) for run in runs]
            primes = [run.get('prime', 7) for run in runs]
            cycles_per_run = [run.get('cycles_detected', 0) for run in runs]
            
            avg_field_size = np.mean(field_sizes) if field_sizes else 32
            avg_prime = np.mean(primes) if primes else 7
            avg_cycles = np.mean(cycles_per_run) if cycles_per_run else 0
            max_cycles = np.max(cycles_per_run) if cycles_per_run else 0
            
            # Normalize metrics
            normalized_total_cycles = min(1.0, total_cycles / 20.0)  # Assume 20 is high
            normalized_avg_field_size = min(1.0, avg_field_size / 64.0)
            normalized_prime = min(1.0, avg_prime / 13.0)  # Normalize to reasonable prime range
            
            # Create feature vector [detection_rate, normalized_cycles, field_complexity, prime_factor]
            features = [
                detection_rate,
                normalized_total_cycles,
                normalized_avg_field_size,
                normalized_prime
            ]
            
            # Confidence based on detection rate and consistency
            cycle_consistency = 1.0 - (np.std(cycles_per_run) / max(1, np.mean(cycles_per_run))) if len(cycles_per_run) > 1 else 1.0
            confidence = (detection_rate * 0.7 + max(0.0, cycle_consistency) * 0.3)
            
            # Emergence strength based on cycles detected and field complexity
            emergence_strength = (
                detection_rate * 0.4 + 
                normalized_total_cycles * 0.3 + 
                normalized_avg_field_size * 0.3
            )
            
            # Metadata
            metadata = {
                'total_cycles_detected': total_cycles,
                'cycle_detection_rate': detection_rate,
                'avg_cycles_per_run': avg_cycles,
                'max_cycles_per_run': max_cycles,
                'num_runs': len(runs),
                'avg_field_size': avg_field_size,
                'avg_prime': avg_prime,
                'topological_complexity': self._classify_topological_complexity(total_cycles, avg_field_size)
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='topological_emergence',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating topological signature: {e}")
            return None
    
    def _create_run_signature(self, run_data: Dict[str, Any]) -> EmergenceSignature:
        """Create emergence signature for individual Hodge run."""
        try:
            field_size = int(run_data.get('field_size', 32))
            prime = int(run_data.get('prime', 7))
            cycles_detected = int(run_data.get('cycles_detected', 0))
            
            # Additional metrics that might be available
            hodge_rank = int(run_data.get('hodge_rank', 1))
            betti_numbers = run_data.get('betti_numbers', [1, 0, 0])
            spectral_gap = float(run_data.get('spectral_gap', 0.5))
            harmonic_forms = int(run_data.get('harmonic_forms', 1))
            
            # Normalize metrics
            normalized_cycles = min(1.0, cycles_detected / 10.0)  # Assume 10 cycles is high for a single run
            normalized_field_size = min(1.0, field_size / 64.0)
            normalized_prime = min(1.0, prime / 13.0)
            normalized_spectral_gap = min(1.0, spectral_gap)
            
            # Create feature vector [cycles, field_size, prime, spectral_gap]
            features = [
                normalized_cycles,
                normalized_field_size,
                normalized_prime,
                normalized_spectral_gap
            ]
            
            # Confidence based on cycles detected and spectral gap
            cycle_factor = min(1.0, cycles_detected / 5.0)  # Normalize to reasonable expectation
            confidence = (cycle_factor * 0.6 + normalized_spectral_gap * 0.4)
            
            # Emergence strength based on topological complexity
            topological_complexity = (
                normalized_cycles * 0.3 + 
                normalized_field_size * 0.3 + 
                normalized_spectral_gap * 0.4
            )
            emergence_strength = topological_complexity
            
            # Metadata
            metadata = {
                'field_size': field_size,
                'prime': prime,
                'cycles_detected': cycles_detected,
                'hodge_rank': hodge_rank,
                'betti_numbers': betti_numbers,
                'spectral_gap': spectral_gap,
                'harmonic_forms': harmonic_forms,
                'prime_class': self._classify_prime(prime),
                'topological_dimension': len([b for b in betti_numbers if b > 0])
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='hodge_decomposition',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating Hodge run signature: {e}")
            return None
    
    def _classify_topological_complexity(self, total_cycles: int, avg_field_size: float) -> str:
        """Classify topological complexity based on cycles and field size."""
        if total_cycles >= 15 and avg_field_size >= 48:
            return 'high_complexity'
        elif total_cycles >= 10 or avg_field_size >= 32:
            return 'moderate_complexity'
        else:
            return 'low_complexity'
    
    def _classify_prime(self, prime: int) -> str:
        """Classify prime number for mathematical context."""
        if prime <= 3:
            return 'small_prime'
        elif prime <= 7:
            return 'moderate_prime'
        elif prime <= 13:
            return 'large_prime'
        else:
            return 'very_large_prime'
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate Hodge theory specific constraints.
        
        Args:
            domain_data: Hodge theory validation data to validate
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        try:
            # Check total cycles detected
            total_cycles = domain_data.get('total_cycles_detected', 0)
            if total_cycles < 0:
                violations.append(f"Invalid total cycles: {total_cycles} < 0")
            
            # Check cycle detection rate
            detection_rate = domain_data.get('cycle_detection_rate', 0.0)
            if detection_rate < 0.0 or detection_rate > 1.0:
                violations.append(f"Detection rate out of range: {detection_rate}")
            elif detection_rate < 0.5:
                violations.append(f"Detection rate too low: {detection_rate:.3f} < 0.5")
            
            # Check individual run constraints
            runs = domain_data.get('runs', [])
            for i, run in enumerate(runs):
                field_size = run.get('field_size', 32)
                if field_size < 8 or field_size > 128:
                    violations.append(f"Run {i}: field size out of range: {field_size}")
                
                prime = run.get('prime', 7)
                if not self._is_prime(prime):
                    violations.append(f"Run {i}: {prime} is not a prime number")
                
                cycles = run.get('cycles_detected', 0)
                if cycles < 0:
                    violations.append(f"Run {i}: negative cycles detected: {cycles}")
                
                spectral_gap = run.get('spectral_gap', 0.5)
                if spectral_gap < 0.0 or spectral_gap > 1.0:
                    violations.append(f"Run {i}: spectral gap out of range: {spectral_gap}")
                
                betti_numbers = run.get('betti_numbers', [1, 0, 0])
                if not isinstance(betti_numbers, list) or any(b < 0 for b in betti_numbers):
                    violations.append(f"Run {i}: invalid Betti numbers: {betti_numbers}")
            
        except Exception as e:
            violations.append(f"Error validating Hodge theory constraints: {e}")
        
        return violations
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
