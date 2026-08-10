"""
MED (Macro Emergence Dynamics) domain adapter for the Unified Emergence Framework v2.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..domain.models import EmergenceSignature
from ..domain.protocols import DomainAdapter, TestRunner

logger = logging.getLogger(__name__)


class MEDDomainAdapter:
    """
    Domain adapter for Macro Emergence Dynamics pattern analysis.
    
    This adapter extracts emergence patterns from MED computational validation,
    focusing on complexity bounds and emergence detection metrics.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize MED domain adapter.
        
        Args:
            test_runner: Test runner for executing MED validations
        """
        self.test_runner = test_runner
        self._domain_name = 'med'
    
    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self._domain_name
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from MED validation results.
        
        Args:
            domain_results: Raw results from MED validations
            
        Returns:
            List of emergence signatures representing complexity bound patterns
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
                        
                        # Create signature from standalone MED simulation data
                        signature = self._create_med_dynamics_signature(metrics, field_size)
                        if signature:
                            signatures.append(signature)
            
            logger.info(f"Extracted {len(signatures)} MED patterns")
            
        except Exception as e:
            logger.error(f"Error extracting MED patterns: {e}")
        
        return signatures
    
    def _create_med_dynamics_signature(self, metrics: Dict[str, Any], field_size: int) -> EmergenceSignature:
        """Create emergence signature for MED dynamics."""
        try:
            # Extract key MED metrics from standalone simulator
            complexity_bound = float(metrics.get('complexity_bound', 0.0))
            emergence_rate = float(metrics.get('emergence_rate', 0.0))
            convergence_time = float(metrics.get('convergence_time', 100.0))
            stability_metric = float(metrics.get('stability_metric', 0.0))
            
            # Normalize convergence time (lower is better, so invert)
            normalized_convergence = max(0.0, 1.0 - min(1.0, convergence_time / 200.0))
            
            # Create feature vector [complexity, emergence_rate, convergence, stability]
            features = [
                complexity_bound,
                emergence_rate,
                normalized_convergence,
                stability_metric
            ]
            
            # Calculate confidence as average of key metrics
            confidence = np.mean([complexity_bound, emergence_rate, stability_metric])
            
            # Calculate emergence strength based on dynamics and complexity
            # Higher complexity bounds and faster emergence indicate stronger emergence
            complexity_factor = complexity_bound
            dynamics_factor = emergence_rate * normalized_convergence
            emergence_strength = (complexity_factor * 0.4 + dynamics_factor * 0.6)
            
            # Create metadata
            metadata = {
                'field_size': field_size,
                'convergence_time': convergence_time,
                'complexity_bound': complexity_bound,
                'emergence_rate': emergence_rate
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='macro_emergence_dynamics',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating MED dynamics signature: {e}")
            return None
    
    def _create_complexity_bound_signature(self, complexity_bound: float, 
                                         best_score: float, runs: List[Dict]) -> EmergenceSignature:
        """Create emergence signature for overall complexity bound satisfaction."""
        try:
            # Calculate aggregate metrics from runs
            scores = [run.get('score', 0.0) for run in runs]
            field_sizes = [run.get('field_size', 32) for run in runs]
            
            avg_score = np.mean(scores) if scores else 0.0
            max_score = np.max(scores) if scores else 0.0
            score_consistency = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
            avg_field_size = np.mean(field_sizes) if field_sizes else 32
            
            # Normalize field size factor
            field_size_factor = min(1.0, avg_field_size / 64.0)
            
            # Create feature vector [bound_satisfaction, avg_score, max_score, consistency]
            features = [
                complexity_bound,
                avg_score,
                max_score,
                max(0.0, score_consistency)  # Ensure non-negative
            ]
            
            # Confidence based on complexity bound satisfaction and score consistency
            confidence = (complexity_bound * 0.6 + score_consistency * 0.4)
            
            # Emergence strength based on scores and field complexity
            emergence_strength = (max_score * 0.5 + avg_score * 0.3 + field_size_factor * 0.2)
            
            # Metadata
            metadata = {
                'complexity_bound_satisfaction': complexity_bound,
                'best_score': best_score,
                'avg_score': avg_score,
                'num_runs': len(runs),
                'avg_field_size': avg_field_size
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='complexity_bound',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating complexity bound signature: {e}")
            return None
    
    def _create_run_signature(self, run_data: Dict[str, Any]) -> EmergenceSignature:
        """Create emergence signature for individual MED run."""
        try:
            score = float(run_data.get('score', 0.0))
            field_size = int(run_data.get('field_size', 32))
            parameters = run_data.get('parameters', {})
            
            # Extract parameter values for feature vector
            alpha = float(parameters.get('alpha', 0.01))
            beta = float(parameters.get('beta', 0.1))
            gamma = float(parameters.get('gamma', 1.0))
            
            # Normalize parameters to [0,1] range (assuming reasonable ranges)
            normalized_alpha = min(1.0, alpha * 100)  # Assuming alpha ~ 0.01
            normalized_beta = min(1.0, beta * 10)     # Assuming beta ~ 0.1
            normalized_gamma = min(1.0, gamma / 2.0)  # Assuming gamma ~ 1.0
            
            # Create feature vector [score, alpha, beta, gamma]
            features = [
                score,
                normalized_alpha,
                normalized_beta,
                normalized_gamma
            ]
            
            # Confidence equals score for individual runs
            confidence = score
            
            # Emergence strength based on score and field complexity
            field_complexity = min(1.0, field_size / 64.0)
            emergence_strength = (score * 0.8 + field_complexity * 0.2)
            
            # Metadata
            metadata = {
                'score': score,
                'field_size': field_size,
                'parameters': parameters,
                'parameter_alpha': alpha,
                'parameter_beta': beta,
                'parameter_gamma': gamma
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='med_run',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating MED run signature: {e}")
            return None
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate MED-specific constraints.
        
        Args:
            domain_data: MED validation data to validate
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        try:
            # Check complexity bound satisfaction
            complexity_bound = domain_data.get('complexity_bound_satisfaction', 0.0)
            if complexity_bound < 0.8:
                violations.append(f"Complexity bound satisfaction too low: {complexity_bound:.3f} < 0.8")
            
            # Check best score
            best_score = domain_data.get('best_score', 0.0)
            if best_score < 0.6:
                violations.append(f"Best score too low: {best_score:.3f} < 0.6")
            
            # Check run consistency
            runs = domain_data.get('runs', [])
            if runs:
                scores = [run.get('score', 0.0) for run in runs]
                if len(scores) > 1:
                    score_std = np.std(scores)
                    if score_std > 0.3:
                        violations.append(f"Score variation too high: std={score_std:.3f} > 0.3")
            
            # Check parameter ranges
            for run in runs:
                params = run.get('parameters', {})
                alpha = params.get('alpha', 0.01)
                if alpha <= 0 or alpha > 1.0:
                    violations.append(f"Alpha parameter out of range: {alpha}")
                
                beta = params.get('beta', 0.1)
                if beta <= 0 or beta > 2.0:
                    violations.append(f"Beta parameter out of range: {beta}")
            
        except Exception as e:
            violations.append(f"Error validating MED constraints: {e}")
        
        return violations
