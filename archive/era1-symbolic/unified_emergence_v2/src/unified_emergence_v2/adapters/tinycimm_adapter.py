"""
TinyCIMM domain adapter for the Unified Emergence Framework v2.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..domain.models import EmergenceSignature
from ..domain.protocols import DomainAdapter, TestRunner

logger = logging.getLogger(__name__)


class TinyCIMMDomainAdapter:
    """
    Domain adapter for TinyCIMM (Tiny Computational Information Management Model) analysis.
    
    This adapter extracts emergence patterns from TinyCIMM computational validations,
    focusing on information management and computational emergence metrics.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize TinyCIMM domain adapter.
        
        Args:
            test_runner: Test runner for executing TinyCIMM validations
        """
        self.test_runner = test_runner
        self._domain_name = 'tinycimm'
    
    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self._domain_name
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from TinyCIMM validation results.
        
        Args:
            domain_results: Raw results from TinyCIMM validations
            
        Returns:
            List of emergence signatures representing computational emergence patterns
        """
        signatures = []
        
        if 'error' in domain_results:
            logger.warning(f"Skipping pattern extraction due to error: {domain_results['error']}")
            return signatures
        
        try:
            runs = domain_results.get('runs', [])
            
            for run_data in runs:
                signature = self._create_computational_emergence_signature(run_data)
                if signature:
                    signatures.append(signature)
            
            logger.info(f"Extracted {len(signatures)} TinyCIMM patterns")
            
        except Exception as e:
            logger.error(f"Error extracting TinyCIMM patterns: {e}")
        
        return signatures
    
    def _create_computational_emergence_signature(self, run_data: Dict[str, Any]) -> EmergenceSignature:
        """Create emergence signature for computational emergence patterns."""
        try:
            # Extract TinyCIMM metrics
            architecture = run_data.get('architecture', 'unknown')
            score = float(run_data.get('score', 0.0))
            field_size = int(run_data.get('field_size', 32))
            
            # Additional metrics that might be available
            information_density = float(run_data.get('information_density', 0.5))
            computational_efficiency = float(run_data.get('computational_efficiency', 0.7))
            emergence_coefficient = float(run_data.get('emergence_coefficient', 0.6))
            convergence_rate = float(run_data.get('convergence_rate', 0.8))
            
            # Architecture encoding for feature vector
            architecture_encodings = {
                'planck': 0.9,
                'quantum': 0.8,
                'classical': 0.6,
                'hybrid': 0.7,
                'unknown': 0.5
            }
            architecture_score = architecture_encodings.get(architecture.lower(), 0.5)
            
            # Normalize field size
            field_complexity = min(1.0, field_size / 64.0)
            
            # Create feature vector [score, architecture, info_density, efficiency]
            features = [
                score,
                architecture_score,
                information_density,
                computational_efficiency
            ]
            
            # Confidence based on score and convergence
            confidence = (score * 0.6 + convergence_rate * 0.4)
            
            # Emergence strength based on emergence coefficient and complexity
            emergence_strength = (
                emergence_coefficient * 0.5 + 
                field_complexity * 0.2 + 
                computational_efficiency * 0.3
            )
            
            # Metadata
            metadata = {
                'architecture': architecture,
                'score': score,
                'field_size': field_size,
                'information_density': information_density,
                'computational_efficiency': computational_efficiency,
                'emergence_coefficient': emergence_coefficient,
                'convergence_rate': convergence_rate,
                'complexity_class': self._classify_complexity(field_size, score)
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='computational_emergence',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating TinyCIMM signature: {e}")
            return None
    
    def _classify_complexity(self, field_size: int, score: float) -> str:
        """Classify computational complexity based on field size and performance."""
        if field_size >= 64 and score >= 0.8:
            return 'high_complexity_high_performance'
        elif field_size >= 64:
            return 'high_complexity_moderate_performance'
        elif score >= 0.8:
            return 'moderate_complexity_high_performance'
        else:
            return 'moderate_complexity_moderate_performance'
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate TinyCIMM specific constraints.
        
        Args:
            domain_data: TinyCIMM validation data to validate
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        try:
            # Check score range
            score = domain_data.get('score', 0.0)
            if score < 0.0 or score > 1.0:
                violations.append(f"Score out of range: {score} not in [0.0, 1.0]")
            elif score < 0.5:
                violations.append(f"Score too low: {score:.3f} < 0.5")
            
            # Check architecture validity
            architecture = domain_data.get('architecture', '').lower()
            valid_architectures = {'planck', 'quantum', 'classical', 'hybrid'}
            if architecture and architecture not in valid_architectures:
                violations.append(f"Unknown architecture: {architecture}")
            
            # Check information density
            info_density = domain_data.get('information_density', 0.5)
            if info_density < 0.0 or info_density > 1.0:
                violations.append(f"Information density out of range: {info_density}")
            elif info_density < 0.3:
                violations.append(f"Information density too low: {info_density:.3f} < 0.3")
            
            # Check computational efficiency
            efficiency = domain_data.get('computational_efficiency', 0.7)
            if efficiency < 0.0 or efficiency > 1.0:
                violations.append(f"Computational efficiency out of range: {efficiency}")
            elif efficiency < 0.5:
                violations.append(f"Computational efficiency too low: {efficiency:.3f} < 0.5")
            
            # Check emergence coefficient
            emergence_coeff = domain_data.get('emergence_coefficient', 0.6)
            if emergence_coeff < 0.0 or emergence_coeff > 1.0:
                violations.append(f"Emergence coefficient out of range: {emergence_coeff}")
            elif emergence_coeff < 0.4:
                violations.append(f"Emergence coefficient too low: {emergence_coeff:.3f} < 0.4")
            
            # Check convergence rate
            convergence = domain_data.get('convergence_rate', 0.8)
            if convergence < 0.0 or convergence > 1.0:
                violations.append(f"Convergence rate out of range: {convergence}")
            elif convergence < 0.6:
                violations.append(f"Convergence rate too low: {convergence:.3f} < 0.6")
            
            # Check field size
            field_size = domain_data.get('field_size', 32)
            if field_size < 8 or field_size > 128:
                violations.append(f"Field size out of reasonable range: {field_size}")
            
        except Exception as e:
            violations.append(f"Error validating TinyCIMM constraints: {e}")
        
        return violations
