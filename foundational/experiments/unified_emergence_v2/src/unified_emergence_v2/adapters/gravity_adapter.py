"""
Gravity domain adapter for the Unified Emergence Framework v2.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..domain.models import EmergenceSignature
from ..domain.protocols import DomainAdapter, TestRunner

logger = logging.getLogger(__name__)


class GravityDomainAdapter:
    """
    Domain adapter for gravitational emergence pattern analysis.
    
    This adapter extracts emergence patterns from gravity simulations,
    focusing on orbital dynamics, energy conservation, and stability metrics.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize gravity domain adapter.
        
        Args:
            test_runner: Test runner for executing gravity simulations
        """
        self.test_runner = test_runner
        self._domain_name = 'gravity'
    
    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self._domain_name
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from gravity simulation results.
        
        Args:
            domain_results: Raw results from gravity simulations
            
        Returns:
            List of emergence signatures representing orbital dynamics patterns
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
                        signature = self._create_orbital_dynamics_signature(metrics, field_size)
                        if signature:
                            signatures.append(signature)
            
            logger.info(f"Extracted {len(signatures)} gravity patterns")
            
        except Exception as e:
            logger.error(f"Error extracting gravity patterns: {e}")
        
        return signatures
    
    def _create_orbital_dynamics_signature(self, metrics: Dict[str, Any], field_size: int) -> EmergenceSignature:
        """Create emergence signature for orbital dynamics."""
        try:
            # Extract key orbital metrics
            orbital_stability = float(metrics.get('orbital_stability', 0.0))
            energy_conservation = float(metrics.get('energy_conservation', 0.0))
            angular_momentum_conservation = float(metrics.get('angular_momentum_conservation', 0.0))
            orbital_eccentricity = float(metrics.get('orbital_eccentricity', 1.0))
            
            # Normalize orbital eccentricity (lower is better, so invert)
            normalized_eccentricity = max(0.0, 1.0 - min(1.0, orbital_eccentricity))
            
            # Create feature vector [stability, energy, momentum, eccentricity]
            features = [
                orbital_stability,
                energy_conservation,
                angular_momentum_conservation,
                normalized_eccentricity
            ]
            
            # Calculate confidence as average of conservation metrics
            confidence = np.mean([orbital_stability, energy_conservation, angular_momentum_conservation])
            
            # Calculate emergence strength based on system complexity and stability
            # Higher field size and stable orbits indicate stronger emergence
            complexity_factor = min(1.0, field_size / 64.0)  # Normalize to [0,1]
            stability_factor = orbital_stability
            emergence_strength = (complexity_factor * 0.3 + stability_factor * 0.7)
            
            # Create metadata
            metadata = {
                'field_size': field_size,
                'orbital_eccentricity': orbital_eccentricity,
                'mean_orbital_radius_au': metrics.get('mean_orbital_radius_au', 0.0),
                'trajectory_points': metrics.get('trajectory_points', 0)
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='orbital_dynamics',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating orbital dynamics signature: {e}")
            return None
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate gravity-specific constraints.
        
        Args:
            domain_data: Gravity simulation data to validate
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        try:
            # Check energy conservation
            energy_conservation = domain_data.get('energy_conservation', 0.0)
            if energy_conservation < 0.9:
                violations.append(f"Energy conservation too low: {energy_conservation:.3f} < 0.9")
            
            # Check angular momentum conservation
            angular_momentum = domain_data.get('angular_momentum_conservation', 0.0)
            if angular_momentum < 0.95:
                violations.append(f"Angular momentum conservation too low: {angular_momentum:.3f} < 0.95")
            
            # Check orbital stability
            orbital_stability = domain_data.get('orbital_stability', 0.0)
            if orbital_stability < 0.8:
                violations.append(f"Orbital stability too low: {orbital_stability:.3f} < 0.8")
            
            # Check orbital eccentricity (should be reasonable for stable orbits)
            eccentricity = domain_data.get('orbital_eccentricity', 1.0)
            if eccentricity > 0.5:
                violations.append(f"Orbital eccentricity too high: {eccentricity:.3f} > 0.5")
            
            # Check trajectory points (should have sufficient data)
            trajectory_points = domain_data.get('trajectory_points', 0)
            if trajectory_points < 100:
                violations.append(f"Insufficient trajectory points: {trajectory_points} < 100")
            
        except Exception as e:
            violations.append(f"Error validating gravity constraints: {e}")
        
        return violations
