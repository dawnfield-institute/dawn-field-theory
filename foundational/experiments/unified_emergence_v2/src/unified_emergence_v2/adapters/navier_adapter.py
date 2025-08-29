"""
Navier-Stokes domain adapter for the Unified Emergence Framework v2.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..domain.models import EmergenceSignature
from ..domain.protocols import DomainAdapter, TestRunner

logger = logging.getLogger(__name__)


class NavierDomainAdapter:
    """
    Domain adapter for Navier-Stokes fluid dynamics emergence pattern analysis.
    
    This adapter extracts emergence patterns from fluid dynamics simulations,
    focusing on turbulence detection and flow stability metrics.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize Navier-Stokes domain adapter.
        
        Args:
            test_runner: Test runner for executing fluid dynamics simulations
        """
        self.test_runner = test_runner
        self._domain_name = 'navier'
    
    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self._domain_name
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from Navier-Stokes simulation results.
        
        Args:
            domain_results: Raw results from fluid dynamics simulations
            
        Returns:
            List of emergence signatures representing turbulence patterns
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
                        
                        # Create signature from standalone Navier-Stokes simulation data
                        signature = self._create_fluid_dynamics_signature(metrics, field_size)
                        if signature:
                            signatures.append(signature)
            
            logger.info(f"Extracted {len(signatures)} Navier-Stokes patterns")
            
        except Exception as e:
            logger.error(f"Error extracting Navier-Stokes patterns: {e}")
        
        return signatures
    
    def _create_fluid_dynamics_signature(self, metrics: Dict[str, Any], field_size: int) -> EmergenceSignature:
        """Create emergence signature for fluid dynamics patterns."""
        try:
            # Extract key fluid dynamics metrics from standalone simulator
            reynolds_number = float(metrics.get('reynolds_number', 0.0))
            turbulence_intensity = float(metrics.get('turbulence_intensity', 0.0))
            vorticity_strength = float(metrics.get('vorticity_strength', 0.0))
            pressure_gradient = float(metrics.get('pressure_gradient', 0.0))
            viscosity_ratio = float(metrics.get('viscosity_ratio', 1.0))
            
            # Normalize Reynolds number to [0,1] range for features
            normalized_reynolds = min(1.0, reynolds_number / 10000.0)
            
            # Normalize viscosity ratio (1.0 is ideal, so distance from 1.0)
            normalized_viscosity = 1.0 - abs(viscosity_ratio - 1.0)
            
            # Create feature vector [reynolds, turbulence, vorticity, pressure, viscosity]
            features = [
                normalized_reynolds,
                turbulence_intensity,
                vorticity_strength,
                pressure_gradient,
                normalized_viscosity
            ]
            
            # Calculate confidence based on fluid stability and coherence
            confidence = np.mean([turbulence_intensity, vorticity_strength, normalized_viscosity])
            
            # Calculate emergence strength based on turbulent complexity
            # Higher Reynolds numbers and coherent turbulence patterns indicate emergence
            reynolds_factor = min(1.0, reynolds_number / 5000.0)  # Moderate Reynolds for emergence
            turbulence_factor = turbulence_intensity * vorticity_strength
            emergence_strength = (reynolds_factor * 0.3 + turbulence_factor * 0.7)
            
            # Create metadata
            metadata = {
                'field_size': field_size,
                'reynolds_number': reynolds_number,
                'turbulence_intensity': turbulence_intensity,
                'vorticity_strength': vorticity_strength
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='fluid_dynamics',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating fluid dynamics signature: {e}")
            return None
    
    def _create_turbulence_signature(self, run_data: Dict[str, Any]) -> EmergenceSignature:
        """Create emergence signature for turbulence patterns."""
        try:
            # Extract fluid dynamics metrics
            reynolds_number = float(run_data.get('reynolds_number', 0.0))
            turbulence_accuracy = float(run_data.get('turbulence_detection_accuracy', 0.0))
            grid_size = int(run_data.get('grid_size', 32))
            
            # Additional metrics that might be available
            velocity_divergence = float(run_data.get('velocity_divergence', 0.0))
            pressure_stability = float(run_data.get('pressure_stability', 0.9))
            vorticity_magnitude = float(run_data.get('vorticity_magnitude', 0.5))
            
            # Normalize Reynolds number to [0,1] range (log scale for very high values)
            normalized_reynolds = min(1.0, np.log10(max(1, reynolds_number)) / 6.0)  # Log scale up to 10^6
            
            # Normalize velocity divergence (lower is better for incompressible flow)
            normalized_divergence = max(0.0, 1.0 - min(1.0, abs(velocity_divergence)))
            
            # Normalize vorticity magnitude
            normalized_vorticity = min(1.0, vorticity_magnitude)
            
            # Create feature vector [reynolds, turbulence_acc, divergence, vorticity]
            features = [
                normalized_reynolds,
                turbulence_accuracy,
                normalized_divergence,
                normalized_vorticity
            ]
            
            # Confidence based on turbulence detection accuracy and pressure stability
            confidence = (turbulence_accuracy * 0.7 + pressure_stability * 0.3)
            
            # Emergence strength based on complexity and flow stability
            grid_complexity = min(1.0, grid_size / 64.0)
            flow_stability = (pressure_stability + normalized_divergence) / 2.0
            emergence_strength = (turbulence_accuracy * 0.4 + flow_stability * 0.4 + grid_complexity * 0.2)
            
            # Metadata
            metadata = {
                'reynolds_number': reynolds_number,
                'turbulence_detection_accuracy': turbulence_accuracy,
                'grid_size': grid_size,
                'velocity_divergence': velocity_divergence,
                'pressure_stability': pressure_stability,
                'vorticity_magnitude': vorticity_magnitude,
                'flow_regime': self._classify_flow_regime(reynolds_number)
            }
            
            return EmergenceSignature(
                domain=self._domain_name,
                pattern_type='turbulence_dynamics',
                features=features,
                confidence=confidence,
                emergence_strength=emergence_strength,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating turbulence signature: {e}")
            return None
    
    def _classify_flow_regime(self, reynolds_number: float) -> str:
        """Classify flow regime based on Reynolds number."""
        if reynolds_number < 2300:
            return 'laminar'
        elif reynolds_number < 4000:
            return 'transitional'
        else:
            return 'turbulent'
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate Navier-Stokes specific constraints.
        
        Args:
            domain_data: Fluid dynamics simulation data to validate
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        try:
            # Check Reynolds number range
            reynolds_number = domain_data.get('reynolds_number', 0.0)
            if reynolds_number <= 0:
                violations.append(f"Invalid Reynolds number: {reynolds_number} <= 0")
            elif reynolds_number > 1e6:
                violations.append(f"Reynolds number too high: {reynolds_number} > 1e6")
            
            # Check turbulence detection accuracy
            turbulence_accuracy = domain_data.get('turbulence_detection_accuracy', 0.0)
            if turbulence_accuracy < 0.7:
                violations.append(f"Turbulence detection accuracy too low: {turbulence_accuracy:.3f} < 0.7")
            
            # Check velocity divergence (should be near zero for incompressible flow)
            velocity_divergence = domain_data.get('velocity_divergence', 0.0)
            if abs(velocity_divergence) > 0.1:
                violations.append(f"Velocity divergence too high: {abs(velocity_divergence):.3f} > 0.1")
            
            # Check pressure stability
            pressure_stability = domain_data.get('pressure_stability', 1.0)
            if pressure_stability < 0.8:
                violations.append(f"Pressure stability too low: {pressure_stability:.3f} < 0.8")
            
            # Check grid size
            grid_size = domain_data.get('grid_size', 32)
            if grid_size < 16:
                violations.append(f"Grid size too small: {grid_size} < 16")
            elif grid_size > 256:
                violations.append(f"Grid size too large: {grid_size} > 256")
            
            # Check flow regime consistency
            if reynolds_number > 4000 and turbulence_accuracy < 0.8:
                violations.append("High Reynolds number but low turbulence detection accuracy")
            
        except Exception as e:
            violations.append(f"Error validating Navier-Stokes constraints: {e}")
        
        return violations
