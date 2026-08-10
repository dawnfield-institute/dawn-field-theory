"""
Advanced Möbius topology implementation with detailed analysis capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TopologyMetrics:
    """Metrics describing topological properties"""
    twist_strength: float
    curvature_variance: float
    boundary_continuity: float
    field_coherence: float
    anti_periodic_quality: float

class MobiusTopology:
    """
    Advanced Möbius topology implementation with anti-periodic boundary conditions
    
    This serves as the computational substrate for Pre-Field Recursion experiments.
    The Möbius structure provides the geometric foundation for PAC conservation
    and natural amplification emergence.
    """
    
    def __init__(self, size: int, twist_strength: float = 1.0, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            
        self.size = size
        self.twist_strength = twist_strength
        self.field = self._initialize_mobius_field()
        self.boundary_conditions = "anti_periodic"
        self._cached_metrics = None
        
    def _initialize_mobius_field(self) -> np.ndarray:
        """
        Initialize field with proper Möbius topology structure
        
        The field satisfies: f(x + period/2) = -f(x) * twist_strength
        """
        # Start with random field
        base_field = np.random.random(self.size) * 2 - 1  # [-1, 1]
        
        # Apply Möbius twist constraint
        half_size = self.size // 2
        
        for i in range(half_size):
            opposite_idx = (i + half_size) % self.size
            # Enforce anti-periodic: f(x + π) = -f(x)
            base_field[opposite_idx] = -self.twist_strength * base_field[i]
            
        # Smooth transitions to maintain continuity
        base_field = self._smooth_field(base_field)
        
        return base_field
    
    def _smooth_field(self, field: np.ndarray, smoothing_passes: int = 3) -> np.ndarray:
        """Apply smoothing while preserving Möbius structure"""
        smoothed = field.copy()
        
        for _ in range(smoothing_passes):
            new_field = smoothed.copy()
            for i in range(self.size):
                neighbors = [
                    smoothed[(i - 1) % self.size],
                    smoothed[i],
                    smoothed[(i + 1) % self.size]
                ]
                new_field[i] = 0.6 * smoothed[i] + 0.2 * (neighbors[0] + neighbors[2])
            
            # Re-enforce Möbius constraint after smoothing
            half_size = self.size // 2
            for i in range(half_size):
                opposite_idx = (i + half_size) % self.size
                # Average the constraint enforcement
                constraint_value = -self.twist_strength * new_field[i]
                new_field[opposite_idx] = 0.8 * new_field[opposite_idx] + 0.2 * constraint_value
                
            smoothed = new_field
            
        return smoothed
    
    def get_local_structure(self, center: int, radius: int = 3) -> Dict:
        """
        Extract detailed local topological structure around a point
        
        Returns comprehensive analysis of local geometry
        """
        indices = []
        values = []
        
        for i in range(-radius, radius + 1):
            idx = (center + i) % self.size
            indices.append(idx)
            values.append(self.field[idx])
            
        local_array = np.array(values)
        
        # Calculate local properties
        local_curvature = self._calculate_local_curvature(local_array)
        local_twist = self._calculate_local_twist(center)
        local_gradient = np.gradient(local_array)
        
        return {
            'indices': indices,
            'values': local_array,
            'center_value': self.field[center],
            'curvature': local_curvature,
            'twist_measure': local_twist,
            'gradient': local_gradient,
            'field_variance': np.var(local_array),
            'coherence': self._calculate_local_coherence(local_array)
        }
    
    def _calculate_local_curvature(self, local_field: np.ndarray) -> float:
        """Calculate local curvature from field values"""
        if len(local_field) < 3:
            return 0.0
            
        # Second derivative approximation
        second_deriv = np.diff(local_field, n=2)
        return np.mean(np.abs(second_deriv))
    
    def _calculate_local_twist(self, center: int) -> float:
        """Calculate local twist strength around a point"""
        half_size = self.size // 2
        opposite_idx = (center + half_size) % self.size
        
        # Measure deviation from perfect anti-periodic condition
        expected_opposite = -self.twist_strength * self.field[center]
        actual_opposite = self.field[opposite_idx]
        
        twist_error = abs(expected_opposite - actual_opposite)
        return 1.0 - min(twist_error, 1.0)  # Normalized twist quality
    
    def _calculate_local_coherence(self, local_field: np.ndarray) -> float:
        """Calculate field coherence in local region"""
        if len(local_field) < 2:
            return 1.0
            
        # Measure smoothness vs randomness
        gradient_variance = np.var(np.gradient(local_field))
        field_magnitude = np.mean(np.abs(local_field))
        
        if field_magnitude < 1e-10:
            return 0.0
            
        coherence = 1.0 / (1.0 + gradient_variance / field_magnitude)
        return coherence
    
    def verify_mobius_structure(self) -> Dict:
        """Verify that the field satisfies Möbius topology requirements"""
        half_size = self.size // 2
        anti_periodic_errors = []
        
        for i in range(half_size):
            opposite_idx = (i + half_size) % self.size
            expected = -self.twist_strength * self.field[i]
            actual = self.field[opposite_idx]
            error = abs(expected - actual)
            anti_periodic_errors.append(error)
            
        mean_error = np.mean(anti_periodic_errors)
        max_error = np.max(anti_periodic_errors)
        
        # Calculate boundary continuity
        boundary_jump = abs(self.field[-1] + self.twist_strength * self.field[0])
        
        return {
            'anti_periodic_mean_error': mean_error,
            'anti_periodic_max_error': max_error,
            'anti_periodic_quality': max(0, 1 - mean_error),
            'boundary_continuity': max(0, 1 - boundary_jump),
            'structure_valid': mean_error < 0.1 and boundary_jump < 0.1,
            'twist_strength': self.twist_strength,
            'field_magnitude': np.mean(np.abs(self.field))
        }

class TopologyAnalyzer:
    """
    Analyzer for topological properties and their effects on amplification
    """
    
    def __init__(self, topology: MobiusTopology):
        self.topology = topology
        
    def analyze_amplification_correlation(self, amplification_measurements: List[float]) -> Dict:
        """
        Analyze correlation between topology and amplification measurements
        
        This helps understand how topological structure influences local amplification
        """
        if not amplification_measurements:
            return {'correlation': 0.0, 'analysis': 'No measurements provided'}
            
        # Extract topological features
        curvatures = []
        twists = []
        coherences = []
        
        for i in range(self.topology.size):
            local_struct = self.topology.get_local_structure(i)
            curvatures.append(local_struct['curvature'])
            twists.append(local_struct['twist_measure'])
            coherences.append(local_struct['coherence'])
            
        topo_features = {
            'mean_curvature': np.mean(curvatures),
            'curvature_variance': np.var(curvatures),
            'mean_twist': np.mean(twists),
            'twist_variance': np.var(twists),
            'mean_coherence': np.mean(coherences),
            'coherence_variance': np.var(coherences)
        }
        
        # Analyze amplification characteristics
        amp_stats = {
            'mean_amplification': np.mean(amplification_measurements),
            'amplification_variance': np.var(amplification_measurements),
            'amplification_range': (np.min(amplification_measurements), 
                                  np.max(amplification_measurements))
        }
        
        # Look for correlations
        correlations = {}
        if len(amplification_measurements) > 1:
            # Simplified correlation analysis
            # In real implementation, would use proper statistical methods
            amp_normalized = (amplification_measurements - amp_stats['mean_amplification']) / max(amp_stats['amplification_variance'], 1e-10)
            
            correlations = {
                'curvature_correlation': self._estimate_correlation(curvatures[:len(amplification_measurements)], amp_normalized),
                'twist_correlation': self._estimate_correlation(twists[:len(amplification_measurements)], amp_normalized),
                'coherence_correlation': self._estimate_correlation(coherences[:len(amplification_measurements)], amp_normalized)
            }
        
        return {
            'topology_features': topo_features,
            'amplification_stats': amp_stats,
            'correlations': correlations,
            'analysis': self._generate_analysis_summary(topo_features, amp_stats, correlations)
        }
    
    def _estimate_correlation(self, topo_values: List[float], amp_values: List[float]) -> float:
        """Estimate correlation between topology and amplification"""
        if len(topo_values) != len(amp_values) or len(topo_values) < 2:
            return 0.0
            
        topo_array = np.array(topo_values)
        amp_array = np.array(amp_values)
        
        # Normalize
        topo_norm = (topo_array - np.mean(topo_array)) / max(np.std(topo_array), 1e-10)
        amp_norm = (amp_array - np.mean(amp_array)) / max(np.std(amp_array), 1e-10)
        
        # Simple correlation coefficient
        correlation = np.mean(topo_norm * amp_norm)
        return np.clip(correlation, -1.0, 1.0)
    
    def _generate_analysis_summary(self, topo_features: Dict, amp_stats: Dict, correlations: Dict) -> str:
        """Generate human-readable analysis summary"""
        summary_parts = []
        
        # Topology description
        summary_parts.append(f"Topology shows mean curvature {topo_features['mean_curvature']:.3f} with variance {topo_features['curvature_variance']:.3f}")
        summary_parts.append(f"Twist quality averages {topo_features['mean_twist']:.3f}")
        summary_parts.append(f"Field coherence averages {topo_features['mean_coherence']:.3f}")
        
        # Amplification description
        summary_parts.append(f"Amplification ranges from {amp_stats['amplification_range'][0]:.1f}x to {amp_stats['amplification_range'][1]:.1f}x")
        summary_parts.append(f"Mean amplification: {amp_stats['mean_amplification']:.1f}x")
        
        # Correlation insights
        if correlations:
            strongest_corr = max(correlations.items(), key=lambda x: abs(x[1]))
            summary_parts.append(f"Strongest topology correlation: {strongest_corr[0]} ({strongest_corr[1]:.3f})")
        
        return ". ".join(summary_parts) + "."
    
    def get_topology_metrics(self) -> TopologyMetrics:
        """Get comprehensive topology metrics"""
        verification = self.topology.verify_mobius_structure()
        
        # Calculate additional metrics
        field = self.topology.field
        curvature_variance = np.var([self.topology._calculate_local_curvature(
            field[max(0, i-2):min(len(field), i+3)]
        ) for i in range(len(field))])
        
        field_coherence = np.mean([self.topology._calculate_local_coherence(
            field[max(0, i-2):min(len(field), i+3)]
        ) for i in range(len(field))])
        
        return TopologyMetrics(
            twist_strength=self.topology.twist_strength,
            curvature_variance=curvature_variance,
            boundary_continuity=verification['boundary_continuity'],
            field_coherence=field_coherence,
            anti_periodic_quality=verification['anti_periodic_quality']
        )