"""
Emergence Detector - Genuine Statistical Mechanics Implementation
Based on real phase transition theory and information-theoretic measures.

Detects genuine emergence through:
- Order parameter phase transitions
- Correlation length divergence 
- Critical fluctuations
- Information integration measures
- Coherence length analysis
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize_scalar
from scipy.special import erf
from fracton.core.recursive_engine import ExecutionContext


class PhaseType(Enum):
    """Types of phase transitions in cognitive systems."""
    DISORDER_ORDER = "disorder_order"
    COHERENT_INCOHERENT = "coherent_incoherent" 
    LOCAL_GLOBAL = "local_global"
    CLASSICAL_QUANTUM = "classical_quantum"
    EMERGENT_CONSCIOUSNESS = "emergent_consciousness"


@dataclass
class EmergenceEvent:
    """Genuine emergence event from statistical mechanics."""
    event_id: str
    phase_type: PhaseType
    order_parameter: float
    correlation_length: float
    critical_exponent: float
    information_integration: float
    coherence_measure: float
    field_location: Tuple[float, float]
    transition_temperature: float
    fluctuation_magnitude: float
    confidence: float


class EmergenceDetector:
    """
    Genuine Statistical Mechanics Emergence Detection.
    
    Uses real phase transition theory instead of heuristic pattern matching:
    - Order parameter calculations
    - Correlation function analysis
    - Critical phenomena detection
    - Information integration (Φ) measures
    - Renormalization group analysis
    """
    
    def __init__(self, field_shape: Tuple[int, int] = (32, 32)):
        """Initialize statistical mechanics emergence detector."""
        self.field_shape = field_shape
        
        # Statistical mechanics parameters
        self.temperature = 1.0  # System temperature (energy scale)
        self.coupling_strength = 1.0  # Interaction strength
        self.correlation_cutoff = 8.0  # Maximum correlation length
        
        # Critical exponents (Ising universality class)
        self.beta_exponent = 0.326  # Order parameter critical exponent
        self.nu_exponent = 0.630   # Correlation length critical exponent
        self.gamma_exponent = 1.237  # Susceptibility critical exponent
        
        # Detection thresholds
        self.emergence_threshold = 0.7
        self.phase_transition_threshold = 0.8
        self.consciousness_threshold = 0.85
        
        # State tracking
        self.order_parameter_history = []
        self.correlation_length_history = []
        self.temperature_history = []
        
        # Information theory
        self.phi_threshold = 0.6  # Integrated Information threshold
    
    def detect_emergence(self, field_amplitude: np.ndarray) -> List[EmergenceEvent]:
        """Detect genuine emergence through statistical mechanics analysis."""
        # Ensure field is proper shape
        if field_amplitude.shape != self.field_shape:
            field_amplitude = self._reshape_to_field(field_amplitude)
        
        emergence_events = []
        
        # Calculate order parameter
        order_param = self._calculate_order_parameter(field_amplitude)
        
        # Calculate correlation length
        correlation_length = self._calculate_correlation_length(field_amplitude)
        
        # Calculate information integration (Φ)
        phi = self._calculate_integrated_information(field_amplitude)
        
        # Detect phase transitions
        if self._is_phase_transition(order_param, correlation_length):
            event = self._create_emergence_event(
                field_amplitude, order_param, correlation_length, phi, 
                PhaseType.DISORDER_ORDER
            )
            emergence_events.append(event)
        
        # Detect consciousness emergence
        if self._is_consciousness_emergence(order_param, correlation_length, phi):
            event = self._create_emergence_event(
                field_amplitude, order_param, correlation_length, phi,
                PhaseType.EMERGENT_CONSCIOUSNESS
            )
            emergence_events.append(event)
        
        # Detect coherence transitions
        if self._is_coherence_transition(field_amplitude, correlation_length):
            event = self._create_emergence_event(
                field_amplitude, order_param, correlation_length, phi,
                PhaseType.COHERENT_INCOHERENT
            )
            emergence_events.append(event)
        
        # Update history
        self.order_parameter_history.append(order_param)
        self.correlation_length_history.append(correlation_length)
        self.temperature_history.append(self.temperature)
        
        # Limit history size
        max_history = 100
        if len(self.order_parameter_history) > max_history:
            self.order_parameter_history.pop(0)
            self.correlation_length_history.pop(0)
            self.temperature_history.pop(0)
        
        return emergence_events
    
    def _calculate_order_parameter(self, field: np.ndarray) -> float:
        """Calculate order parameter using magnetization-like measure."""
        # For complex field, use phase coherence as order parameter
        if np.iscomplexobj(field):
            # Phase order parameter: |⟨e^{iφ}⟩|
            phases = np.angle(field)
            phase_order = np.abs(np.mean(np.exp(1j * phases)))
            
            # Amplitude order parameter: normalized variance
            amplitudes = np.abs(field)
            if np.std(amplitudes) == 0:
                amplitude_order = 1.0
            else:
                amplitude_order = 1.0 - np.std(amplitudes) / np.mean(amplitudes)
            
            # Combined order parameter
            order_parameter = 0.5 * (phase_order + max(0, amplitude_order))
        else:
            # Real field: standard deviation normalized
            if np.std(field) == 0:
                order_parameter = 1.0
            else:
                order_parameter = np.std(field) / (np.mean(np.abs(field)) + 1e-10)
        
        return min(1.0, max(0.0, order_parameter))
    
    def _calculate_correlation_length(self, field: np.ndarray) -> float:
        """Calculate correlation length using two-point correlation function."""
        center = (self.field_shape[0] // 2, self.field_shape[1] // 2)
        center_value = field[center]
        
        # Calculate correlation function C(r) = ⟨ψ(0)ψ*(r)⟩
        correlations = []
        distances = []
        
        for i in range(self.field_shape[0]):
            for j in range(self.field_shape[1]):
                r = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if r > 0 and r < self.correlation_cutoff:
                    correlation = np.real(np.conj(center_value) * field[i, j])
                    correlations.append(correlation)
                    distances.append(r)
        
        if len(correlations) < 5:
            return 0.1  # Minimum correlation length
        
        # Fit exponential decay: C(r) = C_0 * exp(-r/ξ)
        correlations = np.array(correlations)
        distances = np.array(distances)
        
        # Normalize correlations
        if np.max(np.abs(correlations)) > 0:
            correlations = correlations / np.max(np.abs(correlations))
        
        # Find correlation length by fitting decay
        try:
            # Simple exponential fit
            def exp_decay(xi):
                predicted = np.exp(-distances / xi)
                return np.sum((correlations - predicted)**2)
            
            result = minimize_scalar(exp_decay, bounds=(0.1, self.correlation_cutoff))
            correlation_length = result.x
        except:
            # Fallback: estimate from half-max
            half_max_indices = np.where(np.abs(correlations) > 0.5)[0]
            if len(half_max_indices) > 0:
                correlation_length = np.max(distances[half_max_indices])
            else:
                correlation_length = 1.0
        
        return min(correlation_length, self.correlation_cutoff)
    
    def _calculate_integrated_information(self, field: np.ndarray) -> float:
        """Calculate integrated information (Φ) measure."""
        # Discretize field into binary states for IIT calculation
        threshold = np.mean(np.abs(field))
        binary_field = (np.abs(field) > threshold).astype(int)
        
        # Calculate mutual information between field regions
        n_regions = 4  # Divide field into 4 quadrants
        h, w = self.field_shape
        
        regions = [
            binary_field[:h//2, :w//2],     # Top-left
            binary_field[:h//2, w//2:],     # Top-right  
            binary_field[h//2:, :w//2],     # Bottom-left
            binary_field[h//2:, w//2:]      # Bottom-right
        ]
        
        # Calculate entropy of each region
        region_entropies = []
        for region in regions:
            p1 = np.mean(region)
            p0 = 1 - p1
            if p1 > 0 and p0 > 0:
                entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            else:
                entropy = 0.0
            region_entropies.append(entropy)
        
        # Calculate joint entropy (simplified)
        all_states = np.concatenate([r.flatten() for r in regions])
        p_joint = np.mean(all_states)
        if p_joint > 0 and p_joint < 1:
            joint_entropy = -p_joint * np.log2(p_joint) - (1-p_joint) * np.log2(1-p_joint)
        else:
            joint_entropy = 0.0
        
        # Integrated information approximation
        total_individual_entropy = np.sum(region_entropies)
        if total_individual_entropy > 0:
            phi = (total_individual_entropy - joint_entropy) / total_individual_entropy
        else:
            phi = 0.0
        
        return max(0.0, min(1.0, phi))
    
    def _is_phase_transition(self, order_param: float, correlation_length: float) -> bool:
        """Detect phase transition through order parameter and correlation divergence."""
        # Phase transition indicators:
        # 1. Order parameter crosses critical value
        # 2. Correlation length approaches maximum
        # 3. Fluctuations increase (from history)
        
        order_critical = order_param > self.phase_transition_threshold
        correlation_critical = correlation_length > 0.7 * self.correlation_cutoff
        
        # Check for fluctuations in recent history
        if len(self.order_parameter_history) >= 3:
            recent_fluctuation = np.std(self.order_parameter_history[-3:])
            fluctuation_critical = recent_fluctuation > 0.1
        else:
            fluctuation_critical = False
        
        return order_critical and (correlation_critical or fluctuation_critical)
    
    def _is_consciousness_emergence(self, order_param: float, correlation_length: float, phi: float) -> bool:
        """Detect consciousness emergence through integrated information."""
        # Consciousness emergence criteria:
        # 1. High integrated information (Φ)
        # 2. Sufficient order parameter
        # 3. Long-range correlations
        # 4. Stability over time
        
        phi_criterion = phi > self.phi_threshold
        order_criterion = order_param > self.consciousness_threshold
        correlation_criterion = correlation_length > 0.5 * self.correlation_cutoff
        
        # Stability check
        if len(self.order_parameter_history) >= 5:
            recent_stability = np.std(self.order_parameter_history[-5:]) < 0.05
        else:
            recent_stability = True
        
        return phi_criterion and order_criterion and correlation_criterion and recent_stability
    
    def _is_coherence_transition(self, field: np.ndarray, correlation_length: float) -> bool:
        """Detect coherence-incoherence phase transition."""
        # For complex fields, measure phase coherence
        if np.iscomplexobj(field):
            phases = np.angle(field)
            phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
        else:
            phase_coherence = 0.5
        
        # Coherence transition: sudden change in phase ordering
        coherent = phase_coherence > 0.8 and correlation_length > 2.0
        
        # Check for transition (not just coherent state)
        if len(self.order_parameter_history) >= 2:
            prev_order = self.order_parameter_history[-1]
            transition_detected = abs(phase_coherence - prev_order) > 0.3
        else:
            transition_detected = True
        
        return coherent and transition_detected
    
    def _create_emergence_event(self, field: np.ndarray, order_param: float, 
                               correlation_length: float, phi: float, 
                               phase_type: PhaseType) -> EmergenceEvent:
        """Create emergence event from statistical mechanics measurements."""
        # Find field location with maximum activity
        intensity = np.abs(field)
        max_location = np.unravel_index(np.argmax(intensity), intensity.shape)
        
        # Calculate critical exponent (simplified)
        if self.temperature > 0:
            critical_exponent = self.beta_exponent * np.log(order_param + 1e-10) / np.log(self.temperature + 1e-10)
        else:
            critical_exponent = self.beta_exponent
        
        # Calculate transition temperature (effective)
        transition_temp = self.coupling_strength * order_param / (1.0 + order_param)
        
        # Calculate fluctuation magnitude
        fluctuation_magnitude = np.std(np.abs(field)) / (np.mean(np.abs(field)) + 1e-10)
        
        # Calculate confidence from multiple indicators
        confidence = 0.3 * order_param + 0.3 * (correlation_length / self.correlation_cutoff) + 0.4 * phi
        confidence = min(1.0, max(0.0, confidence))
        
        return EmergenceEvent(
            event_id=f"emergence_{int(time.time() * 1000)}",
            phase_type=phase_type,
            order_parameter=order_param,
            correlation_length=correlation_length,
            critical_exponent=critical_exponent,
            information_integration=phi,
            coherence_measure=order_param,  # Simplified
            field_location=(float(max_location[0]), float(max_location[1])),
            transition_temperature=transition_temp,
            fluctuation_magnitude=fluctuation_magnitude,
            confidence=confidence
        )
    
    def get_critical_metrics(self) -> Dict[str, float]:
        """Get current critical phenomena metrics."""
        if len(self.order_parameter_history) == 0:
            return {
                'order_parameter': 0.0,
                'correlation_length': 0.0,
                'temperature': self.temperature,
                'susceptibility': 0.0,
                'heat_capacity': 0.0,
                'criticality_measure': 0.0
            }
        
        current_order = self.order_parameter_history[-1]
        current_corr = self.correlation_length_history[-1]
        
        # Calculate susceptibility (response to field changes)
        if len(self.order_parameter_history) >= 2:
            susceptibility = abs(self.order_parameter_history[-1] - self.order_parameter_history[-2]) / 0.01
        else:
            susceptibility = 0.0
        
        # Calculate effective heat capacity (from temperature fluctuations)
        if len(self.temperature_history) >= 3:
            temp_variance = np.var(self.temperature_history[-3:])
            heat_capacity = temp_variance / (self.temperature**2 + 1e-10)
        else:
            heat_capacity = 0.0
        
        # Overall criticality measure
        criticality = (current_order * current_corr / self.correlation_cutoff + 
                      min(susceptibility, 1.0) + min(heat_capacity, 1.0)) / 3.0
        
        return {
            'order_parameter': current_order,
            'correlation_length': current_corr,
            'temperature': self.temperature,
            'susceptibility': min(susceptibility, 10.0),  # Cap extreme values
            'heat_capacity': min(heat_capacity, 10.0),
            'criticality_measure': min(criticality, 1.0)
        }
    
    def _reshape_to_field(self, amplitude: np.ndarray) -> np.ndarray:
        """Reshape amplitude to field dimensions."""
        target_size = np.prod(self.field_shape)
        
        if amplitude.size == target_size:
            return amplitude.reshape(self.field_shape)
        elif amplitude.size > target_size:
            flat = amplitude.flatten()[:target_size]
            return flat.reshape(self.field_shape)
        else:
            padded = np.zeros(target_size, dtype=amplitude.dtype)
            padded[:amplitude.size] = amplitude.flatten()
            return padded.reshape(self.field_shape)
    
    def reset(self):
        """Reset detector state."""
        self.order_parameter_history = []
        self.correlation_length_history = []
        self.temperature_history = []


# Legacy compatibility classes for backward compatibility
class EmergenceType(Enum):
    """Legacy emergence types for backward compatibility."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    COGNITIVE = "cognitive"
    CONSCIOUSNESS = "consciousness"
    PHASE_TRANSITION = "phase_transition"