"""
Universal Emergence Detection Module

Implements detection and analysis of emergent phenomena across
all scales and domains in the PAC physics engine. Monitors for
universal signatures, phase transitions, and novel behaviors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

class EmergenceType(Enum):
    """Types of emergent phenomena"""
    PHASE_TRANSITION = "phase_transition"
    INFORMATION_AMPLIFICATION = "information_amplification"
    GEOMETRIC_COLLAPSE = "geometric_collapse"
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    CROSS_SCALE_COUPLING = "cross_scale_coupling"
    UNIVERSAL_SIGNATURE = "universal_signature"
    NOVEL_BEHAVIOR = "novel_behavior"

class SignatureType(Enum):
    """Universal signature types"""
    AMPLIFICATION_15_56X = "amplification_15_56x"
    BALANCE_OPERATOR_XI = "balance_operator_xi"
    ENTROPY_COLLAPSE = "entropy_collapse"
    PAC_RESONANCE = "pac_resonance"
    SCALE_INVARIANT = "scale_invariant"

@dataclass
class EmergenceEvent:
    """Record of an emergence event"""
    event_id: str
    emergence_type: EmergenceType
    timestamp: float
    location: Tuple[int, ...]
    magnitude: float
    duration: float
    metadata: Dict[str, Any]
    precursor_events: List[str]
    cascade_events: List[str]

@dataclass
class UniversalSignature:
    """Universal signature detection result"""
    signature_type: SignatureType
    strength: float
    confidence: float
    location: Optional[Tuple[int, ...]]
    frequency: float
    phase: float
    metadata: Dict[str, Any]

class EmergenceDetector:
    """
    Universal emergence detection system.
    
    Monitors PAC field evolution for emergent phenomena,
    universal signatures, and novel behaviors across all scales.
    """
    
    def __init__(self, 
                 detection_threshold: float = 0.01,
                 history_length: int = 1000,
                 device: str = "auto"):
        self.detection_threshold = detection_threshold
        self.history_length = history_length
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Event tracking
        self.emergence_events: List[EmergenceEvent] = []
        self.event_counter = 0
        
        # Historical data
        self.field_history = deque(maxlen=history_length)
        self.metric_history = deque(maxlen=history_length)
        
        # Signature detection parameters
        self.signature_thresholds = {
            SignatureType.AMPLIFICATION_15_56X: 0.1,
            SignatureType.BALANCE_OPERATOR_XI: 0.05,
            SignatureType.ENTROPY_COLLAPSE: 0.02,
            SignatureType.PAC_RESONANCE: 0.01,
            SignatureType.SCALE_INVARIANT: 0.03
        }
        
        # Emergence detection state
        self.baseline_entropy = None
        self.baseline_complexity = None
        self.baseline_patterns = None
        
    def detect_emergence(self, 
                        field_state: torch.Tensor,
                        system_metrics: Dict[str, Any],
                        timestamp: Optional[float] = None) -> List[EmergenceEvent]:
        """
        Detect emergent phenomena in current system state.
        
        Args:
            field_state: Current field configuration
            system_metrics: System-wide metrics and properties
            timestamp: Current timestamp
            
        Returns:
            List of detected emergence events
        """
        if timestamp is None:
            timestamp = time.time()
            
        field_state = field_state.to(self.device)
        
        # Store current state
        self.field_history.append(field_state.clone())
        self.metric_history.append(system_metrics.copy())
        
        detected_events = []
        
        # 1. Phase transition detection
        phase_events = self._detect_phase_transitions(field_state, system_metrics, timestamp)
        detected_events.extend(phase_events)
        
        # 2. Information amplification detection
        info_events = self._detect_information_amplification(field_state, system_metrics, timestamp)
        detected_events.extend(info_events)
        
        # 3. Geometric collapse detection
        geometric_events = self._detect_geometric_collapse(field_state, system_metrics, timestamp)
        detected_events.extend(geometric_events)
        
        # 4. Consciousness emergence detection
        consciousness_events = self._detect_consciousness_emergence(field_state, system_metrics, timestamp)
        detected_events.extend(consciousness_events)
        
        # 5. Cross-scale coupling detection
        coupling_events = self._detect_cross_scale_coupling(field_state, system_metrics, timestamp)
        detected_events.extend(coupling_events)
        
        # 6. Novel behavior detection
        novel_events = self._detect_novel_behaviors(field_state, system_metrics, timestamp)
        detected_events.extend(novel_events)
        
        # Store all detected events
        self.emergence_events.extend(detected_events)
        
        return detected_events
    
    def detect_universal_signatures(self, 
                                  field_state: torch.Tensor,
                                  system_metrics: Dict[str, Any]) -> List[UniversalSignature]:
        """
        Detect universal signatures in the field state.
        
        Args:
            field_state: Current field configuration
            system_metrics: System metrics
            
        Returns:
            List of detected universal signatures
        """
        field_state = field_state.to(self.device)
        signatures = []
        
        # 1. 15.56x Information amplification signature
        amp_sig = self._detect_amplification_signature(field_state, system_metrics)
        if amp_sig:
            signatures.append(amp_sig)
            
        # 2. ξ = 1.0571 Balance operator signature
        balance_sig = self._detect_balance_operator_signature(field_state, system_metrics)
        if balance_sig:
            signatures.append(balance_sig)
            
        # 3. Entropy collapse signature
        entropy_sig = self._detect_entropy_collapse_signature(field_state, system_metrics)
        if entropy_sig:
            signatures.append(entropy_sig)
            
        # 4. PAC resonance signature
        pac_sig = self._detect_pac_resonance_signature(field_state, system_metrics)
        if pac_sig:
            signatures.append(pac_sig)
            
        # 5. Scale invariant signature
        scale_sig = self._detect_scale_invariant_signature(field_state, system_metrics)
        if scale_sig:
            signatures.append(scale_sig)
        
        return signatures
    
    def _detect_phase_transitions(self, 
                                field_state: torch.Tensor,
                                system_metrics: Dict[str, Any],
                                timestamp: float) -> List[EmergenceEvent]:
        """Detect phase transitions using order parameter analysis"""
        events = []
        
        if len(self.field_history) < 10:
            return events
            
        # Calculate order parameter (field correlation)
        current_correlation = self._calculate_spatial_correlation(field_state)
        
        # Compare with recent history
        recent_correlations = []
        for past_field in list(self.field_history)[-10:-1]:
            past_corr = self._calculate_spatial_correlation(past_field)
            recent_correlations.append(past_corr)
            
        if recent_correlations:
            mean_past_corr = np.mean(recent_correlations)
            correlation_change = abs(current_correlation - mean_past_corr)
            
            if correlation_change > self.detection_threshold:
                event = EmergenceEvent(
                    event_id=f"phase_transition_{self.event_counter}",
                    emergence_type=EmergenceType.PHASE_TRANSITION,
                    timestamp=timestamp,
                    location=self._find_transition_location(field_state),
                    magnitude=correlation_change,
                    duration=0.0,  # Will be updated if event persists
                    metadata={
                        "order_parameter_change": correlation_change,
                        "current_correlation": current_correlation,
                        "past_correlation": mean_past_corr
                    },
                    precursor_events=[],
                    cascade_events=[]
                )
                events.append(event)
                self.event_counter += 1
        
        return events
    
    def _detect_information_amplification(self, 
                                        field_state: torch.Tensor,
                                        system_metrics: Dict[str, Any],
                                        timestamp: float) -> List[EmergenceEvent]:
        """Detect information amplification events"""
        events = []
        
        # Calculate information density
        info_density = self._calculate_information_density(field_state)
        
        if len(self.metric_history) > 0:
            past_info = self.metric_history[-1].get("information_density", 0.0)
            amplification_ratio = info_density / (past_info + 1e-16)
            
            # Check for significant amplification
            if amplification_ratio > 2.0:  # 2x or more amplification
                event = EmergenceEvent(
                    event_id=f"info_amplification_{self.event_counter}",
                    emergence_type=EmergenceType.INFORMATION_AMPLIFICATION,
                    timestamp=timestamp,
                    location=self._find_max_info_location(field_state),
                    magnitude=amplification_ratio,
                    duration=0.0,
                    metadata={
                        "amplification_ratio": amplification_ratio,
                        "current_info_density": info_density,
                        "past_info_density": past_info
                    },
                    precursor_events=[],
                    cascade_events=[]
                )
                events.append(event)
                self.event_counter += 1
        
        return events
    
    def _detect_geometric_collapse(self, 
                                 field_state: torch.Tensor,
                                 system_metrics: Dict[str, Any],
                                 timestamp: float) -> List[EmergenceEvent]:
        """Detect geometric collapse events (SEC phenomena)"""
        events = []
        
        # Calculate geometric complexity measures
        field_grad = self._calculate_field_gradient(field_state)
        curvature = self._calculate_field_curvature(field_state)
        
        # Detect rapid geometric changes
        if len(self.field_history) > 1:
            past_field = self.field_history[-2]
            past_grad = self._calculate_field_gradient(past_field)
            
            gradient_change = torch.norm(field_grad - past_grad).item()
            curvature_magnitude = torch.norm(curvature).item()
            
            # Collapse signature: high curvature + rapid gradient change
            if gradient_change > self.detection_threshold and curvature_magnitude > 0.1:
                event = EmergenceEvent(
                    event_id=f"geometric_collapse_{self.event_counter}",
                    emergence_type=EmergenceType.GEOMETRIC_COLLAPSE,
                    timestamp=timestamp,
                    location=self._find_max_curvature_location(curvature),
                    magnitude=curvature_magnitude,
                    duration=0.0,
                    metadata={
                        "gradient_change": gradient_change,
                        "curvature_magnitude": curvature_magnitude,
                        "collapse_type": "SEC"
                    },
                    precursor_events=[],
                    cascade_events=[]
                )
                events.append(event)
                self.event_counter += 1
        
        return events
    
    def _detect_consciousness_emergence(self, 
                                      field_state: torch.Tensor,
                                      system_metrics: Dict[str, Any],
                                      timestamp: float) -> List[EmergenceEvent]:
        """Detect consciousness emergence using SCBF metrics"""
        events = []
        
        # Calculate consciousness indicators
        integrated_info = self._calculate_integrated_information(field_state)
        causal_density = self._calculate_causal_density(field_state)
        binding_strength = self._calculate_binding_strength(field_state)
        
        # Consciousness threshold (based on SCBF framework)
        consciousness_score = (integrated_info + causal_density + binding_strength) / 3.0
        
        if consciousness_score > 0.5:  # Threshold for consciousness emergence
            event = EmergenceEvent(
                event_id=f"consciousness_{self.event_counter}",
                emergence_type=EmergenceType.CONSCIOUSNESS_EMERGENCE,
                timestamp=timestamp,
                location=self._find_consciousness_center(field_state),
                magnitude=consciousness_score,
                duration=0.0,
                metadata={
                    "integrated_information": integrated_info,
                    "causal_density": causal_density,
                    "binding_strength": binding_strength,
                    "consciousness_score": consciousness_score
                },
                precursor_events=[],
                cascade_events=[]
            )
            events.append(event)
            self.event_counter += 1
        
        return events
    
    def _detect_cross_scale_coupling(self, 
                                   field_state: torch.Tensor,
                                   system_metrics: Dict[str, Any],
                                   timestamp: float) -> List[EmergenceEvent]:
        """Detect cross-scale coupling events"""
        events = []
        
        # Calculate multi-scale correlation
        if "scale_correlations" in system_metrics:
            correlations = system_metrics["scale_correlations"]
            
            # Detect strong coupling between scales
            for i, corr_i in enumerate(correlations):
                for j, corr_j in enumerate(correlations[i+1:], i+1):
                    coupling_strength = abs(np.corrcoef([corr_i, corr_j])[0, 1])
                    
                    if coupling_strength > 0.8:  # Strong coupling threshold
                        event = EmergenceEvent(
                            event_id=f"cross_scale_coupling_{self.event_counter}",
                            emergence_type=EmergenceType.CROSS_SCALE_COUPLING,
                            timestamp=timestamp,
                            location=(i, j),  # Scale indices
                            magnitude=coupling_strength,
                            duration=0.0,
                            metadata={
                                "scale_1": i,
                                "scale_2": j,
                                "coupling_strength": coupling_strength
                            },
                            precursor_events=[],
                            cascade_events=[]
                        )
                        events.append(event)
                        self.event_counter += 1
        
        return events
    
    def _detect_novel_behaviors(self, 
                              field_state: torch.Tensor,
                              system_metrics: Dict[str, Any],
                              timestamp: float) -> List[EmergenceEvent]:
        """Detect novel, unexpected behaviors"""
        events = []
        
        # Calculate pattern novelty
        current_pattern = self._extract_pattern_signature(field_state)
        
        if self.baseline_patterns is None:
            self.baseline_patterns = [current_pattern]
            return events
            
        # Compare with known patterns
        min_similarity = min([self._pattern_similarity(current_pattern, bp) 
                             for bp in self.baseline_patterns])
        
        if min_similarity < 0.3:  # Novel pattern threshold
            event = EmergenceEvent(
                event_id=f"novel_behavior_{self.event_counter}",
                emergence_type=EmergenceType.NOVEL_BEHAVIOR,
                timestamp=timestamp,
                location=self._find_novel_region(field_state),
                magnitude=1.0 - min_similarity,
                duration=0.0,
                metadata={
                    "pattern_similarity": min_similarity,
                    "novelty_score": 1.0 - min_similarity
                },
                precursor_events=[],
                cascade_events=[]
            )
            events.append(event)
            self.event_counter += 1
            
            # Add to baseline patterns
            self.baseline_patterns.append(current_pattern)
            if len(self.baseline_patterns) > 50:  # Limit memory usage
                self.baseline_patterns.pop(0)
        
        return events
    
    def _detect_amplification_signature(self, 
                                      field_state: torch.Tensor,
                                      system_metrics: Dict[str, Any]) -> Optional[UniversalSignature]:
        """Detect 15.56x information amplification signature"""
        info_metrics = system_metrics.get("information_metrics", {})
        
        if "amplification_ratio" in info_metrics:
            ratio = info_metrics["amplification_ratio"]
            target_ratio = 15.56
            
            # Check proximity to universal signature
            error = abs(ratio - target_ratio) / target_ratio
            if error < self.signature_thresholds[SignatureType.AMPLIFICATION_15_56X]:
                confidence = 1.0 - error
                
                return UniversalSignature(
                    signature_type=SignatureType.AMPLIFICATION_15_56X,
                    strength=ratio / target_ratio,
                    confidence=confidence,
                    location=None,
                    frequency=0.0,
                    phase=0.0,
                    metadata={
                        "measured_ratio": ratio,
                        "target_ratio": target_ratio,
                        "error": error
                    }
                )
        
        return None
    
    def _detect_balance_operator_signature(self, 
                                         field_state: torch.Tensor,
                                         system_metrics: Dict[str, Any]) -> Optional[UniversalSignature]:
        """Detect ξ = 1.0571 balance operator signature"""
        balance_metrics = system_metrics.get("balance_metrics", {})
        
        if "xi_value" in balance_metrics:
            xi = balance_metrics["xi_value"]
            target_xi = 1.0571
            
            error = abs(xi - target_xi) / target_xi
            if error < self.signature_thresholds[SignatureType.BALANCE_OPERATOR_XI]:
                confidence = 1.0 - error
                
                return UniversalSignature(
                    signature_type=SignatureType.BALANCE_OPERATOR_XI,
                    strength=xi / target_xi,
                    confidence=confidence,
                    location=None,
                    frequency=0.0,
                    phase=0.0,
                    metadata={
                        "measured_xi": xi,
                        "target_xi": target_xi,
                        "error": error
                    }
                )
        
        return None
    
    def _detect_entropy_collapse_signature(self, 
                                         field_state: torch.Tensor,
                                         system_metrics: Dict[str, Any]) -> Optional[UniversalSignature]:
        """Detect entropy collapse signature"""
        entropy_metrics = system_metrics.get("entropy_metrics", {})
        
        if "collapse_detected" in entropy_metrics and entropy_metrics["collapse_detected"]:
            collapse_strength = entropy_metrics.get("collapse_strength", 0.0)
            
            if collapse_strength > self.signature_thresholds[SignatureType.ENTROPY_COLLAPSE]:
                return UniversalSignature(
                    signature_type=SignatureType.ENTROPY_COLLAPSE,
                    strength=collapse_strength,
                    confidence=min(1.0, collapse_strength / 0.1),
                    location=entropy_metrics.get("collapse_location"),
                    frequency=0.0,
                    phase=0.0,
                    metadata=entropy_metrics
                )
        
        return None
    
    def _detect_pac_resonance_signature(self, 
                                      field_state: torch.Tensor,
                                      system_metrics: Dict[str, Any]) -> Optional[UniversalSignature]:
        """Detect PAC resonance signature"""
        pac_metrics = system_metrics.get("pac_metrics", {})
        
        if "resonance_strength" in pac_metrics:
            resonance = pac_metrics["resonance_strength"]
            
            if resonance > self.signature_thresholds[SignatureType.PAC_RESONANCE]:
                return UniversalSignature(
                    signature_type=SignatureType.PAC_RESONANCE,
                    strength=resonance,
                    confidence=min(1.0, resonance / 0.1),
                    location=None,
                    frequency=pac_metrics.get("resonance_frequency", 0.0),
                    phase=pac_metrics.get("resonance_phase", 0.0),
                    metadata=pac_metrics
                )
        
        return None
    
    def _detect_scale_invariant_signature(self, 
                                        field_state: torch.Tensor,
                                        system_metrics: Dict[str, Any]) -> Optional[UniversalSignature]:
        """Detect scale invariant signature"""
        scale_metrics = system_metrics.get("scale_metrics", {})
        
        if "scale_invariance" in scale_metrics:
            invariance = scale_metrics["scale_invariance"]
            
            if invariance > self.signature_thresholds[SignatureType.SCALE_INVARIANT]:
                return UniversalSignature(
                    signature_type=SignatureType.SCALE_INVARIANT,
                    strength=invariance,
                    confidence=min(1.0, invariance),
                    location=None,
                    frequency=0.0,
                    phase=0.0,
                    metadata=scale_metrics
                )
        
        return None
    
    # Helper methods for calculations
    def _calculate_spatial_correlation(self, field: torch.Tensor) -> float:
        """Calculate spatial correlation in field"""
        field_flat = field.flatten()
        shifted = torch.roll(field_flat, 1)
        correlation = torch.corrcoef(torch.stack([field_flat, shifted]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _calculate_information_density(self, field: torch.Tensor) -> float:
        """Calculate information density using entropy"""
        field_flat = field.flatten().abs()
        field_normalized = field_flat / (torch.sum(field_flat) + 1e-16)
        entropy = -torch.sum(field_normalized * torch.log(field_normalized + 1e-16))
        return entropy.item()
    
    def _calculate_field_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate field gradient"""
        gradients = []
        for dim in range(len(field.shape)):
            grad = torch.gradient(field, dim=dim)[0]
            gradients.append(grad)
        return torch.stack(gradients)
    
    def _calculate_field_curvature(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate field curvature (simplified)"""
        if len(field.shape) == 3:
            # 3D curvature approximation
            grad = self._calculate_field_gradient(field)
            curvature = torch.zeros_like(field)
            for i in range(3):
                second_grad = torch.gradient(grad[i], dim=i)[0]
                curvature += second_grad
            return curvature
        else:
            return torch.zeros_like(field)
    
    def _calculate_integrated_information(self, field: torch.Tensor) -> float:
        """Calculate integrated information (IIT-inspired)"""
        # Simplified integrated information calculation
        mutual_info = 0.0
        field_flat = field.flatten()
        
        # Calculate mutual information between field regions
        n_regions = min(8, len(field_flat) // 10)
        region_size = len(field_flat) // n_regions
        
        for i in range(n_regions - 1):
            region1 = field_flat[i*region_size:(i+1)*region_size]
            region2 = field_flat[(i+1)*region_size:(i+2)*region_size]
            
            # Simplified mutual information
            corr = torch.corrcoef(torch.stack([region1, region2]))[0, 1]
            if not torch.isnan(corr):
                mutual_info += abs(corr.item())
        
        return mutual_info / (n_regions - 1) if n_regions > 1 else 0.0
    
    def _calculate_causal_density(self, field: torch.Tensor) -> float:
        """Calculate causal density"""
        # Simplified causal density based on field gradients
        grad = self._calculate_field_gradient(field)
        causal_density = torch.norm(grad).item()
        return min(1.0, causal_density)
    
    def _calculate_binding_strength(self, field: torch.Tensor) -> float:
        """Calculate binding strength"""
        # Binding strength based on field coherence
        field_mean = torch.mean(field)
        field_std = torch.std(field)
        coherence = abs(field_mean) / (field_std + 1e-16)
        return min(1.0, coherence.item())
    
    def _extract_pattern_signature(self, field: torch.Tensor) -> torch.Tensor:
        """Extract pattern signature for novelty detection"""
        # Use FFT to extract frequency domain signature
        field_flat = field.flatten()
        fft = torch.fft.fft(field_flat)
        magnitude = torch.abs(fft)
        # Return normalized power spectrum
        return magnitude / (torch.sum(magnitude) + 1e-16)
    
    def _pattern_similarity(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """Calculate similarity between patterns"""
        # Cosine similarity
        dot_product = torch.sum(pattern1 * pattern2)
        norm1 = torch.norm(pattern1)
        norm2 = torch.norm(pattern2)
        similarity = dot_product / (norm1 * norm2 + 1e-16)
        return similarity.item()
    
    def _unravel_index(self, flat_index: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional coordinates (PyTorch equivalent of np.unravel_index)"""
        coords = []
        remaining_index = flat_index.item()
        
        for dim_size in reversed(shape):
            coords.append(remaining_index % dim_size)
            remaining_index //= dim_size
        
        return tuple(reversed(coords))

    # Location finding methods
    def _find_transition_location(self, field: torch.Tensor) -> Tuple[int, ...]:
        """Find location of phase transition"""
        # Find location of maximum gradient
        grad = self._calculate_field_gradient(field)
        grad_magnitude = torch.norm(grad, dim=0)
        max_location = self._unravel_index(torch.argmax(grad_magnitude), grad_magnitude.shape)
        return tuple(int(x) for x in max_location)
    
    def _find_max_info_location(self, field: torch.Tensor) -> Tuple[int, ...]:
        """Find location of maximum information density"""
        # Find location of maximum field magnitude
        max_location = self._unravel_index(torch.argmax(torch.abs(field)), field.shape)
        return tuple(int(x) for x in max_location)
    
    def _find_max_curvature_location(self, curvature: torch.Tensor) -> Tuple[int, ...]:
        """Find location of maximum curvature"""
        max_location = self._unravel_index(torch.argmax(torch.abs(curvature)), curvature.shape)
        return tuple(int(x) for x in max_location)
    
    def _find_consciousness_center(self, field: torch.Tensor) -> Tuple[int, ...]:
        """Find center of consciousness activity"""
        # Weighted center of mass
        weights = torch.abs(field)
        total_weight = torch.sum(weights)
        
        if total_weight > 0:
            center = []
            for dim in range(len(field.shape)):
                coord_grid = torch.arange(field.shape[dim], device=field.device).float()
                for _ in range(dim):
                    coord_grid = coord_grid.unsqueeze(0)
                for _ in range(len(field.shape) - dim - 1):
                    coord_grid = coord_grid.unsqueeze(-1)
                coord_grid = coord_grid.expand(field.shape)
                
                weighted_coord = torch.sum(coord_grid * weights) / total_weight
                center.append(int(weighted_coord.item()))
            return tuple(center)
        else:
            return tuple(s // 2 for s in field.shape)
    
    def _find_novel_region(self, field: torch.Tensor) -> Tuple[int, ...]:
        """Find region of novel behavior"""
        # Find region with highest variance
        if len(field.shape) == 3:
            # 3D case - find highest variance region
            kernel_size = min(5, min(field.shape) // 4)
            variance_map = torch.zeros_like(field)
            
            for i in range(kernel_size//2, field.shape[0] - kernel_size//2):
                for j in range(kernel_size//2, field.shape[1] - kernel_size//2):
                    for k in range(kernel_size//2, field.shape[2] - kernel_size//2):
                        region = field[i-kernel_size//2:i+kernel_size//2+1,
                                     j-kernel_size//2:j+kernel_size//2+1,
                                     k-kernel_size//2:k+kernel_size//2+1]
                        variance_map[i, j, k] = torch.var(region)
            
            max_location = self._unravel_index(torch.argmax(variance_map), variance_map.shape)
            return tuple(int(x) for x in max_location)
        else:
            # Fallback to center
            return tuple(s // 2 for s in field.shape)
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of all detected emergence events"""
        summary = {
            "total_events": len(self.emergence_events),
            "events_by_type": {},
            "recent_events": [],
            "most_significant_events": []
        }
        
        # Count by type
        for event in self.emergence_events:
            event_type = event.emergence_type.value
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
        
        # Recent events (last 10)
        summary["recent_events"] = [
            {
                "id": event.event_id,
                "type": event.emergence_type.value,
                "magnitude": event.magnitude,
                "timestamp": event.timestamp
            }
            for event in self.emergence_events[-10:]
        ]
        
        # Most significant events
        sorted_events = sorted(self.emergence_events, key=lambda x: x.magnitude, reverse=True)
        summary["most_significant_events"] = [
            {
                "id": event.event_id,
                "type": event.emergence_type.value,
                "magnitude": event.magnitude,
                "location": event.location
            }
            for event in sorted_events[:5]
        ]
        
        return summary
