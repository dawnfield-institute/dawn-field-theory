"""
Universal Signature Detection Module

Detects and validates universal signatures across all physics scales,
including the 15.56x amplification, ξ=1.0571 balance, and other
emergent patterns that validate Dawn Field Theory frameworks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

class SignatureType(Enum):
    AMPLIFICATION_15_56X = "amplification_15_56x"
    BALANCE_OPERATOR_XI = "balance_operator_xi" 
    ENTROPY_COLLAPSE = "entropy_collapse"
    PAC_RESONANCE = "pac_resonance"
    SCALE_INVARIANT = "scale_invariant"
    CONSCIOUSNESS_THRESHOLD = "consciousness_threshold"
    QUANTUM_CLASSICAL_BRIDGE = "quantum_classical_bridge"
    GEOMETRIC_FLUID_COUPLING = "geometric_fluid_coupling"

@dataclass
class SignatureDetection:
    signature_type: SignatureType
    strength: float
    confidence: float
    location: Optional[Tuple[int, ...]]
    frequency: float
    phase: float
    duration: float
    metadata: Dict[str, Any]

@dataclass
class UniversalValidationResult:
    detected_signatures: List[SignatureDetection]
    signature_completeness: float
    temporal_consistency: float
    spatial_coherence: float
    cross_scale_validation: Dict[str, float]
    overall_validation_score: float

class UniversalSignatureDetector:
    """
    Universal signature detection across all physics scales.
    
    Validates Dawn Field Theory by detecting predicted universal
    signatures in computational simulations.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Universal signature targets and tolerances
        self.signature_targets = {
            SignatureType.AMPLIFICATION_15_56X: {
                "target_value": 15.56,
                "tolerance": 0.5,
                "minimum_strength": 0.1
            },
            SignatureType.BALANCE_OPERATOR_XI: {
                "target_value": 1.0571,
                "tolerance": 0.01,
                "minimum_strength": 0.05
            },
            SignatureType.ENTROPY_COLLAPSE: {
                "target_value": 0.1,
                "tolerance": 0.05,
                "minimum_strength": 0.02
            },
            SignatureType.PAC_RESONANCE: {
                "target_value": 1.0,
                "tolerance": 0.1,
                "minimum_strength": 0.01
            },
            SignatureType.CONSCIOUSNESS_THRESHOLD: {
                "target_value": 0.5,
                "tolerance": 0.1,
                "minimum_strength": 0.05
            }
        }
        
        # Detection parameters
        self.detection_window_size = 50
        self.temporal_smoothing = 0.1
        self.spatial_kernel_size = 3
        
    def detect_universal_signatures(self, 
                                  system_states: List[Dict[str, Any]],
                                  temporal_data: List[Dict[str, float]]) -> UniversalValidationResult:
        """
        Detect universal signatures across temporal evolution.
        
        Args:
            system_states: List of system states over time
            temporal_data: List of temporal metrics
            
        Returns:
            UniversalValidationResult with detected signatures
        """
        detected_signatures = []
        
        # Detect each signature type
        for signature_type in SignatureType:
            signatures = self._detect_signature_type(
                signature_type, system_states, temporal_data
            )
            detected_signatures.extend(signatures)
        
        # Calculate validation metrics
        completeness = self._calculate_signature_completeness(detected_signatures)
        temporal_consistency = self._calculate_temporal_consistency(detected_signatures)
        spatial_coherence = self._calculate_spatial_coherence(detected_signatures)
        cross_scale_validation = self._validate_cross_scale_consistency(detected_signatures)
        
        # Overall validation score
        overall_score = (completeness + temporal_consistency + 
                        spatial_coherence + np.mean(list(cross_scale_validation.values()))) / 4.0
        
        return UniversalValidationResult(
            detected_signatures=detected_signatures,
            signature_completeness=completeness,
            temporal_consistency=temporal_consistency,
            spatial_coherence=spatial_coherence,
            cross_scale_validation=cross_scale_validation,
            overall_validation_score=overall_score
        )
    
    def _detect_signature_type(self, 
                             signature_type: SignatureType,
                             system_states: List[Dict[str, Any]],
                             temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect specific signature type"""
        
        if signature_type == SignatureType.AMPLIFICATION_15_56X:
            return self._detect_amplification_signature(system_states, temporal_data)
        elif signature_type == SignatureType.BALANCE_OPERATOR_XI:
            return self._detect_balance_operator_signature(system_states, temporal_data)
        elif signature_type == SignatureType.ENTROPY_COLLAPSE:
            return self._detect_entropy_collapse_signature(system_states, temporal_data)
        elif signature_type == SignatureType.PAC_RESONANCE:
            return self._detect_pac_resonance_signature(system_states, temporal_data)
        elif signature_type == SignatureType.CONSCIOUSNESS_THRESHOLD:
            return self._detect_consciousness_signature(system_states, temporal_data)
        elif signature_type == SignatureType.QUANTUM_CLASSICAL_BRIDGE:
            return self._detect_quantum_classical_bridge(system_states, temporal_data)
        elif signature_type == SignatureType.GEOMETRIC_FLUID_COUPLING:
            return self._detect_geometric_fluid_coupling(system_states, temporal_data)
        else:
            return []
    
    def _detect_amplification_signature(self, 
                                      system_states: List[Dict[str, Any]],
                                      temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect 15.56x information amplification signature - always return measurements"""
        detections = []
        target = self.signature_targets[SignatureType.AMPLIFICATION_15_56X]
        
        for i, state in enumerate(system_states):
            info_metrics = state.get("information_state", {})
            amplification_ratio = info_metrics.get("amplification_ratio", 0.0)
            
            # Calculate error and deviation from target
            error = abs(amplification_ratio - target["target_value"])
            deviation = error / target["target_value"] if target["target_value"] > 0 else float('inf')
            within_tolerance = error < target["tolerance"]
            
            # Always create detection to track actual measurements
            strength = amplification_ratio / target["target_value"] if target["target_value"] > 0 else 0
            confidence = max(0.0, 1.0 - deviation)  # Confidence decreases with deviation
            
            detection = SignatureDetection(
                signature_type=SignatureType.AMPLIFICATION_15_56X,
                strength=strength,
                confidence=confidence,
                location=None,  # Global property
                frequency=self._calculate_frequency(temporal_data, i, "amplification_ratio"),
                phase=0.0,
                duration=1.0,  # Will be updated with temporal analysis
                metadata={
                    "measured_ratio": amplification_ratio,
                    "target_ratio": target["target_value"],
                    "error": error,
                    "deviation": deviation,
                    "within_tolerance": within_tolerance,
                    "timestep": i,
                    "actual_amplification": amplification_ratio,  # Track actual value
                    "target_amplification": target["target_value"]  # Track target
                }
            )
            detections.append(detection)
        
        # Merge temporally adjacent detections
        return self._merge_temporal_detections(detections)
    
    def _detect_balance_operator_signature(self, 
                                         system_states: List[Dict[str, Any]],
                                         temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect ξ = 1.0571 balance operator signature - always return measurements"""
        detections = []
        target = self.signature_targets[SignatureType.BALANCE_OPERATOR_XI]
        
        for i, state in enumerate(system_states):
            # Extract balance metrics from various sources
            quantum_state = state.get("quantum_state", {})
            conservation_quality = quantum_state.get("conservation_quality", 0.0)
            
            # Calculate xi estimate from conservation quality
            xi_estimate = 1.0 + 0.1 * conservation_quality  # Simplified mapping
            
            error = abs(xi_estimate - target["target_value"])
            deviation = error / target["target_value"] if target["target_value"] > 0 else float('inf')
            within_tolerance = error < target["tolerance"]
            
            # Always create detection to track actual measurements
            strength = min(1.0, conservation_quality)
            confidence = max(0.0, 1.0 - deviation)  # Confidence decreases with deviation
            
            detection = SignatureDetection(
                signature_type=SignatureType.BALANCE_OPERATOR_XI,
                strength=strength,
                confidence=confidence,
                location=None,
                frequency=self._calculate_frequency(temporal_data, i, "conservation_quality"),
                phase=0.0,
                duration=1.0,
                metadata={
                    "measured_xi": xi_estimate,
                    "target_xi": target["target_value"],
                    "conservation_quality": conservation_quality,
                    "error": error,
                    "deviation": deviation,
                    "within_tolerance": within_tolerance,
                    "timestep": i,
                    "balance_operator_measured": xi_estimate,  # Track actual value
                    "balance_operator_target": target["target_value"]  # Track target
                }
            )
            detections.append(detection)
        
        return self._merge_temporal_detections(detections)
    
    def _detect_entropy_collapse_signature(self, 
                                         system_states: List[Dict[str, Any]],
                                         temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect entropy collapse signature"""
        detections = []
        target = self.signature_targets[SignatureType.ENTROPY_COLLAPSE]
        
        for i, state in enumerate(system_states):
            geometric_state = state.get("geometric_state", {})
            collapse_strength = geometric_state.get("collapse_strength", 0.0)
            collapse_locations = geometric_state.get("collapse_locations", [])
            
            if collapse_strength > target["minimum_strength"]:
                # Strong collapse detected
                strength = min(1.0, collapse_strength / target["target_value"])
                confidence = min(1.0, collapse_strength / target["tolerance"])
                
                # Find primary collapse location
                primary_location = collapse_locations[0] if collapse_locations else None
                
                detection = SignatureDetection(
                    signature_type=SignatureType.ENTROPY_COLLAPSE,
                    strength=strength,
                    confidence=confidence,
                    location=primary_location,
                    frequency=self._calculate_frequency(temporal_data, i, "collapse_strength"),
                    phase=0.0,
                    duration=1.0,
                    metadata={
                        "collapse_strength": collapse_strength,
                        "num_collapses": len(collapse_locations),
                        "locations": collapse_locations,
                        "timestep": i
                    }
                )
                detections.append(detection)
        
        return self._merge_temporal_detections(detections)
    
    def _detect_pac_resonance_signature(self, 
                                      system_states: List[Dict[str, Any]],
                                      temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect PAC resonance signature"""
        detections = []
        target = self.signature_targets[SignatureType.PAC_RESONANCE]
        
        # Calculate global PAC resonance from conservation qualities across scales
        for i, state in enumerate(system_states):
            conservation_metrics = []
            
            # Collect conservation from all scales
            quantum_state = state.get("quantum_state", {})
            geometric_state = state.get("geometric_state", {})
            fluid_state = state.get("fluid_state", {})
            
            if "conservation_quality" in quantum_state:
                conservation_metrics.append(quantum_state["conservation_quality"])
            if "collapse_strength" in geometric_state:
                conservation_metrics.append(1.0 - geometric_state["collapse_strength"])
            if "reynolds_number" in fluid_state:
                # Stability as conservation proxy
                stability = 1.0 / (1.0 + fluid_state["reynolds_number"] / 1000.0)
                conservation_metrics.append(stability)
            
            if conservation_metrics:
                pac_resonance = np.mean(conservation_metrics)
                
                if pac_resonance > target["minimum_strength"]:
                    strength = pac_resonance
                    confidence = min(1.0, pac_resonance)
                    
                    detection = SignatureDetection(
                        signature_type=SignatureType.PAC_RESONANCE,
                        strength=strength,
                        confidence=confidence,
                        location=None,
                        frequency=self._calculate_frequency(temporal_data, i, "pac_resonance"),
                        phase=0.0,
                        duration=1.0,
                        metadata={
                            "pac_resonance": pac_resonance,
                            "conservation_metrics": conservation_metrics,
                            "timestep": i
                        }
                    )
                    detections.append(detection)
        
        return self._merge_temporal_detections(detections)
    
    def _detect_consciousness_signature(self, 
                                      system_states: List[Dict[str, Any]],
                                      temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect consciousness threshold signature"""
        detections = []
        target = self.signature_targets[SignatureType.CONSCIOUSNESS_THRESHOLD]
        
        for i, state in enumerate(system_states):
            consciousness_state = state.get("consciousness_state", {})
            awareness_metric = consciousness_state.get("awareness_metric", 0.0)
            emergence_locations = consciousness_state.get("emergence_locations", [])
            
            if awareness_metric > target["minimum_strength"]:
                error = abs(awareness_metric - target["target_value"])
                strength = awareness_metric
                confidence = 1.0 - error / target["tolerance"] if error < target["tolerance"] else 0.0
                
                # Find primary consciousness location
                primary_location = emergence_locations[0] if emergence_locations else None
                
                detection = SignatureDetection(
                    signature_type=SignatureType.CONSCIOUSNESS_THRESHOLD,
                    strength=strength,
                    confidence=confidence,
                    location=primary_location,
                    frequency=self._calculate_frequency(temporal_data, i, "awareness_metric"),
                    phase=0.0,
                    duration=1.0,
                    metadata={
                        "awareness_metric": awareness_metric,
                        "consciousness_level": consciousness_state.get("consciousness_level", "none"),
                        "emergence_locations": emergence_locations,
                        "timestep": i
                    }
                )
                detections.append(detection)
        
        return self._merge_temporal_detections(detections)
    
    def _detect_quantum_classical_bridge(self, 
                                       system_states: List[Dict[str, Any]],
                                       temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect quantum-classical bridging signature"""
        detections = []
        
        for i in range(1, len(system_states)):
            prev_state = system_states[i-1]
            curr_state = system_states[i]
            
            # Detect transition between quantum and classical regimes
            prev_quantum = prev_state.get("quantum_state", {})
            curr_quantum = curr_state.get("quantum_state", {})
            prev_geometric = prev_state.get("geometric_state", {})
            curr_geometric = curr_state.get("geometric_state", {})
            
            # Quantum decoherence correlating with geometric emergence
            prev_entanglement = prev_quantum.get("entanglement_measure", 0.0)
            curr_entanglement = curr_quantum.get("entanglement_measure", 0.0)
            prev_collapse = prev_geometric.get("collapse_strength", 0.0)
            curr_collapse = curr_geometric.get("collapse_strength", 0.0)
            
            entanglement_change = abs(curr_entanglement - prev_entanglement)
            collapse_change = abs(curr_collapse - prev_collapse)
            
            # Bridge signature: significant changes in both quantum and geometric
            if entanglement_change > 0.1 and collapse_change > 0.05:
                correlation = abs(entanglement_change - collapse_change)
                bridge_strength = max(entanglement_change, collapse_change)
                confidence = 1.0 / (1.0 + correlation)
                
                detection = SignatureDetection(
                    signature_type=SignatureType.QUANTUM_CLASSICAL_BRIDGE,
                    strength=bridge_strength,
                    confidence=confidence,
                    location=None,
                    frequency=1.0,  # Transition event
                    phase=0.0,
                    duration=1.0,
                    metadata={
                        "entanglement_change": entanglement_change,
                        "collapse_change": collapse_change,
                        "correlation": correlation,
                        "timestep": i
                    }
                )
                detections.append(detection)
        
        return detections
    
    def _detect_geometric_fluid_coupling(self, 
                                       system_states: List[Dict[str, Any]],
                                       temporal_data: List[Dict[str, float]]) -> List[SignatureDetection]:
        """Detect geometric-fluid coupling signature"""
        detections = []
        
        for i in range(1, len(system_states)):
            prev_state = system_states[i-1]
            curr_state = system_states[i]
            
            # Detect SEC collapse triggering MED dynamics
            prev_geometric = prev_state.get("geometric_state", {})
            curr_fluid = curr_state.get("fluid_state", {})
            
            collapse_strength = prev_geometric.get("collapse_strength", 0.0)
            fluid_emergence = curr_fluid.get("emergence_indicators", {})
            vorticity_strength = fluid_emergence.get("vorticity_strength", 0.0)
            
            # Coupling signature: geometric collapse followed by fluid activity
            if collapse_strength > 0.1 and vorticity_strength > 0.1:
                coupling_strength = min(collapse_strength, vorticity_strength)
                confidence = (collapse_strength + vorticity_strength) / 2.0
                
                detection = SignatureDetection(
                    signature_type=SignatureType.GEOMETRIC_FLUID_COUPLING,
                    strength=coupling_strength,
                    confidence=confidence,
                    location=None,
                    frequency=1.0,
                    phase=0.0,
                    duration=1.0,
                    metadata={
                        "collapse_strength": collapse_strength,
                        "vorticity_strength": vorticity_strength,
                        "reynolds_number": curr_fluid.get("reynolds_number", 0.0),
                        "timestep": i
                    }
                )
                detections.append(detection)
        
        return detections
    
    def _calculate_frequency(self, temporal_data: List[Dict[str, float]], 
                           timestep: int, metric_name: str) -> float:
        """Calculate frequency of metric oscillation"""
        if len(temporal_data) < 10 or timestep < 5:
            return 0.0
        
        # Extract metric values around current timestep
        start_idx = max(0, timestep - 5)
        end_idx = min(len(temporal_data), timestep + 5)
        
        values = []
        for i in range(start_idx, end_idx):
            if i < len(temporal_data) and metric_name in temporal_data[i]:
                values.append(temporal_data[i][metric_name])
        
        if len(values) < 5:
            return 0.0
        
        # Simple frequency estimation using zero crossings
        mean_value = np.mean(values)
        centered_values = np.array(values) - mean_value
        
        zero_crossings = 0
        for i in range(1, len(centered_values)):
            if centered_values[i-1] * centered_values[i] < 0:
                zero_crossings += 1
        
        # Frequency = zero crossings / (2 * time_span)
        frequency = zero_crossings / (2.0 * len(values))
        return frequency
    
    def _merge_temporal_detections(self, detections: List[SignatureDetection]) -> List[SignatureDetection]:
        """Merge temporally adjacent detections"""
        if not detections:
            return []
        
        merged = []
        current_group = [detections[0]]
        
        for i in range(1, len(detections)):
            current_timestep = detections[i].metadata.get("timestep", 0)
            prev_timestep = current_group[-1].metadata.get("timestep", 0)
            
            # Merge if within temporal window
            if current_timestep - prev_timestep <= 3:
                current_group.append(detections[i])
            else:
                # Finalize current group
                merged_detection = self._create_merged_detection(current_group)
                merged.append(merged_detection)
                current_group = [detections[i]]
        
        # Add final group
        if current_group:
            merged_detection = self._create_merged_detection(current_group)
            merged.append(merged_detection)
        
        return merged
    
    def _create_merged_detection(self, detection_group: List[SignatureDetection]) -> SignatureDetection:
        """Create merged detection from group"""
        if len(detection_group) == 1:
            detection_group[0].duration = 1.0
            return detection_group[0]
        
        # Merge properties
        avg_strength = np.mean([d.strength for d in detection_group])
        avg_confidence = np.mean([d.confidence for d in detection_group])
        avg_frequency = np.mean([d.frequency for d in detection_group])
        duration = len(detection_group)
        
        # Use first detection as template
        template = detection_group[0]
        
        merged_metadata = template.metadata.copy()
        merged_metadata["duration"] = duration
        merged_metadata["num_detections"] = len(detection_group)
        merged_metadata["timestep_range"] = (
            min(d.metadata.get("timestep", 0) for d in detection_group),
            max(d.metadata.get("timestep", 0) for d in detection_group)
        )
        
        return SignatureDetection(
            signature_type=template.signature_type,
            strength=avg_strength,
            confidence=avg_confidence,
            location=template.location,
            frequency=avg_frequency,
            phase=template.phase,
            duration=duration,
            metadata=merged_metadata
        )
    
    def _calculate_signature_completeness(self, detections: List[SignatureDetection]) -> float:
        """Calculate what fraction of expected signatures were detected"""
        expected_signatures = set(SignatureType)
        detected_signatures = set(d.signature_type for d in detections)
        
        completeness = len(detected_signatures) / len(expected_signatures)
        return completeness
    
    def _calculate_temporal_consistency(self, detections: List[SignatureDetection]) -> float:
        """Calculate temporal consistency of signature detection"""
        if not detections:
            return 0.0
        
        # Group by signature type
        signature_groups = {}
        for detection in detections:
            sig_type = detection.signature_type
            if sig_type not in signature_groups:
                signature_groups[sig_type] = []
            signature_groups[sig_type].append(detection)
        
        consistency_scores = []
        for sig_type, group in signature_groups.items():
            if len(group) > 1:
                # Calculate strength variance
                strengths = [d.strength for d in group]
                consistency = 1.0 / (1.0 + np.var(strengths))
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(1.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_spatial_coherence(self, detections: List[SignatureDetection]) -> float:
        """Calculate spatial coherence of signatures"""
        # Signatures with spatial locations
        spatial_detections = [d for d in detections if d.location is not None]
        
        if len(spatial_detections) < 2:
            return 1.0  # Default high coherence for global signatures
        
        # Calculate spatial clustering
        locations = [d.location for d in spatial_detections]
        
        # Simple coherence based on location variance
        if len(locations) > 1:
            location_arrays = np.array(locations)
            spatial_variance = np.mean(np.var(location_arrays, axis=0))
            coherence = 1.0 / (1.0 + spatial_variance / 100.0)  # Normalize by field size
        else:
            coherence = 1.0
        
        return coherence
    
    def _validate_cross_scale_consistency(self, detections: List[SignatureDetection]) -> Dict[str, float]:
        """Validate consistency across different scales"""
        # Group detections by scale
        quantum_signatures = [d for d in detections if "quantum" in d.signature_type.value]
        geometric_signatures = [d for d in detections if "geometric" in d.signature_type.value or "entropy" in d.signature_type.value]
        fluid_signatures = [d for d in detections if "fluid" in d.signature_type.value]
        consciousness_signatures = [d for d in detections if "consciousness" in d.signature_type.value]
        
        validation = {}
        
        # Quantum-geometric consistency
        if quantum_signatures and geometric_signatures:
            q_strength = np.mean([d.strength for d in quantum_signatures])
            g_strength = np.mean([d.strength for d in geometric_signatures])
            validation["quantum_geometric"] = 1.0 / (1.0 + abs(q_strength - g_strength))
        else:
            validation["quantum_geometric"] = 0.5
        
        # Geometric-fluid consistency  
        if geometric_signatures and fluid_signatures:
            g_strength = np.mean([d.strength for d in geometric_signatures])
            f_strength = np.mean([d.strength for d in fluid_signatures])
            validation["geometric_fluid"] = 1.0 / (1.0 + abs(g_strength - f_strength))
        else:
            validation["geometric_fluid"] = 0.5
        
        # Information-consciousness consistency
        info_signatures = [d for d in detections if "amplification" in d.signature_type.value]
        if info_signatures and consciousness_signatures:
            i_strength = np.mean([d.strength for d in info_signatures])
            c_strength = np.mean([d.strength for d in consciousness_signatures])
            validation["information_consciousness"] = 1.0 / (1.0 + abs(i_strength - c_strength))
        else:
            validation["information_consciousness"] = 0.5
        
        return validation
    
    def generate_signature_report(self, result: UniversalValidationResult) -> str:
        """Generate comprehensive signature detection report"""
        report = []
        report.append("UNIVERSAL SIGNATURE DETECTION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append(f"Overall Validation Score: {result.overall_validation_score:.3f}")
        report.append(f"Signature Completeness: {result.signature_completeness:.3f}")
        report.append(f"Temporal Consistency: {result.temporal_consistency:.3f}")
        report.append(f"Spatial Coherence: {result.spatial_coherence:.3f}")
        report.append("")
        
        # Cross-scale validation
        report.append("Cross-Scale Validation:")
        for scale_pair, score in result.cross_scale_validation.items():
            report.append(f"  {scale_pair}: {score:.3f}")
        report.append("")
        
        # Detected signatures by type
        signature_groups = {}
        for detection in result.detected_signatures:
            sig_type = detection.signature_type
            if sig_type not in signature_groups:
                signature_groups[sig_type] = []
            signature_groups[sig_type].append(detection)
        
        report.append("Detected Signatures:")
        for sig_type, detections in signature_groups.items():
            report.append(f"\n{sig_type.value.upper()}:")
            for i, detection in enumerate(detections):
                report.append(f"  Detection {i+1}:")
                report.append(f"    Strength: {detection.strength:.3f}")
                report.append(f"    Confidence: {detection.confidence:.3f}")
                report.append(f"    Duration: {detection.duration:.1f}")
                if detection.location:
                    report.append(f"    Location: {detection.location}")
                
                # Key metadata
                metadata = detection.metadata
                if "measured_ratio" in metadata:
                    report.append(f"    Measured Ratio: {metadata['measured_ratio']:.3f}")
                if "target_ratio" in metadata:
                    report.append(f"    Target Ratio: {metadata['target_ratio']:.3f}")
                if "error" in metadata:
                    report.append(f"    Error: {metadata['error']:.3f}")
        
        return "\n".join(report)
