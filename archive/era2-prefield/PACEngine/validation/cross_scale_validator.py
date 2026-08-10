"""
Cross-Scale Validator

Validates consistency and coupling between different physics scales
to ensure proper PAC conservation across quantum, geometric, fluid,
information, and consciousness domains.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class CrossScaleValidationResult:
    scale_correlations: Dict[str, float]
    coupling_strengths: Dict[str, float]
    conservation_consistency: float
    temporal_synchronization: float
    emergence_cascade_detected: bool
    validation_passed: bool

class CrossScaleValidator:
    """Validates cross-scale consistency in PAC physics engine"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.coupling_thresholds = {
            "quantum_geometric": 0.3,
            "geometric_fluid": 0.4,
            "fluid_information": 0.3,
            "information_consciousness": 0.5
        }
    
    def validate_cross_scale_consistency(self, meta_states: List[Dict[str, Any]]) -> CrossScaleValidationResult:
        """Validate consistency across all physics scales"""
        
        # Calculate scale correlations
        correlations = self._calculate_scale_correlations(meta_states)
        
        # Measure coupling strengths
        couplings = self._measure_coupling_strengths(meta_states)
        
        # Check conservation consistency
        conservation = self._check_conservation_consistency(meta_states)
        
        # Analyze temporal synchronization
        synchronization = self._analyze_temporal_synchronization(meta_states)
        
        # Detect emergence cascades
        cascade_detected = self._detect_emergence_cascades(meta_states)
        
        # Overall validation
        validation_passed = all([
            correlations.get("overall", 0) > 0.5,
            conservation > 0.8,
            synchronization > 0.6
        ])
        
        return CrossScaleValidationResult(
            scale_correlations=correlations,
            coupling_strengths=couplings,
            conservation_consistency=conservation,
            temporal_synchronization=synchronization,
            emergence_cascade_detected=cascade_detected,
            validation_passed=validation_passed
        )
    
    def _calculate_scale_correlations(self, meta_states: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlations between different scales"""
        if len(meta_states) < 2:
            return {"overall": 0.0}
        
        # Extract time series for each scale
        quantum_series = [state.get("quantum_state", {}).get("conservation_quality", 0) for state in meta_states]
        geometric_series = [1.0 - state.get("geometric_state", {}).get("collapse_strength", 0) for state in meta_states]
        fluid_series = [1.0/(1.0 + state.get("fluid_state", {}).get("reynolds_number", 0)/1000) for state in meta_states]
        info_series = [state.get("information_state", {}).get("resonance_strength", 0) for state in meta_states]
        consciousness_series = [state.get("consciousness_state", {}).get("awareness_metric", 0) for state in meta_states]
        
        # Calculate pairwise correlations
        correlations = {}
        series_dict = {
            "quantum": quantum_series,
            "geometric": geometric_series, 
            "fluid": fluid_series,
            "information": info_series,
            "consciousness": consciousness_series
        }
        
        scale_names = list(series_dict.keys())
        all_correlations = []
        
        for i in range(len(scale_names)):
            for j in range(i+1, len(scale_names)):
                name1, name2 = scale_names[i], scale_names[j]
                series1, series2 = series_dict[name1], series_dict[name2]
                
                if len(series1) > 1 and len(series2) > 1:
                    corr = np.corrcoef(series1, series2)[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{name1}_{name2}"] = abs(corr)
                        all_correlations.append(abs(corr))
        
        correlations["overall"] = np.mean(all_correlations) if all_correlations else 0.0
        return correlations
    
    def _measure_coupling_strengths(self, meta_states: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure coupling strengths between scales"""
        couplings = {}
        
        for i in range(1, len(meta_states)):
            prev_state = meta_states[i-1]
            curr_state = meta_states[i]
            
            # Quantum-Geometric coupling
            q_change = abs(curr_state.get("quantum_state", {}).get("conservation_quality", 0) - 
                          prev_state.get("quantum_state", {}).get("conservation_quality", 0))
            g_change = abs(curr_state.get("geometric_state", {}).get("collapse_strength", 0) - 
                          prev_state.get("geometric_state", {}).get("collapse_strength", 0))
            
            if "quantum_geometric" not in couplings:
                couplings["quantum_geometric"] = []
            couplings["quantum_geometric"].append(min(q_change, g_change))
        
        # Average coupling strengths
        for key in couplings:
            couplings[key] = np.mean(couplings[key])
        
        return couplings
    
    def _check_conservation_consistency(self, meta_states: List[Dict[str, Any]]) -> float:
        """Check PAC conservation consistency across scales"""
        conservation_scores = []
        
        for state in meta_states:
            scale_scores = []
            
            # Quantum conservation
            q_conservation = state.get("quantum_state", {}).get("conservation_quality", 0)
            scale_scores.append(q_conservation)
            
            # Geometric conservation (inverse of collapse)
            g_conservation = 1.0 - state.get("geometric_state", {}).get("collapse_strength", 0)
            scale_scores.append(max(0, g_conservation))
            
            # Information conservation (resonance strength)
            i_conservation = state.get("information_state", {}).get("resonance_strength", 0)
            scale_scores.append(i_conservation)
            
            if scale_scores:
                conservation_scores.append(np.mean(scale_scores))
        
        return np.mean(conservation_scores) if conservation_scores else 0.0
    
    def _analyze_temporal_synchronization(self, meta_states: List[Dict[str, Any]]) -> float:
        """Analyze temporal synchronization between scales"""
        if len(meta_states) < 3:
            return 0.5
        
        # Look for synchronized changes across scales
        synchronization_events = 0
        total_events = 0
        
        for i in range(2, len(meta_states)):
            prev_state = meta_states[i-1]
            curr_state = meta_states[i]
            
            # Detect significant changes in each scale
            scales_changed = 0
            
            # Quantum change
            q_change = abs(curr_state.get("quantum_state", {}).get("conservation_quality", 0) - 
                          prev_state.get("quantum_state", {}).get("conservation_quality", 0))
            if q_change > 0.1:
                scales_changed += 1
            
            # Geometric change
            g_change = abs(curr_state.get("geometric_state", {}).get("collapse_strength", 0) - 
                          prev_state.get("geometric_state", {}).get("collapse_strength", 0))
            if g_change > 0.05:
                scales_changed += 1
            
            # Information change
            i_change = abs(curr_state.get("information_state", {}).get("amplification_ratio", 0) - 
                          prev_state.get("information_state", {}).get("amplification_ratio", 0))
            if i_change > 1.0:
                scales_changed += 1
            
            if scales_changed >= 2:
                synchronization_events += 1
                total_events += 1
            elif scales_changed >= 1:
                total_events += 1
        
        return synchronization_events / total_events if total_events > 0 else 0.0
    
    def _detect_emergence_cascades(self, meta_states: List[Dict[str, Any]]) -> bool:
        """Detect emergence cascade events across scales"""
        cascade_detected = False
        
        for i in range(2, len(meta_states)):
            state = meta_states[i]
            
            # Check for simultaneous high activity across multiple scales
            high_activity_scales = 0
            
            # Quantum entanglement
            if state.get("quantum_state", {}).get("entanglement_measure", 0) > 0.5:
                high_activity_scales += 1
            
            # Geometric collapse
            if state.get("geometric_state", {}).get("collapse_strength", 0) > 0.3:
                high_activity_scales += 1
            
            # Fluid turbulence
            if state.get("fluid_state", {}).get("reynolds_number", 0) > 1000:
                high_activity_scales += 1
            
            # Information amplification
            if state.get("information_state", {}).get("amplification_ratio", 0) > 5.0:
                high_activity_scales += 1
            
            # Consciousness emergence
            if state.get("consciousness_state", {}).get("awareness_metric", 0) > 0.4:
                high_activity_scales += 1
            
            if high_activity_scales >= 3:
                cascade_detected = True
                break
        
        return cascade_detected
