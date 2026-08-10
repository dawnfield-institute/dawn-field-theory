"""
Emergence Tracker

Tracks and analyzes emergent phenomena across all physics scales,
monitoring for novel behaviors, phase transitions, and consciousness
emergence events in the PAC physics engine.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

class EmergenceEventType(Enum):
    PHASE_TRANSITION = "phase_transition"
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    INFORMATION_AMPLIFICATION = "information_amplification"
    GEOMETRIC_COLLAPSE = "geometric_collapse"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    FLUID_TURBULENCE = "fluid_turbulence"
    CROSS_SCALE_COUPLING = "cross_scale_coupling"
    NOVEL_BEHAVIOR = "novel_behavior"

@dataclass
class EmergenceEvent:
    event_id: str
    event_type: EmergenceEventType
    timestamp: float
    location: Optional[Tuple[int, ...]]
    magnitude: float
    duration: float
    precursor_events: List[str]
    cascade_effects: List[str]
    metadata: Dict[str, Any]

@dataclass 
class EmergenceAnalysis:
    total_events: int
    events_by_type: Dict[str, int]
    emergence_rate: float
    cascade_chains: List[List[str]]
    complexity_measure: float
    predictability_score: float

class EmergenceTracker:
    """Tracks emergent phenomena across all physics scales"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Event tracking
        self.tracked_events: List[EmergenceEvent] = []
        self.event_counter = 0
        
        # Detection thresholds
        self.detection_thresholds = {
            EmergenceEventType.PHASE_TRANSITION: 0.1,
            EmergenceEventType.CONSCIOUSNESS_EMERGENCE: 0.3,
            EmergenceEventType.INFORMATION_AMPLIFICATION: 2.0,
            EmergenceEventType.GEOMETRIC_COLLAPSE: 0.05,
            EmergenceEventType.QUANTUM_DECOHERENCE: 0.2,
            EmergenceEventType.FLUID_TURBULENCE: 500.0,
            EmergenceEventType.CROSS_SCALE_COUPLING: 0.5,
            EmergenceEventType.NOVEL_BEHAVIOR: 0.4
        }
        
        # Temporal window for cascade detection
        self.cascade_window = 5
        
    def track_emergence_events(self, meta_states: List[Dict[str, Any]], 
                             timestamps: List[float] = None) -> EmergenceAnalysis:
        """Track emergence events across temporal evolution"""
        
        if timestamps is None:
            timestamps = list(range(len(meta_states)))
        
        # Clear previous tracking
        self.tracked_events = []
        self.event_counter = 0
        
        # Detect events in each timestep
        for i, (state, timestamp) in enumerate(zip(meta_states, timestamps)):
            events = self._detect_events_in_state(state, timestamp, i)
            self.tracked_events.extend(events)
        
        # Detect cascade chains
        cascade_chains = self._detect_cascade_chains()
        
        # Calculate emergence metrics
        analysis = self._analyze_emergence_patterns(cascade_chains)
        
        return analysis
    
    def _detect_events_in_state(self, state: Dict[str, Any], 
                              timestamp: float, step: int) -> List[EmergenceEvent]:
        """Detect emergence events in a single state"""
        events = []
        
        # Consciousness emergence
        consciousness_state = state.get("consciousness_state", {})
        awareness = consciousness_state.get("awareness_metric", 0)
        if awareness > self.detection_thresholds[EmergenceEventType.CONSCIOUSNESS_EMERGENCE]:
            event = EmergenceEvent(
                event_id=f"consciousness_{self.event_counter}",
                event_type=EmergenceEventType.CONSCIOUSNESS_EMERGENCE,
                timestamp=timestamp,
                location=consciousness_state.get("emergence_locations", [None])[0],
                magnitude=awareness,
                duration=1.0,
                precursor_events=[],
                cascade_effects=[],
                metadata={
                    "consciousness_level": consciousness_state.get("consciousness_level", "none"),
                    "binding_strength": consciousness_state.get("binding_strength", 0),
                    "step": step
                }
            )
            events.append(event)
            self.event_counter += 1
        
        # Information amplification
        info_state = state.get("information_state", {})
        amplification = info_state.get("amplification_ratio", 0)
        if amplification > self.detection_thresholds[EmergenceEventType.INFORMATION_AMPLIFICATION]:
            event = EmergenceEvent(
                event_id=f"amplification_{self.event_counter}",
                event_type=EmergenceEventType.INFORMATION_AMPLIFICATION,
                timestamp=timestamp,
                location=None,
                magnitude=amplification,
                duration=1.0,
                precursor_events=[],
                cascade_effects=[],
                metadata={
                    "amplification_ratio": amplification,
                    "resonance_strength": info_state.get("resonance_strength", 0),
                    "step": step
                }
            )
            events.append(event)
            self.event_counter += 1
        
        # Geometric collapse
        geometric_state = state.get("geometric_state", {})
        collapse_strength = geometric_state.get("collapse_strength", 0)
        if collapse_strength > self.detection_thresholds[EmergenceEventType.GEOMETRIC_COLLAPSE]:
            collapse_locations = geometric_state.get("collapse_locations", [])
            primary_location = collapse_locations[0] if collapse_locations else None
            
            event = EmergenceEvent(
                event_id=f"collapse_{self.event_counter}",
                event_type=EmergenceEventType.GEOMETRIC_COLLAPSE,
                timestamp=timestamp,
                location=primary_location,
                magnitude=collapse_strength,
                duration=1.0,
                precursor_events=[],
                cascade_effects=[],
                metadata={
                    "collapse_strength": collapse_strength,
                    "num_collapses": len(collapse_locations),
                    "geometric_phase": geometric_state.get("geometric_phase", "unknown"),
                    "step": step
                }
            )
            events.append(event)
            self.event_counter += 1
        
        # Quantum decoherence
        quantum_state = state.get("quantum_state", {})
        entanglement = quantum_state.get("entanglement_measure", 0)
        if entanglement > self.detection_thresholds[EmergenceEventType.QUANTUM_DECOHERENCE]:
            event = EmergenceEvent(
                event_id=f"quantum_{self.event_counter}",
                event_type=EmergenceEventType.QUANTUM_DECOHERENCE,
                timestamp=timestamp,
                location=None,
                magnitude=entanglement,
                duration=1.0,
                precursor_events=[],
                cascade_effects=[],
                metadata={
                    "entanglement_measure": entanglement,
                    "conservation_quality": quantum_state.get("conservation_quality", 0),
                    "coherence_time": quantum_state.get("coherence_time", 0),
                    "step": step
                }
            )
            events.append(event)
            self.event_counter += 1
        
        # Fluid turbulence
        fluid_state = state.get("fluid_state", {})
        reynolds = fluid_state.get("reynolds_number", 0)
        if reynolds > self.detection_thresholds[EmergenceEventType.FLUID_TURBULENCE]:
            event = EmergenceEvent(
                event_id=f"turbulence_{self.event_counter}",
                event_type=EmergenceEventType.FLUID_TURBULENCE,
                timestamp=timestamp,
                location=None,
                magnitude=reynolds,
                duration=1.0,
                precursor_events=[],
                cascade_effects=[],
                metadata={
                    "reynolds_number": reynolds,
                    "fluid_regime": fluid_state.get("fluid_regime", "unknown"),
                    "vorticity_strength": fluid_state.get("emergence_indicators", {}).get("vorticity_strength", 0),
                    "step": step
                }
            )
            events.append(event)
            self.event_counter += 1
        
        return events
    
    def _detect_cascade_chains(self) -> List[List[str]]:
        """Detect cascade chains of emergence events"""
        cascade_chains = []
        
        # Sort events by timestamp
        sorted_events = sorted(self.tracked_events, key=lambda x: x.timestamp)
        
        # Build temporal chains
        for i, event in enumerate(sorted_events):
            chain = [event.event_id]
            
            # Look for subsequent events within cascade window
            for j in range(i+1, len(sorted_events)):
                next_event = sorted_events[j]
                time_diff = next_event.timestamp - event.timestamp
                
                if time_diff <= self.cascade_window:
                    # Check for causal relationship
                    if self._is_causal_relationship(event, next_event):
                        chain.append(next_event.event_id)
                        # Update cascade effects
                        event.cascade_effects.append(next_event.event_id)
                        next_event.precursor_events.append(event.event_id)
                else:
                    break
            
            if len(chain) > 1:
                cascade_chains.append(chain)
        
        return cascade_chains
    
    def _is_causal_relationship(self, event1: EmergenceEvent, event2: EmergenceEvent) -> bool:
        """Determine if there's a causal relationship between events"""
        
        # Known causal relationships
        causal_pairs = [
            (EmergenceEventType.QUANTUM_DECOHERENCE, EmergenceEventType.GEOMETRIC_COLLAPSE),
            (EmergenceEventType.GEOMETRIC_COLLAPSE, EmergenceEventType.FLUID_TURBULENCE),
            (EmergenceEventType.FLUID_TURBULENCE, EmergenceEventType.INFORMATION_AMPLIFICATION),
            (EmergenceEventType.INFORMATION_AMPLIFICATION, EmergenceEventType.CONSCIOUSNESS_EMERGENCE),
            (EmergenceEventType.GEOMETRIC_COLLAPSE, EmergenceEventType.CONSCIOUSNESS_EMERGENCE)
        ]
        
        event_pair = (event1.event_type, event2.event_type)
        return event_pair in causal_pairs
    
    def _analyze_emergence_patterns(self, cascade_chains: List[List[str]]) -> EmergenceAnalysis:
        """Analyze patterns in emergence events"""
        
        # Count events by type
        events_by_type = {}
        for event in self.tracked_events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        # Calculate emergence rate
        if self.tracked_events:
            time_span = max(e.timestamp for e in self.tracked_events) - min(e.timestamp for e in self.tracked_events)
            emergence_rate = len(self.tracked_events) / (time_span + 1e-6)
        else:
            emergence_rate = 0.0
        
        # Calculate complexity measure
        complexity = self._calculate_complexity_measure(cascade_chains)
        
        # Calculate predictability score
        predictability = self._calculate_predictability_score(cascade_chains)
        
        return EmergenceAnalysis(
            total_events=len(self.tracked_events),
            events_by_type=events_by_type,
            emergence_rate=emergence_rate,
            cascade_chains=cascade_chains,
            complexity_measure=complexity,
            predictability_score=predictability
        )
    
    def _calculate_complexity_measure(self, cascade_chains: List[List[str]]) -> float:
        """Calculate complexity measure of emergence patterns"""
        if not cascade_chains:
            return 0.0
        
        # Complexity based on chain lengths and interconnections
        total_complexity = 0.0
        
        for chain in cascade_chains:
            # Chain length contributes to complexity
            chain_complexity = len(chain) / 10.0  # Normalize
            
            # Branching factor (events that trigger multiple cascades)
            branching_events = 0
            for event_id in chain:
                event = next((e for e in self.tracked_events if e.event_id == event_id), None)
                if event and len(event.cascade_effects) > 1:
                    branching_events += 1
            
            branching_complexity = branching_events / len(chain) if chain else 0
            
            total_complexity += chain_complexity + branching_complexity
        
        return total_complexity / len(cascade_chains)
    
    def _calculate_predictability_score(self, cascade_chains: List[List[str]]) -> float:
        """Calculate predictability score based on known patterns"""
        if not cascade_chains:
            return 0.0
        
        predicted_patterns = 0
        total_patterns = len(cascade_chains)
        
        for chain in cascade_chains:
            # Get event types in chain
            event_types = []
            for event_id in chain:
                event = next((e for e in self.tracked_events if e.event_id == event_id), None)
                if event:
                    event_types.append(event.event_type)
            
            # Check against known patterns
            if self._matches_known_pattern(event_types):
                predicted_patterns += 1
        
        return predicted_patterns / total_patterns
    
    def _matches_known_pattern(self, event_types: List[EmergenceEventType]) -> bool:
        """Check if event sequence matches known emergence patterns"""
        
        # Known emergence patterns
        known_patterns = [
            # Quantum → Geometric → Fluid → Information → Consciousness
            [EmergenceEventType.QUANTUM_DECOHERENCE, EmergenceEventType.GEOMETRIC_COLLAPSE, 
             EmergenceEventType.FLUID_TURBULENCE, EmergenceEventType.INFORMATION_AMPLIFICATION,
             EmergenceEventType.CONSCIOUSNESS_EMERGENCE],
            
            # Geometric → Consciousness (direct path)
            [EmergenceEventType.GEOMETRIC_COLLAPSE, EmergenceEventType.CONSCIOUSNESS_EMERGENCE],
            
            # Information → Consciousness
            [EmergenceEventType.INFORMATION_AMPLIFICATION, EmergenceEventType.CONSCIOUSNESS_EMERGENCE]
        ]
        
        # Check if event sequence is a subsequence of any known pattern
        for pattern in known_patterns:
            if self._is_subsequence(event_types, pattern):
                return True
        
        return False
    
    def _is_subsequence(self, seq1: List, seq2: List) -> bool:
        """Check if seq1 is a subsequence of seq2"""
        i = 0
        for item in seq2:
            if i < len(seq1) and seq1[i] == item:
                i += 1
        return i == len(seq1)
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of emergence tracking results"""
        if not self.tracked_events:
            return {"total_events": 0, "message": "No emergence events detected"}
        
        # Event statistics
        events_by_type = {}
        for event in self.tracked_events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        # Temporal statistics
        timestamps = [e.timestamp for e in self.tracked_events]
        time_span = max(timestamps) - min(timestamps) if timestamps else 0
        
        # Magnitude statistics
        magnitudes = [e.magnitude for e in self.tracked_events]
        
        return {
            "total_events": len(self.tracked_events),
            "events_by_type": events_by_type,
            "time_span": time_span,
            "emergence_rate": len(self.tracked_events) / (time_span + 1e-6),
            "average_magnitude": np.mean(magnitudes) if magnitudes else 0,
            "max_magnitude": max(magnitudes) if magnitudes else 0,
            "most_common_event": max(events_by_type.items(), key=lambda x: x[1])[0] if events_by_type else "none"
        }
