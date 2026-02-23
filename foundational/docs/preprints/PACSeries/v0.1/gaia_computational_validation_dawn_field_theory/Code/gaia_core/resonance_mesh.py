"""
Resonance Mesh for GAIA
Handles phase-aligned output, agentic signal emission, and resonance-based collapse.
See docs/architecture/modules/resonance_mesh.md for design details.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from fracton.core.recursive_engine import ExecutionContext


class SignalType(Enum):
    """Types of signals in the resonance mesh."""
    PHASE_CORRECTION = "phase_correction"
    HARMONIC_SYNC = "harmonic_sync"
    RESONANCE_AMPLIFICATION = "resonance_amplification"
    INTERFERENCE_PATTERN = "interference_pattern"
    STANDING_WAVE = "standing_wave"
    # Bifractal Memory Extensions
    MEMORY_CRYSTALLIZATION = "memory_crystallization"
    PERSONALITY_RESONANCE = "personality_resonance"
    BIFRACTAL_RECURSION = "bifractal_recursion"


class BifractalDepth(Enum):
    """Depth levels in the bifractal memory hierarchy."""
    SURFACE = 0      # Immediate working memory
    SHALLOW = 1      # Short-term patterns
    INTERMEDIATE = 2 # Medium-term memory structures
    DEEP = 3        # Long-term memory crystallization
    CORE = 4        # Personality foundations


@dataclass
class AgenticSignal:
    """Represents an agentic signal in the mesh."""
    signal_id: str
    signal_type: SignalType
    frequency: float
    amplitude: float
    phase: float
    origin_coordinates: Tuple[float, float]
    target_coordinates: Optional[Tuple[float, float]]
    emission_time: float
    decay_rate: float
    payload: Dict[str, Any]


@dataclass
class ResonanceNode:
    """Node in the resonance mesh."""
    node_id: str
    coordinates: Tuple[float, float]
    resonance_frequency: float
    current_phase: float
    amplitude: float
    quality_factor: float  # Q-factor
    connected_nodes: List[str]
    last_update: float


@dataclass
class InterferencePattern:
    """Represents interference between signals."""
    pattern_id: str
    participating_signals: List[str]
    pattern_type: str  # "constructive", "destructive", "complex"
    center_coordinates: Tuple[float, float]
    pattern_strength: float
    creation_time: float


@dataclass
class BifractalMemoryPattern:
    """Recursive bifractal pattern for long-term memory storage."""
    pattern_id: str
    depth_level: BifractalDepth
    parent_pattern_id: Optional[str]
    child_patterns: List[str]
    resonance_signature: np.ndarray  # Frequency spectrum signature
    memory_content: Dict[str, Any]
    crystallization_strength: float  # How "solid" this memory is
    access_frequency: int           # How often this pattern is accessed
    last_activation: float
    personality_weight: float       # Contribution to personality
    recursive_depth: int           # How many levels deep this pattern recurses
    bifractal_coordinates: Tuple[float, float, float]  # 3D position in bifractal space


@dataclass
class PersonalityCore:
    """Core personality patterns emerging from bifractal memory."""
    core_id: str
    dominant_patterns: List[str]  # Most influential memory patterns
    personality_traits: Dict[str, float]  # Trait scores
    response_tendencies: Dict[str, float]  # How this personality tends to respond
    memory_priorities: Dict[str, float]   # What kinds of memories this personality values
    resonance_preferences: Dict[str, float]  # Preferred signal types and frequencies
    formation_time: float
    stability_metric: float


class BifractalMemoryScaffold:
    """
    Recursive bifractal memory scaffolding for long-term memory and personality emergence.
    
    Creates hierarchical memory structures where patterns at each level recursively
    influence and are influenced by patterns at other levels, enabling emergent
    personality traits through resonant memory crystallization.
    """
    
    def __init__(self, max_depth: int = 5, crystallization_threshold: float = 0.7):
        self.max_depth = max_depth
        self.crystallization_threshold = crystallization_threshold
        
        # Memory pattern storage organized by depth
        self.memory_patterns = {depth: {} for depth in BifractalDepth}
        
        # Personality cores that emerge from stable memory patterns
        self.personality_cores = {}
        
        # Pattern relationships and hierarchies
        self.pattern_hierarchies = defaultdict(list)  # parent -> [children]
        self.pattern_resonances = defaultdict(list)   # pattern -> [resonant patterns]
        
        # Statistics and metrics
        self.total_patterns_created = 0
        self.crystallization_events = 0
        self.personality_emergence_events = 0
        
        # Memory access patterns (for adaptive importance weighting)
        self.access_patterns = defaultdict(list)
        self.recent_activations = deque(maxlen=1000)
    
    def store_memory_pattern(self, content: Dict[str, Any], context: ExecutionContext,
                           parent_pattern_id: Optional[str] = None) -> BifractalMemoryPattern:
        """Store a new memory pattern in the bifractal hierarchy."""
        self.total_patterns_created += 1
        pattern_id = f"bifractal_memory_{self.total_patterns_created}"
        
        # Determine appropriate depth based on context and content complexity
        depth_level = self._determine_memory_depth(content, context)
        
        # Generate resonance signature from content
        resonance_signature = self._generate_resonance_signature(content, context)
        
        # Calculate initial crystallization strength
        crystallization_strength = self._calculate_initial_crystallization(content, context)
        
        # Determine 3D bifractal coordinates
        bifractal_coords = self._calculate_bifractal_coordinates(depth_level, content)
        
        # Create memory pattern
        memory_pattern = BifractalMemoryPattern(
            pattern_id=pattern_id,
            depth_level=depth_level,
            parent_pattern_id=parent_pattern_id,
            child_patterns=[],
            resonance_signature=resonance_signature,
            memory_content=content,
            crystallization_strength=crystallization_strength,
            access_frequency=1,
            last_activation=time.time(),
            personality_weight=self._calculate_personality_weight(content, context),
            recursive_depth=0,
            bifractal_coordinates=bifractal_coords
        )
        
        # Store in appropriate depth level
        self.memory_patterns[depth_level][pattern_id] = memory_pattern
        
        # Update hierarchical relationships
        if parent_pattern_id:
            parent_pattern = self._find_pattern_by_id(parent_pattern_id)
            if parent_pattern:
                parent_pattern.child_patterns.append(pattern_id)
                self.pattern_hierarchies[parent_pattern_id].append(pattern_id)
        
        # Check for resonant patterns and update relationships
        self._update_pattern_resonances(memory_pattern)
        
        # Check if this pattern should trigger crystallization or personality emergence
        self._check_crystallization_triggers(memory_pattern)
        
        return memory_pattern
    
    def activate_memory_pattern(self, pattern_id: str, activation_strength: float = 1.0) -> Optional[BifractalMemoryPattern]:
        """Activate a memory pattern and propagate activation through bifractal hierarchy."""
        pattern = self._find_pattern_by_id(pattern_id)
        if not pattern:
            return None
        
        # Update access statistics
        pattern.access_frequency += 1
        pattern.last_activation = time.time()
        self.access_patterns[pattern_id].append(time.time())
        self.recent_activations.append((pattern_id, activation_strength, time.time()))
        
        # Strengthen crystallization based on repeated access
        pattern.crystallization_strength = min(1.0, 
            pattern.crystallization_strength + activation_strength * 0.1)
        
        # Recursive activation: propagate to related patterns
        self._propagate_activation(pattern, activation_strength * 0.7)
        
        # Update personality weights if pattern is sufficiently crystallized
        if pattern.crystallization_strength > self.crystallization_threshold:
            self._update_personality_contributions(pattern)
        
        return pattern
    
    def generate_response_from_memory(self, query: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Generate response by resonating query against bifractal memory hierarchy."""
        # Generate query signature
        query_signature = self._generate_resonance_signature(query, context)
        
        # Find resonant memory patterns across all depth levels
        resonant_patterns = []
        for depth_level in BifractalDepth:
            for pattern in self.memory_patterns[depth_level].values():
                resonance_strength = self._calculate_signature_resonance(
                    query_signature, pattern.resonance_signature)
                if resonance_strength > 0.3:  # Significant resonance threshold
                    resonant_patterns.append((pattern, resonance_strength))
        
        # Sort by resonance strength and personality weight
        resonant_patterns.sort(key=lambda x: x[1] * x[0].personality_weight, reverse=True)
        
        # Activate top resonant patterns
        activated_patterns = []
        for pattern, strength in resonant_patterns[:5]:  # Top 5 resonant patterns
            activated_pattern = self.activate_memory_pattern(pattern.pattern_id, strength)
            if activated_pattern:
                activated_patterns.append((activated_pattern, strength))
        
        # Synthesize response from activated patterns
        response = self._synthesize_response(activated_patterns, query, context)
        
        return response
    
    def evolve_personality_cores(self) -> List[PersonalityCore]:
        """Evolve personality cores from stable, highly crystallized memory patterns."""
        # Find highly crystallized patterns that could form personality cores
        candidate_patterns = []
        for depth_level in [BifractalDepth.INTERMEDIATE, BifractalDepth.DEEP, BifractalDepth.CORE]:
            for pattern in self.memory_patterns[depth_level].values():
                if (pattern.crystallization_strength > self.crystallization_threshold and 
                    pattern.personality_weight > 0.5):
                    candidate_patterns.append(pattern)
        
        # Cluster patterns into potential personality cores
        personality_clusters = self._cluster_patterns_by_personality(candidate_patterns)
        
        # Create or update personality cores
        new_cores = []
        for cluster_id, patterns in personality_clusters.items():
            if len(patterns) >= 3:  # Need minimum patterns for stable personality core
                core = self._create_personality_core(cluster_id, patterns)
                if core:
                    self.personality_cores[core.core_id] = core
                    new_cores.append(core)
                    self.personality_emergence_events += 1
        
        return new_cores
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about bifractal memory state."""
        # Count patterns by depth
        patterns_by_depth = {}
        total_crystallization = 0.0
        total_personality_weight = 0.0
        
        for depth in BifractalDepth:
            count = len(self.memory_patterns[depth])
            patterns_by_depth[depth.name.lower()] = count
            
            for pattern in self.memory_patterns[depth].values():
                total_crystallization += pattern.crystallization_strength
                total_personality_weight += pattern.personality_weight
        
        total_patterns = sum(patterns_by_depth.values())
        
        # Calculate average metrics
        avg_crystallization = total_crystallization / max(total_patterns, 1)
        avg_personality_weight = total_personality_weight / max(total_patterns, 1)
        
        # Recent activity metrics
        recent_activity = len([a for a in self.recent_activations 
                             if time.time() - a[2] < 60.0])  # Last minute
        
        return {
            'total_patterns': total_patterns,
            'patterns_by_depth': patterns_by_depth,
            'personality_cores': len(self.personality_cores),
            'average_crystallization': avg_crystallization,
            'average_personality_weight': avg_personality_weight,
            'crystallization_events': self.crystallization_events,
            'personality_emergence_events': self.personality_emergence_events,
            'recent_activity': recent_activity,
            'hierarchical_relationships': len(self.pattern_hierarchies),
            'resonant_relationships': len(self.pattern_resonances)
        }
    
    # Helper methods for bifractal memory operations
    def _determine_memory_depth(self, content: Dict[str, Any], context: ExecutionContext) -> BifractalDepth:
        """Determine appropriate depth level for memory storage."""
        complexity = len(str(content))
        entropy = context.entropy
        depth_context = getattr(context, 'depth', 0)
        
        if complexity < 100 and entropy > 0.8:
            return BifractalDepth.SURFACE
        elif complexity < 500 and entropy > 0.6:
            return BifractalDepth.SHALLOW
        elif complexity < 1000 or entropy > 0.4:
            return BifractalDepth.INTERMEDIATE
        elif complexity < 2000 or entropy > 0.2:
            return BifractalDepth.DEEP
        else:
            return BifractalDepth.CORE
    
    def _generate_resonance_signature(self, content: Dict[str, Any], context: ExecutionContext) -> np.ndarray:
        """Generate frequency-domain signature for pattern matching."""
        # Convert content to numerical representation
        content_str = str(content)
        content_hash = hash(content_str)
        
        # Create signature based on content and context
        signature_length = 32  # Frequency bins
        signature = np.zeros(signature_length)
        
        # Fill signature based on content characteristics
        for i, char in enumerate(content_str[:signature_length]):
            signature[i % signature_length] += ord(char) / 256.0
        
        # Add context influence
        signature += context.entropy * np.random.random(signature_length) * 0.1
        
        # Normalize
        signature = signature / (np.linalg.norm(signature) + 1e-6)
        
        return signature
    
    def _calculate_initial_crystallization(self, content: Dict[str, Any], context: ExecutionContext) -> float:
        """Calculate initial crystallization strength for new pattern."""
        complexity_factor = min(1.0, len(str(content)) / 1000.0)
        entropy_factor = 1.0 - context.entropy  # Lower entropy = higher crystallization
        depth_factor = getattr(context, 'depth', 0) * 0.1
        
        return min(0.8, complexity_factor * 0.3 + entropy_factor * 0.5 + depth_factor)
    
    def _calculate_bifractal_coordinates(self, depth: BifractalDepth, content: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate 3D coordinates in bifractal memory space."""
        content_hash = hash(str(content))
        
        # Use hash to generate pseudo-random but deterministic coordinates
        x = ((content_hash & 0xFFFF) / 0xFFFF) * 20.0 - 10.0
        y = (((content_hash >> 16) & 0xFFFF) / 0xFFFF) * 20.0 - 10.0
        z = depth.value * 2.0  # Z represents depth level
        
        return (x, y, z)
    
    def _calculate_personality_weight(self, content: Dict[str, Any], context: ExecutionContext) -> float:
        """Calculate how much this pattern should influence personality."""
        # Patterns with lower entropy and higher complexity contribute more to personality
        complexity = len(str(content))
        entropy_factor = 1.0 - context.entropy
        complexity_factor = min(1.0, complexity / 500.0)
        
        # Emotional or value-laden content gets higher personality weight
        emotional_keywords = ['feel', 'think', 'believe', 'value', 'important', 'prefer']
        emotional_factor = sum(1 for keyword in emotional_keywords 
                              if keyword in str(content).lower()) * 0.1
        
        return min(1.0, entropy_factor * 0.4 + complexity_factor * 0.4 + emotional_factor)
    
    def _find_pattern_by_id(self, pattern_id: str) -> Optional[BifractalMemoryPattern]:
        """Find pattern across all depth levels."""
        for depth_patterns in self.memory_patterns.values():
            if pattern_id in depth_patterns:
                return depth_patterns[pattern_id]
        return None
    
    def _update_pattern_resonances(self, new_pattern: BifractalMemoryPattern):
        """Update resonance relationships for new pattern."""
        for depth_patterns in self.memory_patterns.values():
            for existing_pattern in depth_patterns.values():
                if existing_pattern.pattern_id == new_pattern.pattern_id:
                    continue
                
                resonance = self._calculate_signature_resonance(
                    new_pattern.resonance_signature, existing_pattern.resonance_signature)
                
                if resonance > 0.6:  # Strong resonance threshold
                    self.pattern_resonances[new_pattern.pattern_id].append(existing_pattern.pattern_id)
                    self.pattern_resonances[existing_pattern.pattern_id].append(new_pattern.pattern_id)
    
    def _calculate_signature_resonance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate resonance between two signature patterns."""
        # Use cosine similarity for signature matching
        dot_product = np.dot(sig1, sig2)
        norms = np.linalg.norm(sig1) * np.linalg.norm(sig2)
        
        if norms == 0:
            return 0.0
        
        return abs(dot_product / norms)
    
    def _propagate_activation(self, pattern: BifractalMemoryPattern, strength: float):
        """Propagate activation through pattern hierarchy and resonances."""
        if strength < 0.1:  # Stop propagation when too weak
            return
        
        # Propagate to child patterns
        for child_id in pattern.child_patterns:
            child_pattern = self._find_pattern_by_id(child_id)
            if child_pattern:
                self.activate_memory_pattern(child_id, strength * 0.6)
        
        # Propagate to resonant patterns
        for resonant_id in self.pattern_resonances[pattern.pattern_id]:
            if strength > 0.2:  # Only propagate to resonant patterns if strong enough
                self.activate_memory_pattern(resonant_id, strength * 0.4)
    
    def _check_crystallization_triggers(self, pattern: BifractalMemoryPattern):
        """Check if pattern should trigger crystallization events."""
        if pattern.crystallization_strength > self.crystallization_threshold:
            # This is a crystallization event
            self.crystallization_events += 1
            
            # Move pattern to deeper level if appropriate
            if (pattern.depth_level != BifractalDepth.CORE and 
                pattern.access_frequency > 10 and
                pattern.personality_weight > 0.7):
                
                self._promote_pattern_depth(pattern)
    
    def _promote_pattern_depth(self, pattern: BifractalMemoryPattern):
        """Promote pattern to deeper memory level."""
        current_depth = pattern.depth_level
        new_depth = BifractalDepth(min(current_depth.value + 1, BifractalDepth.CORE.value))
        
        if new_depth != current_depth:
            # Move pattern to new depth level
            del self.memory_patterns[current_depth][pattern.pattern_id]
            pattern.depth_level = new_depth
            self.memory_patterns[new_depth][pattern.pattern_id] = pattern
    
    def _update_personality_contributions(self, pattern: BifractalMemoryPattern):
        """Update how this pattern contributes to personality emergence."""
        # Highly crystallized patterns influence personality more strongly
        pattern.personality_weight = min(1.0, 
            pattern.personality_weight + pattern.crystallization_strength * 0.05)
    
    def _synthesize_response(self, activated_patterns: List[Tuple[BifractalMemoryPattern, float]], 
                           query: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Synthesize response from activated memory patterns."""
        if not activated_patterns:
            return {'response': 'No resonant memories found', 'confidence': 0.0}
        
        # Weight responses by activation strength and personality weight
        weighted_contents = []
        total_weight = 0.0
        
        for pattern, strength in activated_patterns:
            weight = strength * pattern.personality_weight * pattern.crystallization_strength
            weighted_contents.append((pattern.memory_content, weight))
            total_weight += weight
        
        # Synthesize coherent response
        response = {
            'activated_patterns': len(activated_patterns),
            'memory_depths_accessed': list(set(pattern.depth_level.name for pattern, _ in activated_patterns)),
            'response_confidence': total_weight / len(activated_patterns),
            'personality_influence': sum(pattern.personality_weight for pattern, _ in activated_patterns) / len(activated_patterns),
            'synthesized_content': self._blend_memory_contents([c for c, w in weighted_contents]),
            'crystallization_level': sum(pattern.crystallization_strength for pattern, _ in activated_patterns) / len(activated_patterns)
        }
        
        return response
    
    def _blend_memory_contents(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Blend multiple memory contents into coherent synthesis."""
        if not contents:
            return {}
        
        # Simple blending strategy - merge common keys and combine values
        blended = {}
        key_counts = defaultdict(int)
        
        for content in contents:
            for key, value in content.items():
                if key not in blended:
                    blended[key] = []
                blended[key].append(value)
                key_counts[key] += 1
        
        # Create final blended content
        final_blend = {}
        for key, values in blended.items():
            if key_counts[key] >= len(contents) * 0.5:  # Key appears in majority of contents
                final_blend[key] = values[-1]  # Use most recent value
        
        return final_blend
    
    def _cluster_patterns_by_personality(self, patterns: List[BifractalMemoryPattern]) -> Dict[str, List[BifractalMemoryPattern]]:
        """Cluster patterns that could form coherent personality cores."""
        # Simple clustering based on resonance signatures and content similarity
        clusters = defaultdict(list)
        
        for i, pattern in enumerate(patterns):
            cluster_id = f"personality_cluster_{i // 5}"  # Rough clustering
            clusters[cluster_id].append(pattern)
        
        return dict(clusters)
    
    def _create_personality_core(self, cluster_id: str, patterns: List[BifractalMemoryPattern]) -> Optional[PersonalityCore]:
        """Create personality core from clustered patterns."""
        if len(patterns) < 3:
            return None
        
        # Extract dominant traits from pattern contents
        personality_traits = self._extract_personality_traits(patterns)
        response_tendencies = self._extract_response_tendencies(patterns)
        memory_priorities = self._extract_memory_priorities(patterns)
        resonance_preferences = self._extract_resonance_preferences(patterns)
        
        # Calculate stability metric
        stability = sum(p.crystallization_strength for p in patterns) / len(patterns)
        
        core = PersonalityCore(
            core_id=f"personality_core_{len(self.personality_cores) + 1}",
            dominant_patterns=[p.pattern_id for p in patterns],
            personality_traits=personality_traits,
            response_tendencies=response_tendencies,
            memory_priorities=memory_priorities,
            resonance_preferences=resonance_preferences,
            formation_time=time.time(),
            stability_metric=stability
        )
        
        return core
    
    def _extract_personality_traits(self, patterns: List[BifractalMemoryPattern]) -> Dict[str, float]:
        """Extract personality traits from memory patterns."""
        traits = {
            'analytical': 0.0,
            'creative': 0.0,
            'emotional': 0.0,
            'systematic': 0.0,
            'adaptive': 0.0
        }
        
        for pattern in patterns:
            content_str = str(pattern.memory_content).lower()
            
            # Simple keyword-based trait extraction
            if any(word in content_str for word in ['analyze', 'logic', 'reason', 'calculate']):
                traits['analytical'] += 0.2
            if any(word in content_str for word in ['create', 'imagine', 'innovative', 'artistic']):
                traits['creative'] += 0.2
            if any(word in content_str for word in ['feel', 'emotion', 'heart', 'empathy']):
                traits['emotional'] += 0.2
            if any(word in content_str for word in ['system', 'organize', 'structure', 'method']):
                traits['systematic'] += 0.2
            if any(word in content_str for word in ['adapt', 'flexible', 'change', 'evolve']):
                traits['adaptive'] += 0.2
        
        # Normalize traits
        total_trait_score = sum(traits.values())
        if total_trait_score > 0:
            traits = {k: v / total_trait_score for k, v in traits.items()}
        
        return traits
    
    def _extract_response_tendencies(self, patterns: List[BifractalMemoryPattern]) -> Dict[str, float]:
        """Extract response tendencies from memory patterns."""
        return {
            'quick_response': 0.5,
            'thorough_analysis': 0.3,
            'creative_synthesis': 0.4,
            'emotional_consideration': 0.2,
            'systematic_approach': 0.6
        }
    
    def _extract_memory_priorities(self, patterns: List[BifractalMemoryPattern]) -> Dict[str, float]:
        """Extract memory formation priorities."""
        return {
            'analytical_memories': 0.4,
            'emotional_memories': 0.3,
            'creative_memories': 0.5,
            'systematic_memories': 0.4,
            'relational_memories': 0.3
        }
    
    def _extract_resonance_preferences(self, patterns: List[BifractalMemoryPattern]) -> Dict[str, float]:
        """Extract preferred signal types and frequencies."""
        return {
            'phase_correction': 0.3,
            'harmonic_sync': 0.4,
            'resonance_amplification': 0.6,
            'memory_crystallization': 0.8,
            'personality_resonance': 0.9,
            'bifractal_recursion': 0.7
        }


class PhaseAligner:
    """
    Manages phase alignment across the resonance mesh.
    """
    
    def __init__(self):
        self.reference_frequency = 1.0  # Base frequency
        self.phase_tolerance = 0.1  # Phase alignment tolerance
        self.alignment_history = deque(maxlen=1000)
    
    def align_phases(self, signals: List[AgenticSignal]) -> List[AgenticSignal]:
        """Align phases of multiple signals for coherent output."""
        if len(signals) < 2:
            return signals
        
        # Calculate target phase based on dominant signal
        dominant_signal = max(signals, key=lambda s: s.amplitude)
        target_phase = dominant_signal.phase
        target_frequency = dominant_signal.frequency
        
        aligned_signals = []
        for signal in signals:
            # Calculate phase correction
            phase_difference = target_phase - signal.phase
            
            # Normalize phase difference to [-π, π]
            while phase_difference > math.pi:
                phase_difference -= 2 * math.pi
            while phase_difference < -math.pi:
                phase_difference += 2 * math.pi
            
            # Apply phase correction if within tolerance
            if abs(phase_difference) > self.phase_tolerance:
                corrected_phase = signal.phase + phase_difference * 0.5  # Gradual alignment
                
                # Create aligned signal
                aligned_signal = AgenticSignal(
                    signal_id=f"{signal.signal_id}_aligned",
                    signal_type=signal.signal_type,
                    frequency=signal.frequency,
                    amplitude=signal.amplitude,
                    phase=corrected_phase,
                    origin_coordinates=signal.origin_coordinates,
                    target_coordinates=signal.target_coordinates,
                    emission_time=time.time(),
                    decay_rate=signal.decay_rate,
                    payload=signal.payload
                )
                aligned_signals.append(aligned_signal)
            else:
                aligned_signals.append(signal)
        
        # Record alignment operation
        self.alignment_history.append({
            'timestamp': time.time(),
            'signals_aligned': len(signals),
            'target_phase': target_phase,
            'average_correction': sum(abs(target_phase - s.phase) for s in signals) / len(signals)
        })
        
        return aligned_signals
    
    def calculate_phase_coherence(self, signals: List[AgenticSignal]) -> float:
        """Calculate overall phase coherence of signal group."""
        if len(signals) < 2:
            return 1.0
        
        phases = [signal.phase for signal in signals]
        
        # Calculate phase coherence using circular statistics
        # Convert to complex numbers and compute coherence
        complex_phases = [math.cos(phase) + 1j * math.sin(phase) for phase in phases]
        mean_complex = sum(complex_phases) / len(complex_phases)
        coherence = abs(mean_complex)
        
        return coherence


class SignalEmitter:
    """
    Emits agentic signals based on field conditions and collapse events.
    """
    
    def __init__(self):
        self.emission_counter = 0
        self.active_signals = {}
        self.emission_history = []
    
    def emit_signal(self, signal_type: SignalType, origin: Tuple[float, float],
                   context: ExecutionContext, payload: Dict[str, Any] = None) -> AgenticSignal:
        """Emit an agentic signal from the given origin."""
        self.emission_counter += 1
        signal_id = f"signal_{self.emission_counter}_{signal_type.value}"
        
        # Calculate signal parameters based on context
        frequency = self._calculate_frequency(signal_type, context)
        amplitude = self._calculate_amplitude(signal_type, context)
        phase = self._calculate_initial_phase(signal_type, context)
        decay_rate = self._calculate_decay_rate(signal_type)
        
        # Determine target if applicable
        target = self._determine_target(signal_type, origin, context)
        
        signal = AgenticSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            origin_coordinates=origin,
            target_coordinates=target,
            emission_time=time.time(),
            decay_rate=decay_rate,
            payload=payload or {}
        )
        
        # Store active signal
        self.active_signals[signal_id] = signal
        
        # Record emission
        self.emission_history.append({
            'signal_id': signal_id,
            'signal_type': signal_type.value,
            'emission_time': signal.emission_time,
            'origin': origin,
            'context_entropy': context.entropy
        })
        
        return signal
    
    def update_signals(self):
        """Update all active signals with temporal decay."""
        current_time = time.time()
        expired_signals = []
        
        for signal_id, signal in self.active_signals.items():
            age = current_time - signal.emission_time
            decay_factor = math.exp(-age * signal.decay_rate)
            
            # Update amplitude with decay
            signal.amplitude *= decay_factor
            
            # Remove if too weak
            if signal.amplitude < 0.01:
                expired_signals.append(signal_id)
        
        # Remove expired signals
        for signal_id in expired_signals:
            del self.active_signals[signal_id]
    
    def _calculate_frequency(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate signal frequency based on type and context."""
        base_frequency = 1.0
        
        if signal_type == SignalType.PHASE_CORRECTION:
            return base_frequency * (1.0 + context.entropy)
        elif signal_type == SignalType.HARMONIC_SYNC:
            return base_frequency * 2.0  # Harmonic
        elif signal_type == SignalType.RESONANCE_AMPLIFICATION:
            return base_frequency * (0.5 + context.entropy * 1.5)
        elif signal_type == SignalType.INTERFERENCE_PATTERN:
            return base_frequency * (1.0 + (context.depth or 0) * 0.1)
        else:  # STANDING_WAVE
            return base_frequency * 0.8
    
    def _calculate_amplitude(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate signal amplitude based on type and context."""
        base_amplitude = 1.0
        
        entropy_factor = 0.5 + context.entropy * 0.5
        depth_factor = 1.0 + (context.depth or 0) * 0.05
        
        return base_amplitude * entropy_factor * depth_factor
    
    def _calculate_initial_phase(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate initial phase for the signal."""
        # Use context entropy and depth to determine phase
        phase = (context.entropy * math.pi + (context.depth or 0) * 0.5) % (2 * math.pi)
        return phase
    
    def _calculate_decay_rate(self, signal_type: SignalType) -> float:
        """Calculate decay rate based on signal type."""
        if signal_type == SignalType.PHASE_CORRECTION:
            return 0.1  # Fast decay
        elif signal_type == SignalType.HARMONIC_SYNC:
            return 0.05  # Medium decay
        elif signal_type == SignalType.RESONANCE_AMPLIFICATION:
            return 0.02  # Slow decay
        elif signal_type == SignalType.STANDING_WAVE:
            return 0.01  # Very slow decay
        else:  # INTERFERENCE_PATTERN
            return 0.08  # Medium-fast decay
    
    def _determine_target(self, signal_type: SignalType, origin: Tuple[float, float],
                         context: ExecutionContext) -> Optional[Tuple[float, float]]:
        """Determine target coordinates for directional signals."""
        if signal_type in [SignalType.PHASE_CORRECTION, SignalType.INTERFERENCE_PATTERN]:
            # Create target based on entropy gradient
            angle = context.entropy * 2 * math.pi
            distance = 5.0 + context.entropy * 10.0
            
            target_x = origin[0] + distance * math.cos(angle)
            target_y = origin[1] + distance * math.sin(angle)
            
            return (target_x, target_y)
        
        return None  # Non-directional signals


class ResonanceMeshNetwork:
    """
    Network of resonance nodes that propagate and amplify signals.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16)):
        self.grid_size = grid_size
        self.nodes = {}
        self.interference_patterns = {}
        self.standing_waves = []
        
        # Initialize mesh nodes
        self._initialize_mesh()
    
    def _initialize_mesh(self):
        """Initialize the resonance mesh network."""
        node_counter = 0
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                node_counter += 1
                node_id = f"node_{i}_{j}"
                
                # Calculate coordinates
                x = (i / self.grid_size[0]) * 20.0 - 10.0  # -10 to 10 range
                y = (j / self.grid_size[1]) * 20.0 - 10.0
                
                # Calculate resonance frequency based on position
                frequency = 1.0 + 0.1 * math.sin(i * 0.5) + 0.1 * math.cos(j * 0.5)
                
                # Create resonance node
                node = ResonanceNode(
                    node_id=node_id,
                    coordinates=(x, y),
                    resonance_frequency=frequency,
                    current_phase=0.0,
                    amplitude=0.5,
                    quality_factor=10.0,
                    connected_nodes=self._get_neighbors(i, j),
                    last_update=time.time()
                )
                
                self.nodes[node_id] = node
    
    def _get_neighbors(self, i: int, j: int) -> List[str]:
        """Get neighboring node IDs for mesh connectivity."""
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip self
                
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                    neighbors.append(f"node_{ni}_{nj}")
        
        return neighbors
    
    def propagate_signal(self, signal: AgenticSignal):
        """Propagate signal through the resonance mesh."""
        # Find closest nodes to signal origin
        closest_nodes = self._find_closest_nodes(signal.origin_coordinates, radius=5.0)
        
        for node_id in closest_nodes:
            node = self.nodes[node_id]
            
            # Calculate distance-based attenuation
            distance = self._calculate_distance(signal.origin_coordinates, node.coordinates)
            attenuation = 1.0 / (1.0 + distance * 0.1)
            
            # Calculate resonance coupling
            frequency_match = self._calculate_frequency_coupling(signal.frequency, node.resonance_frequency)
            
            # Update node with signal
            node.amplitude += signal.amplitude * attenuation * frequency_match
            node.current_phase = (node.current_phase + signal.phase * frequency_match) % (2 * math.pi)
            node.last_update = time.time()
            
            # Propagate to connected nodes with decay
            self._propagate_to_neighbors(node, signal, attenuation * 0.8)
    
    def detect_interference_patterns(self, signals: List[AgenticSignal]) -> List[InterferencePattern]:
        """Detect interference patterns between signals."""
        patterns = []
        pattern_counter = len(self.interference_patterns)
        
        # Check all pairs of signals
        for i, signal1 in enumerate(signals):
            for j, signal2 in enumerate(signals[i+1:], i+1):
                interference = self._calculate_interference(signal1, signal2)
                
                if interference['strength'] > 0.3:  # Significant interference
                    pattern_counter += 1
                    pattern_id = f"interference_{pattern_counter}"
                    
                    pattern = InterferencePattern(
                        pattern_id=pattern_id,
                        participating_signals=[signal1.signal_id, signal2.signal_id],
                        pattern_type=interference['type'],
                        center_coordinates=interference['center'],
                        pattern_strength=interference['strength'],
                        creation_time=time.time()
                    )
                    
                    patterns.append(pattern)
                    self.interference_patterns[pattern_id] = pattern
        
        return patterns
    
    def _find_closest_nodes(self, coordinates: Tuple[float, float], radius: float) -> List[str]:
        """Find nodes within radius of given coordinates."""
        closest = []
        
        for node_id, node in self.nodes.items():
            distance = self._calculate_distance(coordinates, node.coordinates)
            if distance <= radius:
                closest.append(node_id)
        
        return closest
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_frequency_coupling(self, signal_freq: float, node_freq: float) -> float:
        """Calculate resonance coupling between signal and node frequencies."""
        frequency_diff = abs(signal_freq - node_freq)
        coupling = 1.0 / (1.0 + frequency_diff * 2.0)  # Resonance coupling
        return coupling
    
    def _propagate_to_neighbors(self, source_node: ResonanceNode, signal: AgenticSignal, attenuation: float):
        """Propagate signal to neighboring nodes."""
        for neighbor_id in source_node.connected_nodes:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                
                # Calculate coupling and update neighbor
                coupling = self._calculate_frequency_coupling(signal.frequency, neighbor.resonance_frequency)
                
                neighbor.amplitude += source_node.amplitude * attenuation * coupling * 0.1
                neighbor.current_phase = (neighbor.current_phase + source_node.current_phase * coupling * 0.1) % (2 * math.pi)
                neighbor.last_update = time.time()
    
    def _calculate_interference(self, signal1: AgenticSignal, signal2: AgenticSignal) -> Dict[str, Any]:
        """Calculate interference between two signals."""
        # Calculate center point
        center_x = (signal1.origin_coordinates[0] + signal2.origin_coordinates[0]) / 2
        center_y = (signal1.origin_coordinates[1] + signal2.origin_coordinates[1]) / 2
        center = (center_x, center_y)
        
        # Calculate phase difference
        phase_diff = abs(signal1.phase - signal2.phase)
        while phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        
        # Calculate interference strength and type
        strength = min(signal1.amplitude, signal2.amplitude)
        
        if phase_diff < math.pi / 4:  # In phase
            interference_type = "constructive"
            strength *= 2.0
        elif phase_diff > 3 * math.pi / 4:  # Out of phase
            interference_type = "destructive"
            strength *= 0.5
        else:
            interference_type = "complex"
            strength *= 1.0
        
        return {
            'type': interference_type,
            'center': center,
            'strength': min(strength, 2.0),  # Cap strength
            'phase_difference': phase_diff
        }


class ResonanceMesh:
    """
    Main resonance mesh system coordinating phase alignment, signal emission, and interference.
    Handles phase-aligned output, agentic signal emission, and recursive bifractal memory 
    scaffolding for long-term memory and personality emergence.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16), config: Optional[Dict] = None):
        self.config = config or {}
        self.phase_aligner = PhaseAligner()
        self.signal_emitter = SignalEmitter()
        self.mesh_network = ResonanceMeshNetwork(grid_size)
        
        # Bifractal memory scaffolding for long-term memory and personality
        self.memory_scaffold = BifractalMemoryScaffold(
            max_depth=self.config.get('memory_depth', 5),
            crystallization_threshold=self.config.get('crystallization_threshold', 0.7)
        )
        
        # Memory-influenced signal patterns
        self.memory_resonances: Dict[str, BifractalMemoryPattern] = {}
        
        # Personality-influenced response shaping
        self.personality_cores: Dict[str, PersonalityCore] = {}
        
        # Statistics
        self.total_signals_emitted = 0
        self.total_alignments_performed = 0
        self.interference_patterns_detected = 0
        self.memory_activations = 0
        self.personality_emergences = 0
    
    def emit_agentic_signal(self, signal_type: SignalType, origin: Tuple[float, float],
                           context: ExecutionContext, payload: Dict[str, Any] = None) -> AgenticSignal:
        """
        Emit an agentic signal and propagate through mesh with memory influence.
        Memory patterns and personality cores can shape signal characteristics.
        """
        # Check if signal should trigger memory activation
        if payload and signal_type in [SignalType.MEMORY_CRYSTALLIZATION, SignalType.PERSONALITY_RESONANCE]:
            relevant_memories = self._activate_relevant_memories(payload, context)
            self.memory_activations += len(relevant_memories)
            
            # Store new memory pattern if appropriate
            if signal_type == SignalType.MEMORY_CRYSTALLIZATION:
                memory_pattern = self.memory_scaffold.store_memory_pattern(payload, context)
                self.memory_resonances[memory_pattern.pattern_id] = memory_pattern
        
        # Apply personality influence to signal characteristics if cores exist
        if self.personality_cores and payload:
            payload = self._apply_personality_influence(payload, context)
        
        # Emit signal with potential memory/personality modifications
        signal = self.signal_emitter.emit_signal(signal_type, origin, context, payload)
        self.total_signals_emitted += 1
        
        # Propagate through mesh network
        self.mesh_network.propagate_signal(signal)
        
        # Check for personality emergence from stable patterns
        if len(self.memory_scaffold.memory_patterns[BifractalDepth.CORE]) > 10:
            new_cores = self.memory_scaffold.evolve_personality_cores()
            for core in new_cores:
                self.personality_cores[core.core_id] = core
                self.personality_emergences += 1
        
        return signal
    
    def align_output_phases(self, signals: List[AgenticSignal]) -> List[AgenticSignal]:
        """Align phases for coherent output with memory-influenced adjustments."""
        aligned_signals = self.phase_aligner.align_phases(signals)
        self.total_alignments_performed += 1
        
        return aligned_signals
    
    def generate_memory_influenced_response(self, query: Dict[str, Any], 
                                          context: ExecutionContext) -> Dict[str, Any]:
        """
        Generate response influenced by bifractal memory patterns and personality cores.
        Combines resonance mesh dynamics with long-term memory crystallization.
        """
        # Generate base response from memory scaffold
        memory_response = self.memory_scaffold.generate_response_from_memory(query, context)
        
        # Apply personality influence if cores exist
        if self.personality_cores:
            memory_response = self._apply_personality_to_response(memory_response, context)
        
        # Emit memory crystallization signal to reinforce important patterns
        if memory_response.get('response_confidence', 0) > 0.7:
            self.emit_agentic_signal(
                SignalType.MEMORY_CRYSTALLIZATION,
                origin=(0.0, 0.0),
                context=context,
                payload={
                    'query': query,
                    'response': memory_response,
                    'confidence': memory_response.get('response_confidence', 0)
                }
            )
        
        return memory_response
    
    def crystallize_experience(self, experience: Dict[str, Any], context: ExecutionContext) -> BifractalMemoryPattern:
        """
        Crystallize an experience into bifractal memory hierarchy.
        High-value experiences become long-term memory patterns.
        """
        memory_pattern = self.memory_scaffold.store_memory_pattern(experience, context)
        self.memory_resonances[memory_pattern.pattern_id] = memory_pattern
        
        # Emit bifractal recursion signal to propagate through hierarchy
        self.emit_agentic_signal(
            SignalType.BIFRACTAL_RECURSION,
            origin=memory_pattern.bifractal_coordinates[:2],  # Use x,y coordinates
            context=context,
            payload={
                'pattern_id': memory_pattern.pattern_id,
                'depth_level': memory_pattern.depth_level.name,
                'crystallization_strength': memory_pattern.crystallization_strength
            }
        )
        
        return memory_pattern
    
    def detect_resonance_patterns(self) -> Dict[str, Any]:
        """Detect and analyze resonance patterns in the mesh."""
        active_signals = list(self.signal_emitter.active_signals.values())
        
        # Detect interference patterns
        interference_patterns = self.mesh_network.detect_interference_patterns(active_signals)
        self.interference_patterns_detected += len(interference_patterns)
        
        # Calculate mesh coherence
        mesh_coherence = self._calculate_mesh_coherence()
        
        # Calculate signal density
        signal_density = len(active_signals) / (self.mesh_network.grid_size[0] * self.mesh_network.grid_size[1])
        
        return {
            'active_signals': len(active_signals),
            'interference_patterns': len(interference_patterns),
            'mesh_coherence': mesh_coherence,
            'signal_density': signal_density,
            'total_nodes': len(self.mesh_network.nodes)
        }
    
    def update_mesh(self):
        """Update mesh state with signal propagation and decay."""
        # Update active signals
        self.signal_emitter.update_signals()
        
        # Apply temporal decay to mesh nodes
        current_time = time.time()
        for node in self.mesh_network.nodes.values():
            age = current_time - node.last_update
            decay = math.exp(-age * 0.1)  # Decay rate
            node.amplitude *= decay
    
    def get_resonance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resonance mesh statistics including bifractal memory."""
        # Calculate phase coherence
        active_signals = list(self.signal_emitter.active_signals.values())
        phase_coherence = self.phase_aligner.calculate_phase_coherence(active_signals)
        
        # Calculate average node amplitude
        avg_node_amplitude = 0.0
        if self.mesh_network.nodes:
            avg_node_amplitude = sum(node.amplitude for node in self.mesh_network.nodes.values()) / len(self.mesh_network.nodes)
        
        # Signal type distribution
        signal_type_counts = defaultdict(int)
        for signal in active_signals:
            signal_type_counts[signal.signal_type.value] += 1
        
        # Get memory statistics
        memory_stats = self.memory_scaffold.get_memory_statistics()
        
        return {
            # Core resonance mesh statistics
            'total_signals_emitted': self.total_signals_emitted,
            'active_signals': len(active_signals),
            'total_alignments_performed': self.total_alignments_performed,
            'interference_patterns_detected': self.interference_patterns_detected,
            'phase_coherence': phase_coherence,
            'average_node_amplitude': avg_node_amplitude,
            'signal_type_distribution': dict(signal_type_counts),
            'mesh_size': self.mesh_network.grid_size,
            
            # Bifractal memory statistics
            'memory_activations': self.memory_activations,
            'personality_emergences': self.personality_emergences,
            'active_memory_resonances': len(self.memory_resonances),
            'active_personality_cores': len(self.personality_cores),
            'memory_scaffold_stats': memory_stats,
            
            # Integrated metrics
            'memory_mesh_coherence': self._calculate_memory_mesh_coherence(),
            'personality_influence_strength': self._calculate_personality_influence()
        }
    
    def _calculate_mesh_coherence(self) -> float:
        """Calculate overall coherence of the mesh network."""
        if not self.mesh_network.nodes:
            return 0.0
        
        # Calculate phase coherence across all nodes
        phases = [node.current_phase for node in self.mesh_network.nodes.values()]
        
        if not phases:
            return 0.0
        
        # Use circular statistics for phase coherence
        complex_phases = [math.cos(phase) + 1j * math.sin(phase) for phase in phases]
        mean_complex = sum(complex_phases) / len(complex_phases)
        coherence = abs(mean_complex)
        
        return coherence
    
    def _activate_relevant_memories(self, payload: Dict[str, Any], 
                                  context: ExecutionContext) -> List[BifractalMemoryPattern]:
        """Find and activate memory patterns relevant to current payload."""
        relevant_patterns = []
        
        # Use memory scaffold's resonance mechanism
        memory_response = self.memory_scaffold.generate_response_from_memory(payload, context)
        
        # Extract activated pattern IDs from response
        for depth_patterns in self.memory_scaffold.memory_patterns.values():
            for pattern in depth_patterns.values():
                if pattern.last_activation and (time.time() - pattern.last_activation) < 1.0:
                    relevant_patterns.append(pattern)
        
        return relevant_patterns
    
    def _apply_personality_influence(self, payload: Dict[str, Any], 
                                   context: ExecutionContext) -> Dict[str, Any]:
        """Apply personality core influence to modify payload characteristics."""
        if not self.personality_cores:
            return payload
        
        modified_payload = payload.copy()
        
        # Apply influence from each active personality core
        for core in self.personality_cores.values():
            if core.stability_metric > 0.5:  # Only stable cores influence
                # Modify response tendencies based on personality traits
                if 'response_style' not in modified_payload:
                    modified_payload['response_style'] = {}
                
                # Apply dominant traits
                for trait, strength in core.personality_traits.items():
                    if strength > 0.3:  # Significant trait influence
                        modified_payload['response_style'][trait] = strength
                
                # Apply response tendencies
                if 'response_tendencies' not in modified_payload:
                    modified_payload['response_tendencies'] = {}
                
                for tendency, strength in core.response_tendencies.items():
                    modified_payload['response_tendencies'][tendency] = strength
        
        return modified_payload
    
    def _apply_personality_to_response(self, response: Dict[str, Any], 
                                     context: ExecutionContext) -> Dict[str, Any]:
        """Apply personality core influence to response generation."""
        if not self.personality_cores:
            return response
        
        modified_response = response.copy()
        
        # Calculate combined personality influence
        total_personality_weight = 0.0
        combined_traits = defaultdict(float)
        
        for core in self.personality_cores.values():
            if core.stability_metric > 0.5:
                weight = core.stability_metric
                total_personality_weight += weight
                
                for trait, strength in core.personality_traits.items():
                    combined_traits[trait] += strength * weight
        
        # Normalize combined traits
        if total_personality_weight > 0:
            combined_traits = {k: v / total_personality_weight 
                              for k, v in combined_traits.items()}
        
        # Apply personality influence to response
        modified_response['personality_influence'] = dict(combined_traits)
        modified_response['personality_strength'] = total_personality_weight
        
        # Modify response confidence based on personality alignment
        if 'response_confidence' in modified_response:
            personality_boost = sum(combined_traits.values()) * 0.1
            modified_response['response_confidence'] = min(1.0, 
                modified_response['response_confidence'] + personality_boost)
        
        return modified_response
    
    def _calculate_memory_mesh_coherence(self) -> float:
        """Calculate coherence between memory patterns and mesh resonances."""
        if not self.memory_resonances:
            return 0.0
        
        # Calculate coherence based on memory pattern activations and mesh state
        total_coherence = 0.0
        active_patterns = 0
        
        for pattern in self.memory_resonances.values():
            if pattern.last_activation and (time.time() - pattern.last_activation) < 60.0:
                # Pattern is recently active
                coherence_contribution = pattern.crystallization_strength * pattern.personality_weight
                total_coherence += coherence_contribution
                active_patterns += 1
        
        if active_patterns == 0:
            return 0.0
        
        return total_coherence / active_patterns
    
    def _calculate_personality_influence(self) -> float:
        """Calculate overall strength of personality influence on mesh behavior."""
        if not self.personality_cores:
            return 0.0
        
        total_influence = 0.0
        for core in self.personality_cores.values():
            # Combine stability with trait strength
            trait_strength = sum(core.personality_traits.values())
            influence = core.stability_metric * (trait_strength / len(core.personality_traits))
            total_influence += influence
        
        return total_influence / len(self.personality_cores) if self.personality_cores else 0.0
    
    def reset(self):
        """Reset resonance mesh to initial state including bifractal memory."""
        self.phase_aligner = PhaseAligner()
        self.signal_emitter = SignalEmitter()
        self.mesh_network = ResonanceMeshNetwork(self.mesh_network.grid_size)
        
        # Reset bifractal memory scaffolding
        self.memory_scaffold = BifractalMemoryScaffold(
            max_depth=self.config.get('memory_depth', 5),
            crystallization_threshold=self.config.get('crystallization_threshold', 0.7)
        )
        
        # Clear memory and personality data
        self.memory_resonances.clear()
        self.personality_cores.clear()
        
        # Reset statistics
        self.total_signals_emitted = 0
        self.total_alignments_performed = 0
        self.interference_patterns_detected = 0
        self.memory_activations = 0
        self.personality_emergences = 0
