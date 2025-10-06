"""
PAC-Native Superfluid Memory for GAIA v3.0
Built on Fracton SDK for physics-governed superfluid dynamics with automatic conservation.
All memory operations maintain f(parent) = Σf(children) through native PAC regulation.
"""

import numpy as np
import time
import math
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Add Fracton SDK path
sys.path.append('../../../../fracton')

# Import PAC-native Fracton SDK for memory operations (required)
import fracton
from fracton import (
    # Core PAC-native memory components
    PhysicsMemoryField, MemoryField,
    PhysicsRecursiveExecutor,
    # PAC regulation and validation
    pac_recursive, validate_pac_conservation,
    enable_pac_self_regulation, get_system_pac_metrics,
    # Physics primitives for superfluid dynamics
    klein_gordon_evolution, enforce_pac_conservation,
    resonance_field_interaction
)
from fracton.core.recursive_engine import ExecutionContext


@dataclass
class MemoryImprint:
    """Represents a stable structure in superfluid memory."""
    structure_id: str
    field_coordinates: Tuple[float, float, float]  # 3D position
    entropy_signature: float
    stability_score: float
    phase_coherence: float
    recursion_depth: int
    temporal_decay: float
    creation_time: float
    last_reinforcement: float
    reinforcement_count: int
    vortex_strength: float = 0.0


@dataclass
class VortexPattern:
    """Represents a coherent feedback loop in memory."""
    pattern_id: str
    center_coordinates: Tuple[float, float, float]
    radius: float
    angular_velocity: float
    stability_index: float
    participating_imprints: List[str]
    temporal_symmetry: float
    creation_time: float


class MemoryFieldTensor:
    """
    Dynamic 3D+ tensor storing symbolic field attractors.
    Functions as quantum-esque field trace of symbolic ancestry.
    """
    
    def __init__(self, shape: Tuple[int, int, int] = (32, 32, 16)):
        self.shape = shape
        self.field = np.zeros(shape, dtype=np.float32)
        self.entropy_density = np.zeros(shape, dtype=np.float32)
        self.phase_coherence = np.zeros(shape, dtype=np.float32)
        self.recursion_signature = np.zeros(shape, dtype=np.float32)
        
        # Track field evolution
        self.field_history = []
        self.update_count = 0
    
    def add_imprint(self, imprint: MemoryImprint, intensity: float = 1.0):
        """Add memory imprint to field tensor."""
        x, y, z = imprint.field_coordinates
        
        # Map coordinates to tensor indices
        ix = int(abs(x) * self.shape[0] / 20) % self.shape[0]
        iy = int(abs(y) * self.shape[1] / 20) % self.shape[1] 
        iz = int(abs(z) * self.shape[2] / 10) % self.shape[2]
        
        # Create Gaussian spread around imprint location
        sigma = 2.0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-2, 3):
                    nx, ny, nz = (ix + dx) % self.shape[0], (iy + dy) % self.shape[1], (iz + dz) % self.shape[2]
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    weight = intensity * math.exp(-distance*distance / (2*sigma*sigma))
                    
                    # Update field components
                    self.field[nx, ny, nz] += weight * imprint.stability_score
                    self.entropy_density[nx, ny, nz] += weight * imprint.entropy_signature
                    self.phase_coherence[nx, ny, nz] += weight * imprint.phase_coherence
                    self.recursion_signature[nx, ny, nz] += weight * imprint.recursion_depth / 10.0
    
    def apply_temporal_decay(self, decay_rate: float = 0.99):
        """Apply temporal decay to memory field."""
        self.field *= decay_rate
        self.entropy_density *= decay_rate
        self.phase_coherence *= decay_rate
        self.recursion_signature *= decay_rate
    
    def get_field_strength(self, coordinates: Tuple[float, float, float]) -> float:
        """Get field strength at given coordinates."""
        x, y, z = coordinates
        ix = int(abs(x) * self.shape[0] / 20) % self.shape[0]
        iy = int(abs(y) * self.shape[1] / 20) % self.shape[1]
        iz = int(abs(z) * self.shape[2] / 10) % self.shape[2]
        
        return float(self.field[ix, iy, iz])
    
    def find_attractors(self, threshold: float = 0.5) -> List[Tuple[int, int, int]]:
        """Find high-intensity regions that act as attractors."""
        attractors = []
        
        for ix in range(self.shape[0]):
            for iy in range(self.shape[1]):
                for iz in range(self.shape[2]):
                    if self.field[ix, iy, iz] > threshold:
                        # Check if this is a local maximum
                        is_maximum = True
                        current_val = self.field[ix, iy, iz]
                        
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    
                                    nx = (ix + dx) % self.shape[0]
                                    ny = (iy + dy) % self.shape[1]
                                    nz = (iz + dz) % self.shape[2]
                                    
                                    if self.field[nx, ny, nz] > current_val:
                                        is_maximum = False
                                        break
                                if not is_maximum:
                                    break
                            if not is_maximum:
                                break
                        
                        if is_maximum:
                            attractors.append((ix, iy, iz))
        
        return attractors


class StabilityEvaluator:
    """
    Analyzes incoming structures for long-term memory viability.
    Uses entropy convergence and recurrence frequency.
    """
    
    def __init__(self):
        self.structure_history = defaultdict(list)
        self.stability_threshold = 0.6
        self.recurrence_window = 100  # Time window for recurrence analysis
    
    def evaluate_stability(self, structure_data: Dict[str, Any], 
                          context: ExecutionContext) -> float:
        """Evaluate if structure should be stored in long-term memory."""
        structure_type = structure_data.get('type', 'unknown')
        entropy_resolved = structure_data.get('entropy_resolved', 0.0)
        
        # Record structure occurrence
        current_time = time.time()
        self.structure_history[structure_type].append({
            'timestamp': current_time,
            'entropy': entropy_resolved,
            'depth': context.depth or 0,
            'context_entropy': context.entropy
        })
        
        # Clean old history
        cutoff_time = current_time - self.recurrence_window
        self.structure_history[structure_type] = [
            entry for entry in self.structure_history[structure_type]
            if entry['timestamp'] > cutoff_time
        ]
        
        # Calculate stability metrics
        recurrence_frequency = len(self.structure_history[structure_type])
        entropy_convergence = self._calculate_entropy_convergence(structure_type)
        temporal_stability = self._calculate_temporal_stability(structure_type)
        depth_consistency = self._calculate_depth_consistency(structure_type)
        
        # Combine metrics
        stability_score = (
            0.3 * min(recurrence_frequency / 10.0, 1.0) +  # Frequency component
            0.25 * entropy_convergence +                    # Convergence component
            0.25 * temporal_stability +                     # Temporal component
            0.2 * depth_consistency                        # Depth component
        )
        
        return min(stability_score, 1.0)
    
    def _calculate_entropy_convergence(self, structure_type: str) -> float:
        """Calculate how much entropy values are converging."""
        history = self.structure_history[structure_type]
        if len(history) < 3:
            return 0.0
        
        entropies = [entry['entropy'] for entry in history[-10:]]  # Last 10 entries
        if len(entropies) < 2:
            return 0.0
        
        # Calculate variance - lower variance means more convergence
        variance = np.var(entropies)
        convergence = 1.0 / (1.0 + variance)  # Inverse relationship
        
        return convergence
    
    def _calculate_temporal_stability(self, structure_type: str) -> float:
        """Calculate temporal consistency of structure appearance."""
        history = self.structure_history[structure_type]
        if len(history) < 3:
            return 0.0
        
        timestamps = [entry['timestamp'] for entry in history]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.0
        
        # More regular intervals = higher stability
        interval_variance = np.var(intervals)
        stability = 1.0 / (1.0 + interval_variance / 100.0)  # Normalize
        
        return min(stability, 1.0)
    
    def _calculate_depth_consistency(self, structure_type: str) -> float:
        """Calculate consistency of recursion depth."""
        history = self.structure_history[structure_type]
        if len(history) < 2:
            return 0.0
        
        depths = [entry['depth'] for entry in history]
        depth_variance = np.var(depths)
        consistency = 1.0 / (1.0 + depth_variance)
        
        return min(consistency, 1.0)


class MemoryImrintEncoder:
    """
    Converts stable collapses into field imprints.
    Includes phase encoding and time decay signatures.
    """
    
    def __init__(self):
        self.imprint_counter = 0
        self.phase_offset = 0.0
    
    def encode_imprint(self, structure_data: Dict[str, Any], stability_score: float,
                      context: ExecutionContext) -> MemoryImprint:
        """Convert collapse result into memory imprint."""
        self.imprint_counter += 1
        
        # Calculate 3D coordinates from 2D plus depth
        base_coords = structure_data.get('coordinates', (0.0, 0.0))
        z_coord = (context.depth or 0) * 2.0  # Scale depth to z-dimension
        coordinates = (base_coords[0], base_coords[1], z_coord)
        
        # Calculate phase coherence
        phase_coherence = self._calculate_phase_coherence(structure_data, context)
        
        # Calculate temporal decay rate
        temporal_decay = self._calculate_decay_rate(stability_score)
        
        # Calculate vortex strength
        vortex_strength = self._calculate_vortex_strength(structure_data)
        
        imprint = MemoryImprint(
            structure_id=f"imprint_{self.imprint_counter}_{structure_data.get('type', 'unknown')}",
            field_coordinates=coordinates,
            entropy_signature=structure_data.get('entropy_resolved', 0.0),
            stability_score=stability_score,
            phase_coherence=phase_coherence,
            recursion_depth=context.depth or 0,
            temporal_decay=temporal_decay,
            creation_time=time.time(),
            last_reinforcement=time.time(),
            reinforcement_count=1,
            vortex_strength=vortex_strength
        )
        
        return imprint
    
    def _calculate_phase_coherence(self, structure_data: Dict[str, Any], 
                                  context: ExecutionContext) -> float:
        """Calculate phase coherence for the imprint."""
        # Use entropy and recursion depth to calculate phase
        entropy = structure_data.get('entropy_resolved', 0.0)
        depth = context.depth or 0
        
        # Phase calculation with harmonic components
        phase = (entropy * math.pi + depth * 0.5 + self.phase_offset) % (2 * math.pi)
        coherence = abs(math.cos(phase))  # Coherence based on phase alignment
        
        return coherence
    
    def _calculate_decay_rate(self, stability_score: float) -> float:
        """Calculate temporal decay rate based on stability."""
        # More stable structures decay slower
        base_decay = 0.99
        stability_bonus = stability_score * 0.01
        return min(base_decay + stability_bonus, 0.999)
    
    def _calculate_vortex_strength(self, structure_data: Dict[str, Any]) -> float:
        """Calculate potential for vortex formation."""
        # Structures with high curvature or force tend to form vortices
        curvature = structure_data.get('curvature', 0.0)
        force = structure_data.get('force', 0.0)
        
        vortex_strength = abs(curvature) * 0.5 + abs(force) * 0.3
        return min(vortex_strength, 1.0)


class VortexTracker:
    """
    Detects coherent feedback loops and orbiting patterns.
    Supports structural invariance in symbolic evolution.
    """
    
    def __init__(self, field_tensor: MemoryFieldTensor):
        self.field_tensor = field_tensor
        self.active_vortices = {}
        self.vortex_counter = 0
        self.detection_threshold = 0.4
    
    def detect_vortices(self, imprints: List[MemoryImprint]) -> List[VortexPattern]:
        """Detect vortex patterns in memory imprints."""
        new_vortices = []
        
        # Group imprints by proximity
        clusters = self._cluster_imprints(imprints)
        
        for cluster in clusters:
            if len(cluster) >= 3:  # Need at least 3 points for vortex
                vortex = self._analyze_cluster_for_vortex(cluster)
                if vortex and vortex.stability_index > self.detection_threshold:
                    new_vortices.append(vortex)
        
        # Update active vortices
        self._update_vortex_registry(new_vortices)
        
        return new_vortices
    
    def track_temporal_symmetry(self, vortex: VortexPattern, 
                               time_window: float = 10.0) -> float:
        """Track temporal symmetry in vortex evolution."""
        # Simplified temporal symmetry calculation
        # In a real implementation, this would analyze the vortex's evolution over time
        
        current_time = time.time()
        age = current_time - vortex.creation_time
        
        # Symmetry decreases with age unless reinforced
        base_symmetry = math.exp(-age / time_window)
        
        # Modify based on angular velocity (more regular = more symmetric)
        velocity_factor = 1.0 - abs(vortex.angular_velocity - 1.0)  # Optimal at velocity = 1.0
        
        temporal_symmetry = base_symmetry * velocity_factor
        return max(temporal_symmetry, 0.0)
    
    def _cluster_imprints(self, imprints: List[MemoryImprint]) -> List[List[MemoryImprint]]:
        """Group imprints by spatial proximity."""
        clusters = []
        used = set()
        
        for i, imprint in enumerate(imprints):
            if i in used:
                continue
            
            cluster = [imprint]
            used.add(i)
            
            # Find nearby imprints
            for j, other in enumerate(imprints):
                if j in used or j == i:
                    continue
                
                distance = self._calculate_3d_distance(
                    imprint.field_coordinates, 
                    other.field_coordinates
                )
                
                if distance < 5.0:  # Proximity threshold
                    cluster.append(other)
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _analyze_cluster_for_vortex(self, cluster: List[MemoryImprint]) -> Optional[VortexPattern]:
        """Analyze cluster for vortex formation."""
        if len(cluster) < 3:
            return None
        
        # Calculate center of mass
        center_x = sum(imp.field_coordinates[0] for imp in cluster) / len(cluster)
        center_y = sum(imp.field_coordinates[1] for imp in cluster) / len(cluster)
        center_z = sum(imp.field_coordinates[2] for imp in cluster) / len(cluster)
        center = (center_x, center_y, center_z)
        
        # Calculate average distance (radius)
        distances = [self._calculate_3d_distance(center, imp.field_coordinates) for imp in cluster]
        radius = sum(distances) / len(distances)
        
        # Estimate angular velocity based on vortex strengths
        total_vortex_strength = sum(imp.vortex_strength for imp in cluster)
        angular_velocity = total_vortex_strength / len(cluster)
        
        # Calculate stability index
        stability_index = self._calculate_vortex_stability(cluster, center, radius)
        
        # Calculate temporal symmetry
        temporal_symmetry = 1.0  # Initialize high, will be updated over time
        
        self.vortex_counter += 1
        vortex = VortexPattern(
            pattern_id=f"vortex_{self.vortex_counter}",
            center_coordinates=center,
            radius=radius,
            angular_velocity=angular_velocity,
            stability_index=stability_index,
            participating_imprints=[imp.structure_id for imp in cluster],
            temporal_symmetry=temporal_symmetry,
            creation_time=time.time()
        )
        
        return vortex
    
    def _calculate_vortex_stability(self, cluster: List[MemoryImprint], 
                                   center: Tuple[float, float, float], 
                                   radius: float) -> float:
        """Calculate stability index for vortex pattern."""
        # Factors contributing to stability:
        # 1. Uniformity of distances from center
        # 2. Consistency of vortex strengths
        # 3. Phase coherence alignment
        
        distances = [self._calculate_3d_distance(center, imp.field_coordinates) for imp in cluster]
        distance_variance = np.var(distances) / max(radius * radius, 1e-10)
        distance_stability = 1.0 / (1.0 + distance_variance)
        
        vortex_strengths = [imp.vortex_strength for imp in cluster]
        strength_variance = np.var(vortex_strengths)
        strength_stability = 1.0 / (1.0 + strength_variance)
        
        phase_coherences = [imp.phase_coherence for imp in cluster]
        phase_consistency = sum(phase_coherences) / len(phase_coherences)
        
        # Combine stability factors
        stability = (0.4 * distance_stability + 
                    0.3 * strength_stability + 
                    0.3 * phase_consistency)
        
        return min(stability, 1.0)
    
    def _calculate_3d_distance(self, pos1: Tuple[float, float, float], 
                              pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance in 3D."""
        return math.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
    
    def _update_vortex_registry(self, new_vortices: List[VortexPattern]):
        """Update registry of active vortices."""
        current_time = time.time()
        
        # Remove old vortices
        expired_vortices = []
        for vortex_id, vortex in self.active_vortices.items():
            age = current_time - vortex.creation_time
            if age > 60.0 or vortex.stability_index < 0.2:  # 1 minute timeout or low stability
                expired_vortices.append(vortex_id)
        
        for vortex_id in expired_vortices:
            del self.active_vortices[vortex_id]
        
        # Add new vortices
        for vortex in new_vortices:
            self.active_vortices[vortex.pattern_id] = vortex


class SuperfluidMemory:
    """
    PAC-Native Superfluid Memory for GAIA v3.0
    Built on Fracton SDK for physics-governed superfluid memory with automatic conservation.
    All memory operations maintain f(parent) = Σf(children) through native PAC regulation.
    """
    
    def __init__(self, field_shape: Tuple[int, int, int] = (32, 32, 16)):
        # Use Fracton's physics memory as foundation
        self.physics_memory = PhysicsMemoryField(
            field_shape=field_shape[:2],  # Use 2D for core field
            xi_target=1.0571,
            enable_conservation=True,
            superfluid_mode=True
        )
        
        # Enable PAC self-regulation
        self.pac_regulator = enable_pac_self_regulation()
        
        # Physics recursive executor for memory operations
        self.physics_executor = PhysicsRecursiveExecutor(
            max_depth=5,
            pac_regulation=True
        )
        
        # Enhanced components
        self.field_shape = field_shape
        self.active_imprints = {}
        self.imprint_counter = 0
        
        # Statistics
        self.total_imprints_created = 0
        self.memory_updates = 0
        self.conservation_violations_resolved = 0
    
    @pac_recursive("superfluid_memory_add")
    def add_memory(self, structure_data: Dict[str, Any], 
                   context=None) -> Optional[MemoryImprint]:
        """Add memory structure with PAC-native superfluid dynamics."""
        self.memory_updates += 1
        
        # Get initial state for conservation tracking
        initial_metrics = self.physics_memory.get_physics_metrics()
        
        # Encode structure into physics field
        structure_field = self._encode_structure_to_field(structure_data)
        
        # Add to memory using Klein-Gordon superfluid evolution
        current_field = self.physics_memory.get_field_state()
        evolved_field = klein_gordon_evolution(
            field=current_field,
            dt=0.01,
            mass_squared=0.01,  # Low mass for superfluid behavior
            source_term=structure_field
        )
        
        # Update physics memory
        self.physics_memory.update_field_state(evolved_field)
        
        # Check and resolve conservation violations
        final_metrics = self.physics_memory.get_physics_metrics()
        conservation_residual = abs(final_metrics.get('conservation_residual', 0.0))
        
        if conservation_residual > 1e-6:
            corrected_field = enforce_pac_conservation(
                field=evolved_field,
                target_xi=1.0571
            )
            self.physics_memory.update_field_state(corrected_field)
            self.conservation_violations_resolved += 1
        
        # Create memory imprint
        self.imprint_counter += 1
        imprint = MemoryImprint(
            structure_id=f"pac_imprint_{self.imprint_counter}",
            field_coordinates=self._find_field_centroid(structure_field),
            entropy_signature=final_metrics.get('entropy_measure', 0.5),
            stability_score=1.0 - conservation_residual,
            phase_coherence=final_metrics.get('field_coherence', 0.5),
            recursion_depth=context.get('depth', 0) if context else 0,
            temporal_decay=0.01,
            creation_time=time.time(),
            last_reinforcement=time.time(),
            reinforcement_count=1,
            vortex_strength=self._calculate_vortex_strength(structure_field)
        )
        
        self.active_imprints[imprint.structure_id] = imprint
        self.total_imprints_created += 1
        
        return imprint
    
    @pac_recursive("superfluid_memory_retrieve") 
    def retrieve_memory(self, query_pattern: Any) -> List[MemoryImprint]:
        """Retrieve memories using PAC-native resonance matching."""
        # Encode query into physics field
        query_field = self._encode_structure_to_field(query_pattern)
        
        # Use resonance field interaction for memory retrieval
        memory_field = self.physics_memory.get_field_state()
        resonance_result = resonance_field_interaction(
            field1=memory_field,
            field2=query_field,
            coupling_strength=1.571  # PAC coupling
        )
        
        # Find resonant imprints based on field overlap
        resonant_imprints = []
        for imprint in self.active_imprints.values():
            # Calculate resonance strength
            imprint_coords = imprint.field_coordinates
            if len(imprint_coords) >= 2:
                x, y = int(imprint_coords[0]), int(imprint_coords[1])
                if 0 <= x < resonance_result.shape[0] and 0 <= y < resonance_result.shape[1]:
                    resonance_strength = abs(resonance_result[x, y])
                    if resonance_strength > 0.1:  # Resonance threshold
                        resonant_imprints.append(imprint)
        
        return resonant_imprints
    
    def _encode_structure_to_field(self, structure_data: Any) -> np.ndarray:
        """Encode structure data into physics field representation."""
        if isinstance(structure_data, str):
            # Hash-based encoding for strings
            import hashlib
            hash_bytes = hashlib.md5(structure_data.encode()).digest()
            field_size = int(np.prod(self.field_shape[:2]))
            hash_values = np.frombuffer(hash_bytes, dtype=np.uint8)
            
            if len(hash_values) >= field_size:
                field = hash_values[:field_size].astype(np.float32)
            else:
                field = np.zeros(field_size, dtype=np.float32)
                field[:len(hash_values)] = hash_values.astype(np.float32)
            
            field = field.reshape(self.field_shape[:2])
            return (field - np.mean(field)) / (np.std(field) + 1e-8)
        
        elif isinstance(structure_data, dict):
            # Encode dictionary as structured field
            field = np.random.normal(0, 0.1, self.field_shape[:2])
            # Add structure based on dict keys/values
            for i, (k, v) in enumerate(structure_data.items()):
                if i < field.size:
                    field.flat[i] += hash(str(k)) % 100 * 0.01
            return field
        
        else:
            # Default: small random field
            return np.random.normal(0, 0.05, self.field_shape[:2])
    
    def _find_field_centroid(self, field: np.ndarray) -> Tuple[float, float, float]:
        """Find the centroid (center of mass) of a field."""
        if field.size == 0:
            return (0.0, 0.0, 0.0)
        
        # Calculate weighted centroid
        intensity = np.abs(field) ** 2
        total_intensity = np.sum(intensity)
        
        if total_intensity < 1e-10:
            return (field.shape[0]/2, field.shape[1]/2, 0.0)
        
        y_indices, x_indices = np.meshgrid(range(field.shape[1]), range(field.shape[0]))
        
        centroid_x = np.sum(x_indices * intensity) / total_intensity
        centroid_y = np.sum(y_indices * intensity) / total_intensity
        
        return (float(centroid_x), float(centroid_y), 0.0)
    
    def _calculate_vortex_strength(self, field: np.ndarray) -> float:
        """Calculate vortex strength from field phase structure."""
        if field.dtype == np.complex128 or field.dtype == np.complex64:
            phase = np.angle(field)
        else:
            # Convert real field to complex for phase calculation
            phase = np.arctan2(np.gradient(field, axis=1), np.gradient(field, axis=0))
        
        # Calculate phase circulation
        phase_grad_x = np.gradient(phase, axis=0)
        phase_grad_y = np.gradient(phase, axis=1)
        circulation = np.sum(np.abs(phase_grad_x) + np.abs(phase_grad_y))
        
        return circulation / field.size
        """Add new structure to PAC superfluid memory with amplitude conservation."""
        self.memory_updates += 1
        
        # Convert structure to complex amplitude pattern
        amplitude_pattern = self._structure_to_amplitude_pattern(structure_data, context)
        
        # Evaluate stability through amplitude coherence (replaces arbitrary threshold)
        # Use pattern coherence measure: max amplitude * coherence factor
        max_amplitude = np.max(np.abs(amplitude_pattern))
        mean_amplitude = np.mean(np.abs(amplitude_pattern))
        pattern_coherence = max_amplitude * mean_amplitude  # Coherence = peak × average
        stability_score = pattern_coherence
        
        # Debug output for testing
        print(f"PAC Memory: max_amplitude={max_amplitude:.4f}, mean_amplitude={mean_amplitude:.4f}, stability_score={stability_score:.4f}")
        
        # PAC conservation: must rebalance field when adding new pattern
        if not hasattr(self, 'memory_amplitude_field'):
            # Initialize complex memory field on first use
            shape = getattr(self.memory_field_tensor, 'shape', (32, 32, 16))[:2]  # Use 2D for simplicity
            self.memory_amplitude_field = np.zeros(shape, dtype=complex)
        
        # Add pattern while conserving total amplitude
        self.memory_amplitude_field = self._rebalance_amplitude_field(
            self.memory_amplitude_field, 
            amplitude_pattern
        )
        
        # Detect vortices from phase singularities (real physics)
        phase_field = np.angle(self.memory_amplitude_field)
        vortices = self._detect_phase_singularities(phase_field)
        self.total_vortices_detected = len(vortices)
        
        # Only store if amplitude pattern creates coherent structure (lowered threshold)
        coherence_threshold = 0.01  # Much lower threshold for PAC patterns
        if stability_score < coherence_threshold:
            print(f"PAC Memory: Pattern rejected - stability_score {stability_score:.4f} < threshold {coherence_threshold}")
            return None
        
        # Create memory imprint from amplitude pattern
        imprint = self.imprint_encoder.encode_imprint(structure_data, stability_score, context)
        
        # Store amplitude pattern in imprint for PAC physics
        imprint.amplitude_pattern = amplitude_pattern
        imprint.phase_singularities = vortices
        
        # Add to active imprints
        self.active_imprints[imprint.structure_id] = imprint
        self.total_imprints_created += 1
        
        # Update legacy field tensor for compatibility (derived from amplitude)
        self.memory_field_tensor.add_imprint(imprint)
        
        return imprint
    
    def reinforce_memory(self, structure_id: str, reinforcement_strength: float = 1.0):
        """Reinforce existing memory imprint."""
        if structure_id in self.active_imprints:
            imprint = self.active_imprints[structure_id]
            imprint.last_reinforcement = time.time()
            imprint.reinforcement_count += 1
            
            # Strengthen in field tensor
            self.memory_field_tensor.add_imprint(imprint, reinforcement_strength)
    
    def get_memory_coherence(self, coordinates: Tuple[float, float, float]) -> float:
        """Get memory field coherence at given coordinates."""
        return self.memory_field_tensor.get_field_strength(coordinates)
    
    def get_memory_attractors(self, threshold: float = 0.5) -> List[Tuple[int, int, int]]:
        """Get current memory attractors (high-coherence regions)."""
        return self.memory_field_tensor.find_attractors(threshold)
    
    def update_memory_field(self):
        """Update memory field with temporal decay and maintenance."""
        # Apply temporal decay
        self.memory_field_tensor.apply_temporal_decay()
        
        # Remove weak imprints
        current_time = time.time()
        expired_imprints = []
        
        for imprint_id, imprint in self.active_imprints.items():
            age = current_time - imprint.last_reinforcement
            decay_factor = imprint.temporal_decay ** age
            
            if decay_factor < 0.1:  # Very weak
                expired_imprints.append(imprint_id)
        
        for imprint_id in expired_imprints:
            del self.active_imprints[imprint_id]
        
        # Update vortex tracking
        active_imprints_list = list(self.active_imprints.values())
        self.vortex_tracker.detect_vortices(active_imprints_list)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        active_vortices = len(self.vortex_tracker.active_vortices)
        attractors = len(self.get_memory_attractors())
        
        avg_stability = 0.0
        if self.active_imprints:
            avg_stability = sum(imp.stability_score for imp in self.active_imprints.values()) / len(self.active_imprints)
        
        return {
            'active_imprints': len(self.active_imprints),
            'total_imprints_created': self.total_imprints_created,
            'active_vortices': active_vortices,
            'total_vortices_detected': self.total_vortices_detected,
            'memory_attractors': attractors,
            'average_stability': avg_stability,
            'memory_updates': self.memory_updates,
            'field_tensor_shape': self.memory_field_tensor.shape
        }
    
    def _check_vortex_formation(self):
        """Check for new vortex formation after adding imprint."""
        active_imprints_list = list(self.active_imprints.values())
        new_vortices = self.vortex_tracker.detect_vortices(active_imprints_list)
        
        if new_vortices:
            self.total_vortices_detected += len(new_vortices)
    
    def reset(self):
        """Reset superfluid memory to initial state."""
        self.memory_field_tensor = MemoryFieldTensor(self.memory_field_tensor.shape)
        self.stability_evaluator = StabilityEvaluator()
        self.imprint_encoder = MemoryImrintEncoder()
        self.vortex_tracker = VortexTracker(self.memory_field_tensor)
        
        self.active_imprints.clear()
        self.total_imprints_created = 0
        self.total_vortices_detected = 0
        self.memory_updates = 0
        
        # Reset PAC fields
        if hasattr(self, 'memory_amplitude_field'):
            del self.memory_amplitude_field

    def _structure_to_amplitude_pattern(self, structure_data: Dict[str, Any], context: ExecutionContext) -> np.ndarray:
        """Convert symbolic structure to complex amplitude pattern."""
        # Get field dimensions
        shape = getattr(self.memory_field_tensor, 'shape', (32, 32, 16))[:2]
        
        # Create amplitude pattern from structure data
        amplitude_pattern = np.zeros(shape, dtype=complex)
        
        # Use structure properties to create amplitude/phase pattern
        structure_id = structure_data.get('structure_id', '')
        entropy_resolved = structure_data.get('entropy_resolved', 0.5)
        
        # Create spatial pattern from structure ID hash
        import hashlib
        id_hash = int(hashlib.md5(structure_id.encode()).hexdigest()[:8], 16)
        
        # Map hash to field locations
        num_points = min(10, max(1, int(entropy_resolved * 20)))  # More entropy = more points
        np.random.seed(id_hash % (2**31))  # Reproducible random pattern
        
        for i in range(num_points):
            x = np.random.randint(0, shape[0])
            y = np.random.randint(0, shape[1])
            
            # Amplitude magnitude from entropy (boosted for better coherence)
            magnitude = entropy_resolved * 0.5  # Increased from 0.1 to 0.5
            
            # Phase from structure properties  
            phase = 2 * np.pi * (i / num_points)
            if 'collapse_type' in structure_data:
                phase += hash(str(structure_data['collapse_type'])) * 0.001
            
            # Add Gaussian amplitude blob
            sigma = max(1, shape[0] // 8)
            xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
            distance = (xx - x)**2 + (yy - y)**2
            gaussian = np.exp(-distance / (2 * sigma**2))
            
            amplitude_pattern += magnitude * gaussian * np.exp(1j * phase)
        
        return amplitude_pattern
    
    def _rebalance_amplitude_field(self, existing_field: np.ndarray, new_pattern: np.ndarray) -> np.ndarray:
        """Rebalance memory field while conserving total amplitude."""
        # Resize new pattern to match existing field if needed
        if new_pattern.shape != existing_field.shape:
            # Simple resize by taking the minimum dimensions
            min_x = min(new_pattern.shape[0], existing_field.shape[0])
            min_y = min(new_pattern.shape[1], existing_field.shape[1])
            resized_pattern = np.zeros_like(existing_field)
            resized_pattern[:min_x, :min_y] = new_pattern[:min_x, :min_y]
            new_pattern = resized_pattern
        
        # Add patterns
        combined = existing_field + new_pattern
        
        # Calculate total probability
        total_prob = np.sum(np.abs(combined) ** 2)
        target_prob = existing_field.size  # Normalize to field size
        
        # Renormalize to conserve total probability
        if total_prob > 1e-10:
            conservation_factor = np.sqrt(target_prob / total_prob)
            combined *= conservation_factor
        
        return combined
    
    def _detect_phase_singularities(self, phase_field: np.ndarray) -> List[Tuple[int, int]]:
        """Detect phase singularities (vortices) in phase field - real physics."""
        singularities = []
        
        try:
            # Look for points where phase circulation is ±2π
            for i in range(1, phase_field.shape[0] - 1):
                for j in range(1, phase_field.shape[1] - 1):
                    # Get phases around this point in clockwise order
                    phases = [
                        phase_field[i-1, j],     # top
                        phase_field[i-1, j+1],   # top-right  
                        phase_field[i, j+1],     # right
                        phase_field[i+1, j+1],   # bottom-right
                        phase_field[i+1, j],     # bottom
                        phase_field[i+1, j-1],   # bottom-left
                        phase_field[i, j-1],     # left
                        phase_field[i-1, j-1]    # top-left
                    ]
                    
                    # Calculate total phase circulation
                    circulation = 0
                    for k in range(len(phases)):
                        diff = phases[k] - phases[k-1]
                        # Unwrap phase difference
                        if diff > np.pi:
                            diff -= 2 * np.pi
                        elif diff < -np.pi:
                            diff += 2 * np.pi
                        circulation += diff
                    
                    # Singularity if circulation ≈ ±2π
                    if abs(abs(circulation) - 2 * np.pi) < 0.5:
                        singularities.append((i, j))
                        
        except Exception as e:
            print(f"PAC vortex detection error: {e}")
        
        return singularities
