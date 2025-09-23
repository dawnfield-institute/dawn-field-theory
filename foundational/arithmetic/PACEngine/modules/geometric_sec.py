"""
Symbolic Entropy Collapse (SEC) Module

Implements geometric collapse phenomena through PAC conservation,
where symbolic structures undergo entropy collapse leading to
emergent geometric configurations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

class CollapseType(Enum):
    """Types of geometric collapse"""
    ENTROPY_DRIVEN = "entropy_driven"
    CURVATURE_INDUCED = "curvature_induced"
    TOPOLOGICAL = "topological"
    SYMBOLIC_RESONANCE = "symbolic_resonance"
    CRITICAL_TRANSITION = "critical_transition"

class GeometricPhase(Enum):
    """Phases of geometric evolution"""
    DISPERSED = "dispersed"
    CLUSTERING = "clustering"
    COLLAPSING = "collapsing"
    CRYSTALLIZED = "crystallized"
    EMERGENT = "emergent"

@dataclass
class SECResult:
    """Result of SEC geometric evolution"""
    geometric_field: torch.Tensor
    entropy_profile: torch.Tensor
    collapse_locations: List[Tuple[int, ...]]
    geometric_phase: GeometricPhase
    collapse_strength: float
    symbolic_resonance: float
    emergent_structures: List[Dict[str, Any]]

class GeometricSECModule:
    """
    Symbolic Entropy Collapse through PAC conservation.
    
    Implements geometric phenomena where high-dimensional symbolic
    spaces collapse into lower-dimensional geometric structures
    while maintaining PAC conservation.
    """
    
    def __init__(self, 
                 collapse_threshold: float = 0.1,
                 symbolic_dimension: int = 256,
                 device: str = "auto"):
        self.collapse_threshold = collapse_threshold
        self.symbolic_dimension = symbolic_dimension
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # SEC parameters
        self.entropy_decay_rate = 0.05
        self.curvature_sensitivity = 1.0
        self.resonance_frequency = 1.0571  # Universal balance frequency
        self.geometric_scale_factor = 0.1
        
        # Collapse detection
        self.min_collapse_size = 3
        self.max_collapse_size = 50
        
    def evolve_geometric_sec(self, 
                           symbolic_field: torch.Tensor,
                           geometric_constraints: Optional[Dict] = None,
                           dt: float = 0.01) -> SECResult:
        """
        Evolve symbolic field through SEC dynamics.
        
        Args:
            symbolic_field: High-dimensional symbolic representation
            geometric_constraints: Geometric boundary conditions
            dt: Time step for evolution
            
        Returns:
            SECResult with geometric evolution details
        """
        symbolic_field = symbolic_field.to(self.device)
        
        # Calculate current entropy profile
        entropy_profile = self._calculate_entropy_profile(symbolic_field)
        
        # Detect potential collapse regions
        collapse_candidates = self._identify_collapse_regions(symbolic_field, entropy_profile)
        
        # Apply SEC dynamics
        evolved_field = self._apply_sec_dynamics(symbolic_field, entropy_profile, dt)
        
        # Apply geometric constraints
        if geometric_constraints:
            evolved_field = self._apply_geometric_constraints(evolved_field, geometric_constraints)
        
        # Detect actual collapses
        collapse_locations = self._detect_active_collapses(evolved_field, entropy_profile)
        
        # Analyze geometric phase
        geometric_phase = self._determine_geometric_phase(evolved_field, entropy_profile)
        
        # Calculate collapse strength
        collapse_strength = self._calculate_collapse_strength(collapse_locations, entropy_profile)
        
        # Detect symbolic resonance
        symbolic_resonance = self._detect_symbolic_resonance(evolved_field)
        
        # Identify emergent structures
        emergent_structures = self._identify_emergent_structures(evolved_field, collapse_locations)
        
        return SECResult(
            geometric_field=evolved_field,
            entropy_profile=entropy_profile,
            collapse_locations=collapse_locations,
            geometric_phase=geometric_phase,
            collapse_strength=collapse_strength,
            symbolic_resonance=symbolic_resonance,
            emergent_structures=emergent_structures
        )
    
    def create_symbolic_field(self, 
                            field_type: str = "random",
                            dimensions: Tuple[int, ...] = (32, 32, 32),
                            symbolic_complexity: float = 1.0) -> torch.Tensor:
        """
        Create initial symbolic field configuration.
        
        Args:
            field_type: Type of initial field ("random", "structured", "symbolic")
            dimensions: Field dimensions
            symbolic_complexity: Complexity of symbolic structures
            
        Returns:
            Symbolic field tensor
        """
        if field_type == "random":
            field = torch.randn(dimensions, device=self.device)
        elif field_type == "structured":
            field = self._create_structured_symbolic_field(dimensions, symbolic_complexity)
        elif field_type == "symbolic":
            field = self._create_symbolic_resonance_field(dimensions, symbolic_complexity)
        else:
            field = torch.zeros(dimensions, device=self.device)
        
        # Normalize to maintain PAC conservation
        field = self._normalize_symbolic_field(field)
        
        return field
    
    def simulate_entropy_collapse(self, 
                                initial_field: torch.Tensor,
                                collapse_trigger: str = "entropy_threshold",
                                evolution_steps: int = 1000) -> Dict[str, Any]:
        """
        Simulate complete entropy collapse process.
        
        Args:
            initial_field: Initial symbolic field
            collapse_trigger: Mechanism triggering collapse
            evolution_steps: Number of evolution steps
            
        Returns:
            Dictionary with collapse simulation results
        """
        initial_field = initial_field.to(self.device)
        
        # Evolution tracking
        field_history = []
        entropy_history = []
        collapse_events = []
        phase_transitions = []
        
        current_field = initial_field.clone()
        dt = 0.01
        
        for step in range(evolution_steps):
            # Evolve field
            result = self.evolve_geometric_sec(current_field, dt=dt)
            current_field = result.geometric_field
            
            # Record state
            field_history.append(current_field.clone().cpu())
            entropy_history.append(result.entropy_profile.clone().cpu())
            
            # Record collapse events
            if result.collapse_locations:
                collapse_events.append({
                    "step": step,
                    "locations": result.collapse_locations,
                    "strength": result.collapse_strength,
                    "phase": result.geometric_phase.value
                })
            
            # Check for phase transitions
            if step > 0:
                prev_phase = phase_transitions[-1]["phase"] if phase_transitions else "dispersed"
                if result.geometric_phase.value != prev_phase:
                    phase_transitions.append({
                        "step": step,
                        "from_phase": prev_phase,
                        "to_phase": result.geometric_phase.value,
                        "transition_strength": result.collapse_strength
                    })
            
            # Early termination if fully crystallized
            if result.geometric_phase == GeometricPhase.CRYSTALLIZED:
                break
        
        return {
            "final_field": current_field,
            "field_evolution": field_history,
            "entropy_evolution": entropy_history,
            "collapse_events": collapse_events,
            "phase_transitions": phase_transitions,
            "total_collapses": len(collapse_events),
            "final_phase": result.geometric_phase.value,
            "final_entropy": torch.mean(result.entropy_profile).item()
        }
    
    def _calculate_entropy_profile(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate local entropy profile across the field"""
        # Local entropy calculation using sliding window
        if len(field.shape) == 3:
            h, w, d = field.shape
            entropy_profile = torch.zeros_like(field)
            window_size = 3
            
            for i in range(window_size//2, h - window_size//2):
                for j in range(window_size//2, w - window_size//2):
                    for k in range(window_size//2, d - window_size//2):
                        # Extract local region
                        local_region = field[i-window_size//2:i+window_size//2+1,
                                           j-window_size//2:j+window_size//2+1,
                                           k-window_size//2:k+window_size//2+1]
                        
                        # Calculate local entropy
                        local_entropy = self._calculate_local_entropy(local_region)
                        entropy_profile[i, j, k] = local_entropy
        else:
            # Fallback for other dimensions
            field_flat = field.flatten()
            hist = torch.histc(field_flat, bins=50)
            hist = hist / torch.sum(hist)
            hist = hist[hist > 0]
            entropy = -torch.sum(hist * torch.log(hist))
            entropy_profile = torch.full_like(field, entropy)
        
        return entropy_profile
    
    def _calculate_local_entropy(self, local_region: torch.Tensor) -> float:
        """Calculate entropy of a local field region"""
        region_flat = local_region.flatten()
        
        # Create histogram
        hist = torch.histc(region_flat, bins=min(20, len(region_flat)))
        hist = hist / torch.sum(hist)
        
        # Remove zero bins
        hist = hist[hist > 0]
        
        if len(hist) > 1:
            entropy = -torch.sum(hist * torch.log(hist))
            return entropy.item()
        else:
            return 0.0
    
    def _identify_collapse_regions(self, 
                                 field: torch.Tensor,
                                 entropy_profile: torch.Tensor) -> List[Tuple[int, ...]]:
        """Identify regions with potential for entropy collapse"""
        collapse_candidates = []
        
        # Find low entropy regions
        low_entropy_mask = entropy_profile < self.collapse_threshold
        
        if torch.any(low_entropy_mask):
            # Find connected components of low entropy
            if len(field.shape) == 3:
                # 3D connected components (simplified)
                labeled_regions = self._label_connected_components_3d(low_entropy_mask)
                
                for region_label in torch.unique(labeled_regions):
                    if region_label == 0:  # Skip background
                        continue
                    
                    # Find region center
                    region_mask = labeled_regions == region_label
                    region_indices = torch.where(region_mask)
                    
                    if len(region_indices[0]) >= self.min_collapse_size:
                        center = tuple(int(torch.mean(idx.float()).item()) for idx in region_indices)
                        collapse_candidates.append(center)
        
        return collapse_candidates
    
    def _apply_sec_dynamics(self, 
                          field: torch.Tensor,
                          entropy_profile: torch.Tensor,
                          dt: float) -> torch.Tensor:
        """Apply SEC evolution dynamics"""
        evolved_field = field.clone()
        
        # Calculate gradients
        if len(field.shape) == 3:
            # 3D gradient calculation
            grad_x = torch.gradient(field, dim=0)[0]
            grad_y = torch.gradient(field, dim=1)[0]
            grad_z = torch.gradient(field, dim=2)[0]
            
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        else:
            # Fallback gradient
            gradient_magnitude = torch.abs(torch.gradient(field.flatten())[0].reshape(field.shape))
        
        # Calculate curvature (simplified)
        curvature = self._calculate_field_curvature(field)
        
        # SEC evolution equation:
        # ∂φ/∂t = -α∇²φ - β(entropy)φ + γ(curvature)φ + δ(resonance)
        
        # Diffusion term (smoothing)
        laplacian = self._calculate_laplacian(field)
        diffusion_term = -0.01 * laplacian
        
        # Entropy-driven collapse term
        entropy_term = -self.entropy_decay_rate * entropy_profile * field
        
        # Curvature amplification term
        curvature_term = self.curvature_sensitivity * curvature * field
        
        # Symbolic resonance term
        resonance_term = self._calculate_resonance_field(field)
        
        # Combine terms
        field_derivative = (diffusion_term + entropy_term + 
                          curvature_term + resonance_term)
        
        # Update field
        evolved_field += dt * field_derivative
        
        # Apply PAC conservation
        evolved_field = self._enforce_pac_conservation(evolved_field, field)
        
        return evolved_field
    
    def _calculate_field_curvature(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate field curvature tensor"""
        if len(field.shape) == 3:
            # Calculate second derivatives
            d2_dx2 = torch.gradient(torch.gradient(field, dim=0)[0], dim=0)[0]
            d2_dy2 = torch.gradient(torch.gradient(field, dim=1)[0], dim=1)[0]
            d2_dz2 = torch.gradient(torch.gradient(field, dim=2)[0], dim=2)[0]
            
            # Simplified scalar curvature
            curvature = d2_dx2 + d2_dy2 + d2_dz2
        else:
            # Fallback for other dimensions
            curvature = torch.zeros_like(field)
        
        return curvature
    
    def _calculate_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate Laplacian of field"""
        return self._calculate_field_curvature(field)  # Same as scalar curvature
    
    def _calculate_resonance_field(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate symbolic resonance field"""
        # Create resonance pattern based on universal frequency
        if len(field.shape) == 3:
            h, w, d = field.shape
            x = torch.arange(h, device=self.device).float()
            y = torch.arange(w, device=self.device).float()
            z = torch.arange(d, device=self.device).float()
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            
            # Resonance pattern
            resonance = (0.1 * torch.sin(self.resonance_frequency * X / h * 2 * math.pi) *
                        torch.cos(self.resonance_frequency * Y / w * 2 * math.pi) *
                        torch.sin(self.resonance_frequency * Z / d * 2 * math.pi))
        else:
            resonance = 0.1 * torch.sin(self.resonance_frequency * torch.arange(field.numel(), device=self.device).float()).reshape(field.shape)
        
        return resonance
    
    def _enforce_pac_conservation(self, new_field: torch.Tensor, old_field: torch.Tensor) -> torch.Tensor:
        """Enforce PAC conservation during field evolution"""
        # Simple energy conservation
        old_energy = torch.sum(old_field**2)
        new_energy = torch.sum(new_field**2)
        
        if new_energy > 0:
            conservation_factor = torch.sqrt(old_energy / new_energy)
            return new_field * conservation_factor
        else:
            return new_field
    
    def _detect_active_collapses(self, 
                               field: torch.Tensor,
                               entropy_profile: torch.Tensor) -> List[Tuple[int, ...]]:
        """Detect actively collapsing regions"""
        active_collapses = []
        
        # Find regions with very low entropy and high field gradient
        low_entropy_mask = entropy_profile < self.collapse_threshold * 0.5
        
        if len(field.shape) == 3:
            gradient_magnitude = self._calculate_gradient_magnitude(field)
            high_gradient_mask = gradient_magnitude > torch.mean(gradient_magnitude) + torch.std(gradient_magnitude)
            
            # Regions with both low entropy and high gradient
            collapse_mask = low_entropy_mask & high_gradient_mask
            
            if torch.any(collapse_mask):
                # Find connected components
                labeled_regions = self._label_connected_components_3d(collapse_mask)
                
                for region_label in torch.unique(labeled_regions):
                    if region_label == 0:
                        continue
                    
                    region_mask = labeled_regions == region_label
                    region_indices = torch.where(region_mask)
                    
                    if len(region_indices[0]) >= self.min_collapse_size:
                        center = tuple(int(torch.mean(idx.float()).item()) for idx in region_indices)
                        active_collapses.append(center)
        
        return active_collapses
    
    def _calculate_gradient_magnitude(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate magnitude of field gradient"""
        if len(field.shape) == 3:
            grad_x = torch.gradient(field, dim=0)[0]
            grad_y = torch.gradient(field, dim=1)[0]
            grad_z = torch.gradient(field, dim=2)[0]
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        else:
            grad = torch.gradient(field.flatten())[0]
            magnitude = torch.abs(grad).reshape(field.shape)
        
        return magnitude
    
    def _determine_geometric_phase(self, 
                                 field: torch.Tensor,
                                 entropy_profile: torch.Tensor) -> GeometricPhase:
        """Determine current geometric phase of the system"""
        mean_entropy = torch.mean(entropy_profile)
        entropy_std = torch.std(entropy_profile)
        gradient_magnitude = torch.mean(self._calculate_gradient_magnitude(field))
        
        # Phase classification based on entropy and gradient characteristics
        if mean_entropy > 2.0 and entropy_std < 0.5:
            return GeometricPhase.DISPERSED
        elif mean_entropy > 1.0 and entropy_std > 0.5:
            return GeometricPhase.CLUSTERING
        elif mean_entropy < 1.0 and gradient_magnitude > 0.5:
            return GeometricPhase.COLLAPSING
        elif mean_entropy < 0.5 and gradient_magnitude < 0.1:
            return GeometricPhase.CRYSTALLIZED
        else:
            return GeometricPhase.EMERGENT
    
    def _calculate_collapse_strength(self, 
                                   collapse_locations: List[Tuple[int, ...]],
                                   entropy_profile: torch.Tensor) -> float:
        """Calculate overall collapse strength"""
        if not collapse_locations:
            return 0.0
        
        total_strength = 0.0
        for location in collapse_locations:
            if all(0 <= loc < entropy_profile.shape[i] for i, loc in enumerate(location)):
                local_entropy = entropy_profile[location]
                strength = max(0.0, self.collapse_threshold - local_entropy.item())
                total_strength += strength
        
        return total_strength / len(collapse_locations)
    
    def _detect_symbolic_resonance(self, field: torch.Tensor) -> float:
        """Detect symbolic resonance strength"""
        # Calculate resonance with universal frequency
        if len(field.shape) == 3:
            h, w, d = field.shape
            # Create reference resonance pattern
            x = torch.arange(h, device=self.device).float()
            y = torch.arange(w, device=self.device).float()
            z = torch.arange(d, device=self.device).float()
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            reference_pattern = torch.sin(self.resonance_frequency * X / h * 2 * math.pi)
            
            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([field.flatten(), reference_pattern.flatten()]))[0, 1]
            resonance_strength = abs(correlation.item()) if not torch.isnan(correlation) else 0.0
        else:
            # Fallback
            resonance_strength = 0.0
        
        return resonance_strength
    
    def _identify_emergent_structures(self, 
                                    field: torch.Tensor,
                                    collapse_locations: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        """Identify emergent geometric structures"""
        structures = []
        
        for i, location in enumerate(collapse_locations):
            # Analyze local structure around collapse point
            if len(field.shape) == 3 and all(0 <= loc < field.shape[j] for j, loc in enumerate(location)):
                # Extract local region
                radius = min(5, min(field.shape) // 4)
                x, y, z = location
                
                x_start = max(0, x - radius)
                x_end = min(field.shape[0], x + radius + 1)
                y_start = max(0, y - radius)
                y_end = min(field.shape[1], y + radius + 1)
                z_start = max(0, z - radius)
                z_end = min(field.shape[2], z + radius + 1)
                
                local_region = field[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Analyze structure properties
                structure_type = self._classify_structure_type(local_region)
                symmetry = self._calculate_local_symmetry(local_region)
                stability = self._assess_structure_stability(local_region)
                
                structure = {
                    "id": f"structure_{i}",
                    "location": location,
                    "type": structure_type,
                    "symmetry": symmetry,
                    "stability": stability,
                    "size": local_region.shape,
                    "intensity": torch.mean(torch.abs(local_region)).item()
                }
                
                structures.append(structure)
        
        return structures
    
    def _classify_structure_type(self, local_region: torch.Tensor) -> str:
        """Classify type of emergent structure"""
        # Simple structure classification based on field properties
        field_std = torch.std(local_region)
        field_skewness = self._calculate_skewness(local_region)
        
        if field_std < 0.1:
            return "uniform"
        elif abs(field_skewness) > 1.0:
            return "asymmetric"
        elif field_std > 1.0:
            return "turbulent"
        else:
            return "structured"
    
    def _calculate_local_symmetry(self, local_region: torch.Tensor) -> float:
        """Calculate local symmetry measure"""
        # Simple symmetry measure based on reflection
        if len(local_region.shape) == 3:
            # Check reflection symmetries
            x_symmetry = torch.mean(torch.abs(local_region - torch.flip(local_region, [0])))
            y_symmetry = torch.mean(torch.abs(local_region - torch.flip(local_region, [1])))
            z_symmetry = torch.mean(torch.abs(local_region - torch.flip(local_region, [2])))
            
            avg_symmetry = (x_symmetry + y_symmetry + z_symmetry) / 3
            symmetry_score = 1.0 / (1.0 + avg_symmetry.item())
        else:
            symmetry_score = 0.5
        
        return symmetry_score
    
    def _assess_structure_stability(self, local_region: torch.Tensor) -> float:
        """Assess stability of emergent structure"""
        # Stability based on local curvature and gradient consistency
        gradient_mag = self._calculate_gradient_magnitude(local_region)
        curvature = self._calculate_field_curvature(local_region)
        
        gradient_consistency = 1.0 / (1.0 + torch.std(gradient_mag).item())
        curvature_smoothness = 1.0 / (1.0 + torch.std(curvature).item())
        
        stability = (gradient_consistency + curvature_smoothness) / 2
        return stability
    
    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calculate skewness of tensor values"""
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)
        
        if std_val > 0:
            centered = tensor - mean_val
            skewness = torch.mean((centered / std_val) ** 3)
            return skewness.item()
        else:
            return 0.0
    
    def _label_connected_components_3d(self, binary_mask: torch.Tensor) -> torch.Tensor:
        """Simple 3D connected components labeling"""
        # Simplified connected components (not optimal but functional)
        labels = torch.zeros_like(binary_mask, dtype=torch.int32)
        current_label = 1
        
        if len(binary_mask.shape) == 3:
            h, w, d = binary_mask.shape
            
            for i in range(h):
                for j in range(w):
                    for k in range(d):
                        if binary_mask[i, j, k] and labels[i, j, k] == 0:
                            # Start flood fill
                            self._flood_fill_3d(binary_mask, labels, i, j, k, current_label)
                            current_label += 1
        
        return labels
    
    def _flood_fill_3d(self, binary_mask: torch.Tensor, labels: torch.Tensor, 
                      x: int, y: int, z: int, label: int):
        """3D flood fill for connected components"""
        h, w, d = binary_mask.shape
        stack = [(x, y, z)]
        
        while stack:
            cx, cy, cz = stack.pop()
            
            if (cx < 0 or cx >= h or cy < 0 or cy >= w or cz < 0 or cz >= d or
                not binary_mask[cx, cy, cz] or labels[cx, cy, cz] != 0):
                continue
            
            labels[cx, cy, cz] = label
            
            # Add neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        stack.append((cx + dx, cy + dy, cz + dz))
    
    def _create_structured_symbolic_field(self, 
                                        dimensions: Tuple[int, ...],
                                        complexity: float) -> torch.Tensor:
        """Create structured symbolic field"""
        field = torch.randn(dimensions, device=self.device)
        
        # Add structure through modulation
        if len(dimensions) == 3:
            h, w, d = dimensions
            x = torch.arange(h, device=self.device).float() / h
            y = torch.arange(w, device=self.device).float() / w
            z = torch.arange(d, device=self.device).float() / d
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            
            # Add structured modulation
            structure = (torch.sin(complexity * 2 * math.pi * X) *
                        torch.cos(complexity * 2 * math.pi * Y) *
                        torch.sin(complexity * 2 * math.pi * Z))
            
            field = field * (1 + 0.5 * structure)
        
        return field
    
    def _create_symbolic_resonance_field(self, 
                                       dimensions: Tuple[int, ...],
                                       complexity: float) -> torch.Tensor:
        """Create field with symbolic resonance patterns"""
        field = torch.zeros(dimensions, device=self.device)
        
        if len(dimensions) == 3:
            h, w, d = dimensions
            
            # Create multiple resonance modes
            for mode in range(int(complexity * 5) + 1):
                freq_x = (mode + 1) * self.resonance_frequency
                freq_y = (mode + 1) * self.resonance_frequency * 1.1
                freq_z = (mode + 1) * self.resonance_frequency * 0.9
                
                x = torch.arange(h, device=self.device).float()
                y = torch.arange(w, device=self.device).float()
                z = torch.arange(d, device=self.device).float()
                
                X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                
                mode_field = (torch.sin(freq_x * X / h * 2 * math.pi) *
                             torch.cos(freq_y * Y / w * 2 * math.pi) *
                             torch.sin(freq_z * Z / d * 2 * math.pi))
                
                field += mode_field / (mode + 1)  # Decreasing amplitude
        
        return field
    
    def _normalize_symbolic_field(self, field: torch.Tensor) -> torch.Tensor:
        """Normalize symbolic field for PAC conservation"""
        field_norm = torch.norm(field)
        if field_norm > 0:
            return field / field_norm
        else:
            return field
    
    def get_sec_metrics(self, result: SECResult) -> Dict[str, float]:
        """Get comprehensive SEC metrics"""
        metrics = {
            "total_entropy": torch.sum(result.entropy_profile).item(),
            "mean_entropy": torch.mean(result.entropy_profile).item(),
            "entropy_variance": torch.var(result.entropy_profile).item(),
            "num_collapses": len(result.collapse_locations),
            "collapse_strength": result.collapse_strength,
            "symbolic_resonance": result.symbolic_resonance,
            "geometric_phase": result.geometric_phase.value,
            "num_emergent_structures": len(result.emergent_structures),
            "field_energy": torch.sum(result.geometric_field**2).item(),
            "field_complexity": torch.std(result.geometric_field).item()
        }
        
        return metrics
