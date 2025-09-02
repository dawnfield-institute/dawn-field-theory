"""
GAIA v2.0 - Collapse Core Implementation
Physics-Informed Geometric Collapse Dynamics

TORCH ONLY - NO NUMPY
This implementation uses PyTorch with CUDA acceleration exclusively.
All tensor operations are performed on GPU when available.

This module implements the enhanced Collapse Core that:
1. Receives collapse triggers from Field Engine
2. Applies geometric collapse guidance using information curvature
3. Crystallizes symbolic structures from entropy resolution
4. Tracks thermodynamic costs and ancestry lineage
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GAIA Collapse Core using device: {device}")

# Import shared data structures
try:
    from .data_structures import FieldState, CollapseEvent, SymbolicStructure
except ImportError:
    # Fallback for direct execution
    from data_structures import FieldState, CollapseEvent, SymbolicStructure

# SCBF Integration - temporarily disabled for testing
SCBF_AVAILABLE = False
logging.warning("SCBF integration disabled for testing")


class CollapseType(Enum):
    """Types of collapse events"""
    FRACTAL_SYMBOL = "fractal_symbol"
    MEMORY_WAVE = "memory_wave"
    AGENT_SIGNAL = "agent_signal"
    GEOMETRIC_COLLAPSE = "geometric_collapse"
    THERMODYNAMIC_CRYSTALLIZATION = "thermodynamic_crystallization"


@dataclass
class SymbolicStructure:
    """Represents a crystallized symbolic structure from collapse"""
    structure_id: str
    collapse_location: Tuple[int, ...]
    symbolic_content: torch.Tensor
    entropy_signature: float
    thermodynamic_cost: float
    creation_timestamp: float
    ancestry_trace: List[str]
    geometric_properties: Dict[str, float]


@dataclass
class CurvatureTensor:
    """Information curvature tensor for geometric guidance"""
    gaussian_curvature: torch.Tensor
    mean_curvature: torch.Tensor
    ricci_scalar: torch.Tensor
    information_metric: torch.Tensor


class CollapseCore:
    """
    Enhanced Collapse Core implementing physics-informed geometric collapse
    
    This module converts field imbalance into crystallized symbolic structures
    using geometric guidance and thermodynamic optimization.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...],
                 geometric_guidance: bool = True,
                 thermodynamic_optimization: bool = True,
                 scbf_logging: bool = True,
                 temperature: float = 1.0):
        """
        Initialize the Collapse Core
        
        Args:
            field_shape: Shape of the associated field
            geometric_guidance: Enable information curvature guidance
            thermodynamic_optimization: Enable Landauer cost optimization
            scbf_logging: Enable SCBF ancestry tracking
            temperature: Thermodynamic temperature for cost calculations
        """
        self.field_shape = field_shape
        self.geometric_guidance = geometric_guidance
        self.thermodynamic_optimization = thermodynamic_optimization
        self.scbf_logging = scbf_logging and SCBF_AVAILABLE
        self.temperature = temperature
        
        # Constants
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant
        self.pi_harmonic_base = torch.tensor(torch.pi / 2, device=device)
        
        # State tracking
        self.symbolic_structures: List[SymbolicStructure] = []
        self.collapse_history: List[CollapseEvent] = []
        self.curvature_cache: Optional[CurvatureTensor] = None
        
        # SCBF Integration - disabled for testing
        if self.scbf_logging:
            # self.ancestry_metrics = ActivationAncestryMetrics()
            # self.lineage_metrics = BifractalLineageMetrics()
            # self.logger = ExperimentLogger(experiment_name="gaia_collapse_core")
            logging.info("SCBF logging requested but disabled for testing")
            self.scbf_logging = False
        
        logging.info(f"Collapse Core initialized with geometric_guidance={geometric_guidance}")
    
    def process_collapse_event(self, 
                             collapse_event: CollapseEvent,
                             field_state: FieldState) -> Optional[SymbolicStructure]:
        """
        Process a collapse event and potentially crystallize symbolic structure
        
        Args:
            collapse_event: The collapse event from Field Engine
            field_state: Current state of the field
            
        Returns:
            SymbolicStructure if crystallization occurred, None otherwise
        """
        logging.debug(f"Processing collapse event at {collapse_event.location}")
        
        # 1. Evaluate collapse conditions
        if not self._should_crystallize(collapse_event, field_state):
            logging.debug("Collapse conditions not met for crystallization")
            return None
        
        # 2. Determine collapse type
        collapse_type = self._determine_collapse_type(collapse_event, field_state)
        
        # 3. Apply geometric guidance if enabled
        if self.geometric_guidance:
            guidance_vector = self._compute_geometric_guidance(collapse_event.location, field_state)
        else:
            guidance_vector = None
        
        # 4. Crystallize symbolic structure
        symbolic_structure = self._crystallize_structure(
            collapse_event, field_state, collapse_type, guidance_vector
        )
        
        # 5. Apply thermodynamic optimization
        if self.thermodynamic_optimization:
            symbolic_structure = self._optimize_thermodynamic_cost(symbolic_structure)
        
        # 6. Log to SCBF if enabled
        if self.scbf_logging:
            self._log_scbf_ancestry(symbolic_structure)
        
        # 7. Store and return
        self.symbolic_structures.append(symbolic_structure)
        self.collapse_history.append(collapse_event)
        
        logging.info(f"Crystallized {collapse_type.value} structure {symbolic_structure.structure_id}")
        return symbolic_structure
    
    def _should_crystallize(self, collapse_event: CollapseEvent, field_state: FieldState) -> bool:
        """Determine if collapse should result in crystallization"""
        # Basic criteria: sufficient entropy delta and field pressure
        entropy_threshold = 0.1
        pressure_threshold = 0.3
        
        if collapse_event.entropy_delta < entropy_threshold:
            return False
        
        if collapse_event.field_pressure_pre < pressure_threshold:
            return False
        
        # Geometric criteria if enabled
        if self.geometric_guidance:
            curvature = self._compute_local_curvature(collapse_event.location, field_state)
            if curvature < 0.05:  # Too flat for interesting structure
                return False
        
        return True
    
    def _determine_collapse_type(self, collapse_event: CollapseEvent, field_state: FieldState) -> CollapseType:
        """Determine the type of collapse based on field conditions"""
        location = collapse_event.location
        
        # Analyze local field properties
        local_entropy = field_state.entropy_tensor[location]
        local_energy = field_state.energy_field[location]
        local_info = field_state.information_field[location]
        
        # Calculate field balance ratio
        balance_ratio = local_energy / (local_info + 1e-8)
        
        # Determine type based on field characteristics
        if self.geometric_guidance and self._has_high_curvature(location, field_state):
            return CollapseType.GEOMETRIC_COLLAPSE
        elif balance_ratio > 2.0:  # High energy dominance
            return CollapseType.FRACTAL_SYMBOL
        elif balance_ratio < 0.5:  # High information dominance
            return CollapseType.MEMORY_WAVE
        elif local_entropy > torch.mean(field_state.entropy_tensor).item() * 1.5:
            return CollapseType.THERMODYNAMIC_CRYSTALLIZATION
        else:
            return CollapseType.AGENT_SIGNAL
    
    def _compute_geometric_guidance(self, 
                                  location: Tuple[int, ...], 
                                  field_state: FieldState) -> torch.Tensor:
        """Compute geometric guidance vector using information curvature"""
        if self.curvature_cache is None:
            self.curvature_cache = self._compute_curvature_tensor(field_state)
        
        # Extract local curvature properties
        gaussian_curv = self.curvature_cache.gaussian_curvature[location]
        mean_curv = self.curvature_cache.mean_curvature[location]
        
        # Compute guidance vector based on curvature gradients
        guidance = torch.zeros(len(location), device=device, dtype=torch.float32)
        
        for i, loc in enumerate(location):
            if loc > 0 and loc < self.field_shape[i] - 1:
                # Gradient of Gaussian curvature
                gradient = (self.curvature_cache.gaussian_curvature[
                    tuple(loc + 1 if j == i else l for j, l in enumerate(location))
                ] - self.curvature_cache.gaussian_curvature[
                    tuple(loc - 1 if j == i else l for j, l in enumerate(location))
                ]) / 2.0
                guidance[i] = gradient
        
        # Normalize guidance vector
        norm = torch.norm(guidance)
        if norm > 1e-8:
            guidance = guidance / norm
        
        return guidance
    
    def _compute_curvature_tensor(self, field_state: FieldState) -> CurvatureTensor:
        """Compute information curvature tensor from field state"""
        # Combine energy and information fields into metric tensor
        combined_field = field_state.energy_field + 1j * field_state.information_field
        
        # Compute first and second derivatives
        grad_field = torch.gradient(combined_field)
        hessian_components = []
        
        for i, grad_component in enumerate(grad_field):
            hess_row = torch.gradient(grad_component)
            hessian_components.append(hess_row)
        
        # Approximate Gaussian curvature (for 2D case, extend for higher dimensions)
        if len(self.field_shape) == 2:
            # K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
            f_x, f_y = grad_field
            f_xx, f_xy = hessian_components[0]
            f_yx, f_yy = hessian_components[1]
            
            numerator = torch.real(f_xx * f_yy - f_xy * f_yx)
            denominator = (1 + torch.abs(f_x)**2 + torch.abs(f_y)**2)**2
            gaussian_curvature = numerator / (denominator + 1e-8)
            
            # Mean curvature: H = (f_xx + f_yy) / (1 + f_x^2 + f_y^2)^(3/2)
            mean_curvature = torch.real(f_xx + f_yy) / (denominator**(3/4) + 1e-8)
        else:
            # For higher dimensions, use simplified approximation
            gaussian_curvature = torch.abs(torch.real(combined_field))
            mean_curvature = torch.abs(torch.imag(combined_field))
        
        # Ricci scalar (simplified)
        ricci_scalar = gaussian_curvature + mean_curvature
        
        # Information metric (based on field gradients)
        information_metric = torch.abs(combined_field)**2
        
        return CurvatureTensor(
            gaussian_curvature=gaussian_curvature,
            mean_curvature=mean_curvature,
            ricci_scalar=ricci_scalar,
            information_metric=information_metric
        )
    
    def _has_high_curvature(self, location: Tuple[int, ...], field_state: FieldState) -> bool:
        """Check if location has high information curvature"""
        if self.curvature_cache is None:
            self.curvature_cache = self._compute_curvature_tensor(field_state)
        
        local_curvature = self.curvature_cache.gaussian_curvature[location]
        mean_curvature = torch.mean(self.curvature_cache.gaussian_curvature)
        
        return local_curvature > mean_curvature * 1.5
    
    def _compute_local_curvature(self, location: Tuple[int, ...], field_state: FieldState) -> float:
        """Compute local curvature at specific location"""
        if self.curvature_cache is None:
            self.curvature_cache = self._compute_curvature_tensor(field_state)
        
        return float(self.curvature_cache.gaussian_curvature[location])
    
    def _crystallize_structure(self,
                             collapse_event: CollapseEvent,
                             field_state: FieldState,
                             collapse_type: CollapseType,
                             guidance_vector: Optional[torch.Tensor]) -> SymbolicStructure:
        """Crystallize symbolic structure from collapse event"""
        location = collapse_event.location
        
        # Extract local field region for symbolic content
        region_size = 3  # 3x3 region around collapse
        local_region = self._extract_local_region(field_state, location, region_size)
        
        # Apply collapse-type specific crystallization
        if collapse_type == CollapseType.GEOMETRIC_COLLAPSE:
            symbolic_content = self._geometric_crystallization(local_region, guidance_vector)
        elif collapse_type == CollapseType.FRACTAL_SYMBOL:
            symbolic_content = self._fractal_crystallization(local_region)
        elif collapse_type == CollapseType.MEMORY_WAVE:
            symbolic_content = self._memory_crystallization(local_region)
        elif collapse_type == CollapseType.THERMODYNAMIC_CRYSTALLIZATION:
            symbolic_content = self._thermodynamic_crystallization(local_region)
        else:  # AGENT_SIGNAL
            symbolic_content = self._signal_crystallization(local_region)
        
        # Calculate geometric properties
        geometric_props = self._compute_geometric_properties(symbolic_content, guidance_vector)
        
        # Generate unique structure ID
        structure_id = f"{collapse_type.value}_{len(self.symbolic_structures):04d}_{collapse_event.timestamp}"
        
        # Create symbolic structure
        structure = SymbolicStructure(
            structure_id=structure_id,
            collapse_location=location,
            symbolic_content=symbolic_content,
            entropy_signature=collapse_event.entropy_delta,
            thermodynamic_cost=0.0,  # Will be computed in optimization
            creation_timestamp=collapse_event.timestamp,
            ancestry_trace=[],  # Will be populated by SCBF
            geometric_properties=geometric_props
        )
        
        return structure
    
    def _extract_local_region(self, 
                            field_state: FieldState, 
                            center: Tuple[int, ...], 
                            size: int) -> Dict[str, torch.Tensor]:
        """Extract local field region around collapse point"""
        # Calculate region bounds
        bounds = []
        for i, (c, s) in enumerate(zip(center, self.field_shape)):
            start = max(0, c - size // 2)
            end = min(s, c + size // 2 + 1)
            bounds.append((start, end))
        
        # Extract regions
        slices = tuple(slice(start, end) for start, end in bounds)
        
        return {
            'energy': field_state.energy_field[slices],
            'information': field_state.information_field[slices],
            'entropy': field_state.entropy_tensor[slices]
        }
    
    def _geometric_crystallization(self, local_region: Dict[str, torch.Tensor], 
                                 guidance_vector: Optional[torch.Tensor]) -> torch.Tensor:
        """Crystallize structure using geometric guidance"""
        # Combine fields with geometric weighting
        energy = local_region['energy']
        info = local_region['information']
        
        if guidance_vector is not None:
            # Weight fields based on guidance direction
            weight = torch.abs(guidance_vector).mean()
            combined = energy * (1 + weight) + info * (1 - weight)
        else:
            combined = energy + info
        
        # Apply geometric transformation (curvature-aware)
        return combined / (torch.max(combined) + 1e-8)
    
    def _fractal_crystallization(self, local_region: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Crystallize fractal symbolic structure"""
        energy = local_region['energy']
        
        # Create self-similar pattern
        pattern = energy.copy()
        for scale in [0.5, 0.25]:
            scaled_pattern = self._scale_pattern(pattern, scale)
            pattern += scaled_pattern
        
        return pattern / (torch.max(pattern) + 1e-8)
    
    def _memory_crystallization(self, local_region: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Crystallize memory wave structure"""
        info = local_region['information']
        
        # Emphasize stable, low-entropy regions
        stability_mask = 1.0 / (local_region['entropy'] + 1e-8)
        stable_info = info * stability_mask
        
        return stable_info / (torch.max(stable_info) + 1e-8)
    
    def _thermodynamic_crystallization(self, local_region: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Crystallize structure with thermodynamic optimization"""
        # Minimize energy while preserving information
        energy = local_region['energy']
        info = local_region['information']
        
        # Thermodynamic balance: minimize E while maximizing I
        efficiency = info / (energy + 1e-8)
        optimized = efficiency * torch.sqrt(energy * info)
        
        return optimized / (torch.max(optimized) + 1e-8)
    
    def _signal_crystallization(self, local_region: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Crystallize agent signal structure"""
        # Phase-coherent signal structure
        energy = local_region['energy']
        info = local_region['information']
        
        # Phase alignment
        phase = torch.angle(energy.to(torch.complex64) + 1j * info.to(torch.complex64))
        magnitude = torch.abs(energy.to(torch.complex64) + 1j * info.to(torch.complex64))
        
        signal = magnitude * torch.cos(phase)
        return signal / (torch.max(signal) + 1e-8)
    
    def _scale_pattern(self, pattern: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale pattern for fractal generation using torch interpolation"""
        # Use torch's interpolation for scaling
        try:
            # Add batch and channel dimensions for interpolation
            pattern_4d = pattern.unsqueeze(0).unsqueeze(0)
            
            # Calculate target size
            new_size = tuple(int(s * scale) for s in pattern.shape)
            
            # Downsample
            scaled_down = torch.nn.functional.interpolate(
                pattern_4d, size=new_size, mode='bilinear', align_corners=False
            )
            
            # Upsample back to original size
            scaled_back = torch.nn.functional.interpolate(
                scaled_down, size=pattern.shape, mode='bilinear', align_corners=False
            )
            
            # Remove batch and channel dimensions
            result = scaled_back.squeeze(0).squeeze(0)
            return result
        except Exception:
            # Fallback if interpolation fails
            return pattern * scale
    
    def _compute_geometric_properties(self, 
                                    symbolic_content: torch.Tensor,
                                    guidance_vector: Optional[torch.Tensor]) -> Dict[str, float]:
        """Compute geometric properties of crystallized structure"""
        props = {}
        
        # Basic statistics
        props['mean_value'] = float(torch.mean(symbolic_content))
        props['std_value'] = float(torch.std(symbolic_content))
        props['max_value'] = float(torch.max(symbolic_content))
        props['min_value'] = float(torch.min(symbolic_content))
        
        # Geometric measures
        props['compactness'] = float(torch.sum(symbolic_content > 0.5) / symbolic_content.numel())
        props['symmetry'] = self._compute_symmetry(symbolic_content)
        
        # Gradient-based measures
        gradients = torch.gradient(symbolic_content)
        gradient_magnitude = torch.sqrt(sum(g**2 for g in gradients))
        props['roughness'] = float(torch.mean(gradient_magnitude))
        
        # Guidance alignment if available
        if guidance_vector is not None:
            props['guidance_alignment'] = float(torch.dot(guidance_vector, gradients[0].flatten()[:len(guidance_vector)]))
        else:
            props['guidance_alignment'] = 0.0
        
        return props
    
    def _compute_symmetry(self, pattern: torch.Tensor) -> float:
        """Compute symmetry measure of pattern"""
        # Simple symmetry: compare with horizontal flip
        if len(pattern.shape) >= 2:
            flipped = torch.flip(pattern, dims=[1])
            symmetry = 1.0 - torch.mean(torch.abs(pattern - flipped))
        else:
            flipped = torch.flip(pattern, dims=[0])
            symmetry = 1.0 - torch.mean(torch.abs(pattern - flipped)).item()
        
        return float(max(0.0, symmetry))
    
    def _optimize_thermodynamic_cost(self, structure: SymbolicStructure) -> SymbolicStructure:
        """Optimize structure for thermodynamic efficiency"""
        if not self.thermodynamic_optimization:
            return structure
        
        # Calculate Landauer erasure cost
        n_bits = structure.symbolic_content.numel() if hasattr(structure.symbolic_content, 'numel') else len(structure.symbolic_content)
        erasure_cost = self.k_boltzmann * self.temperature * torch.log(torch.tensor(2.0, device=device)) * n_bits
        
        # Calculate information preserved
        information_preserved = -torch.sum(structure.symbolic_content * torch.log(structure.symbolic_content + 1e-8)).item()
        
        # Efficiency metric
        efficiency = information_preserved / (erasure_cost + 1e-8)
        
        # Apply efficiency optimization (modify symbolic content slightly)
        if efficiency < 1.0:
            # Increase efficiency by reducing redundancy
            structure.symbolic_content = self._reduce_redundancy(structure.symbolic_content)
            
        # Update thermodynamic cost
        structure.thermodynamic_cost = float(erasure_cost)
        structure.geometric_properties['thermodynamic_efficiency'] = float(efficiency)
        
        return structure
    
    def _reduce_redundancy(self, content: torch.Tensor) -> torch.Tensor:
        """Reduce redundancy in symbolic content for efficiency"""
        # Simple compression: quantize to reduce bit depth
        quantized = torch.round(content * 8) / 8  # 3-bit quantization
        return quantized
    
    def _log_scbf_ancestry(self, structure: SymbolicStructure) -> None:
        """Log structure ancestry to SCBF system"""
        if not self.scbf_logging:
            return
        
        ancestry_data = {
            'structure_id': structure.structure_id,
            'creation_timestamp': structure.creation_timestamp,
            'collapse_location': structure.collapse_location,
            'entropy_signature': structure.entropy_signature,
            'thermodynamic_cost': structure.thermodynamic_cost,
            'geometric_properties': structure.geometric_properties
        }
        
        self.logger.log_metrics(ancestry_data)
    
    def get_symbolic_structures(self) -> List[SymbolicStructure]:
        """Get all crystallized symbolic structures"""
        return self.symbolic_structures.copy()
    
    def get_structure_by_id(self, structure_id: str) -> Optional[SymbolicStructure]:
        """Get specific structure by ID"""
        for structure in self.symbolic_structures:
            if structure.structure_id == structure_id:
                return structure
        return None
    
    def reset(self) -> None:
        """Reset collapse core to initial state"""
        self.symbolic_structures.clear()
        self.collapse_history.clear()
        self.curvature_cache = None
        logging.info("Collapse Core reset to initial state")


if __name__ == "__main__":
    # Simple test of Collapse Core
    logging.basicConfig(level=logging.INFO)
    
    # Create test collapse event and field state
    from .field_engine import FieldEngine
    
    engine = FieldEngine(field_shape=(8, 8))
    test_stimulus = torch.rand((3, 3), device=device)
    engine.inject_stimulus(test_stimulus, "energy")
    
    # Run until we get a collapse
    collapse_core = CollapseCore(field_shape=(8, 8))
    
    for step in range(10):
        collapse_event = engine.step()
        if collapse_event:
            field_state = engine.get_field_state()
            structure = collapse_core.process_collapse_event(collapse_event, field_state)
            if structure:
                print(f"Created structure: {structure.structure_id}")
                print(f"Geometric properties: {structure.geometric_properties}")
                break
    
    print(f"Test complete. Structures created: {len(collapse_core.symbolic_structures)}")
