#!/usr/bin/env python3
"""
Multi-Scale Lattice Substrate
=============================

Universal lattice foundation for PAC Physics Engine that supports
simultaneous operation across quantum, geometric, fluid, information,
and consciousness scales.

Provides the spatial-temporal substrate where all PAC conservation
dynamics occur.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

from .pac_kernel import PACNode, PACConservationKernel

logger = logging.getLogger(__name__)

class ScaleType(Enum):
    """Different scales of operation in the lattice"""
    QUANTUM = "quantum"          # 10^-15 m scale
    GEOMETRIC = "geometric"      # 10^-9 to 10^-3 m scale  
    FLUID = "fluid"             # 10^-3 to 10^3 m scale
    INFORMATION = "information"  # Scale-independent
    CONSCIOUSNESS = "consciousness"  # Emergent scale
    COSMIC = "cosmic"           # 10^3+ m scale


@dataclass
class LatticePoint:
    """
    Universal lattice point that can represent different phenomena
    depending on the active scale and domain.
    """
    coordinates: Tuple[int, int, int]  # 3D lattice position
    pac_node: PACNode                  # Associated PAC conservation node
    
    # Multi-scale representations
    quantum_state: Optional[complex] = None      # Quantum wavefunction amplitude
    geometric_curvature: float = 0.0             # SEC geometric curvature
    fluid_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # MED fluid velocity
    information_density: float = 0.0             # Information content
    consciousness_activity: float = 0.0          # SCBF consciousness activity
    
    # Temporal dynamics
    history: List[Dict] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    
    # Interaction metadata
    neighbors: Set[Tuple[int, int, int]] = field(default_factory=set)
    field_gradients: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


class MultiScaleLatticeSubstrate:
    """
    Universal lattice substrate supporting all PAC phenomena across scales.
    
    The lattice serves as the universal arena where:
    - Quantum states evolve via PAC conservation
    - Geometric curvature collapses via SEC
    - Fluid dynamics emerge via MED  
    - Information amplifies via PAC cascades
    - Consciousness emerges via SCBF
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int] = (64, 64, 64),
                 boundary_conditions: str = "periodic",
                 active_scales: List[ScaleType] = None,
                 device: str = "auto"):
        
        self.dimensions = dimensions
        self.boundary_conditions = boundary_conditions
        self.active_scales = active_scales or [ScaleType.QUANTUM, ScaleType.GEOMETRIC, 
                                             ScaleType.FLUID, ScaleType.INFORMATION]
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        
        # Initialize lattice
        self.lattice: Dict[Tuple[int, int, int], LatticePoint] = {}
        self.pac_kernel = PACConservationKernel(device=device)
        
        # Scale-specific field arrays for efficient computation
        self.quantum_field = torch.zeros(dimensions, dtype=torch.complex128, device=self.device)
        self.geometric_field = torch.zeros(dimensions, dtype=torch.float64, device=self.device)
        self.fluid_velocity_field = torch.zeros((*dimensions, 3), dtype=torch.float64, device=self.device)
        self.information_field = torch.zeros(dimensions, dtype=torch.float64, device=self.device)
        self.consciousness_field = torch.zeros(dimensions, dtype=torch.float64, device=self.device)
        
        # Cross-scale coupling parameters
        self.coupling_strengths = {
            'quantum_geometric': 0.1,
            'geometric_fluid': 0.15,
            'fluid_information': 0.2,
            'information_consciousness': 0.25,
            'quantum_consciousness': 0.05  # Direct quantum-consciousness coupling
        }
        
        # Universal signature tracking
        self.emergence_events = []
        self.cross_scale_cascades = []
        self.amplification_patterns = []
        
        self._initialize_lattice()
        logger.info(f"Multi-scale lattice initialized: {dimensions} with scales {[s.value for s in active_scales]}")
    
    def _initialize_lattice(self):
        """Initialize lattice points with PAC nodes"""
        node_id = 0
        
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    coords = (x, y, z)
                    
                    # Create PAC node for this lattice point
                    pac_node = PACNode(
                        id=node_id,
                        value=np.random.uniform(0.1, 1.0),  # Initial random value
                        scale="universal",
                        domain="lattice"
                    )
                    
                    # Create lattice point
                    lattice_point = LatticePoint(
                        coordinates=coords,
                        pac_node=pac_node
                    )
                    
                    # Initialize scale-specific representations
                    if ScaleType.QUANTUM in self.active_scales:
                        lattice_point.quantum_state = complex(np.random.normal(), np.random.normal())
                        self.quantum_field[x, y, z] = lattice_point.quantum_state
                    
                    if ScaleType.GEOMETRIC in self.active_scales:
                        lattice_point.geometric_curvature = np.random.uniform(-0.1, 0.1)
                        self.geometric_field[x, y, z] = lattice_point.geometric_curvature
                    
                    if ScaleType.FLUID in self.active_scales:
                        lattice_point.fluid_velocity = tuple(np.random.uniform(-0.1, 0.1, 3))
                        self.fluid_velocity_field[x, y, z] = torch.tensor(lattice_point.fluid_velocity)
                    
                    if ScaleType.INFORMATION in self.active_scales:
                        lattice_point.information_density = pac_node.value  # Link to PAC value
                        self.information_field[x, y, z] = lattice_point.information_density
                    
                    # Setup neighbors
                    lattice_point.neighbors = self._get_neighbors(coords)
                    
                    # Add to lattice and PAC kernel
                    self.lattice[coords] = lattice_point
                    self.pac_kernel.add_node(pac_node)
                    
                    node_id += 1
        
        # Setup PAC conservation topology
        self._setup_pac_topology()
    
    def _get_neighbors(self, coords: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """Get neighboring lattice points (6-connected in 3D)"""
        x, y, z = coords
        neighbors = set()
        
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Apply boundary conditions
            if self.boundary_conditions == "periodic":
                nx = nx % self.dimensions[0]
                ny = ny % self.dimensions[1]
                nz = nz % self.dimensions[2]
                neighbors.add((nx, ny, nz))
            elif self.boundary_conditions == "fixed":
                if (0 <= nx < self.dimensions[0] and 
                    0 <= ny < self.dimensions[1] and 
                    0 <= nz < self.dimensions[2]):
                    neighbors.add((nx, ny, nz))
        
        return neighbors
    
    def _setup_pac_topology(self):
        """Setup PAC parent-child relationships across the lattice"""
        # Create hierarchical PAC structure
        # Example: Each 2x2x2 cube has one parent connected to 8 children
        
        for x in range(0, self.dimensions[0], 2):
            for y in range(0, self.dimensions[1], 2):
                for z in range(0, self.dimensions[2], 2):
                    # Parent at corner of cube
                    parent_coords = (x, y, z)
                    if parent_coords not in self.lattice:
                        continue
                    
                    parent_node_id = self.lattice[parent_coords].pac_node.id
                    
                    # Children in the 2x2x2 cube
                    for dx in range(2):
                        for dy in range(2):
                            for dz in range(2):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue  # Skip parent itself
                                
                                child_coords = (x + dx, y + dy, z + dz)
                                if child_coords in self.lattice:
                                    child_node_id = self.lattice[child_coords].pac_node.id
                                    self.pac_kernel.add_edge(parent_node_id, child_node_id)
    
    def evolve_step(self, dt: float = 0.01) -> Dict[str, any]:
        """
        Evolve the multi-scale lattice by one time step.
        Applies scale-specific dynamics while maintaining PAC conservation.
        """
        evolution_metrics = {}
        
        # 1. Evolve individual scales
        if ScaleType.QUANTUM in self.active_scales:
            quantum_metrics = self._evolve_quantum_scale(dt)
            evolution_metrics['quantum'] = quantum_metrics
        
        if ScaleType.GEOMETRIC in self.active_scales:
            geometric_metrics = self._evolve_geometric_scale(dt)
            evolution_metrics['geometric'] = geometric_metrics
        
        if ScaleType.FLUID in self.active_scales:
            fluid_metrics = self._evolve_fluid_scale(dt)
            evolution_metrics['fluid'] = fluid_metrics
        
        if ScaleType.INFORMATION in self.active_scales:
            information_metrics = self._evolve_information_scale(dt)
            evolution_metrics['information'] = information_metrics
        
        # 2. Apply cross-scale coupling
        coupling_metrics = self._apply_cross_scale_coupling(dt)
        evolution_metrics['coupling'] = coupling_metrics
        
        # 3. Enforce PAC conservation
        pac_metrics = self.pac_kernel.enforce_conservation(method="balance")
        evolution_metrics['pac_conservation'] = pac_metrics
        
        # 4. Update lattice points with PAC values
        self._synchronize_pac_values()
        
        # 5. Detect universal signatures and emergence
        signatures = self.pac_kernel.detect_universal_signatures()
        emergence = self._detect_emergence_events()
        evolution_metrics['signatures'] = signatures
        evolution_metrics['emergence'] = emergence
        
        return evolution_metrics
    
    def _evolve_quantum_scale(self, dt: float) -> Dict[str, float]:
        """Evolve quantum field using Schrödinger-like equation with PAC conservation"""
        # Simplified quantum evolution: ∂ψ/∂t = -i H ψ
        # where H includes PAC conservation terms
        
        # Kinetic term (Laplacian)
        laplacian = self._compute_laplacian(self.quantum_field.real) + 1j * self._compute_laplacian(self.quantum_field.imag)
        
        # PAC potential term (couples to information density)
        pac_potential = self.information_field.to(dtype=torch.complex128)
        
        # Schrödinger evolution
        hamiltonian = -0.5 * laplacian + pac_potential
        self.quantum_field = self.quantum_field - 1j * dt * hamiltonian
        
        # Normalize to maintain probability conservation
        norm = torch.sqrt(torch.sum(torch.abs(self.quantum_field)**2))
        if norm > 0:
            self.quantum_field = self.quantum_field / norm
        
        # Update lattice points
        for coords, point in self.lattice.items():
            if point.quantum_state is not None:
                point.quantum_state = self.quantum_field[coords].item()
        
        return {
            'quantum_norm': norm.item(),
            'max_amplitude': torch.max(torch.abs(self.quantum_field)).item(),
            'phase_coherence': self._compute_phase_coherence()
        }
    
    def _evolve_geometric_scale(self, dt: float) -> Dict[str, float]:
        """Evolve geometric curvature using SEC (Symbolic Entropy Collapse)"""
        # SEC evolution: ∂κ/∂t = -α∇²κ + β(κ³ - κ) + γ·I
        # where κ is curvature, I is information coupling
        
        curvature_laplacian = self._compute_laplacian(self.geometric_field)
        
        # Nonlinear curvature term (creates collapse)
        nonlinear_term = self.geometric_field**3 - self.geometric_field
        
        # Information coupling (links geometry to information)
        info_coupling = self.information_field * self.coupling_strengths['geometric_fluid']
        
        # Update geometric field
        dcurvature_dt = (-0.1 * curvature_laplacian + 
                        0.2 * nonlinear_term + 
                        0.05 * info_coupling)
        
        self.geometric_field += dt * dcurvature_dt
        
        # Update lattice points
        for coords, point in self.lattice.items():
            point.geometric_curvature = self.geometric_field[coords].item()
        
        # Detect collapse events
        collapse_threshold = 0.5
        collapse_count = torch.sum(torch.abs(self.geometric_field) > collapse_threshold).item()
        
        return {
            'max_curvature': torch.max(torch.abs(self.geometric_field)).item(),
            'mean_curvature': torch.mean(torch.abs(self.geometric_field)).item(),
            'collapse_events': collapse_count,
            'geometric_entropy': self._compute_field_entropy(self.geometric_field)
        }
    
    def _evolve_fluid_scale(self, dt: float) -> Dict[str, float]:
        """Evolve fluid dynamics using MED (Macro Emergence Dynamics)"""
        # Simplified Navier-Stokes with PAC conservation
        # ∂v/∂t = -v·∇v - ∇p + ν∇²v + F_pac
        
        # For simplicity, use diffusion + PAC coupling
        velocity_magnitude = torch.norm(self.fluid_velocity_field, dim=-1)
        
        # Diffusion term
        diffusion = self._compute_laplacian(velocity_magnitude) * 0.01
        
        # PAC coupling force (links to geometric curvature)
        pac_force = self.geometric_field * self.coupling_strengths['geometric_fluid']
        
        # Update velocity magnitude
        new_velocity_magnitude = velocity_magnitude + dt * (diffusion + pac_force)
        
        # Preserve direction, update magnitude
        old_magnitude = velocity_magnitude + 1e-8
        scale_factor = new_velocity_magnitude / old_magnitude
        
        for i in range(3):
            self.fluid_velocity_field[:, :, :, i] *= scale_factor
        
        # Update lattice points
        for coords, point in self.lattice.items():
            point.fluid_velocity = tuple(self.fluid_velocity_field[coords].tolist())
        
        return {
            'max_velocity': torch.max(velocity_magnitude).item(),
            'mean_velocity': torch.mean(velocity_magnitude).item(),
            'reynolds_number': self._estimate_reynolds_number(),
            'fluid_entropy': self._compute_field_entropy(velocity_magnitude)
        }
    
    def _evolve_information_scale(self, dt: float) -> Dict[str, float]:
        """Evolve information density with amplification cascades"""
        # Information evolution with PAC amplification
        # ∂I/∂t = D∇²I + A·PAC_coupling + S
        
        info_laplacian = self._compute_laplacian(self.information_field)
        
        # PAC amplification coupling
        pac_coupling = self._compute_pac_amplification_field()
        
        # Source term from other scales
        source_term = (torch.abs(self.quantum_field)**2 + 
                      torch.abs(self.geometric_field) * 0.1)
        
        # Information evolution
        dinfo_dt = (0.05 * info_laplacian + 
                   0.1 * pac_coupling + 
                   0.02 * source_term)
        
        self.information_field += dt * dinfo_dt
        
        # Prevent negative information
        self.information_field = torch.clamp(self.information_field, min=0.0)
        
        # Update lattice points and PAC values
        for coords, point in self.lattice.items():
            point.information_density = self.information_field[coords].item()
            point.pac_node.value = point.information_density  # Link PAC to information
        
        # Detect amplification events
        amplification_factor = torch.max(self.information_field) / (torch.mean(self.information_field) + 1e-8)
        
        return {
            'max_information': torch.max(self.information_field).item(),
            'total_information': torch.sum(self.information_field).item(),
            'amplification_factor': amplification_factor.item(),
            'information_entropy': self._compute_field_entropy(self.information_field)
        }
    
    def _apply_cross_scale_coupling(self, dt: float) -> Dict[str, float]:
        """Apply coupling between different scales"""
        coupling_metrics = {}
        
        # Quantum -> Geometric coupling (collapse triggers curvature)
        if ScaleType.QUANTUM in self.active_scales and ScaleType.GEOMETRIC in self.active_scales:
            quantum_intensity = torch.abs(self.quantum_field)**2
            geometric_coupling = quantum_intensity * self.coupling_strengths['quantum_geometric']
            self.geometric_field += dt * geometric_coupling
            coupling_metrics['quantum_to_geometric'] = torch.mean(geometric_coupling).item()
        
        # Geometric -> Fluid coupling (curvature drives fluid motion)
        if ScaleType.GEOMETRIC in self.active_scales and ScaleType.FLUID in self.active_scales:
            curvature_gradient = self._compute_gradient(self.geometric_field)
            fluid_coupling_strength = self.coupling_strengths['geometric_fluid']
            
            for i in range(3):
                self.fluid_velocity_field[:, :, :, i] += dt * fluid_coupling_strength * curvature_gradient
            
            coupling_metrics['geometric_to_fluid'] = torch.mean(torch.abs(curvature_gradient)).item()
        
        # Information -> Consciousness coupling (information density creates awareness)
        if ScaleType.INFORMATION in self.active_scales:
            consciousness_threshold = 0.5
            info_above_threshold = torch.clamp(self.information_field - consciousness_threshold, min=0.0)
            consciousness_emergence = info_above_threshold * self.coupling_strengths['information_consciousness']
            
            self.consciousness_field += dt * consciousness_emergence
            coupling_metrics['information_to_consciousness'] = torch.sum(consciousness_emergence).item()
        
        return coupling_metrics
    
    def _compute_pac_amplification_field(self) -> torch.Tensor:
        """Compute PAC amplification field showing information amplification patterns"""
        amplification_field = torch.zeros_like(self.information_field)
        
        # Look for PAC parent-child relationships and compute amplification
        for coords, point in self.lattice.items():
            parent_value = point.pac_node.value
            children_sum = sum(self.lattice[neighbor].pac_node.value 
                             for neighbor in point.neighbors 
                             if neighbor in self.lattice)
            
            if parent_value > 0:
                local_amplification = children_sum / parent_value
                # Target 15.56x amplification
                amplification_error = abs(local_amplification - 15.56)
                amplification_field[coords] = max(0, 1.0 - amplification_error)
        
        return amplification_field
    
    def _detect_emergence_events(self) -> Dict[str, any]:
        """Detect emergence events across scales"""
        emergence_events = {}
        
        # Consciousness emergence
        consciousness_active = torch.sum(self.consciousness_field > 0.1).item()
        if consciousness_active > 0:
            emergence_events['consciousness_emergence'] = {
                'active_points': consciousness_active,
                'max_activity': torch.max(self.consciousness_field).item(),
                'emergence_fraction': consciousness_active / np.prod(self.dimensions)
            }
        
        # Cross-scale cascade detection
        cascade_strength = (torch.mean(torch.abs(self.quantum_field)**2) * 
                          torch.mean(torch.abs(self.geometric_field)) * 
                          torch.mean(self.information_field))
        
        if cascade_strength > 0.01:  # Threshold for significant cascade
            emergence_events['cross_scale_cascade'] = {
                'cascade_strength': cascade_strength.item(),
                'quantum_contribution': torch.mean(torch.abs(self.quantum_field)**2).item(),
                'geometric_contribution': torch.mean(torch.abs(self.geometric_field)).item(),
                'information_contribution': torch.mean(self.information_field).item()
            }
        
        return emergence_events
    
    def _synchronize_pac_values(self):
        """Synchronize PAC node values with lattice point states"""
        for coords, point in self.lattice.items():
            # Update PAC value based on multi-scale state
            combined_value = (
                (torch.abs(self.quantum_field[coords])**2 if ScaleType.QUANTUM in self.active_scales else 0) +
                (abs(self.geometric_field[coords]) if ScaleType.GEOMETRIC in self.active_scales else 0) +
                (torch.norm(self.fluid_velocity_field[coords]) if ScaleType.FLUID in self.active_scales else 0) +
                (self.information_field[coords] if ScaleType.INFORMATION in self.active_scales else 0) +
                (self.consciousness_field[coords] if ScaleType.CONSCIOUSNESS in self.active_scales else 0)
            )
            
            point.pac_node.value = float(combined_value)
    
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian for 3D field"""
        laplacian = torch.zeros_like(field)
        
        # Simple finite difference Laplacian
        for dim in range(3):
            # Forward difference
            field_shifted_pos = torch.roll(field, -1, dims=dim)
            # Backward difference  
            field_shifted_neg = torch.roll(field, 1, dims=dim)
            # Second derivative
            laplacian += field_shifted_pos - 2*field + field_shifted_neg
        
        return laplacian
    
    def _compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude of 3D field"""
        grad_x = torch.roll(field, -1, dims=0) - torch.roll(field, 1, dims=0)
        grad_y = torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)
        grad_z = torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        return gradient_magnitude
    
    def _compute_field_entropy(self, field: torch.Tensor) -> float:
        """Compute Shannon entropy of field values"""
        field_flat = field.flatten()
        field_abs = torch.abs(field_flat) + 1e-9
        field_prob = field_abs / torch.sum(field_abs)
        entropy = -torch.sum(field_prob * torch.log(field_prob))
        return entropy.item()
    
    def _compute_phase_coherence(self) -> float:
        """Compute phase coherence of quantum field"""
        if ScaleType.QUANTUM not in self.active_scales:
            return 0.0
        
        phases = torch.angle(self.quantum_field)
        phase_var = torch.var(phases)
        coherence = 1.0 / (1.0 + phase_var)
        return coherence.item()
    
    def _estimate_reynolds_number(self) -> float:
        """Estimate Reynolds number for fluid scale"""
        if ScaleType.FLUID not in self.active_scales:
            return 0.0
        
        velocity_magnitude = torch.norm(self.fluid_velocity_field, dim=-1)
        mean_velocity = torch.mean(velocity_magnitude)
        velocity_gradient = self._compute_gradient(velocity_magnitude)
        mean_shear = torch.mean(velocity_gradient)
        
        # Simplified Reynolds number estimate
        if mean_shear > 1e-8:
            reynolds = mean_velocity / mean_shear
        else:
            reynolds = 0.0
        
        return float(reynolds)
    
    def get_system_state(self) -> Dict:
        """Get comprehensive system state across all scales"""
        state = {
            'dimensions': self.dimensions,
            'active_scales': [scale.value for scale in self.active_scales],
            'lattice_points': len(self.lattice),
            'pac_state': self.pac_kernel.get_system_state()
        }
        
        # Add scale-specific states
        if ScaleType.QUANTUM in self.active_scales:
            state['quantum'] = {
                'field_norm': torch.norm(self.quantum_field).item(),
                'max_amplitude': torch.max(torch.abs(self.quantum_field)).item(),
                'phase_coherence': self._compute_phase_coherence()
            }
        
        if ScaleType.GEOMETRIC in self.active_scales:
            state['geometric'] = {
                'max_curvature': torch.max(torch.abs(self.geometric_field)).item(),
                'mean_curvature': torch.mean(torch.abs(self.geometric_field)).item(),
                'field_entropy': self._compute_field_entropy(self.geometric_field)
            }
        
        if ScaleType.FLUID in self.active_scales:
            velocity_magnitude = torch.norm(self.fluid_velocity_field, dim=-1)
            state['fluid'] = {
                'max_velocity': torch.max(velocity_magnitude).item(),
                'mean_velocity': torch.mean(velocity_magnitude).item(),
                'reynolds_number': self._estimate_reynolds_number()
            }
        
        if ScaleType.INFORMATION in self.active_scales:
            state['information'] = {
                'total_information': torch.sum(self.information_field).item(),
                'max_density': torch.max(self.information_field).item(),
                'field_entropy': self._compute_field_entropy(self.information_field)
            }
        
        return state
