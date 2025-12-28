"""
PAC Mathematical Operations Module

Implements specialized mathematical operations for PAC conservation,
including exact conservation calculations, residual analysis, and
mathematical utilities for multi-scale physics simulation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ConservationMode(Enum):
    """Conservation calculation modes"""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    ADAPTIVE = "adaptive"

class ResidualType(Enum):
    """Types of conservation residuals"""
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    LINF_NORM = "linf_norm"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

@dataclass
class ConservationResult:
    """Result of PAC conservation calculation"""
    pre_total: float
    post_total: float
    residual: float
    relative_error: float
    conservation_quality: float
    method_used: str
    convergence_iterations: int
    numerical_stability: float

class PACMathematicalOperations:
    """
    Core mathematical operations for PAC conservation.
    
    Provides exact and approximate methods for enforcing
    f(parent) = Σf(children) across arbitrary field configurations.
    """
    
    def __init__(self, 
                 tolerance: float = 1e-12,
                 max_iterations: int = 1000,
                 device: str = "auto"):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Convergence parameters
        self.convergence_threshold = 1e-14
        self.stability_window = 10
        
    def enforce_exact_conservation(self, 
                                 field: torch.Tensor,
                                 parent_indices: torch.Tensor,
                                 child_indices: torch.Tensor,
                                 weights: Optional[torch.Tensor] = None) -> ConservationResult:
        """
        Enforce exact PAC conservation f(parent) = Σf(children).
        
        Args:
            field: The field values to conserve
            parent_indices: Indices of parent nodes
            child_indices: Indices corresponding to children of each parent
            weights: Optional weights for conservation (default: uniform)
            
        Returns:
            ConservationResult with enforcement details
        """
        field = field.to(self.device)
        parent_indices = parent_indices.to(self.device)
        child_indices = child_indices.to(self.device)
        
        if weights is None:
            weights = torch.ones_like(field)
        else:
            weights = weights.to(self.device)
            
        # Calculate initial totals
        pre_parent_total = torch.sum(field[parent_indices] * weights[parent_indices])
        pre_child_total = torch.sum(field[child_indices] * weights[child_indices])
        initial_residual = abs(pre_parent_total - pre_child_total).item()
        
        # If already conserved, return early
        if initial_residual < self.tolerance:
            return ConservationResult(
                pre_total=pre_parent_total.item(),
                post_total=pre_child_total.item(),
                residual=initial_residual,
                relative_error=initial_residual / (abs(pre_parent_total.item()) + 1e-16),
                conservation_quality=np.exp(-initial_residual),
                method_used="already_conserved",
                convergence_iterations=0,
                numerical_stability=1.0
            )
        
        # Apply iterative conservation enforcement
        modified_field = field.clone()
        residual_history = []
        
        for iteration in range(self.max_iterations):
            # Calculate current totals
            parent_total = torch.sum(modified_field[parent_indices] * weights[parent_indices])
            child_total = torch.sum(modified_field[child_indices] * weights[child_indices])
            current_residual = parent_total - child_total
            
            residual_history.append(abs(current_residual).item())
            
            # Check convergence
            if abs(current_residual) < self.tolerance:
                break
                
            # Distribute residual to minimize field perturbation
            correction = self._calculate_minimal_correction(
                modified_field, parent_indices, child_indices, 
                current_residual, weights
            )
            
            modified_field += correction
            
            # Check for numerical instability
            if iteration > self.stability_window:
                recent_residuals = residual_history[-self.stability_window:]
                if all(r > residual_history[0] for r in recent_residuals[-3:]):
                    # Diverging - apply stabilization
                    modified_field = self._apply_stabilization(
                        field, modified_field, parent_indices, child_indices, weights
                    )
                    break
        
        # Calculate final metrics
        final_parent_total = torch.sum(modified_field[parent_indices] * weights[parent_indices])
        final_child_total = torch.sum(modified_field[child_indices] * weights[child_indices])
        final_residual = abs(final_parent_total - final_child_total).item()
        
        # Calculate numerical stability
        field_change_norm = torch.norm(modified_field - field).item()
        total_field_norm = torch.norm(field).item()
        stability = 1.0 / (1.0 + field_change_norm / (total_field_norm + 1e-16))
        
        # Update original field
        field.copy_(modified_field)
        
        return ConservationResult(
            pre_total=pre_parent_total.item(),
            post_total=final_child_total.item(),
            residual=final_residual,
            relative_error=final_residual / (abs(final_parent_total.item()) + 1e-16),
            conservation_quality=np.exp(-final_residual),
            method_used="iterative_enforcement",
            convergence_iterations=iteration + 1,
            numerical_stability=stability
        )
    
    def _calculate_minimal_correction(self, 
                                    field: torch.Tensor,
                                    parent_indices: torch.Tensor,
                                    child_indices: torch.Tensor,
                                    residual: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """Calculate minimal field correction to achieve conservation"""
        correction = torch.zeros_like(field)
        
        # Distribute residual proportionally to minimize L2 norm of correction
        total_weight = torch.sum(weights[child_indices])
        
        if total_weight > 0:
            # Distribute residual to children proportionally
            for idx in child_indices:
                correction[idx] = residual * weights[idx] / total_weight
        
        return correction
    
    def _apply_stabilization(self, 
                           original_field: torch.Tensor,
                           current_field: torch.Tensor,
                           parent_indices: torch.Tensor,
                           child_indices: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        """Apply numerical stabilization when iterative method diverges"""
        # Blend with original field to maintain stability
        alpha = 0.5  # Stabilization factor
        stabilized_field = alpha * original_field + (1 - alpha) * current_field
        
        # Apply single-step exact correction
        parent_total = torch.sum(stabilized_field[parent_indices] * weights[parent_indices])
        child_total = torch.sum(stabilized_field[child_indices] * weights[child_indices])
        residual = parent_total - child_total
        
        # Direct redistribution
        total_child_weight = torch.sum(weights[child_indices])
        if total_child_weight > 0:
            for idx in child_indices:
                stabilized_field[idx] += residual * weights[idx] / total_child_weight
        
        return stabilized_field
    
    def calculate_conservation_residuals(self, 
                                       field: torch.Tensor,
                                       parent_indices: torch.Tensor,
                                       child_indices: torch.Tensor,
                                       residual_types: List[ResidualType] = None) -> Dict[str, float]:
        """
        Calculate various types of conservation residuals.
        
        Args:
            field: Field values
            parent_indices: Parent node indices
            child_indices: Child node indices
            residual_types: Types of residuals to calculate
            
        Returns:
            Dictionary of residual values
        """
        if residual_types is None:
            residual_types = [ResidualType.L2_NORM, ResidualType.RELATIVE]
            
        field = field.to(self.device)
        parent_indices = parent_indices.to(self.device)
        child_indices = child_indices.to(self.device)
        
        parent_total = torch.sum(field[parent_indices])
        child_total = torch.sum(field[child_indices])
        residual_vector = parent_total - child_total
        
        residuals = {}
        
        for residual_type in residual_types:
            if residual_type == ResidualType.L1_NORM:
                residuals["l1_norm"] = torch.abs(residual_vector).item()
            elif residual_type == ResidualType.L2_NORM:
                residuals["l2_norm"] = torch.sqrt(residual_vector**2).item()
            elif residual_type == ResidualType.LINF_NORM:
                residuals["linf_norm"] = torch.abs(residual_vector).item()
            elif residual_type == ResidualType.RELATIVE:
                total_magnitude = abs(parent_total.item()) + abs(child_total.item())
                residuals["relative"] = abs(residual_vector.item()) / (total_magnitude + 1e-16)
            elif residual_type == ResidualType.ABSOLUTE:
                residuals["absolute"] = abs(residual_vector.item())
        
        return residuals
    
    def analyze_conservation_stability(self, 
                                     field_history: List[torch.Tensor],
                                     parent_indices: torch.Tensor,
                                     child_indices: torch.Tensor) -> Dict[str, float]:
        """
        Analyze conservation stability over time.
        
        Args:
            field_history: List of field states over time
            parent_indices: Parent node indices
            child_indices: Child node indices
            
        Returns:
            Stability analysis metrics
        """
        residual_history = []
        
        for field in field_history:
            field = field.to(self.device)
            parent_total = torch.sum(field[parent_indices])
            child_total = torch.sum(field[child_indices])
            residual = abs(parent_total - child_total).item()
            residual_history.append(residual)
        
        residual_array = np.array(residual_history)
        
        return {
            "mean_residual": np.mean(residual_array),
            "std_residual": np.std(residual_array),
            "max_residual": np.max(residual_array),
            "min_residual": np.min(residual_array),
            "trend_slope": np.polyfit(range(len(residual_array)), residual_array, 1)[0],
            "stability_coefficient": 1.0 / (1.0 + np.std(residual_array))
        }
    
    def compute_conservation_quality_metric(self, 
                                          residual: float,
                                          field_magnitude: float = 1.0) -> float:
        """
        Compute a quality metric for conservation.
        
        Args:
            residual: Conservation residual
            field_magnitude: Scale of the field for normalization
            
        Returns:
            Quality metric between 0 and 1
        """
        # Exponential decay quality metric
        normalized_residual = residual / (field_magnitude + 1e-16)
        quality = np.exp(-normalized_residual)
        return min(1.0, max(0.0, quality))
    
    def suggest_optimization_parameters(self, 
                                      field_properties: Dict[str, float]) -> Dict[str, float]:
        """
        Suggest optimization parameters based on field properties.
        
        Args:
            field_properties: Properties of the field (magnitude, sparsity, etc.)
            
        Returns:
            Suggested parameters for conservation enforcement
        """
        magnitude = field_properties.get("magnitude", 1.0)
        sparsity = field_properties.get("sparsity", 0.0)
        dimension = field_properties.get("dimension", 3)
        
        # Adaptive parameter suggestions
        suggested_tolerance = max(1e-14, magnitude * 1e-12)
        suggested_max_iterations = min(10000, max(100, int(1000 * sparsity)))
        
        return {
            "tolerance": suggested_tolerance,
            "max_iterations": suggested_max_iterations,
            "convergence_threshold": suggested_tolerance * 0.01,
            "stability_factor": 0.1 + 0.4 * (1.0 - sparsity)
        }

# Utility functions for common PAC mathematical operations
def calculate_pac_field_properties(field: torch.Tensor) -> Dict[str, float]:
    """Calculate properties of a PAC field for optimization"""
    field_flat = field.flatten()
    
    return {
        "magnitude": torch.norm(field).item(),
        "mean": torch.mean(field).item(),
        "std": torch.std(field).item(),
        "sparsity": (field_flat == 0).float().mean().item(),
        "dimension": len(field.shape),
        "total_elements": field.numel(),
        "dynamic_range": (torch.max(field) - torch.min(field)).item()
    }

def create_conservation_matrix(parent_child_mapping: Dict[int, List[int]], 
                             total_nodes: int,
                             device: str = "cpu") -> torch.Tensor:
    """Create conservation constraint matrix for linear algebra methods"""
    device = torch.device(device)
    num_constraints = len(parent_child_mapping)
    conservation_matrix = torch.zeros(num_constraints, total_nodes, device=device)
    
    for constraint_idx, (parent, children) in enumerate(parent_child_mapping.items()):
        # Parent coefficient: +1
        conservation_matrix[constraint_idx, parent] = 1.0
        
        # Children coefficients: -1 each
        for child in children:
            conservation_matrix[constraint_idx, child] = -1.0
    
    return conservation_matrix

def solve_conservation_linear_system(conservation_matrix: torch.Tensor,
                                    current_field: torch.Tensor,
                                    target_residuals: torch.Tensor = None) -> torch.Tensor:
    """Solve conservation as a linear system for exact enforcement"""
    if target_residuals is None:
        target_residuals = torch.zeros(conservation_matrix.shape[0], device=conservation_matrix.device)
    
    # Solve: conservation_matrix @ field_correction = target_residuals - current_residuals
    current_residuals = conservation_matrix @ current_field
    rhs = target_residuals - current_residuals
    
    # Use least squares if system is overdetermined
    try:
        correction = torch.linalg.lstsq(conservation_matrix.T, rhs).solution
        return correction
    except:
        # Fallback to pseudo-inverse
        pseudo_inv = torch.linalg.pinv(conservation_matrix.T)
        correction = pseudo_inv @ rhs
        return correction
