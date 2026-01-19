#!/usr/bin/env python3
"""
projections.py - Symmetric/Antisymmetric projection operators

The key insight: Maxwell uses antisymmetric projection (→ curl)
                 Gravity uses symmetric projection (→ divergence)

Both project from the same Möbius pre-field.
"""

import numpy as np
from typing import Tuple

# =============================================================================
# TENSOR DECOMPOSITION
# =============================================================================

def symmetric_part(tensor: np.ndarray) -> np.ndarray:
    """
    Extract symmetric part of a 2-tensor.
    S_ij = (T_ij + T_ji) / 2
    
    For a 3x3 tensor, this gives 6 independent components.
    Relates to: stress-energy tensor, metric perturbation
    """
    return (tensor + tensor.T) / 2


def antisymmetric_part(tensor: np.ndarray) -> np.ndarray:
    """
    Extract antisymmetric part of a 2-tensor.
    A_ij = (T_ij - T_ji) / 2
    
    For a 3x3 tensor, this gives 3 independent components.
    Relates to: electromagnetic field tensor F_μν
    """
    return (tensor - tensor.T) / 2


def decompose_tensor(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose tensor into symmetric + antisymmetric parts."""
    S = symmetric_part(tensor)
    A = antisymmetric_part(tensor)
    return S, A


# =============================================================================
# DIFFERENTIAL OPERATORS
# =============================================================================

def gradient_3d(field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D gradient of scalar field."""
    grad_x = np.gradient(field, dx, axis=2)
    grad_y = np.gradient(field, dx, axis=1)
    grad_z = np.gradient(field, dx, axis=0)
    return grad_x, grad_y, grad_z


def divergence_3d(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute 3D divergence of vector field.
    ∇·F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z
    
    This is the GRAVITY operator - sources create scalar potential.
    """
    dFx_dx = np.gradient(Fx, dx, axis=2)
    dFy_dy = np.gradient(Fy, dx, axis=1)
    dFz_dz = np.gradient(Fz, dx, axis=0)
    return dFx_dx + dFy_dy + dFz_dz


def curl_3d(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D curl of vector field.
    (∇×F)_x = ∂Fz/∂y - ∂Fy/∂z
    (∇×F)_y = ∂Fx/∂z - ∂Fz/∂x
    (∇×F)_z = ∂Fy/∂x - ∂Fx/∂y
    
    This is the MAXWELL operator - circulation creates field.
    """
    # Partials
    dFz_dy = np.gradient(Fz, dx, axis=1)
    dFy_dz = np.gradient(Fy, dx, axis=0)
    dFx_dz = np.gradient(Fx, dx, axis=0)
    dFz_dx = np.gradient(Fz, dx, axis=2)
    dFy_dx = np.gradient(Fy, dx, axis=2)
    dFx_dy = np.gradient(Fx, dx, axis=1)
    
    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    
    return curl_x, curl_y, curl_z


def laplacian_3d(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute 3D Laplacian.
    ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    
    Appears in both EM (wave equation) and gravity (Poisson equation).
    """
    d2_dx2 = np.gradient(np.gradient(field, dx, axis=2), dx, axis=2)
    d2_dy2 = np.gradient(np.gradient(field, dx, axis=1), dx, axis=1)
    d2_dz2 = np.gradient(np.gradient(field, dx, axis=0), dx, axis=0)
    return d2_dx2 + d2_dy2 + d2_dz2


# =============================================================================
# PROJECTION FROM PRE-FIELD
# =============================================================================

def project_antisymmetric(prefield: np.ndarray, hidden_axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project pre-field antisymmetrically (→ Maxwell).
    
    The hidden dimension becomes the curl structure.
    This extracts the PHASE information.
    """
    # Integrate out hidden dimension with oscillatory weight
    Nz = prefield.shape[hidden_axis]
    phase = np.exp(2j * np.pi * np.arange(Nz) / Nz)
    
    # Weight by phase and take imaginary parts (antisymmetric)
    if hidden_axis == 0:
        weighted = prefield * phase[:, np.newaxis, np.newaxis]
        projected = np.mean(weighted, axis=0)
    else:
        raise NotImplementedError("Only axis=0 supported")
    
    # Return as 3 components (curl has 3 components in 3D)
    Fx = np.real(projected)
    Fy = np.imag(projected)
    Fz = np.abs(projected) - np.mean(np.abs(projected))
    
    return Fx, Fy, Fz


def project_symmetric(prefield: np.ndarray, hidden_axis: int = 0) -> np.ndarray:
    """
    Project pre-field symmetrically (→ Gravity).
    
    This extracts the AMPLITUDE information.
    Returns scalar potential (not vector).
    """
    # Simple mean over hidden dimension (no phase weighting)
    projected = np.mean(np.abs(prefield), axis=hidden_axis)
    
    return projected


# =============================================================================
# DEPTH-BASED PROJECTION
# =============================================================================

def depth_2_projection(field_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MED depth-2 projection.
    
    Returns: (curl_field, divergence_source)
    The same 4D pre-field produces BOTH EM and gravity!
    """
    # Antisymmetric → EM (curl structure)
    em_x, em_y, em_z = project_antisymmetric(field_4d)
    
    # Symmetric → Gravity (divergence source)
    grav_potential = project_symmetric(field_4d)
    
    return (em_x, em_y, em_z), grav_potential


def depth_183_suppression(field: np.ndarray, depth_ratio: float = None) -> np.ndarray:
    """
    Apply the F₁₈₃ suppression factor.
    
    Gravity is suppressed by factor ~10⁻³⁸ relative to EM
    because it involves 183 levels of Fibonacci recursion.
    """
    from constants import LOG10_F183
    
    if depth_ratio is None:
        suppression = 10**(-LOG10_F183)
    else:
        suppression = 10**(-LOG10_F183 * depth_ratio)
    
    return field * suppression
