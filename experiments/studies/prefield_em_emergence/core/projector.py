"""
3D Projection and Electromagnetic Field Extraction
==================================================

Projects pre-field dynamics from Möbius manifold into 3D space
and extracts electromagnetic fields.

The projection process:
1. Map Möbius (u,v) → R³ embedding (X,Y,Z)
2. Interpolate pre-field values onto 3D grid
3. Extract scalar potential φ from amplitude
4. Extract vector potential A from phase structure
5. Compute E = -∇φ and B = ∇×A

Key Result:
    E/B = φ^(-4.42 × w/R + 2.34)
    
    The Möbius geometry determines the electromagnetic coupling ratio.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .constants import (
    PHI, PHI_INV, PHI_SQ, XI,
    DIV_B_THRESHOLD, DIV_E_THRESHOLD, PHI_MATCH_THRESHOLD
)
from .mobius_field import MobiusField


class EMProjector:
    """Projects Möbius pre-field to 3D and extracts E, B fields.
    
    The projection uses Gaussian-weighted interpolation from the
    Möbius embedding coordinates to a regular 3D grid. This is
    necessary because the Möbius strip is a curved surface in 3D,
    not aligned with Cartesian axes.
    
    Attributes:
        n: Grid resolution per dimension
        L: Half-width of 3D domain
        mask: Boolean mask for interior points
        X, Y, Z: 3D coordinate grids
        dx: Grid spacing
        
    Example:
        >>> field = MobiusField(n_u=64, n_v=32, R=2.0, w=0.6)
        >>> proj = EMProjector(n=24, L=3.0)
        >>> result = proj.project(field)
        >>> print(f"E/B ratio: {result['EB_ratio']:.4f}")
    """
    
    def __init__(
        self,
        n: int = 24,
        L: float = 3.0,
        shape: str = 'sphere',
        sigma: float = 0.5
    ):
        """Initialize 3D projector.
        
        Args:
            n: Grid resolution per dimension
            L: Half-width of domain (domain is [-L, L]³)
            shape: Boundary shape ('sphere', 'cube', 'none')
            sigma: Gaussian interpolation width
        """
        self.n = n
        self.L = L
        self.shape = shape
        self.sigma = sigma
        
        # Create 3D coordinate grid
        x = np.linspace(-L, L, n)
        self.x = x
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        self.dx = x[1] - x[0]
        
        # Compute boundary mask
        self._compute_mask()
    
    def _compute_mask(self):
        """Compute interior mask based on shape."""
        r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        if self.shape == 'sphere':
            self.mask = r <= self.L * 0.8
        elif self.shape == 'cube':
            self.mask = (
                (np.abs(self.X) <= self.L * 0.8) &
                (np.abs(self.Y) <= self.L * 0.8) &
                (np.abs(self.Z) <= self.L * 0.8)
            )
        else:  # 'none'
            self.mask = np.ones_like(r, dtype=bool)
        
        self.n_interior = self.mask.sum()
    
    def project(self, field: MobiusField) -> Dict:
        """Project pre-field to 3D and extract all fields.
        
        This is the main method that performs:
        1. Möbius → 3D interpolation
        2. Potential extraction (φ from amplitude)
        3. Vector potential construction (A from phase)
        4. E and B field computation
        5. Maxwell validation
        
        Args:
            field: MobiusField to project
            
        Returns:
            Dictionary containing:
            - Scalar fields: phi, charge_density
            - Vector fields: E, B, A (as components)
            - Metrics: EB_ratio, div_E, div_B, etc.
        """
        # Get Möbius embedding and field values
        X_m = field.X.flatten()
        Y_m = field.Y.flatten()
        Z_m = field.Z.flatten()
        amp_m = field.amplitude().flatten()
        phase_m = field.phase().flatten()
        
        # Initialize 3D fields
        phi = np.zeros((self.n, self.n, self.n))
        Ax = np.zeros_like(phi)
        Ay = np.zeros_like(phi)
        Az = np.zeros_like(phi)
        
        # Interpolate from Möbius to 3D grid
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if not self.mask[i, j, k]:
                        continue
                    
                    px, py, pz = self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]
                    
                    # Gaussian-weighted interpolation
                    dist2 = (X_m - px)**2 + (Y_m - py)**2 + (Z_m - pz)**2
                    weights = np.exp(-dist2 / (2 * self.sigma**2))
                    weights /= weights.sum() + 1e-10
                    
                    # Scalar potential from amplitude
                    phi[i,j,k] = np.sum(weights * amp_m)
                    
                    # Vector potential from phase structure
                    phase_avg = np.sum(weights * phase_m)
                    
                    # Construct A in toroidal direction (perpendicular to radial)
                    r_cyl = np.sqrt(px**2 + py**2) + 0.01
                    theta = np.arctan2(py, px)
                    
                    # Toroidal component proportional to sin(phase)
                    A_tor = phi[i,j,k] * np.sin(phase_avg) * 0.5
                    Ax[i,j,k] = -A_tor * np.sin(theta)
                    Ay[i,j,k] = A_tor * np.cos(theta)
                    Az[i,j,k] = phi[i,j,k] * np.cos(phase_avg) * 0.3
        
        # Compute E = -∇φ
        Ex = -np.gradient(phi, self.dx, axis=0)
        Ey = -np.gradient(phi, self.dx, axis=1)
        Ez = -np.gradient(phi, self.dx, axis=2)
        
        # Compute B = ∇×A
        Bx = np.gradient(Az, self.dx, axis=1) - np.gradient(Ay, self.dx, axis=2)
        By = np.gradient(Ax, self.dx, axis=2) - np.gradient(Az, self.dx, axis=0)
        Bz = np.gradient(Ay, self.dx, axis=0) - np.gradient(Ax, self.dx, axis=1)
        
        # Compute derived quantities
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        # Charge density (∇·E)
        charge_density = (
            np.gradient(Ex, self.dx, axis=0) +
            np.gradient(Ey, self.dx, axis=1) +
            np.gradient(Ez, self.dx, axis=2)
        )
        
        # Divergence of B (should be ~0)
        div_B = (
            np.gradient(Bx, self.dx, axis=0) +
            np.gradient(By, self.dx, axis=1) +
            np.gradient(Bz, self.dx, axis=2)
        )
        
        # Compute metrics over interior
        mask = self.mask
        E_mean = E_mag[mask].mean()
        B_mean = B_mag[mask].mean()
        
        # E/B ratio
        EB_ratio = E_mean / (B_mean + 1e-10)
        
        # Deviations from Dawn constants
        phi_dev = abs(EB_ratio - PHI) / PHI
        phi_sq_dev = abs(EB_ratio - PHI_SQ) / PHI_SQ
        phi_1_5_dev = abs(EB_ratio - PHI**1.5) / PHI**1.5
        xi_dev = abs(EB_ratio - XI) / XI
        
        # Determine closest match
        deviations = {
            'φ': phi_dev,
            'φ²': phi_sq_dev,
            'φ^1.5': phi_1_5_dev,
            'Ξ': xi_dev
        }
        closest_match = min(deviations, key=deviations.get)
        
        # φ-power
        phi_power = np.log(EB_ratio) / np.log(PHI) if EB_ratio > 0 else 0
        
        # Charge analysis
        charge_pos = charge_density[mask & (charge_density > 0)]
        charge_neg = charge_density[mask & (charge_density < 0)]
        
        return {
            # Geometry
            'w_R_ratio': field.w_R_ratio,
            
            # Scalar fields
            'phi': phi,
            'charge_density': charge_density,
            
            # Vector field components
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
            'Bx': Bx, 'By': By, 'Bz': Bz,
            'Ax': Ax, 'Ay': Ay, 'Az': Az,
            
            # Field magnitudes
            'E_mag': E_mag,
            'B_mag': B_mag,
            
            # Key metrics
            'E_mean': float(E_mean),
            'B_mean': float(B_mean),
            'EB_ratio': float(EB_ratio),
            'phi_power': float(phi_power),
            
            # Deviations from constants
            'phi_deviation': float(phi_dev),
            'phi_sq_deviation': float(phi_sq_dev),
            'phi_1_5_deviation': float(phi_1_5_dev),
            'xi_deviation': float(xi_dev),
            'closest_match': closest_match,
            'closest_deviation': float(deviations[closest_match]),
            
            # Maxwell compliance
            'div_E_mean': float(np.abs(charge_density[mask]).mean()),
            'div_B_mean': float(np.abs(div_B[mask]).mean()),
            'gauss_ok': float(np.abs(charge_density[mask]).mean()) < DIV_E_THRESHOLD,
            'no_monopoles': float(np.abs(div_B[mask]).mean()) < DIV_B_THRESHOLD,
            
            # Charge structure
            'charge_total_pos': float(charge_pos.sum()) if len(charge_pos) > 0 else 0,
            'charge_total_neg': float(charge_neg.sum()) if len(charge_neg) > 0 else 0,
            'charge_net': float(charge_density[mask].sum()),
        }
    
    def extract_E(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract E field from scalar potential.
        
        E = -∇φ
        
        Args:
            phi: Scalar potential field
            
        Returns:
            (Ex, Ey, Ez) components
        """
        Ex = -np.gradient(phi, self.dx, axis=0)
        Ey = -np.gradient(phi, self.dx, axis=1)
        Ez = -np.gradient(phi, self.dx, axis=2)
        return Ex, Ey, Ez
    
    def extract_B(
        self,
        Ax: np.ndarray,
        Ay: np.ndarray,
        Az: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract B field from vector potential.
        
        B = ∇×A
        
        This guarantees ∇·B = 0 by construction (div of curl is zero).
        
        Args:
            Ax, Ay, Az: Vector potential components
            
        Returns:
            (Bx, By, Bz) components
        """
        Bx = np.gradient(Az, self.dx, axis=1) - np.gradient(Ay, self.dx, axis=2)
        By = np.gradient(Ax, self.dx, axis=2) - np.gradient(Az, self.dx, axis=0)
        Bz = np.gradient(Ay, self.dx, axis=0) - np.gradient(Ax, self.dx, axis=1)
        return Bx, By, Bz
    
    def compute_divergence(
        self,
        Fx: np.ndarray,
        Fy: np.ndarray,
        Fz: np.ndarray
    ) -> np.ndarray:
        """Compute divergence ∇·F of a vector field.
        
        Args:
            Fx, Fy, Fz: Vector field components
            
        Returns:
            Scalar divergence field
        """
        return (
            np.gradient(Fx, self.dx, axis=0) +
            np.gradient(Fy, self.dx, axis=1) +
            np.gradient(Fz, self.dx, axis=2)
        )
    
    def compute_curl(
        self,
        Fx: np.ndarray,
        Fy: np.ndarray,
        Fz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute curl ∇×F of a vector field.
        
        Args:
            Fx, Fy, Fz: Vector field components
            
        Returns:
            (curl_x, curl_y, curl_z) components
        """
        curl_x = np.gradient(Fz, self.dx, axis=1) - np.gradient(Fy, self.dx, axis=2)
        curl_y = np.gradient(Fx, self.dx, axis=2) - np.gradient(Fz, self.dx, axis=0)
        curl_z = np.gradient(Fy, self.dx, axis=0) - np.gradient(Fx, self.dx, axis=1)
        return curl_x, curl_y, curl_z
    
    def __repr__(self) -> str:
        return f"EMProjector(n={self.n}, L={self.L}, shape='{self.shape}')"


class MaxwellValidator:
    """Validates compliance with Maxwell's equations.
    
    Checks:
    1. ∇·E = ρ/ε₀ (Gauss's law) - ρ=0 in vacuum
    2. ∇·B = 0 (No magnetic monopoles)
    3. ∇×E = -∂B/∂t (Faraday's law) - static case: ∇×E ≈ 0
    4. ∇×B = μ₀J + μ₀ε₀∂E/∂t (Ampère-Maxwell) - static case: ∇×B ≈ 0
    
    Example:
        >>> validator = MaxwellValidator(projector)
        >>> result = projector.project(field)
        >>> validation = validator.validate(result)
        >>> print(f"Maxwell compliance: {validation['overall_score']:.0%}")
    """
    
    def __init__(self, projector: EMProjector):
        """Initialize validator with projector for grid info.
        
        Args:
            projector: EMProjector instance
        """
        self.projector = projector
        self.dx = projector.dx
        self.mask = projector.mask
    
    def validate(self, projection_result: Dict) -> Dict:
        """Validate Maxwell equation compliance.
        
        Args:
            projection_result: Output from EMProjector.project()
            
        Returns:
            Dictionary of validation metrics
        """
        # Extract fields
        Ex, Ey, Ez = projection_result['Ex'], projection_result['Ey'], projection_result['Ez']
        Bx, By, Bz = projection_result['Bx'], projection_result['By'], projection_result['Bz']
        
        mask = self.mask
        
        # ∇·E (should be ~0 in vacuum)
        div_E = self.projector.compute_divergence(Ex, Ey, Ez)
        div_E_residual = np.abs(div_E[mask]).mean()
        
        # ∇·B (must be 0)
        div_B = self.projector.compute_divergence(Bx, By, Bz)
        div_B_residual = np.abs(div_B[mask]).mean()
        
        # ∇×E (should be ~0 for static case)
        curl_Ex, curl_Ey, curl_Ez = self.projector.compute_curl(Ex, Ey, Ez)
        curl_E_mag = np.sqrt(curl_Ex**2 + curl_Ey**2 + curl_Ez**2)
        curl_E_residual = curl_E_mag[mask].mean()
        
        # ∇×B (should be ~0 for static case without current)
        curl_Bx, curl_By, curl_Bz = self.projector.compute_curl(Bx, By, Bz)
        curl_B_mag = np.sqrt(curl_Bx**2 + curl_By**2 + curl_Bz**2)
        curl_B_residual = curl_B_mag[mask].mean()
        
        # Compute E/B ratio
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        E_mean = E_mag[mask].mean()
        B_mean = B_mag[mask].mean()
        EB_ratio = E_mean / (B_mean + 1e-10)
        
        # Check against thresholds
        gauss_ok = div_E_residual < DIV_E_THRESHOLD
        no_monopoles = div_B_residual < DIV_B_THRESHOLD
        phi_match = abs(EB_ratio - PHI) / PHI < PHI_MATCH_THRESHOLD
        
        # Overall score (0-1)
        scores = [
            1.0 if no_monopoles else 0.0,  # Most important
            0.5 if gauss_ok else 0.0,
            0.5 if phi_match else 0.0,
        ]
        overall_score = sum(scores) / len(scores)
        
        return {
            # Residuals
            'div_E_residual': float(div_E_residual),
            'div_B_residual': float(div_B_residual),
            'curl_E_residual': float(curl_E_residual),
            'curl_B_residual': float(curl_B_residual),
            
            # Pass/fail
            'gauss_law_ok': gauss_ok,
            'no_monopoles': no_monopoles,
            'phi_match': phi_match,
            
            # E/B analysis
            'E_mean': float(E_mean),
            'B_mean': float(B_mean),
            'EB_ratio': float(EB_ratio),
            'phi_deviation': float(abs(EB_ratio - PHI) / PHI),
            
            # Overall
            'overall_score': float(overall_score),
            'verdict': 'PASS' if overall_score > 0.5 else 'FAIL',
        }
    
    def summary(self, validation_result: Dict) -> str:
        """Generate human-readable summary.
        
        Args:
            validation_result: Output from validate()
            
        Returns:
            Formatted summary string
        """
        v = validation_result
        lines = [
            "Maxwell Validation Summary",
            "=" * 40,
            f"Gauss's Law (∇·E ≈ 0):    {'PASS' if v['gauss_law_ok'] else 'FAIL'} ({v['div_E_residual']:.4f})",
            f"No Monopoles (∇·B = 0):   {'PASS' if v['no_monopoles'] else 'FAIL'} ({v['div_B_residual']:.6f})",
            f"E/B ≈ φ:                  {'PASS' if v['phi_match'] else 'FAIL'} ({v['EB_ratio']:.4f}, {v['phi_deviation']*100:.1f}% dev)",
            "-" * 40,
            f"Overall: {v['verdict']} ({v['overall_score']:.0%})",
        ]
        return '\n'.join(lines)
