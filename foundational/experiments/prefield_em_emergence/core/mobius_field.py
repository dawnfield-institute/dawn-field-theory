"""
Möbius Pre-Field Implementation
===============================

Implements the pre-field state Ψ on a Möbius manifold M.

The Möbius strip is parameterized by (u, v) where:
- u ∈ [0, 2π]: angle around the strip
- v ∈ [-w, w]: distance from centerline

The half-twist topology means: ψ(u + 2π, v) = ψ(u, -v)

Key Insight:
    A Möbius strip CANNOT be embedded in 2D without self-intersection.
    It requires minimum 3D for a proper embedding. This is why projecting
    pre-field dynamics to 3D is not arbitrary - it's mathematically necessary.

Standard Embedding:
    X = (R + v·cos(u/2))·cos(u)
    Y = (R + v·cos(u/2))·sin(u)
    Z = v·sin(u/2)
"""

import numpy as np
from typing import Tuple, Optional
from .constants import PHI, PI_FREQ


class MobiusField:
    """Pre-field state on a Möbius manifold.
    
    The pre-field is a complex-valued function Ψ: M → ℂ initialized with
    π-harmonic structure that promotes stable attractor formation during
    SEC evolution.
    
    Attributes:
        n_u: Angular resolution (must be even for proper half-twist)
        n_v: Radial resolution
        R: Major radius (how big the strip circles)
        w: Half-width (how wide the strip is)
        w_R_ratio: Width-to-radius ratio (key geometric parameter)
        psi: Complex field values Ψ(u, v)
        X, Y, Z: 3D embedding coordinates
        
    Example:
        >>> field = MobiusField(n_u=64, n_v=32, R=2.0, w=0.6)
        >>> field.w_R_ratio
        0.3
        >>> field.potential().shape
        (64, 32)
    """
    
    def __init__(
        self,
        n_u: int = 64,
        n_v: int = 32,
        R: float = 2.0,
        w: float = 0.5,
        twist: float = 1.0,
        seed: Optional[int] = None
    ):
        """Initialize Möbius pre-field.
        
        Args:
            n_u: Angular resolution (should be even)
            n_v: Radial resolution
            R: Major radius
            w: Half-width of strip
            twist: Twist multiplier (1.0 = standard half-twist)
            seed: Random seed for reproducibility
        """
        if n_u % 2 != 0:
            raise ValueError("n_u must be even for proper Möbius topology")
        
        self.n_u = n_u
        self.n_v = n_v
        self.R = R
        self.w = w
        self.twist = twist
        self.w_R_ratio = w / R
        
        if seed is not None:
            np.random.seed(seed)
        
        # Coordinate grids
        self.u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        self.v = np.linspace(-w, w, n_v)
        self.U, self.V = np.meshgrid(self.u, self.v, indexing='ij')
        
        # Grid spacing
        self.du = self.u[1] - self.u[0]
        self.dv = self.v[1] - self.v[0]
        
        # Initialize field with π-harmonic structure
        self._initialize_field()
        
        # Compute 3D embedding
        self._compute_embedding()
    
    def _initialize_field(self):
        """Initialize Ψ with π-harmonic phase structure.
        
        The initialization uses:
        - Phase: π × sin(u) × cos(πv/w) for angular coherence
        - Amplitude: 1 + modulation for non-trivial structure
        
        This structure promotes stable attractor formation during SEC evolution.
        """
        # π-harmonic phase (key for stability)
        phase = np.pi * np.sin(self.U) * np.cos(np.pi * self.V / self.w)
        
        # Amplitude with radial modulation
        amplitude = 1.0 + 0.3 * np.cos(2 * self.U) * np.exp(-self.V**2 / (0.3**2))
        
        # Small noise for symmetry breaking
        noise = 0.01 * np.random.randn(self.n_u, self.n_v)
        amplitude += noise
        amplitude = np.clip(amplitude, 0.1, 5.0)
        
        # Construct complex field
        self.psi = amplitude * np.exp(1j * phase)
    
    def _compute_embedding(self):
        """Compute the standard Möbius embedding in R³.
        
        Parameterization:
            X = (R + v·cos(twist·u/2))·cos(u)
            Y = (R + v·cos(twist·u/2))·sin(u)
            Z = v·sin(twist·u/2)
        
        The twist parameter allows for non-standard topologies:
            twist = 1.0: Standard half-twist (Möbius strip)
            twist = 2.0: Full twist (orientable cylinder with twist)
            twist = 0.5: Quarter-twist (partial Möbius)
        """
        twist_angle = self.twist * self.U / 2
        
        self.X = (self.R + self.V * np.cos(twist_angle)) * np.cos(self.U)
        self.Y = (self.R + self.V * np.cos(twist_angle)) * np.sin(self.U)
        self.Z = self.V * np.sin(twist_angle)
    
    # =========================================================================
    # Field Properties
    # =========================================================================
    
    def potential(self) -> np.ndarray:
        """Get potential field P = |ψ|².
        
        The potential represents the "energy density" or "information density"
        of the pre-field at each point.
        
        Returns:
            2D array of potential values
        """
        return np.abs(self.psi) ** 2
    
    def phase(self) -> np.ndarray:
        """Get phase field φ = arg(ψ).
        
        The phase encodes the "direction" of the field, which becomes
        the vector potential direction after 3D projection.
        
        Returns:
            2D array of phase values in [-π, π]
        """
        return np.angle(self.psi)
    
    def amplitude(self) -> np.ndarray:
        """Get amplitude field A = |ψ|.
        
        Returns:
            2D array of amplitude values
        """
        return np.abs(self.psi)
    
    # =========================================================================
    # Gradient and Differential Operators
    # =========================================================================
    
    def gradient_potential(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of potential ∇P in (u, v) coordinates.
        
        Returns:
            (dP_du, dP_dv): Gradient components
        """
        P = self.potential()
        dP_du = np.gradient(P, self.du, axis=0)
        dP_dv = np.gradient(P, self.dv, axis=1)
        return dP_du, dP_dv
    
    def gradient_phase(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of phase ∇φ in (u, v) coordinates.
        
        Uses unwrapping to handle phase discontinuities.
        
        Returns:
            (dφ_du, dφ_dv): Gradient components
        """
        ph = self.phase()
        # Unwrap to handle 2π discontinuities
        ph_unwrap_u = np.unwrap(ph, axis=0)
        ph_unwrap_v = np.unwrap(ph, axis=1)
        
        dph_du = np.gradient(ph_unwrap_u, self.du, axis=0)
        dph_dv = np.gradient(ph_unwrap_v, self.dv, axis=1)
        
        return dph_du, dph_dv
    
    def gradient_magnitude(self) -> np.ndarray:
        """Compute |∇P| - magnitude of potential gradient.
        
        High values indicate rapid spatial variation.
        
        Returns:
            2D array of gradient magnitudes
        """
        dP_du, dP_dv = self.gradient_potential()
        return np.sqrt(dP_du**2 + dP_dv**2)
    
    def phase_gradient_magnitude(self) -> np.ndarray:
        """Compute |∇φ| - magnitude of phase gradient.
        
        High values indicate potential phase singularities (topological defects).
        
        Returns:
            2D array of phase gradient magnitudes
        """
        dph_du, dph_dv = self.gradient_phase()
        return np.sqrt(dph_du**2 + dph_dv**2)
    
    def laplacian_potential(self) -> np.ndarray:
        """Compute Laplacian ∇²P of potential.
        
        Used in SEC diffusion term.
        
        Returns:
            2D array of Laplacian values
        """
        P = self.potential()
        d2P_du2 = np.gradient(np.gradient(P, self.du, axis=0), self.du, axis=0)
        d2P_dv2 = np.gradient(np.gradient(P, self.dv, axis=1), self.dv, axis=1)
        return d2P_du2 + d2P_dv2
    
    # =========================================================================
    # Entropy and Information Measures
    # =========================================================================
    
    def entropy(self) -> np.ndarray:
        """Compute local Shannon entropy H = -P·log(P).
        
        Normalized so that P represents probability distribution.
        
        Returns:
            2D array of local entropy values
        """
        P = self.potential()
        P_norm = P / (P.sum() + 1e-10)
        P_norm = np.clip(P_norm, 1e-15, 1.0)
        return -P_norm * np.log(P_norm)
    
    def total_entropy(self) -> float:
        """Compute total entropy of the field.
        
        Returns:
            Scalar total entropy
        """
        return float(self.entropy().sum())
    
    def information(self) -> np.ndarray:
        """Compute local information I = P (potential as information density).
        
        Returns:
            2D array (same as potential)
        """
        return self.potential()
    
    # =========================================================================
    # PAC Conservation Metrics
    # =========================================================================
    
    def pac_residual(self) -> float:
        """Compute PAC (Potential-Actualization Conservation) residual.
        
        PAC principle: f(Parent) = Σ f(Children)
        
        Here we use spectral decomposition as "actualization":
        - Potential = spatial sum of |ψ|²
        - Actualization = spectral energy
        
        The residual measures how far from conservation:
            residual = |P_total - A_total| / (P_total + A_total)
        
        Returns:
            PAC residual (lower is more conserved)
        """
        P = self.potential()
        P_total = P.sum()
        
        # Spectral actualization
        fft_P = np.fft.fft2(P)
        spectral_energy = np.abs(fft_P) ** 2
        A_total = spectral_energy.sum() / P.size
        
        # Normalized residual
        residual = np.abs(P_total - A_total) / (P_total + A_total + 1e-10)
        return float(residual)
    
    def pac_balance(self) -> float:
        """Compute PAC balance ratio P/A.
        
        Should approach 1 at equilibrium.
        
        Returns:
            Balance ratio
        """
        P = self.potential()
        P_total = P.sum()
        
        fft_P = np.fft.fft2(P)
        A_total = np.abs(fft_P[0, 0]) ** 2 / P.size
        
        return float(P_total / (A_total + 1e-10))
    
    # =========================================================================
    # Phase Singularity Detection
    # =========================================================================
    
    def find_singularities(self, threshold: float = 2.0) -> np.ndarray:
        """Find locations of potential phase singularities.
        
        Phase singularities are topological defects where the phase is
        undefined (all phases meet at one point). They manifest as
        very high phase gradient magnitudes.
        
        Args:
            threshold: Multiple of mean gradient to consider singular
            
        Returns:
            Boolean mask where True indicates potential singularity
        """
        grad_mag = self.phase_gradient_magnitude()
        return grad_mag > threshold * grad_mag.mean()
    
    def singularity_count(self, threshold: float = 2.0) -> int:
        """Count number of potential phase singularities.
        
        Returns:
            Number of grid points with high phase gradient
        """
        return int(self.find_singularities(threshold).sum())
    
    # =========================================================================
    # State Modification
    # =========================================================================
    
    def set_psi(self, psi: np.ndarray):
        """Set the field state directly.
        
        Args:
            psi: New complex field values (must match shape)
        """
        if psi.shape != (self.n_u, self.n_v):
            raise ValueError(f"Shape mismatch: expected {(self.n_u, self.n_v)}, got {psi.shape}")
        self.psi = psi.astype(complex)
    
    def normalize(self, target_total: Optional[float] = None):
        """Normalize field to preserve total potential.
        
        Args:
            target_total: Target total potential (default: preserve current)
        """
        current = self.potential().sum()
        if target_total is None:
            target_total = current
        
        if current > 1e-10:
            scale = np.sqrt(target_total / current)
            self.psi *= scale
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"MobiusField(n_u={self.n_u}, n_v={self.n_v}, "
            f"R={self.R}, w={self.w}, w/R={self.w_R_ratio:.3f})"
        )
    
    def summary(self) -> dict:
        """Get summary statistics of the field.
        
        Returns:
            Dictionary of key metrics
        """
        return {
            'geometry': {
                'n_u': self.n_u,
                'n_v': self.n_v,
                'R': self.R,
                'w': self.w,
                'w_R_ratio': self.w_R_ratio,
                'twist': self.twist,
            },
            'field': {
                'potential_total': float(self.potential().sum()),
                'potential_mean': float(self.potential().mean()),
                'potential_std': float(self.potential().std()),
                'amplitude_range': [float(self.amplitude().min()), 
                                   float(self.amplitude().max())],
            },
            'entropy': {
                'total': self.total_entropy(),
                'mean_local': float(self.entropy().mean()),
            },
            'pac': {
                'residual': self.pac_residual(),
                'balance': self.pac_balance(),
            },
            'topology': {
                'singularity_count': self.singularity_count(),
                'phase_gradient_max': float(self.phase_gradient_magnitude().max()),
            },
        }
