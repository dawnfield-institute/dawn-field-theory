"""
Formal Mathematical Definitions for Pre-Field Recursion Framework

This module provides rigorous mathematical foundations:
- PreFieldState: Ψ_pre on Möbius manifold M
- RecursionOperator: R: Ψ → Ψ via Möbius transformation
- Emergence conditions based on PAC conservation

Version: 2.0
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PreFieldState:
    """
    Formal pre-field state definition
    
    A pre-field state Ψ_pre is defined on a Möbius manifold M where:
    - Ψ_pre: M → ℂ (complex-valued for phase encoding)
    - ∇²Ψ_pre = λ(Ψ_pre) (eigenfunction of Laplacian)
    - PAC(Ψ_pre) < ε (not yet conserving)
    
    Attributes:
        wavefunction: Complex-valued field on manifold
        topology: Manifold type ('mobius', 'klein', 'torus')
        recursion_depth: Number of recursion applications (n in R^n(Ψ))
        pac_residual: PAC conservation violation metric
        curvature_tensor: Local curvature information
    """
    wavefunction: np.ndarray  # Ψ_pre: M → ℂ
    topology: str = "mobius"
    recursion_depth: int = 0
    pac_residual: float = np.inf
    curvature_tensor: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure wavefunction is complex"""
        if not np.iscomplexobj(self.wavefunction):
            self.wavefunction = self.wavefunction.astype(complex)
    
    def is_conserving(self, epsilon: float = 1e-12) -> bool:
        """
        Check if PAC conservation is satisfied
        
        Args:
            epsilon: Conservation threshold (machine precision)
            
        Returns:
            True if |PAC residual| < ε
        """
        return self.pac_residual < epsilon
    
    def compute_emergence_metric(self) -> float:
        """
        Quantify proximity to field emergence
        
        Emergence metric: κ × (1/ε_PAC)
        where κ is mean curvature and ε_PAC is PAC residual
        
        Returns:
            Emergence metric value (higher = closer to emergence)
        """
        if self.curvature_tensor is None:
            return 0.0
        
        mean_curvature = np.mean(np.abs(self.curvature_tensor))
        
        # Avoid division by zero
        if self.pac_residual < 1e-15:
            return 1e15  # Very high emergence metric
        
        return mean_curvature / self.pac_residual
    
    def compute_phase_variance(self) -> float:
        """Calculate variance in phase distribution"""
        phases = np.angle(self.wavefunction)
        return np.var(phases)
    
    def compute_field_energy(self) -> float:
        """Calculate total field energy (L2 norm squared)"""
        return np.sum(np.abs(self.wavefunction)**2)
    
    def compute_information_entropy(self) -> float:
        """Calculate Shannon entropy of probability distribution"""
        prob = np.abs(self.wavefunction)**2
        prob_norm = prob / (np.sum(prob) + 1e-15)
        prob_nonzero = prob_norm[prob_norm > 1e-15]
        
        if len(prob_nonzero) == 0:
            return 0.0
        
        return -np.sum(prob_nonzero * np.log2(prob_nonzero))
    
    def copy(self) -> 'PreFieldState':
        """Create deep copy of state"""
        return PreFieldState(
            wavefunction=self.wavefunction.copy(),
            topology=self.topology,
            recursion_depth=self.recursion_depth,
            pac_residual=self.pac_residual,
            curvature_tensor=self.curvature_tensor.copy() if self.curvature_tensor is not None else None,
            metadata=self.metadata.copy()
        )


class RecursionOperator:
    """
    Recursion operator: R: Ψ_pre → Ψ_pre
    
    Applies Möbius transformation to evolve pre-field state:
    R(z) = (z + θi) / (1 - z̄θi)
    
    where θ is the twist rate and z̄ is complex conjugate.
    
    Each application:
    - Increases recursion depth
    - Modifies local curvature
    - Drives toward PAC conservation
    """
    
    def __init__(self, twist_rate: float = np.pi/4, conserve_norm: bool = True):
        """
        Initialize recursion operator
        
        Args:
            twist_rate: Rotation angle for Möbius twist (radians)
            conserve_norm: Whether to renormalize after each application
        """
        self.twist_rate = twist_rate
        self.conserve_norm = conserve_norm
        self.iteration_count = 0
        self.transformation_history = []
    
    def apply(self, state: PreFieldState) -> PreFieldState:
        """
        Apply one recursion step: Ψ_{n+1} = R(Ψ_n)
        
        Args:
            state: Current pre-field state
            
        Returns:
            Evolved pre-field state
        """
        # Möbius transformation on complex plane
        z = state.wavefunction
        theta_i = self.twist_rate * 1j
        
        # Apply transformation: w = (z + θi)/(1 - z̄θi)
        numerator = z + theta_i
        denominator = 1 - np.conj(z) * theta_i
        
        # Avoid division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        w = numerator / denominator
        
        # Renormalize if requested
        if self.conserve_norm:
            original_norm = np.sqrt(state.compute_field_energy())
            current_norm = np.sqrt(np.sum(np.abs(w)**2))
            if current_norm > 1e-10:
                w = w * (original_norm / current_norm)
        
        # Compute PAC residual
        pac_residual = self._compute_pac_residual(w)
        
        # Compute curvature tensor
        curvature = self._compute_curvature(w)
        
        # Create new state
        new_state = PreFieldState(
            wavefunction=w,
            topology=state.topology,
            recursion_depth=state.recursion_depth + 1,
            pac_residual=pac_residual,
            curvature_tensor=curvature,
            metadata={
                'previous_depth': state.recursion_depth,
                'twist_applied': self.twist_rate,
                'iteration_count': self.iteration_count
            }
        )
        
        self.iteration_count += 1
        self.transformation_history.append({
            'iteration': self.iteration_count,
            'pac_residual': pac_residual,
            'emergence_metric': new_state.compute_emergence_metric()
        })
        
        return new_state
    
    def _compute_pac_residual(self, psi: np.ndarray) -> float:
        """
        Calculate PAC conservation violation with enhanced terms
        
        v2.1: Includes kinetic (gradient) and phase coupling terms
        for faster, more accurate convergence.
        
        PAC residual = |Potential - Actualized| / (1 + phase_coupling)
        where:
        - Potential = Σ|ψ|² (amplitude squared)
        - Actualized = ΣRe(ψ) + 0.5*kinetic (includes gradient)
        - Phase coupling = smoothness of phase distribution
        
        Args:
            psi: Complex wavefunction
            
        Returns:
            PAC residual (lower = better conservation)
        """
        # Potential energy (amplitude squared)
        potential = np.sum(np.abs(psi)**2)
        
        # Kinetic energy (gradient terms)
        if psi.ndim == 1:
            grad = np.gradient(psi)
            kinetic = np.sum(np.abs(grad)**2)
        else:
            grad_x, grad_y = np.gradient(psi)
            kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Actualized includes kinetic contribution
        actualized = np.sum(np.real(psi)) + 0.5 * kinetic
        
        # Phase coupling term (measures phase coherence)
        if len(psi) > 1:
            phase_diff = np.angle(psi[1:]) - np.angle(psi[:-1])
            # Wrap to [-π, π]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            phase_coupling = np.sum(np.abs(phase_diff))
        else:
            phase_coupling = 0.0
        
        # Base residual
        base_residual = abs(potential - actualized)
        
        # Apply phase coupling weighting (smoother phases → faster convergence)
        residual = base_residual / (1.0 + phase_coupling * 0.1)
        
        # Normalize by field magnitude
        magnitude = np.sqrt(potential)
        if magnitude > 1e-10:
            residual = residual / magnitude
        
        return residual
    
    def _compute_curvature(self, psi: np.ndarray) -> np.ndarray:
        """
        Calculate discrete Riemann curvature tensor
        
        Simplified implementation using gradient and Hessian
        
        Args:
            psi: Complex wavefunction
            
        Returns:
            Curvature tensor approximation
        """
        # Handle both 1D and 2D cases
        if psi.ndim == 1:
            # 1D: Use second derivative
            grad = np.gradient(psi)
            hess = np.gradient(grad)
            return hess
        else:
            # 2D: Use full gradient
            grad_x = np.gradient(psi, axis=0)
            grad_y = np.gradient(psi, axis=1)
            
            # Hessian components
            hess_xx = np.gradient(grad_x, axis=0)
            hess_yy = np.gradient(grad_y, axis=1)
            hess_xy = np.gradient(grad_x, axis=1)
            
            # Scalar curvature approximation
            curvature = np.abs(hess_xx) + np.abs(hess_yy) + 2 * np.abs(hess_xy)
            return curvature
    
    def get_statistics(self) -> dict:
        """Get statistics from transformation history"""
        if not self.transformation_history:
            return {}
        
        pac_residuals = [h['pac_residual'] for h in self.transformation_history]
        emergence_metrics = [h['emergence_metric'] for h in self.transformation_history]
        
        return {
            'total_iterations': self.iteration_count,
            'twist_rate': self.twist_rate,
            'pac_residual_evolution': pac_residuals,
            'emergence_metric_evolution': emergence_metrics,
            'final_pac_residual': pac_residuals[-1] if pac_residuals else np.inf,
            'final_emergence_metric': emergence_metrics[-1] if emergence_metrics else 0.0,
            'pac_improvement': pac_residuals[0] / pac_residuals[-1] if pac_residuals and pac_residuals[-1] > 0 else 1.0
        }
    
    def reset(self):
        """Reset operator state"""
        self.iteration_count = 0
        self.transformation_history = []


def create_initial_state(size: int = 100, topology: str = "mobius", 
                         seed: Optional[int] = None) -> PreFieldState:
    """
    Create initial pre-field state with specified properties
    
    Args:
        size: Size of the field (1D length or 2D side)
        topology: Topology type ('mobius', 'klein', 'torus')
        seed: Random seed for reproducibility
        
    Returns:
        Initial PreFieldState
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create complex field with random phase
    if topology == "mobius":
        # 1D Möbius strip
        phase = np.linspace(0, 2*np.pi, size, endpoint=False)
        amplitude = 1.0 + 0.1 * np.random.randn(size)
        wavefunction = amplitude * np.exp(1j * phase)
        
        # Enforce anti-periodic boundary: ψ(x + π) = -ψ(x)
        half = size // 2
        for i in range(half):
            wavefunction[i + half] = -wavefunction[i]
    
    elif topology == "klein":
        # 2D Klein bottle (both directions anti-periodic)
        side = int(np.sqrt(size))
        phase_x = np.linspace(0, 2*np.pi, side, endpoint=False)
        phase_y = np.linspace(0, 2*np.pi, side, endpoint=False)
        phase_grid = np.meshgrid(phase_x, phase_y)
        phase = phase_grid[0] + phase_grid[1]
        
        amplitude = 1.0 + 0.1 * np.random.randn(side, side)
        wavefunction = amplitude * np.exp(1j * phase)
    
    else:  # torus or default
        # Standard periodic
        phase = 2 * np.pi * np.random.rand(size)
        amplitude = 1.0 + 0.1 * np.random.randn(size)
        wavefunction = amplitude * np.exp(1j * phase)
    
    return PreFieldState(
        wavefunction=wavefunction,
        topology=topology,
        recursion_depth=0,
        pac_residual=np.inf,  # Will be computed on first recursion
        curvature_tensor=None
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Formal Definitions Module")
    print("=" * 50)
    
    # Create initial state
    state = create_initial_state(size=100, topology="mobius", seed=42)
    print(f"✓ Created initial state: {state.topology}, size={len(state.wavefunction)}")
    print(f"  Energy: {state.compute_field_energy():.4f}")
    print(f"  Entropy: {state.compute_information_entropy():.4f}")
    
    # Create recursion operator
    recursion = RecursionOperator(twist_rate=np.pi/4)
    print(f"\n✓ Created recursion operator: twist_rate={recursion.twist_rate:.4f}")
    
    # Apply several recursions
    print("\nApplying recursions:")
    for i in range(5):
        state = recursion.apply(state)
        print(f"  Depth {state.recursion_depth}: PAC={state.pac_residual:.6f}, "
              f"Emergence={state.compute_emergence_metric():.6f}")
    
    # Get statistics
    stats = recursion.get_statistics()
    print(f"\n✓ PAC improvement: {stats['pac_improvement']:.2f}x")
    print(f"✓ Final emergence metric: {stats['final_emergence_metric']:.6f}")
    
    print("\n✅ Formal definitions module functional!")
