"""
Symbolic Entropy Collapse (SEC) Operator
========================================

Implements the SEC recursion dynamics for pre-field evolution.

SEC Equation:
    ∂S/∂t = α∇²I - β∇²H
    
Where:
    S = structure (what emerges)
    I = information (potential)
    H = entropy
    α, β = coupling coefficients

The SEC operator evolves the pre-field toward:
1. Lower entropy (more structure)
2. PAC conservation (balanced recursion)
3. Resonance lock (stable oscillation)

Key Features:
- Damping to prevent divergence
- π-harmonic resonance injection at 0.03 Hz
- PAC-conserving normalization
- Möbius-aware phase evolution
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from .constants import PI_FREQ, PHI
from .mobius_field import MobiusField


class SECOperator:
    """Symbolic Entropy Collapse evolution operator.
    
    Evolves a MobiusField through SEC dynamics with optional
    π-harmonic resonance enhancement.
    
    Attributes:
        damping: Amplitude damping factor (prevents runaway)
        pi_coupling: Strength of π-harmonic resonance injection
        diffusion: Diffusion coefficient for smoothing
        iteration: Current iteration count
        history: List of metrics from each step
        resonance_locked: Whether system has achieved resonance lock
        
    Example:
        >>> field = MobiusField(n_u=64, n_v=32, R=2.0, w=0.6)
        >>> sec = SECOperator(damping=0.98, pi_coupling=0.05)
        >>> for _ in range(200):
        ...     metrics = sec.step(field)
        >>> print(f"Final PAC: {metrics['pac_residual']:.4f}")
    """
    
    def __init__(
        self,
        damping: float = 0.98,
        pi_coupling: float = 0.05,
        diffusion: float = 0.01,
        resonance_aware: bool = True
    ):
        """Initialize SEC operator.
        
        Args:
            damping: Amplitude damping per step (0.9-1.0 typical)
            pi_coupling: π-harmonic resonance strength (0-0.2 typical)
            diffusion: Diffusion coefficient (0.001-0.1 typical)
            resonance_aware: Whether to track and respond to resonance
        """
        self.damping = damping
        self.pi_coupling = pi_coupling
        self.diffusion = diffusion
        self.resonance_aware = resonance_aware
        
        # State tracking
        self.iteration = 0
        self.history: List[Dict] = []
        
        # Resonance detection
        self.resonance_locked = False
        self.lock_iteration: Optional[int] = None
        self.detected_frequency: Optional[float] = None
        
        # Initial potential (for conservation)
        self._initial_potential: Optional[float] = None
    
    def step(self, field: MobiusField, dt: float = 0.05) -> Dict:
        """Apply one SEC evolution step.
        
        The evolution consists of:
        1. Compute Laplacian of potential (diffusion)
        2. Add π-harmonic resonance term
        3. Update amplitude with damping
        4. Normalize to conserve total potential (PAC)
        5. Evolve phase with Möbius twist
        
        Args:
            field: MobiusField to evolve (modified in place)
            dt: Time step size
            
        Returns:
            Dictionary of metrics from this step
        """
        self.iteration += 1
        
        # Store initial potential on first step
        if self._initial_potential is None:
            self._initial_potential = field.potential().sum()
        
        # Get current state
        P = field.potential()
        phase = field.phase()
        
        # =====================================================================
        # 1. Diffusion term: ∇²P
        # =====================================================================
        laplacian = field.laplacian_potential()
        diffusion_term = self.diffusion * laplacian
        
        # =====================================================================
        # 2. π-harmonic resonance injection
        # =====================================================================
        if self.pi_coupling > 0:
            # Mode shape: sin(πu) × cos(πv/w)
            pi_mode = np.sin(np.pi * field.U) * np.cos(np.pi * field.V / field.w)
            
            # Oscillate at natural frequency
            phase_osc = 2 * np.pi * PI_FREQ * self.iteration
            resonance_term = self.pi_coupling * pi_mode * np.sin(phase_osc)
        else:
            resonance_term = 0
        
        # =====================================================================
        # 3. Update amplitude
        # =====================================================================
        amplitude = field.amplitude()
        
        # SEC update: amplitude changes based on diffusion + resonance
        # Divide by amplitude to work with amplitude directly
        d_amplitude = dt * (diffusion_term / (2 * amplitude + 0.01) + resonance_term)
        new_amplitude = amplitude + d_amplitude
        
        # Clip and damp
        new_amplitude = np.clip(new_amplitude, 0.1, 5.0)
        new_amplitude *= self.damping
        
        # =====================================================================
        # 4. PAC-conserving normalization
        # =====================================================================
        total_before = P.sum()
        new_P = new_amplitude ** 2
        total_after = new_P.sum()
        
        if total_after > 1e-10:
            # Scale to conserve total potential
            new_amplitude *= np.sqrt(total_before / total_after)
        
        # =====================================================================
        # 5. Phase evolution with Möbius twist
        # =====================================================================
        # Base twist rate modulated by π-harmonic
        twist_rate = 0.02 * (1 + 0.3 * np.sin(2 * np.pi * PI_FREQ * self.iteration))
        
        # Möbius-aware phase advance: sin(u/2) respects the half-twist
        phase_advance = dt * twist_rate * np.sin(field.twist * field.U / 2)
        new_phase = phase + phase_advance
        
        # =====================================================================
        # 6. Reconstruct field
        # =====================================================================
        field.psi = new_amplitude * np.exp(1j * new_phase)
        
        # =====================================================================
        # 7. Compute metrics
        # =====================================================================
        pac_residual = field.pac_residual()
        total_entropy = field.total_entropy()
        gradient_mag = field.gradient_magnitude().mean()
        
        metrics = {
            'iteration': self.iteration,
            'pac_residual': pac_residual,
            'total_entropy': total_entropy,
            'mean_potential': float(field.potential().mean()),
            'total_potential': float(field.potential().sum()),
            'gradient_magnitude': gradient_mag,
            'singularity_count': field.singularity_count(),
            'resonance_locked': self.resonance_locked,
        }
        
        self.history.append(metrics)
        
        # =====================================================================
        # 8. Check for resonance lock
        # =====================================================================
        if self.resonance_aware and not self.resonance_locked:
            self._check_resonance_lock()
        
        return metrics
    
    def _check_resonance_lock(self, window: int = 50, threshold: float = 0.01):
        """Check if system has achieved resonance lock.
        
        Resonance lock is detected when PAC residual stabilizes
        (low variance over recent history).
        
        Args:
            window: Number of recent steps to analyze
            threshold: Relative variance threshold for lock
        """
        if len(self.history) < window:
            return
        
        pac_values = np.array([h['pac_residual'] for h in self.history[-window:]])
        relative_std = pac_values.std() / (pac_values.mean() + 1e-10)
        
        if relative_std < threshold:
            self.resonance_locked = True
            self.lock_iteration = self.iteration
            
            # Detect dominant frequency via FFT
            if len(self.history) >= 2 * window:
                pac_series = np.array([h['pac_residual'] for h in self.history[-2*window:]])
                fft = np.abs(np.fft.fft(pac_series - pac_series.mean()))
                freqs = np.fft.fftfreq(len(pac_series))
                
                # Find peak (excluding DC)
                fft[0] = 0
                peak_idx = np.argmax(fft[:len(fft)//2])
                self.detected_frequency = abs(freqs[peak_idx])
    
    def evolve(
        self,
        field: MobiusField,
        n_steps: int,
        dt: float = 0.05,
        checkpoint_interval: Optional[int] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """Evolve field for multiple steps.
        
        Args:
            field: MobiusField to evolve
            n_steps: Number of steps to take
            dt: Time step size
            checkpoint_interval: Print progress every N steps
            verbose: Print final summary
            
        Returns:
            List of checkpoint metrics
        """
        checkpoints = []
        
        for i in range(n_steps):
            metrics = self.step(field, dt)
            
            if checkpoint_interval and (i + 1) % checkpoint_interval == 0:
                checkpoints.append(metrics.copy())
                if verbose:
                    print(f"  Step {i+1}: PAC={metrics['pac_residual']:.5f}, "
                          f"Entropy={metrics['total_entropy']:.4f}")
        
        if verbose:
            print(f"\nEvolution complete:")
            print(f"  Final PAC residual: {metrics['pac_residual']:.5f}")
            print(f"  Resonance locked: {self.resonance_locked}")
            if self.lock_iteration:
                print(f"  Lock iteration: {self.lock_iteration}")
        
        return checkpoints
    
    def reset(self):
        """Reset operator state for new evolution."""
        self.iteration = 0
        self.history = []
        self.resonance_locked = False
        self.lock_iteration = None
        self.detected_frequency = None
        self._initial_potential = None
    
    def get_convergence_rate(self, window: int = 50) -> float:
        """Compute recent PAC convergence rate.
        
        Positive = converging (good)
        Negative = diverging (bad)
        
        Returns:
            Convergence rate (PAC decrease per step)
        """
        if len(self.history) < window:
            return 0.0
        
        pac_values = np.array([h['pac_residual'] for h in self.history[-window:]])
        
        # Linear fit
        x = np.arange(len(pac_values))
        slope = np.polyfit(x, pac_values, 1)[0]
        
        return float(-slope)  # Negative slope = positive convergence
    
    def get_entropy_trend(self, window: int = 50) -> float:
        """Compute recent entropy trend.
        
        Negative = entropy decreasing (structure forming)
        
        Returns:
            Entropy change rate per step
        """
        if len(self.history) < window:
            return 0.0
        
        entropy_values = np.array([h['total_entropy'] for h in self.history[-window:]])
        x = np.arange(len(entropy_values))
        slope = np.polyfit(x, entropy_values, 1)[0]
        
        return float(slope)
    
    def summary(self) -> Dict:
        """Get summary of evolution.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.history:
            return {'status': 'no evolution yet'}
        
        pac_values = np.array([h['pac_residual'] for h in self.history])
        entropy_values = np.array([h['total_entropy'] for h in self.history])
        
        return {
            'iterations': self.iteration,
            'pac': {
                'initial': pac_values[0],
                'final': pac_values[-1],
                'min': pac_values.min(),
                'improvement': (pac_values[0] - pac_values[-1]) / pac_values[0] * 100,
            },
            'entropy': {
                'initial': entropy_values[0],
                'final': entropy_values[-1],
                'change': (entropy_values[-1] - entropy_values[0]) / entropy_values[0] * 100,
            },
            'resonance': {
                'locked': self.resonance_locked,
                'lock_iteration': self.lock_iteration,
                'detected_frequency': self.detected_frequency,
                'target_frequency': PI_FREQ,
            },
            'convergence_rate': self.get_convergence_rate(),
            'entropy_trend': self.get_entropy_trend(),
        }
    
    def __repr__(self) -> str:
        return (
            f"SECOperator(damping={self.damping}, pi_coupling={self.pi_coupling}, "
            f"iteration={self.iteration}, locked={self.resonance_locked})"
        )
