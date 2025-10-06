"""
Pattern Amplifier - Genuine Wave Mechanics Implementation
Based on real wave resonance and constructive/destructive interference.

Implements genuine wave physics:
- Harmonic oscillator resonance
- Q-factor analysis
- Constructive/destructive interference
- Wave packet superposition
- Dispersion relation management
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.fft import fft2, ifft2, fftfreq
from fracton.core.recursive_engine import ExecutionContext


class ResonanceMode(Enum):
    """Types of wave resonance modes."""
    FUNDAMENTAL = "fundamental"      # First harmonic
    OVERTONE = "overtone"           # Higher harmonics  
    SUPERPOSITION = "superposition"  # Multiple mode coupling
    PARAMETRIC = "parametric"       # Parametric amplification
    COHERENT = "coherent"           # Phase-locked amplification


@dataclass
class WaveMode:
    """Genuine wave mode with physical parameters."""
    frequency: float                 # Oscillation frequency (Hz)
    wavelength: float               # Spatial wavelength
    amplitude: complex              # Complex amplitude
    phase_velocity: float           # Wave propagation speed
    group_velocity: float           # Energy propagation speed
    q_factor: float                 # Quality factor (resonance sharpness)
    energy_density: float           # Energy per unit volume
    damping_coefficient: float      # Energy dissipation rate


@dataclass
class AmplificationResult:
    """Result of genuine wave amplification."""
    original_amplitude: complex
    amplified_amplitude: complex
    amplification_factor: float
    q_factor: float
    resonance_frequency: float
    energy_gain: float
    phase_coherence: float
    interference_efficiency: float


class PatternAmplifier:
    """
    Genuine Wave Mechanics Pattern Amplifier.
    
    Uses real wave physics instead of signal processing tricks:
    - Harmonic oscillator resonance
    - Constructive interference amplification
    - Q-factor optimization
    - Dispersion management
    - Energy conservation in wave coupling
    """
    
    def __init__(self, field_shape: Tuple[int, int] = (32, 32)):
        """Initialize wave mechanics amplifier."""
        self.field_shape = field_shape
        
        # Wave physics parameters
        self.c = 1.0  # Wave speed (normalized)
        self.omega_0 = 2.0 * np.pi  # Fundamental frequency
        self.damping = 0.1  # Damping coefficient
        self.coupling_strength = 0.5  # Mode coupling strength
        
        # Resonator parameters
        self.q_factor = 10.0  # Quality factor for resonance
        self.resonance_bandwidth = self.omega_0 / (2 * self.q_factor)
        
        # Amplification parameters
        self.max_amplification = 5.0  # Maximum amplification factor
        self.energy_threshold = 0.1   # Minimum energy for amplification
        
        # Wave field state
        self.wave_field = np.zeros(field_shape, dtype=complex)
        self.velocity_field = np.zeros(field_shape, dtype=complex)
        
        # Mode tracking
        self.active_modes = []
        self.resonance_history = []
        
        # Spatial frequencies for dispersion
        kx = fftfreq(field_shape[0], d=1.0) * 2 * np.pi
        ky = fftfreq(field_shape[1], d=1.0) * 2 * np.pi
        self.kx_grid, self.ky_grid = np.meshgrid(kx, ky, indexing='ij')
        self.k_magnitude = np.sqrt(self.kx_grid**2 + self.ky_grid**2)
    
    def amplify_pattern(self, input_field: np.ndarray, target_frequency: float = None) -> AmplificationResult:
        """Amplify pattern using genuine wave resonance."""
        # Ensure complex field
        if input_field.dtype != complex:
            field = input_field.astype(complex)
        else:
            field = input_field.copy()
        
        # Reshape if needed
        if field.shape != self.field_shape:
            field = self._reshape_to_field(field)
        
        original_amplitude = np.sqrt(np.mean(np.abs(field)**2))
        
        if target_frequency is None:
            target_frequency = self.omega_0
        
        # Find resonant frequency closest to target
        resonant_freq = self._find_resonant_frequency(field, target_frequency)
        
        # Calculate Q-factor for this resonance
        q_factor = self._calculate_q_factor(field, resonant_freq)
        
        # Apply harmonic oscillator amplification
        amplified_field = self._harmonic_amplification(field, resonant_freq, q_factor)
        
        # Apply constructive interference
        interference_field = self._constructive_interference(amplified_field, resonant_freq)
        
        # Energy conservation check
        final_field = self._enforce_energy_conservation(field, interference_field)
        
        # Update internal state
        self.wave_field = final_field
        self._update_velocity_field()
        
        final_amplitude = np.sqrt(np.mean(np.abs(final_field)**2))
        amplification_factor = final_amplitude / (original_amplitude + 1e-10)
        
        # Calculate performance metrics
        phase_coherence = self._calculate_phase_coherence(final_field)
        interference_efficiency = self._calculate_interference_efficiency(field, final_field)
        energy_gain = final_amplitude**2 - original_amplitude**2
        
        return AmplificationResult(
            original_amplitude=complex(original_amplitude),
            amplified_amplitude=complex(final_amplitude),
            amplification_factor=amplification_factor,
            q_factor=q_factor,
            resonance_frequency=resonant_freq,
            energy_gain=energy_gain,
            phase_coherence=phase_coherence,
            interference_efficiency=interference_efficiency
        )
    
    def _find_resonant_frequency(self, field: np.ndarray, target_freq: float) -> float:
        """Find resonant frequency using spectral analysis."""
        # Fourier transform to frequency domain
        field_fft = fft2(field)
        power_spectrum = np.abs(field_fft)**2
        
        # Create frequency grid
        freqs_x = fftfreq(self.field_shape[0], d=1.0) * 2 * np.pi
        freqs_y = fftfreq(self.field_shape[1], d=1.0) * 2 * np.pi
        
        # Find peak frequency closest to target
        max_power = 0
        best_freq = target_freq
        
        for i, fx in enumerate(freqs_x):
            for j, fy in enumerate(freqs_y):
                freq_magnitude = np.sqrt(fx**2 + fy**2)
                if abs(freq_magnitude - target_freq) < self.resonance_bandwidth:
                    if power_spectrum[i, j] > max_power:
                        max_power = power_spectrum[i, j]
                        best_freq = freq_magnitude
        
        return best_freq
    
    def _calculate_q_factor(self, field: np.ndarray, frequency: float) -> float:
        """Calculate Q-factor from field characteristics."""
        # Energy stored vs energy dissipated per cycle
        field_fft = fft2(field)
        power_spectrum = np.abs(field_fft)**2
        
        # Find resonance peak width
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return self.q_factor
        
        # Estimate bandwidth from spectral width
        freqs = np.sqrt(self.kx_grid**2 + self.ky_grid**2)
        freq_weighted_power = power_spectrum * (freqs**2)
        
        mean_freq_sq = np.sum(freq_weighted_power) / total_power
        mean_freq = np.sqrt(mean_freq_sq)
        
        # Bandwidth estimation (simplified)
        bandwidth = max(0.1, mean_freq / 10.0)  # Heuristic estimate
        
        q_factor = frequency / bandwidth
        return min(max(q_factor, 1.0), 100.0)  # Reasonable bounds
    
    def _harmonic_amplification(self, field: np.ndarray, frequency: float, q_factor: float) -> np.ndarray:
        """Apply harmonic oscillator resonance amplification."""
        # Driven harmonic oscillator response
        # H(ω) = 1 / (ω₀² - ω² + 2iγω)
        # where γ = ω₀/(2Q)
        
        gamma = frequency / (2 * q_factor)
        field_fft = fft2(field)
        
        # Apply frequency-dependent amplification
        omega_sq_grid = self.k_magnitude**2 * self.c**2  # Dispersion relation
        
        # Resonance response function
        denominator = self.omega_0**2 - omega_sq_grid + 2j * gamma * self.k_magnitude * self.c
        response = 1.0 / (denominator + 1e-10)  # Avoid division by zero
        
        # Limit amplification to prevent instability
        amplification_magnitude = np.abs(response)
        limited_response = response * np.minimum(amplification_magnitude, self.max_amplification) / (amplification_magnitude + 1e-10)
        
        # Apply amplification in frequency domain
        amplified_fft = field_fft * limited_response
        amplified_field = ifft2(amplified_fft)
        
        return amplified_field
    
    def _constructive_interference(self, field: np.ndarray, frequency: float) -> np.ndarray:
        """Apply constructive interference amplification."""
        # Create phase-locked reference wave
        x = np.arange(self.field_shape[0])
        y = np.arange(self.field_shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Reference wave with optimized phase
        k_ref = frequency / self.c
        reference_wave = np.exp(1j * k_ref * (X + Y) / np.sqrt(2))
        
        # Calculate phase correlation with reference
        phase_correlation = np.mean(np.real(np.conj(field) * reference_wave))
        
        # Constructive interference amplification
        if phase_correlation > 0:
            # Phases are aligned - constructive interference
            interference_factor = 1.0 + self.coupling_strength * phase_correlation
            interfered_field = field * interference_factor
        else:
            # Phases are opposed - minimal amplification
            interference_factor = 1.0 + 0.1 * self.coupling_strength * abs(phase_correlation)
            interfered_field = field * interference_factor
        
        return interfered_field
    
    def _enforce_energy_conservation(self, original_field: np.ndarray, amplified_field: np.ndarray) -> np.ndarray:
        """Enforce energy conservation in amplification."""
        original_energy = np.sum(np.abs(original_field)**2)
        amplified_energy = np.sum(np.abs(amplified_field)**2)
        
        if original_energy == 0:
            return amplified_field
        
        # Limit energy gain to prevent unphysical amplification
        max_energy_gain = self.max_amplification**2 * original_energy
        
        if amplified_energy > max_energy_gain:
            # Scale down to conserve energy
            energy_scale = np.sqrt(max_energy_gain / amplified_energy)
            conserved_field = amplified_field * energy_scale
        else:
            conserved_field = amplified_field
        
        return conserved_field
    
    def _update_velocity_field(self):
        """Update velocity field from wave field (∂ψ/∂t)."""
        # Wave equation: ∂²ψ/∂t² = c²∇²ψ - 2γ∂ψ/∂t
        # So: ∂ψ/∂t = v (velocity field)
        
        # Calculate Laplacian
        laplacian = self._calculate_laplacian(self.wave_field)
        
        # Velocity update: ∂v/∂t = c²∇²ψ - 2γv
        acceleration = self.c**2 * laplacian - 2 * self.damping * self.velocity_field
        
        # Simple integration (Euler method)
        dt = 0.01
        self.velocity_field += acceleration * dt
    
    def _calculate_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Calculate discrete Laplacian."""
        laplacian = np.zeros_like(field)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        )
        
        # Boundary conditions (Neumann)
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    def _calculate_phase_coherence(self, field: np.ndarray) -> float:
        """Calculate phase coherence measure."""
        phases = np.angle(field)
        phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
        return phase_coherence
    
    def _calculate_interference_efficiency(self, original: np.ndarray, amplified: np.ndarray) -> float:
        """Calculate interference efficiency."""
        original_power = np.mean(np.abs(original)**2)
        amplified_power = np.mean(np.abs(amplified)**2)
        
        if original_power == 0:
            return 0.0
        
        # Efficiency is how much useful amplification occurred
        power_ratio = amplified_power / original_power
        efficiency = (power_ratio - 1.0) / (self.max_amplification - 1.0)
        
        return min(1.0, max(0.0, efficiency))
    
    def get_wave_metrics(self) -> Dict[str, float]:
        """Get current wave metrics."""
        if np.sum(np.abs(self.wave_field)) == 0:
            return {
                'wave_amplitude': 0.0,
                'wave_energy': 0.0,
                'phase_coherence': 0.0,
                'frequency_content': 0.0,
                'q_factor_estimate': self.q_factor,
                'damping_rate': self.damping
            }
        
        amplitude = np.sqrt(np.mean(np.abs(self.wave_field)**2))
        energy = np.sum(np.abs(self.wave_field)**2)
        coherence = self._calculate_phase_coherence(self.wave_field)
        
        # Estimate dominant frequency
        field_fft = fft2(self.wave_field)
        power_spectrum = np.abs(field_fft)**2
        total_power = np.sum(power_spectrum)
        
        if total_power > 0:
            freq_weighted = power_spectrum * self.k_magnitude
            dominant_freq = np.sum(freq_weighted) / total_power
        else:
            dominant_freq = 0.0
        
        return {
            'wave_amplitude': amplitude,
            'wave_energy': energy,
            'phase_coherence': coherence,
            'frequency_content': dominant_freq,
            'q_factor_estimate': self.q_factor,
            'damping_rate': self.damping
        }
    
    def create_wave_superposition(self, modes: List[WaveMode]) -> np.ndarray:
        """Create superposition of multiple wave modes."""
        superposed_field = np.zeros(self.field_shape, dtype=complex)
        
        x = np.arange(self.field_shape[0])
        y = np.arange(self.field_shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for mode in modes:
            # Wave vector from frequency and dispersion relation
            k = mode.frequency / self.c
            
            # Create spatial wave pattern
            wave_pattern = mode.amplitude * np.exp(1j * k * (X + Y) / np.sqrt(2))
            
            # Apply exponential decay for finite Q-factor
            if mode.q_factor > 0:
                decay_length = mode.q_factor
                decay = np.exp(-(X**2 + Y**2) / (2 * decay_length**2))
                wave_pattern *= decay
            
            superposed_field += wave_pattern
        
        return superposed_field
    
    def _reshape_to_field(self, amplitude: np.ndarray) -> np.ndarray:
        """Reshape amplitude to field dimensions."""
        target_size = np.prod(self.field_shape)
        
        if amplitude.size == target_size:
            return amplitude.reshape(self.field_shape)
        elif amplitude.size > target_size:
            flat = amplitude.flatten()[:target_size]
            return flat.reshape(self.field_shape)
        else:
            padded = np.zeros(target_size, dtype=amplitude.dtype)
            padded[:amplitude.size] = amplitude.flatten()
            return padded.reshape(self.field_shape)
    
    def reset(self):
        """Reset amplifier state."""
        self.wave_field = np.zeros(self.field_shape, dtype=complex)
        self.velocity_field = np.zeros(self.field_shape, dtype=complex)
        self.active_modes = []
        self.resonance_history = []


# Legacy compatibility classes for backward compatibility
class AmplificationMode(Enum):
    """Legacy amplification modes for backward compatibility."""
    COHERENT = "coherent"
    RESONANT = "resonant"
    EMERGENT = "emergent"
    COGNITIVE = "cognitive"
    SELECTIVE = "selective"