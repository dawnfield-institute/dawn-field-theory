"""
Experiment 31: Klein-Gordon Field Dynamics

exp_30 showed PAC alone doesn't shift frequency.
GAIA uses Klein-Gordon field evolution: ∂²ψ/∂t² = c²∇²ψ - m²ψ

Hypothesis: Adding wave-like field dynamics (Laplacian evolution) will
            produce oscillations at the 0.020 Hz frequency.

Key insight from GAIA's conservation_engine.py:
- Uses discrete Laplacian for field dynamics
- Potential ↔ Actualization coupling
- Klein-Gordon equation governs evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import json
from datetime import datetime
import hashlib

# Constants
PHI = (1 + np.sqrt(5)) / 2
XI = 1.0571  # PAC balance operator


def generate_entropy_seed(hash_input: str, shape: tuple) -> np.ndarray:
    """SHA-based entropy seeding."""
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape).astype(np.float32)


class KleinGordonDynamics:
    """
    Full PAC dynamics with Klein-Gordon wave evolution.
    
    From GAIA's conservation_engine.py:
    - Discrete Laplacian ∇²ψ
    - Klein-Gordon: ∂²ψ/∂t² = c²∇²ψ - m²ψ
    - Potential ↔ Actualization coupling
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        depth: int = 32,
        c: float = 1.0,      # Wave speed
        mass: float = 0.1,    # Field mass (controls oscillation frequency)
        coupling: float = 0.1,  # P↔A coupling
        dt: float = 0.1,      # Time step
        damping: float = 0.01  # Small damping for stability
    ):
        self.grid_size = grid_size
        self.depth = depth
        self.c = c
        self.mass = mass
        self.coupling = coupling
        self.dt = dt
        self.damping = damping
        
        self.reset()
    
    def reset(self, seed_prefix: str = "CIMM"):
        """Initialize fields."""
        shape = (self.grid_size, self.grid_size, self.depth)
        
        # Potential field (main field)
        self.psi = generate_entropy_seed(f"{seed_prefix}:psi", shape) * 0.1
        # Field velocity (∂ψ/∂t)
        self.psi_dot = np.zeros(shape, dtype=np.float32)
        
        # Actualization field (coupled)
        self.phi = np.zeros(shape, dtype=np.float32)
        self.phi_dot = np.zeros(shape, dtype=np.float32)
        
        # Collapse tracking
        self.collapse_field = np.zeros(shape, dtype=np.float32)
        
        # History
        self.total_energy_history = []
        self.psi_amplitude_history = []
    
    def _laplacian_3d(self, field: np.ndarray) -> np.ndarray:
        """
        Discrete 3D Laplacian using central differences.
        
        ∇²ψ = ψ(x+1) + ψ(x-1) + ψ(y+1) + ψ(y-1) + ψ(z+1) + ψ(z-1) - 6ψ
        """
        laplacian = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        )
        return laplacian
    
    def step(self):
        """
        One time step using Klein-Gordon dynamics.
        
        ∂²ψ/∂t² = c²∇²ψ - m²ψ + coupling*φ
        ∂²φ/∂t² = c²∇²φ - m²φ + coupling*ψ
        """
        # Calculate Laplacians
        laplacian_psi = self._laplacian_3d(self.psi)
        laplacian_phi = self._laplacian_3d(self.phi)
        
        # Klein-Gordon accelerations
        psi_ddot = (
            self.c**2 * laplacian_psi - 
            self.mass**2 * self.psi + 
            self.coupling * self.phi -
            self.damping * self.psi_dot  # Damping for stability
        )
        
        phi_ddot = (
            self.c**2 * laplacian_phi - 
            self.mass**2 * self.phi + 
            self.coupling * self.psi -
            self.damping * self.phi_dot
        )
        
        # Verlet integration (symplectic, conserves energy better)
        self.psi_dot += psi_ddot * self.dt
        self.psi += self.psi_dot * self.dt
        
        self.phi_dot += phi_ddot * self.dt
        self.phi += self.phi_dot * self.dt
        
        # PAC constraint: check Ψ(k) = Ψ(k+1) + Ψ(k+2)
        psi_k1 = np.roll(self.psi, -1, axis=2)
        psi_k2 = np.roll(self.psi, -2, axis=2)
        pac_residual = self.psi - (psi_k1 + psi_k2)
        
        # Soft PAC correction
        self.psi -= 0.01 * pac_residual
        
        # Xi balance (energy conservation with slight growth)
        total_energy = self.get_total_energy()
        if len(self.total_energy_history) > 0:
            initial_energy = self.total_energy_history[0]
            if total_energy > XI * initial_energy:
                scale = np.sqrt(XI * initial_energy / total_energy)
                self.psi *= scale
                self.phi *= scale
                self.psi_dot *= scale
                self.phi_dot *= scale
        
        # Collapse dynamics (SEC-like threshold at 1/φ)
        sec_threshold = 1.0 / PHI
        amplitude = np.abs(self.psi)
        collapse_mask = amplitude > sec_threshold
        self.collapse_field[collapse_mask] += 1.0
        
        # Record history
        self.total_energy_history.append(total_energy)
        self.psi_amplitude_history.append(float(np.mean(np.abs(self.psi))))
        
        return collapse_mask.sum()
    
    def get_total_energy(self) -> float:
        """Total field energy (kinetic + potential + gradient)."""
        kinetic = 0.5 * (np.sum(self.psi_dot**2) + np.sum(self.phi_dot**2))
        potential = 0.5 * self.mass**2 * (np.sum(self.psi**2) + np.sum(self.phi**2))
        coupling_energy = -self.coupling * np.sum(self.psi * self.phi)
        
        # Gradient energy from Laplacian
        grad_psi = np.gradient(self.psi)
        grad_phi = np.gradient(self.phi)
        gradient_energy = 0.5 * self.c**2 * (
            np.sum(grad_psi[0]**2 + grad_psi[1]**2 + grad_psi[2]**2) +
            np.sum(grad_phi[0]**2 + grad_phi[1]**2 + grad_phi[2]**2)
        )
        
        return float(kinetic + potential + gradient_energy + coupling_energy)
    
    def get_dominant_frequency(self, signal: np.ndarray, dt: float) -> float:
        """Extract dominant frequency from signal."""
        # Detrend
        detrended = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1), np.arange(len(signal)))
        
        # FFT
        n = len(signal)
        spectrum = np.abs(fft(detrended))[:n//2]
        freqs = fftfreq(n, dt)[:n//2]
        
        # Find peaks (skip DC)
        min_idx = max(1, n // 100)
        max_idx = n // 4
        
        peaks, _ = find_peaks(spectrum[min_idx:max_idx], height=np.max(spectrum[min_idx:max_idx]) * 0.1)
        peaks += min_idx
        
        if len(peaks) > 0:
            return float(freqs[peaks[np.argmax(spectrum[peaks])]])
        else:
            return float(freqs[np.argmax(spectrum[min_idx:max_idx]) + min_idx])


def run_klein_gordon_simulation(
    seed_prefix: str,
    steps: int = 10000,
    dt: float = 0.1,
    mass: float = 0.1,
    grid_size: int = 32
) -> dict:
    """Run Klein-Gordon simulation and extract frequency."""
    
    print(f"\n{'='*60}")
    print(f"Running Klein-Gordon: {seed_prefix}")
    print(f"  mass = {mass}, dt = {dt}, steps = {steps}")
    print(f"{'='*60}")
    
    sim = KleinGordonDynamics(
        grid_size=grid_size,
        depth=16,  # Smaller for speed
        mass=mass,
        dt=dt
    )
    sim.reset(seed_prefix)
    
    for step in range(steps):
        n_collapses = sim.step()
        
        if step % 2000 == 0:
            E = sim.get_total_energy()
            amp = sim.psi_amplitude_history[-1]
            print(f"  Step {step}: E={E:.2f}, |ψ|={amp:.6f}, collapses={n_collapses}")
    
    # Extract frequency from amplitude history
    signal = np.array(sim.psi_amplitude_history)
    dominant_freq = sim.get_dominant_frequency(signal, dt)
    
    # Also check energy oscillations
    energy_signal = np.array(sim.total_energy_history)
    energy_freq = sim.get_dominant_frequency(energy_signal, dt)
    
    print(f"\n  Dominant frequencies:")
    print(f"    Amplitude: {dominant_freq:.6f} Hz")
    print(f"    Energy: {energy_freq:.6f} Hz")
    
    # Natural frequency from Klein-Gordon
    # ω = sqrt(m² + k²) where k is spatial frequency
    # For lowest mode: ω ≈ m
    natural_freq = mass / (2 * np.pi)
    print(f"    Natural (m/2π): {natural_freq:.6f} Hz")
    
    return {
        'seed': seed_prefix,
        'mass': mass,
        'dt': dt,
        'steps': steps,
        'dominant_freq': dominant_freq,
        'energy_freq': energy_freq,
        'natural_freq': natural_freq,
        'amplitude_history': signal.tolist(),
        'energy_history': energy_signal.tolist()
    }


def find_mass_for_target_frequency(target_freq: float = 0.020) -> float:
    """
    Find the mass parameter that produces 0.020 Hz.
    
    From Klein-Gordon: ω = m (in natural units)
    So m = 2π × f = 2π × 0.020 ≈ 0.126
    """
    return 2 * np.pi * target_freq


def main():
    """Test if Klein-Gordon dynamics produce 0.020 Hz."""
    
    print("="*70)
    print("EXPERIMENT 31: Klein-Gordon Field Dynamics")
    print("="*70)
    print("\nQuestion: Does wave-like field evolution produce 0.020 Hz?")
    print("\nKey insight: Klein-Gordon frequency ω = sqrt(m² + k²)")
    print("             For m = 2π × 0.020 ≈ 0.126, expect f = 0.020 Hz")
    print("="*70)
    
    # Calculate target mass
    target_freq = 0.020
    target_mass = find_mass_for_target_frequency(target_freq)
    print(f"\nTarget mass for {target_freq} Hz: m = {target_mass:.4f}")
    
    results = []
    
    # Test 1: Default mass (arbitrary)
    result1 = run_klein_gordon_simulation(
        "CIMM:default",
        steps=10000,
        dt=0.1,
        mass=0.1,  # Arbitrary
        grid_size=32
    )
    results.append(result1)
    
    # Test 2: Mass tuned for 0.020 Hz
    result2 = run_klein_gordon_simulation(
        "CIMM:tuned",
        steps=10000,
        dt=0.1,
        mass=target_mass,
        grid_size=32
    )
    results.append(result2)
    
    # Test 3: Different seed with tuned mass
    result3 = run_klein_gordon_simulation(
        "CIMM:cosmic",
        steps=10000,
        dt=0.1,
        mass=target_mass,
        grid_size=32
    )
    results.append(result3)
    
    # Analysis
    print("\n" + "="*70)
    print("FREQUENCY ANALYSIS")
    print("="*70)
    
    print(f"\n| Seed    | Mass   | Amplitude Freq | Energy Freq | Natural Freq |")
    print(f"|---------|--------|----------------|-------------|--------------|")
    
    for r in results:
        print(f"| {r['seed'].split(':')[1]:7} | {r['mass']:6.4f} | "
              f"{r['dominant_freq']:14.6f} | {r['energy_freq']:11.6f} | "
              f"{r['natural_freq']:12.6f} |")
    
    # Key question: does tuned mass produce 0.020 Hz?
    print("\n" + "-"*70)
    print("KEY QUESTION: Does m = 2π × 0.020 produce f = 0.020 Hz?")
    print("-"*70)
    
    tuned_results = [r for r in results if abs(r['mass'] - target_mass) < 0.01]
    if tuned_results:
        avg_freq = np.mean([r['dominant_freq'] for r in tuned_results])
        error = abs(avg_freq - target_freq) / target_freq * 100
        
        print(f"\nTuned mass ({target_mass:.4f}) produced:")
        print(f"  Average frequency: {avg_freq:.6f} Hz")
        print(f"  Target frequency:  {target_freq:.6f} Hz")
        print(f"  Error: {error:.2f}%")
        
        if error < 20:
            print(f"\n✅ Klein-Gordon with m = 2π×{target_freq} produces ~{target_freq} Hz!")
            print("   This confirms the frequency comes from the field mass/coupling.")
        else:
            print(f"\n⚠️ Frequency doesn't match exactly. Check boundary conditions.")
    
    # Deep insight
    print("\n" + "="*70)
    print("THEORETICAL INSIGHT")
    print("="*70)
    print("""
The 0.020 Hz frequency in GAIA emerges from:

1. Klein-Gordon equation: ω = m (natural units)
2. GAIA sets parameters that effectively give m ≈ 2π × 0.020

This means 0.020 Hz isn't "discovered" — it's encoded in the parameters.

HOWEVER: The question is whether these parameters are NECESSARY
for stable PAC dynamics, or just chosen arbitrarily.

If m = 2π × 0.020 is the ONLY value that gives stable PAC evolution,
then 0.020 Hz IS fundamental to PAC.
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_31_klein_gordon_dynamics',
        'hypothesis': 'Klein-Gordon dynamics with tuned mass produce 0.020 Hz',
        'target_freq': target_freq,
        'target_mass': float(target_mass),
        'results': results
    }
    
    # Visualization
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 8))
    
    for i, r in enumerate(results):
        # Amplitude history
        ax = axes[0, i]
        signal = np.array(r['amplitude_history'])
        time = np.arange(len(signal)) * r['dt']
        ax.plot(time[:2000], signal[:2000], 'b-', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('|ψ|')
        ax.set_title(f"{r['seed'].split(':')[1]} (m={r['mass']:.3f})")
        ax.grid(True, alpha=0.3)
        
        # Spectrum
        ax = axes[1, i]
        signal_detrended = signal - np.mean(signal)
        n = len(signal)
        spectrum = np.abs(fft(signal_detrended))[:n//2]
        freqs = fftfreq(n, r['dt'])[:n//2]
        mask = (freqs > 0.001) & (freqs < 0.1)
        ax.plot(freqs[mask], spectrum[mask], 'b-', alpha=0.7)
        ax.axvline(x=target_freq, color='r', linestyle='--', label='Target: 0.020 Hz')
        ax.axvline(x=r['dominant_freq'], color='g', linestyle='-',
                   label=f"Found: {r['dominant_freq']:.4f} Hz")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/exp_31_klein_gordon_spectra.png', dpi=150)
    plt.close()
    
    with open('../results/exp_31_klein_gordon_dynamics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - exp_31_klein_gordon_dynamics.json")
    print(f"  - exp_31_klein_gordon_spectra.png")
    
    return output


if __name__ == "__main__":
    results = main()
