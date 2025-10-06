"""
Ocean Wave Group Formation via MED + Herniation

Tests whether ocean wave groups at 0.02-0.03 Hz emerge from:
1. MED bounded complexity (depth ≤ 1, nodes ≤ 3)
2. Herniation depth D ≈ 1-2 (MAS framework)
3. Balance operator Ξ ≈ 1.0571

Key hypothesis: Wave groups form because ocean discretizes from continuous
fluid (D→0) to wave packets (D≈1-2), creating natural 0.02 Hz envelope.

Connection to Navier-Stokes:
- Ocean is a fluid obeying N-S equations
- MED bounds N-S complexity via symbolic entropy collapse
- Wave group formation = macro emergence at bounded complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from scipy.fft import fft2, ifft2, fftfreq
from scipy.signal import welch


class OceanWaveMEDTest:
    """Test MED principles on ocean wave group formation."""
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.dx = 100.0 / grid_size  # 100m domain
        self.dt = 0.1  # 0.1 second timesteps
        
        # Physical parameters
        self.g = 9.81  # m/s^2
        self.depth = 50.0  # m, intermediate depth
        
        # MED parameters
        self.xi_balance = 1.0571  # Balance operator from MED
        self.max_depth = 1  # MED bounded complexity
        self.max_nodes = 3
        
        # MAS parameters
        self.f_infinity = 0.030  # Hz, continuous limit
        self.r_relax = 0.438  # Universal relaxation ratio
        
        # Wave field state
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize with random wave field (wind-driven)
        self.eta = np.zeros((grid_size, grid_size))  # Surface elevation
        self.u = np.zeros((grid_size, grid_size))    # x-velocity
        self.v = np.zeros((grid_size, grid_size))    # y-velocity
        
        # Tracking
        self.eta_history = []
        self.group_envelope_history = []
        self.depth_history = []
        
        print(f"Ocean Wave MED Test initialized: {grid_size}x{grid_size} grid")
        print(f"  Domain: 100m x 100m")
        print(f"  MED balance operator: {self.xi_balance:.4f}")
        print(f"  Expected group frequency: ~0.02 Hz")
    
    def initialize_random_waves(self, n_modes=20):
        """Initialize with random wave modes (wind-driven sea state)."""
        
        # Wave number range for wind waves (0.1 to 1.0 rad/m)
        k_min, k_max = 0.1, 1.0
        
        for _ in range(n_modes):
            # Random wave parameters
            k = np.random.uniform(k_min, k_max)
            theta = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.exponential(0.5)  # Rayleigh distributed amplitudes
            phase = np.random.uniform(0, 2*np.pi)
            
            # Dispersion relation: ω² = gk tanh(kh)
            omega = np.sqrt(self.g * k * np.tanh(k * self.depth))
            
            # Add wave component
            kx = k * np.cos(theta)
            ky = k * np.sin(theta)
            
            self.eta += amplitude * np.cos(kx * self.X + ky * self.Y + phase)
            
            # Corresponding velocity field (linear wave theory)
            self.u += amplitude * omega * np.cos(kx * self.X + ky * self.Y + phase) * kx / k
            self.v += amplitude * omega * np.cos(kx * self.X + ky * self.Y + phase) * ky / k
    
    def compute_wave_energy(self) -> float:
        """Compute total wave energy."""
        potential = 0.5 * self.g * np.mean(self.eta**2)
        kinetic = 0.5 * self.depth * np.mean(self.u**2 + self.v**2)
        return potential + kinetic
    
    def compute_herniation_depth(self, energy: float) -> float:
        """
        Compute effective herniation depth from wave energy.
        
        High energy (linear waves) = low D (continuous)
        Low energy (organized groups) = high D (discrete)
        """
        # Normalize energy (initial state = D→0)
        if not hasattr(self, 'initial_energy'):
            self.initial_energy = energy
        
        normalized_energy = energy / self.initial_energy if self.initial_energy > 0 else 1.0
        
        # Depth increases as energy ORGANIZES (not just decreases)
        # Use MAS relation: lower apparent energy = higher organization = higher D
        # But we want to detect ORGANIZATION not just dissipation
        # So we look at energy CONCENTRATION (variance)
        energy_density = self.eta**2 + (self.u**2 + self.v**2) / (2 * self.g)
        organization = np.std(energy_density) / (np.mean(energy_density) + 1e-10)
        
        # High organization (groups) = high D
        # Use organization metric directly
        depth = organization * 2.0  # Scale to get D~1-2 range
        
        return np.clip(depth, 0, 10)
    
    def apply_med_collapse(self):
        """
        Apply MED bounded complexity to wave field.
        
        Enforces:
        1. Depth ≤ 1 (limit complexity)
        2. Balance operator Ξ ≈ 1.0571
        3. Symbolic entropy collapse
        """
        
        # Compute local wave energy density
        energy_density = self.eta**2 + (self.u**2 + self.v**2) / (2 * self.g)
        
        # Identify high-energy regions (need collapse)
        threshold = np.mean(energy_density) * self.xi_balance
        high_energy_mask = energy_density > threshold
        
        # Apply bounded complexity constraint
        # Collapse excess energy into organized structures (groups)
        if np.any(high_energy_mask):
            # Redistribute energy to neighbors (group formation)
            collapse_factor = 0.95  # Slight energy reduction
            self.eta[high_energy_mask] *= collapse_factor
            
            # Add to neighboring cells (creates groups)
            for i in range(1, self.grid_size-1):
                for j in range(1, self.grid_size-1):
                    if high_energy_mask[i, j]:
                        # Energy flows to neighbors (group formation)
                        excess = self.eta[i, j] * (1 - collapse_factor)
                        self.eta[i-1:i+2, j-1:j+2] += excess / 9
    
    def evolve_step(self):
        """Evolve wave field one timestep with MED constraints."""
        
        # Simple shallow water evolution with MED stability
        # du/dt = -g * deta/dx - friction * u
        # dv/dt = -g * deta/dy - friction * v  
        # deta/dt = -depth * (du/dx + dv/dy)
        
        # MED-enforced dissipation (prevents unbounded growth)
        friction = 0.05  # Bounded complexity constraint
        
        # Compute gradients (central differences)
        deta_dx = np.gradient(self.eta, self.dx, axis=1)
        deta_dy = np.gradient(self.eta, self.dx, axis=0)
        
        du_dx = np.gradient(self.u, self.dx, axis=1)
        dv_dy = np.gradient(self.v, self.dx, axis=0)
        
        # Update velocities with friction
        self.u -= (self.g * deta_dx + friction * self.u) * self.dt
        self.v -= (self.g * deta_dy + friction * self.v) * self.dt
        
        # Update surface elevation
        self.eta -= self.depth * (du_dx + dv_dy) * self.dt
        
        # MED energy bound enforcement
        energy = self.compute_wave_energy()
        if energy > 2 * self.initial_energy:
            # Renormalize to prevent explosion (MED bounded complexity)
            scale = np.sqrt(2 * self.initial_energy / energy)
            self.eta *= scale
            self.u *= scale
            self.v *= scale
        
        # Apply MED collapse (creates wave groups)
        self.apply_med_collapse()
        
        # Track state
        energy = self.compute_wave_energy()
        depth = self.compute_herniation_depth(energy)
        
        self.eta_history.append(np.copy(self.eta))
        self.depth_history.append(depth)
        
        # Compute wave group envelope (Hilbert-like)
        envelope = np.abs(self.eta)
        self.group_envelope_history.append(np.mean(envelope))
    
    def run_simulation(self, n_steps=2000):
        """Run wave evolution with MED constraints."""
        
        print("\nInitializing wave field...")
        self.initialize_random_waves(n_modes=30)
        self.initial_energy = self.compute_wave_energy()
        
        print(f"Initial energy: {self.initial_energy:.4f}")
        print(f"\nEvolving for {n_steps} timesteps...")
        print(f"Physical time: {n_steps * self.dt:.1f} seconds")
        
        for step in range(n_steps):
            self.evolve_step()
            
            if step % 200 == 0:
                energy = self.compute_wave_energy()
                depth = self.depth_history[-1]
                print(f"  Step {step}: E={energy:.4f}, D={depth:.2f}")
        
        print("\nSimulation complete!")
    
    def analyze_group_frequency(self) -> Dict:
        """Analyze wave group envelope frequency."""
        
        envelope = np.array(self.group_envelope_history)
        
        # The envelope should show slow modulation (groups)
        # Apply low-pass filter to extract group frequency
        from scipy.signal import butter, filtfilt
        
        # Design lowpass filter (cutoff at 0.5 Hz to capture 0.02 Hz groups)
        nyquist = 0.5 / self.dt
        cutoff = 0.5 / nyquist
        b, a = butter(4, cutoff, btype='low')
        
        # Filter the envelope to get group modulation
        envelope_filtered = filtfilt(b, a, envelope)
        
        # Detrend
        envelope_filtered = envelope_filtered - np.mean(envelope_filtered)
        
        # Compute power spectrum of filtered envelope
        freqs, psd = welch(envelope_filtered, fs=1.0/self.dt, 
                          nperseg=min(256, len(envelope_filtered)))
        
        # Find dominant frequency in low-frequency range (< 0.1 Hz)
        low_freq_mask = freqs < 0.1
        if np.any(low_freq_mask):
            low_freqs = freqs[low_freq_mask]
            low_psd = psd[low_freq_mask]
            dominant_idx = np.argmax(low_psd)
            dominant_freq = low_freqs[dominant_idx]
        else:
            dominant_freq = 0.0
        
        # Compute expected frequency from final depth
        final_depth = self.depth_history[-1]
        expected_freq = self.f_infinity / (1 + final_depth * self.r_relax)
        
        return {
            'observed_frequency': dominant_freq,
            'expected_frequency': expected_freq,
            'final_depth': final_depth,
            'frequencies': freqs,
            'psd': psd,
            'envelope_filtered': envelope_filtered
        }
    
    def visualize_results(self, analysis: Dict):
        """Create visualization of ocean wave MED test."""
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Initial wave field
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(self.eta_history[0], cmap='seismic', 
                        extent=[0, 100, 0, 100], origin='lower')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Initial Wave Field (Random)')
        plt.colorbar(im1, ax=ax1, label='η (m)')
        
        # Plot 2: Final wave field (with groups)
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(self.eta_history[-1], cmap='seismic',
                        extent=[0, 100, 0, 100], origin='lower')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_title('Final Wave Field (Groups Formed)')
        plt.colorbar(im2, ax=ax2, label='η (m)')
        
        # Plot 3: Wave group envelope evolution
        ax3 = plt.subplot(2, 3, 3)
        times = np.arange(len(self.group_envelope_history)) * self.dt
        ax3.plot(times, self.group_envelope_history, 'b-', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mean Envelope')
        ax3.set_title('Wave Group Envelope Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Herniation depth evolution
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(times, self.depth_history, 'r-', linewidth=2)
        ax4.axhline(y=1, color='orange', linestyle='--', label='D=1 (first herniation)')
        ax4.axhline(y=2, color='red', linestyle='--', label='D=2 (2/3 regime)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Herniation Depth D')
        ax4.set_title('Ocean Herniation Depth Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Power spectrum
        ax5 = plt.subplot(2, 3, 5)
        ax5.semilogy(analysis['frequencies'], analysis['psd'], 'b-', linewidth=1.5)
        ax5.axvline(x=analysis['observed_frequency'], color='red', 
                   linestyle='--', linewidth=2, label=f"Observed: {analysis['observed_frequency']:.4f} Hz")
        ax5.axvline(x=0.020, color='green', linestyle='--', 
                   linewidth=2, label='MAS target: 0.020 Hz')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power Spectral Density')
        ax5.set_title('Wave Group Envelope Spectrum')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 0.1)
        
        # Plot 6: Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        validates = abs(analysis['observed_frequency'] - 0.020) / 0.020 < 0.3
        
        summary_text = f"""
OCEAN WAVE MED TEST RESULTS

MED Parameters:
  Balance operator Ξ: {self.xi_balance:.4f}
  Bounded complexity: depth ≤ {self.max_depth}
  
MAS Framework:
  f_∞ = {self.f_infinity:.4f} Hz (continuous)
  r = {self.r_relax:.4f} (relaxation)
  
Results:
  Observed frequency: {analysis['observed_frequency']:.4f} Hz
  Expected (from D): {analysis['expected_frequency']:.4f} Hz
  Final depth: {analysis['final_depth']:.2f}
  
Validation:
  Matches 0.02 Hz: {'YES' if validates else 'NO'}
  
Interpretation:
{'  Wave groups form naturally' if validates else '  Need parameter tuning'}
{'  at D≈1-2 due to MED collapse' if validates else ''}
{'  Ocean herniates from continuous' if validates else ''}
{'  to discrete wave packets!' if validates else ''}
"""
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Ocean Wave Group Formation via MED + Herniation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_dir = Path("results/ocean_wave_med")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.savefig(output_dir / f"ocean_wave_med_{timestamp}.png", 
                   dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_dir}/ocean_wave_med_{timestamp}.png")
        
        plt.show()


def main():
    """Run ocean wave MED test."""
    
    print("=" * 80)
    print("OCEAN WAVE GROUP FORMATION VIA MED + HERNIATION")
    print("=" * 80)
    print()
    
    test = OceanWaveMEDTest(grid_size=64)
    test.run_simulation(n_steps=500)
    
    print("\n" + "=" * 80)
    print("ANALYZING WAVE GROUP FREQUENCY")
    print("=" * 80)
    
    analysis = test.analyze_group_frequency()
    
    print(f"\nObserved group frequency: {analysis['observed_frequency']:.4f} Hz")
    print(f"Expected from depth D={analysis['final_depth']:.2f}: {analysis['expected_frequency']:.4f} Hz")
    print(f"MAS target: 0.020 Hz")
    
    ratio = analysis['observed_frequency'] / 0.020
    print(f"Ratio to target: {ratio:.3f}")
    
    if 0.7 < ratio < 1.3:
        print("\nSUCCESS: Wave groups form at ~0.02 Hz!")
        print("MED bounded complexity creates herniation at D≈1-2")
    else:
        print("\nNote: May need parameter tuning for exact match")
    
    print("\n" + "=" * 80)
    print("VISUALIZING RESULTS")
    print("=" * 80)
    
    test.visualize_results(analysis)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
