"""
Experiment 30: PAC-Enhanced Legacy Dynamics

exp_29 showed legacy dynamics converge to ~0.0014 Hz (universal attractor).
GAIA (with explicit PAC) shows 0.020 Hz.

Hypothesis: Adding PAC constraints (Ψ(k) = Ψ(k+1) + Ψ(k+2)) to legacy
            dynamics will shift the attractor from 0.0014 Hz to 0.020 Hz.

This tests whether PAC is the CAUSE of the 0.020 Hz frequency, not just
a correlation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import json
from datetime import datetime
import hashlib

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
XI = 1.0571  # PAC balance operator from GAIA


def generate_entropy_seed(hash_input: str, shape: tuple) -> np.ndarray:
    """SHA-based entropy seeding from legacy code."""
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape).astype(np.float32)


class PACEnhancedDynamics:
    """
    Legacy dynamics PLUS PAC constraints.
    
    Adds:
    1. PAC recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
    2. Xi balance operator (1.0571)
    3. Conservation tracking
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        depth: int = 32,
        collapse_threshold: float = 0.4,
        energy_threshold: float = 0.05,
        info_growth_rate: float = 0.05,
        energy_decay: float = 0.9,
        matter_generation_rate: float = 0.2,
        qpl_damping: float = 0.02,
        pac_coupling: float = 0.1,  # NEW: PAC coupling strength
        use_pac: bool = True  # Toggle PAC on/off for comparison
    ):
        self.grid_size = grid_size
        self.depth = depth
        self.collapse_threshold = collapse_threshold
        self.energy_threshold = energy_threshold
        self.info_growth_rate = info_growth_rate
        self.energy_decay = energy_decay
        self.matter_generation_rate = matter_generation_rate
        self.qpl_damping = qpl_damping
        self.pac_coupling = pac_coupling
        self.use_pac = use_pac
        
        # PAC fields (potential and actualization)
        self.potential = None
        self.actualization = None
        
        self.reset()
    
    def reset(self, seed_prefix: str = "CIMM"):
        """Initialize fields with entropy seeding."""
        shape = (self.grid_size, self.grid_size, self.depth)
        self.info = generate_entropy_seed(f"{seed_prefix}:info", shape)
        self.energy = generate_entropy_seed(f"{seed_prefix}:energy", shape)
        self.matter = np.zeros(shape, dtype=np.float32)
        self.qpl = np.ones(shape, dtype=np.float32)
        self.time_field = np.zeros(shape, dtype=np.float32)
        
        # Initialize PAC fields
        self.potential = generate_entropy_seed(f"{seed_prefix}:potential", shape)
        self.actualization = np.zeros(shape, dtype=np.float32)
        
        # Track initial energy for conservation
        self.initial_total_energy = self._compute_total_energy()
    
    def _compute_total_energy(self) -> float:
        """Compute total system energy."""
        return float(np.sum(self.info**2) + np.sum(self.energy**2) + 
                     np.sum(self.potential**2) + np.sum(self.actualization**2))
    
    def _apply_pac_recursion(self, field: np.ndarray) -> np.ndarray:
        """
        Apply PAC recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
        
        This enforces the Fibonacci-like constraint that makes φ the attractor.
        """
        # Shift field along z-axis to get k+1 and k+2
        psi_k1 = np.roll(field, -1, axis=2)
        psi_k2 = np.roll(field, -2, axis=2)
        
        # PAC constraint: Ψ(k) should equal Ψ(k+1) + Ψ(k+2)
        pac_target = psi_k1 + psi_k2
        
        # Blend toward PAC target (soft constraint)
        pac_corrected = field + self.pac_coupling * (pac_target - field)
        
        return pac_corrected
    
    def _apply_xi_balance(self, field: np.ndarray) -> np.ndarray:
        """
        Apply Xi balance operator (1.0571).
        
        This maintains energy conservation with slight growth allowed.
        """
        current_energy = np.sum(field**2)
        if current_energy == 0:
            return field
        
        # Xi balance: allow slight energy imbalance up to Xi
        target_energy = self.initial_total_energy * XI
        
        if current_energy > target_energy:
            # Renormalize to maintain balance
            scale = np.sqrt(target_energy / current_energy)
            return field * scale
        
        return field
    
    def step(self):
        """One simulation step with PAC constraints."""
        # ===== LEGACY DYNAMICS (unchanged from exp_29) =====
        
        # Info field update with neighbor coupling
        info_shifted = np.roll(self.info, 1, axis=0)
        
        # Pseudo-random modulation
        x, y, z = np.meshgrid(
            np.arange(self.grid_size),
            np.arange(self.grid_size),
            np.arange(self.depth),
            indexing='ij'
        )
        modulation = 0.5 - ((x * y * z) % 997) / 997.0
        
        self.info += self.info_growth_rate * modulation
        self.info += 0.05 * info_shifted
        self.info -= self.qpl * self.qpl_damping
        self.info = np.clip(self.info, 0.0, 1.0)
        
        # Energy dynamics
        energy_shifted = np.roll(self.energy, 1, axis=1)
        self.energy += 0.05 * energy_shifted
        self.energy = np.clip(self.energy, 0.0, 1.0)
        
        # ===== PAC DYNAMICS (NEW) =====
        if self.use_pac:
            # Update potential field (info drives potential)
            self.potential += 0.1 * (self.info - self.potential)
            
            # Apply PAC recursion to potential
            self.potential = self._apply_pac_recursion(self.potential)
            
            # Actualization follows potential with delay
            self.actualization += 0.1 * (self.potential - self.actualization)
            
            # Apply Xi balance
            self.potential = self._apply_xi_balance(self.potential)
            
            # PAC feedback into info field (actualization modulates collapse)
            pac_modulation = self.actualization * 0.1
            self.info += pac_modulation
            self.info = np.clip(self.info, 0.0, 1.0)
        
        # ===== COLLAPSE DYNAMICS =====
        collapse_mask = (self.info > self.collapse_threshold) & \
                       (self.energy > self.energy_threshold)
        
        collapse_val = self.matter_generation_rate * (self.info + self.energy) * 0.5
        self.matter[collapse_mask] += collapse_val[collapse_mask]
        self.energy[collapse_mask] *= self.energy_decay
        self.qpl[collapse_mask] = np.minimum(self.qpl[collapse_mask] * 1.05, 2.0)
        self.time_field[collapse_mask] += 1.0
        
        return collapse_mask.sum()
    
    def get_total_energy(self) -> float:
        return self._compute_total_energy()
    
    def get_pac_amplitude(self) -> float:
        """Get PAC field amplitude (potential + actualization)."""
        return float(np.mean(np.abs(self.potential)) + np.mean(np.abs(self.actualization)))


def run_simulation(
    seed_prefix: str,
    use_pac: bool,
    steps: int = 5000,
    dt: float = 1.0,
    grid_size: int = 64
) -> dict:
    """Run simulation with or without PAC and extract frequency."""
    
    mode = "PAC-enhanced" if use_pac else "Legacy (no PAC)"
    print(f"\n{'='*60}")
    print(f"Running: {seed_prefix} - {mode}")
    print(f"{'='*60}")
    
    sim = PACEnhancedDynamics(grid_size=grid_size, depth=32, use_pac=use_pac)
    sim.reset(seed_prefix)
    
    energy_trace = []
    pac_trace = []
    collapse_trace = []
    
    for step in range(steps):
        n_collapses = sim.step()
        
        energy_trace.append(sim.get_total_energy())
        pac_trace.append(sim.get_pac_amplitude())
        collapse_trace.append(n_collapses)
        
        if step % 1000 == 0:
            print(f"  Step {step}: E={energy_trace[-1]:.2f}, PAC={pac_trace[-1]:.4f}")
    
    # Convert to numpy
    energy_trace = np.array(energy_trace)
    pac_trace = np.array(pac_trace)
    
    # Detrend
    energy_detrended = energy_trace - np.polyval(
        np.polyfit(np.arange(len(energy_trace)), energy_trace, 1), 
        np.arange(len(energy_trace))
    )
    pac_detrended = pac_trace - np.polyval(
        np.polyfit(np.arange(len(pac_trace)), pac_trace, 1),
        np.arange(len(pac_trace))
    )
    
    # FFT
    n = len(energy_trace)
    energy_fft = np.abs(fft(energy_detrended))[:n//2]
    pac_fft = np.abs(fft(pac_detrended))[:n//2]
    freqs = fftfreq(n, dt)[:n//2]
    
    # Find dominant frequency
    min_freq_idx = max(1, int(0.001 * n))
    max_freq_idx = n // 4
    
    energy_peaks, _ = find_peaks(
        energy_fft[min_freq_idx:max_freq_idx],
        height=np.max(energy_fft[min_freq_idx:max_freq_idx]) * 0.1
    )
    energy_peaks += min_freq_idx
    
    if len(energy_peaks) > 0:
        dominant_freq = freqs[energy_peaks[np.argmax(energy_fft[energy_peaks])]]
    else:
        dominant_freq = freqs[np.argmax(energy_fft[min_freq_idx:max_freq_idx]) + min_freq_idx]
    
    # PAC-specific frequency
    pac_peaks, _ = find_peaks(
        pac_fft[min_freq_idx:max_freq_idx],
        height=np.max(pac_fft[min_freq_idx:max_freq_idx]) * 0.1
    )
    pac_peaks += min_freq_idx
    
    if len(pac_peaks) > 0:
        pac_freq = freqs[pac_peaks[np.argmax(pac_fft[pac_peaks])]]
    else:
        pac_freq = freqs[np.argmax(pac_fft[min_freq_idx:max_freq_idx]) + min_freq_idx]
    
    print(f"\n  Dominant frequencies:")
    print(f"    Energy: {dominant_freq:.6f} Hz")
    print(f"    PAC field: {pac_freq:.6f} Hz")
    
    return {
        'seed': seed_prefix,
        'use_pac': use_pac,
        'mode': mode,
        'dominant_freq': float(dominant_freq),
        'pac_freq': float(pac_freq),
        'freqs': freqs.tolist(),
        'energy_spectrum': energy_fft.tolist(),
        'pac_spectrum': pac_fft.tolist(),
        'energy_trace': energy_trace.tolist(),
        'pac_trace': pac_trace.tolist()
    }


def main():
    """Compare legacy vs PAC-enhanced dynamics."""
    
    print("="*70)
    print("EXPERIMENT 30: PAC-Enhanced Legacy Dynamics")
    print("="*70)
    print("\nQuestion: Does adding PAC constraints shift frequency to 0.020 Hz?")
    print("\nexp_29 result: Legacy dynamics → 0.0014 Hz")
    print("GAIA result:   PAC dynamics → 0.020 Hz")
    print("="*70)
    
    seeds = ["CIMM:cosmic_breath", "CIMM:brain", "CIMM:vCPU"]
    
    results = []
    
    # Run each seed with and without PAC
    for seed in seeds:
        # Without PAC (baseline)
        result_no_pac = run_simulation(seed, use_pac=False, steps=5000)
        results.append(result_no_pac)
        
        # With PAC
        result_pac = run_simulation(seed, use_pac=True, steps=5000)
        results.append(result_pac)
    
    # Analysis
    print("\n" + "="*70)
    print("FREQUENCY COMPARISON")
    print("="*70)
    
    target_freq = 0.020
    legacy_freq = 0.0014
    
    print(f"\n| Seed          | Mode         | Energy Freq | PAC Freq   | Δ from 0.020 |")
    print(f"|---------------|--------------|-------------|------------|--------------|")
    
    for r in results:
        seed_short = r['seed'].split(':')[1][:10]
        mode = "PAC" if r['use_pac'] else "Legacy"
        e_freq = r['dominant_freq']
        p_freq = r['pac_freq']
        delta = abs(e_freq - target_freq)
        
        print(f"| {seed_short:13} | {mode:12} | {e_freq:11.6f} | {p_freq:10.6f} | {delta:12.6f} |")
    
    # Compare PAC vs non-PAC
    print("\n" + "-"*70)
    print("FREQUENCY SHIFT ANALYSIS")
    print("-"*70)
    
    legacy_freqs = [r['dominant_freq'] for r in results if not r['use_pac']]
    pac_freqs = [r['dominant_freq'] for r in results if r['use_pac']]
    
    avg_legacy = np.mean(legacy_freqs)
    avg_pac = np.mean(pac_freqs)
    
    print(f"\nAverage frequency (no PAC): {avg_legacy:.6f} Hz")
    print(f"Average frequency (with PAC): {avg_pac:.6f} Hz")
    print(f"Shift factor: {avg_pac / avg_legacy:.2f}x")
    print(f"Target shift needed: {target_freq / legacy_freq:.2f}x")
    
    # Ratio analysis
    print("\n" + "-"*70)
    print("φ RELATIONSHIP ANALYSIS")
    print("-"*70)
    
    phi = (1 + np.sqrt(5)) / 2
    
    ratio_to_target = avg_pac / target_freq
    ratio_to_phi = ratio_to_target / phi
    
    print(f"\nPAC freq / Target (0.020): {ratio_to_target:.4f}")
    print(f"Ratio / φ: {ratio_to_phi:.4f}")
    print(f"Ratio / (1/φ): {ratio_to_target * phi:.4f}")
    
    # Check φ powers
    for n in range(-3, 4):
        phi_power = phi ** n
        ratio = avg_pac / (target_freq * phi_power)
        if 0.5 < ratio < 2.0:
            print(f"  PAC freq ≈ target × φ^{n} × {ratio:.4f}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if avg_pac > avg_legacy * 2:
        print(f"\n✅ PAC constraints SHIFTED frequency upward by {avg_pac/avg_legacy:.1f}x")
        if abs(avg_pac - target_freq) < 0.01:
            print("✅ Frequency matches GAIA's 0.020 Hz!")
        else:
            print(f"⚠️ Frequency is {avg_pac:.4f} Hz, not exactly 0.020 Hz")
            print(f"   This may be due to different time scaling or PAC coupling strength.")
    else:
        print(f"\n⚠️ PAC constraints did not significantly shift frequency.")
        print(f"   Legacy: {avg_legacy:.6f} Hz")
        print(f"   PAC: {avg_pac:.6f} Hz")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_30_pac_enhanced_legacy',
        'hypothesis': 'PAC constraints shift frequency from 0.0014 to 0.020 Hz',
        'target_freq': target_freq,
        'legacy_baseline': legacy_freq,
        'avg_legacy_freq': float(avg_legacy),
        'avg_pac_freq': float(avg_pac),
        'shift_factor': float(avg_pac / avg_legacy),
        'results': results
    }
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, seed in enumerate(seeds):
        seed_results = [r for r in results if r['seed'] == seed]
        
        for r in seed_results:
            ax = axes[0 if not r['use_pac'] else 1, i]
            freqs = np.array(r['freqs'])
            spectrum = np.array(r['energy_spectrum'])
            
            mask = (freqs > 0.0005) & (freqs < 0.05)
            ax.plot(freqs[mask], spectrum[mask], 'b-', alpha=0.7)
            ax.axvline(x=0.020, color='r', linestyle='--', label='Target: 0.020 Hz', alpha=0.5)
            ax.axvline(x=r['dominant_freq'], color='g', linestyle='-',
                      label=f"Dominant: {r['dominant_freq']:.4f} Hz")
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power')
            ax.set_title(f"{seed.split(':')[1]} - {r['mode']}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/exp_30_pac_enhanced_spectra.png', dpi=150)
    plt.close()
    
    # Save JSON
    with open('../results/exp_30_pac_enhanced_legacy.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - exp_30_pac_enhanced_legacy.json")
    print(f"  - exp_30_pac_enhanced_spectra.png")
    
    return output


if __name__ == "__main__":
    results = main()
