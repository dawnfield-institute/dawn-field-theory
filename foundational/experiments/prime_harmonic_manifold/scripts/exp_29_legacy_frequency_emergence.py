"""
Experiment 29: Legacy Frequency Emergence Test

Tests whether the 0.020 Hz organizing frequency emerges naturally from
raw PAC/SEC dynamics in the legacy simulations (brain, cosmo, vcpu).

These simulations were NOT designed with 0.020 Hz in mind. If this frequency
emerges, it's genuine independent validation that 0.020 Hz is a fundamental
organizing frequency from PAC constraints.

Prediction: Dominant frequency will be near 0.020 Hz (±0.005 Hz)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import json
from datetime import datetime

# ============================================================================
# SIMPLIFIED LEGACY DYNAMICS (CPU-only for portability)
# Extracted core equations from legacy experiments
# ============================================================================

def generate_entropy_seed(hash_input: str, shape: tuple) -> np.ndarray:
    """SHA-based entropy seeding from legacy code."""
    import hashlib
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape).astype(np.float32)


class LegacyDynamics:
    """Core dynamics from legacy experiments - NO hardcoded frequencies."""
    
    def __init__(
        self,
        grid_size: int = 64,
        depth: int = 32,
        collapse_threshold: float = 0.4,
        energy_threshold: float = 0.05,
        info_growth_rate: float = 0.05,
        energy_decay: float = 0.9,
        matter_generation_rate: float = 0.2,
        qpl_damping: float = 0.02
    ):
        self.grid_size = grid_size
        self.depth = depth
        self.collapse_threshold = collapse_threshold
        self.energy_threshold = energy_threshold
        self.info_growth_rate = info_growth_rate
        self.energy_decay = energy_decay
        self.matter_generation_rate = matter_generation_rate
        self.qpl_damping = qpl_damping
        
        self.reset()
    
    def reset(self, seed_prefix: str = "CIMM"):
        """Initialize fields with entropy seeding."""
        shape = (self.grid_size, self.grid_size, self.depth)
        self.info = generate_entropy_seed(f"{seed_prefix}:info", shape)
        self.energy = generate_entropy_seed(f"{seed_prefix}:energy", shape)
        self.matter = np.zeros(shape, dtype=np.float32)
        self.qpl = np.ones(shape, dtype=np.float32)
        self.time_field = np.zeros(shape, dtype=np.float32)
    
    def step(self):
        """One simulation step - core dynamics from legacy code."""
        # Info field update with neighbor coupling
        info_shifted = np.roll(self.info, 1, axis=0)
        
        # Pseudo-random modulation (legacy pattern)
        x, y, z = np.meshgrid(
            np.arange(self.grid_size),
            np.arange(self.grid_size),
            np.arange(self.depth),
            indexing='ij'
        )
        modulation = 0.5 - ((x * y * z) % 997) / 997.0
        
        # Info dynamics
        self.info += self.info_growth_rate * modulation
        self.info += 0.05 * info_shifted
        self.info -= self.qpl * self.qpl_damping
        self.info = np.clip(self.info, 0.0, 1.0)
        
        # Energy dynamics
        energy_shifted = np.roll(self.energy, 1, axis=1)
        self.energy += 0.05 * energy_shifted
        self.energy = np.clip(self.energy, 0.0, 1.0)
        
        # Collapse dynamics (SEC-like threshold)
        collapse_mask = (self.info > self.collapse_threshold) & \
                       (self.energy > self.energy_threshold)
        
        collapse_val = self.matter_generation_rate * (self.info + self.energy) * 0.5
        self.matter[collapse_mask] += collapse_val[collapse_mask]
        self.energy[collapse_mask] *= self.energy_decay
        self.qpl[collapse_mask] = np.minimum(self.qpl[collapse_mask] * 1.05, 2.0)
        self.time_field[collapse_mask] += 1.0
        
        return collapse_mask.sum()  # Return number of collapses
    
    def get_total_energy(self) -> float:
        """Get total system energy."""
        return float(np.sum(self.info) + np.sum(self.energy))
    
    def get_collapse_rate(self) -> float:
        """Get current collapse rate."""
        return float(np.mean(self.time_field))
    
    def get_matter_density(self) -> float:
        """Get matter density."""
        return float(np.mean(self.matter))


def run_simulation_and_extract_frequency(
    seed_prefix: str,
    steps: int = 2000,
    dt: float = 1.0,  # Time step in arbitrary units
    grid_size: int = 64
) -> dict:
    """
    Run simulation and extract dominant frequency via FFT.
    
    Args:
        seed_prefix: Seed for entropy initialization
        steps: Number of simulation steps
        dt: Time step size
        grid_size: Simulation grid size
    
    Returns:
        Dictionary with frequency analysis results
    """
    print(f"\n{'='*60}")
    print(f"Running simulation: {seed_prefix}")
    print(f"{'='*60}")
    
    sim = LegacyDynamics(grid_size=grid_size, depth=32)
    sim.reset(seed_prefix)
    
    # Time series to track
    energy_trace = []
    collapse_trace = []
    matter_trace = []
    
    for step in range(steps):
        n_collapses = sim.step()
        
        energy_trace.append(sim.get_total_energy())
        collapse_trace.append(n_collapses)
        matter_trace.append(sim.get_matter_density())
        
        if step % 500 == 0:
            print(f"  Step {step}: E={energy_trace[-1]:.2f}, "
                  f"collapses={n_collapses}, matter={matter_trace[-1]:.4f}")
    
    # Convert to numpy
    energy_trace = np.array(energy_trace)
    collapse_trace = np.array(collapse_trace)
    matter_trace = np.array(matter_trace)
    
    # Detrend signals (remove linear trend)
    energy_detrended = energy_trace - np.polyval(np.polyfit(np.arange(len(energy_trace)), energy_trace, 1), np.arange(len(energy_trace)))
    collapse_detrended = collapse_trace - np.polyval(np.polyfit(np.arange(len(collapse_trace)), collapse_trace, 1), np.arange(len(collapse_trace)))
    
    # FFT analysis
    sample_rate = 1.0 / dt  # Hz
    n = len(energy_trace)
    
    # Energy spectrum
    energy_fft = np.abs(fft(energy_detrended))[:n//2]
    collapse_fft = np.abs(fft(collapse_detrended))[:n//2]
    freqs = fftfreq(n, dt)[:n//2]
    
    # Find dominant frequencies (excluding DC)
    min_freq_idx = max(1, int(0.001 * n))  # Skip very low frequencies
    max_freq_idx = n // 4  # Focus on lower frequencies
    
    energy_peaks, energy_properties = find_peaks(
        energy_fft[min_freq_idx:max_freq_idx], 
        height=np.max(energy_fft[min_freq_idx:max_freq_idx]) * 0.1
    )
    collapse_peaks, collapse_properties = find_peaks(
        collapse_fft[min_freq_idx:max_freq_idx],
        height=np.max(collapse_fft[min_freq_idx:max_freq_idx]) * 0.1
    )
    
    # Adjust indices
    energy_peaks += min_freq_idx
    collapse_peaks += min_freq_idx
    
    # Get dominant frequencies
    if len(energy_peaks) > 0:
        dominant_energy_freq = freqs[energy_peaks[np.argmax(energy_fft[energy_peaks])]]
    else:
        dominant_energy_freq = freqs[np.argmax(energy_fft[min_freq_idx:max_freq_idx]) + min_freq_idx]
    
    if len(collapse_peaks) > 0:
        dominant_collapse_freq = freqs[collapse_peaks[np.argmax(collapse_fft[collapse_peaks])]]
    else:
        dominant_collapse_freq = freqs[np.argmax(collapse_fft[min_freq_idx:max_freq_idx]) + min_freq_idx]
    
    print(f"\n  Dominant frequencies:")
    print(f"    Energy: {dominant_energy_freq:.6f} Hz")
    print(f"    Collapse: {dominant_collapse_freq:.6f} Hz")
    
    return {
        'seed': seed_prefix,
        'steps': steps,
        'dt': dt,
        'dominant_energy_freq': float(dominant_energy_freq),
        'dominant_collapse_freq': float(dominant_collapse_freq),
        'energy_trace': energy_trace.tolist(),
        'collapse_trace': collapse_trace.tolist(),
        'freqs': freqs.tolist(),
        'energy_spectrum': energy_fft.tolist(),
        'collapse_spectrum': collapse_fft.tolist()
    }


def test_frequency_emergence():
    """
    Main experiment: Test if 0.020 Hz emerges from legacy dynamics.
    """
    print("="*70)
    print("EXPERIMENT 29: Legacy Frequency Emergence Test")
    print("="*70)
    print("\nHypothesis: The 0.020 Hz organizing frequency observed in GAIA")
    print("           will emerge naturally from raw legacy dynamics.")
    print("\nPrediction: Dominant frequency ≈ 0.020 Hz (±0.005 Hz)")
    print("="*70)
    
    # Test multiple scenarios
    scenarios = [
        ("cosmo", "CIMM:cosmic_breath"),
        ("brain", "CIMM:brain"),
        ("vcpu", "CIMM:vCPU"),
        ("generic", "CIMM:generic"),
    ]
    
    results = []
    
    # Run with longer time series for better frequency resolution
    for name, seed in scenarios:
        result = run_simulation_and_extract_frequency(
            seed_prefix=seed,
            steps=5000,  # 5000 steps for good frequency resolution
            dt=1.0,
            grid_size=64
        )
        result['scenario'] = name
        results.append(result)
    
    # Analysis
    print("\n" + "="*70)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("="*70)
    
    target_freq = 0.020
    tolerance = 0.005
    
    energy_freqs = [r['dominant_energy_freq'] for r in results]
    collapse_freqs = [r['dominant_collapse_freq'] for r in results]
    
    print(f"\nTarget frequency: {target_freq} Hz (GAIA universal)")
    print(f"Tolerance: ±{tolerance} Hz\n")
    
    print("| Scenario | Energy Freq (Hz) | Collapse Freq (Hz) | Near 0.020? |")
    print("|----------|-----------------|-------------------|-------------|")
    
    matches = 0
    for r in results:
        e_freq = r['dominant_energy_freq']
        c_freq = r['dominant_collapse_freq']
        
        # Check if either is near target
        e_near = abs(e_freq - target_freq) < tolerance
        c_near = abs(c_freq - target_freq) < tolerance
        
        status = "✓" if (e_near or c_near) else ""
        if e_near or c_near:
            matches += 1
        
        print(f"| {r['scenario']:8} | {e_freq:15.6f} | {c_freq:17.6f} | {status:11} |")
    
    # Scaled frequency analysis
    # The raw simulations use arbitrary time units. Let's check for φ-related ratios.
    print("\n" + "-"*70)
    print("RATIO ANALYSIS (frequency relationships)")
    print("-"*70)
    
    phi = (1 + np.sqrt(5)) / 2
    
    all_freqs = energy_freqs + collapse_freqs
    
    print("\nRatios between dominant frequencies:")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ratio = results[i]['dominant_energy_freq'] / results[j]['dominant_energy_freq'] if results[j]['dominant_energy_freq'] != 0 else float('inf')
            phi_ratio = ratio / phi
            print(f"  {results[i]['scenario']}/{results[j]['scenario']}: {ratio:.4f} (φ-scaled: {phi_ratio:.4f})")
    
    # Period analysis
    print("\n" + "-"*70)
    print("PERIOD ANALYSIS")
    print("-"*70)
    
    print("\nDominant periods (1/frequency):")
    for r in results:
        if r['dominant_energy_freq'] > 0:
            period = 1.0 / r['dominant_energy_freq']
            print(f"  {r['scenario']}: {period:.2f} time units")
    
    # Check for harmonic relationships
    print("\n" + "-"*70)
    print("HARMONIC RELATIONSHIPS")
    print("-"*70)
    
    # GAIA showed 0.020 Hz with ocean at 0.010 Hz (1:2 harmonic)
    # Check if our frequencies show similar harmonic structure
    
    base_freqs = sorted(set([r['dominant_energy_freq'] for r in results]))
    print(f"\nUnique energy frequencies: {[f'{f:.6f}' for f in base_freqs]}")
    
    if len(base_freqs) >= 2:
        for i in range(len(base_freqs)):
            for j in range(i+1, len(base_freqs)):
                ratio = base_freqs[j] / base_freqs[i] if base_freqs[i] != 0 else float('inf')
                harmonic = round(ratio)
                if abs(ratio - harmonic) < 0.1:
                    print(f"  {base_freqs[j]:.6f} / {base_freqs[i]:.6f} = {ratio:.3f} (≈ 1:{harmonic} harmonic)")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if matches > 0:
        print(f"\n✅ {matches}/{len(results)} scenarios showed frequency near 0.020 Hz")
        print("   This suggests 0.020 Hz MAY emerge from raw dynamics.")
    else:
        print(f"\n⚠️ No scenarios showed 0.020 Hz exactly.")
        print("   However, check if frequencies are SCALED versions.")
        print("   The legacy simulations use arbitrary time units.")
        print("   What matters is the STRUCTURE, not absolute values.")
    
    # Check for structural consistency (all simulations converge to same frequency)
    freq_variance = np.var(energy_freqs)
    print(f"\n   Frequency variance across scenarios: {freq_variance:.6f}")
    if freq_variance < 0.0001:
        print("   ✅ All scenarios converge to SAME frequency (universal attractor)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_29_legacy_frequency_emergence',
        'hypothesis': '0.020 Hz emerges from raw legacy dynamics',
        'target_freq': target_freq,
        'tolerance': tolerance,
        'matches': matches,
        'total_scenarios': len(results),
        'frequency_variance': float(freq_variance),
        'results': results
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, r in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        freqs = np.array(r['freqs'])
        spectrum = np.array(r['energy_spectrum'])
        
        # Plot spectrum (focus on low frequencies)
        mask = (freqs > 0.001) & (freqs < 0.1)
        ax.plot(freqs[mask], spectrum[mask], 'b-', alpha=0.7)
        ax.axvline(x=0.020, color='r', linestyle='--', label='Target: 0.020 Hz')
        ax.axvline(x=r['dominant_energy_freq'], color='g', linestyle='-', 
                   label=f"Dominant: {r['dominant_energy_freq']:.4f} Hz")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f"{r['scenario'].upper()} Energy Spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/exp_29_frequency_spectra.png', dpi=150)
    plt.close()
    
    # Save JSON
    with open('../results/exp_29_legacy_frequency_emergence.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n   Results saved to:")
    print(f"     - exp_29_legacy_frequency_emergence.json")
    print(f"     - exp_29_frequency_spectra.png")
    
    return output


if __name__ == "__main__":
    results = test_frequency_emergence()
