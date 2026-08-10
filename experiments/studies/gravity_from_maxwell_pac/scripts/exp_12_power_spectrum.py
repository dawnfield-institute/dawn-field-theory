#!/usr/bin/env python3
"""
exp_12_power_spectrum.py

Power spectrum analysis of PAC web structures.
Tests if the emergent structure is scale-free (fractal-like).

Real cosmic structures show power-law scaling P(k) ∝ k^n
- n ≈ 1 for primordial fluctuations
- n ≈ -1.5 for observed matter power spectrum (on large scales)

If LOCAL PAC gravity produces scale-free structure, we have evidence
that Newtonian 1/r² is not required - locality + SEC suffices.

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import ndimage
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from constants import PHI, XI, print_header, print_result

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# PAC SYSTEM (2D for cleaner power spectrum)
# =============================================================================

class PACSystemPowerSpec:
    """2D PAC system optimized for power spectrum analysis."""
    
    def __init__(self, n_nodes: int, box_size: float, interaction_radius: float,
                 pac_strength: float, sec_balance: float, seed: int = 42):
        self.n_nodes = n_nodes
        self.box_size = box_size
        self.r0 = interaction_radius
        self.pac_strength = pac_strength
        self.sec_balance = sec_balance
        self.dt = 0.05
        
        torch.manual_seed(seed)
        
        # Grid initialization with perturbation
        n_per_dim = int(np.ceil(np.sqrt(n_nodes)))
        actual_nodes = n_per_dim * n_per_dim  # Use full grid
        self.n_nodes = actual_nodes
        
        spacing = box_size / n_per_dim
        grid = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        xx, yy = torch.meshgrid(grid, grid, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        positions += torch.randn_like(positions) * spacing * 0.1
        self.positions = positions % box_size
        
        self.velocities = torch.zeros_like(self.positions)
        self.masses = 1.0 + 0.1 * torch.randn(actual_nodes, device=DEVICE)
        self.entropy = 0.1 * torch.rand(actual_nodes, device=DEVICE)
    
    def step(self):
        n = self.n_nodes
        r0 = self.r0
        box = self.box_size
        
        pos_i = self.positions.unsqueeze(1)
        pos_j = self.positions.unsqueeze(0)
        deltas = pos_i - pos_j
        deltas = deltas - box * torch.round(deltas / box)
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)
        distances.fill_diagonal_(float('inf'))
        
        cutoff = 3 * r0
        mask = distances < cutoff
        mass_prod = self.masses.unsqueeze(1) * self.masses.unsqueeze(0)
        
        force_mag = torch.where(
            mask,
            self.pac_strength * mass_prod * torch.exp(-distances / r0) / (distances + 0.1),
            torch.zeros_like(distances)
        )
        force_dir = deltas / (distances.unsqueeze(2) + 0.1)
        gravity = torch.sum(force_mag.unsqueeze(2) * force_dir, dim=1)
        
        entropy_i = self.entropy.unsqueeze(1)
        entropy_j = self.entropy.unsqueeze(0)
        entropy_diff = entropy_i - entropy_j
        
        pressure_mag = torch.where(
            distances < 2 * r0,
            self.sec_balance * entropy_diff * torch.exp(-distances / r0),
            torch.zeros_like(distances)
        )
        pressure_dir = -deltas / (distances.unsqueeze(2) + 0.1)
        pressure = torch.sum(pressure_mag.unsqueeze(2) * pressure_dir, dim=1)
        
        total_force = gravity + pressure
        self.velocities = 0.99 * self.velocities + total_force * self.dt / self.masses.unsqueeze(1)
        
        speeds = torch.norm(self.velocities, dim=1, keepdim=True)
        self.velocities = torch.where(speeds > 2.0, self.velocities * 2.0 / speeds, self.velocities)
        
        self.positions = (self.positions + self.velocities * self.dt) % box
        
        local_count = torch.sum(distances < r0, dim=1).float()
        mean_count = local_count.mean()
        density_deviation = (local_count - mean_count) / (mean_count + 1)
        entropy_change = self.sec_balance * density_deviation
        self.entropy = torch.clamp(self.entropy + entropy_change, min=0.0)
    
    def compute_density_field(self, resolution: int = 128) -> np.ndarray:
        """Compute high-resolution density field for FFT."""
        field = torch.zeros(resolution, resolution, device=DEVICE)
        
        ix = (self.positions[:, 0] / self.box_size * resolution).long() % resolution
        iy = (self.positions[:, 1] / self.box_size * resolution).long() % resolution
        
        # Use actual number of particles in system
        n_actual = min(len(self.masses), len(ix))
        for i in range(n_actual):
            field[ix[i], iy[i]] += self.masses[i]
        
        # Smooth for better FFT
        field_np = field.cpu().numpy()
        field_np = ndimage.gaussian_filter(field_np, sigma=1.0)
        
        return field_np


def compute_power_spectrum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged power spectrum of density field."""
    
    # FFT
    fft = np.fft.fft2(field)
    power = np.abs(fft) ** 2
    power = np.fft.fftshift(power)
    
    # Radial averaging
    ny, nx = field.shape
    y, x = np.ogrid[:ny, :nx]
    center = (ny // 2, nx // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Radial bins
    max_r = int(np.sqrt(center[0]**2 + center[1]**2))
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    
    radial_power = tbin / (nr + 1e-8)
    
    # Frequency scale
    k = np.arange(len(radial_power))
    k[0] = 0.1  # Avoid log(0)
    
    # Return valid range
    valid = k[1:max_r//2]
    power_valid = radial_power[1:max_r//2]
    
    return valid, power_valid


def fit_power_law(k: np.ndarray, power: np.ndarray) -> Tuple[float, float, float]:
    """Fit power law P(k) ∝ k^n in log-log space."""
    
    # Filter out zeros and negatives
    valid = (k > 0) & (power > 0)
    log_k = np.log10(k[valid])
    log_power = np.log10(power[valid])
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_power)
    
    return slope, r_value**2, std_err


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("Experiment 12: Power Spectrum Analysis")
    
    # Configuration
    n_nodes = 5000
    box_size = 100.0
    n_steps = 500
    interaction_radius = 8.0
    pac_strength = 0.8
    sec_balance = XI
    resolution = 128  # For FFT
    
    print(f"\n=== Configuration ===")
    print(f"Nodes: {n_nodes}")
    print(f"Box size: {box_size}")
    print(f"FFT resolution: {resolution}")
    print(f"SEC balance: {sec_balance:.4f}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    system = PACSystemPowerSpec(n_nodes, box_size, interaction_radius, 
                                pac_strength, sec_balance, seed=42)
    
    import time
    start = time.time()
    
    # Track power spectrum evolution
    power_history = []
    checkpoints = [0, 100, 250, 500]
    
    for step in range(n_steps + 1):
        if step in checkpoints:
            field = system.compute_density_field(resolution)
            k, power = compute_power_spectrum(field)
            slope, r2, stderr = fit_power_law(k, power)
            power_history.append({
                'step': step,
                'k': k.tolist(),
                'power': power.tolist(),
                'slope': slope,
                'r2': r2
            })
            print(f"  Step {step}: slope = {slope:.3f}, R² = {r2:.3f}")
        
        if step < n_steps:
            system.step()
    
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Final analysis
    final = power_history[-1]
    
    print(f"\n=== Power Spectrum Analysis ===")
    print(f"Final power law slope: {final['slope']:.3f}")
    print(f"R² (goodness of fit): {final['r2']:.3f}")
    
    # Interpret
    slope = final['slope']
    if -2.5 < slope < -0.5:
        interpretation = "SCALE-FREE (power-law)"
        is_fractal = True
    elif slope < -2.5:
        interpretation = "STEEP (concentrated at small scales)"
        is_fractal = False
    else:
        interpretation = "SHALLOW (concentrated at large scales)"
        is_fractal = False
    
    print(f"Interpretation: {interpretation}")
    
    print_result(
        "Scale-free (fractal-like) structure",
        is_fractal,
        f"slope={slope:.2f}, n ∈ [-2.5, -0.5] expected"
    )
    
    # Compare to cosmic matter power spectrum
    # Observed: n ≈ -1.5 on large scales
    cosmic_similarity = 1.0 - abs(slope - (-1.5)) / 1.5
    print_result(
        "Similar to cosmic matter spectrum (n ≈ -1.5)",
        0.3 < cosmic_similarity < 1.0,
        f"similarity = {cosmic_similarity:.2f}"
    )
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Final density field
    final_field = system.compute_density_field(resolution)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(final_field, origin='lower', cmap='inferno')
    ax1.set_title('Final Density Field')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # Panel 2: Power spectrum evolution
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(power_history)))
    
    for i, ph in enumerate(power_history):
        ax2.loglog(ph['k'], ph['power'], color=colors[i], 
                   label=f"Step {ph['step']}: n={ph['slope']:.2f}")
    
    ax2.set_xlabel('k (wavenumber)')
    ax2.set_ylabel('P(k) (power)')
    ax2.set_title('Power Spectrum Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Final power spectrum with fit
    ax3 = axes[1, 0]
    k = np.array(final['k'])
    power = np.array(final['power'])
    
    ax3.loglog(k, power, 'b-', linewidth=2, label='PAC web')
    
    # Fit line
    log_k = np.log10(k[(k > 0) & (power > 0)])
    slope_final = final['slope']
    fit_power = 10**(slope_final * log_k + np.log10(power[1]))
    ax3.loglog(10**log_k, fit_power, 'r--', linewidth=2, 
               label=f'Fit: P ∝ k^{slope_final:.2f}')
    
    # Reference slopes
    ax3.loglog(k, k**(-1.5) * power[10] / k[10]**(-1.5), 'g:', alpha=0.5, 
               label='n = -1.5 (cosmic)')
    ax3.loglog(k, k**(-1.0) * power[10] / k[10]**(-1.0), 'm:', alpha=0.5, 
               label='n = -1.0')
    
    ax3.set_xlabel('k (wavenumber)')
    ax3.set_ylabel('P(k) (power)')
    ax3.set_title('Power Spectrum with Fit')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Slope evolution
    ax4 = axes[1, 1]
    steps = [ph['step'] for ph in power_history]
    slopes = [ph['slope'] for ph in power_history]
    r2s = [ph['r2'] for ph in power_history]
    
    ax4.plot(steps, slopes, 'b-o', linewidth=2, markersize=8, label='Slope')
    ax4.axhline(y=-1.5, color='g', linestyle='--', alpha=0.5, label='Cosmic n=-1.5')
    ax4.axhline(y=-1.0, color='m', linestyle='--', alpha=0.5, label='Scale-free n=-1')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(steps, r2s, 'r-s', linewidth=2, markersize=8, label='R²')
    ax4_twin.set_ylabel('R² (goodness of fit)', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Power Law Slope (n)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4.set_title('Power Spectrum Evolution')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Power Spectrum Analysis: LOCAL PAC Gravity (SEC={sec_balance:.3f})', 
                 fontsize=14)
    plt.tight_layout()
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig_path = results_dir / f'exp_12_power_spectrum_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {fig_path}")
    
    plt.show()
    
    # Save results
    results = {
        'experiment': 'exp_12_power_spectrum',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_nodes': n_nodes,
            'box_size': box_size,
            'resolution': resolution,
            'sec_balance': sec_balance,
            'n_steps': n_steps
        },
        'final_analysis': {
            'slope': final['slope'],
            'r2': final['r2'],
            'interpretation': interpretation,
            'is_scale_free': is_fractal,
            'cosmic_similarity': cosmic_similarity
        },
        'evolution': power_history,
        'elapsed_seconds': elapsed
    }
    
    results_file = results_dir / f'exp_12_power_spectrum_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
