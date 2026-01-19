#!/usr/bin/env python3
"""
exp_10_phase_transition_sweep.py

Parameter sweep to find the CLUMP → WEB phase transition.

Hypothesis: There is a critical SEC balance value where the system
transitions from gravitational collapse (clump) to web structure.

This is analogous to:
- Percolation threshold in network theory
- Critical temperature in phase transitions
- Edge of chaos in cellular automata

We expect the critical point to show:
- Maximum structure complexity
- φ-related ratios at criticality
- Power-law scaling near transition

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from constants import PHI, XI, print_header, print_result

# XI = 1.0571428571428572 - the SEC balance constant from oscillation_attractor_dynamics

# =============================================================================
# DEVICE SETUP
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    n_nodes: int = 1500  # Fewer nodes for speed
    box_size: float = 100.0
    dt: float = 0.05
    n_steps: int = 600  # Longer to develop structure
    interaction_radius: float = 6.0
    pac_strength: float = 0.8
    memory_decay: float = 0.95
    seed: int = 42


# =============================================================================
# FAST GPU SIMULATION (simplified for sweep)
# =============================================================================

class FastPACSystem:
    """Optimized PAC system for parameter sweeps."""
    
    def __init__(self, config: SweepConfig, sec_balance: float):
        self.config = config
        self.sec_balance = sec_balance
        torch.manual_seed(config.seed)
        
        n = config.n_nodes
        n_per_dim = int(np.ceil(n ** 0.5))
        spacing = config.box_size / n_per_dim
        
        # Grid initialization
        grid_x = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        grid_y = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        positions = torch.stack([xx.flatten()[:n], yy.flatten()[:n]], dim=1)
        positions += torch.randn_like(positions) * spacing * 0.1
        positions = positions % config.box_size
        
        self.positions = positions
        self.velocities = torch.zeros_like(positions)
        self.masses = 1.0 + 0.1 * torch.randn(n, device=DEVICE)
        # Initialize with small random entropy to break symmetry
        self.entropy = 0.1 * torch.rand(n, device=DEVICE)
    
    def step(self):
        """Single simulation step."""
        n = self.config.n_nodes
        r0 = self.config.interaction_radius
        box = self.config.box_size
        
        # Pairwise distances
        pos_i = self.positions.unsqueeze(1)
        pos_j = self.positions.unsqueeze(0)
        deltas = pos_i - pos_j
        deltas = deltas - box * torch.round(deltas / box)
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)
        distances.fill_diagonal_(float('inf'))
        
        # Local gravity
        cutoff = 3 * r0
        mask = distances < cutoff
        mass_prod = self.masses.unsqueeze(1) * self.masses.unsqueeze(0)
        
        force_mag = torch.where(
            mask,
            self.config.pac_strength * mass_prod * torch.exp(-distances / r0) / (distances + 0.1),
            torch.zeros_like(distances)
        )
        force_dir = deltas / (distances.unsqueeze(2) + 0.1)
        gravity = torch.sum(force_mag.unsqueeze(2) * force_dir, dim=1)
        
        # Entropy pressure
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
        
        # Update
        total_force = gravity + pressure
        self.velocities = 0.99 * self.velocities + total_force * self.config.dt / self.masses.unsqueeze(1)
        
        speeds = torch.norm(self.velocities, dim=1, keepdim=True)
        self.velocities = torch.where(speeds > 2.0, self.velocities * 2.0 / speeds, self.velocities)
        
        self.positions = (self.positions + self.velocities * self.config.dt) % box
        
        # SEC dynamics - entropy grows/shrinks based on local density deviation
        local_count = torch.sum(distances < r0, dim=1).float()
        
        # Use actual mean as baseline (not theoretical - that's wrong for discrete grid)
        mean_count = local_count.mean()
        
        # Entropy grows when denser than average, decays when sparser
        # Higher SEC balance = stronger density response
        density_deviation = (local_count - mean_count) / (mean_count + 1)
        entropy_change = self.sec_balance * density_deviation
        
        # Apply: positive deviation increases entropy, negative decreases
        self.entropy = torch.clamp(self.entropy + entropy_change, min=0.0)
    
    def compute_metrics(self) -> Dict:
        """Compute structure metrics."""
        # Density field - use higher resolution
        resolution = 80
        field = torch.zeros(resolution, resolution, device=DEVICE)
        ix = (self.positions[:, 0] / self.config.box_size * resolution).long() % resolution
        iy = (self.positions[:, 1] / self.config.box_size * resolution).long() % resolution
        indices = iy * resolution + ix
        field_flat = field.view(-1)
        field_flat.scatter_add_(0, indices, self.masses)
        field = field.view(resolution, resolution)
        
        field_np = field.cpu().numpy()
        
        # Void fraction (cells with very low density)
        mean_density = np.mean(field_np)
        void_fraction = np.sum(field_np < 0.1 * mean_density) / field_np.size
        
        # Filament fraction (high density cells)
        if mean_density > 0:
            threshold = np.percentile(field_np[field_np > 0], 80) if np.sum(field_np > 0) > 10 else mean_density * 2
            filament_fraction = np.sum(field_np > threshold) / field_np.size
        else:
            filament_fraction = 0.0
        
        # Density CV
        density_cv = np.std(field_np) / (mean_density + 1e-8)
        
        # Clustering (sampled)
        positions = self.positions.cpu().numpy()
        r0 = self.config.interaction_radius
        box = self.config.box_size
        n_sample = min(200, len(positions))
        
        total_triangles = 0
        total_possible = 0
        
        for i in range(n_sample):
            delta = positions - positions[i]
            delta = delta - box * np.round(delta / box)
            distances = np.sqrt(np.sum(delta**2, axis=1))
            neighbors = np.where((distances > 0) & (distances < r0))[0]
            k = len(neighbors)
            if k < 2:
                continue
            
            for j_idx, j in enumerate(neighbors):
                for l in neighbors[j_idx+1:]:
                    d_jl = positions[l] - positions[j]
                    d_jl = d_jl - box * np.round(d_jl / box)
                    if np.linalg.norm(d_jl) < r0:
                        total_triangles += 1
            total_possible += k * (k - 1) / 2
        
        clustering = total_triangles / total_possible if total_possible > 0 else 0.0
        
        return {
            'void_fraction': float(void_fraction),
            'filament_fraction': float(filament_fraction),
            'density_cv': float(density_cv),
            'clustering': float(clustering)
        }


def run_single_simulation(config: SweepConfig, sec_balance: float, run_idx: int = 0) -> Dict:
    """Run a single simulation with given SEC balance."""
    # Use SAME initial conditions but different physics (sec_balance)
    torch.manual_seed(config.seed)  # Same start for fair comparison
    system = FastPACSystem(config, sec_balance)
    
    for step in range(config.n_steps):
        system.step()
        # Debug every 100 steps
        if step == 100 and run_idx == 0:
            print(f"\n  DEBUG step 100: entropy range [{system.entropy.min():.3f}, {system.entropy.max():.3f}]")
    
    # Debug: check max entropy reached
    max_ent = system.entropy.max().item()
    mean_ent = system.entropy.mean().item()
    
    metrics = system.compute_metrics()
    metrics['sec_balance'] = sec_balance
    metrics['max_entropy'] = max_ent
    metrics['mean_entropy'] = mean_ent
    
    # Determine structure type - use filament > 0.04 (not 0.05)
    is_web = (metrics['filament_fraction'] > 0.04 and 
              metrics['void_fraction'] > 0.3 and 
              metrics['density_cv'] > 0.8)
    metrics['structure_type'] = 'web' if is_web else 'clump'
    
    return metrics


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def run_sweep(sec_values: np.ndarray, config: SweepConfig) -> List[Dict]:
    """Run parameter sweep over SEC balance values."""
    results = []
    
    for i, sec in enumerate(sec_values):
        print(f"  [{i+1}/{len(sec_values)}] SEC={sec:.3f} ... ", end='', flush=True)
        metrics = run_single_simulation(config, sec, i)
        max_ent = metrics.get('max_entropy', 0)
        print(f"→ {metrics['structure_type'].upper()} (voids={metrics['void_fraction']:.2f}, entropy={max_ent:.1f})")
        results.append(metrics)
    
    return results


def find_critical_point(results: List[Dict]) -> Tuple[float, Dict]:
    """Find the critical SEC balance where phase transition occurs."""
    # Look for maximum density CV (complexity peak at transition)
    max_cv = 0
    critical_idx = 0
    
    for i, r in enumerate(results):
        if r['density_cv'] > max_cv:
            max_cv = r['density_cv']
            critical_idx = i
    
    # Also find transition point (clump → web)
    transition_idx = None
    for i in range(1, len(results)):
        if results[i-1]['structure_type'] == 'clump' and results[i]['structure_type'] == 'web':
            transition_idx = i
            break
    
    if transition_idx is not None:
        return results[transition_idx]['sec_balance'], results[transition_idx]
    return results[critical_idx]['sec_balance'], results[critical_idx]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sweep_results(results: List[Dict], save_path: Path = None):
    """Plot the parameter sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sec_values = [r['sec_balance'] for r in results]
    
    # Panel 1: Void fraction vs SEC
    ax1 = axes[0, 0]
    ax1.plot(sec_values, [r['void_fraction'] for r in results], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Web threshold')
    ax1.axhline(y=1/PHI, color='gold', linestyle='--', alpha=0.7, label=f'1/φ = {1/PHI:.3f}')
    ax1.set_xlabel('SEC Balance')
    ax1.set_ylabel('Void Fraction')
    ax1.set_title('Void Fraction vs SEC Balance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Filament fraction vs SEC
    ax2 = axes[0, 1]
    ax2.plot(sec_values, [r['filament_fraction'] for r in results], 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Web threshold')
    ax2.set_xlabel('SEC Balance')
    ax2.set_ylabel('Filament Fraction')
    ax2.set_title('Filament Fraction vs SEC Balance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Density CV (complexity) vs SEC
    ax3 = axes[1, 0]
    cvs = [r['density_cv'] for r in results]
    ax3.plot(sec_values, cvs, 'm-o', linewidth=2, markersize=6)
    
    # Mark maximum (critical point)
    max_idx = np.argmax(cvs)
    ax3.axvline(x=sec_values[max_idx], color='red', linestyle=':', linewidth=2, 
                label=f'Max complexity at SEC={sec_values[max_idx]:.2f}')
    ax3.scatter([sec_values[max_idx]], [cvs[max_idx]], color='red', s=150, zorder=5, marker='*')
    
    ax3.set_xlabel('SEC Balance')
    ax3.set_ylabel('Density CV (Complexity)')
    ax3.set_title('Structure Complexity vs SEC Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Clustering vs SEC
    ax4 = axes[1, 1]
    ax4.plot(sec_values, [r['clustering'] for r in results], 'c-o', linewidth=2, markersize=6)
    ax4.axhline(y=1/PHI, color='gold', linestyle='--', alpha=0.7, label=f'1/φ = {1/PHI:.3f}')
    ax4.axhline(y=1/2, color='orange', linestyle='--', alpha=0.5, label='1/2')
    ax4.set_xlabel('SEC Balance')
    ax4.set_ylabel('Clustering Coefficient')
    ax4.set_title('Clustering vs SEC Balance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Color background by structure type
    for ax in axes.flat:
        for i, r in enumerate(results[:-1]):
            color = 'lightgreen' if r['structure_type'] == 'web' else 'lightyellow'
            ax.axvspan(sec_values[i], sec_values[i+1], alpha=0.1, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_phase_diagram(results: List[Dict], save_path: Path = None):
    """Plot a phase diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sec_values = [r['sec_balance'] for r in results]
    void_frac = [r['void_fraction'] for r in results]
    filament_frac = [r['filament_fraction'] for r in results]
    
    # Color by structure type
    colors = ['red' if r['structure_type'] == 'clump' else 'blue' for r in results]
    sizes = [r['density_cv'] * 100 + 50 for r in results]
    
    scatter = ax.scatter(void_frac, filament_frac, c=sec_values, s=sizes, 
                        cmap='viridis', edgecolors='black', linewidths=1)
    
    # Add arrows showing evolution with SEC
    for i in range(len(results)-1):
        dx = void_frac[i+1] - void_frac[i]
        dy = filament_frac[i+1] - filament_frac[i]
        ax.annotate('', xy=(void_frac[i+1], filament_frac[i+1]),
                   xytext=(void_frac[i], filament_frac[i]),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3))
    
    # Mark regions
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Web boundary')
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    
    ax.text(0.1, 0.2, 'CLUMP\nRegion', fontsize=12, ha='center', color='red', alpha=0.7)
    ax.text(0.6, 0.15, 'WEB\nRegion', fontsize=12, ha='center', color='blue', alpha=0.7)
    
    cbar = plt.colorbar(scatter, label='SEC Balance')
    ax.set_xlabel('Void Fraction', fontsize=12)
    ax.set_ylabel('Filament Fraction', fontsize=12)
    ax.set_title('Phase Diagram: Clump → Web Transition', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase diagram to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("Experiment 10: Phase Transition Sweep")
    
    config = SweepConfig(
        n_nodes=2000,
        box_size=100.0,
        n_steps=400,
        interaction_radius=6.0,
        pac_strength=0.8,
        memory_decay=0.95,
        seed=42
    )
    
    # Fine sweep around 1/φ ≈ 0.618 and Ξ ≈ 1.057
    sec_values = np.linspace(0.3, 1.3, 21)
    
    print(f"\n=== Configuration ===")
    print(f"Nodes: {config.n_nodes}")
    print(f"Steps per run: {config.n_steps}")
    print(f"SEC values to test: {len(sec_values)}")
    print(f"Range: {sec_values[0]:.2f} to {sec_values[-1]:.2f}")
    
    print(f"\n=== Running Sweep ===")
    import time
    start = time.time()
    results = run_sweep(sec_values, config)
    elapsed = time.time() - start
    print(f"\nSweep completed in {elapsed:.1f}s ({elapsed/len(sec_values):.1f}s per run)")
    
    # Find critical point
    critical_sec, critical_metrics = find_critical_point(results)
    
    print(f"\n=== Critical Point Analysis ===")
    print(f"Critical SEC balance: {critical_sec:.3f}")
    print(f"At transition:")
    print(f"  Void fraction: {critical_metrics['void_fraction']:.3f}")
    print(f"  Filament fraction: {critical_metrics['filament_fraction']:.3f}")
    print(f"  Density CV: {critical_metrics['density_cv']:.3f}")
    print(f"  Clustering: {critical_metrics['clustering']:.3f}")
    
    # Check for φ-related ratios
    print(f"\n=== Fibonacci/φ/Ξ Analysis ===")
    print(f"Ξ (Xi) balance constant = {XI:.6f}")
    print(f"1/φ = {1/PHI:.4f}")
    print(f"1/φ² = {1/PHI**2:.4f}")
    print(f"2/3 (F₃/F₄) = {2/3:.4f}")
    print(f"Critical SEC = {critical_sec:.4f}")
    print(f"Critical SEC / Ξ = {critical_sec/XI:.4f}")
    
    # Check if critical point relates to Ξ
    xi_ratio = critical_sec / XI
    print_result(
        f"Critical SEC near Ξ (1.057)",
        abs(xi_ratio - 1) < 0.2,
        f"Ratio = {xi_ratio:.3f}"
    )
    
    # Check if critical point relates to φ
    phi_ratio = critical_sec / (1/PHI)
    print_result(
        f"Critical SEC near 1/φ",
        abs(phi_ratio - 1) < 0.3,
        f"Ratio = {phi_ratio:.3f}"
    )
    
    # Visualize
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_path = results_dir / f'exp_10_sweep_{timestamp}.png'
    plot_sweep_results(results, plot_path)
    
    phase_path = results_dir / f'exp_10_phase_diagram_{timestamp}.png'
    plot_phase_diagram(results, phase_path)
    
    # Save results
    output = {
        'experiment': 'exp_10_phase_transition_sweep',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_nodes': config.n_nodes,
            'n_steps': config.n_steps,
            'interaction_radius': config.interaction_radius,
            'pac_strength': config.pac_strength
        },
        'critical_point': {
            'sec_balance': critical_sec,
            'metrics': critical_metrics
        },
        'sweep_results': results,
        'analysis': {
            'phi': PHI,
            'critical_sec_over_inv_phi': critical_sec * PHI
        }
    }
    
    results_file = results_dir / f'exp_10_sweep_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
