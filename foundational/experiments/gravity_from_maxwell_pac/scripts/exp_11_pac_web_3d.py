#!/usr/bin/env python3
"""
exp_11_pac_web_3d.py

3D cosmic web simulation with LOCAL gravitational interactions.
GPU-accelerated with PyTorch + CUDA.

In 3D, we expect to see:
- Filaments (1D structures)
- Sheets (2D structures)
- Nodes (0D concentrations)
- Voids (3D empty regions)

This matches the observed cosmic web topology.

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from constants import PHI, XI, print_header, print_result

# =============================================================================
# DEVICE SETUP
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 3D PAC SYSTEM
# =============================================================================

class PACSystem3D:
    """GPU-accelerated 3D particle system with PAC/SEC dynamics."""
    
    def __init__(self, n_nodes: int, box_size: float, interaction_radius: float,
                 pac_strength: float, sec_balance: float, seed: int = 42):
        self.n_nodes = n_nodes
        self.box_size = box_size
        self.r0 = interaction_radius
        self.pac_strength = pac_strength
        self.sec_balance = sec_balance
        self.dt = 0.05
        
        torch.manual_seed(seed)
        
        # Initialize on 3D grid with perturbation
        n_per_dim = int(np.ceil(n_nodes ** (1/3)))
        spacing = box_size / n_per_dim
        
        grid = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing='ij')
        
        positions = torch.stack([xx.flatten()[:n_nodes], 
                                 yy.flatten()[:n_nodes], 
                                 zz.flatten()[:n_nodes]], dim=1)
        positions += torch.randn_like(positions) * spacing * 0.1
        self.positions = positions % box_size
        
        self.velocities = torch.zeros_like(self.positions)
        self.masses = 1.0 + 0.1 * torch.randn(n_nodes, device=DEVICE)
        self.entropy = 0.1 * torch.rand(n_nodes, device=DEVICE)
    
    def step(self):
        """Single simulation step."""
        n = self.n_nodes
        r0 = self.r0
        box = self.box_size
        
        # Pairwise distances in 3D
        pos_i = self.positions.unsqueeze(1)  # [N, 1, 3]
        pos_j = self.positions.unsqueeze(0)  # [1, N, 3]
        deltas = pos_i - pos_j               # [N, N, 3]
        deltas = deltas - box * torch.round(deltas / box)  # Periodic BC
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)  # [N, N]
        distances.fill_diagonal_(float('inf'))
        
        # Local gravity (exponential falloff)
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
        
        # Update velocities and positions
        total_force = gravity + pressure
        self.velocities = 0.99 * self.velocities + total_force * self.dt / self.masses.unsqueeze(1)
        
        speeds = torch.norm(self.velocities, dim=1, keepdim=True)
        self.velocities = torch.where(speeds > 2.0, self.velocities * 2.0 / speeds, self.velocities)
        
        self.positions = (self.positions + self.velocities * self.dt) % box
        
        # SEC dynamics
        local_count = torch.sum(distances < r0, dim=1).float()
        mean_count = local_count.mean()
        density_deviation = (local_count - mean_count) / (mean_count + 1)
        entropy_change = self.sec_balance * density_deviation
        self.entropy = torch.clamp(self.entropy + entropy_change, min=0.0)
    
    def compute_density_field(self, resolution: int = 32) -> torch.Tensor:
        """Compute 3D density field."""
        field = torch.zeros(resolution, resolution, resolution, device=DEVICE)
        
        ix = (self.positions[:, 0] / self.box_size * resolution).long() % resolution
        iy = (self.positions[:, 1] / self.box_size * resolution).long() % resolution
        iz = (self.positions[:, 2] / self.box_size * resolution).long() % resolution
        
        # 3D scatter
        for i in range(self.n_nodes):
            field[ix[i], iy[i], iz[i]] += self.masses[i]
        
        return field
    
    def compute_metrics(self) -> Dict:
        """Compute 3D structure metrics."""
        field = self.compute_density_field(32)
        field_np = field.cpu().numpy()
        
        mean_density = np.mean(field_np)
        
        # Void fraction (cells < 10% mean)
        void_fraction = np.sum(field_np < 0.1 * mean_density) / field_np.size
        
        # Filament fraction (cells > 80th percentile of non-zero)
        nonzero = field_np[field_np > 0]
        if len(nonzero) > 10:
            threshold = np.percentile(nonzero, 80)
            filament_fraction = np.sum(field_np > threshold) / field_np.size
        else:
            filament_fraction = 0.0
        
        # Density CV
        density_cv = np.std(field_np) / (mean_density + 1e-8)
        
        # Clustering (sampled)
        positions = self.positions.cpu().numpy()
        r0 = self.r0
        box = self.box_size
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
            'clustering': float(clustering),
            'max_entropy': float(self.entropy.max().item()),
            'mean_entropy': float(self.entropy.mean().item())
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_3d_web(system: PACSystem3D, save_path: Path = None):
    """Visualize the 3D web structure."""
    positions = system.positions.cpu().numpy()
    masses = system.masses.cpu().numpy()
    entropy = system.entropy.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 6))
    
    # Panel 1: 3D scatter
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Color by local density (entropy as proxy)
    colors = entropy / (entropy.max() + 1e-8)
    sizes = masses * 5
    
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                         c=colors, cmap='plasma', s=sizes, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D PAC Web Structure')
    fig.colorbar(scatter, ax=ax1, label='Entropy', shrink=0.5)
    
    # Panel 2: XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:, 0], positions[:, 1], c=colors, cmap='plasma', 
               s=sizes, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # Panel 3: Density histogram (3D)
    ax3 = fig.add_subplot(133)
    field = system.compute_density_field(32).cpu().numpy()
    
    # Flatten and plot histogram
    ax3.hist(field.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(field), color='red', linestyle='--', label=f'Mean: {np.mean(field):.2f}')
    ax3.axvline(x=np.percentile(field[field > 0], 80), color='orange', linestyle='--', 
               label=f'80th %ile: {np.percentile(field[field > 0], 80):.2f}')
    ax3.set_xlabel('Density')
    ax3.set_ylabel('Count')
    ax3.set_title('3D Density Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_3d_slices(system: PACSystem3D, save_path: Path = None):
    """Visualize slices through the 3D density field."""
    field = system.compute_density_field(32).cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Z slices
    n_slices = 8
    for i, ax in enumerate(axes.flat):
        z_idx = int(i * field.shape[2] / n_slices)
        slice_data = field[:, :, z_idx]
        
        im = ax.imshow(slice_data, origin='lower', cmap='inferno', 
                       vmin=0, vmax=np.percentile(field, 95))
        ax.set_title(f'Z = {z_idx}')
        ax.axis('off')
    
    plt.suptitle('Slices Through 3D Cosmic Web', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved slices to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("Experiment 11: 3D PAC Web Emergence")
    
    # Configuration
    # In 3D, material spreads more - need stronger gravity, lower SEC
    n_nodes = 4000  # More nodes in 3D
    box_size = 60.0  # Smaller box = higher density
    n_steps = 600
    interaction_radius = 10.0  # Larger interaction for 3D
    pac_strength = 1.5  # Stronger gravity
    sec_balance = XI / PHI  # Lower SEC for 3D (0.653)
    
    print(f"\n=== Configuration ===")
    print(f"Nodes: {n_nodes}")
    print(f"Box size: {box_size}")
    print(f"Interaction radius: {interaction_radius}")
    print(f"SEC balance: {sec_balance:.4f} (Xi)")
    print(f"Steps: {n_steps}")
    
    # Create system
    print(f"\nInitializing 3D system...")
    system = PACSystem3D(n_nodes, box_size, interaction_radius, 
                         pac_strength, sec_balance, seed=42)
    
    # Run simulation
    print(f"Running {n_steps} steps...")
    import time
    start = time.time()
    
    for step in range(n_steps):
        system.step()
        if step % 100 == 0:
            metrics = system.compute_metrics()
            print(f"  Step {step}: voids={metrics['void_fraction']:.2f}, entropy={metrics['max_entropy']:.1f}")
    
    elapsed = time.time() - start
    print(f"\nSimulation completed in {elapsed:.1f}s")
    
    # Final metrics
    metrics = system.compute_metrics()
    
    print(f"\n=== Final Metrics ===")
    print(f"Void fraction: {metrics['void_fraction']:.3f}")
    print(f"Filament fraction: {metrics['filament_fraction']:.3f}")
    print(f"Density CV: {metrics['density_cv']:.3f}")
    print(f"Clustering: {metrics['clustering']:.3f}")
    print(f"Max entropy: {metrics['max_entropy']:.1f}")
    
    # Check for web structure (adjusted for 3D topology)
    # 3D webs have higher void fraction naturally
    is_web = (metrics['filament_fraction'] > 0.01 and 
              metrics['void_fraction'] > 0.5 and 
              metrics['density_cv'] > 1.0 and
              metrics['clustering'] > 0.3)
    
    print_result(
        "3D Web structure emerged",
        is_web,
        f"Filaments={metrics['filament_fraction']:.2f}, Voids={metrics['void_fraction']:.2f}"
    )
    
    # Visualize
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    viz_path = results_dir / f'exp_11_3d_web_{timestamp}.png'
    visualize_3d_web(system, viz_path)
    
    slices_path = results_dir / f'exp_11_3d_slices_{timestamp}.png'
    visualize_3d_slices(system, slices_path)
    
    # Save results
    results = {
        'experiment': 'exp_11_pac_web_3d',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_nodes': n_nodes,
            'box_size': box_size,
            'interaction_radius': interaction_radius,
            'pac_strength': pac_strength,
            'sec_balance': sec_balance,
            'n_steps': n_steps
        },
        'metrics': metrics,
        'elapsed_seconds': elapsed,
        'is_web': is_web
    }
    
    results_file = results_dir / f'exp_11_3d_web_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
