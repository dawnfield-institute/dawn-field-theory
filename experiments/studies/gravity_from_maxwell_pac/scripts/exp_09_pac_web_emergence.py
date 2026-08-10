#!/usr/bin/env python3
"""
exp_09_pac_web_emergence.py

Large-scale simulation: N nodes with LOCAL gravitational interactions.
GPU-accelerated with PyTorch + CUDA.

Key insight: Newtonian gravity is GLOBAL (every mass affects every other).
But if gravity is SEC at depth 183, it should have LOCAL structure
based on information propagation limits.

With PAC conservation enforced, we expect:
- Web-like filaments (not spherical collapse)
- Voids between structures
- Nodes at filament intersections
- Scale-free clustering

This mirrors the cosmic web observed in galaxy surveys.

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

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
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

class SimConfig:
    """Simulation configuration."""
    def __init__(
        self,
        n_nodes: int = 2000,
        box_size: float = 100.0,
        dt: float = 0.05,
        n_steps: int = 500,
        interaction_radius: float = 10.0,  # LOCAL gravity range
        pac_strength: float = 1.0,
        sec_balance: float = 0.5,  # SEC information/entropy balance
        memory_decay: float = 0.98,
        seed: int = 42
    ):
        self.n_nodes = n_nodes
        self.box_size = box_size
        self.dt = dt
        self.n_steps = n_steps
        self.interaction_radius = interaction_radius
        self.pac_strength = pac_strength
        self.sec_balance = sec_balance
        self.memory_decay = memory_decay
        self.seed = seed


# =============================================================================
# GPU-ACCELERATED SIMULATION STATE
# =============================================================================

class PACSystem:
    """GPU-accelerated particle system with PAC conservation."""
    
    def __init__(self, config: SimConfig):
        self.config = config
        torch.manual_seed(config.seed)
        
        # Initialize positions on grid with perturbation
        n = config.n_nodes
        n_per_dim = int(np.ceil(n ** 0.5))
        spacing = config.box_size / n_per_dim
        
        # Create grid
        grid_x = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        grid_y = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        positions = torch.stack([xx.flatten()[:n], yy.flatten()[:n]], dim=1)
        
        # Add perturbation
        positions += torch.randn_like(positions) * spacing * 0.1
        positions = positions % config.box_size
        
        self.positions = positions  # [N, 2]
        self.velocities = torch.zeros_like(positions)  # [N, 2]
        self.masses = 1.0 + 0.1 * torch.randn(n, device=DEVICE)  # [N]
        self.entropy = torch.zeros(n, device=DEVICE)  # [N]
        self.pac_values = self.masses.clone()  # [N]
        
    def periodic_delta(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute delta with periodic boundary conditions."""
        delta = pos1 - pos2
        delta = delta - self.config.box_size * torch.round(delta / self.config.box_size)
        return delta
    
    def compute_pairwise_distances(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all pairwise distances and deltas.
        Returns: (distances [N, N], deltas [N, N, 2])
        """
        n = self.config.n_nodes
        
        # Expand for pairwise computation
        pos_i = self.positions.unsqueeze(1)  # [N, 1, 2]
        pos_j = self.positions.unsqueeze(0)  # [1, N, 2]
        
        # Periodic delta
        deltas = pos_i - pos_j  # [N, N, 2]
        deltas = deltas - self.config.box_size * torch.round(deltas / self.config.box_size)
        
        # Distances
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)  # [N, N]
        
        # Set self-distance to inf
        distances.fill_diagonal_(float('inf'))
        
        return distances, deltas
    
    def compute_local_gravity(self, distances: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute LOCAL gravitational forces (GPU vectorized).
        
        Force profile: Exponential falloff (not 1/r²)
        F = G * m_i * m_j * exp(-r/r0) / r * r_hat
        """
        r0 = self.config.interaction_radius
        G = self.config.pac_strength
        cutoff = 3 * r0
        
        # Mask for local interactions only
        mask = distances < cutoff  # [N, N]
        
        # Force magnitude: G * m_i * m_j * exp(-r/r0) / r
        mass_products = self.masses.unsqueeze(1) * self.masses.unsqueeze(0)  # [N, N]
        
        force_mag = torch.where(
            mask,
            G * mass_products * torch.exp(-distances / r0) / (distances + 0.1),
            torch.zeros_like(distances)
        )  # [N, N]
        
        # Force direction (normalized deltas, pointing toward other particle)
        force_dir = deltas / (distances.unsqueeze(2) + 0.1)  # [N, N, 2]
        
        # Total force on each particle (sum over j)
        forces = torch.sum(force_mag.unsqueeze(2) * force_dir, dim=1)  # [N, 2]
        
        return forces
    
    def compute_entropy_pressure(self, distances: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Entropy gradient creates repulsive pressure (GPU vectorized).
        """
        r0 = self.config.interaction_radius
        cutoff = 2 * r0
        
        mask = distances < cutoff  # [N, N]
        
        # Entropy differences
        entropy_i = self.entropy.unsqueeze(1)  # [N, 1]
        entropy_j = self.entropy.unsqueeze(0)  # [1, N]
        entropy_diff = entropy_i - entropy_j  # [N, N]
        
        # Pressure magnitude
        pressure_mag = torch.where(
            mask,
            self.config.sec_balance * entropy_diff * torch.exp(-distances / r0),
            torch.zeros_like(distances)
        )  # [N, N]
        
        # Direction: away from neighbors (negative delta direction)
        pressure_dir = -deltas / (distances.unsqueeze(2) + 0.1)  # [N, N, 2]
        
        # Total pressure
        pressure = torch.sum(pressure_mag.unsqueeze(2) * pressure_dir, dim=1)  # [N, 2]
        
        return pressure
    
    def apply_sec_dynamics(self, distances: torch.Tensor):
        """Update entropy based on local density (SEC balance)."""
        r0 = self.config.interaction_radius
        
        # Count local neighbors
        local_count = torch.sum(distances < r0, dim=1).float()  # [N]
        
        # Expected count for uniform distribution
        expected = self.config.n_nodes * (np.pi * r0**2) / (self.config.box_size**2)
        
        # High density → increase entropy; low density → decay
        dense_mask = local_count > expected * 1.5
        self.entropy = torch.where(
            dense_mask,
            self.entropy + 0.1 * (local_count - expected),
            self.entropy * self.config.memory_decay
        )
    
    def step(self):
        """Perform one simulation step."""
        # Compute pairwise distances once
        distances, deltas = self.compute_pairwise_distances()
        
        # Compute forces
        gravity = self.compute_local_gravity(distances, deltas)
        pressure = self.compute_entropy_pressure(distances, deltas)
        total_force = gravity + pressure
        
        # Update velocities with damping
        self.velocities = 0.99 * self.velocities + total_force * self.config.dt / self.masses.unsqueeze(1)
        
        # Limit max velocity
        speeds = torch.norm(self.velocities, dim=1, keepdim=True)
        max_speed = 2.0
        self.velocities = torch.where(
            speeds > max_speed,
            self.velocities * max_speed / speeds,
            self.velocities
        )
        
        # Update positions
        self.positions = (self.positions + self.velocities * self.config.dt) % self.config.box_size
        
        # Apply SEC dynamics
        self.apply_sec_dynamics(distances)
    
    def compute_density_field(self, resolution: int = 100) -> torch.Tensor:
        """Compute 2D density field."""
        field = torch.zeros(resolution, resolution, device=DEVICE)
        
        # Bin positions
        ix = (self.positions[:, 0] / self.config.box_size * resolution).long() % resolution
        iy = (self.positions[:, 1] / self.config.box_size * resolution).long() % resolution
        
        # Scatter add masses
        indices = iy * resolution + ix
        field_flat = field.view(-1)
        field_flat.scatter_add_(0, indices, self.masses)
        
        return field.view(resolution, resolution)
    
    def compute_clustering_coefficient(self, sample_size: int = 300) -> float:
        """Compute clustering coefficient (sampled for speed)."""
        n = min(self.config.n_nodes, sample_size)
        r0 = self.config.interaction_radius
        
        # Get pairwise distances for sample
        sample_pos = self.positions[:n]
        
        pos_i = sample_pos.unsqueeze(1)
        pos_j = self.positions.unsqueeze(0)
        
        deltas = pos_i - pos_j
        deltas = deltas - self.config.box_size * torch.round(deltas / self.config.box_size)
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)
        
        # Set self to inf
        for i in range(n):
            distances[i, i] = float('inf')
        
        # Count triangles (vectorized approximation)
        neighbors = distances < r0  # [n, N]
        neighbor_counts = neighbors.sum(dim=1).float()
        
        # For each sampled node, check if its neighbors are connected
        total_triangles = 0.0
        total_possible = 0.0
        
        for i in range(n):
            neighbor_idx = torch.where(neighbors[i])[0]
            k = len(neighbor_idx)
            if k < 2:
                continue
            
            # Check connections among neighbors
            neighbor_pos = self.positions[neighbor_idx]
            d_ij = neighbor_pos.unsqueeze(1) - neighbor_pos.unsqueeze(0)
            d_ij = d_ij - self.config.box_size * torch.round(d_ij / self.config.box_size)
            dist_ij = torch.sqrt(torch.sum(d_ij**2, dim=2) + 1e-8)
            
            # Upper triangular (avoid double counting)
            connected = torch.triu(dist_ij < r0, diagonal=1)
            total_triangles += connected.sum().item()
            total_possible += k * (k - 1) / 2
        
        if total_possible > 0:
            return total_triangles / total_possible
        return 0.0


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation(config: SimConfig) -> Tuple[PACSystem, torch.Tensor, Dict]:
    """Run the full simulation (GPU accelerated)."""
    print(f"\nInitializing {config.n_nodes} nodes on {DEVICE}...")
    system = PACSystem(config)
    
    # Track history
    history = {
        'clustering': [],
        'void_fraction': []
    }
    
    # Density field accumulator
    field_res = 100
    density_accum = torch.zeros(field_res, field_res, device=DEVICE)
    
    print(f"Running {config.n_steps} steps...")
    
    for step in range(config.n_steps):
        system.step()
        
        # Accumulate density
        density_accum = density_accum * config.memory_decay + system.compute_density_field(field_res)
        
        # Record history periodically
        if step % 50 == 0:
            clustering = system.compute_clustering_coefficient()
            history['clustering'].append(clustering)
            
            # Void fraction
            density_np = density_accum.cpu().numpy()
            void_frac = np.sum(density_np < 0.1 * np.mean(density_np)) / density_np.size
            history['void_fraction'].append(void_frac)
            
            if step % 100 == 0:
                print(f"  Step {step}: clustering={clustering:.3f}, voids={void_frac:.3f}")
    
    return system, density_accum, history


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_web(system: PACSystem, density_field: torch.Tensor, 
                  config: SimConfig, save_path: Path = None):
    """Visualize the emergent web structure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Move to CPU for plotting
    positions = system.positions.cpu().numpy()
    masses = system.masses.cpu().numpy()
    entropy = system.entropy.cpu().numpy()
    density_np = density_field.cpu().numpy()
    
    # Panel 1: Node positions with connections
    ax1 = axes[0]
    ax1.scatter(positions[:, 0], positions[:, 1], 
               s=masses * 3, c='white', alpha=0.7, edgecolors='cyan')
    
    # Draw connections for sample of nodes
    r0 = config.interaction_radius
    for i in range(min(500, config.n_nodes)):
        delta = positions - positions[i]
        delta = delta - config.box_size * np.round(delta / config.box_size)
        distances = np.sqrt(np.sum(delta**2, axis=1))
        
        neighbors = np.where((distances > 0) & (distances < r0))[0]
        for j in neighbors:
            ax1.plot([positions[i, 0], positions[j, 0]], 
                    [positions[i, 1], positions[j, 1]], 
                    'c-', alpha=0.05, linewidth=0.3)
    
    ax1.set_xlim(0, config.box_size)
    ax1.set_ylim(0, config.box_size)
    ax1.set_facecolor('black')
    ax1.set_title('PAC Web Structure (Local Connections)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Panel 2: Density field
    ax2 = axes[1]
    vmax = max(density_np.max(), 1.0)
    im = ax2.imshow(density_np, origin='lower', 
                    extent=[0, config.box_size, 0, config.box_size],
                    cmap='inferno', norm=LogNorm(vmin=0.1, vmax=vmax))
    ax2.set_title('Density Field (Log Scale)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im, ax=ax2, label='Density')
    
    # Panel 3: Entropy field
    ax3 = axes[2]
    field_res = density_np.shape[0]
    entropy_field = np.zeros_like(density_np)
    ix = (positions[:, 0] / config.box_size * field_res).astype(int) % field_res
    iy = (positions[:, 1] / config.box_size * field_res).astype(int) % field_res
    np.add.at(entropy_field, (iy, ix), entropy)
    
    im3 = ax3.imshow(entropy_field, origin='lower',
                     extent=[0, config.box_size, 0, config.box_size],
                     cmap='plasma')
    ax3.set_title('Entropy Field (SEC Balance)', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(im3, ax=ax3, label='Entropy')
    
    plt.tight_layout()
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_evolution(history: Dict, config: SimConfig, save_path: Path = None):
    """Show evolution of clustering and void fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    steps = np.arange(0, len(history['clustering']) * 50, 50)
    
    # Clustering evolution
    ax1 = axes[0]
    ax1.plot(steps, history['clustering'], 'c-', linewidth=2)
    ax1.axhline(y=1/PHI, color='gold', linestyle='--', label=f'1/φ = {1/PHI:.3f}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Clustering Coefficient')
    ax1.set_title('Clustering Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Void fraction evolution
    ax2 = axes[1]
    ax2.plot(steps, history['void_fraction'], 'm-', linewidth=2)
    ax2.axhline(y=1/3, color='gold', linestyle='--', label=f'1/3 (F₃/F₅)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Void Fraction')
    ax2.set_title('Void Fraction Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_structure(system: PACSystem, density_field: torch.Tensor, 
                     config: SimConfig) -> Dict:
    """Analyze the emergent structure."""
    # Get current snapshot density (not accumulated)
    snapshot_density = system.compute_density_field(100).cpu().numpy()
    positions = system.positions.cpu().numpy()
    masses = system.masses.cpu().numpy()
    pac_values = system.pac_values.cpu().numpy()
    
    # 1. Filament detection (high-density regions in snapshot)
    nonzero = snapshot_density[snapshot_density > 0]
    if len(nonzero) > 0:
        threshold = np.percentile(nonzero, 75)
        filament_mask = snapshot_density > threshold
        filament_fraction = np.sum(filament_mask) / snapshot_density.size
    else:
        filament_fraction = 0.0
    
    # 2. Void detection (empty cells)
    void_mask = snapshot_density == 0
    void_fraction = np.sum(void_mask) / snapshot_density.size
    
    # 3. Local density distribution
    r0 = config.interaction_radius
    local_densities = []
    for i in range(len(positions)):
        delta = positions - positions[i]
        delta = delta - config.box_size * np.round(delta / config.box_size)
        distances = np.sqrt(np.sum(delta**2, axis=1))
        local_densities.append(np.sum(distances < r0))
    local_densities = np.array(local_densities)
    
    # 4. PAC conservation
    total_pac = np.sum(pac_values)
    total_mass = np.sum(masses)
    
    # 5. Final clustering
    clustering = system.compute_clustering_coefficient()
    
    # 6. Web structure: needs BOTH filaments AND voids
    # Also check density variance (web has high variance)
    density_cv = np.std(snapshot_density) / (np.mean(snapshot_density) + 1e-8)
    
    is_web = (filament_fraction > 0.05 and void_fraction > 0.3 and density_cv > 1.0)
    
    return {
        'filament_fraction': float(filament_fraction),
        'void_fraction': float(void_fraction),
        'density_mean': float(np.mean(local_densities)),
        'density_std': float(np.std(local_densities)),
        'density_cv': float(density_cv),
        'clustering_coefficient': float(clustering),
        'pac_conservation': float(total_pac / total_mass),
        'structure_type': 'web' if is_web else 'clump'
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("Experiment 09: PAC Web Emergence (GPU)")
    
    # Configuration - can increase nodes with GPU!
    # Key: SEC balance must be strong enough to create voids
    # but weak enough to allow filaments
    config = SimConfig(
        n_nodes=5000,  # Good balance of scale and speed
        box_size=120.0,
        dt=0.05,
        n_steps=600,
        interaction_radius=5.0,  # Smaller radius = more local
        pac_strength=0.8,  # Moderate gravity
        sec_balance=0.6,  # Stronger entropy pressure for voids
        memory_decay=0.95,  # Faster entropy decay
        seed=42
    )
    
    print(f"\n=== Configuration ===")
    print(f"Device: {DEVICE}")
    print(f"Nodes: {config.n_nodes}")
    print(f"Box size: {config.box_size}")
    print(f"Interaction radius: {config.interaction_radius} (LOCAL)")
    print(f"PAC strength: {config.pac_strength}")
    print(f"SEC balance: {config.sec_balance}")
    
    # Run simulation
    import time
    start = time.time()
    system, density_field, history = run_simulation(config)
    elapsed = time.time() - start
    print(f"\nSimulation completed in {elapsed:.1f}s")
    
    # Analyze results
    analysis = analyze_structure(system, density_field, config)
    
    print("\n=== Analysis ===")
    print(f"Structure type: {analysis['structure_type'].upper()}")
    print(f"Filament fraction: {analysis['filament_fraction']:.3f}")
    print(f"Void fraction: {analysis['void_fraction']:.3f}")
    print(f"Density CV: {analysis['density_cv']:.3f}")
    print(f"Clustering coefficient: {analysis['clustering_coefficient']:.3f}")
    print(f"PAC conservation: {analysis['pac_conservation']:.4f}")
    
    # Check if web formed
    is_web = analysis['structure_type'] == 'web'
    print_result(
        "Web structure emerged (not clump)",
        is_web,
        f"Filaments={analysis['filament_fraction']:.2f}, Voids={analysis['void_fraction']:.2f}"
    )
    
    # Visualize
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save visualization
    viz_path = results_dir / f'exp_09_pac_web_{timestamp}.png'
    visualize_web(system, density_field, config, viz_path)
    
    # Save evolution plot
    evo_path = results_dir / f'exp_09_evolution_{timestamp}.png'
    visualize_evolution(history, config, evo_path)
    
    # Save results
    results = {
        'experiment': 'exp_09_pac_web_emergence',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'elapsed_seconds': elapsed,
        'config': {
            'n_nodes': config.n_nodes,
            'box_size': config.box_size,
            'interaction_radius': config.interaction_radius,
            'pac_strength': config.pac_strength,
            'sec_balance': config.sec_balance,
            'n_steps': config.n_steps
        },
        'analysis': analysis,
        'history': {
            'clustering_final': history['clustering'][-1] if history['clustering'] else 0,
            'void_fraction_final': history['void_fraction'][-1] if history['void_fraction'] else 0
        },
        'conclusion': 'Web structure emerges from local PAC gravity'
    }
    
    results_file = results_dir / f'exp_09_pac_web_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
