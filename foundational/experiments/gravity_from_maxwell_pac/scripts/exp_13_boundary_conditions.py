#!/usr/bin/env python3
"""
exp_13_boundary_conditions.py

Tests whether Ξ ≈ 1.057 is a saturation bound for CLOSED systems.

Hypothesis (from refined interpretation):
  Ξ is not a universal constant, but the maximum sustainable computational
  asymmetry for closed recursive systems under PAC conservation.

Predictions:
  1. CLOSED (periodic boundary) → converges toward Ξ
  2. OPEN (particle injection)  → drifts or overshoots Ξ
  3. LEAKY (particle removal)   → hovers below Ξ

We measure the "asymmetry ratio" - the ratio of structure gain to entropy cost.
This should saturate at Ξ for closed systems only.

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# BOUNDARY CONDITION TYPES
# =============================================================================

@dataclass
class BoundaryConfig:
    """Configuration for boundary condition experiment."""
    name: str
    injection_rate: float  # particles per step (0 = closed)
    removal_rate: float    # fraction removed per step (0 = closed)
    description: str


BOUNDARY_CONDITIONS = [
    BoundaryConfig("CLOSED", 0.0, 0.0, "Periodic boundary, no particle flux"),
    BoundaryConfig("OPEN_INJECT", 5.0, 0.0, "Inject 5 particles per step"),
    BoundaryConfig("LEAKY", 0.0, 0.002, "Remove 0.2% of particles per step"),
    BoundaryConfig("BALANCED", 3.0, 0.001, "Inject 3, remove 0.1% per step"),
]


# =============================================================================
# PAC SYSTEM WITH CONFIGURABLE BOUNDARIES
# =============================================================================

class PACSystemBoundary:
    """GPU-accelerated PAC system with configurable boundary conditions."""
    
    def __init__(self, n_nodes: int, box_size: float, interaction_radius: float,
                 pac_strength: float, sec_balance: float, 
                 boundary: BoundaryConfig, seed: int = 42):
        self.initial_n = n_nodes
        self.box_size = box_size
        self.r0 = interaction_radius
        self.pac_strength = pac_strength
        self.sec_balance = sec_balance
        self.boundary = boundary
        self.dt = 0.05
        self.max_nodes = n_nodes + 2000  # Buffer for injection
        
        torch.manual_seed(seed)
        
        # Initialize on grid with perturbation
        n_per_dim = int(np.ceil(np.sqrt(n_nodes)))
        self.n_nodes = n_per_dim * n_per_dim
        
        spacing = box_size / n_per_dim
        grid = torch.arange(n_per_dim, device=DEVICE, dtype=torch.float32) * spacing + spacing/2
        xx, yy = torch.meshgrid(grid, grid, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        positions += torch.randn_like(positions) * spacing * 0.1
        self.positions = positions % box_size
        
        self.velocities = torch.zeros_like(self.positions)
        self.masses = 1.0 + 0.1 * torch.randn(self.n_nodes, device=DEVICE)
        self.entropy = 0.1 * torch.rand(self.n_nodes, device=DEVICE)
        
        # Track total mass for PAC monitoring
        self.initial_mass = self.masses.sum().item()
        self.total_injected = 0.0
        self.total_removed = 0.0
    
    def inject_particles(self, count: int):
        """Inject new particles at random positions."""
        if count <= 0 or self.n_nodes >= self.max_nodes:
            return
        
        count = min(count, self.max_nodes - self.n_nodes)
        
        new_pos = torch.rand(count, 2, device=DEVICE) * self.box_size
        new_vel = torch.zeros(count, 2, device=DEVICE)
        new_mass = 1.0 + 0.1 * torch.randn(count, device=DEVICE)
        new_entropy = 0.1 * torch.rand(count, device=DEVICE)
        
        self.positions = torch.cat([self.positions, new_pos], dim=0)
        self.velocities = torch.cat([self.velocities, new_vel], dim=0)
        self.masses = torch.cat([self.masses, new_mass], dim=0)
        self.entropy = torch.cat([self.entropy, new_entropy], dim=0)
        
        self.n_nodes += count
        self.total_injected += new_mass.sum().item()
    
    def remove_particles(self, fraction: float):
        """Remove fraction of particles (random selection)."""
        if fraction <= 0 or self.n_nodes <= 100:
            return
        
        n_remove = max(1, int(self.n_nodes * fraction))
        n_remove = min(n_remove, self.n_nodes - 100)  # Keep at least 100
        
        # Random selection
        keep_mask = torch.ones(self.n_nodes, dtype=torch.bool, device=DEVICE)
        remove_indices = torch.randperm(self.n_nodes, device=DEVICE)[:n_remove]
        keep_mask[remove_indices] = False
        
        removed_mass = self.masses[remove_indices].sum().item()
        
        self.positions = self.positions[keep_mask]
        self.velocities = self.velocities[keep_mask]
        self.masses = self.masses[keep_mask]
        self.entropy = self.entropy[keep_mask]
        
        self.n_nodes = len(self.masses)
        self.total_removed += removed_mass
    
    def compute_pa_ratio(self) -> float:
        """
        Compute P/A ratio (Potential / Actualization).
        
        Analogous to SEC's definition:
        - Potential: variance in LOW density regions (dispersed, unrealized)
        - Actualization: variance in HIGH density regions (structured, realized)
        
        This measures the balance between expansion and collapse.
        For closed recursive systems, this should approach Ξ ≈ 1.057.
        """
        n = self.n_nodes
        r0 = self.r0
        box = self.box_size
        
        # Compute local density for each particle
        pos_i = self.positions.unsqueeze(1)
        pos_j = self.positions.unsqueeze(0)
        deltas = pos_i - pos_j
        deltas = deltas - box * torch.round(deltas / box)
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)
        distances.fill_diagonal_(float('inf'))
        
        # Local density = number of neighbors within r0
        local_density = torch.sum(distances < r0, dim=1).float()
        
        # Split into low and high density regions
        median_density = local_density.median()
        
        low_density_mask = local_density < median_density
        high_density_mask = local_density >= median_density
        
        # Potential: variance in low-density (dispersed) regions
        if low_density_mask.sum() > 10:
            potential = local_density[low_density_mask].var().item()
        else:
            potential = 0.001
        
        # Actualization: variance in high-density (structured) regions  
        if high_density_mask.sum() > 10:
            actualization = local_density[high_density_mask].var().item()
        else:
            actualization = 0.001
        
        # P/A ratio - balance between dispersion and concentration
        pa_ratio = (potential + 0.001) / (actualization + 0.001)
        
        return float(pa_ratio)
    
    def compute_asymmetry_ratio(self) -> float:
        """
        Compute the asymmetry ratio: structure gain / entropy cost.
        
        This measures how efficiently the system converts entropy into structure.
        For closed systems, this should approach Ξ ≈ 1.057.
        """
        n = self.n_nodes
        r0 = self.r0
        box = self.box_size
        
        # Compute local density for each particle
        pos_i = self.positions.unsqueeze(1)
        pos_j = self.positions.unsqueeze(0)
        deltas = pos_i - pos_j
        deltas = deltas - box * torch.round(deltas / box)
        distances = torch.sqrt(torch.sum(deltas**2, dim=2) + 1e-8)
        distances.fill_diagonal_(float('inf'))
        
        # Local count (structure measure)
        local_count = torch.sum(distances < r0, dim=1).float()
        
        # Structure gain: deviation from uniform
        mean_count = local_count.mean()
        structure_variance = local_count.var()
        
        # Entropy cost: total entropy in system
        total_entropy = self.entropy.sum()
        
        # Asymmetry ratio
        if total_entropy < 1e-6:
            return 0.0
        
        # Normalize by system size
        normalized_structure = structure_variance / (mean_count + 1)
        normalized_entropy = total_entropy / n
        
        asymmetry = normalized_structure / (normalized_entropy + 0.01)
        
        return float(asymmetry.item())
    
    def compute_clustering_ratio(self) -> float:
        """Alternative measure: clustering coefficient / entropy density."""
        # Sample-based clustering
        positions = self.positions.cpu().numpy()
        r0 = self.r0
        box = self.box_size
        n_sample = min(100, len(positions))
        
        total_triangles = 0
        total_possible = 0
        
        indices = np.random.choice(len(positions), n_sample, replace=False)
        for i in indices:
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
        
        clustering = total_triangles / (total_possible + 1)
        
        # Entropy density
        entropy_density = self.entropy.mean().item()
        
        # Ratio
        return clustering / (entropy_density + 0.01)
    
    def step(self):
        """Single simulation step with boundary effects."""
        n = self.n_nodes
        r0 = self.r0
        box = self.box_size
        
        # Physics step
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
        
        # SEC dynamics
        local_count = torch.sum(distances < r0, dim=1).float()
        mean_count = local_count.mean()
        density_deviation = (local_count - mean_count) / (mean_count + 1)
        entropy_change = self.sec_balance * density_deviation
        self.entropy = torch.clamp(self.entropy + entropy_change, min=0.0)
        
        # Apply boundary conditions
        if self.boundary.injection_rate > 0:
            self.inject_particles(int(self.boundary.injection_rate))
        
        if self.boundary.removal_rate > 0:
            self.remove_particles(self.boundary.removal_rate)
    
    def get_pac_status(self) -> Dict:
        """Get PAC conservation status."""
        current_mass = self.masses.sum().item()
        expected_mass = self.initial_mass + self.total_injected - self.total_removed
        pac_conservation = current_mass / expected_mass if expected_mass > 0 else 1.0
        
        return {
            'current_mass': current_mass,
            'initial_mass': self.initial_mass,
            'total_injected': self.total_injected,
            'total_removed': self.total_removed,
            'expected_mass': expected_mass,
            'pac_conservation': pac_conservation,
            'n_nodes': self.n_nodes
        }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_boundary_experiment(boundary: BoundaryConfig, n_steps: int = 400) -> Dict:
    """Run experiment with specific boundary condition."""
    print(f"\n--- {boundary.name}: {boundary.description} ---")
    
    system = PACSystemBoundary(
        n_nodes=2500,
        box_size=80.0,
        interaction_radius=8.0,
        pac_strength=0.8,
        sec_balance=1.0,  # Generic SEC balance - NOT Ξ! We measure if Ξ emerges.
        boundary=boundary,
        seed=42
    )
    
    # Track P/A ratio over time (the correct metric for Ξ)
    pa_history = []
    asymmetry_history = []
    clustering_history = []
    n_nodes_history = []
    
    for step in range(n_steps + 1):
        if step % 50 == 0:
            pa = system.compute_pa_ratio()
            ar = system.compute_asymmetry_ratio()
            cr = system.compute_clustering_ratio()
            pa_history.append({'step': step, 'pa_ratio': pa})
            asymmetry_history.append({'step': step, 'asymmetry': ar})
            clustering_history.append({'step': step, 'clustering_ratio': cr})
            n_nodes_history.append({'step': step, 'n_nodes': system.n_nodes})
            
            if step % 100 == 0:
                print(f"  Step {step}: n={system.n_nodes}, P/A={pa:.4f}, Ξ={XI:.4f}")
        
        if step < n_steps:
            system.step()
    
    # Final metrics
    final_pa = pa_history[-1]['pa_ratio']
    final_ar = asymmetry_history[-1]['asymmetry']
    final_cr = clustering_history[-1]['clustering_ratio']
    pac_status = system.get_pac_status()
    
    # Analyze convergence toward Ξ (using P/A ratio)
    mid_pa = pa_history[len(pa_history)//2]['pa_ratio']
    convergence = final_pa / (mid_pa + 0.001) if mid_pa > 0.001 else 0
    
    # Distance from Ξ
    distance_from_xi = abs(final_pa - XI) / XI
    
    return {
        'boundary': boundary.name,
        'description': boundary.description,
        'pa_history': pa_history,
        'asymmetry_history': asymmetry_history,
        'clustering_history': clustering_history,
        'n_nodes_history': n_nodes_history,
        'final_pa_ratio': final_pa,
        'final_asymmetry': final_ar,
        'final_clustering_ratio': final_cr,
        'distance_from_xi': distance_from_xi,
        'convergence_ratio': convergence,
        'pac_status': pac_status
    }


def main():
    print_header("Experiment 13: Boundary Conditions and Ξ Convergence")
    
    print(f"\nHypothesis: Ξ ≈ {XI:.4f} is a saturation bound for CLOSED systems only")
    print(f"\nPredictions:")
    print(f"  CLOSED  → converges toward Ξ")
    print(f"  OPEN    → drifts or overshoots Ξ")
    print(f"  LEAKY   → hovers below Ξ")
    
    import time
    start = time.time()
    
    # Run all boundary conditions
    results = []
    for boundary in BOUNDARY_CONDITIONS:
        result = run_boundary_experiment(boundary, n_steps=400)
        results.append(result)
    
    elapsed = time.time() - start
    print(f"\n\nCompleted in {elapsed:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Distance from Ξ = {XI:.4f}")
    print(f"{'='*60}")
    print(f"{'Boundary':<15} {'Final P/A':>10} {'Dist from Ξ':>12} {'Prediction':>12}")
    print(f"{'-'*60}")
    
    for r in results:
        pred_status = ""
        if r['boundary'] == "CLOSED":
            pred_status = "CLOSEST" if r['distance_from_xi'] < 0.3 else "MISS"
        elif r['boundary'] == "OPEN_INJECT":
            pred_status = "OVERSHOOT" if r['final_pa_ratio'] > XI else "UNDER"
        elif r['boundary'] == "LEAKY":
            pred_status = "BELOW" if r['final_pa_ratio'] < XI else "ABOVE"
        else:
            pred_status = "MIXED"
        
        print(f"{r['boundary']:<15} {r['final_pa_ratio']:>10.4f} {r['distance_from_xi']:>12.4f} {pred_status:>12}")
    
    # Test predictions
    closed_result = next(r for r in results if r['boundary'] == "CLOSED")
    open_result = next(r for r in results if r['boundary'] == "OPEN_INJECT")
    leaky_result = next(r for r in results if r['boundary'] == "LEAKY")
    
    # Prediction 1: Closed should be closest to Ξ
    closed_closest = closed_result['distance_from_xi'] < min(
        open_result['distance_from_xi'], 
        leaky_result['distance_from_xi']
    )
    print_result(
        "CLOSED system closest to Ξ",
        closed_closest,
        f"d={closed_result['distance_from_xi']:.3f} vs open={open_result['distance_from_xi']:.3f}, leaky={leaky_result['distance_from_xi']:.3f}"
    )
    
    # Prediction 2: Open should overshoot or drift
    open_different = open_result['distance_from_xi'] > 0.1
    print_result(
        "OPEN system differs from Ξ",
        open_different,
        f"distance={open_result['distance_from_xi']:.3f}"
    )
    
    # Prediction 3: Leaky should be below
    leaky_below = leaky_result['final_pa_ratio'] < XI
    print_result(
        "LEAKY system below Ξ",
        leaky_below,
        f"P/A={leaky_result['final_pa_ratio']:.4f} < Ξ={XI:.4f}"
    )
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'CLOSED': 'blue', 'OPEN_INJECT': 'red', 'LEAKY': 'green', 'BALANCED': 'purple'}
    
    # Panel 1: P/A ratio over time
    ax1 = axes[0, 0]
    for r in results:
        steps = [h['step'] for h in r['pa_history']]
        pas = [h['pa_ratio'] for h in r['pa_history']]
        ax1.plot(steps, pas, '-o', color=colors[r['boundary']], 
                label=r['boundary'], linewidth=2, markersize=4)
    
    ax1.axhline(y=XI, color='black', linestyle='--', linewidth=2, label=f'Ξ = {XI:.4f}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('P/A Ratio')
    ax1.set_title('P/A Ratio Evolution (Potential / Actualization)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Distance from Ξ over time
    ax2 = axes[0, 1]
    for r in results:
        steps = [h['step'] for h in r['pa_history']]
        dists = [abs(h['pa_ratio'] - XI) / XI for h in r['pa_history']]
        ax2.plot(steps, dists, '-o', color=colors[r['boundary']], 
                label=r['boundary'], linewidth=2, markersize=4)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('|P/A - Ξ| / Ξ')
    ax2.set_title('Distance from Ξ Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Node count over time
    ax3 = axes[1, 0]
    for r in results:
        steps = [h['step'] for h in r['n_nodes_history']]
        nodes = [h['n_nodes'] for h in r['n_nodes_history']]
        ax3.plot(steps, nodes, '-o', color=colors[r['boundary']], 
                label=r['boundary'], linewidth=2, markersize=4)
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Number of Nodes')
    ax3.set_title('System Size Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Final comparison bar chart
    ax4 = axes[1, 1]
    names = [r['boundary'] for r in results]
    final_pas = [r['final_pa_ratio'] for r in results]
    bar_colors = [colors[n] for n in names]
    
    bars = ax4.bar(names, final_pas, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=XI, color='black', linestyle='--', linewidth=2, label=f'Ξ = {XI:.4f}')
    ax4.set_ylabel('Final P/A Ratio')
    ax4.set_title('Final P/A by Boundary Type')
    ax4.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, final_pas):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Boundary Conditions and Ξ Convergence\n(Hypothesis: Ξ is saturation bound for CLOSED systems)', 
                 fontsize=14)
    plt.tight_layout()
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig_path = results_dir / f'exp_13_boundary_conditions_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {fig_path}")
    
    plt.show()
    
    # Save results
    output = {
        'experiment': 'exp_13_boundary_conditions',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Ξ is saturation bound for closed recursive systems',
        'xi': XI,
        'results': results,
        'predictions': {
            'closed_closest': closed_closest,
            'open_different': open_different,
            'leaky_below': leaky_below
        },
        'elapsed_seconds': elapsed
    }
    
    results_file = results_dir / f'exp_13_boundary_conditions_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
