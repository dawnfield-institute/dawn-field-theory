"""
Dark Matter 3D Simulation with Temporal Gradient Approach
=========================================================

Novel approach: Use cosmological distance as temporal proxy:
- Start with high-redshift (young) structures
- Progress to low-redshift (evolved) structures
- Match simulation evolution to cosmic time progression

Based on SEC framework with proven parameters from infodynamics validation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # Keep for matplotlib compatibility only
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from typing import Dict, List, Tuple, Any
from astro_data_fetcher import AstroDataFetcher
from sec_auto_tuning_engine import SECAutoTuningEngine, SECParameters, SECTargetMetrics, SECOptimizationResult

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Temporal Gradient Parameters ---

# SEC Parameters - Based on proven optimal values from framework experiments
# Using validated parameters from: α=0.005857, ξ=1.0571, entropy_threshold=0.55
# From infodynamics_arithmetic_v1.md and MED framework validation
sec_params = SECParameters(
    rho_thresh=1.0571,         # Optimal ξ threshold from MED validation
    dispersion_strength=0.55,  # Matched to entropy threshold for balance (Ξ ≈ 1)
    clustering_strength=0.25,  # Crystallization threshold from predictive collapse
    branching_bias=0.12,       # Collapse curvature threshold (kappa)
    centroid_strength=0.0      # NO centroid pull - let system be chaotic
)

# Temporal progression parameters
n_particles = 15000
n_redshift_bins = 5  # Number of evolutionary stages
steps_per_bin = 1000  # Evolution steps per redshift bin
total_steps = n_redshift_bins * steps_per_bin

# Physics parameters
G = 4.498e-6  # Gravitational constant (kpc³/M☉/Gyr²)
dt = 0.01
damping = 0.999
noise_strength = 0.0001
spatial_bounds = 20.0

# Analysis parameters
snapshot_interval = 200
analysis_radius = 3.0

# --- Helper Functions (PyTorch CUDA) ---
def compute_density_cuda(positions, radius=2.5, chunk_size=2000):
    """Compute local density around each particle using CUDA with chunked processing"""
    positions = positions.contiguous()
    n = positions.shape[0]
    densities = torch.zeros(n, device=positions.device)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_positions = positions[i:end_i].contiguous()
        
        diff = chunk_positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        
        # Count neighbors within radius
        neighbors = (distances < radius).float()
        neighbors[distances == 0] = 0  # Exclude self
        densities[i:end_i] = torch.sum(neighbors, dim=1)
    
    return densities

def compute_fractal_dimension_cuda(positions, n_scales=15):
    """Compute fractal dimension using box-counting method (pure PyTorch)"""
    positions = positions.contiguous()
    
    # Create logarithmically spaced scales
    max_extent = torch.max(torch.abs(positions))
    min_scale = max_extent / 100
    max_scale = max_extent / 2
    scales = torch.logspace(torch.log10(min_scale), torch.log10(max_scale), 
                           n_scales, device=positions.device)
    
    counts = torch.zeros(n_scales, device=positions.device)
    
    for i, scale in enumerate(scales):
        # Discretize positions into boxes
        box_coords = torch.floor(positions / scale).int()
        
        # Count unique boxes using torch operations
        # Convert to linear indices for uniqueness check
        box_indices = (box_coords[:, 0] * 1000000 + 
                      box_coords[:, 1] * 1000 + 
                      box_coords[:, 2])
        
        unique_boxes = torch.unique(box_indices)
        counts[i] = len(unique_boxes)
    
    # Linear regression in log space (pure PyTorch)
    log_scales = torch.log(scales)
    log_counts = torch.log(counts + 1e-10)  # Avoid log(0)
    
    # Remove invalid points
    valid_mask = torch.isfinite(log_counts) & (counts > 0)
    if torch.sum(valid_mask) < 3:
        return torch.tensor(2.0, device=positions.device)
    
    log_scales = log_scales[valid_mask]
    log_counts = log_counts[valid_mask]
    
    # PyTorch linear regression
    n = len(log_scales)
    mean_x = torch.mean(log_scales)
    mean_y = torch.mean(log_counts)
    
    numerator = torch.sum((log_scales - mean_x) * (log_counts - mean_y))
    denominator = torch.sum((log_scales - mean_x) ** 2)
    
    if denominator == 0:
        return torch.tensor(2.0, device=positions.device)
    
    slope = numerator / denominator
    return -slope  # Negative because log(count) decreases with log(scale)

def compute_clustering_metrics_cuda(positions, n_bins=20):
    """Compute spatial entropy and clustering using pure PyTorch operations"""
    positions = positions.contiguous()
    
    # Spatial entropy calculation
    bounds = torch.max(torch.abs(positions))
    bin_edges = torch.linspace(-bounds, bounds, n_bins + 1, device=positions.device)
    
    # 3D histogram using torch.bincount
    x_indices = torch.bucketize(positions[:, 0], bin_edges) - 1
    y_indices = torch.bucketize(positions[:, 1], bin_edges) - 1
    z_indices = torch.bucketize(positions[:, 2], bin_edges) - 1
    
    # Clamp indices to valid range
    x_indices = torch.clamp(x_indices, 0, n_bins - 1)
    y_indices = torch.clamp(y_indices, 0, n_bins - 1)
    z_indices = torch.clamp(z_indices, 0, n_bins - 1)
    
    # Convert 3D indices to linear indices
    linear_indices = x_indices * n_bins * n_bins + y_indices * n_bins + z_indices
    
    # Count occurrences using bincount
    counts = torch.bincount(linear_indices, minlength=n_bins**3)
    
    # Calculate entropy
    total_points = positions.shape[0]
    probabilities = counts.float() / total_points
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    
    if len(probabilities) == 0:
        spatial_entropy = torch.tensor(0.0, device=positions.device)
    else:
        spatial_entropy = -torch.sum(probabilities * torch.log2(probabilities))
    
    # Density variance
    densities = compute_density_cuda(positions)
    density_variance = torch.var(densities)
    
    return spatial_entropy, density_variance

def compute_trend_pytorch(values, window=10):
    """Compute trend of recent values using PyTorch linear regression"""
    if len(values) < window:
        return torch.tensor(0.0, device=values[0].device if len(values) > 0 else device)
    
    recent_values = torch.stack(values[-window:])
    x = torch.arange(len(recent_values), dtype=torch.float32, device=recent_values.device)
    
    # Linear regression
    n = len(x)
    mean_x = torch.mean(x)
    mean_y = torch.mean(recent_values)
    
    numerator = torch.sum((x - mean_x) * (recent_values - mean_y))
    denominator = torch.sum((x - mean_x) ** 2)
    
    if denominator == 0:
        return torch.tensor(0.0, device=recent_values.device)
    
    slope = numerator / denominator
    return slope

def apply_sec_forces_cuda(positions, velocities, sec_params, n_particles, chunk_size=2000):
    """Apply SEC forces using chunked processing for memory efficiency"""
    positions = positions.contiguous()
    forces = torch.zeros_like(positions)
    
    rho_thresh = sec_params.rho_thresh
    dispersion_strength = sec_params.dispersion_strength
    clustering_strength = sec_params.clustering_strength
    branching_bias = sec_params.branching_bias
    
    # Process in chunks
    for i in range(0, n_particles, chunk_size):
        end_i = min(i + chunk_size, n_particles)
        chunk_positions = positions[i:end_i].contiguous()
        chunk_densities = compute_density_cuda(chunk_positions)
        
        # Compute pairwise forces for this chunk
        diff = positions.unsqueeze(0) - chunk_positions.unsqueeze(1)  # [chunk_size, n_particles, 3]
        distances = torch.norm(diff, dim=2) + 1e-6  # [chunk_size, n_particles]
        directions = diff / distances.unsqueeze(2)  # [chunk_size, n_particles, 3]
        
        # SEC Logic: disperse when density is low (most particles)
        low_density_mask = (chunk_densities < rho_thresh * n_particles).unsqueeze(1)  # [chunk_size, 1]
        
        # Repulsion for low-density regions (dispersion)
        repulsion_strength = 1.0 / (distances ** 2 + 0.1)  # [chunk_size, n_particles]
        repulsion_forces = directions * repulsion_strength.unsqueeze(2)  # [chunk_size, n_particles, 3]
        repulsion_total = torch.sum(repulsion_forces, dim=1) * dispersion_strength
        
        # Attraction for high-density regions (limited clustering)
        high_density_mask = ~low_density_mask
        attraction_strength = 1.0 / (distances + 1.0)  # [chunk_size, n_particles]
        attraction_forces = -directions * attraction_strength.unsqueeze(2)  # [chunk_size, n_particles, 3]
        attraction_total = torch.sum(attraction_forces, dim=1) * clustering_strength
        
        # Combine forces based on density
        chunk_forces = (low_density_mask.float() * repulsion_total + 
                       high_density_mask.float() * attraction_total)
        
        forces[i:end_i] = chunk_forces
    
    # Add radial branching bias (pushes outward from center)
    center = torch.mean(positions, dim=0)
    radial_directions = positions - center
    radial_distances = torch.norm(radial_directions, dim=1, keepdim=True) + 1e-6
    radial_directions = radial_directions / radial_distances
    forces += branching_bias * radial_directions
    
    return forces

def initialize_particles_from_target(target_positions, n_particles):
    """Initialize simulation particles based on target structure with some randomization"""
    n_target = len(target_positions)
    
    if n_particles <= n_target:
        # Sample subset of target positions
        indices = torch.randperm(n_target)[:n_particles]
        positions = target_positions[indices].clone()
    else:
        # Replicate and add noise
        repeat_factor = n_particles // n_target
        remainder = n_particles % n_target
        
        positions = target_positions.repeat(repeat_factor, 1)
        if remainder > 0:
            positions = torch.cat([positions, target_positions[:remainder]], dim=0)
    
    # Add small random perturbations to break symmetry
    positions += torch.randn_like(positions) * 0.5
    
    # Initialize velocities
    velocities = torch.randn_like(positions) * 0.1
    
    return positions, velocities

def run_temporal_gradient_simulation():
    """Main simulation using temporal gradient approach"""
    print("=== TEMPORAL GRADIENT DARK MATTER SIMULATION ===")
    print(f"Approach: High-redshift (young) → Low-redshift (evolved) structures")
    print(f"Redshift bins: {n_redshift_bins}, Steps per bin: {steps_per_bin}")
    print(f"Total evolution: {total_steps} steps")
    
    # Fetch temporal gradient data
    fetcher = AstroDataFetcher(device=device)
    print("\nFetching temporal gradient data...")
    
    try:
        position_bins, metadata = fetcher.fetch_temporal_gradient_data(
            total_limit=5000, z_bins=n_redshift_bins
        )
        print(f"✓ Loaded {len(position_bins)} redshift bins")
        for i, bin_meta in enumerate(metadata['bin_metadata']):
            print(f"  Bin {i+1}: z={bin_meta['redshift_range'][0]:.2f}-{bin_meta['redshift_range'][1]:.2f}, "
                  f"age≈{bin_meta.get('age_gyr', 0):.1f} Gyr, {bin_meta['count']} galaxies")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Initialize simulation with youngest structures (highest redshift)
    print(f"\nInitializing with youngest structures (bin 1, highest redshift)...")
    initial_target = position_bins[0]
    positions, velocities = initialize_particles_from_target(initial_target, n_particles)
    
    print(f"✓ Initialized {n_particles} particles")
    print(f"SEC Parameters: rho_thresh={sec_params.rho_thresh:.4f}, "
          f"clustering={sec_params.clustering_strength:.4f}, "
          f"branching={sec_params.branching_bias:.4f}")
    
    # Simulation tracking
    step = 0
    start_time = time.time()
    
    # Metrics tracking
    fractal_dims = []
    entropies = []
    density_vars = []
    kinetic_energies = []
    similarities = []
    
    current_bin = 0
    steps_in_current_bin = 0
    
    print(f"\nStarting temporal gradient evolution...")
    print(f"Current target: Bin 1/{n_redshift_bins} (youngest structures)")
    
    while step < total_steps:
        # Check if we should progress to next evolutionary stage
        if steps_in_current_bin >= steps_per_bin and current_bin < len(position_bins) - 1:
            current_bin += 1
            steps_in_current_bin = 0
            print(f"\n🌌 Progressing to Bin {current_bin + 1}/{n_redshift_bins} "
                  f"(more evolved structures)")
            
            # Update target for similarity comparison
            current_target = position_bins[current_bin]
        else:
            current_target = position_bins[current_bin]
        
        # Apply SEC forces
        forces = apply_sec_forces_cuda(positions, velocities, sec_params, n_particles)
        
        # Update physics
        velocities += forces * dt
        velocities *= damping  # Apply damping
        
        # Add small amount of noise to prevent stagnation
        velocities += torch.randn_like(velocities) * noise_strength
        
        positions += velocities * dt
        
        # Boundary conditions (periodic-like)
        positions = torch.clamp(positions, -spatial_bounds, spatial_bounds)
        
        # Compute metrics periodically
        if step % snapshot_interval == 0:
            with torch.no_grad():
                # Compute simulation metrics
                fractal_dim = compute_fractal_dimension_cuda(positions)
                spatial_entropy, density_variance = compute_clustering_metrics_cuda(positions)
                kinetic_energy = torch.sum(velocities ** 2) / 2
                
                # Compute similarity to current target
                sim_fractal_dim = compute_fractal_dimension_cuda(current_target)
                sim_entropy, sim_density_var = compute_clustering_metrics_cuda(current_target)
                
                # Similarity calculations
                fractal_similarity = 1.0 - abs(fractal_dim - sim_fractal_dim) / max(fractal_dim, sim_fractal_dim)
                entropy_similarity = 1.0 - abs(spatial_entropy - sim_entropy) / max(spatial_entropy, sim_entropy, 1e-6)
                density_similarity = 1.0 - abs(density_variance - sim_density_var) / max(density_variance, sim_density_var, 1e-6)
                
                overall_similarity = (fractal_similarity * 0.4 + 
                                    entropy_similarity * 0.4 + 
                                    density_similarity * 0.2)
                
                # Store metrics
                fractal_dims.append(fractal_dim.cpu())
                entropies.append(spatial_entropy.cpu())
                density_vars.append(density_variance.cpu())
                kinetic_energies.append(kinetic_energy.cpu())
                similarities.append(overall_similarity.cpu())
                
                # Progress update
                elapsed = time.time() - start_time
                bin_progress = steps_in_current_bin / steps_per_bin * 100
                
                print(f"Step {step:4d} | Bin {current_bin+1}/{n_redshift_bins} ({bin_progress:5.1f}%) | "
                      f"Fractal_D={fractal_dim:.3f} | Entropy={spatial_entropy:.3f} | "
                      f"Similarity={overall_similarity:.3f} | Time={elapsed:.1f}s")
        
        step += 1
        steps_in_current_bin += 1
    
    # Final analysis
    total_time = time.time() - start_time
    print(f"\n=== TEMPORAL GRADIENT SIMULATION COMPLETE ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Performance: {step / total_time:.1f} steps/second")
    print(f"Evolved through {n_redshift_bins} evolutionary stages")
    
    # Final metrics against most evolved target
    final_target = position_bins[-1]
    final_fractal = compute_fractal_dimension_cuda(positions)
    final_entropy, final_density_var = compute_clustering_metrics_cuda(positions)
    
    target_fractal = compute_fractal_dimension_cuda(final_target)
    target_entropy, target_density_var = compute_clustering_metrics_cuda(final_target)
    
    print(f"\n=== FINAL COMPARISON (Most Evolved Structures) ===")
    print(f"Simulation fractal dimension: {final_fractal:.3f}")
    print(f"Target fractal dimension: {target_fractal:.3f}")
    print(f"Simulation entropy: {final_entropy:.3f}")
    print(f"Target entropy: {target_entropy:.3f}")
    print(f"Simulation density variance: {final_density_var:.1f}")
    print(f"Target density variance: {target_density_var:.1f}")
    
    final_similarity = (
        (1.0 - abs(final_fractal - target_fractal) / max(final_fractal, target_fractal)) * 0.4 +
        (1.0 - abs(final_entropy - target_entropy) / max(final_entropy, target_entropy, 1e-6)) * 0.4 +
        (1.0 - abs(final_density_var - target_density_var) / max(final_density_var, target_density_var, 1e-6)) * 0.2
    )
    
    print(f"Final similarity score: {final_similarity:.3f}")
    
    # Create evolution plot
    create_temporal_evolution_plot(fractal_dims, entropies, similarities, n_redshift_bins, steps_per_bin)
    
    return {
        'positions': positions.cpu().numpy(),
        'metrics': {
            'fractal_dims': [f.item() for f in fractal_dims],
            'entropies': [e.item() for e in entropies],
            'similarities': [s.item() for s in similarities],
            'final_similarity': final_similarity.item()
        },
        'metadata': metadata
    }

def create_temporal_evolution_plot(fractal_dims, entropies, similarities, n_bins, steps_per_bin):
    """Create plot showing evolution across temporal gradient"""
    steps = np.arange(len(fractal_dims)) * snapshot_interval
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fractal dimension evolution
    ax1.plot(steps, fractal_dims, 'b-', linewidth=2)
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for bin transitions
    for i in range(1, n_bins):
        bin_step = i * steps_per_bin
        ax1.axvline(bin_step, color='red', linestyle='--', alpha=0.5, 
                   label=f'Bin {i+1}' if i == 1 else '')
    
    # Entropy evolution
    ax2.plot(steps, entropies, 'g-', linewidth=2)
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Spatial Entropy')
    ax2.set_title('Spatial Entropy Evolution')
    ax2.grid(True, alpha=0.3)
    
    for i in range(1, n_bins):
        bin_step = i * steps_per_bin
        ax2.axvline(bin_step, color='red', linestyle='--', alpha=0.5)
    
    # Similarity evolution
    ax3.plot(steps, similarities, 'purple', linewidth=2)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Similarity to Target')
    ax3.set_title('Similarity to Current Target Bin')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    for i in range(1, n_bins):
        bin_step = i * steps_per_bin
        ax3.axvline(bin_step, color='red', linestyle='--', alpha=0.5)
    
    # Bin progression chart
    bin_labels = [f'Bin {i+1}\n(z~{i*0.2:.1f})' for i in range(n_bins)]
    bin_positions = [(i + 0.5) * steps_per_bin for i in range(n_bins)]
    
    ax4.barh(range(n_bins), [steps_per_bin] * n_bins, 
            color=plt.cm.viridis(np.linspace(0, 1, n_bins)))
    ax4.set_yticks(range(n_bins))
    ax4.set_yticklabels([f'Bin {i+1}' for i in range(n_bins)])
    ax4.set_xlabel('Steps')
    ax4.set_title('Temporal Gradient Progression\n(Young → Evolved)')
    
    plt.tight_layout()
    plt.savefig('temporal_gradient_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved evolution plot: temporal_gradient_evolution.png")
    plt.show()

if __name__ == "__main__":
    try:
        print("🌌 Dark Matter Temporal Gradient Simulation")
        print("Using cosmological distance as evolutionary proxy")
        print("=" * 60)
        
        results = run_temporal_gradient_simulation()
        
        print("\n🎯 Simulation completed successfully!")
        print(f"Final similarity: {results['metrics']['final_similarity']:.3f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleaned up")
