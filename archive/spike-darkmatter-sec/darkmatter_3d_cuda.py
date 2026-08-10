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

# --- Parameters (Scaled Up for Macro View) ---

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
n_particles = 15000       # Reduced to fit GPU memory
n_steps = 4000           # Moderate simulation length for testing (will auto-stop on convergence)
dt = 0.005               # Smaller timestep for stability with longer simulation

# SEC Auto-Tuning Configuration
use_auto_tuning = False    # Disable for now - need to restructure simulation as function first
auto_tune_method = 'differential_evolution'  # More exploratory optimization for better parameter space exploration

# Initial SEC Parameters (radical anti-clustering approach)
sec_params = SECParameters(
    rho_thresh=0.1,        # Very high threshold - most particles will repel
    dispersion_strength=0.3,   # Beyond normal bounds for maximum scatter
    clustering_strength=0.01,  # Minimal clustering
    branching_bias=0.002,      # Almost no branching
    centroid_strength=0.0      # NO centroid pull - let system be chaotic
)

# Analysis parameters
snapshot_interval = 100  # Less frequent snapshots
analysis_radius = 3.0
spatial_bounds = 20.0    # Larger simulation space

# Convergence detection parameters
convergence_window = 20   # Check convergence over last N snapshots
convergence_threshold = 0.005  # Slightly tighter threshold to allow more evolution

# --- Helper Functions (PyTorch CUDA) ---
def compute_density_cuda(positions, radius=2.5, chunk_size=2000):
    """Compute local density around each particle using CUDA with chunked processing (optimized)"""
    # Ensure positions are contiguous for better performance
    positions = positions.contiguous()
    n = positions.shape[0]
    densities = torch.zeros(n, device=positions.device)
    
    # Process in chunks to avoid memory issues
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_positions = positions[i:end_i].contiguous()
        
        # Compute distances for this chunk against all particles
        diff = chunk_positions.unsqueeze(1) - positions.unsqueeze(0)  # [chunk_size, n, 3]
        distances = torch.norm(diff, dim=2)  # [chunk_size, n]
        
        # Count neighbors within radius (excluding self)
        mask = (distances < radius) & (distances > 0)
        chunk_densities = mask.sum(dim=1).float()
        densities[i:end_i] = chunk_densities
    
    return densities

def compute_fractal_dimension_cuda(positions, max_radius=15.0, n_radii=20):
    """Estimate fractal dimension using box-counting method with pure CUDA (optimized)"""
    # Ensure positions are contiguous
    positions = positions.contiguous()
    
    radii = torch.logspace(-0.5, torch.log10(torch.tensor(max_radius, device=device)), n_radii, device=device)
    counts = []
    
    center = torch.mean(positions, dim=0)
    
    # Compute all distances at once for efficiency
    dists = torch.norm(positions - center, dim=1)
    
    for r in radii:
        # Count particles within radius r
        count = torch.sum(dists <= r).float()
        counts.append(torch.clamp(count, min=1.0))  # avoid log(0)
    
    # Fit log(count) vs log(radius) to get fractal dimension using PyTorch
    log_radii = torch.log(radii)
    log_counts = torch.log(torch.stack(counts))
    
    # Linear regression using PyTorch (faster than numpy)
    # y = ax + b, solve for a (fractal dimension)
    n = len(log_radii)
    sum_x = torch.sum(log_radii)
    sum_y = torch.sum(log_counts)
    sum_xx = torch.sum(log_radii * log_radii)
    sum_xy = torch.sum(log_radii * log_counts)
    
    # Calculate slope (fractal dimension)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator.abs() > 1e-6:
        fractal_dim = (n * sum_xy - sum_x * sum_y) / denominator
    else:
        fractal_dim = torch.tensor(2.0, device=device)
    
    return torch.abs(fractal_dim).item()

def compute_clustering_metrics_cuda(positions):
    """Compute various clustering and emergence metrics using CUDA"""
    n = positions.shape[0]
    if n < 2:
        return {}
    
    # Global statistics
    center = torch.mean(positions, dim=0)
    dists_from_center = torch.norm(positions - center, dim=1)
    
    # Sample pairwise distances for efficiency (too expensive for 50k particles)
    sample_size = min(5000, n)
    indices = torch.randperm(n, device=device)[:sample_size]
    sample_positions = positions[indices]
    
    # Pairwise distances for sample
    diff = sample_positions.unsqueeze(1) - sample_positions.unsqueeze(0)
    pairwise_dists = torch.norm(diff, dim=2)
    pairwise_dists = pairwise_dists[torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)]
    
    # Local density variance (emergence indicator)
    densities = compute_density_cuda(positions)
    density_variance = torch.var(densities).item()
    
    # 3D Spatial entropy using pure PyTorch (optimized for contiguous memory)
    # Divide space into 3D grid and compute occupancy entropy
    grid_size = 15  # Reduced for 3D
    
    # Ensure positions are contiguous for better performance
    positions_contiguous = positions.contiguous()
    
    # Get position bounds
    x_min, x_max = torch.min(positions_contiguous[:, 0]), torch.max(positions_contiguous[:, 0])
    y_min, y_max = torch.min(positions_contiguous[:, 1]), torch.max(positions_contiguous[:, 1])
    z_min, z_max = torch.min(positions_contiguous[:, 2]), torch.max(positions_contiguous[:, 2])
    
    # Add small epsilon to avoid boundary issues
    epsilon = 1e-6
    x_range = x_max - x_min + epsilon
    y_range = y_max - y_min + epsilon
    z_range = z_max - z_min + epsilon
    
    # Fast binning using integer division (more efficient than bucketize)
    x_normalized = (positions_contiguous[:, 0] - x_min) / x_range
    y_normalized = (positions_contiguous[:, 1] - y_min) / y_range
    z_normalized = (positions_contiguous[:, 2] - z_min) / z_range
    
    # Convert to grid indices
    x_indices = torch.clamp((x_normalized * grid_size).long(), 0, grid_size - 1)
    y_indices = torch.clamp((y_normalized * grid_size).long(), 0, grid_size - 1)
    z_indices = torch.clamp((z_normalized * grid_size).long(), 0, grid_size - 1)
    
    # Create linear indices for efficient histogram computation
    linear_indices = x_indices * grid_size * grid_size + y_indices * grid_size + z_indices
    
    # Compute histogram using bincount (much faster than manual loop)
    hist_flat = torch.bincount(linear_indices, minlength=grid_size**3).float()
    
    # Compute entropy using PyTorch
    hist_norm = hist_flat / torch.sum(hist_flat)
    hist_norm = hist_norm[hist_norm > 0]  # remove empty bins
    
    if len(hist_norm) > 1:
        spatial_entropy = -torch.sum(hist_norm * torch.log2(hist_norm + 1e-10)).item()
    else:
        spatial_entropy = 0.0
    
    # Fractal dimension
    fractal_dim = compute_fractal_dimension_cuda(positions)
    
    # Radial distribution analysis using PyTorch
    radial_distances = dists_from_center
    radial_std = torch.std(radial_distances).item()
    radial_mean = torch.mean(radial_distances).item()
    
    return {
        'mean_distance_from_center': radial_mean,
        'std_distance_from_center': radial_std,
        'mean_pairwise_distance': torch.mean(pairwise_dists).item(),
        'std_pairwise_distance': torch.std(pairwise_dists).item(),
        'density_variance': density_variance,
        'spatial_entropy': spatial_entropy,
        'fractal_dimension': fractal_dim,
        'num_particles': n,
        'max_density': torch.max(densities).item(),
        'min_density': torch.min(densities).item(),
        'mean_density': torch.mean(densities).item()
    }

def compute_trend_pytorch(values):
    """Compute linear trend using PyTorch"""
    if len(values) < 2:
        return 0.0
    
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    x_tensor = torch.arange(len(values), dtype=torch.float32, device=device)
    
    # Linear regression: y = ax + b, solve for a (trend)
    n = len(values)
    sum_x = torch.sum(x_tensor)
    sum_y = torch.sum(values_tensor)
    sum_xx = torch.sum(x_tensor * x_tensor)
    sum_xy = torch.sum(x_tensor * values_tensor)
    
    # Calculate slope (trend)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator.abs() > 1e-6:
        trend = (n * sum_xy - sum_x * sum_y) / denominator
    else:
        trend = torch.tensor(0.0, device=device)
    
    return trend.item()

def compute_emergence_signature(metrics_history):
    """Compute emergence signature from time series of metrics"""
    if len(metrics_history) < 2:
        return {}
    
    # Extract time series
    fractal_dims = [m.get('fractal_dimension', 0) for m in metrics_history]
    entropies = [m.get('spatial_entropy', 0) for m in metrics_history]
    density_vars = [m.get('density_variance', 0) for m in metrics_history]
    
    # Compute trends using PyTorch (faster than numpy)
    fractal_trend = compute_trend_pytorch(fractal_dims)
    entropy_trend = compute_trend_pytorch(entropies) 
    density_trend = compute_trend_pytorch(density_vars)
    
    # Emergence indicators
    final_fractal = fractal_dims[-1] if fractal_dims else 0
    final_entropy = entropies[-1] if entropies else 0
    final_density_var = density_vars[-1] if density_vars else 0
    
    # Enhanced emergence strength calculation for 3D
    # Account for increased complexity in 3D space and reduced particle count
    emergence_strength = min(1.0, 
                           (final_fractal / 3.0) * 0.35 +     # 3D fractal dimension
                           (final_entropy / 8.0) * 0.25 +     # Higher entropy in 3D
                           (final_density_var / 15000.0) * 0.25 +  # Scaled for 15k particles
                           (abs(fractal_trend) * 1000) * 0.15)     # Trend significance
    
    return {
        'fractal_dimension': final_fractal,
        'spatial_entropy': final_entropy,
        'density_variance': final_density_var,
        'fractal_trend': fractal_trend,
        'entropy_trend': entropy_trend,
        'density_trend': density_trend,
        'emergence_strength': emergence_strength,
        'complexity_index': final_fractal * final_entropy / 10.0  # New metric
    }

# Initialize simulation
torch.manual_seed(42)
positions = torch.rand((n_particles, 3), device=device) * 2 * spatial_bounds - spatial_bounds
velocities = torch.randn((n_particles, 3), device=device) * 0.05

# Store data for analysis
snapshots = []
metrics_history = []
energy_history = []

# --- SEC AUTO-TUNING ---
print("=== SEC AUTO-TUNING PHASE ===")
astro_fetcher = AstroDataFetcher(device=device)
real_positions, real_metadata = astro_fetcher.get_comparison_dataset('clusters', limit=3000)
real_metrics = astro_fetcher.compute_real_data_metrics(real_positions)

print(f"Real data loaded: {real_metadata['count']} {real_metadata['source']} points")
print(f"Target fractal dimension: {real_metrics['fractal_dimension']:.3f}")
print(f"Target spatial entropy: {real_metrics['spatial_entropy']:.3f}")
print(f"Target density variance: {real_metrics['density_variance']:.1f}")

# Create target metrics for auto-tuning
target_metrics = SECTargetMetrics(
    fractal_dimension=real_metrics['fractal_dimension'],
    spatial_entropy=real_metrics['spatial_entropy'],
    density_variance=real_metrics['density_variance']
)

if use_auto_tuning:
    print(f"\nInitializing SEC Auto-Tuning Engine...")
    sec_engine = SECAutoTuningEngine(device=device)
    
    print(f"Starting parameters: {sec_params}")
    
    tuning_result = sec_engine.optimize_parameters(sec_params, target_metrics, method=auto_tune_method)
    
    if tuning_result.success:
        sec_params = tuning_result.optimal_params
        print(f"✓ Auto-tuning successful!")
        print(f"  Balance improvement: {tuning_result.balance_improvement*100:.1f}%")
        print(f"  Predicted similarity: {tuning_result.predicted_similarity:.3f}")
        print(f"  Optimization time: {tuning_result.optimization_time:.1f}s")
        print(f"  Method: {tuning_result.method}")
        print(f"  Iterations: {tuning_result.iterations}")
        print(f"Tuned parameters: {sec_params}")
    else:
        print(f"⚠ Auto-tuning failed, using initial parameters")
else:
    print("SEC auto-tuning disabled, using initial parameters")

# Extract parameters for simulation
rho_thresh = sec_params.rho_thresh
dispersion_strength = sec_params.dispersion_strength
clustering_strength = sec_params.clustering_strength
branching_bias = sec_params.branching_bias
centroid_strength = sec_params.centroid_strength
cutoff_radius = 2.5  # Fixed for performance

print("Starting 3D Dark Matter Fractal Dispersal Simulation (CUDA + SEC Auto-Tuning)...")
print(f"Particles: {n_particles:,}, Steps: {n_steps}, dt: {dt}")
print(f"SEC Parameters: rho_thresh={rho_thresh:.4f}, clustering={clustering_strength:.4f}, branching={branching_bias:.4f}")
print(f"Spatial bounds: ±{spatial_bounds}, Device: {device}")

start_time = time.time()

# --- 3D Simulation Loop (CUDA Accelerated) ---
for step in range(n_steps):
    forces = torch.zeros_like(positions)
    global_centroid = torch.mean(positions, dim=0)
    
    # Calculate total kinetic energy for monitoring
    kinetic_energy = 0.5 * torch.sum(velocities**2).item()
    energy_history.append(kinetic_energy)
    
    # Compute densities for all particles
    densities = compute_density_cuda(positions)
    
    # Efficient force calculation using chunked vectorized operations
    chunk_size = 2000  # Process in chunks to manage memory
    
    for i in range(0, n_particles, chunk_size):
        end_i = min(i + chunk_size, n_particles)
        chunk_positions = positions[i:end_i]
        chunk_densities = densities[i:end_i]
        
        # Pairwise differences and distances for this chunk
        diff = chunk_positions.unsqueeze(1) - positions.unsqueeze(0)  # [chunk_size, n, 3]
        distances = torch.norm(diff, dim=2, keepdim=True)  # [chunk_size, n, 1]
        distances = distances + 1e-6  # Avoid division by zero
        
        # Mask for particles within cutoff radius
        mask = (distances.squeeze(-1) < cutoff_radius) & (distances.squeeze(-1) > 0)
        
        # Force directions (normalized differences)
        force_directions = diff / distances  # [chunk_size, n, 3]
        
        # Apply density-based rules
        low_density_mask = (chunk_densities < rho_thresh * n_particles).unsqueeze(1)  # [chunk_size, 1]
        
        # Repulsion for low density particles
        repulsion_forces = -force_directions / (distances**2)  # [chunk_size, n, 3]
        repulsion_forces = repulsion_forces * mask.unsqueeze(-1) * low_density_mask.unsqueeze(-1)
        repulsion_total = torch.sum(repulsion_forces, dim=1) * dispersion_strength
        
        # Attraction for high density particles
        attraction_forces = force_directions / (distances**2)  # [chunk_size, n, 3]
        attraction_forces = attraction_forces * mask.unsqueeze(-1) * (~low_density_mask).unsqueeze(-1)
        attraction_total = torch.sum(attraction_forces, dim=1) * clustering_strength
        
        # Store forces for this chunk
        forces[i:end_i] += repulsion_total + attraction_total
    
    # Centroid pull
    forces += centroid_strength * (global_centroid - positions)
    
    # 3D Fractal branching bias: radial outward push with spherical coordinates
    r = torch.norm(positions, dim=1, keepdim=True)
    radial_directions = positions / (r + 1e-6)
    forces += branching_bias * radial_directions
    
    # Update positions and velocities
    velocities += forces * dt
    positions += velocities * dt
    
    # Optional: Apply soft boundary conditions to keep particles in bounds
    boundary_force = 0.01
    out_of_bounds = torch.abs(positions) > spatial_bounds * 0.9
    boundary_forces = -torch.sign(positions) * out_of_bounds * boundary_force
    velocities += boundary_forces * dt
    
    # Collect snapshots and metrics
    if step % snapshot_interval == 0:
        # Store snapshot (sample for memory efficiency)
        sample_indices = torch.randperm(n_particles, device=device)[:2000]
        snapshots.append(positions[sample_indices].cpu().numpy())
        
        # Compute metrics
        metrics = compute_clustering_metrics_cuda(positions)
        metrics['step'] = step
        metrics['kinetic_energy'] = kinetic_energy
        metrics_history.append(metrics)
        
        # Clean up GPU memory periodically
        if step % (snapshot_interval * 2) == 0:
            torch.cuda.empty_cache()
        
        if step % (snapshot_interval * 4) == 0:  # Progress update
            elapsed = time.time() - start_time
            print(f"Step {step}: KE={kinetic_energy:.1f}, "
                  f"Fractal_D={metrics.get('fractal_dimension', 0):.3f}, "
                  f"Entropy={metrics.get('spatial_entropy', 0):.3f}, "
                  f"Time={elapsed:.1f}s")
            
            # Disable convergence check temporarily to let system evolve longer
            # if len(energy_history) >= convergence_window:
            #     recent_energies = energy_history[-convergence_window:]
            #     energy_change = abs(recent_energies[-1] - recent_energies[0]) / max(recent_energies[0], 1e-6)
            #     
            #     if energy_change < convergence_threshold:
            #         print(f"🎯 Convergence detected at step {step}! Energy change: {energy_change:.4f}")
            #         print(f"System has settled - stopping early for efficiency")
            #         break

simulation_time = time.time() - start_time
print(f"Simulation completed in {simulation_time:.1f} seconds!")

# Compute final emergence signature
emergence_sig = compute_emergence_signature(metrics_history)

# --- SIMULATION vs REAL DATA COMPARISON ---
print(f"\n=== SIMULATION vs REAL DATA COMPARISON ===")

# Compute similarity metrics using final emergence signature
sim_fractal = 1.0 - abs(emergence_sig.get('fractal_dimension', 0) - real_metrics['fractal_dimension']) / 3.0
sim_entropy = 1.0 - abs(emergence_sig.get('spatial_entropy', 0) - real_metrics['spatial_entropy']) / 8.0
sim_density = 1.0 - abs(emergence_sig.get('density_variance', 0) - real_metrics['density_variance']) / max(emergence_sig.get('density_variance', 1), real_metrics['density_variance'])
overall_similarity = (sim_fractal + sim_entropy + sim_density) / 3.0

print(f"Simulation fractal dimension: {emergence_sig.get('fractal_dimension', 0):.3f}")
print(f"Real fractal dimension: {real_metrics['fractal_dimension']:.3f}")
print(f"Fractal dimension similarity: {sim_fractal:.3f}")

print(f"Simulation spatial entropy: {emergence_sig.get('spatial_entropy', 0):.3f}")
print(f"Real spatial entropy: {real_metrics['spatial_entropy']:.3f}")
print(f"Spatial entropy similarity: {sim_entropy:.3f}")

print(f"Simulation density variance: {emergence_sig.get('density_variance', 0):.1f}")
print(f"Real density variance: {real_metrics['density_variance']:.1f}")
print(f"Density variance similarity: {sim_density:.3f}")

print(f"Overall similarity score: {overall_similarity:.3f}")

# Display auto-tuning results if used
if use_auto_tuning and 'tuning_result' in locals():
    print(f"\n=== SEC AUTO-TUNING RESULTS ===")
    print(f"Tuning method: {tuning_result.method}")
    print(f"Predicted similarity: {tuning_result.predicted_similarity:.3f}")
    print(f"Actual similarity achieved: {overall_similarity:.3f}")
    print(f"Prediction accuracy: {(1.0 - abs(tuning_result.predicted_similarity - overall_similarity)):.3f}")
    print(f"Balance improvement: {tuning_result.balance_improvement*100:.1f}%")

print("\n=== 3D EMERGENCE ANALYSIS ===")
print(f"Final Fractal Dimension: {emergence_sig.get('fractal_dimension', 0):.3f}")
print(f"Final Spatial Entropy: {emergence_sig.get('spatial_entropy', 0):.3f}")
print(f"Final Density Variance: {emergence_sig.get('density_variance', 0):.1f}")
print(f"Final Complexity Index: {emergence_sig.get('complexity_index', 0):.3f}")
print(f"Fractal Trend: {emergence_sig.get('fractal_trend', 0):.6f}")
print(f"Entropy Trend: {emergence_sig.get('entropy_trend', 0):.6f}")
print(f"Density Trend: {emergence_sig.get('density_trend', 0):.6f}")
print(f"Emergence Strength: {emergence_sig.get('emergence_strength', 0):.3f}")

# --- Enhanced 3D Visualization with Real Data Comparison ---
fig = plt.figure(figsize=(24, 20))

# 1. 3D Spatial snapshots (2x2 grid)
snapshot_indices = [0, len(snapshots)//3, 2*len(snapshots)//3, -1]
for i, idx in enumerate(snapshot_indices):
    if idx < len(snapshots):
        ax = fig.add_subplot(5, 4, i+1, projection='3d')
        pos = snapshots[idx]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=0.5, alpha=0.6, c='blue')
        ax.set_title(f"Simulation Step {idx * snapshot_interval}")
        ax.set_xlim([-spatial_bounds, spatial_bounds])
        ax.set_ylim([-spatial_bounds, spatial_bounds])
        ax.set_zlim([-spatial_bounds, spatial_bounds])

# 1b. Real data visualization
ax = fig.add_subplot(5, 4, 5, projection='3d')
real_pos_sample = real_positions[:2000].cpu().numpy()  # Sample for visualization
ax.scatter(real_pos_sample[:, 0], real_pos_sample[:, 1], real_pos_sample[:, 2], 
           s=0.8, alpha=0.7, c='green')
ax.set_title(f"Real Data: {real_metadata['source']}")
bounds = real_metadata['spatial_extent']
ax.set_xlim([-bounds, bounds])
ax.set_ylim([-bounds, bounds])
ax.set_zlim([-bounds, bounds])

# 2. Time series plots
plt.subplot(5, 4, 6)
steps = [m['step'] for m in metrics_history]
fractal_dims = [m.get('fractal_dimension', 0) for m in metrics_history]
plt.plot(steps, fractal_dims, 'b-', linewidth=2, label='Simulation')
plt.axhline(y=real_metrics['fractal_dimension'], color='green', linestyle='--', linewidth=2, label='Real Data')
plt.title('3D Fractal Dimension Evolution')
plt.xlabel('Step')
plt.ylabel('Fractal Dimension')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 7)
entropies = [m.get('spatial_entropy', 0) for m in metrics_history]
plt.plot(steps, entropies, 'r-', linewidth=2, label='Simulation')
plt.axhline(y=real_metrics['spatial_entropy'], color='green', linestyle='--', linewidth=2, label='Real Data')
plt.title('3D Spatial Entropy Evolution')
plt.xlabel('Step')
plt.ylabel('Spatial Entropy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 8)
density_vars = [m.get('density_variance', 0) for m in metrics_history]
plt.plot(steps, density_vars, 'g-', linewidth=2, label='Simulation')
plt.axhline(y=real_metrics['density_variance'], color='green', linestyle='--', linewidth=2, label='Real Data')
plt.title('Density Variance Evolution')
plt.xlabel('Step')
plt.ylabel('Density Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 9)
plt.plot(range(len(energy_history)), energy_history, 'm-', linewidth=2)
plt.title('Kinetic Energy Evolution')
plt.xlabel('Step')
plt.ylabel('Kinetic Energy')
plt.grid(True, alpha=0.3)

# 3. Final state analysis
ax = fig.add_subplot(5, 4, 10, projection='3d')
final_pos = snapshots[-1] if snapshots else positions.cpu().numpy()[:2000]
ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], s=1, alpha=0.7, c='red')
ax.set_title('Final 3D Configuration')
ax.set_xlim([-spatial_bounds, spatial_bounds])
ax.set_ylim([-spatial_bounds, spatial_bounds])
ax.set_zlim([-spatial_bounds, spatial_bounds])

# 4. Similarity comparison
plt.subplot(5, 4, 11)
comparison_metrics = ['Fractal\nDimension', 'Spatial\nEntropy', 'Density\nVariance', 'Overall\nSimilarity']
similarity_scores = [sim_fractal, sim_entropy, sim_density, overall_similarity]
bars = plt.bar(comparison_metrics, similarity_scores, color=['blue', 'red', 'green', 'purple'])
plt.title('Similarity to Real Data')
plt.ylabel('Similarity Score')
plt.ylim(0, 1)
for bar, val in zip(bars, similarity_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# 5. Enhanced metrics summary
plt.subplot(5, 4, 12)
metrics_names = ['Fractal\nDimension', 'Spatial\nEntropy', 'Density\nVariance', 'Emergence\nStrength', 'Complexity\nIndex']
metrics_values = [
    emergence_sig.get('fractal_dimension', 0) / 3.0,  # normalize for 3D
    emergence_sig.get('spatial_entropy', 0) / 8.0,    # normalize for 3D
    min(1.0, emergence_sig.get('density_variance', 0) / 15000.0),  # normalize for 15k
    emergence_sig.get('emergence_strength', 0),
    min(1.0, emergence_sig.get('complexity_index', 0) / 10.0)  # normalize
]
bars = plt.bar(metrics_names, metrics_values, color=['blue', 'red', 'green', 'purple', 'orange'])
plt.title('Final 3D Emergence Metrics')
plt.ylabel('Normalized Value')
plt.ylim(0, 1)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# 6. Density distribution analysis
plt.subplot(5, 4, 13)
final_metrics = metrics_history[-1] if metrics_history else {}
densities_final = [m.get('mean_density', 0) for m in metrics_history]
plt.plot(steps, densities_final, 'orange', linewidth=2)
plt.title('Mean Particle Density Evolution')
plt.xlabel('Step')
plt.ylabel('Mean Density')
plt.grid(True, alpha=0.3)

# 7. Performance metrics
plt.subplot(5, 4, 14)
performance_data = [
    ('Particles', n_particles),
    ('Steps', n_steps),
    ('Time (s)', simulation_time),
    ('Steps/sec', n_steps/simulation_time)
]
perf_names = [p[0] for p in performance_data]
perf_values = [p[1] for p in performance_data]

# Normalize values for display
normalized_values = [
    perf_values[0] / 100000,  # particles
    perf_values[1] / 5000,    # steps
    min(1.0, perf_values[2] / 100),  # time
    min(1.0, perf_values[3] / 50)    # steps/sec
]

bars = plt.bar(perf_names, normalized_values, color=['cyan', 'magenta', 'yellow', 'lime'])
plt.title('Performance Metrics')
plt.ylabel('Normalized Value')
plt.ylim(0, 1)
for bar, val, raw_val in zip(bars, normalized_values, perf_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{raw_val:.1f}', ha='center', va='bottom', fontsize=8)

# 8. Radial distribution comparison
plt.subplot(5, 4, 15)
if snapshots:
    final_snapshot = snapshots[-1]
    sim_radial = np.sqrt(np.sum(final_snapshot**2, axis=1))
    real_radial = np.sqrt(np.sum(real_pos_sample**2, axis=1))
    
    plt.hist(sim_radial, bins=30, alpha=0.7, color='blue', density=True, label='Simulation')
    plt.hist(real_radial, bins=30, alpha=0.7, color='green', density=True, label='Real Data')
    plt.title('Radial Distribution Comparison')
    plt.xlabel('Distance from Origin')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 9. Cross-sectional comparison
plt.subplot(5, 4, 16)
if snapshots:
    final_snapshot = snapshots[-1]
    sim_slice = final_snapshot[np.abs(final_snapshot[:, 2]) < 2]
    real_slice = real_pos_sample[np.abs(real_pos_sample[:, 2]) < 2]
    
    plt.scatter(sim_slice[:, 0], sim_slice[:, 1], s=0.5, alpha=0.6, c='blue', label='Simulation')
    plt.scatter(real_slice[:, 0], real_slice[:, 1], s=0.5, alpha=0.6, c='green', label='Real Data')
    plt.title('XY Cross-Section Comparison')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

# 10. Emergence strength evolution
plt.subplot(5, 4, 17)
if len(metrics_history) > 1:
    emergence_strengths = []
    for i in range(1, len(metrics_history)):
        temp_sig = compute_emergence_signature(metrics_history[:i+1])
        emergence_strengths.append(temp_sig.get('emergence_strength', 0))
    
    plt.plot(steps[1:], emergence_strengths, 'purple', linewidth=2)
    plt.title('Emergence Strength Evolution')
    plt.xlabel('Step')
    plt.ylabel('Emergence Strength')
    plt.grid(True, alpha=0.3)

# 11. Complexity evolution
plt.subplot(5, 4, 18)
complexity_indices = []
for metrics in metrics_history:
    fractal = metrics.get('fractal_dimension', 0)
    entropy = metrics.get('spatial_entropy', 0)
    complexity = fractal * entropy / 10.0
    complexity_indices.append(complexity)

plt.plot(steps, complexity_indices, 'darkgreen', linewidth=2)
plt.title('Complexity Index Evolution')
plt.xlabel('Step')
plt.ylabel('Complexity Index')
plt.grid(True, alpha=0.3)

# 12. SEC Auto-Tuning Results Visualization
plt.subplot(5, 4, 19)
if use_auto_tuning and 'tuning_result' in locals() and tuning_result.success:
    param_names = ['Rho\nThresh', 'Dispersion\nStrength', 'Clustering\nStrength', 'Branching\nBias']
    
    # Original parameters from initial sec_params used for tuning
    if 'sec_engine' in locals():
        # Get initial parameters that were used for tuning
        initial_params = SECParameters(rho_thresh=0.02, dispersion_strength=0.08, 
                                     clustering_strength=0.12, branching_bias=0.03)
        current_vals = initial_params.to_array()
    else:
        current_vals = [0.02, 0.08, 0.12, 0.03]  # Default initial values
    
    optimal_vals = tuning_result.optimal_params.to_array()
    
    x_pos = np.arange(len(param_names))
    width = 0.35
    
    # Normalize for visualization
    all_vals = list(current_vals) + list(optimal_vals)
    max_val = max(all_vals)
    current_norm = [v/max_val for v in current_vals]
    optimal_norm = [v/max_val for v in optimal_vals]
    
    plt.bar(x_pos - width/2, current_norm, width, label='Initial', color='blue', alpha=0.7)
    plt.bar(x_pos + width/2, optimal_norm, width, label='SEC Optimized', color='orange', alpha=0.7)
    plt.xlabel('Parameters')
    plt.ylabel('Normalized Values')
    plt.title(f'SEC Auto-Tuning Results\n(Similarity: {overall_similarity:.3f}, Predicted: {tuning_result.predicted_similarity:.3f})')
    plt.xticks(x_pos, param_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'SEC Auto-Tuning\nDisabled/Failed', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('SEC Auto-Tuning Results')

# 13. Real vs Sim metrics comparison
plt.subplot(5, 4, 20)
sim_values = [emergence_sig.get('fractal_dimension', 0), 
              emergence_sig.get('spatial_entropy', 0),
              emergence_sig.get('density_variance', 0) / 1000]  # scale for vis
real_values = [real_metrics['fractal_dimension'],
               real_metrics['spatial_entropy'], 
               real_metrics['density_variance'] / 1000]  # scale for vis

# Predicted values using SEC engine predictions
if use_auto_tuning and 'tuning_result' in locals() and tuning_result.success:
    params = tuning_result.optimal_params.to_array()
    pred_values = [
        1.2 + (params[3] * 40) + (params[2] * 5) - (params[1] * 2),  # fractal
        6.0 - (params[2] * 30) - (params[3] * 50) + (params[1] * 10),  # entropy  
        (200 + (params[2] * 2000) + (params[0] * 5000) - (params[1] * 1000)) / 1000  # density scaled
    ]
else:
    pred_values = sim_values

metric_labels = ['Fractal D', 'Entropy', 'Density V/1000']
x_pos = np.arange(len(metric_labels))
width = 0.25

plt.bar(x_pos - width, sim_values, width, label='Current Sim', color='blue', alpha=0.7)
plt.bar(x_pos, real_values, width, label='Real Data', color='green', alpha=0.7)
if use_auto_tuning and 'tuning_result' in locals() and tuning_result.success:
    plt.bar(x_pos + width, pred_values, width, label='SEC Predicted', color='orange', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Metric Comparison with SEC Auto-Tuning')
plt.xticks(x_pos, metric_labels)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('darkmatter_3d_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== 3D SIMULATION SUMMARY ===")
print(f"Total particles: {n_particles:,}")
print(f"Total snapshots collected: {len(snapshots)}")
print(f"Total metrics calculated: {len(metrics_history)}")
print(f"Simulation time: {simulation_time:.1f} seconds")
print(f"Performance: {n_steps/simulation_time:.1f} steps/second")
print(f"Energy change: {energy_history[0]:.1f} -> {energy_history[-1]:.1f} ({(energy_history[-1]/energy_history[0]-1)*100:+.1f}%)")
print(f"Final emergence assessment: {'STRONG' if emergence_sig.get('emergence_strength', 0) > 0.6 else 'MODERATE' if emergence_sig.get('emergence_strength', 0) > 0.3 else 'WEAK'}")
print(f"Final complexity index: {emergence_sig.get('complexity_index', 0):.3f}")
print(f"Similarity to real dark matter structures: {overall_similarity:.3f} ({'HIGH' if overall_similarity > 0.7 else 'MODERATE' if overall_similarity > 0.4 else 'LOW'})")
print(f"Real data source: {real_metadata['source']} ({real_metadata['count']} points)")

if use_auto_tuning and 'tuning_result' in locals() and tuning_result.success:
    print(f"\n=== SEC AUTO-TUNING SUMMARY ===")
    print(f"SEC optimization successful with {tuning_result.balance_improvement*100:.1f}% balance improvement")
    print(f"Predicted similarity after tuning: {tuning_result.predicted_similarity:.3f}")
    print(f"Actual similarity achieved: {overall_similarity:.3f}")
    print(f"Prediction accuracy: {(1.0 - abs(tuning_result.predicted_similarity - overall_similarity)):.3f}")
    print(f"Optimization method: {tuning_result.method}")
    print(f"Optimization time: {tuning_result.optimization_time:.1f}s")
    print(f"Recommended parameters for reproduction:")
    print(f"  rho_thresh = {tuning_result.optimal_params.rho_thresh:.4f}")
    print(f"  dispersion_strength = {tuning_result.optimal_params.dispersion_strength:.4f}")
    print(f"  clustering_strength = {tuning_result.optimal_params.clustering_strength:.4f}")
    print(f"  branching_bias = {tuning_result.optimal_params.branching_bias:.4f}")
else:
    print(f"\n=== SEC AUTO-TUNING ===")
    print(f"SEC auto-tuning was disabled or failed - using initial parameters")

# Memory cleanup
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"GPU memory cleaned up")
