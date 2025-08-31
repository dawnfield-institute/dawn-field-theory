import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import time
from typing import Dict, List, Tuple, Any

# --- Parameters ---
n_particles = 200        # number of particles (reduced for speed)
n_steps = 800            # number of simulation steps
dt = 0.02                # timestep

# Recursive/Fractal Dispersal Parameters
rho_thresh = 0.05
dispersion_strength = 0.1
clustering_strength = 0.15
centroid_strength = 0.02
branching_bias = 0.05     # pushes towards fractal-like dispersal
cutoff_radius = 3.0

# Analysis parameters
snapshot_interval = 50   # steps between snapshots for analysis
analysis_radius = 5.0    # radius for local analysis

np.random.seed(42)
positions = np.random.uniform(-5, 5, (n_particles, 2))
velocities = np.random.normal(0, 0.1, (n_particles, 2))

# --- Helper Functions ---
def compute_density(i, positions, radius=cutoff_radius):
    """Compute local density around particle i"""
    dists = np.linalg.norm(positions - positions[i], axis=1)
    return np.sum(dists < radius) - 1

def compute_fractal_dimension(positions, max_radius=10.0, n_radii=20):
    """Estimate fractal dimension using box-counting method"""
    radii = np.logspace(-1, np.log10(max_radius), n_radii)
    counts = []
    
    center = np.mean(positions, axis=0)
    
    for r in radii:
        # Count particles within radius r
        dists = np.linalg.norm(positions - center, axis=1)
        count = np.sum(dists <= r)
        counts.append(max(1, count))  # avoid log(0)
    
    # Fit log(count) vs log(radius) to get fractal dimension
    log_radii = np.log(radii)
    log_counts = np.log(counts)
    
    # Linear regression to get slope (fractal dimension)
    A = np.vstack([log_radii, np.ones(len(log_radii))]).T
    fractal_dim, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
    
    return abs(fractal_dim)

def compute_clustering_metrics(positions):
    """Compute various clustering and emergence metrics"""
    n = len(positions)
    if n < 2:
        return {}
    
    # Global statistics
    center = np.mean(positions, axis=0)
    dists_from_center = np.linalg.norm(positions - center, axis=1)
    
    # Pairwise distances
    pairwise_dists = pdist(positions)
    
    # Local density variance (emergence indicator)
    densities = [compute_density(i, positions) for i in range(n)]
    density_variance = np.var(densities)
    
    # Spatial entropy
    # Divide space into grid and compute occupancy entropy
    grid_size = 20
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_bins, y_bins])
    hist_norm = hist / np.sum(hist)
    hist_norm = hist_norm[hist_norm > 0]  # remove empty bins
    spatial_entropy = entropy(hist_norm.flatten())
    
    # Fractal dimension
    fractal_dim = compute_fractal_dimension(positions)
    
    return {
        'mean_distance_from_center': np.mean(dists_from_center),
        'std_distance_from_center': np.std(dists_from_center),
        'mean_pairwise_distance': np.mean(pairwise_dists),
        'std_pairwise_distance': np.std(pairwise_dists),
        'density_variance': density_variance,
        'spatial_entropy': spatial_entropy,
        'fractal_dimension': fractal_dim,
        'num_particles': n
    }

def compute_emergence_signature(metrics_history):
    """Compute emergence signature from time series of metrics"""
    if len(metrics_history) < 2:
        return {}
    
    # Extract time series
    fractal_dims = [m.get('fractal_dimension', 0) for m in metrics_history]
    entropies = [m.get('spatial_entropy', 0) for m in metrics_history]
    density_vars = [m.get('density_variance', 0) for m in metrics_history]
    
    # Compute trends and stability
    fractal_trend = np.polyfit(range(len(fractal_dims)), fractal_dims, 1)[0] if len(fractal_dims) > 1 else 0
    entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0
    
    # Emergence indicators
    final_fractal = fractal_dims[-1] if fractal_dims else 0
    final_entropy = entropies[-1] if entropies else 0
    final_density_var = density_vars[-1] if density_vars else 0
    
    # Simple emergence strength calculation
    # Higher fractal dimension + stable entropy + structured density variation = emergence
    emergence_strength = min(1.0, (final_fractal / 2.0) * 0.4 + 
                            (final_entropy / 5.0) * 0.3 + 
                            (final_density_var / 10.0) * 0.3)
    
    return {
        'fractal_dimension': final_fractal,
        'spatial_entropy': final_entropy,
        'density_variance': final_density_var,
        'fractal_trend': fractal_trend,
        'entropy_trend': entropy_trend,
        'emergence_strength': emergence_strength
    }

# Initialize simulation
np.random.seed(42)
positions = np.random.uniform(-5, 5, (n_particles, 2))
velocities = np.random.normal(0, 0.1, (n_particles, 2))

# Store data for analysis
snapshots = []
metrics_history = []
energy_history = []

print("Starting Dark Matter Fractal Dispersal Simulation...")
print(f"Particles: {n_particles}, Steps: {n_steps}, dt: {dt}")
print(f"Parameters: rho_thresh={rho_thresh}, clustering={clustering_strength}, branching={branching_bias}")

# Store snapshots for visualization
snapshots = []

# --- Simulation Loop ---
for step in range(n_steps):
    forces = np.zeros_like(positions)
    global_centroid = np.mean(positions, axis=0)
    
    for i in range(n_particles):
        rho = compute_density(i, positions)
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-5
        mask = (dists.flatten() < cutoff_radius)
        diffs = diffs[mask]
        dists = dists[mask].reshape(-1,1)
        
        if rho < rho_thresh * n_particles:
            repulsion = np.sum(-diffs / dists**2, axis=0)
            forces[i] += dispersion_strength * repulsion
# --- Simulation Loop ---
for step in range(n_steps):
    forces = np.zeros_like(positions)
    global_centroid = np.mean(positions, axis=0)
    
    # Calculate total kinetic energy for monitoring
    kinetic_energy = 0.5 * np.sum(velocities**2)
    energy_history.append(kinetic_energy)
    
    for i in range(n_particles):
        rho = compute_density(i, positions)
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-5
        mask = (dists.flatten() < cutoff_radius)
        diffs = diffs[mask]
        dists = dists[mask].reshape(-1,1)
        
        if rho < rho_thresh * n_particles:
            # Low density: dispersal force
            repulsion = np.sum(-diffs / dists**2, axis=0)
            forces[i] += dispersion_strength * repulsion
        else:
            # High density: clustering force
            attraction = np.sum(diffs / dists**2, axis=0)
            forces[i] += clustering_strength * attraction
        
        # Centroid pull
        forces[i] += centroid_strength * (global_centroid - positions[i])
        
        # Fractal branching bias: slight outward directional push
        angle = np.arctan2(positions[i,1], positions[i,0])
        branch_vec = np.array([np.cos(angle), np.sin(angle)])
        forces[i] += branching_bias * branch_vec
    
    # Update positions and velocities
    velocities += forces * dt
    positions += velocities * dt
    
    # Collect snapshots and metrics
    if step % snapshot_interval == 0:
        snapshots.append(positions.copy())
        metrics = compute_clustering_metrics(positions)
        metrics['step'] = step
        metrics['kinetic_energy'] = kinetic_energy
        metrics_history.append(metrics)
        
        if step % (snapshot_interval * 4) == 0:  # Progress update
            print(f"Step {step}: KE={kinetic_energy:.3f}, "
                  f"Fractal_D={metrics.get('fractal_dimension', 0):.3f}, "
                  f"Entropy={metrics.get('spatial_entropy', 0):.3f}")

print("Simulation completed!")

# Compute final emergence signature
emergence_sig = compute_emergence_signature(metrics_history)
print("\n=== EMERGENCE ANALYSIS ===")
print(f"Final Fractal Dimension: {emergence_sig.get('fractal_dimension', 0):.3f}")
print(f"Final Spatial Entropy: {emergence_sig.get('spatial_entropy', 0):.3f}")
print(f"Final Density Variance: {emergence_sig.get('density_variance', 0):.3f}")
print(f"Fractal Trend: {emergence_sig.get('fractal_trend', 0):.6f}")
print(f"Entropy Trend: {emergence_sig.get('entropy_trend', 0):.6f}")
print(f"Emergence Strength: {emergence_sig.get('emergence_strength', 0):.3f}")

# --- Enhanced Visualization ---
fig = plt.figure(figsize=(20, 12))

# 1. Spatial snapshots
ax1 = plt.subplot(3, 4, (1, 4))
n_snapshots = min(4, len(snapshots))
for i, (pos, step_num) in enumerate(zip(snapshots[:n_snapshots], range(0, n_snapshots * snapshot_interval, snapshot_interval))):
    plt.subplot(3, 4, i+1)
    plt.scatter(pos[:,0], pos[:,1], s=3, alpha=0.7, c='blue')
    plt.title(f"Step {step_num}")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

# 2. Time series plots
plt.subplot(3, 4, 5)
steps = [m['step'] for m in metrics_history]
fractal_dims = [m.get('fractal_dimension', 0) for m in metrics_history]
plt.plot(steps, fractal_dims, 'b-', linewidth=2)
plt.title('Fractal Dimension Evolution')
plt.xlabel('Step')
plt.ylabel('Fractal Dimension')
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 6)
entropies = [m.get('spatial_entropy', 0) for m in metrics_history]
plt.plot(steps, entropies, 'r-', linewidth=2)
plt.title('Spatial Entropy Evolution')
plt.xlabel('Step')
plt.ylabel('Spatial Entropy')
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 7)
density_vars = [m.get('density_variance', 0) for m in metrics_history]
plt.plot(steps, density_vars, 'g-', linewidth=2)
plt.title('Density Variance Evolution')
plt.xlabel('Step')
plt.ylabel('Density Variance')
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 8)
plt.plot(range(len(energy_history)), energy_history, 'm-', linewidth=2)
plt.title('Kinetic Energy Evolution')
plt.xlabel('Step')
plt.ylabel('Kinetic Energy')
plt.grid(True, alpha=0.3)

# 3. Final state analysis
plt.subplot(3, 4, 9)
final_pos = snapshots[-1] if snapshots else positions
plt.scatter(final_pos[:,0], final_pos[:,1], s=10, alpha=0.7, c='red')
plt.title('Final Configuration')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# 4. Emergence metrics summary
plt.subplot(3, 4, 10)
metrics_names = ['Fractal\nDimension', 'Spatial\nEntropy', 'Density\nVariance', 'Emergence\nStrength']
metrics_values = [
    emergence_sig.get('fractal_dimension', 0),
    emergence_sig.get('spatial_entropy', 0) / 5,  # normalize for visualization
    emergence_sig.get('density_variance', 0) / 10,  # normalize for visualization
    emergence_sig.get('emergence_strength', 0)
]
bars = plt.bar(metrics_names, metrics_values, color=['blue', 'red', 'green', 'purple'])
plt.title('Final Emergence Metrics')
plt.ylabel('Normalized Value')
plt.ylim(0, 1)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom')

# 5. Phase space plot (velocity vs position)
plt.subplot(3, 4, 11)
plt.scatter(np.linalg.norm(final_pos, axis=1), np.linalg.norm(velocities, axis=1), 
           s=8, alpha=0.6, c='orange')
plt.xlabel('Distance from Origin')
plt.ylabel('Velocity Magnitude')
plt.title('Phase Space (Final State)')
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 12)
if len(metrics_history) > 1:
    # Emergence strength over time
    emergence_strengths = []
    for i in range(1, len(metrics_history)):
        temp_sig = compute_emergence_signature(metrics_history[:i+1])
        emergence_strengths.append(temp_sig.get('emergence_strength', 0))
    
    plt.plot(steps[1:], emergence_strengths, 'purple', linewidth=2)
    plt.title('Emergence Strength Evolution')
    plt.xlabel('Step')
    plt.ylabel('Emergence Strength')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== SIMULATION SUMMARY ===")
print(f"Total snapshots collected: {len(snapshots)}")
print(f"Total metrics calculated: {len(metrics_history)}")
print(f"Energy change: {energy_history[0]:.3f} -> {energy_history[-1]:.3f} ({(energy_history[-1]/energy_history[0]-1)*100:+.1f}%)")
print(f"Final emergence assessment: {'STRONG' if emergence_sig.get('emergence_strength', 0) > 0.6 else 'MODERATE' if emergence_sig.get('emergence_strength', 0) > 0.3 else 'WEAK'}")
