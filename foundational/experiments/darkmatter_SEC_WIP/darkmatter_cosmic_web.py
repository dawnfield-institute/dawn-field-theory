#!/usr/bin/env python3
"""
🌌 DARK MATTER COSMIC WEB SIMULATION
Large-scale structure formation: Dark matter distribution at cosmic web scales

This simulation focuses on:
- COSMIC WEB SCALE (not galaxy scale)
- DARK MATTER DISTRIBUTION (not galaxy formation)
- FILAMENTARY STRUCTURE (not dense clustering)

Key differences from galaxy simulation:
- MUCH WEAKER gravitational strength (cosmic web scale)
- HIGHER VISCOSITY to match larger scale physics
- LOWER DENSITY for web-like structure
- FOCUS ON FILAMENTS not dense nodes

Based on darkmatter_temporal_gradient.py but adapted for cosmic web scales
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import hashlib
import os
from astro_data_fetcher import AstroDataFetcher
from sec_auto_tuning_engine import SECParameters

# Ensure debug plots directory exists
os.makedirs('debug_plots_cosmic_web', exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 🌌 COSMIC WEB SCALE PHYSICS PARAMETERS
# Key insight: We need MUCH weaker gravity and HIGHER viscosity for cosmic web scales

# Temporal progression parameters - Cosmic Web Scale
n_particles = 25000          # Same particle count but representing cosmic web nodes
n_redshift_bins = 5          # Number of evolutionary stages  
steps_per_bin = 1000         # Evolution steps per redshift bin
total_steps = n_redshift_bins * steps_per_bin

# 🔑 COSMIC WEB PHYSICS - The key changes for large-scale structure
G = 4.498e-8  # MUCH WEAKER gravitational constant (cosmic web scale, not galaxy scale)
dt = 0.01
damping = 0.999
noise_strength = 0.0001
spatial_bounds = 50.0  # LARGER spatial scale for cosmic web

# Cosmological parameters (same as galaxy sim)
omega_m = 0.3            # Matter density parameter
omega_lambda = 0.7       # Dark energy density parameter  
H0 = 67.4               # Hubble constant (km/s/Mpc)

# Convert Hubble constant to simulation units
# H0 = 67.4 km/s/Mpc = 67.4 km/s/1000kpc = 0.0674 km/s/kpc
# 1 Gyr = 3.156e16 s, 1 kpc = 3.086e16 km
# H0_sim = H0 * (km/s/kpc) * (kpc/km) * (s/Gyr) = H0 * (1/3.086e16) * (3.156e16) = H0 * 1.022e-6
H0_sim = H0 * 1.022e-6  # Hubble constant in simulation units (1/Gyr)

def scale_factor(redshift):
    """Compute scale factor a(z) = 1/(1+z)"""
    if isinstance(redshift, torch.Tensor):
        return 1.0 / (1.0 + redshift)
    else:
        return 1.0 / (1.0 + redshift)

def hubble_parameter(redshift):
    """Compute H(z) = H0 * sqrt(Ωm*(1+z)³ + ΩΛ) in simulation units"""
    z_plus_1 = 1.0 + redshift
    if isinstance(redshift, torch.Tensor):
        return H0_sim * torch.sqrt(omega_m * z_plus_1**3 + omega_lambda)
    else:
        return H0_sim * (omega_m * z_plus_1**3 + omega_lambda)**0.5

def conformal_time_derivative(redshift):
    """Compute da/dt in conformal time for co-moving coordinates"""
    return hubble_parameter(redshift) * scale_factor(redshift)

# 🌌 COSMIC WEB TIDAL FORCES - Stronger for large-scale structure
def compute_cosmic_web_tidal_tensor(positions, redshift, device):
    """
    Compute tidal force tensor optimized for cosmic web formation.
    Stronger anisotropy to promote filamentary structure at large scales.
    """
    # Tidal strength scales with cosmic evolution - STRONGER for cosmic web
    a = scale_factor(redshift)
    tidal_strength = 5e-7 * (1.0 + redshift)**0.5  # 50x stronger than galaxy scale
    
    # Create STRONG anisotropic tidal field for cosmic web structure
    # Major axis: x-direction (main cosmic web spine)
    # Minor axes: y,z-directions (strong compression for filament formation)
    tidal_tensor = torch.zeros((3, 3), device=device)
    tidal_tensor[0, 0] = tidal_strength * 3.0   # Strong stretching along spine
    tidal_tensor[1, 1] = -tidal_strength * 1.5  # Strong compression  
    tidal_tensor[2, 2] = -tidal_strength * 1.5  # Strong compression
    
    return tidal_tensor

def apply_cosmic_web_tidal_forces(positions, velocities, redshift, dt, device):
    """
    Apply cosmic web tidal forces optimized for large-scale filamentary structure.
    """
    tidal_tensor = compute_cosmic_web_tidal_tensor(positions, redshift, device)
    
    # Apply tidal acceleration: a_tidal = T · r 
    tidal_forces = torch.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            tidal_forces[:, i] += tidal_tensor[i, j] * positions[:, j]
    
    # STRONGER scaling for cosmic web (not subtle like galaxy scale)
    tidal_forces *= 0.5  # Strong external cosmic web influence
    
    return tidal_forces

# Universal Physical Constants for Emergent Gravity Anchoring
c_light = 299792.458  # Speed of light (km/s)
k_B = 1.380649e-23   # Boltzmann constant (J/K)
m_p = 1.672621898e-27  # Proton mass (kg)
t_U = 13.8e9 * 365.25 * 24 * 3600  # Age of universe (seconds)
R_U = c_light * t_U / 1000  # Universe horizon radius (kpc)
T_CMB = 2.725  # CMB temperature (K)

# 🔑 COSMIC WEB EMERGENT GRAVITY PARAMETERS
# Much higher viscosity (α) for cosmic web scale physics
# Lower force coupling (β) for weaker gravitational clustering

# α: MUCH HIGHER viscosity coefficient for cosmic web scale
alpha_base = 1.0 / (c_light * t_U * 1e3)  # 1000x higher viscosity for cosmic web

# β: MUCH LOWER force coefficient for weaker clustering  
beta_base = k_B * T_CMB / (m_p * c_light**2 * R_U**2) * 1e-3  # 1000x weaker for web structure

print(f"🌌 Cosmic Web Emergent Gravity Parameters:")
print(f"α_base = {alpha_base:.6e} (cosmic web viscosity - 1000x higher)")
print(f"β_base = {beta_base:.6e} (weak clustering - 1000x lower)")
print(f"QBE Controller: Optimized for large-scale filamentary structure")
print(f"Physical basis: High viscosity + weak clustering = cosmic web")

# 🔑 COSMIC WEB SEC PARAMETERS
sec_params = SECParameters(
    rho_thresh=0.1,        # MUCH LOWER density threshold for web structure
    clustering_strength=0.05,  # MUCH WEAKER clustering for filaments  
    branching_bias=0.02,   # VERY LOW branching for clean filaments
    centroid_strength=0.0  # NO centroid pull - pure filamentary flow
)

print(f"🌌 Cosmic Web SEC Parameters:")
print(f"rho_thresh={sec_params.rho_thresh:.4f} (low density for web structure)")
print(f"clustering={sec_params.clustering_strength:.4f} (weak for filaments)")
print(f"branching={sec_params.branching_bias:.4f} (minimal branching)")
print(f"Cosmic Web Physics: Weak clustering + high viscosity = filamentary structure")

def apply_sec_forces_cuda(positions, velocities, sec_params, n_particles, time_step=0):
    """
    Simplified SEC forces for cosmic web scale - much weaker than galaxy scale.
    Focuses on weak clustering and filamentary structure formation.
    """
    forces = torch.zeros_like(positions)
    
    # Simplified gravitational forces with cosmic web parameters
    for i in range(min(n_particles, 1000)):  # Limit for performance
        r_vec = positions - positions[i:i+1]
        r_dist = torch.norm(r_vec, dim=1) + 1e-6  # Softening
        
        # Very weak gravitational force for cosmic web scale
        f_magnitude = G * sec_params.clustering_strength / (r_dist**2 + sec_params.rho_thresh)
        
        # Apply cosmic web viscosity (high viscosity for large scale)
        viscosity_factor = torch.exp(-alpha_base * r_dist)
        
        # Landauer scaffolding for cosmic web (weak clustering)
        landauer_factor = 1.0 / (1.0 + beta_base * r_dist**2)
        
        # Combine forces with cosmic web scaling
        f_vec = f_magnitude.unsqueeze(1) * r_vec
        f_vec *= viscosity_factor.unsqueeze(1) * landauer_factor.unsqueeze(1)
        
        # Set self-force to zero
        f_vec[0] = 0.0
        
        forces[i] = torch.sum(f_vec, dim=0)
    
    return forces

def compute_fractal_dimension_cuda(positions):
    """Simplified fractal dimension computation for cosmic web structure"""
    n_particles = positions.shape[0]
    if n_particles < 10:
        return torch.tensor(1.5)
    
    # Use correlation dimension approximation
    # Sample subset for performance
    sample_size = min(n_particles, 2000)
    sample_indices = torch.randperm(n_particles)[:sample_size]
    sample_positions = positions[sample_indices]
    
    distances = torch.cdist(sample_positions, sample_positions)
    distances = distances[distances > 0]  # Remove self-distances
    
    if len(distances) == 0:
        return torch.tensor(1.5)
    
    median_dist = torch.median(distances)
    count_close = torch.sum(distances < median_dist * 0.5).float()
    
    if count_close > 0:
        # Rough approximation of fractal dimension
        fractal_dim = torch.log(count_close) / torch.log(2.0)
        return torch.clamp(fractal_dim, 1.0, 3.0)
    else:
        return torch.tensor(1.5)

def compute_clustering_metrics_cuda(positions):
    """Simplified clustering metrics for cosmic web structure"""
    n_particles = positions.shape[0]
    
    # Spatial entropy calculation
    # Discretize space and compute entropy
    bins = 20
    try:
        hist = torch.histogramdd(positions, bins=bins, 
                               range=[(-spatial_bounds, spatial_bounds)]*3)[0]
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove empty bins
        
        if len(hist) == 0:
            spatial_entropy = torch.tensor(0.0)
        else:
            # Normalize to probabilities
            probs = hist.float() / torch.sum(hist)
            spatial_entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    except:
        spatial_entropy = torch.tensor(5.0)  # Default value
    
    # Density variance calculation
    try:
        # Simple density variance using position variance
        pos_var = torch.var(positions, dim=0)
        density_variance = torch.mean(pos_var)
    except:
        density_variance = torch.tensor(1000.0)  # Default value
    
    return spatial_entropy, density_variance

# Oscillatory Hum - Cosmic Web Scale  
OSCILLATORY_HUM_ENABLED = True
HUM_BASE_FREQ = 0.05      # SLOWER cosmic web oscillations
HUM_AMPLITUDE = 0.01      # WEAKER modulation for large scale
HUM_SEC_COUPLING = 0.005  # REDUCED SEC coupling

# Add these functions after the tidal force functions and before the main function

def generate_sha_entropy_seed(hash_input, shape):
    """
    Generate deterministic pseudo-random field using SHA-256 entropy seeding.
    Creates evenly distributed initial conditions based on cryptographic hash.
    From legacy cosmo.py - proven method for uniform cosmic structure seeding.
    """
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape)

def generate_cosmic_web_initial_conditions(n_particles, spatial_bounds=50.0, device='cuda'):
    """
    Generate initial conditions optimized for cosmic web formation using SHA entropy seeding.
    Lower density, evenly distributed using cryptographic entropy for realistic cosmic structure.
    """
    print(f"🌌 Generating SHA-seeded cosmic web initial conditions...")
    print(f"   Approach: Cryptographic entropy for even distribution")
    print(f"   Purpose: Realistic cosmic web filament formation (not galaxy clustering)")
    
    # Use SHA-256 entropy seeding for evenly distributed cosmic structure
    # Different hash inputs create different but deterministic patterns
    hash_positions = "CIMM:cosmic_web:positions:dark_matter"
    hash_velocities = "CIMM:cosmic_web:velocities:initial"
    
    # Generate positions using SHA entropy (evenly distributed)
    pos_entropy = generate_sha_entropy_seed(hash_positions, (n_particles, 3))
    
    # Convert to simulation coordinates (centered around origin)
    positions = torch.tensor(pos_entropy, device=device, dtype=torch.float32)
    positions = (positions - 0.5) * 2.0 * spatial_bounds  # Scale to [-bounds, +bounds]
    
    # Generate velocities using separate SHA entropy
    vel_entropy = generate_sha_entropy_seed(hash_velocities, (n_particles, 3))
    velocities = torch.tensor(vel_entropy, device=device, dtype=torch.float32)
    velocities = (velocities - 0.5) * 0.002  # Very small initial velocities for cosmic web
    
    # Add subtle anisotropic perturbations to seed filamentary structure
    # This mimics quantum fluctuations in the early universe
    hash_perturbations = "CIMM:cosmic_web:quantum_fluctuations"
    perturb_entropy = generate_sha_entropy_seed(hash_perturbations, (n_particles, 3))
    perturbations = torch.tensor(perturb_entropy, device=device, dtype=torch.float32)
    
    # Apply anisotropic perturbations (stronger along x-axis for filament seeding)
    perturbations[:, 0] *= 0.1  # Weak perturbations along main axis
    perturbations[:, 1] *= 0.05  # Stronger compression along y
    perturbations[:, 2] *= 0.05  # Stronger compression along z
    
    # Add perturbations to positions
    positions += perturbations
    
    print(f"✓ Generated SHA-seeded cosmic web with {n_particles} particles")
    print(f"   Method: SHA-256 cryptographic entropy for even distribution")
    print(f"   Position range: [{positions.min():.2f}, {positions.max():.2f}]")
    print(f"   Velocity RMS: {torch.sqrt(torch.mean(velocities**2)):.6f}")
    print(f"   Distribution: Evenly spread with anisotropic quantum fluctuation seeds")
    
    return positions, velocities

def create_cosmic_web_debug_visualization(positions, step, current_bin, fractal_dim, entropy, similarity):
    """Create debug visualization optimized for cosmic web structure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    pos_np = positions.cpu().numpy()
    
    # XY projection - main cosmic web view
    ax1.scatter(pos_np[:, 0], pos_np[:, 1], alpha=0.3, s=0.5, c='blue')
    ax1.set_title(f'Cosmic Web XY - Step {step}')
    ax1.set_xlabel('X (cosmic web scale)')
    ax1.set_ylabel('Y (cosmic web scale)')
    ax1.grid(True, alpha=0.3)
    
    # XZ projection - side view of web
    ax2.scatter(pos_np[:, 0], pos_np[:, 2], alpha=0.3, s=0.5, c='red')
    ax2.set_title(f'Cosmic Web XZ - Bin {current_bin+1}')
    ax2.set_xlabel('X (cosmic web scale)')
    ax2.set_ylabel('Z (cosmic web scale)')
    ax2.grid(True, alpha=0.3)
    
    # Density histogram - should show filamentary structure
    ax3.hist2d(pos_np[:, 0], pos_np[:, 1], bins=50, alpha=0.8, cmap='viridis')
    ax3.set_title(f'Cosmic Web Density')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # Metrics
    ax4.text(0.1, 0.8, f'Step: {step}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Bin: {current_bin+1}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Fractal Dim: {fractal_dim:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Entropy: {entropy:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f'Similarity: {similarity:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.2, 'COSMIC WEB SCALE', fontsize=16, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.1, 'Dark Matter Distribution', fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    filename = f'debug_plots_cosmic_web/cosmic_web_step_{step:05d}_bin{current_bin+1}_sim{similarity:.3f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Cosmic web visualization saved: {filename}")

def main():
    """Main cosmic web simulation"""
    print("🌌 Dark Matter Cosmic Web Simulation")
    print("Large-scale structure formation at cosmic web scales")
    print("="*80)
    print("=== COSMIC WEB DARK MATTER SIMULATION ===")
    print("Approach: Weak gravity + high viscosity for filamentary structure")
    print("Scale: COSMIC WEB (not galaxy formation)")
    print("Target: Dark matter distribution in large-scale structure")
    print(f"🌌 Cosmic Web Physics: Weak clustering for filamentary structure")
    print(f"Redshift bins: {n_redshift_bins}, Steps per bin: {steps_per_bin}")
    print(f"Total evolution: {total_steps} steps")
    
    # Fetch temporal gradient data (same as galaxy sim)
    fetcher = AstroDataFetcher(device=device)
    print("\nFetching cosmic web reference data...")
    
    try:
        position_bins, metadata = fetcher.fetch_temporal_gradient_data(
            total_limit=15000, z_bins=n_redshift_bins  # Same reference data
        )
        print(f"✓ Loaded {len(position_bins)} redshift bins")
        for i, bin_meta in enumerate(metadata['bin_metadata']):
            print(f"  Bin {i+1}: z={bin_meta['redshift_range'][0]:.2f}-{bin_meta['redshift_range'][1]:.2f}, "
                  f"age≈{bin_meta.get('age_gyr', 0):.1f} Gyr, {bin_meta['count']} galaxies")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Initialize cosmic web simulation
    print(f"\n🌌 Cosmic Web Structure Simulation:")
    print(f"   Physics: Low density + weak gravity → filamentary cosmic web")
    print(f"   Approach: Intersecting planes → natural cosmic web evolution")
    print(f"   Target: Large-scale dark matter distribution")
    
    positions, velocities = generate_cosmic_web_initial_conditions(n_particles, 
                                                                 spatial_bounds=spatial_bounds,
                                                                 device=device)
    
    print(f"✓ Initialized {n_particles} particles for cosmic web simulation")
    print(f"SEC Parameters: rho_thresh={sec_params.rho_thresh:.4f}, "
          f"clustering={sec_params.clustering_strength:.4f}, "
          f"branching={sec_params.branching_bias:.4f}")
    print(f"Cosmic Web Parameters: α_base={alpha_base:.6e}, β_base={beta_base:.6e}")
    print(f"🌌 Cosmic Web Scale: Ωₘ={omega_m}, Ωₗ={omega_lambda}, H(z) evolution")
    print(f"Integration: SEC + weak gravity + cosmological expansion + cosmic web tidal forces")
    print(f"Evolution: Intersecting planes → natural cosmic web filament formation")
    
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
    snapshot_interval = 200
    
    print(f"\nStarting cosmic web evolution...")
    print(f"Current target: Bin 1/{n_redshift_bins} (youngest cosmic web structures)")
    
    while step < total_steps:
        # Check if we should progress to next evolutionary stage
        if steps_in_current_bin >= steps_per_bin and current_bin < len(position_bins) - 1:
            current_bin += 1
            steps_in_current_bin = 0
            print(f"\n🌌 Progressing to Bin {current_bin + 1}/{n_redshift_bins} "
                  f"(more evolved cosmic web)")
            
            # Update target for similarity comparison
            current_target = position_bins[current_bin]
        else:
            current_target = position_bins[current_bin]
        
        # Apply SEC forces with oscillatory hum modulation (cosmic web scale)
        forces = apply_sec_forces_cuda(positions, velocities, sec_params, n_particles, time_step=step)
        
        # Add oscillatory hum for cosmic web (slower, weaker)
        if OSCILLATORY_HUM_ENABLED:
            # Cosmic web oscillations (slower than galaxy scale)
            sec_field_avg = torch.mean(torch.norm(forces, dim=1))
            dynamic_freq = HUM_BASE_FREQ * (1.0 + HUM_SEC_COUPLING * sec_field_avg.item())
            hum_modulation = 1.0 + HUM_AMPLITUDE * torch.cos(dynamic_freq * step)
            forces *= hum_modulation.unsqueeze(1)
        
        # 🌌 COSMIC WEB COSMOLOGICAL PHYSICS
        # Get current cosmological redshift and calculate scale factor
        current_redshift_range = metadata['bin_metadata'][current_bin]['redshift_range']
        current_z = (current_redshift_range[0] + current_redshift_range[1]) / 2.0  # Midpoint redshift
        
        # Calculate cosmological parameters at current redshift
        a = scale_factor(current_z)  # Scale factor a(z) = 1/(1+z)
        H_z = hubble_parameter(current_z)  # H(z) with matter and dark energy evolution
        H_z_sim = H_z * (1e-3 / 3.156e16) * 3.156e16  # Convert to simulation units
        
        # Apply proper cosmological expansion: v = H(z) * r
        center_of_mass = torch.mean(positions, dim=0)
        displacement_from_center = positions - center_of_mass
        hubble_velocity = H_z_sim * displacement_from_center
        
        # Scale expansion by scale factor evolution (proper co-moving physics)
        scale_evolution_factor = a  # Current scale factor relative to z=0
        hubble_velocity *= scale_evolution_factor
        
        # Apply cosmologically-correct Hubble expansion
        velocities += hubble_velocity * dt
        
        # 🌌 COSMIC WEB TIDAL FORCES - Stronger for large-scale structure
        tidal_forces = apply_cosmic_web_tidal_forces(positions, velocities, current_z, dt, device)
        velocities += tidal_forces * dt
        
        # Update physics with SEC forces (weak for cosmic web)
        velocities += forces * dt
        velocities *= damping
        
        # Add small amount of noise to prevent stagnation
        velocities += torch.randn_like(velocities) * noise_strength
        
        positions += velocities * dt
        
        # Boundary conditions (periodic-like, larger scale)
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
                
                # Progress update with cosmological information
                elapsed = time.time() - start_time
                bin_progress = steps_in_current_bin / steps_per_bin * 100
                
                # Get current cosmological parameters for display
                current_redshift_range = metadata['bin_metadata'][current_bin]['redshift_range']
                current_z = (current_redshift_range[0] + current_redshift_range[1]) / 2.0
                a = scale_factor(current_z)
                H_z = hubble_parameter(current_z)
                
                # Calculate cosmic web tidal strength for display
                tidal_strength = 5e-7 * (1.0 + current_z)**0.5 * 0.5
                
                print(f"Step {step:4d} | Bin {current_bin+1}/{n_redshift_bins} ({bin_progress:5.1f}%) | "
                      f"z={current_z:.2f} a={a:.3f} H={H_z:.3e} T={tidal_strength:.2e} | "
                      f"Fractal_D={fractal_dim:.3f} | Entropy={spatial_entropy:.3f} | "
                      f"Similarity={overall_similarity:.3f} | Time={elapsed:.1f}s")
                
                # Create debug visualization every 400 steps
                if step % 400 == 0:
                    create_cosmic_web_debug_visualization(positions, step, current_bin, 
                                                        fractal_dim.item(), spatial_entropy.item(), 
                                                        overall_similarity.item())
        
        step += 1
        steps_in_current_bin += 1
    
    print(f"\n🌌 Cosmic Web Simulation Complete!")
    print(f"Final similarity: {similarities[-1]:.3f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Cleanup
    torch.cuda.empty_cache()
    print("GPU memory cleaned up")

if __name__ == "__main__":
    main()
