#!/usr/bin/env python3
"""
🌌 SIMPLIFIED COSMIC WEB DARK MATTER SIMULATION
Large-scale structure formation: Dark matter distribution at cosmic web scales

Key differences from galaxy simulation:
- MUCH WEAKER gravitational strength (cosmic web scale)
- HIGHER VISCOSITY to match larger scale physics
- SHA ENTROPY SEEDING for even distribution
- FOCUS ON FILAMENTS not dense nodes

Simplified version without SEC dependencies for testing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import hashlib
import os

# Ensure debug plots directory exists
os.makedirs('debug_plots_cosmic_web', exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 🌌 COSMIC WEB SCALE PHYSICS PARAMETERS
n_particles = 25000          # Representing cosmic web nodes
steps = 2000                 # Simulation steps
spatial_bounds = 50.0        # LARGER spatial scale for cosmic web

# 🔑 COSMIC WEB PHYSICS - The key changes for large-scale structure
G = 1e-5  # MUCH WEAKER gravitational constant (cosmic web scale)
dt = 0.01
damping = 0.995  # Slightly more damping for cosmic web
noise_strength = 0.0001

# Cosmological parameters
omega_m = 0.3            # Matter density parameter
omega_lambda = 0.7       # Dark energy density parameter  
H0 = 67.4               # Hubble constant (km/s/Mpc)
H0_sim = H0 * 1.022e-6  # Hubble constant in simulation units (1/Gyr)

def scale_factor(redshift):
    """Compute scale factor a(z) = 1/(1+z)"""
    return 1.0 / (1.0 + redshift)

def hubble_parameter(redshift):
    """Compute H(z) = H0 * sqrt(Ωm*(1+z)³ + ΩΛ) in simulation units"""
    z_plus_1 = 1.0 + redshift
    if isinstance(redshift, torch.Tensor):
        return H0_sim * torch.sqrt(omega_m * z_plus_1**3 + omega_lambda)
    else:
        return H0_sim * (omega_m * z_plus_1**3 + omega_lambda)**0.5

def generate_sha_entropy_seed(hash_input, shape):
    """
    Generate deterministic pseudo-random field using SHA-256 entropy seeding.
    Creates evenly distributed initial conditions based on cryptographic hash.
    """
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape)

def generate_cosmic_web_initial_conditions(n_particles, spatial_bounds=50.0, device='cuda'):
    """
    Generate initial conditions using SHA entropy seeding for even distribution.
    """
    print(f"🌌 Generating SHA-seeded cosmic web initial conditions...")
    print(f"   Method: Cryptographic entropy for even distribution")
    
    # Use SHA-256 entropy seeding for evenly distributed cosmic structure
    hash_positions = "CIMM:cosmic_web:positions:dark_matter"
    hash_velocities = "CIMM:cosmic_web:velocities:initial"
    
    # Generate positions using SHA entropy (evenly distributed)
    pos_entropy = generate_sha_entropy_seed(hash_positions, (n_particles, 3))
    positions = torch.tensor(pos_entropy, device=device, dtype=torch.float32)
    positions = (positions - 0.5) * 2.0 * spatial_bounds  # Scale to [-bounds, +bounds]
    
    # Generate velocities using separate SHA entropy
    vel_entropy = generate_sha_entropy_seed(hash_velocities, (n_particles, 3))
    velocities = torch.tensor(vel_entropy, device=device, dtype=torch.float32)
    velocities = (velocities - 0.5) * 0.002  # Very small initial velocities
    
    print(f"✓ Generated SHA-seeded cosmic web with {n_particles} particles")
    print(f"   Position range: [{positions.min():.2f}, {positions.max():.2f}]")
    print(f"   Velocity RMS: {torch.sqrt(torch.mean(velocities**2)):.6f}")
    
    return positions, velocities

def compute_cosmic_web_tidal_forces(positions, redshift, device):
    """
    Compute tidal forces optimized for cosmic web formation.
    Anisotropic forces to promote filamentary structure.
    """
    # Tidal strength scales with cosmic evolution
    tidal_strength = 5e-7 * (1.0 + redshift)**0.5
    
    # Create anisotropic tidal field for cosmic web structure
    tidal_tensor = torch.zeros((3, 3), device=device)
    tidal_tensor[0, 0] = tidal_strength * 3.0   # Stretching along x
    tidal_tensor[1, 1] = -tidal_strength * 1.5  # Compression along y
    tidal_tensor[2, 2] = -tidal_strength * 1.5  # Compression along z
    
    # Apply tidal acceleration: a_tidal = T · r
    tidal_forces = torch.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            tidal_forces[:, i] += tidal_tensor[i, j] * positions[:, j]
    
    return tidal_forces * 0.5  # Strong external cosmic web influence

def compute_simple_gravity_forces(positions, device):
    """
    Compute simplified gravitational forces for cosmic web scale.
    Much weaker than galaxy-scale gravity.
    """
    n_particles = positions.shape[0]
    forces = torch.zeros_like(positions)
    
    # Simplified gravity calculation (not optimized, but works for testing)
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_dist = torch.norm(r_vec) + 1e-6  # Softening
                
                # Weak gravitational force for cosmic web
                f_magnitude = G / (r_dist**2)
                f_vec = f_magnitude * r_vec / r_dist
                
                # Add viscosity damping for cosmic web scale
                viscosity_factor = torch.exp(-0.01 * r_dist)
                forces[i] += f_vec * viscosity_factor
    
    return forces

def compute_fractal_dimension_simple(positions):
    """Simple fractal dimension estimate"""
    # Box counting method simplified
    n_particles = positions.shape[0]
    if n_particles < 10:
        return 1.5
    
    # Use correlation dimension approximation
    distances = torch.cdist(positions, positions)
    distances = distances[distances > 0]  # Remove self-distances
    
    if len(distances) == 0:
        return 1.5
    
    median_dist = torch.median(distances)
    count_close = torch.sum(distances < median_dist * 0.5).float()
    
    if count_close > 0:
        # Rough approximation of fractal dimension
        fractal_dim = torch.log(count_close) / torch.log(2.0)
        return torch.clamp(fractal_dim, 1.0, 3.0).item()
    else:
        return 1.5

def compute_entropy_simple(positions):
    """Simple spatial entropy estimate"""
    # Discretize space and compute entropy
    bins = 20
    hist, _ = torch.histogramdd(positions, bins=bins, 
                               range=[(-spatial_bounds, spatial_bounds)]*3)
    hist = hist.flatten()
    hist = hist[hist > 0]  # Remove empty bins
    
    if len(hist) == 0:
        return 0.0
    
    # Normalize to probabilities
    probs = hist.float() / torch.sum(hist)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    return entropy.item()

def create_cosmic_web_visualization(positions, step, fractal_dim, entropy, current_z):
    """Create debug visualization for cosmic web structure"""
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
    ax2.set_title(f'Cosmic Web XZ - z={current_z:.2f}')
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
    ax4.text(0.1, 0.7, f'Redshift: {current_z:.2f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Fractal Dim: {fractal_dim:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Entropy: {entropy:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, 'COSMIC WEB SCALE', fontsize=16, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.2, 'SHA Entropy Seeded', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.1, 'Dark Matter Distribution', fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    filename = f'debug_plots_cosmic_web/cosmic_web_step_{step:05d}_z{current_z:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Cosmic web visualization saved: {filename}")

def main():
    """Main cosmic web simulation"""
    print("🌌 Simplified Dark Matter Cosmic Web Simulation")
    print("="*60)
    print("=== COSMIC WEB DARK MATTER SIMULATION ===")
    print("SHA entropy seeding + weak gravity + tidal forces")
    print("Scale: COSMIC WEB (not galaxy formation)")
    print(f"Particles: {n_particles}, Steps: {steps}")
    
    # Initialize cosmic web simulation with SHA entropy seeding
    print(f"\n🌌 Cosmic Web Structure Simulation:")
    print(f"   Physics: SHA seeding + weak gravity → filamentary cosmic web")
    print(f"   Target: Large-scale dark matter distribution")
    
    positions, velocities = generate_cosmic_web_initial_conditions(n_particles, 
                                                                 spatial_bounds=spatial_bounds,
                                                                 device=device)
    
    print(f"✓ Initialized {n_particles} particles for cosmic web simulation")
    print(f"Cosmic Web Parameters: G={G:.2e}, damping={damping}")
    print(f"🌌 Cosmic Web Scale: Ωₘ={omega_m}, Ωₗ={omega_lambda}")
    print(f"Integration: Weak gravity + cosmological expansion + cosmic web tidal forces")
    
    # Simulation tracking
    start_time = time.time()
    snapshot_interval = 200
    
    print(f"\nStarting cosmic web evolution...")
    
    for step in range(steps):
        # Current cosmological time (simplified - assume z decreases linearly)
        current_z = 1.0 * (1.0 - step / steps)  # z from 1.0 to 0.0
        
        # Calculate cosmological parameters at current redshift
        a = scale_factor(current_z)
        H_z = hubble_parameter(current_z)
        
        # Apply cosmological expansion: v = H(z) * r
        center_of_mass = torch.mean(positions, dim=0)
        displacement_from_center = positions - center_of_mass
        hubble_velocity = H_z * displacement_from_center
        
        # Scale expansion by scale factor evolution
        scale_evolution_factor = a
        hubble_velocity *= scale_evolution_factor
        
        # Apply cosmological expansion
        velocities += hubble_velocity * dt
        
        # Apply cosmic web tidal forces
        tidal_forces = compute_cosmic_web_tidal_forces(positions, current_z, device)
        velocities += tidal_forces * dt
        
        # Apply weak gravitational forces (every 5 steps to save computation)
        if step % 5 == 0:
            gravity_forces = compute_simple_gravity_forces(positions, device)
            velocities += gravity_forces * dt * 5  # Scale up for missed steps
        
        # Apply damping and noise
        velocities *= damping
        velocities += torch.randn_like(velocities) * noise_strength
        
        # Update positions
        positions += velocities * dt
        
        # Boundary conditions
        positions = torch.clamp(positions, -spatial_bounds, spatial_bounds)
        
        # Progress and visualization
        if step % snapshot_interval == 0:
            fractal_dim = compute_fractal_dimension_simple(positions)
            entropy = compute_entropy_simple(positions)
            elapsed = time.time() - start_time
            
            print(f"Step {step:4d} | z={current_z:.2f} a={a:.3f} H={H_z:.3e} | "
                  f"Fractal_D={fractal_dim:.3f} | Entropy={entropy:.3f} | Time={elapsed:.1f}s")
            
            if step % 400 == 0:
                create_cosmic_web_visualization(positions, step, fractal_dim, entropy, current_z)
    
    print(f"\n🌌 Cosmic Web Simulation Complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU memory cleaned up")

if __name__ == "__main__":
    main()
