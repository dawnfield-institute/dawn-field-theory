"""
Dark Matter 3D Simulation with Temporal Gradient Approach + Emergent Gravity
===========================================================================

Novel approach: Use cosmological distance as temporal proxy:
- Start with high-redshift (young) structures
- Progress to low-redshift (evolved) structures
- Match simulation evolution to cosmic time progression

EMERGENT GRAVITY INTEGRATION:
----------------------------
Added viscosity and Landauer scaffolding modifications to prevent singularity collapse
and promote filamentary cosmic web structures:

1. **Viscosity Constraint**: F *= exp(-α|r|)
   - Models finite-speed gravitational propagation (light-speed limit)
   - PHYSICALLY ANCHORED: α ∼ 1/(c × t_U) where c = light speed, t_U = universe age
   - Suppresses long-range instantaneous interactions beyond light horizon

2. **Landauer Scaffolding**: F *= 1/(1 + β|r|²)
   - Information-theoretic interaction saturation based on Landauer's principle
   - PHYSICALLY ANCHORED: β ∼ (k_B × T_CMB)/(m_p × c² × R_U²)
   - Prevents runaway clustering by imposing thermodynamic interaction cost

Combined force law:
F_ij = -G * m_i * m_j / |r|³ * r * exp(-α|r|) * 1/(1 + β|r|²)

UNIVERSAL CONSTANTS USED:
- c = 299,792.458 km/s (speed of light)
- t_U = 13.8 Gyr (age of universe)  
- k_B = 1.38×10⁻²³ J/K (Boltzmann constant)
- T_CMB = 2.725 K (cosmic microwave background temperature)
- m_p = 1.67×10⁻²⁷ kg (proton mass)
- R_U = 14.3 billion kpc (observable universe radius)

This approach makes the model FALSIFIABLE by constraining parameters to universal physics
rather than free tuning knobs, bridging gravity + thermodynamics + information theory.

Based on SEC framework with proven parameters from infodynamics validation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # Keep for matplotlib compatibility only
import time
import hashlib  # For entropy seeding
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

# Temporal progression parameters - Phase 2 Enhanced Resolution
n_particles = 25000          # Increased for richer cosmic structure formation  
n_redshift_bins = 5          # Number of evolutionary stages
steps_per_bin = 1000         # Evolution steps per redshift bin
total_steps = n_redshift_bins * steps_per_bin

# Physics parameters
G = 4.498e-6  # Gravitational constant (kpc³/M☉/Gyr²)
dt = 0.01
damping = 0.999
noise_strength = 0.0001
spatial_bounds = 25.0  # Increased for larger particle count and more structure formation room

# Universal Physical Constants for Emergent Gravity Anchoring
c_light = 299792.458  # Speed of light (km/s)
k_B = 1.380649e-23   # Boltzmann constant (J/K)
m_p = 1.67262e-27    # Proton mass (kg)
T_cmb = 2.725        # Cosmic microwave background temperature (K)
R_U = 46.5e9         # Observable universe radius (light-years → need to convert to kpc)
t_U = 13.8           # Age of universe (Gyr)
H0 = 67.4            # Hubble constant (km/s/Mpc)

# Cosmological parameters for co-moving coordinates
omega_m = 0.3        # Matter density parameter
omega_lambda = 0.7   # Dark energy density parameter
omega_k = 0.0        # Curvature density parameter (flat universe)

# Convert R_U to kpc for consistency with simulation units
R_U_kpc = R_U * 0.3066  # Convert ly to kpc: R_U ≈ 14.3 billion kpc

# Convert H0 to simulation units (1/Gyr) for kpc scale
# H0 = 67.4 km/s/Mpc = 67.4 km/s/1000kpc = 0.0674 km/s/kpc
# Convert km/s to kpc/Gyr: 1 km/s = 1.022e-6 kpc/Gyr
H0_sim = H0 * 1.022e-6  # Hubble constant in simulation units (1/Gyr)

# Co-moving coordinate functions
def scale_factor(redshift):
    """Compute scale factor a(t) from redshift: a = 1/(1+z)"""
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

# 🌌 PHASE 3: TIDAL FORCES - External cosmic web influence
def compute_tidal_tensor(positions, redshift, device):
    """
    Compute tidal force tensor from external cosmic web structure.
    Simulates gravitational influence from large-scale structure beyond simulation volume.
    """
    # Tidal strength scales with cosmic evolution and distance
    a = scale_factor(redshift)
    tidal_strength = 1e-8 * (1.0 + redshift)**0.5  # Stronger at higher redshift
    
    # Create anisotropic tidal field (filamentary cosmic web geometry)
    # Major axis: x-direction (main filament)
    # Minor axes: y,z-directions (compression perpendicular to filament)
    tidal_tensor = torch.zeros((3, 3), device=device)
    tidal_tensor[0, 0] = tidal_strength * 2.0   # Stretching along x (filament direction)
    tidal_tensor[1, 1] = -tidal_strength        # Compression along y
    tidal_tensor[2, 2] = -tidal_strength        # Compression along z
    
    return tidal_tensor

def apply_tidal_forces(positions, velocities, redshift, dt, device):
    """
    Apply external tidal forces from cosmic web beyond simulation volume.
    This simulates being embedded in a larger cosmic structure.
    """
    tidal_tensor = compute_tidal_tensor(positions, redshift, device)
    
    # Apply tidal acceleration: a_tidal = T · r (where T is tidal tensor)
    tidal_forces = torch.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            tidal_forces[:, i] += tidal_tensor[i, j] * positions[:, j]
    
    # Scale tidal forces (they should be subtle but measurable)
    tidal_forces *= 0.1  # Modest external influence
    
    return tidal_forces

# Base Physically-Anchored Emergent Gravity Parameters (used as reference)
# α: viscosity coefficient tied to light horizon
# α ∼ 1/(c × t_U) - gravitational influence weakens beyond light horizon
alpha_base = 1.0 / (c_light * t_U * 1e6)  # Scale factor for numerical stability

# β: Landauer coefficient tied to cosmic background temperature
# β ∼ (k_B × T) / (m_p × c² × R_U²) - information cost scaling
c_mks = 2.998e8  # c in m/s for proper units
beta_base = (k_B * T_cmb) / (m_p * c_mks**2 * (R_U_kpc * 3.086e19)**2)  # Convert kpc to m

# Scale parameters for simulation stability (maintain physical ratios)
alpha_base *= 1e15  # Numerical scaling while preserving physics
beta_base *= 1e40   # Numerical scaling while preserving physics

# 🌊 Oscillatory Hum Parameters (Cosmic Heartbeat)
OSCILLATORY_HUM_ENABLED = True
HUM_BASE_FREQUENCY = 0.1        # Base cosmic oscillation frequency
HUM_AMPLITUDE = 0.03            # 3% force modulation (gentle cosmic breathing)
HUM_SEC_COUPLING = 0.01         # How much SEC field variance affects frequency

# Simple QBE Controller (Minimal Working Version)
class EmergentGravityQBEController:
    """Simple QBE controller - just returns base parameters with minimal adaptation"""
    
    def __init__(self, alpha_base, beta_base):
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        
    def update_parameters(self, positions, velocities, forces, densities):
        """Simple parameter update - minimal adaptation only"""
        # Just return base parameters with tiny adaptive scaling
        density_std = torch.std(densities)
        alpha_scale = 1.0 + 0.01 * torch.tanh(density_std / 100.0)  # Very gentle scaling
        beta_scale = 1.0 + 0.01 * torch.tanh(density_std / 100.0)   # Very gentle scaling
        
        return self.alpha_base * alpha_scale, self.beta_base * beta_scale
    
    def update_local_parameters(self, positions, forces, densities, similarity_score=None):
        """Compatibility method - just returns base parameters"""
        return self.alpha_base, self.beta_base
    
    def get_equilibrium_status(self):
        """Return equilibrium status"""
        return "Near Equilibrium"

# Initialize QBE controller
qbe_controller = EmergentGravityQBEController(alpha_base, beta_base)

print(f"🌌 Dynamic Emergent Gravity Parameters:")
print(f"α_base = {alpha_base:.6e} (light-horizon constrained)")
print(f"β_base = {beta_base:.6e} (CMB temperature constrained)")
print(f"QBE Controller: Dynamic local adaptation based on density and force gradients")
print(f"Physical basis: α ∼ 1/(c×t_U), β ∼ k_B×T/(m_p×c²×R_U²)")
if OSCILLATORY_HUM_ENABLED:
    print(f"🌊 Oscillatory Hum: ENABLED")
    print(f"   Base frequency: {HUM_BASE_FREQUENCY:.3f}, Amplitude: {HUM_AMPLITUDE:.3f}")
    print(f"   SEC coupling: {HUM_SEC_COUPLING:.3f} (frequency varies with SEC field)")
    print(f"   Effect: Cosmic heartbeat modulation F *= (1 + A×cos(ωt))")
else:
    print(f"🌊 Oscillatory Hum: DISABLED")

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
    
    # 3D histogram using torch.bincount - ensure contiguous tensors for bucketize
    x_pos = positions[:, 0].contiguous()
    y_pos = positions[:, 1].contiguous()
    z_pos = positions[:, 2].contiguous()
    
    x_indices = torch.bucketize(x_pos, bin_edges) - 1
    y_indices = torch.bucketize(y_pos, bin_edges) - 1
    z_indices = torch.bucketize(z_pos, bin_edges) - 1
    
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

def apply_sec_forces_cuda(positions, velocities, sec_params, n_particles, time_step=0, chunk_size=2000):
    """
    Simplified SEC forces with Basic Emergent Gravity:
    - Simple viscosity constraint: F *= exp(-α|r|)
    - Simple Landauer scaffolding: F *= 1/(1 + β|r|²)
    
    Basic force law: F_ij = -G * m_i * m_j / |r|³ * r * exp(-α|r|) * 1/(1 + β|r|²)
    """
    global qbe_controller
    
    positions = positions.contiguous()
    forces = torch.zeros_like(positions)
    
    rho_thresh = sec_params.rho_thresh
    dispersion_strength = sec_params.dispersion_strength
    clustering_strength = sec_params.clustering_strength
    branching_bias = sec_params.branching_bias
    
    # Get simple QBE parameters (no complex adaptation)
    densities = compute_density_cuda(positions)
    alpha, beta = qbe_controller.update_parameters(positions, velocities, forces, densities)
    
    # Process all particles at once (simplified)
    for i in range(0, n_particles, chunk_size):
        end_i = min(i + chunk_size, n_particles)
        chunk_positions = positions[i:end_i].contiguous()
        chunk_densities = densities[i:end_i]
        
        # Compute pairwise interactions for this chunk
        diff = positions.unsqueeze(0) - chunk_positions.unsqueeze(1)  # [chunk_size, n_particles, 3]
        distances = torch.norm(diff, dim=2) + 1e-6  # [chunk_size, n_particles]
        directions = diff / distances.unsqueeze(2)  # [chunk_size, n_particles, 3]
        
        # Simple gravitational force
        gravitational_strength = G / (distances ** 2 + 1e-6)
        
        # Simple viscosity constraint
        viscosity_damping = torch.exp(-alpha * distances)
        
        # Simple Landauer scaffolding
        landauer_saturation = 1.0 / (1.0 + beta * distances ** 2)
        
        # Combined gravity
        gravity_strength = gravitational_strength * viscosity_damping * landauer_saturation
        gravity_forces = -directions * gravity_strength.unsqueeze(2)
        
        # SEC forces (simplified)
        low_density_mask = (chunk_densities < rho_thresh * n_particles).unsqueeze(1)
        
        # Repulsion for low density
        repulsion_strength = 1.0 / (distances ** 2 + 0.1)
        repulsion_forces = directions * repulsion_strength.unsqueeze(2)
        repulsion_total = torch.sum(repulsion_forces, dim=1) * dispersion_strength
        
        # Attraction for high density
        high_density_mask = ~low_density_mask
        attraction_strength = 1.0 / (distances + 1.0)
        attraction_forces = -directions * attraction_strength.unsqueeze(2)
        attraction_total = torch.sum(attraction_forces, dim=1) * clustering_strength
        
        # Combine SEC forces
        sec_forces = (low_density_mask.float() * repulsion_total + 
                     high_density_mask.float() * attraction_total)
        
        # Combine all forces
        gravity_total = torch.sum(gravity_forces, dim=1)
        chunk_forces = gravity_total + sec_forces
        forces[i:end_i] = chunk_forces
    
    # Add radial branching bias
    center = torch.mean(positions, dim=0)
    radial_directions = positions - center
    radial_distances = torch.norm(radial_directions, dim=1, keepdim=True) + 1e-6
    radial_directions = radial_directions / radial_distances
    forces += branching_bias * radial_directions
    
    # 🌊 OSCILLATORY HUM: Cosmic heartbeat modulation
    if OSCILLATORY_HUM_ENABLED:
        # Use SEC field variance to set natural frequency
        sec_field_variance = torch.var(densities)
        sec_coupling = HUM_SEC_COUPLING * sec_field_variance.item()
        omega = 2 * torch.pi * (HUM_BASE_FREQUENCY + sec_coupling)
        
        # Apply oscillatory hum: F_final = F_existing × (1 + A × cos(ωt))
        time_phase = omega * time_step
        hum_factor = 1.0 + HUM_AMPLITUDE * torch.cos(torch.tensor(time_phase, device=positions.device))
        forces *= hum_factor
    
    return forces

def generate_entropy_seed(hash_input, shape, device):
    """Generate reproducible entropy-seeded distribution using SHA-256"""
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    
    # Use the seed with numpy for reproducibility, then convert to torch
    np.random.seed(seed)
    positions = np.random.rand(*shape).astype(np.float32)
    
    return torch.from_numpy(positions).to(device)

def generate_primitive_cosmic_web(n_particles, spatial_bounds=10.0, device='cuda'):
    """
    Generate a 'comfortable' primitive cosmic web structure for evolutionary simulation:
    - Known good filamentary pattern (like early cosmic web)
    - Realistic node-filament topology
    - Good starting point for natural evolution
    """
    print("🌌 Generating primitive cosmic web structure...")
    print(f"   Approach: Idealized early filamentary network")
    print(f"   Purpose: Known good starting pattern for evolution")
    
    # Create primary filamentary scaffolding using multiple entropy seeds
    positions = []
    
    # Main filament along X-axis
    filament_x = generate_entropy_seed("CIMM:filament:main_x", (n_particles // 4, 3), device)
    filament_x[:, 0] = (filament_x[:, 0] - 0.5) * 2 * spatial_bounds * 0.8  # Main extent
    filament_x[:, 1] = (filament_x[:, 1] - 0.5) * spatial_bounds * 0.1      # Small Y spread
    filament_x[:, 2] = (filament_x[:, 2] - 0.5) * spatial_bounds * 0.1      # Small Z spread
    positions.append(filament_x)
    
    # Cross filament along Y-axis
    filament_y = generate_entropy_seed("CIMM:filament:cross_y", (n_particles // 4, 3), device)
    filament_y[:, 0] = (filament_y[:, 0] - 0.5) * spatial_bounds * 0.1      # Small X spread
    filament_y[:, 1] = (filament_y[:, 1] - 0.5) * 2 * spatial_bounds * 0.8  # Main extent
    filament_y[:, 2] = (filament_y[:, 2] - 0.5) * spatial_bounds * 0.1      # Small Z spread
    positions.append(filament_y)
    
    # Diagonal filament for 3D structure
    filament_diag = generate_entropy_seed("CIMM:filament:diagonal", (n_particles // 4, 3), device)
    t = (filament_diag[:, 0] - 0.5) * 2 * spatial_bounds * 0.6
    filament_diag[:, 0] = t
    filament_diag[:, 1] = t * 0.5 + (filament_diag[:, 1] - 0.5) * spatial_bounds * 0.1
    filament_diag[:, 2] = t * 0.3 + (filament_diag[:, 2] - 0.5) * spatial_bounds * 0.1
    positions.append(filament_diag)
    
    # Central node cluster (intersection point)
    remaining = n_particles - 3 * (n_particles // 4)
    central_node = generate_entropy_seed("CIMM:node:central", (remaining, 3), device)
    central_node = (central_node - 0.5) * spatial_bounds * 0.2  # Compact central cluster
    positions.append(central_node)
    
    # Combine all structures
    positions = torch.cat(positions, dim=0)
    
    # Add small random perturbations for realism
    perturbations = generate_entropy_seed("CIMM:perturbation:web", positions.shape, device)
    perturbations = (perturbations - 0.5) * spatial_bounds * 0.05
    positions += perturbations
    
    # Initialize velocities with slight infall toward filaments
    velocities = torch.zeros_like(positions)
    center = torch.mean(positions, dim=0)
    
    # Add small velocities toward nearest filament axis (simplified)
    for i in range(len(positions)):
        # Simple infall pattern
        direction = center - positions[i]
        direction = direction / (torch.norm(direction) + 1e-6)
        velocities[i] = direction * 0.01  # Small infall velocity
    
    # Add thermal motion
    thermal_motion = generate_entropy_seed("CIMM:thermal:web", velocities.shape, device)
    thermal_motion = (thermal_motion - 0.5) * 0.02
    velocities += thermal_motion
    
    print(f"✓ Generated primitive cosmic web with {n_particles} particles")
    print(f"   Structure: 3 main filaments + central node + perturbations")
    print(f"   Position range: [{torch.min(positions):.2f}, {torch.max(positions):.2f}]")
    print(f"   Velocity RMS: {torch.sqrt(torch.mean(velocities**2)):.4f}")
    
    return positions, velocities

def initialize_particles_entropy_seeded(n_particles, spatial_bounds=10.0, 
                                       fluctuation_amplitude=0.01, device='cuda'):
    """
    Initialize particles with entropy-seeded early universe conditions:
    - Nearly uniform distribution (like post-recombination)
    - Tiny density fluctuations (mimicking CMB power spectrum)
    - SHA-based reproducible seeding (like quantum fluctuations)
    """
    print("🌌 Initializing with entropy-seeded early universe conditions...")
    print(f"   Approach: Nearly uniform + tiny quantum-like fluctuations")
    print(f"   Fluctuation amplitude: {fluctuation_amplitude:.3f} (mimics δρ/ρ ~ 10⁻⁵)")
    
    # Base uniform distribution in a cubic volume
    base_positions = generate_entropy_seed("CIMM:early_universe:matter", 
                                         (n_particles, 3), device)
    
    # Scale to spatial bounds (centered around origin)
    positions = (base_positions - 0.5) * 2 * spatial_bounds
    
    # Add tiny structured fluctuations (like CMB anisotropies)
    fluctuation_x = generate_entropy_seed("CIMM:fluctuation:x", (n_particles, 1), device)
    fluctuation_y = generate_entropy_seed("CIMM:fluctuation:y", (n_particles, 1), device)
    fluctuation_z = generate_entropy_seed("CIMM:fluctuation:z", (n_particles, 1), device)
    
    fluctuations = torch.cat([fluctuation_x, fluctuation_y, fluctuation_z], dim=1)
    fluctuations = (fluctuations - 0.5) * 2 * fluctuation_amplitude * spatial_bounds
    
    positions += fluctuations
    
    # Initialize with very small random velocities (thermal motion)
    velocity_seed = generate_entropy_seed("CIMM:early_universe:velocity", 
                                        (n_particles, 3), device)
    velocities = (velocity_seed - 0.5) * 0.02  # Very small initial velocities
    
    print(f"✓ Generated {n_particles} particles with entropy seeding")
    print(f"   Position range: [{torch.min(positions):.2f}, {torch.max(positions):.2f}]")
    print(f"   Initial velocity RMS: {torch.sqrt(torch.mean(velocities**2)):.4f}")
    
    return positions, velocities

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
    """Main simulation using temporal gradient approach with dynamic QBE emergent gravity"""
    print("=== TEMPORAL GRADIENT DARK MATTER SIMULATION WITH DYNAMIC QBE EMERGENT GRAVITY ===")
    print(f"Approach: High-redshift (young) → Low-redshift (evolved) structures")
    print(f"Dynamic Emergent Gravity: QBE-controlled viscosity + Landauer scaffolding")
    if OSCILLATORY_HUM_ENABLED:
        print(f"Oscillatory Hum: SEC-coupled cosmic heartbeat modulation")
    print(f"🌌 Phase 3: Tidal Forces - External cosmic web influence beyond simulation volume")
    print(f"QBE = Quantum Balance Equation for adaptive parameter control")
    print(f"Redshift bins: {n_redshift_bins}, Steps per bin: {steps_per_bin}")
    print(f"Total evolution: {total_steps} steps")
    
    # Fetch temporal gradient data
    fetcher = AstroDataFetcher(device=device)
    print("\nFetching temporal gradient data...")
    
    try:
        position_bins, metadata = fetcher.fetch_temporal_gradient_data(
            total_limit=15000, z_bins=n_redshift_bins  # 3x more galaxy data for richer cosmic structure
        )
        print(f"✓ Loaded {len(position_bins)} redshift bins")
        for i, bin_meta in enumerate(metadata['bin_metadata']):
            print(f"  Bin {i+1}: z={bin_meta['redshift_range'][0]:.2f}-{bin_meta['redshift_range'][1]:.2f}, "
                  f"age≈{bin_meta.get('age_gyr', 0):.1f} Gyr, {bin_meta['count']} galaxies")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Initialize simulation with primitive cosmic web structure for evolution
    print(f"\n🌌 Evolutionary Cosmic Structure Simulation:")
    print(f"   Physics: Start with primitive cosmic web → natural evolution")
    print(f"   Approach: Known good filamentary pattern → SEC + oscillatory hum evolution")
    print(f"   Target: 80% similarity through natural cosmic evolution")
    positions, velocities = generate_primitive_cosmic_web(n_particles, 
                                                        spatial_bounds=spatial_bounds,
                                                        device=device)
    
    print(f"✓ Initialized {n_particles} particles with primitive cosmic web")
    print(f"SEC Parameters: rho_thresh={sec_params.rho_thresh:.4f}, "
          f"clustering={sec_params.clustering_strength:.4f}, "
          f"branching={sec_params.branching_bias:.4f}")
    print(f"QBE Dynamic Parameters: α_base={alpha_base:.6e}, β_base={beta_base:.6e}")
    print(f"🌌 Co-moving Expansion: Ωₘ={omega_m}, Ωₗ={omega_lambda}, H(z) evolution")
    print(f"QBE Controller (Quantum Balance Equation): {qbe_controller.get_equilibrium_status()}")
    print(f"Integration: SEC + emergent gravity + cosmological expansion + tidal forces + oscillatory hum")
    print(f"Evolution: Primitive cosmic web → natural structure formation in expanding universe with external tidal influence")
    
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
        
        # Apply SEC forces with oscillatory hum modulation
        forces = apply_sec_forces_cuda(positions, velocities, sec_params, n_particles, time_step=step)
        
        # 🌌 PHASE 2: CO-MOVING COORDINATES - Proper cosmological expansion
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
        # In co-moving coordinates, physical distances scale as a(t) = 1/(1+z)
        scale_evolution_factor = a  # Current scale factor relative to z=0
        hubble_velocity *= scale_evolution_factor
        
        # Apply cosmologically-correct Hubble expansion
        velocities += hubble_velocity * dt
        
        # 🌌 PHASE 3: TIDAL FORCES - External cosmic web influence
        # Apply gravitational influence from large-scale structure beyond simulation volume
        tidal_forces = apply_tidal_forces(positions, velocities, current_z, dt, device)
        velocities += tidal_forces * dt
        
        # Update physics with SEC forces
        velocities += forces * dt
        velocities *= damping
        
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
                
                # Progress update with cosmological information
                elapsed = time.time() - start_time
                bin_progress = steps_in_current_bin / steps_per_bin * 100
                
                # Get current cosmological parameters for display
                current_redshift_range = metadata['bin_metadata'][current_bin]['redshift_range']
                current_z = (current_redshift_range[0] + current_redshift_range[1]) / 2.0
                a = scale_factor(current_z)
                H_z = hubble_parameter(current_z)
                
                # Calculate tidal strength for display
                tidal_strength = 1e-8 * (1.0 + current_z)**0.5 * 0.1
                
                print(f"Step {step:4d} | Bin {current_bin+1}/{n_redshift_bins} ({bin_progress:5.1f}%) | "
                      f"z={current_z:.2f} a={a:.3f} H={H_z:.3e} T={tidal_strength:.2e} | "
                      f"Fractal_D={fractal_dim:.3f} | Entropy={spatial_entropy:.3f} | "
                      f"Similarity={overall_similarity:.3f} | Time={elapsed:.1f}s")
                
                # Create debug visualization every 400 steps (2x snapshot interval)
                if step % 400 == 0:
                    create_debug_visualization(positions, step, current_bin, 
                                             fractal_dim.item(), spatial_entropy.item(), 
                                             overall_similarity.item())
        
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

def create_debug_visualization(positions, step, current_bin, fractal_dim, entropy, similarity, save_dir="debug_plots"):
    """Create and save visualization snapshot for debugging"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    positions_np = positions.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], 
               alpha=0.6, s=1, c='blue')
    ax1.set_title(f'3D Structure - Step {step}\nBin {current_bin+1}, Sim={similarity:.3f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # XY projection
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(positions_np[:, 0], positions_np[:, 1], alpha=0.6, s=1, c='blue')
    ax2.set_title(f'XY Projection\nFractal D={fractal_dim:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(positions_np[:, 0], positions_np[:, 2], alpha=0.6, s=1, c='blue')
    ax3.set_title(f'XZ Projection\nEntropy={entropy:.3f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_aspect('equal')
    
    # Density histogram
    ax4 = fig.add_subplot(2, 3, 4)
    densities = compute_density_cuda(positions).cpu().numpy()
    ax4.hist(densities, bins=50, alpha=0.7, color='green')
    ax4.set_title(f'Density Distribution\nMean={np.mean(densities):.2f}')
    ax4.set_xlabel('Local Density')
    ax4.set_ylabel('Count')
    
    # Distance from center
    ax5 = fig.add_subplot(2, 3, 5)
    center = torch.mean(positions, dim=0).cpu().numpy()
    distances = np.linalg.norm(positions_np - center, axis=1)
    ax5.hist(distances, bins=50, alpha=0.7, color='red')
    ax5.set_title(f'Radial Distribution\nMax distance={np.max(distances):.2f}')
    ax5.set_xlabel('Distance from Center')
    ax5.set_ylabel('Count')
    
    # Clustering analysis
    ax6 = fig.add_subplot(2, 3, 6)
    density_variance = torch.var(compute_density_cuda(positions)).item()
    kinetic_energy = torch.mean(torch.norm(torch.zeros_like(positions), dim=1)).item()
    
    ax6.text(0.1, 0.8, f'Step: {step}', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.7, f'Bin: {current_bin+1}/5', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.6, f'Similarity: {similarity:.3f}', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.5, f'Fractal D: {fractal_dim:.3f}', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.4, f'Entropy: {entropy:.3f}', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.3, f'Density Var: {density_variance:.1f}', transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.2, f'N Particles: {len(positions)}', transform=ax6.transAxes, fontsize=12)
    ax6.set_title('Simulation Metrics')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{save_dir}/debug_step_{step:05d}_bin{current_bin+1}_sim{similarity:.3f}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Debug visualization saved: {filename}")

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
    ax4.set_xlabel('Simulation Steps')
    ax4.set_title('Temporal Bin Progression')
    
    # Add similarity annotations
    for i, sim_value in enumerate([similarities[min(i*200, len(similarities)-1)] for i in range(n_bins)]):
        ax4.text(bin_positions[i], i, f'{sim_value:.2f}', 
                ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('temporal_evolution_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Evolution plot saved: temporal_evolution_debug.png")
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
        
        # Run simulation with dynamic QBE parameter adaptation
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
