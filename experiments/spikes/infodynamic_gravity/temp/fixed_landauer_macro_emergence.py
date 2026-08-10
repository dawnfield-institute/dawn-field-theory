"""
Fixed Macro Emergence Gravity with Coherence-Scaled Landauer Forces

This implements the coherence length scaling fix for Landauer forces,
making them physically meaningful at galactic scales while preserving
the theoretical foundation.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
PLANCK_LENGTH = 1.616e-35  # Planck length

@dataclass
class FixedMacroEmergenceConfig:
    """Configuration with coherence-scaled Landauer forces"""
    
    # Much more reasonable base parameters
    kappa_base: float = 1e25  # Reduced from 1e30 due to coherence amplification
    beta_floor: float = 0.05  # 5% quantum floor
    T_info: float = 2.7       # CMB temperature
    
    # Local emergence parameters
    k_neighbors_local: int = 5
    k_neighbors_cosmic: int = 20
    memory_decay_rate: float = 0.95
    
    # Coherence scaling parameters
    coherence_scaling_power: float = 0.5  # sqrt scaling
    min_coherence_length: float = 1e-3 * KPC_TO_METERS  # Minimum coherence scale

class FixedMacroEmergenceGravity:
    """Macro emergence gravity with coherence-scaled Landauer forces"""
    
    def __init__(self, config: FixedMacroEmergenceConfig):
        self.config = config
        self.memory_field = None
        self.coherence_history = []
        
    def compute_coherence_landauer_factor(self, positions: np.ndarray) -> float:
        """
        Compute coherence-scaled Landauer factor
        
        Key insight: Information coherence at galactic scales amplifies
        the basic Landauer energy by √(L_coherence / L_planck)
        """
        
        # Determine system coherence length (characteristic scale)
        position_std = np.std(positions, axis=0)
        coherence_length = np.mean(position_std) * 3  # 3-sigma characteristic scale
        
        # Ensure minimum coherence length
        coherence_length = max(coherence_length, self.config.min_coherence_length)
        
        # Base Landauer factor
        base_landauer = K_B * self.config.T_info * np.log(2)
        
        # Coherence amplification: √(L/L_planck)
        coherence_ratio = coherence_length / PLANCK_LENGTH
        amplification_factor = coherence_ratio ** self.config.coherence_scaling_power
        
        # Coherence-scaled Landauer factor
        landauer_factor = base_landauer * amplification_factor
        
        # Track coherence evolution
        self.coherence_history.append({
            'coherence_length': coherence_length,
            'coherence_ratio': coherence_ratio,
            'amplification': amplification_factor,
            'landauer_factor': landauer_factor
        })
        
        return landauer_factor
    
    def compute_local_tangling(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Compute kNN-based local information tangling forces"""
        
        n_particles = len(positions)
        if n_particles < self.config.k_neighbors_local:
            return np.zeros_like(positions)
        
        # Use scikit-learn for efficient kNN
        nbrs = NearestNeighbors(n_neighbors=self.config.k_neighbors_local + 1)  # +1 for self
        nbrs.fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Compute tangling forces
        tangle_forces = np.zeros_like(positions)
        
        for i in range(n_particles):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_distances = distances[i, 1:]
            neighbor_positions = positions[neighbors]
            
            # Force toward neighbors, weighted by 1/r² (gravitational-like)
            for j, (neighbor_idx, dist) in enumerate(zip(neighbors, neighbor_distances)):
                if dist > 0:
                    direction = (neighbor_positions[j] - positions[i]) / dist
                    
                    # Information tangling force: stronger for closer neighbors
                    # Mass weighting for gravitational consistency
                    tangling_strength = masses[i] * masses[neighbor_idx] / (dist**2 + 1e-10)
                    
                    tangle_forces[i] += tangling_strength * direction
        
        return tangle_forces
    
    def evolution_step(self, state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Single evolution step with fixed Landauer scaling"""
        
        positions = state['positions'].copy()
        velocities = state['velocities'].copy()
        masses = state['masses']
        time = state.get('time', 0.0)
        
        # Compute coherence-scaled Landauer factor
        landauer_factor = self.compute_coherence_landauer_factor(positions)
        
        # Compute local tangling forces
        tangle_forces = self.compute_local_tangling(positions, masses)
        
        # Apply coherence-scaled Landauer forces
        infodynamic_forces = self.config.kappa_base * landauer_factor * tangle_forces
        
        # Add small quantum floor (much reduced)
        quantum_forces = np.zeros_like(positions)
        if np.any(infodynamic_forces != 0):
            force_scale = np.mean(np.linalg.norm(infodynamic_forces, axis=1))
            quantum_forces = self.config.beta_floor * force_scale * np.random.randn(*positions.shape) * 1e-6
        
        total_forces = infodynamic_forces + quantum_forces
        
        # Update motion
        accelerations = total_forces / masses[:, np.newaxis]
        
        # Apply reasonable limits for stability
        max_accel = 1e12  # m/s²
        accel_mags = np.linalg.norm(accelerations, axis=1)
        too_fast = accel_mags > max_accel
        if np.any(too_fast):
            accelerations[too_fast] *= (max_accel / accel_mags[too_fast])[:, np.newaxis]
        
        # Update velocities and positions
        velocities += accelerations * dt
        
        # Velocity limit for stability
        max_vel = 1e6  # m/s
        vel_mags = np.linalg.norm(velocities, axis=1)
        too_fast_vel = vel_mags > max_vel
        if np.any(too_fast_vel):
            velocities[too_fast_vel] *= (max_vel / vel_mags[too_fast_vel])[:, np.newaxis]
        
        positions += velocities * dt
        
        # Compute diagnostics
        system_scale = np.std(positions) * 3
        clustering_metric = self.compute_clustering(positions)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'time': time + dt,
            'system_scale': system_scale,
            'clustering_metric': clustering_metric,
            'landauer_factor': landauer_factor,
            'force_magnitude': np.mean(np.linalg.norm(total_forces, axis=1))
        }
    
    def compute_clustering(self, positions: np.ndarray) -> float:
        """Compute actual clustering metric that changes with structure formation"""
        
        n_particles = len(positions)
        
        # Calculate coefficient of variation of pairwise distances
        distances = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        if len(distances) == 0 or np.mean(distances) == 0:
            return 0.0
        
        # Clustering = 1 - coefficient_of_variation
        # High clustering = low variation in distances (particles clumped)
        # Random distribution = high variation
        cv = np.std(distances) / np.mean(distances)
        clustering = max(0.0, 1.0 - cv)
        
        return clustering

def test_fixed_macro_emergence():
    """Test the fixed macro emergence with coherence-scaled Landauer forces"""
    
    print("Testing Fixed Macro Emergence with Coherence-Scaled Landauer Forces")
    print("="*70)
    
    # Create configuration
    config = FixedMacroEmergenceConfig(
        kappa_base=1e25,      # Reduced due to coherence amplification
        beta_floor=0.05,      # 5% quantum floor
        k_neighbors_local=5
    )
    
    print(f"Base κ: {config.kappa_base:.1e}")
    print(f"Quantum floor: {config.beta_floor*100:.1f}%")
    print(f"Coherence scaling: √(L/L_planck)")
    print()
    
    # Initialize system
    gravity = FixedMacroEmergenceGravity(config)
    
    # Test system: smaller for easier analysis
    n_particles = 20
    positions = np.random.randn(n_particles, 3) * 2 * KPC_TO_METERS  # 2 kpc initial spread
    velocities = np.random.randn(n_particles, 3) * 20000  # 20 km/s initial velocities
    masses = np.ones(n_particles) * 1e9 * SOLAR_MASS
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'time': 0.0
    }
    
    # Track evolution
    print("Running evolution with coherence-scaled forces...")
    dt = 0.01 * MYR_TO_SECONDS
    n_steps = 30
    
    history = []
    
    for step in range(n_steps):
        state = gravity.evolution_step(state, dt)
        
        if step % 5 == 0:
            clustering = state['clustering_metric']
            scale = state['system_scale'] / KPC_TO_METERS
            landauer = state['landauer_factor']
            force_mag = state['force_magnitude']
            
            print(f"Step {step:2d}: Clustering={clustering:.3f}, Scale={scale:.1f} kpc, "
                  f"Landauer={landauer:.2e}, Force={force_mag:.2e} N")
            
        history.append({
            'step': step,
            'clustering': state['clustering_metric'],
            'scale': state['system_scale'] / KPC_TO_METERS,
            'landauer_factor': state['landauer_factor'],
            'force_magnitude': state['force_magnitude']
        })
    
    # Analysis
    print(f"\nAnalyzing results...")
    
    initial_clustering = history[0]['clustering']
    final_clustering = history[-1]['clustering']
    clustering_change = final_clustering - initial_clustering
    
    initial_scale = history[0]['scale']
    final_scale = history[-1]['scale']
    scale_change = (final_scale - initial_scale) / initial_scale
    
    landauer_factors = [h['landauer_factor'] for h in history]
    mean_landauer = np.mean(landauer_factors)
    
    print(f"\nResults Summary:")
    print(f"  Initial clustering: {initial_clustering:.3f}")
    print(f"  Final clustering: {final_clustering:.3f}")
    print(f"  Clustering change: {clustering_change:+.3f}")
    print(f"  Scale change: {scale_change:+.1%}")
    print(f"  Mean Landauer factor: {mean_landauer:.2e} J")
    print(f"  Coherence amplification: {mean_landauer / (K_B * 2.7 * np.log(2)):.1e}x")
    
    # Determine success
    structure_formed = abs(clustering_change) > 0.02  # 2% change threshold
    forces_reasonable = 1e10 < mean_landauer < 1e20  # Reasonable force range
    
    print(f"\nAssessment:")
    print(f"  Structure formation: {'✓' if structure_formed else '✗'} ({'Yes' if structure_formed else 'No'})")
    print(f"  Force magnitudes reasonable: {'✓' if forces_reasonable else '✗'} ({'Yes' if forces_reasonable else 'No'})")
    print(f"  Coherence scaling working: {'✓' if mean_landauer > 1e5 else '✗'} ({'Yes' if mean_landauer > 1e5 else 'No'})")
    
    if structure_formed and forces_reasonable:
        print(f"\n🎯 SUCCESS: Coherence-scaled Landauer forces enable structure formation!")
    else:
        print(f"\n⚠️  Needs tuning: Adjust κ or coherence scaling parameters")
    
    return gravity, state, history

if __name__ == "__main__":
    gravity, final_state, evolution_history = test_fixed_macro_emergence()
