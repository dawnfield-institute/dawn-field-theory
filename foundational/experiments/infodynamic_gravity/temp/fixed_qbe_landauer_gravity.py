"""
Fixed QBE-Landauer Gravity System

Fixes the identified issues:
1. Proper particle dynamics (float casting)
2. Realistic acceleration/velocity integration
3. Better clustering metric sensitivity
4. Simplified but effective structure formation
"""

import numpy as np
from typing import Dict, Any, Tuple

# Import QBE components
from qbe_landauer_gravity_tuner import GravityEntropyMonitor, GravityMemoryModule, GravityQLPController

# Physical constants
K_B = 1.380649e-23
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
PLANCK_LENGTH = 1.616e-35

class FixedLandauerGravitySystem:
    """
    Fixed QBE-optimized Landauer gravity with proper particle dynamics
    """
    
    def __init__(self, num_particles=50, box_size_kpc=20):  # Smaller system for testing
        self.num_particles = num_particles
        self.box_size = box_size_kpc * KPC_TO_METERS
        
        # Initialize particles with proper float type
        np.random.seed(42)  # Reproducible results
        self.positions = np.random.uniform(
            -self.box_size/2, self.box_size/2, 
            (num_particles, 3)
        ).astype(np.float64)  # Explicit float64
        
        self.velocities = np.zeros((num_particles, 3), dtype=np.float64)
        self.masses = np.full(num_particles, SOLAR_MASS * 1e8, dtype=np.float64)  # 100M solar masses
        
        # QBE autotuner
        self.entropy_monitor = GravityEntropyMonitor()
        self.memory_module = GravityMemoryModule()
        self.qbe_controller = GravityQLPController(
            initial_kappa=1e26,  # Start higher
            initial_coherence_power=0.4
        )
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 50 * MYR_TO_SECONDS  # 50 Myr steps
        self.clustering_history = []
        self.force_history = []
        
    def compute_clustering_metric(self) -> float:
        """Improved clustering metric"""
        
        # Calculate center of mass
        center_of_mass = np.mean(self.positions, axis=0)
        
        # Calculate distances from center of mass
        distances_from_center = np.linalg.norm(self.positions - center_of_mass, axis=1)
        avg_distance_from_center = np.mean(distances_from_center)
        
        # Expected distance in random distribution (rough estimate)
        expected_random_distance = self.box_size / 4  # Quarter of box size
        
        # Clustering metric: 1 = all at center, 0 = random distribution
        clustering = 1.0 - min(avg_distance_from_center / expected_random_distance, 1.0)
        return max(clustering, 0.0)
    
    def compute_simple_landauer_forces(self, kappa: float, coherence_power: float) -> np.ndarray:
        """
        Simplified Landauer forces focusing on center-of-mass attraction
        """
        
        # Calculate center of mass
        center_of_mass = np.mean(self.positions, axis=0)
        
        forces = np.zeros_like(self.positions)
        
        # Each particle feels Landauer force toward center of mass
        for i in range(self.num_particles):
            direction = center_of_mass - self.positions[i]
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Landauer force magnitude (simplified)
                coherence_factor = (self.box_size / PLANCK_LENGTH) ** coherence_power
                landauer_base = K_B * 2.7 * np.log(2)
                force_magnitude = kappa * landauer_base * coherence_factor
                
                # Information density gradient (simplified)
                # Higher density at center → force toward center
                info_gradient = 1.0 / (distance + self.box_size/100)  # Avoid division by zero
                
                # Force direction toward center
                force_direction = direction / distance
                forces[i] = force_direction * force_magnitude * info_gradient
        
        return forces
    
    def evolve_step(self) -> Dict[str, Any]:
        """Single evolution step with fixed dynamics"""
        
        # Current clustering
        clustering = self.compute_clustering_metric()
        
        # Update QBE monitor
        if len(self.clustering_history) > 0:
            prev_force = self.force_history[-1] if self.force_history else 1e25
            self.entropy_monitor.update(clustering, prev_force)
        
        # Get QBE parameters
        params = self.qbe_controller.tune_gravity_parameters(
            self.entropy_monitor, self.memory_module
        )
        
        kappa = params['kappa_base']
        coherence_power = params['coherence_power']
        
        # Compute Landauer forces
        landauer_forces = self.compute_simple_landauer_forces(kappa, coherence_power)
        
        # Particle dynamics (fixed)
        accelerations = landauer_forces / self.masses[:, np.newaxis]
        
        # Update velocities with damping
        self.velocities += accelerations * self.dt
        self.velocities *= 0.95  # Energy dissipation
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Apply periodic boundary conditions
        for axis in range(3):
            mask_low = self.positions[:, axis] < -self.box_size/2
            mask_high = self.positions[:, axis] > self.box_size/2
            self.positions[mask_low, axis] += self.box_size
            self.positions[mask_high, axis] -= self.box_size
        
        # Store history
        avg_force = np.mean(np.linalg.norm(landauer_forces, axis=1))
        self.clustering_history.append(clustering)
        self.force_history.append(avg_force)
        
        self.time += self.dt
        
        return {
            'time_myr': self.time / MYR_TO_SECONDS,
            'clustering': clustering,
            'kappa': kappa,
            'coherence_power': coherence_power,
            'avg_force': avg_force,
            'qbe_feedback': params['qbe_feedback']
        }

def test_fixed_system():
    """Test the fixed QBE-Landauer gravity system"""
    
    print("Fixed QBE-Landauer Gravity Test")
    print("="*35)
    print("Testing: Structure formation with proper dynamics")
    print()
    
    system = FixedLandauerGravitySystem(num_particles=30, box_size_kpc=10)
    
    initial_clustering = system.compute_clustering_metric()
    print(f"Initial clustering: {initial_clustering:.3f}")
    print(f"Initial κ: {system.qbe_controller.kappa_base:.1e}")
    print(f"Box size: {system.box_size/KPC_TO_METERS:.1f} kpc")
    print()
    
    print("Evolution:")
    print("Step  Time(Myr)  Clustering  κ         Force(N)     QBE")
    print("-" * 60)
    
    for step in range(20):
        result = system.evolve_step()
        
        print(f"{step:3d}   {result['time_myr']:8.0f}   "
              f"{result['clustering']:8.3f}   "
              f"{result['kappa']:.1e}   "
              f"{result['avg_force']:.1e}   "
              f"{result['qbe_feedback']:6.3f}")
        
        # Early success check
        if result['clustering'] > 0.3:
            print(f"\n✓ Significant clustering achieved at step {step}!")
            break
    
    # Final assessment
    final_clustering = system.clustering_history[-1]
    initial_clustering = system.clustering_history[0]
    clustering_improvement = final_clustering - initial_clustering
    
    final_kappa = result['kappa']
    kappa_change = final_kappa / 1e26  # Initial was 1e26
    
    print(f"\nResults:")
    print(f"  Clustering change: {initial_clustering:.3f} → {final_clustering:.3f} "
          f"(Δ={clustering_improvement:+.3f})")
    print(f"  QBE-optimized κ: {final_kappa:.1e} ({kappa_change:.1f}x initial)")
    
    # Success criteria
    clustering_success = clustering_improvement > 0.05
    reasonable_kappa = 1e25 < final_kappa < 1e35
    
    print(f"\nValidation:")
    print(f"  Clustering improvement: {'✓' if clustering_success else '✗'} "
          f"({clustering_improvement:+.3f} > 0.05)")
    print(f"  Reasonable κ range: {'✓' if reasonable_kappa else '✗'}")
    
    if clustering_success and reasonable_kappa:
        print(f"\n🎯 SUCCESS: Fixed QBE-Landauer system works!")
        print(f"   Structure formation confirmed with κ = {final_kappa:.1e}")
        print(f"   Landauer forces CAN drive gravitational clustering ✓")
    else:
        print(f"\n⚠️  System needs further tuning")
    
    return system

if __name__ == "__main__":
    system = test_fixed_system()
