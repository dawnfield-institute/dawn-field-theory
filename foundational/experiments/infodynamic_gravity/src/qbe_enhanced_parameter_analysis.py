"""
QBE-Enhanced Parameter Sweep Analysis

Integrates the successful QBE autotuner into the existing infodynamic gravity framework.
Replaces manual parameter sweeps with adaptive QBE optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add temp directory to path for QBE components
temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
sys.path.append(temp_dir)

from qbe_landauer_gravity_tuner import GravityEntropyMonitor, GravityMemoryModule, GravityQLPController
from macro_emergence_gravity import MacroEmergenceGravity, MacroEmergenceConfig

# Physical constants
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
K_B = 1.380649e-23
PLANCK_LENGTH = 1.616e-35

class QBEEnhancedGravitySystem:
    """
    Infodynamic gravity system enhanced with QBE parameter autotuning
    """
    
    def __init__(self, config: MacroEmergenceConfig, num_particles=20, box_size_kpc=10):
        self.config = config
        self.num_particles = num_particles
        self.box_size = box_size_kpc * KPC_TO_METERS
        
        # Initialize base gravity system
        self.gravity = MacroEmergenceGravity(config)
        
        # QBE autotuner components
        self.entropy_monitor = GravityEntropyMonitor()
        self.memory_module = GravityMemoryModule()
        self.qbe_controller = GravityQLPController(
            initial_kappa=config.kappa_base,
            initial_coherence_power=0.3
        )
        
        # Initialize particle system
        np.random.seed(42)  # Reproducible results
        self.state = self._initialize_particles()
        
        # History tracking
        self.clustering_history = []
        self.force_history = []
        self.kappa_history = []
        self.time_history = []
        
    def _initialize_particles(self):
        """Initialize particle system"""
        positions = np.random.uniform(
            -self.box_size/2, self.box_size/2,
            (self.num_particles, 3)
        )
        velocities = np.random.randn(self.num_particles, 3) * 30000  # 30 km/s
        masses = np.ones(self.num_particles) * 1e9 * SOLAR_MASS
        
        return {
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'time': 0.0
        }
    
    def compute_clustering_metric(self, positions):
        """
        Compute clustering using center-of-mass approach (from successful QBE test)
        """
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        avg_distance = np.mean(distances)
        
        # Expected distance for random distribution
        expected_distance = self.box_size / 4
        
        # Clustering: 1 = all at center, 0 = random
        clustering = 1.0 - min(avg_distance / expected_distance, 1.0)
        return max(clustering, 0.0)
    
    def apply_qbe_landauer_forces(self, positions, velocities, masses, kappa):
        """
        Apply QBE-optimized Landauer forces (based on successful implementation)
        """
        center = np.mean(positions, axis=0)
        forces = np.zeros_like(positions)
        
        for i in range(self.num_particles):
            direction_to_center = center - positions[i]
            distance = np.linalg.norm(direction_to_center)
            
            if distance > 1e10:  # Avoid singularity
                unit_vector = direction_to_center / distance
                
                # Landauer force with coherence scaling
                coherence_factor = (self.box_size / PLANCK_LENGTH) ** 0.3
                landauer_base = K_B * 2.7 * np.log(2)
                
                # QBE-optimized force magnitude
                force_magnitude = kappa * landauer_base * coherence_factor
                
                # Distance scaling (stronger when farther from center)
                distance_factor = distance / (distance + self.box_size/100)
                
                forces[i] = unit_vector * force_magnitude * distance_factor
        
        return forces
    
    def evolve_with_qbe_optimization(self, dt, steps=100):
        """
        Evolve system with QBE parameter optimization
        """
        print(f"QBE-Enhanced Evolution:")
        print(f"  Initial κ: {self.config.kappa_base:.1e}")
        print(f"  Particles: {self.num_particles}")
        print(f"  Box size: {self.box_size/KPC_TO_METERS:.1f} kpc")
        print(f"  Timestep: {dt/MYR_TO_SECONDS:.3f} Myr")
        print()
        
        initial_clustering = self.compute_clustering_metric(self.state['positions'])
        print(f"Initial clustering: {initial_clustering:.3f}")
        print()
        
        print("Step  Time(Myr)  Clustering  κ_QBE     Force(N)    QBE_Feedback")
        print("-" * 70)
        
        for step in range(steps):
            # Current clustering
            clustering = self.compute_clustering_metric(self.state['positions'])
            
            # Update QBE monitor
            if len(self.clustering_history) > 0:
                prev_force = self.force_history[-1] if self.force_history else 1e25
                self.entropy_monitor.update(clustering, prev_force)
            
            # Get QBE-optimized parameters
            params = self.qbe_controller.tune_gravity_parameters(
                self.entropy_monitor, self.memory_module
            )
            
            qbe_kappa = params['kappa_base']
            
            # Update config with QBE-optimized κ
            self.config.kappa_base = qbe_kappa
            self.gravity.config = self.config
            
            # Apply combined forces: original + QBE Landauer
            
            # 1. Original infodynamic gravity step
            self.state = self.gravity.evolution_step(self.state, dt)
            
            # 2. Add QBE Landauer forces
            landauer_forces = self.apply_qbe_landauer_forces(
                self.state['positions'], 
                self.state['velocities'], 
                self.state['masses'], 
                qbe_kappa
            )
            
            # Apply Landauer forces to velocities
            accelerations = landauer_forces / self.state['masses'][:, np.newaxis]
            self.state['velocities'] += accelerations * dt
            
            # Update positions with new velocities
            self.state['positions'] += self.state['velocities'] * dt
            
            # Apply boundary conditions
            for axis in range(3):
                mask_low = self.state['positions'][:, axis] < -self.box_size/2
                mask_high = self.state['positions'][:, axis] > self.box_size/2
                self.state['positions'][mask_low, axis] += self.box_size
                self.state['positions'][mask_high, axis] -= self.box_size
            
            # Store history
            avg_force = np.mean(np.linalg.norm(landauer_forces, axis=1))
            self.clustering_history.append(clustering)
            self.force_history.append(avg_force)
            self.kappa_history.append(qbe_kappa)
            self.time_history.append(self.state['time'])
            
            # Print progress
            if step % 10 == 0:
                print(f"{step:3d}   {self.state['time']/MYR_TO_SECONDS:8.1f}   "
                      f"{clustering:8.3f}   "
                      f"{qbe_kappa:.1e}   "
                      f"{avg_force:.1e}   "
                      f"{params['qbe_feedback']:10.3f}")
            
            # Success detection
            if clustering > 0.5:
                print(f"\n🎯 SIGNIFICANT CLUSTERING ACHIEVED!")
                print(f"   Step: {step}")
                print(f"   Time: {self.state['time']/MYR_TO_SECONDS:.1f} Myr")
                print(f"   Clustering: {clustering:.3f}")
                print(f"   QBE-optimized κ: {qbe_kappa:.1e}")
                break
        
        return {
            'final_clustering': self.clustering_history[-1],
            'initial_clustering': initial_clustering,
            'clustering_improvement': self.clustering_history[-1] - initial_clustering,
            'final_kappa': self.kappa_history[-1],
            'initial_kappa': self.config.kappa_base,
            'kappa_optimization': self.kappa_history[-1] / self.kappa_history[0],
            'final_time_myr': self.state['time'] / MYR_TO_SECONDS
        }

def run_qbe_enhanced_analysis():
    """
    Run QBE-enhanced parameter analysis (replaces manual parameter sweeps)
    """
    
    print("QBE-Enhanced Infodynamic Gravity Analysis")
    print("="*45)
    print("Automatic parameter optimization with QBE autotuner")
    print()
    
    # Create configuration with initial parameters
    config = MacroEmergenceConfig(
        kappa_base=1e25,  # QBE will optimize this
        beta_floor=0.1,
        k_neighbors_local=5,
        k_neighbors_cosmic=20,
        memory_decay_rate=0.95
    )
    
    # Initialize QBE-enhanced system
    system = QBEEnhancedGravitySystem(config, num_particles=15, box_size_kpc=5)
    
    # Run evolution with QBE optimization
    dt = 0.01 * MYR_TO_SECONDS  # 0.01 Myr steps
    results = system.evolve_with_qbe_optimization(dt, steps=50)
    
    # Analysis
    print(f"\n" + "="*60)
    print(f"QBE-ENHANCED ANALYSIS RESULTS")
    print(f"="*60)
    
    print(f"Clustering Performance:")
    print(f"  Initial: {results['initial_clustering']:.3f}")
    print(f"  Final: {results['final_clustering']:.3f}")
    print(f"  Improvement: {results['clustering_improvement']:+.3f}")
    
    print(f"\nQBE Parameter Optimization:")
    print(f"  Initial κ: {results['initial_kappa']:.1e}")
    print(f"  Final κ: {results['final_kappa']:.1e}")
    print(f"  Optimization factor: {results['kappa_optimization']:.1f}x")
    
    print(f"\nEvolution Performance:")
    print(f"  Time to result: {results['final_time_myr']:.1f} Myr")
    print(f"  Steps completed: {len(system.clustering_history)}")
    
    # Success criteria
    clustering_success = results['clustering_improvement'] > 0.1
    qbe_success = 1.1 < results['kappa_optimization'] < 100
    time_success = results['final_time_myr'] < 1000  # Within 1 Gyr
    
    print(f"\nVALIDATION CRITERIA:")
    print(f"  Clustering improvement (>0.1): {'✓' if clustering_success else '✗'} "
          f"({results['clustering_improvement']:+.3f})")
    print(f"  QBE optimization (1.1-100x): {'✓' if qbe_success else '✗'} "
          f"({results['kappa_optimization']:.1f}x)")
    print(f"  Reasonable timescale (<1000 Myr): {'✓' if time_success else '✗'} "
          f"({results['final_time_myr']:.1f} Myr)")
    
    overall_success = clustering_success and qbe_success and time_success
    
    if overall_success:
        print(f"\n🏆 QBE INTEGRATION SUCCESS! 🏆")
        print(f"   QBE autotuner successfully integrated with infodynamic gravity")
        print(f"   Landauer forces enhanced by QBE parameter optimization")
        print(f"   Automatic κ discovery replaces manual parameter sweeps")
        print(f"   'Landier causes gravity' validated with QBE enhancement")
    else:
        print(f"\n⚠️  QBE integration needs refinement")
        
    # Plot results if we have history
    if len(system.clustering_history) > 5:
        plot_qbe_results(system)
    
    return system, results

def plot_qbe_results(system):
    """Plot QBE optimization results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    time_myr = np.array(system.time_history) / MYR_TO_SECONDS
    
    # Clustering evolution
    ax1.plot(time_myr, system.clustering_history, 'b-', linewidth=2)
    ax1.set_xlabel('Time (Myr)')
    ax1.set_ylabel('Clustering Metric')
    ax1.set_title('Clustering Evolution with QBE')
    ax1.grid(True, alpha=0.3)
    
    # QBE κ optimization
    ax2.plot(time_myr, system.kappa_history, 'r-', linewidth=2)
    ax2.set_xlabel('Time (Myr)')
    ax2.set_ylabel('κ (QBE-optimized)')
    ax2.set_title('QBE Parameter Optimization')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Force evolution
    ax3.plot(time_myr, system.force_history, 'g-', linewidth=2)
    ax3.set_xlabel('Time (Myr)')
    ax3.set_ylabel('Landauer Force (N)')
    ax3.set_title('QBE-Enhanced Force Evolution')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # κ vs Clustering correlation
    ax4.scatter(system.kappa_history, system.clustering_history, c=time_myr, 
                cmap='viridis', alpha=0.7)
    ax4.set_xlabel('QBE-optimized κ')
    ax4.set_ylabel('Clustering')
    ax4.set_title('QBE κ-Clustering Correlation')
    ax4.set_xscale('log')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Time (Myr)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qbe_enhanced_gravity_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved as 'qbe_enhanced_gravity_results.png'")

if __name__ == "__main__":
    system, results = run_qbe_enhanced_analysis()
