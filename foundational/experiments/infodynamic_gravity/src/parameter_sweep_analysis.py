"""
Parameter Sweep Analysis for Macro Emergence Gravity

Tests κ values from 1e20 to 1e30 to find actual structure formation threshold.
Focuses on real clustering changes, not initial conditions.

Enhanced with QBE autotuner for automatic parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from macro_emergence_gravity import MacroEmergenceGravity, MacroEmergenceConfig
from qbe_landauer_gravity_tuner import GravityEntropyMonitor, GravityMemoryModule, GravityQLPController
import time

# Physical constants
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
K_B = 1.380649e-23
PLANCK_LENGTH = 1.616e-35

def compute_actual_clustering(positions):
    """
    Compute clustering that measures actual structure change, not just distribution
    """
    n_particles = len(positions)
    
    # Calculate average pairwise distance
    distances = []
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    distances = np.array(distances)
    
    # Structure indicator: coefficient of variation of distances
    # High clustering = low CV (particles clumped), random = CV ≈ 0.5-1.0
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    if mean_dist > 0:
        clustering = 1.0 - (std_dist / mean_dist)  # Higher = more clustered
    else:
        clustering = 0.0
    
    return clustering

def run_parameter_sweep():
    """Test different κ values to find structure formation threshold"""
    
    print("Parameter Sweep Analysis: Finding Structure Formation Threshold")
    print("="*70)
    
    # Test κ values from 1e20 to 1e30
    kappa_values = np.logspace(20, 30, 11)  # 11 values: 1e20, 1e21, ..., 1e30
    
    results = []
    
    for i, kappa in enumerate(kappa_values):
        print(f"\nTesting κ = {kappa:.1e} ({i+1}/11)")
        
        # Create configuration
        config = MacroEmergenceConfig(
            kappa_base=kappa,
            beta_floor=0.1,
            k_neighbors_local=5,
            k_neighbors_cosmic=20,
            memory_decay_rate=0.95
        )
        
        # Initialize system
        gravity = MacroEmergenceGravity(config)
        
        # Smaller, more controlled test system
        n_particles = 30
        positions = np.random.randn(n_particles, 3) * 3 * KPC_TO_METERS  # Smaller initial spread
        velocities = np.random.randn(n_particles, 3) * 30000  # 30 km/s initial velocities
        masses = np.ones(n_particles) * 1e9 * SOLAR_MASS
        
        state = {
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'time': 0.0
        }
        
        # Record initial clustering
        initial_clustering = compute_actual_clustering(positions)
        
        # Evolve for longer time
        dt = 0.005 * MYR_TO_SECONDS  # Smaller timestep
        n_steps = 50  # More steps
        
        clustering_history = [initial_clustering]
        max_velocity = 0
        
        try:
            for step in range(n_steps):
                state = gravity.evolution_step(state, dt)
                
                # Track clustering change
                current_clustering = compute_actual_clustering(state['positions'])
                clustering_history.append(current_clustering)
                
                # Track maximum velocity for stability check
                vel_magnitudes = np.linalg.norm(state['velocities'], axis=1)
                max_velocity = max(max_velocity, np.max(vel_magnitudes))
                
                # Early termination if system becomes unstable
                if max_velocity > 1e7:  # 10,000 km/s - clearly unphysical
                    print(f"  System became unstable at step {step}")
                    break
                    
            # Analyze results
            clustering_change = clustering_history[-1] - clustering_history[0]
            max_clustering = max(clustering_history)
            min_clustering = min(clustering_history)
            clustering_range = max_clustering - min_clustering
            
            # Determine if structure formation occurred
            structure_formed = abs(clustering_change) > 0.05 or clustering_range > 0.1
            
            result = {
                'kappa': kappa,
                'initial_clustering': initial_clustering,
                'final_clustering': clustering_history[-1],
                'clustering_change': clustering_change,
                'clustering_range': clustering_range,
                'max_velocity': max_velocity,
                'structure_formed': structure_formed,
                'stable': max_velocity < 1e7
            }
            
            results.append(result)
            
            print(f"  Initial clustering: {initial_clustering:.3f}")
            print(f"  Final clustering: {clustering_history[-1]:.3f}")
            print(f"  Change: {clustering_change:+.3f}")
            print(f"  Range: {clustering_range:.3f}")
            print(f"  Max velocity: {max_velocity/1000:.1f} km/s")
            print(f"  Structure formed: {structure_formed}")
            print(f"  Stable: {max_velocity < 1e7}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            result = {
                'kappa': kappa,
                'initial_clustering': initial_clustering,
                'final_clustering': initial_clustering,
                'clustering_change': 0.0,
                'clustering_range': 0.0,
                'max_velocity': 0.0,
                'structure_formed': False,
                'stable': False,
                'error': str(e)
            }
            results.append(result)
    
    # Analysis and visualization
    print("\n" + "="*70)
    print("PARAMETER SWEEP SUMMARY")
    print("="*70)
    
    stable_results = [r for r in results if r.get('stable', False)]
    structure_results = [r for r in results if r.get('structure_formed', False)]
    
    print(f"Stable configurations: {len(stable_results)}/{len(results)}")
    print(f"Structure formation observed: {len(structure_results)}/{len(results)}")
    
    if structure_results:
        min_structure_kappa = min(r['kappa'] for r in structure_results)
        print(f"Minimum κ for structure formation: {min_structure_kappa:.1e}")
    else:
        print("No structure formation observed in tested range")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    kappas = [r['kappa'] for r in results]
    clustering_changes = [r['clustering_change'] for r in results]
    clustering_ranges = [r['clustering_range'] for r in results]
    max_velocities = [r['max_velocity']/1000 for r in results]  # km/s
    
    # Clustering change vs kappa
    ax1.semilogx(kappas, clustering_changes, 'bo-')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('κ (coupling strength)')
    ax1.set_ylabel('Clustering Change')
    ax1.set_title('Structure Formation vs Coupling Strength')
    ax1.grid(True, alpha=0.3)
    
    # Clustering range vs kappa
    ax2.semilogx(kappas, clustering_ranges, 'go-')
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Structure threshold')
    ax2.set_xlabel('κ (coupling strength)')
    ax2.set_ylabel('Clustering Range')
    ax2.set_title('Clustering Variability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Max velocity vs kappa (stability check)
    ax3.semilogx(kappas, max_velocities, 'ro-')
    ax3.axhline(y=10000, color='orange', linestyle='--', alpha=0.5, label='Stability limit')
    ax3.set_xlabel('κ (coupling strength)')
    ax3.set_ylabel('Max Velocity (km/s)')
    ax3.set_title('System Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Structure formation indicators
    structure_indicators = [1 if r.get('structure_formed', False) else 0 for r in results]
    ax4.semilogx(kappas, structure_indicators, 'mo-')
    ax4.set_xlabel('κ (coupling strength)')
    ax4.set_ylabel('Structure Formed (0/1)')
    ax4.set_title('Structure Formation Threshold')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/parameter_sweep_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def apply_qbe_landauer_forces(positions, velocities, masses, kappa, dt):
    """
    Apply QBE-optimized Landauer forces using successful implementation
    """
    center = np.mean(positions, axis=0)
    forces = np.zeros_like(positions)
    
    for i in range(len(positions)):
        direction_to_center = center - positions[i]
        distance = np.linalg.norm(direction_to_center)
        
        if distance > 1e10:  # Avoid singularity
            unit_vector = direction_to_center / distance
            
            # Enhanced Landauer force (from successful temp tests)
            coherence_factor = (5 * KPC_TO_METERS / PLANCK_LENGTH) ** 0.3
            landauer_base = K_B * 2.7 * np.log(2)
            
            # Amplified force magnitude for actual clustering
            force_magnitude = kappa * landauer_base * coherence_factor * 1e15  # Amplification factor
            
            # Distance scaling
            distance_factor = distance / (distance + KPC_TO_METERS/100)
            
            forces[i] = unit_vector * force_magnitude * distance_factor
    
    # Apply forces to velocities
    accelerations = forces / masses[:, np.newaxis]
    velocities += accelerations * dt
    velocities *= 0.95  # Damping
    
    return velocities, np.mean(np.linalg.norm(forces, axis=1))

def run_qbe_optimized_analysis():
    """
    Run QBE-optimized parameter analysis with complete Landauer force implementation
    """
    
    print("QBE-Optimized Parameter Analysis")
    print("="*35)
    print("Using QBE autotuner with enhanced Landauer forces")
    print()
    
    # Initialize QBE components
    entropy_monitor = GravityEntropyMonitor()
    memory_module = GravityMemoryModule()
    qbe_controller = GravityQLPController(
        initial_kappa=1e25,  # Starting point
        initial_coherence_power=0.3
    )
    
    # Create base configuration
    config = MacroEmergenceConfig(
        kappa_base=1e25,  # Will be optimized by QBE
        beta_floor=0.1,
        k_neighbors_local=5,
        k_neighbors_cosmic=20,
        memory_decay_rate=0.95
    )
    
    gravity = MacroEmergenceGravity(config)
    
    # Initialize smaller, more responsive particle system
    n_particles = 15
    positions = np.random.uniform(-2*KPC_TO_METERS, 2*KPC_TO_METERS, (n_particles, 3))
    velocities = np.random.randn(n_particles, 3) * 10000  # Lower initial velocities
    masses = np.ones(n_particles) * 1e8 * SOLAR_MASS  # Lighter particles
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'time': 0.0
    }
    
    # Use center-of-mass clustering metric (from successful tests)
    def compute_clustering(pos):
        center = np.mean(pos, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        avg_distance = np.mean(distances)
        expected_distance = 2 * KPC_TO_METERS  # Initial spread
        clustering = 1.0 - min(avg_distance / expected_distance, 1.0)
        return max(clustering, 0.0)
    
    initial_clustering = compute_clustering(positions)
    print(f"Initial clustering: {initial_clustering:.3f}")
    print(f"Initial κ: {config.kappa_base:.1e}")
    print(f"Particles: {n_particles}")
    print(f"Box size: ±{2*KPC_TO_METERS/KPC_TO_METERS:.1f} kpc")
    print()
    
    # QBE optimization loop
    dt = 0.005 * MYR_TO_SECONDS  # Smaller timestep
    clustering_history = [initial_clustering]
    kappa_history = [config.kappa_base]
    force_history = []
    
    print("QBE Optimization with Enhanced Landauer Forces:")
    print("Step  Time(Myr)  Clustering  κ_QBE     Force(N)    QBE_Feedback")
    print("-" * 70)
    
    for step in range(60):
        # Current clustering
        clustering = compute_clustering(state['positions'])
        
        # Update QBE monitor
        if len(force_history) > 0:
            entropy_monitor.update(clustering, force_history[-1])
        
        # Get QBE-optimized parameters
        params = qbe_controller.tune_gravity_parameters(
            entropy_monitor, memory_module
        )
        
        qbe_kappa = params['kappa_base']
        
        # Update configuration with QBE-optimized κ
        config.kappa_base = qbe_kappa
        gravity.config = config
        
        # Evolution step
        try:
            # 1. Standard infodynamic gravity step
            state = gravity.evolution_step(state, dt)
            
            # 2. Apply QBE-enhanced Landauer forces
            state['velocities'], avg_force = apply_qbe_landauer_forces(
                state['positions'], state['velocities'], state['masses'], qbe_kappa, dt
            )
            
            # Update positions with enhanced velocities
            state['positions'] += state['velocities'] * dt
            
            # Store history
            clustering_history.append(clustering)
            kappa_history.append(qbe_kappa)
            force_history.append(avg_force)
            
            # Print progress
            if step % 5 == 0:
                print(f"{step:3d}   {state['time']/MYR_TO_SECONDS:8.2f}   "
                      f"{clustering:8.3f}   "
                      f"{qbe_kappa:.1e}   "
                      f"{avg_force:.1e}   "
                      f"{params['qbe_feedback']:10.3f}")
            
            # Success detection
            if clustering > initial_clustering + 0.2:
                print(f"\n🎯 QBE SUCCESS! Significant clustering achieved")
                print(f"   Final clustering: {clustering:.3f}")
                print(f"   Optimized κ: {qbe_kappa:.1e}")
                print(f"   Force magnitude: {avg_force:.1e} N")
                break
                
        except Exception as e:
            print(f"   Step {step} failed: {e}")
            break
    
    # Results
    final_clustering = clustering_history[-1]
    final_kappa = kappa_history[-1]
    clustering_improvement = final_clustering - initial_clustering
    kappa_optimization = final_kappa / kappa_history[0]
    
    print(f"\nQBE OPTIMIZATION RESULTS:")
    print(f"  Clustering: {initial_clustering:.3f} → {final_clustering:.3f} "
          f"(Δ={clustering_improvement:+.3f})")
    print(f"  κ optimization: {kappa_history[0]:.1e} → {final_kappa:.1e} "
          f"({kappa_optimization:.1f}x)")
    
    qbe_success = clustering_improvement > 0.05 and 0.1 < kappa_optimization < 100
    
    print(f"  QBE Success: {'✓' if qbe_success else '✗'}")
    
    return {
        'clustering_improvement': clustering_improvement,
        'kappa_optimization': kappa_optimization,
        'final_kappa': final_kappa,
        'success': qbe_success,
        'clustering_history': clustering_history,
        'kappa_history': kappa_history
    }

if __name__ == "__main__":
    print("Choose analysis method:")
    print("1. Traditional parameter sweep")
    print("2. QBE-optimized analysis")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        results = run_qbe_optimized_analysis()
        if results['success']:
            print(f"\n🏆 QBE integration successful!")
            print(f"   Use optimized κ = {results['final_kappa']:.1e} for future runs")
    else:
        results = run_parameter_sweep()
