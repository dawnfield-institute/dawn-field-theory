"""
Landauer Parameter Optimization Validator

Fine-tune the coherence scaling parameters to find the sweet spot where:
1. Forces are strong enough to create structure formation
2. Forces are not so strong they cause numerical instability
3. Landauer principle is properly preserved
"""

import numpy as np
import matplotlib.pyplot as plt
from fixed_landauer_macro_emergence import FixedMacroEmergenceGravity, FixedMacroEmergenceConfig

# Physical constants
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
G = 6.67430e-11

def test_parameter_combinations():
    """Test different combinations of κ and coherence scaling power"""
    
    print("LANDAUER PARAMETER OPTIMIZATION")
    print("="*50)
    
    # Test parameters
    kappa_values = [1e15, 1e18, 1e20, 1e22, 1e25]
    coherence_powers = [0.25, 0.3, 0.4, 0.5, 0.6]  # Different scaling powers
    
    results = []
    
    for kappa in kappa_values:
        for power in coherence_powers:
            print(f"\nTesting κ={kappa:.0e}, coherence_power={power:.2f}")
            
            # Create configuration
            config = FixedMacroEmergenceConfig(
                kappa_base=kappa,
                beta_floor=0.02,  # Very small quantum floor
                coherence_scaling_power=power,
                k_neighbors_local=5
            )
            
            # Initialize system
            gravity = FixedMacroEmergenceGravity(config)
            
            # Small test system
            n_particles = 15
            positions = np.random.randn(n_particles, 3) * 1.5 * KPC_TO_METERS
            velocities = np.random.randn(n_particles, 3) * 15000  # 15 km/s
            masses = np.ones(n_particles) * 1e9 * SOLAR_MASS
            
            state = {
                'positions': positions,
                'velocities': velocities,
                'masses': masses,
                'time': 0.0
            }
            
            # Short evolution test
            dt = 0.005 * MYR_TO_SECONDS
            n_steps = 20
            
            try:
                initial_clustering = gravity.compute_clustering(positions)
                max_force = 0
                min_force = float('inf')
                final_clustering = initial_clustering
                
                for step in range(n_steps):
                    state = gravity.evolution_step(state, dt)
                    
                    force_mag = state['force_magnitude']
                    max_force = max(max_force, force_mag)
                    min_force = min(min_force, force_mag)
                    
                    # Check for instability
                    if force_mag > 1e40 or not np.isfinite(force_mag):
                        print(f"  UNSTABLE at step {step}: force = {force_mag:.2e}")
                        break
                
                final_clustering = state['clustering_metric']
                landauer_factor = state['landauer_factor']
                
                # Analyze results
                clustering_change = final_clustering - initial_clustering
                stable = max_force < 1e40 and np.isfinite(max_force)
                structure_formed = abs(clustering_change) > 0.01
                
                # Compare to gravitational force scale
                typical_separation = 1.0 * KPC_TO_METERS
                grav_force = G * (1e9 * SOLAR_MASS)**2 / typical_separation**2
                force_ratio = max_force / grav_force if grav_force > 0 else 0
                
                result = {
                    'kappa': kappa,
                    'coherence_power': power,
                    'initial_clustering': initial_clustering,
                    'final_clustering': final_clustering,
                    'clustering_change': clustering_change,
                    'max_force': max_force,
                    'landauer_factor': landauer_factor,
                    'stable': stable,
                    'structure_formed': structure_formed,
                    'force_ratio_to_gravity': force_ratio
                }
                
                results.append(result)
                
                print(f"  Clustering: {initial_clustering:.3f} → {final_clustering:.3f} (Δ={clustering_change:+.3f})")
                print(f"  Max force: {max_force:.2e} N (vs gravity: {force_ratio:.1e})")
                print(f"  Landauer: {landauer_factor:.2e} J")
                print(f"  Stable: {stable}, Structure: {structure_formed}")
                
            except Exception as e:
                print(f"  FAILED: {e}")
                result = {
                    'kappa': kappa,
                    'coherence_power': power,
                    'stable': False,
                    'structure_formed': False,
                    'error': str(e)
                }
                results.append(result)
    
    return results

def analyze_optimization_results(results):
    """Analyze the optimization results to find best parameters"""
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*50)
    
    # Filter successful runs
    stable_results = [r for r in results if r.get('stable', False)]
    structure_results = [r for r in results if r.get('structure_formed', False)]
    
    print(f"Stable configurations: {len(stable_results)}/{len(results)}")
    print(f"Structure formation: {len(structure_results)}/{len(results)}")
    
    if not stable_results:
        print("No stable configurations found!")
        return
    
    # Find configurations with reasonable force ratios
    reasonable_force_results = [
        r for r in stable_results 
        if 1e-5 < r.get('force_ratio_to_gravity', 0) < 1e5
    ]
    
    print(f"Reasonable force magnitudes: {len(reasonable_force_results)}/{len(stable_results)}")
    
    # Best candidates
    if reasonable_force_results:
        # Sort by clustering change (absolute value)
        best_candidates = sorted(
            reasonable_force_results, 
            key=lambda x: abs(x.get('clustering_change', 0)), 
            reverse=True
        )
        
        print(f"\nTop 3 candidates:")
        print(f"{'κ':>8} {'Power':>6} {'ΔCluster':>10} {'Force Ratio':>12} {'Landauer':>12}")
        print("-" * 60)
        
        for i, candidate in enumerate(best_candidates[:3]):
            kappa = candidate['kappa']
            power = candidate['coherence_power']
            delta_cluster = candidate['clustering_change']
            force_ratio = candidate['force_ratio_to_gravity']
            landauer = candidate['landauer_factor']
            
            print(f"{kappa:>8.0e} {power:>6.2f} {delta_cluster:>+10.3f} {force_ratio:>12.1e} {landauer:>12.2e}")
        
        # Recommend best parameters
        best = best_candidates[0]
        print(f"\nRECOMMENDED PARAMETERS:")
        print(f"  κ = {best['kappa']:.0e}")
        print(f"  Coherence scaling power = {best['coherence_power']:.2f}")
        print(f"  Expected clustering change = {best['clustering_change']:+.3f}")
        print(f"  Force ratio to gravity = {best['force_ratio_to_gravity']:.1e}")
        
        return best
    else:
        print("No configurations with reasonable force magnitudes found!")
        return None

def create_optimized_configuration(best_params):
    """Create the optimized configuration for production use"""
    
    if best_params is None:
        print("Cannot create optimized configuration - no good parameters found")
        return None
    
    print(f"\n{'='*50}")
    print("CREATING OPTIMIZED CONFIGURATION")
    print("="*50)
    
    config = FixedMacroEmergenceConfig(
        kappa_base=best_params['kappa'],
        beta_floor=0.02,  # 2% quantum floor
        coherence_scaling_power=best_params['coherence_power'],
        k_neighbors_local=5,
        k_neighbors_cosmic=20,
        memory_decay_rate=0.95
    )
    
    print(f"Optimized Configuration:")
    print(f"  κ = {config.kappa_base:.0e}")
    print(f"  β = {config.beta_floor*100:.1f}%")
    print(f"  Coherence power = {config.coherence_scaling_power:.2f}")
    print(f"  Local neighbors = {config.k_neighbors_local}")
    print(f"  Expected Landauer amplification = ~{best_params['landauer_factor']/(2.58e-23):.1e}x")
    
    return config

def test_optimized_configuration(config):
    """Test the optimized configuration with longer evolution"""
    
    if config is None:
        return
    
    print(f"\n{'='*50}")
    print("TESTING OPTIMIZED CONFIGURATION")
    print("="*50)
    
    gravity = FixedMacroEmergenceGravity(config)
    
    # Test system
    n_particles = 25
    positions = np.random.randn(n_particles, 3) * 2 * KPC_TO_METERS
    velocities = np.random.randn(n_particles, 3) * 10000  # 10 km/s
    masses = np.ones(n_particles) * 1e9 * SOLAR_MASS
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'time': 0.0
    }
    
    print("Running extended evolution test...")
    dt = 0.01 * MYR_TO_SECONDS
    n_steps = 50
    
    initial_clustering = gravity.compute_clustering(positions)
    
    for step in range(n_steps):
        state = gravity.evolution_step(state, dt)
        
        if step % 10 == 0:
            clustering = state['clustering_metric']
            scale = state['system_scale'] / KPC_TO_METERS
            force = state['force_magnitude']
            
            print(f"Step {step:2d}: Clustering={clustering:.3f}, Scale={scale:.1f} kpc, Force={force:.2e} N")
    
    final_clustering = state['clustering_metric']
    clustering_change = final_clustering - initial_clustering
    
    print(f"\nFinal Assessment:")
    print(f"  Initial clustering: {initial_clustering:.3f}")
    print(f"  Final clustering: {final_clustering:.3f}")
    print(f"  Total change: {clustering_change:+.3f}")
    print(f"  Structure formation: {'✓ YES' if abs(clustering_change) > 0.02 else '✗ No'}")
    
    return clustering_change

if __name__ == "__main__":
    # Run optimization
    results = test_parameter_combinations()
    best_params = analyze_optimization_results(results)
    optimized_config = create_optimized_configuration(best_params)
    clustering_change = test_optimized_configuration(optimized_config)
