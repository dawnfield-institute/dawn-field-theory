"""
Corrected Minimal Clustering Test

Fix the force magnitudes and timesteps to get realistic particle motion.
The previous test had forces/accelerations that were way too strong.
"""

import numpy as np

def corrected_clustering_test():
    """Test with corrected force magnitudes"""
    
    print("Corrected 3-Particle Clustering Test")
    print("="*37)
    
    # 3 particles in a line (in meters, not some arbitrary units)
    positions = np.array([
        [-1e18, 0.0, 0.0],  # 1e18 meters apart (~0.1 kpc)
        [0.0, 0.0, 0.0], 
        [1e18, 0.0, 0.0]
    ], dtype=np.float64)
    
    velocities = np.zeros_like(positions)
    
    print("Initial separation:", 2e18/3.086e19, "kpc")
    
    # Clustering metric
    def compute_clustering(pos):
        center = np.mean(pos, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        avg_distance = np.mean(distances)
        initial_distance = 1e18 * 2/3  # Initial average distance from center
        clustering = 1.0 - min(avg_distance / initial_distance, 1.0)
        return max(clustering, 0.0)
    
    initial_clustering = compute_clustering(positions)
    print(f"Initial clustering: {initial_clustering:.3f}")
    print()
    
    # Realistic parameters
    dt = 1e13  # 10 million years in seconds
    mass = 1e42  # 100 billion solar masses in kg
    
    print("Evolution:")
    print("Step  Separation(kpc)  Clustering  Center_Distance(kpc)")
    print("-" * 55)
    
    for step in range(15):
        # Center of mass
        center = np.mean(positions, axis=0)
        
        # Forces toward center (gravity-like)
        forces = np.zeros_like(positions)
        for i in range(3):
            direction = center - positions[i]
            distance = np.linalg.norm(direction)
            if distance > 0:
                force_direction = direction / distance
                
                # Reasonable force: like gravity between galaxy clusters
                # F ≈ GMm/r² but with Landauer enhancement
                G = 6.67e-11
                effective_mass = mass * 1000  # Landauer enhancement factor
                gravitational_force = G * effective_mass * mass / (distance**2)
                
                forces[i] = force_direction * gravitational_force
        
        # Update motion with proper integration
        accelerations = forces / mass
        velocities += accelerations * dt
        
        # Add damping (energy dissipation)
        velocities *= 0.99  # Very mild damping
        
        positions += velocities * dt
        
        # Calculate metrics
        clustering = compute_clustering(positions)
        center = np.mean(positions, axis=0)
        avg_distance_from_center = np.mean(np.linalg.norm(positions - center, axis=1))
        
        # Convert to kpc for display
        separation_kpc = np.max(positions[:, 0]) - np.min(positions[:, 0])
        separation_kpc /= 3.086e19
        
        center_distance_kpc = avg_distance_from_center / 3.086e19
        
        print(f"{step:3d}     {separation_kpc:10.2f}   {clustering:8.3f}       {center_distance_kpc:12.2f}")
        
        if clustering > 0.8:
            print(f"\n✓ Strong clustering achieved at step {step}!")
            break
        
        if center_distance_kpc < 0.01:  # Less than 0.01 kpc apart
            print(f"\n✓ Particles very close together!")
            break
        
        # Safety check for runaway
        if center_distance_kpc > 1000:  # More than 1000 kpc apart
            print(f"\n✗ Particles flying apart - forces too strong")
            break
    
    final_clustering = clustering
    clustering_change = final_clustering - initial_clustering
    
    print(f"\nResult:")
    print(f"  Clustering: {initial_clustering:.3f} → {final_clustering:.3f} (Δ={clustering_change:+.3f})")
    
    if clustering_change > 0.2:
        print(f"✓ Corrected clustering test PASSED")
        return True
    else:
        print(f"✗ Still having issues - may need even smaller forces")
        return False

def test_landauer_force_scales():
    """Test what force scales actually work for clustering"""
    
    print("\nTesting Different Force Scales")
    print("="*30)
    
    # Test different force multipliers
    force_multipliers = [1e-5, 1e-3, 1e-1, 1.0, 1e1, 1e3, 1e5]
    
    for multiplier in force_multipliers:
        print(f"\nTesting force multiplier: {multiplier:.0e}")
        
        # Simple 2-particle test
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([1e18, 0.0, 0.0])  # 1e18 m apart
        vel2 = np.array([0.0, 0.0, 0.0])
        
        # Test 5 steps
        dt = 1e13
        mass = 1e42
        
        for step in range(5):
            # Force on particle 2 toward particle 1
            direction = pos1 - pos2
            distance = np.linalg.norm(direction)
            force_direction = direction / distance
            
            # Base gravitational force
            G = 6.67e-11
            base_force = G * mass * mass / (distance**2)
            force = base_force * multiplier
            
            # Update particle 2
            acceleration = force / mass
            vel2 += force_direction * acceleration * dt
            vel2 *= 0.99  # Damping
            pos2 += vel2 * dt
            
            new_distance = np.linalg.norm(pos1 - pos2)
            
        distance_change = (1e18 - new_distance) / 3.086e19  # kpc
        
        if distance_change > 0:
            print(f"  ✓ Particles moved closer by {distance_change:.3f} kpc")
            if 0.001 < distance_change < 1:  # Reasonable clustering
                print(f"    🎯 GOOD force scale!")
        else:
            print(f"  ✗ Particles moved apart by {abs(distance_change):.3f} kpc")

if __name__ == "__main__":
    success = corrected_clustering_test()
    test_landauer_force_scales()
    
    if success:
        print(f"\n🎯 Found working clustering parameters!")
    else:
        print(f"\n⚠️  Need to find correct force scaling")
