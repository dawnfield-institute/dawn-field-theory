"""
FIXED: Correct Force Direction

The bug was in force direction calculation!
direction = center - position gives WRONG direction
Should be: direction = center - position, force = direction * magnitude
This gives attraction toward center.
"""

import numpy as np

def finally_fixed_clustering_test():
    """Test with CORRECT force directions"""
    
    print("FIXED: Correct Force Direction Test")
    print("="*35)
    
    # 3 particles
    positions = np.array([
        [-1e17, 0.0, 0.0],  # 0.1 kpc apart
        [0.0, 0.0, 0.0], 
        [1e17, 0.0, 0.0]
    ], dtype=np.float64)
    
    velocities = np.zeros_like(positions)
    
    print("Initial positions (x-axis, kpc):", positions[:, 0] / 3.086e19)
    
    def compute_clustering(pos):
        center = np.mean(pos, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        avg_distance = np.mean(distances)
        initial_distance = 1e17 * 2/3  # Initial avg distance from center
        clustering = 1.0 - min(avg_distance / initial_distance, 1.0)
        return max(clustering, 0.0)
    
    initial_clustering = compute_clustering(positions)
    print(f"Initial clustering: {initial_clustering:.3f}")
    print()
    
    # Conservative parameters
    dt = 1e12  # 1 million years
    mass = 1e42
    
    print("Evolution:")
    print("Step  X-positions(kpc)           Clustering  AvgDist(kpc)")
    print("-" * 60)
    
    for step in range(20):
        # Center of mass
        center = np.mean(positions, axis=0)
        
        # Forces toward center - CORRECT calculation
        forces = np.zeros_like(positions)
        for i in range(3):
            # Vector FROM particle TO center
            direction_to_center = center - positions[i]
            distance = np.linalg.norm(direction_to_center)
            
            if distance > 1e10:  # Avoid division by zero
                # Unit vector toward center
                unit_vector = direction_to_center / distance
                
                # Modest attractive force
                force_magnitude = 1e25  # Much smaller than before
                
                # Force = magnitude × direction_toward_center
                forces[i] = unit_vector * force_magnitude
        
        # Update motion
        accelerations = forces / mass
        velocities += accelerations * dt
        velocities *= 0.98  # Damping
        positions += velocities * dt
        
        # Metrics
        clustering = compute_clustering(positions)
        center = np.mean(positions, axis=0)
        avg_distance_from_center = np.mean(np.linalg.norm(positions - center, axis=1))
        
        # Display in kpc
        x_positions_kpc = positions[:, 0] / 3.086e19
        avg_dist_kpc = avg_distance_from_center / 3.086e19
        
        print(f"{step:3d}   [{x_positions_kpc[0]:6.3f}, {x_positions_kpc[1]:6.3f}, {x_positions_kpc[2]:6.3f}]   "
              f"{clustering:8.3f}    {avg_dist_kpc:8.3f}")
        
        if clustering > 0.7:
            print(f"\n✓ Strong clustering achieved!")
            break
        
        if avg_dist_kpc > 10:  # Spreading too far
            print(f"\n✗ Still spreading apart")
            break
    
    final_clustering = clustering
    clustering_change = final_clustering - initial_clustering
    
    print(f"\nResult:")
    print(f"  Clustering: {initial_clustering:.3f} → {final_clustering:.3f} (Δ={clustering_change:+.3f})")
    
    success = clustering_change > 0.1
    print(f"  Success: {'✓' if success else '✗'}")
    
    return success

def verify_force_direction():
    """Double-check force direction calculation"""
    
    print("\nVerifying Force Direction Logic")
    print("="*30)
    
    # Particle at [10, 0, 0], center at [0, 0, 0]
    particle_pos = np.array([10.0, 0.0, 0.0])
    center_pos = np.array([0.0, 0.0, 0.0])
    
    # Direction FROM particle TO center
    direction_to_center = center_pos - particle_pos
    print(f"Particle at: {particle_pos}")
    print(f"Center at: {center_pos}")
    print(f"Direction to center: {direction_to_center}")
    print(f"Unit vector: {direction_to_center / np.linalg.norm(direction_to_center)}")
    
    # Force should be negative x-direction (toward center)
    expected_force_direction = np.array([-1.0, 0.0, 0.0])
    calculated_direction = direction_to_center / np.linalg.norm(direction_to_center)
    
    correct = np.allclose(calculated_direction, expected_force_direction)
    print(f"Force direction correct: {'✓' if correct else '✗'}")
    
    return correct

if __name__ == "__main__":
    direction_ok = verify_force_direction()
    if direction_ok:
        success = finally_fixed_clustering_test()
        if success:
            print(f"\n🎯 CLUSTERING WORKS! Found the bug!")
        else:
            print(f"\n⚠️ Still need force magnitude tuning")
    else:
        print(f"\n⚠️ Force direction logic still wrong")
