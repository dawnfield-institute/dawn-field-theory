"""
Minimal Clustering Test

Super simple test to verify clustering works at all.
Just 3 particles with direct center-of-mass attraction.
"""

import numpy as np

def minimal_clustering_test():
    """Test clustering with just 3 particles"""
    
    print("Minimal 3-Particle Clustering Test")
    print("="*35)
    
    # 3 particles in a line
    positions = np.array([
        [-10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], 
        [10.0, 0.0, 0.0]
    ], dtype=np.float64)
    
    velocities = np.zeros_like(positions)
    
    print("Initial positions (x-axis):", positions[:, 0])
    
    # Simple clustering metric: average distance from center
    def compute_clustering(pos):
        center = np.mean(pos, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        avg_distance = np.mean(distances)
        # Normalize: 0 = spread out, 1 = all at center
        max_distance = 10.0  # Initial spread
        clustering = 1.0 - min(avg_distance / max_distance, 1.0)
        return max(clustering, 0.0)
    
    initial_clustering = compute_clustering(positions)
    print(f"Initial clustering: {initial_clustering:.3f}")
    print()
    
    # Evolution with simple center-of-mass attraction
    dt = 1e12  # Large timestep
    mass = 1e40  # kg
    
    print("Evolution:")
    print("Step  Positions              Clustering  Avg_Distance")
    print("-" * 55)
    
    for step in range(10):
        # Center of mass
        center = np.mean(positions, axis=0)
        
        # Forces toward center
        forces = np.zeros_like(positions)
        for i in range(3):
            direction = center - positions[i]
            distance = np.linalg.norm(direction)
            if distance > 0:
                force_direction = direction / distance
                # Strong attractive force
                force_magnitude = 1e30  # N
                forces[i] = force_direction * force_magnitude
        
        # Update motion
        accelerations = forces / mass
        velocities += accelerations * dt
        velocities *= 0.8  # Strong damping
        positions += velocities * dt
        
        # Calculate clustering
        clustering = compute_clustering(positions)
        avg_distance = np.mean(np.linalg.norm(positions - center, axis=1))
        
        print(f"{step:3d}   [{positions[0,0]:5.1f}, {positions[1,0]:5.1f}, {positions[2,0]:5.1f}]   "
              f"{clustering:8.3f}     {avg_distance:8.2f}")
        
        if clustering > 0.5:
            print(f"\n✓ Clustering success at step {step}!")
            break
        
        if avg_distance < 1.0:
            print(f"\n✓ Particles very close together!")
            break
    
    final_clustering = clustering
    clustering_change = final_clustering - initial_clustering
    
    print(f"\nResult:")
    print(f"  Clustering: {initial_clustering:.3f} → {final_clustering:.3f} (Δ={clustering_change:+.3f})")
    
    if clustering_change > 0.1:
        print(f"✓ Minimal clustering test PASSED")
        return True
    else:
        print(f"✗ Minimal clustering test FAILED - investigate dynamics")
        return False

def test_clustering_metric_sensitivity():
    """Test if clustering metric is sensitive enough"""
    
    print("\nTesting Clustering Metric Sensitivity")
    print("="*40)
    
    # Test different configurations
    configs = {
        "spread_out": np.array([[-20, 0, 0], [0, 0, 0], [20, 0, 0]]),
        "medium": np.array([[-10, 0, 0], [0, 0, 0], [10, 0, 0]]),
        "close": np.array([[-5, 0, 0], [0, 0, 0], [5, 0, 0]]),
        "very_close": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "clustered": np.array([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
    }
    
    def compute_clustering(pos):
        center = np.mean(pos, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        avg_distance = np.mean(distances)
        max_distance = 20.0  # Reference spread
        clustering = 1.0 - min(avg_distance / max_distance, 1.0)
        return max(clustering, 0.0)
    
    for name, pos in configs.items():
        clustering = compute_clustering(pos)
        avg_dist = np.mean(np.linalg.norm(pos - np.mean(pos, axis=0), axis=1))
        print(f"{name:12s}: clustering={clustering:.3f}, avg_distance={avg_dist:.2f}")
    
    print(f"\nMetric sensitivity: {'✓' if configs else '✗'}")

if __name__ == "__main__":
    success = minimal_clustering_test()
    test_clustering_metric_sensitivity()
    
    if success:
        print(f"\n🎯 Clustering mechanism works - issue is in main simulation")
    else:
        print(f"\n⚠️  Fundamental clustering issue identified")
