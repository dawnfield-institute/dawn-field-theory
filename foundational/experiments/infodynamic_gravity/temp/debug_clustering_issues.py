"""
Debug QBE-Landauer Clustering Issues

Diagnose why clustering remains at 0.000 despite increasing forces.
Likely issues:
1. Clustering metric calculation
2. Force application method
3. Particle dynamics implementation
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
K_B = 1.380649e-23
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30
PLANCK_LENGTH = 1.616e-35

def debug_clustering_metric():
    """Test clustering metric calculation"""
    
    print("Testing Clustering Metric Calculation")
    print("="*40)
    
    # Test 1: Random distribution (should give ~0)
    np.random.seed(42)
    random_positions = np.random.uniform(-50, 50, (100, 3))
    
    # Test 2: Clustered distribution (should give >0)
    clustered_positions = np.random.normal(0, 10, (100, 3))  # Tighter clustering
    
    # Test 3: Highly clustered (should give close to 1)
    highly_clustered = np.random.normal(0, 1, (100, 3))  # Very tight clustering
    
    def compute_clustering_metric(positions, box_size=100):
        """Simplified clustering metric"""
        distances = []
        for i in range(len(positions)):
            other_positions = np.delete(positions, i, axis=0)
            if len(other_positions) > 0:
                min_dist = np.min(np.linalg.norm(other_positions - positions[i], axis=1))
                distances.append(min_dist)
        
        if not distances:
            return 0.0
            
        avg_distance = np.mean(distances)
        # Expected distance in random distribution
        expected_random_distance = box_size / np.sqrt(len(positions))
        
        # Clustering: 1 = perfectly clustered, 0 = random
        clustering = 1.0 - min(avg_distance / expected_random_distance, 1.0)
        return max(clustering, 0.0)
    
    random_clustering = compute_clustering_metric(random_positions, 100)
    clustered_clustering = compute_clustering_metric(clustered_positions, 100)  
    highly_clustered_clustering = compute_clustering_metric(highly_clustered, 100)
    
    print(f"Random distribution clustering: {random_clustering:.3f}")
    print(f"Clustered distribution clustering: {clustered_clustering:.3f}")
    print(f"Highly clustered clustering: {highly_clustered_clustering:.3f}")
    
    # Check average distances
    random_avg_dist = np.mean([np.min(np.linalg.norm(np.delete(random_positions, i, axis=0) - random_positions[i], axis=1)) for i in range(len(random_positions))])
    clustered_avg_dist = np.mean([np.min(np.linalg.norm(np.delete(clustered_positions, i, axis=0) - clustered_positions[i], axis=1)) for i in range(len(clustered_positions))])
    
    print(f"\nAverage nearest neighbor distances:")
    print(f"Random: {random_avg_dist:.2f}")
    print(f"Clustered: {clustered_avg_dist:.2f}")
    
    return random_clustering, clustered_clustering, highly_clustered_clustering

def debug_force_calculation():
    """Test force calculation and information density"""
    
    print("\nTesting Force Calculation")
    print("="*30)
    
    # Simple test setup
    box_size = 100 * KPC_TO_METERS
    
    # Test 1: Uniform distribution
    uniform_pos = np.random.uniform(-box_size/2, box_size/2, (10, 3))
    
    # Test 2: One cluster
    cluster_pos = np.random.normal(0, box_size/20, (10, 3))
    
    def compute_simple_info_density(positions, box_size):
        """Simplified information density"""
        info_density = np.zeros(len(positions))
        interaction_radius = box_size / 10
        
        for i, pos in enumerate(positions):
            distances = np.linalg.norm(positions - pos, axis=1)
            neighbors = np.sum(distances < interaction_radius) - 1  # Exclude self
            
            if neighbors > 0:
                info_density[i] = np.log(neighbors + 1)  # Add 1 to avoid log(0)
            else:
                info_density[i] = 0
                
        return info_density
    
    uniform_info = compute_simple_info_density(uniform_pos, box_size)
    cluster_info = compute_simple_info_density(cluster_pos, box_size)
    
    print(f"Uniform info density: mean={np.mean(uniform_info):.3f}, std={np.std(uniform_info):.3f}")
    print(f"Clustered info density: mean={np.mean(cluster_info):.3f}, std={np.std(cluster_info):.3f}")
    
    # Test force magnitudes
    kappa = 1e25
    coherence_power = 0.5
    L_coherence = box_size * 0.1  # 10% of box size
    coherence_factor = (L_coherence / PLANCK_LENGTH) ** coherence_power
    landauer_base = K_B * 2.7 * np.log(2)
    force_magnitude = kappa * landauer_base * coherence_factor
    
    print(f"\nForce calculation:")
    print(f"κ = {kappa:.1e}")
    print(f"Coherence factor = {coherence_factor:.1e}")
    print(f"Force magnitude = {force_magnitude:.1e} N")
    
    return force_magnitude

def debug_particle_dynamics():
    """Test simple particle movement"""
    
    print("\nTesting Particle Dynamics")
    print("="*25)
    
    # Simple 2-particle test
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([10, 0, 0])  # 10 units apart
    
    # Simulate attraction force
    direction = pos1 - pos2  # Force on particle 2 toward particle 1
    distance = np.linalg.norm(direction)
    force_direction = direction / distance
    
    force_magnitude = 1e30  # Strong force
    mass = SOLAR_MASS * 1e9
    
    acceleration = force_magnitude / mass
    dt = 100 * MYR_TO_SECONDS
    
    # Initial velocity (at rest)
    velocity = np.array([0, 0, 0])
    
    print(f"Initial separation: {distance:.1f} units")
    print(f"Force magnitude: {force_magnitude:.1e} N")
    print(f"Acceleration: {acceleration:.1e} m/s²")
    
    # Simulate a few steps
    for step in range(5):
        # Update velocity
        velocity += force_direction * acceleration * dt
        
        # Update position
        pos2 += velocity * dt
        
        # Recalculate
        direction = pos1 - pos2
        new_distance = np.linalg.norm(direction)
        force_direction = direction / new_distance if new_distance > 0 else np.array([0, 0, 0])
        
        print(f"Step {step+1}: separation = {new_distance:.1f}, velocity = {np.linalg.norm(velocity):.1e} m/s")
        
        if new_distance > distance * 1.1:  # Moving away instead of together
            print("⚠️  Particles moving apart - force direction issue!")
            break
        elif new_distance < 1:  # Too close
            print("✓ Particles clustered")
            break
        
        distance = new_distance

def simplified_clustering_test():
    """Test minimal clustering system"""
    
    print("\nSimplified Clustering Test")
    print("="*30)
    
    # Very simple setup: 5 particles
    num_particles = 5
    box_size = 100
    
    # Start with spread out particles
    positions = np.array([
        [-40, 0, 0],
        [-20, 0, 0], 
        [0, 0, 0],
        [20, 0, 0],
        [40, 0, 0]
    ], dtype=float)
    
    velocities = np.zeros_like(positions)
    
    print("Initial positions:", positions[:, 0])
    
    # Simple attractive force toward center
    for step in range(10):
        forces = np.zeros_like(positions)
        
        # Each particle feels force toward center of mass
        center = np.mean(positions, axis=0)
        
        for i in range(num_particles):
            direction = center - positions[i]
            distance = np.linalg.norm(direction)
            if distance > 0:
                force_direction = direction / distance
                forces[i] = force_direction * 1e30  # Strong force
        
        # Update velocities and positions
        mass = 1e42  # kg
        dt = 1e14  # seconds
        
        accelerations = forces / mass
        velocities += accelerations * dt
        velocities *= 0.9  # Damping
        positions += velocities * dt
        
        # Calculate clustering
        distances = []
        for i in range(num_particles):
            others = np.delete(positions, i, axis=0)
            min_dist = np.min(np.linalg.norm(others - positions[i], axis=1))
            distances.append(min_dist)
        
        avg_distance = np.mean(distances)
        initial_avg = 20.0  # Expected initial average
        clustering = 1.0 - (avg_distance / initial_avg)
        clustering = max(0, clustering)
        
        print(f"Step {step+1}: avg_dist={avg_distance:.1f}, clustering={clustering:.3f}")
        
        if clustering > 0.5:
            print("✓ Significant clustering achieved!")
            break

if __name__ == "__main__":
    debug_clustering_metric()
    debug_force_calculation()
    debug_particle_dynamics()
    simplified_clustering_test()
