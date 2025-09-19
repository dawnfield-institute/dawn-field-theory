"""
FINAL DIAGNOSTIC: Root Cause Analysis

The QBE system is working, forces are correct direction, but NO clustering.
This must be either:
1. Clustering metric is broken
2. Forces still wrong scale
3. Integration timestep issues
4. Boundary conditions interfering

Let's isolate each component.
"""

import numpy as np

KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.154e13
SOLAR_MASS = 1.989e30

def test_clustering_metric_directly():
    """Test clustering metric with known configurations"""
    
    print("Testing Clustering Metric Directly")
    print("="*35)
    
    box_size = 2 * KPC_TO_METERS
    
    def compute_clustering(positions, box_size):
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        avg_distance = np.mean(distances)
        initial_expected = box_size / 4
        clustering = 1.0 - min(avg_distance / initial_expected, 1.0)
        return max(clustering, 0.0)
    
    # Test cases
    test_cases = {
        "random": np.random.uniform(-box_size/2, box_size/2, (10, 3)),
        "center_cluster": np.random.normal(0, box_size/20, (10, 3)),
        "tight_cluster": np.random.normal(0, box_size/100, (10, 3)),
        "at_center": np.zeros((10, 3))  # All particles at exact center
    }
    
    for name, positions in test_cases.items():
        clustering = compute_clustering(positions, box_size)
        center = np.mean(positions, axis=0)
        avg_dist = np.mean(np.linalg.norm(positions - center, axis=1))
        avg_dist_kpc = avg_dist / KPC_TO_METERS
        
        print(f"{name:15s}: clustering={clustering:.3f}, avg_dist={avg_dist_kpc:.4f} kpc")
    
    print()

def test_force_effect_directly():
    """Test if forces actually move particles"""
    
    print("Testing Force Effect Directly")
    print("="*30)
    
    # Start with 3 particles in a line
    positions = np.array([
        [-5e17, 0.0, 0.0],  # 0.5 kpc separation
        [0.0, 0.0, 0.0],
        [5e17, 0.0, 0.0]
    ], dtype=np.float64)
    
    velocities = np.zeros_like(positions)
    box_size = 2 * KPC_TO_METERS
    
    print("Initial positions (kpc):", positions[:, 0] / KPC_TO_METERS)
    
    # Apply forces for several steps
    dt = 1e12  # 1 Myr
    mass = SOLAR_MASS * 1e8
    
    for step in range(10):
        # Center of mass
        center = np.mean(positions, axis=0)
        
        # Forces toward center
        forces = np.zeros_like(positions)
        for i in range(3):
            direction_to_center = center - positions[i]
            distance = np.linalg.norm(direction_to_center)
            
            if distance > 1e10:
                unit_vector = direction_to_center / distance
                force_magnitude = 1e30  # Very strong force
                forces[i] = unit_vector * force_magnitude
        
        # Update motion
        accelerations = forces / mass
        velocities += accelerations * dt
        velocities *= 0.9  # Strong damping
        positions += velocities * dt
        
        positions_kpc = positions[:, 0] / KPC_TO_METERS
        spread = np.max(positions_kpc) - np.min(positions_kpc)
        
        print(f"Step {step}: positions={positions_kpc}, spread={spread:.4f} kpc")
        
        if spread < 0.01:
            print("✓ Particles clustered!")
            break
        elif spread > 2:
            print("✗ Particles spreading!")
            break
    
    print()

def test_boundary_conditions():
    """Test if periodic boundaries are interfering"""
    
    print("Testing Boundary Conditions")
    print("="*27)
    
    box_size = 2 * KPC_TO_METERS
    
    # Particle near boundary
    position = np.array([0.9 * box_size/2, 0, 0])  # Near +x boundary
    center = np.array([0, 0, 0])
    
    print(f"Particle at: {position[0]/KPC_TO_METERS:.3f} kpc")
    print(f"Box extends: ±{box_size/2/KPC_TO_METERS:.3f} kpc")
    
    # Force toward center
    direction = center - position
    distance = np.linalg.norm(direction)
    unit_vector = direction / distance
    
    print(f"Force direction: {unit_vector}")
    print(f"Should point toward negative x: {unit_vector[0] < 0}")
    
    # Simulate movement
    velocity = np.array([0, 0, 0])
    dt = 1e12
    mass = SOLAR_MASS * 1e8
    force_magnitude = 1e30
    
    for step in range(5):
        acceleration = unit_vector * force_magnitude / mass
        velocity += acceleration * dt
        velocity *= 0.9
        position += velocity * dt
        
        # Apply periodic boundary
        for axis in range(3):
            if position[axis] < -box_size/2:
                position[axis] += box_size
                print(f"  Applied +x boundary wrap")
            elif position[axis] > box_size/2:
                position[axis] -= box_size
                print(f"  Applied -x boundary wrap")
        
        print(f"Step {step}: position={position[0]/KPC_TO_METERS:.3f} kpc")
        
        # Recalculate for next step
        direction = center - position
        distance = np.linalg.norm(direction)
        if distance > 0:
            unit_vector = direction / distance
    
    print()

def test_clustering_computation_bug():
    """Test if there's a bug in the actual clustering computation"""
    
    print("Testing Clustering Computation for Bugs")
    print("="*40)
    
    box_size = 2 * KPC_TO_METERS
    
    # Create obvious clustering scenario
    positions = np.array([
        [0.0, 0.0, 0.0],           # At center
        [1e15, 0.0, 0.0],          # Very close to center (0.001 kpc)
        [2e15, 0.0, 0.0],          # Very close to center
        [5e17, 0.0, 0.0],          # Far from center (0.5 kpc)
        [5e17, 0.0, 0.0]           # Far from center
    ], dtype=np.float64)
    
    print("Positions (kpc):")
    for i, pos in enumerate(positions):
        print(f"  Particle {i}: {pos[0]/KPC_TO_METERS:.3f}")
    
    # Step by step calculation
    center = np.mean(positions, axis=0)
    print(f"\nCenter of mass: {center[0]/KPC_TO_METERS:.3f} kpc")
    
    distances = np.linalg.norm(positions - center, axis=1)
    print(f"Distances from center (kpc): {distances/KPC_TO_METERS}")
    
    avg_distance = np.mean(distances)
    print(f"Average distance: {avg_distance/KPC_TO_METERS:.3f} kpc")
    
    initial_expected = box_size / 4
    print(f"Expected random distance: {initial_expected/KPC_TO_METERS:.3f} kpc")
    
    ratio = avg_distance / initial_expected
    print(f"Ratio: {ratio:.3f}")
    
    clustering = 1.0 - min(ratio, 1.0)
    clustering = max(clustering, 0.0)
    print(f"Final clustering: {clustering:.3f}")
    
    print(f"Should be > 0 for this configuration: {'✓' if clustering > 0 else '✗'}")

if __name__ == "__main__":
    test_clustering_metric_directly()
    test_force_effect_directly()
    test_boundary_conditions()
    test_clustering_computation_bug()
