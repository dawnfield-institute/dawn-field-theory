
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Parameters
num_particles = 200
timesteps = 400
dt = 0.02
field_res = 300
field_extent = 15
memory_decay = 0.97
k_neighbors = 10

# Initialize particles
positions = np.random.uniform(-field_extent/2, field_extent/2, (num_particles, 2))
velocities = np.random.uniform(-0.5, 0.5, (num_particles, 2))
masses = np.random.uniform(0.5, 5.0, num_particles)
entropy_field = np.zeros((field_res, field_res))

# Convert position to field index
def to_field_idx(pos):
    scale = field_res / (2 * field_extent)
    x_idx = int((pos[0] + field_extent) * scale)
    y_idx = int((pos[1] + field_extent) * scale)
    return np.clip(x_idx, 0, field_res - 1), np.clip(y_idx, 0, field_res - 1)

# Run simulation
traces = [positions.copy()]
nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')

for _ in range(timesteps):
    new_velocities = velocities.copy()
    nn.fit(positions)
    distances, indices = nn.kneighbors(positions)

    for i in range(num_particles):
        force = np.zeros(2)
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            delta = positions[j] - positions[i]
            if dist > 0:
                tangle_strength = np.exp(-dist) * masses[j]
                direction = delta / (dist + 1e-9)
                force += tangle_strength * direction
        new_velocities[i] += force * dt

    velocities = new_velocities
    positions += velocities * dt
    traces.append(positions.copy())

    entropy_field *= memory_decay
    for pos in positions:
        x_idx, y_idx = to_field_idx(pos)
        entropy_field[y_idx, x_idx] += 0.5

# Plot results
fig, ax = plt.subplots(figsize=(12, 12))
extent = [-field_extent, field_extent, -field_extent, field_extent]
ax.imshow(entropy_field, extent=extent, origin='lower', cmap='inferno', alpha=0.8)
ax.scatter(positions[:, 0], positions[:, 1], s=10, color='white')
ax.set_aspect('equal')
ax.set_facecolor("black")
ax.set_title("Macro Emergence from Informational Tangle (kNN Optimized)")
plt.axis('off')
plt.show()
