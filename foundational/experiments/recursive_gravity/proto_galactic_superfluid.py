
import numpy as np
import matplotlib.pyplot as plt

# Parameters
field_res = 300
field_extent = 20
dt = 0.05
timesteps = 250
memory_decay = 0.98
num_clusters = 5
cluster_particles = 400

# Fields
density_field = np.zeros((field_res, field_res))
entropy_field = np.zeros_like(density_field)

# Initialize clusters
positions = []
for _ in range(num_clusters):
    center = np.random.uniform(-field_extent / 2, field_extent / 2, 2)
    spread = np.random.uniform(1.0, 3.0)
    cluster = np.random.normal(center, spread, (cluster_particles, 2))
    positions.append(cluster)
positions = np.vstack(positions)

# Convert position array to field index
def to_field_indices(pos_array):
    scale = field_res / (2 * field_extent)
    x_idx = ((pos_array[:, 0] + field_extent) * scale).astype(int)
    y_idx = ((pos_array[:, 1] + field_extent) * scale).astype(int)
    x_idx = np.clip(x_idx, 0, field_res - 1)
    y_idx = np.clip(y_idx, 0, field_res - 1)
    return x_idx, y_idx

# Run simulation
for _ in range(timesteps):
    density_field.fill(0)
    x_idx, y_idx = to_field_indices(positions)
    np.add.at(density_field, (y_idx, x_idx), 1)

    grad_y, grad_x = np.gradient(density_field)
    motion_vectors = np.stack([-grad_x, -grad_y], axis=-1)

    move_vectors = motion_vectors[y_idx, x_idx] * dt
    positions += move_vectors

    entropy_field *= memory_decay
    x_idx_new, y_idx_new = to_field_indices(positions)
    np.add.at(entropy_field, (y_idx_new, x_idx_new), 0.5)

# Plot result
fig, ax = plt.subplots(figsize=(12, 12))
extent = [-field_extent, field_extent, -field_extent, field_extent]
ax.imshow(entropy_field, extent=extent, origin='lower', cmap='inferno', alpha=0.9)
ax.set_facecolor("black")
ax.set_title("Scaled Proto-Galactic Superfluid Simulation")
plt.axis('off')
plt.show()
