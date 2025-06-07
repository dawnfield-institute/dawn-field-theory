# Emergent Orbit via Informational Tangle (No Gravity) + Entropy Field Overlay + Recursive Memory Kernel

import numpy as np
import matplotlib.pyplot as plt

# Parameters
timesteps = 300
dt = 0.1
mass_ratio = 10.0  # Big to small mass ratio
recursion_mass_big = mass_ratio
recursion_mass_small = 1.0
field_res = 200
field_extent = 10
entropy_field = np.zeros((field_res, field_res))
memory_decay = 0.98  # Recursive memory decay rate

# Helper to convert position to field index
def to_field_idx(pos):
    scale = field_res / (2 * field_extent)
    x_idx = int((pos[0] + field_extent) * scale)
    y_idx = int((pos[1] + field_extent) * scale)
    return np.clip(x_idx, 0, field_res - 1), np.clip(y_idx, 0, field_res - 1)

# Initial positions and velocities
big_pos = np.array([-1.0, 0.0])
small_pos = np.array([1.0, 0.0])
big_vel = np.array([0.0, 0.1])
small_vel = np.array([0.0, -1.0])

# Traces
big_trace = [big_pos.copy()]
small_trace = [small_pos.copy()]

# Simulation loop
for _ in range(timesteps):
    delta = small_pos - big_pos
    dist = np.linalg.norm(delta)
    direction = delta / (dist + 1e-9)
    tangle_strength = np.exp(-dist)

    small_feedback = -tangle_strength * recursion_mass_big * direction
    big_feedback = tangle_strength * recursion_mass_small * direction

    small_vel += small_feedback * dt
    big_vel += big_feedback * dt
    small_pos += small_vel * dt
    big_pos += big_vel * dt

    small_trace.append(small_pos.copy())
    big_trace.append(big_pos.copy())

    # Recursive memory decay
    entropy_field *= memory_decay

    # Update entropy field
    for pos in [big_pos, small_pos]:
        x_idx, y_idx = to_field_idx(pos)
        entropy_field[y_idx, x_idx] += tangle_strength * 0.5

# Convert to arrays
big_trace = np.array(big_trace)
small_trace = np.array(small_trace)

# Plot results with entropy overlay
fig, ax = plt.subplots(figsize=(10, 10))
extent = [-field_extent, field_extent, -field_extent, field_extent]
ax.imshow(entropy_field, extent=extent, origin='lower', cmap='inferno', alpha=0.7)
ax.plot(big_trace[:, 0], big_trace[:, 1], color='blue', label="Big Mass Trace")
ax.plot(small_trace[:, 0], small_trace[:, 1], color='orange', label="Small Mass Trace")
ax.scatter(big_trace[0, 0], big_trace[0, 1], color="cyan", s=50, label="Big Start")
ax.scatter(small_trace[0, 0], small_trace[0, 1], color="red", s=30, label="Small Start")
ax.set_facecolor("black")
ax.set_aspect('equal')
ax.set_title("Emergent Orbit via Informational Tangle (Field + Memory)")
ax.legend()
plt.axis('off')
plt.show()
