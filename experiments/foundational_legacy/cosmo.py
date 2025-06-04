# CIMM Cosmological Simulation with SHA Entropy + Numba GPU Acceleration
# Adds GPU-backed entropy collapse dynamics using Numba for massively parallel execution

import numpy as np
import matplotlib.pyplot as plt
import hashlib
from numba import cuda, float32
import math
import random

# --- Simulation Parameters ---
GRID_SIZE = 100
DEPTH = 50
steps = 1000
collapse_threshold = 0.4
energy_threshold = 0.05
info_growth_rate = 0.05
energy_decay = 0.9
matter_generation_rate = 0.2
gravity_softening = 1e-2
QPL_damping = 0.02
KERNEL_RADIUS = 3

# --- SHA-Based Entropy Seeding ---
def generate_entropy_seed(hash_input, shape):
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape)

# SHA seed initialization
hash_info = "CIMM:cosmic_breath:info"
hash_energy = "CIMM:cosmic_breath:energy"
info_host = generate_entropy_seed(hash_info, (GRID_SIZE, GRID_SIZE, DEPTH)).astype(np.float32)
energy_host = generate_entropy_seed(hash_energy, (GRID_SIZE, GRID_SIZE, DEPTH)).astype(np.float32)
matter_host = np.zeros((GRID_SIZE, GRID_SIZE, DEPTH), dtype=np.float32)
QPL_host = np.ones((GRID_SIZE, GRID_SIZE, DEPTH), dtype=np.float32)
time_host = np.zeros((GRID_SIZE, GRID_SIZE, DEPTH), dtype=np.float32)

# --- GPU Kernels ---
@cuda.jit
def simulate_step(info, energy, matter, QPL, time):
    x, y, z = cuda.grid(3)
    if x >= info.shape[0] or y >= info.shape[1] or z >= info.shape[2]:
        return

    # Pseudo-random update without Python's random module (Numba-compatible workaround)
    val_info = info[x, y, z] + info_growth_rate * (0.5 - (x * y * z % 1000) / 1000.0)
    val_info += 0.05 * info[(x - 1) % GRID_SIZE, y, z]
    val_info -= QPL[x, y, z] * QPL_damping
    val_info = min(1.0, max(0.0, val_info))

    val_energy = energy[x, y, z]
    val_energy += 0.05 * energy[x, (y - 1) % GRID_SIZE, z]
    val_energy = min(1.0, max(0.0, val_energy))

    if val_info > collapse_threshold and val_energy > energy_threshold:
        collapse_val = matter_generation_rate * (val_info + val_energy) * 0.5
        matter[x, y, z] += collapse_val
        energy[x, y, z] = val_energy * energy_decay
        QPL[x, y, z] *= 1.05
        if QPL[x, y, z] > 2.0:
            QPL[x, y, z] = 2.0
        time[x, y, z] += 1.0

    info[x, y, z] = val_info
    energy[x, y, z] = val_energy

# --- Run GPU Simulation ---
info_dev = cuda.to_device(info_host)
energy_dev = cuda.to_device(energy_host)
matter_dev = cuda.to_device(matter_host)
QPL_dev = cuda.to_device(QPL_host)
time_dev = cuda.to_device(time_host)

threadsperblock = (8, 8, 4)
blockspergrid_x = math.ceil(GRID_SIZE / threadsperblock[0])
blockspergrid_y = math.ceil(GRID_SIZE / threadsperblock[1])
blockspergrid_z = math.ceil(DEPTH / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

for step in range(steps):
    simulate_step[blockspergrid, threadsperblock](info_dev, energy_dev, matter_dev, QPL_dev, time_dev)

# --- Copy Back and Visualize ---
matter_map = matter_dev.copy_to_host()
x, y, z = np.nonzero(matter_map)
v = matter_map[x, y, z]
v_norm = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)
colors = plt.cm.inferno(v_norm)
sizes = v_norm * 60

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=colors, s=sizes, marker='o', alpha=0.7)
ax.set_title("GPU-Accelerated 3D Matter Distribution (SHA-seeded)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
