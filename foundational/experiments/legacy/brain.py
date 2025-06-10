# CIMM GPU-Compatible Cosmological Simulation with Extended Intelligence Load Testing

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math
from numba import cuda, float32
cuda.select_device(0)  # Try 1 or 2 if not default

# --- Parameters ---
GRID_SIZE = 256
DEPTH = 64
collapse_threshold = 0.4
energy_threshold = 0.05
info_growth_rate = 0.05
energy_decay = 0.9
matter_generation_rate = 0.2
QPL_damping = 0.02

# --- SHA-Based Entropy Seeding ---
def generate_entropy_seed(hash_input, shape):
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape).astype(np.float32)

# --- GPU Kernel ---
@cuda.jit
def simulate_step(info, energy, matter, QPL, time):
    x, y, z = cuda.grid(3)
    if x >= info.shape[0] or y >= info.shape[1] or z >= info.shape[2]:
        return

    val_info = info[x, y, z] + info_growth_rate * (0.5 - ((x * y * z) % 997 / 997.0))
    val_info += 0.05 * info[(x - 1) % info.shape[0], y, z]
    val_info -= QPL[x, y, z] * QPL_damping
    val_info = min(1.0, max(0.0, val_info))

    val_energy = energy[x, y, z] + 0.05 * energy[x, (y - 1) % info.shape[1], z]
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

# --- Intelligence Challenge Escalation on GPU ---
def run_gpu_intelligence_test(binary_input, steps):
    pattern_length = len(binary_input)
    shape = (GRID_SIZE, GRID_SIZE, DEPTH)
    info = generate_entropy_seed("CIMM:gpu", shape)
    energy = generate_entropy_seed("CIMM:gpu", shape)
    matter = np.zeros(shape, dtype=np.float32)
    QPL = np.ones(shape, dtype=np.float32)
    time = np.zeros(shape, dtype=np.float32)

    z = DEPTH // 2
    for i, bit in enumerate(binary_input):
        val = 1.0 if bit == "1" else 0.0
        info[i * 10:(i + 1) * 10, 10:20, z] = val

    info_d = cuda.to_device(info)
    energy_d = cuda.to_device(energy)
    matter_d = cuda.to_device(matter)
    QPL_d = cuda.to_device(QPL)
    time_d = cuda.to_device(time)

    threads = (16, 16, 4)
    blocks = (math.ceil(GRID_SIZE / threads[0]), math.ceil(GRID_SIZE / threads[1]), math.ceil(DEPTH / threads[2]))

    for step in range(steps):
        simulate_step[blocks, threads](info_d, energy_d, matter_d, QPL_d, time_d)

    time_host = time_d.copy_to_host()
    input_zone = time_host[0:pattern_length * 10, 10:20, z]
    lifespan = np.mean(input_zone)
    return lifespan

# --- Execute Extended GPU Challenge Test ---
base_input = "1010"
lengths = []
lifespans = []

for i in range(2, 30):
    pattern = base_input * i
    steps = 800 + i * 100
    avg_life = run_gpu_intelligence_test(pattern, steps)
    lengths.append(len(pattern))
    lifespans.append(avg_life)
    if avg_life < 5:
        break

plt.figure(figsize=(12, 6))
plt.plot(lengths, lifespans, marker='o')
plt.title("GPU Extended Test: Avg Collapse Lifespan vs Input Pattern Length")
plt.xlabel("Input Pattern Length (bits)")
plt.ylabel("Average Collapse Lifespan")
plt.grid(True)
plt.tight_layout()
plt.show()
