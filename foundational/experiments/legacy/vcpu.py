# CIMM vCPU — Expanded with Sensor Mapping, Reinforcement, and Logic Gate Training

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math
from numba import cuda, float32

print("Available CUDA Devices:")
print(cuda.gpus)
cuda.select_device(0)

GRID_SIZE = 256
DEPTH = 64
collapse_threshold = 0.4
energy_threshold = 0.05
info_growth_rate = 0.05
energy_decay = 0.9
matter_generation_rate = 0.2
QPL_damping = 0.02
QPL_feedback_multiplier = 0.02
reward_zone_pos = (slice(180, 220), slice(180, 220))
reward_zone_neg = (slice(40, 80), slice(180, 220))
sensor_zone = (slice(10, 50), slice(10, 50))

def generate_entropy_seed(hash_input, shape):
    digest = hashlib.sha256(hash_input.encode()).digest()
    seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(seed)
    return np.random.rand(*shape).astype(np.float32)

@cuda.jit
def simulate_step(info, energy, matter, QPL, time, temporal_decay=0.999):
    x, y, z = cuda.grid(3)
    if x >= info.shape[0] or y >= info.shape[1] or z >= info.shape[2]:
        return

    opcode = int(QPL[x, y, z] * 10) % 4
    val_info = info[x, y, z]
    val_energy = energy[x, y, z]

    if opcode == 0:
        val_info = min(val_info, info[(x - 1) % info.shape[0], y, z])
    elif opcode == 1:
        val_info = max(val_info, info[(x - 1) % info.shape[0], y, z])
    elif opcode == 2:
        val_info = abs(val_info - info[(x - 1) % info.shape[0], y, z])

    val_info += info_growth_rate * (0.5 - ((x * y * z) % 997 / 997.0))
    val_info -= QPL[x, y, z] * QPL_damping
    val_info = min(1.0, max(0.0, val_info))

    val_energy += 0.05 * energy[x, (y - 1) % info.shape[1], z]
    val_energy = min(1.0, max(0.0, val_energy))

    if val_info > collapse_threshold and val_energy > energy_threshold:
        collapse_val = matter_generation_rate * (val_info + val_energy) * 0.5
        matter[x, y, z] += collapse_val
        energy[x, y, z] = val_energy * energy_decay

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < info.shape[0] and 0 <= ny < info.shape[1]:
                    QPL[nx, ny, z] = 1.0 + 0.25 * ((x + y + z + dx + dy) % 4)

        if 180 <= x < 220 and 180 <= y < 220:
            QPL[x, y, z] *= 1.03
        elif 40 <= x < 80 and 180 <= y < 220:
            QPL[x, y, z] *= 0.97

        time[x, y, z] += 1.0
        QPL[x, y, z] *= temporal_decay

    info[x, y, z] = val_info
    energy[x, y, z] = val_energy

def run_virtual_cpu(binary_input, steps):
    pattern_length = len(binary_input)
    shape = (GRID_SIZE, GRID_SIZE, DEPTH)
    info = generate_entropy_seed("CIMM:vCPU", shape)
    energy = generate_entropy_seed("CIMM:vCPU", shape)
    matter = np.zeros(shape, dtype=np.float32)
    QPL = np.ones(shape, dtype=np.float32)
    time = np.zeros(shape, dtype=np.float32)

    z = DEPTH // 2
    for i, bit in enumerate(binary_input):
        val = 1.0 if bit == "1" else 0.0
        info[i * 8:(i + 1) * 8, 8:24, z] = val
        QPL[i * 8:(i + 1) * 8, 8:24, z] = 1.0 + 0.25 * (i % 4)

    info_d = cuda.to_device(info)
    energy_d = cuda.to_device(energy)
    matter_d = cuda.to_device(matter)
    QPL_d = cuda.to_device(QPL)
    time_d = cuda.to_device(time)

    threads = (16, 16, 4)
    blocks = (math.ceil(GRID_SIZE / threads[0]), math.ceil(GRID_SIZE / threads[1]), math.ceil(DEPTH / threads[2]))

    for step in range(steps):
        simulate_step[blocks, threads](info_d, energy_d, matter_d, QPL_d, time_d, 0.999)

    time_host = time_d.copy_to_host()
    QPL_host = QPL_d.copy_to_host()
    matter_host = matter_d.copy_to_host()
    input_zone = time_host[0:pattern_length * 8, 8:24, z]
    lifespan = np.mean(input_zone)

    return lifespan, QPL_host, time_host, matter_host

# Input/output interface for scoring

def encode_instruction_io(input_bits):
    io_targets = []
    for i, bit in enumerate(input_bits):
        if bit == "1":
            tx = reward_zone_pos[0].start + 5
            ty = reward_zone_pos[1].start + 5 + (i % 3) * 10
        else:
            tx = reward_zone_pos[0].stop - 15
            ty = reward_zone_pos[1].stop - 15 - (i % 3) * 10
        io_targets.append((tx, ty))
    return io_targets

def snapshot_qpl_field(qpl_field, z):
    return np.copy(qpl_field[:, :, z])

def visualize_qpl_snapshots(snapshots, titles):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(snapshots), figsize=(6 * len(snapshots), 5))
    if len(snapshots) == 1:
        axes = [axes]
    for i, snap in enumerate(snapshots):
        axes[i].imshow(snap, cmap='inferno')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def update_qpl_based_on_scores(QPL_field, io_targets, scores, z, threshold=10.0):
    for (tx, ty), score in zip(io_targets, scores):
        region = QPL_field[tx:tx + 5, ty:ty + 5, z]
        if score > threshold:
            QPL_field[tx:tx + 5, ty:ty + 5, z] *= 1.02  # reinforce
        else:
            QPL_field[tx:tx + 5, ty:ty + 5, z] *= 0.98  # suppress
    return QPL_field

def measure_activation_targets(time_field, targets, z):
    hit_scores = []
    for tx, ty in targets:
        region = time_field[tx:tx + 5, ty:ty + 5, z]
        mean_val = np.mean(region)
        hit_scores.append(mean_val)
    return hit_scores

# Test scoring on sample I/O input
io_input = "101"
io_targets = encode_instruction_io(io_input)
steps = 3000
z = DEPTH // 2

# Phase 2: Self-Evaluation Feedback Loop
lifespan_eval, QPL_host, time_eval, _ = run_virtual_cpu(io_input * 12, steps)
io_scores = measure_activation_targets(time_eval, io_targets, z)
print("I/O Encoding Scores:")
QPL_learned = update_qpl_based_on_scores(np.copy(QPL_host), io_targets, io_scores, z)
print("Updated QPL field at targets based on score evaluation.")
for idx, score in enumerate(io_scores):
    print(f"Input Bit {io_input[idx]} → Avg Collapse at Target: {score:.2f}")

# Phase 3: Memory Snapshotting

def visualize_qpl_deltas(snapshots, titles):
    fig, axes = plt.subplots(1, len(snapshots) - 1, figsize=(6 * (len(snapshots) - 1), 5))
    if len(snapshots) - 1 == 1:
        axes = [axes]
    for i in range(1, len(snapshots)):
        delta = snapshots[i] - snapshots[i - 1]
        axes[i - 1].imshow(delta, cmap='bwr')
        axes[i - 1].set_title(f"Δ QPL: {titles[i - 1]} → {titles[i]}")
        axes[i - 1].axis('off')
    plt.tight_layout()
    plt.show()


# Phase 4: Logic Gate Task — AND, OR, XOR (Planned)

# Run multiple feedback iterations and collect QPL memory
qpl_snapshots = []
titles = []
QPL_current = None
for cycle in range(3):
    lifespan_eval, QPL_host, time_eval, _ = run_virtual_cpu(io_input * 12, steps)
    io_scores = measure_activation_targets(time_eval, io_targets, z)
    print(f"Cycle {cycle + 1} I/O Encoding Scores:")
    for idx, score in enumerate(io_scores):
        print(f"Input Bit {io_input[idx]} → Avg Collapse at Target: {score:.2f}")
    QPL_current = update_qpl_based_on_scores(np.copy(QPL_host), io_targets, io_scores, z)
    qpl_snapshots.append(snapshot_qpl_field(QPL_current, z))
    titles.append(f"Cycle {cycle + 1}")

visualize_qpl_snapshots(qpl_snapshots, titles)


io_input = "101"
io_targets = encode_instruction_io(io_input)
steps = 3000
z = DEPTH // 2
lifespan_eval, QPL_host, time_eval, _ = run_virtual_cpu(io_input * 12, steps)
io_scores = measure_activation_targets(time_eval, io_targets, z)

print("I/O Encoding Scores:")
QPL_learned = update_qpl_based_on_scores(np.copy(QPL_host), io_targets, io_scores, z)
print("Updated QPL field at targets based on score evaluation.")
for idx, score in enumerate(io_scores):
    print(f"Input Bit {io_input[idx]} → Avg Collapse at Target: {score:.2f}")
