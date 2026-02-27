# landauer_thermal_decay_demo_v1.py

"""
Simulates symbolic memory erasure in an entropic field to estimate thermodynamic cost
via Landauer principle. Now includes temperature scaling, structured vs random field comparison,
and memory reinforcement options.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

# Parameters
field_shape = (100, 100)
decay_rate = 0.05
steps = 20
initial_noise_amplitude = 5
temperature = 1.0  # Thermal multiplier
structured_mode = False  # Toggle between random or structured memory
memory_reinforcement = False
beta = 0.05  # Reinforcement strength


def symbolic_entropy_field(shape, amplitude=1.0):
    return np.random.randn(*shape) * amplitude


def structured_memory_field(shape):
    field = np.zeros(shape)
    cx, cy = shape[0] // 2, shape[1] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            field[i, j] = np.exp(-dist**2 / 200) * 5.0
    return field


def erase_information(field, decay, memory=None):
    updated = field * (1 - decay)
    if memory is not None:
        updated += beta * memory
    return updated


def compute_energy(field):
    return np.sum(field**2)


def plot_field(field, cost, out_path):
    plt.figure()
    plt.imshow(field, cmap='plasma')
    plt.colorbar(label="Field Intensity")
    plt.title(f"Erasure Result — Energy Cost ≈ {cost:.2f}")
    plt.savefig(out_path)
    plt.close()


def plot_energy_curve(energies, out_path):
    plt.figure()
    plt.plot(range(len(energies)), energies, marker='o')
    plt.title("Energy During Erasure (Landauer Decay)")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def save_energy_csv(energies, out_path):
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Energy"])
        for i, e in enumerate(energies):
            writer.writerow([i, e])


def plot_intermediate_fields(snapshots, out_dir):
    for step, field in snapshots.items():
        plt.figure()
        plt.imshow(field, cmap='plasma')
        plt.title(f"Field at Step {step}")
        plt.colorbar(label="Field Intensity")
        plt.savefig(os.path.join(out_dir, f"field_step_{step}.png"))
        plt.close()


# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"reference_material/{timestamp}"
os.makedirs(out_dir, exist_ok=True)

# Initialize field
if structured_mode:
    initial_field = structured_memory_field(field_shape)
else:
    initial_field = symbolic_entropy_field(field_shape, amplitude=initial_noise_amplitude)

initial_energy = compute_energy(initial_field)
field = initial_field.copy()
energy_over_time = [initial_energy]
snapshots = {0: field.copy()}
memory = field.copy() if memory_reinforcement else None

# Run decay simulation
for step in range(1, steps + 1):
    actual_decay = decay_rate * temperature
    field = erase_information(field, decay=actual_decay, memory=memory)
    energy = compute_energy(field)
    energy_over_time.append(energy)
    if step in [1, 5, 10, 15, steps]:
        snapshots[step] = field.copy()

final_energy = energy_over_time[-1]
energy_cost = initial_energy - final_energy

# Save plot
plot_path = os.path.join(out_dir, "erasure_result_plot.png")
plot_field(field, energy_cost, plot_path)

# Save decay curve and CSV
curve_path = os.path.join(out_dir, "energy_curve.png")
plot_energy_curve(energy_over_time, curve_path)

csv_path = os.path.join(out_dir, "energy_over_time.csv")
save_energy_csv(energy_over_time, csv_path)

# Save intermediate field plots
plot_intermediate_fields(snapshots, out_dir)

# Save log
log_path = os.path.join(out_dir, "erasure_log.txt")
with open(log_path, 'w') as f:
    f.write("Landauer Erasure Field Cost Map Simulation\n")
    f.write(f"Field Shape: {field_shape}\n")
    f.write(f"Initial Noise Amplitude: {initial_noise_amplitude}\n")
    f.write(f"Decay Rate: {decay_rate} (scaled by temperature={temperature})\n")
    f.write(f"Steps: {steps}\n")
    f.write(f"Structured Mode: {structured_mode}\n")
    f.write(f"Memory Reinforcement: {memory_reinforcement} (beta={beta})\n")
    f.write(f"Initial Energy: {initial_energy:.2f}\n")
    f.write(f"Final Energy: {final_energy:.2f}\n")
    f.write(f"Erasure Energy Cost: {energy_cost:.2f}\n")

print(f"Saved erasure results to {out_dir}")