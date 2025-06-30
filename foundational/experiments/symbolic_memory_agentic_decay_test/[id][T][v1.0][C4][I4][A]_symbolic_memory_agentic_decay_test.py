# [id][T][v1.0][C4][I4][A]_symbolic_memory_agentic_decay_test.py

"""
Symbolic Memory Reinforcement + Decay Test (Agentic Dynamics)

This simulation models a symbolic memory field where certain regions experience reinforcement
(simulating agentic attention or learning) while others decay via thermodynamic loss.
It now includes:
  - Directed decay to specific symbolic regions (simulated forgetting)
  - Feedback amplification based on local coherence
  - Field interaction dynamics with a secondary symbolic agent
  - Overlay visualization of reinforcement zones for clarity
  - Optional control mode for decay-only comparison
  - Parameter toggle for exploratory runs
  - Separate overlay highlighting of reinforcement zones
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

# Parameters
field_shape = (100, 100)
decay_rate = 0.05
steps = 50
reinforcement_centers = [(30, 30), (70, 70)]
reinforcement_radius = 8
reinforcement_strength = 0.15
initial_noise_amplitude = 4.0
coherence_threshold = 3.0  # For feedback phase
feedback_boost = 0.2
interaction_amplitude = 1.5  # For secondary field influence
control_mode = False  # Set to True to disable reinforcement


def initialize_field(shape, amplitude):
    return np.random.randn(*shape) * amplitude


def create_reinforcement_map(shape, centers, radius, strength):
    mask = np.zeros(shape)
    for cx, cy in centers:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.sqrt((i - cx)**2 + (j - cy)**2) <= radius:
                    mask[i, j] = strength
    return mask


def generate_interaction_field(shape):
    interaction = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            interaction[i, j] = np.sin(2 * np.pi * i / shape[0]) * np.cos(2 * np.pi * j / shape[1])
    return interaction * interaction_amplitude


def update_field(field, decay_rate, reinforcement_mask, step, interaction_field):
    field *= (1 - decay_rate)

    if not control_mode:
        if step <= 30:
            field += reinforcement_mask * field
        elif step <= 40:
            boost_mask = (np.abs(field) > coherence_threshold).astype(float)
            field += boost_mask * field * feedback_boost

        if 21 <= step <= 30:
            for i in range(field_shape[0]):
                for j in range(field_shape[1]):
                    if np.sqrt((i - reinforcement_centers[0][0])**2 + (j - reinforcement_centers[0][1])**2) <= reinforcement_radius:
                        field[i, j] *= 0.8

    if step >= 41:
        field += interaction_field * 0.05

    return field


def compute_energy(field):
    return np.sum(field**2)


def run_simulation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"reference_material/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    field = initialize_field(field_shape, initial_noise_amplitude)
    reinforcement_mask = create_reinforcement_map(field_shape, reinforcement_centers,
                                                  reinforcement_radius, reinforcement_strength)
    interaction_field = generate_interaction_field(field_shape)

    energy_log = [compute_energy(field)]
    snapshots = {0: field.copy()}

    for step in range(1, steps + 1):
        field = update_field(field, decay_rate, reinforcement_mask, step, interaction_field)
        energy_log.append(compute_energy(field))
        if step in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            snapshots[step] = field.copy()

    for step, snap in snapshots.items():
        plt.figure()
        im = plt.imshow(snap, cmap='plasma')
        if not control_mode:
            contours = plt.contour(reinforcement_mask, levels=[0.01], colors='white', linewidths=1.0)
            plt.clabel(contours, inline=True, fontsize=6, fmt='R')
        plt.title(f"Field Step {step}")
        if np.ptp(snap) > 0:
            plt.colorbar(im)
        plt.savefig(os.path.join(out_dir, f"field_step_{step}.png"))
        plt.close()

    plt.figure()
    plt.plot(energy_log, marker='o')
    plt.title("Symbolic Field Energy Over Time")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "energy_curve.png"))
    plt.close()

    with open(os.path.join(out_dir, "energy_over_time.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Energy"])
        for i, e in enumerate(energy_log):
            writer.writerow([i, e])

    print(f"Simulation complete. Outputs saved to {out_dir}")


if __name__ == '__main__':
    run_simulation()
