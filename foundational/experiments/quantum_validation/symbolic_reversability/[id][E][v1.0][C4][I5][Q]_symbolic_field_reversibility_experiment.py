import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime


# Simulation parameters
steps = 150
phase_lengths = [50, 50, 50]  # forward, reverse, forward again
field_size = 100
activation_value = 1.0
decay_rates = [0.0, 0.01, 0.02, 0.05]  # Sweep over these decay rates
input_pattern = np.sin(np.linspace(0, 2 * np.pi, field_size))

def apply_input(field, pattern, reverse=False):
    if reverse:
        return field - pattern * activation_value
    else:
        return field + pattern * activation_value

def decay(field, rate):
    return field * (1 - rate)

for decay_rate in decay_rates:
    field = np.zeros(field_size)
    entropy_trace = []
    energy_trace = []
    snapshots = {}
    for t in range(steps):
        phase = t // sum(phase_lengths[:2]) if t >= sum(phase_lengths[:2]) else (t // phase_lengths[0])
        reverse = (phase == 1)
        field = decay(field, decay_rate)
        field = apply_input(field, input_pattern, reverse=reverse)
        entropy = np.sum(-np.abs(field) * np.log1p(np.abs(field)))
        energy = np.sum(field ** 2)
        entropy_trace.append(entropy)
        energy_trace.append(energy)
        if t in [0, phase_lengths[0]-1, sum(phase_lengths[:2])-1, steps-1]:
            snapshots[f"step{t}"] = field.copy()

    initial_state = snapshots["step0"]
    final_state = snapshots[f"step{steps-1}"]
    fidelity = 1.0 - np.linalg.norm(initial_state - final_state) / (np.linalg.norm(initial_state) + 1e-12)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/reversibility_test_{timestamp}_decay{decay_rate}"
    os.makedirs(result_dir, exist_ok=True)

    # Save entropy trace plot
    plt.figure()
    plt.plot(entropy_trace, label="Entropy")
    plt.plot(energy_trace, label="Field Energy")
    plt.title(f"Entropy & Energy Trace (decay={decay_rate})")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "entropy_energy_trace.png"))
    plt.close()

    # Save summary
    summary = {
        "fidelity": fidelity,
        "entropy_trace": entropy_trace,
        "energy_trace": energy_trace,
        "steps": steps,
        "phase_lengths": phase_lengths,
        "decay_rate": decay_rate
    }
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
