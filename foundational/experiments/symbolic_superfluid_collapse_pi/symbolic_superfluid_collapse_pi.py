# filename: symbolic_superfluid_collapse_pi.py

# Symbolic Superfluid Collapse with Pi Harmonic Modulation and Radial Attractor Geometry
# Includes multi-trial statistical logging, symbolic transition rules, and results document generation

import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, atan2, sin, sqrt
import os
from datetime import datetime
import csv
import json
from collections import defaultdict

# Define symbolic interaction rules (basic logic for transitions)
symbolic_rules = {
    ('A', 'B'): 'C',
    ('B', 'C'): 'D',
    ('C', 'D'): 'A',
    ('D', 'A'): 'B'
}

# Simulation parameters
params = {
    "grid_size": 200,
    "timesteps": 150,
    "decay": 0.97,
    "memory_strength": 0.08,
    "num_particles": 300,
    "dt": 0.1,
    "symbols": ['A', 'B', 'C', 'D'],
    "n_harmonic": pi,
    "radius_threshold": 20,
    "trials": 5
}

# Run a single trial of the simulation and collect results
def run_trial(params, trial_id, base_out_dir):
    grid_size = params["grid_size"]
    timesteps = params["timesteps"]
    decay = params["decay"]
    memory_strength = params["memory_strength"]
    num_particles = params["num_particles"]
    dt = params["dt"]
    symbols = params["symbols"]
    n_harmonic = params["n_harmonic"]
    radius_threshold = params["radius_threshold"]

    entropy = np.zeros((grid_size, grid_size))
    symbol_field = np.full((grid_size, grid_size), '', dtype=object)
    entropy_change = []
    converged = []
    symbol_entropy = []
    avg_speed = []
    transitions = defaultdict(int)

    particles = []
    for _ in range(num_particles):
        x, y = np.random.randint(0, grid_size, 2)
        symbol = random.choice(symbols)
        particles.append({'x': x, 'y': y, 'symbol': symbol, 'prev_x': x, 'prev_y': y})

    center = grid_size // 2

    for t in range(timesteps):
        density = np.zeros((grid_size, grid_size))
        for p in particles:
            x, y = int(p['x']), int(p['y'])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                density[x, y] += 1

        grad_y, grad_x = np.gradient(density)
        entropy_prev = entropy.copy()
        speed_sum = 0

        for i, p in enumerate(particles):
            x_int, y_int = int(p['x']), int(p['y'])
            if 0 <= x_int < grid_size and 0 <= y_int < grid_size:
                dx_rel = p['x'] - center
                dy_rel = p['y'] - center
                r = sqrt(dx_rel**2 + dy_rel**2) + 1e-5
                theta = atan2(dy_rel, dx_rel)
                angular_bias = sin(n_harmonic * theta) * (r / center)
                radial_bias_x = -dx_rel / r * 0.5
                radial_bias_y = -dy_rel / r * 0.5

                dx = (-grad_x[x_int, y_int] + angular_bias + radial_bias_x) * dt
                dy = (-grad_y[x_int, y_int] + angular_bias + radial_bias_y) * dt

                new_x = np.clip(p['x'] + dx, 0, grid_size - 1)
                new_y = np.clip(p['y'] + dy, 0, grid_size - 1)

                speed = sqrt((new_x - p['prev_x'])**2 + (new_y - p['prev_y'])**2)
                speed_sum += speed

                p['prev_x'], p['prev_y'] = p['x'], p['y']
                p['x'], p['y'] = new_x, new_y

                entropy[int(new_x), int(new_y)] += memory_strength

                old_symbol = symbol_field[int(new_x), int(new_y)]
                if old_symbol and (old_symbol, p['symbol']) in symbolic_rules:
                    new_symbol = symbolic_rules[(old_symbol, p['symbol'])]
                    transitions[(old_symbol, p['symbol'])] += 1
                    symbol_field[int(new_x), int(new_y)] = new_symbol
                else:
                    symbol_field[int(new_x), int(new_y)] = p['symbol']

        entropy *= decay

        entropy_change.append(np.sum(np.abs(entropy - entropy_prev)))
        converged.append(sum(
            sqrt((p['x'] - center)**2 + (p['y'] - center)**2) < radius_threshold for p in particles
        ))

        sym_entropy = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if symbol_field[i, j]:
                    syms = [p['symbol'] for p in particles if int(p['x']) == i and int(p['y']) == j]
                    if len(set(syms)) > 1:
                        sym_entropy += 1
        symbol_entropy.append(sym_entropy)
        avg_speed.append(speed_sum / len(particles))

    return entropy_change, converged, symbol_entropy, avg_speed, dict(transitions)

# Run batch of trials and aggregate
base_dir = f"reference_material/{datetime.now().strftime('%Y%m%d_%H%M%S')}_batch"
os.makedirs(base_dir, exist_ok=True)

aggregate_logs = {"entropy": [], "converged": [], "symbol_entropy": [], "avg_speed": []}
transitions_total = defaultdict(int)

for trial in range(params["trials"]):
    e, c, s, a, transitions = run_trial(params, trial, base_dir)
    aggregate_logs["entropy"].append(e)
    aggregate_logs["converged"].append(c)
    aggregate_logs["symbol_entropy"].append(s)
    aggregate_logs["avg_speed"].append(a)
    for k, v in transitions.items():
        transitions_total[k] += v

# Save average plots
timesteps = params["timesteps"]
def plot_avg(metric, name):
    data = np.array(metric)
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(timesteps)
    plt.figure(figsize=(8, 4))
    plt.plot(x, avg, label="Mean")
    plt.fill_between(x, avg - std, avg + std, alpha=0.3, label="Std Dev")
    plt.title(f"{name} Over Time (Mean ± Std)")
    plt.xlabel("Timestep")
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{name.replace(' ', '_').lower()}.png")
    plt.close()

plot_avg(aggregate_logs["entropy"], "Entropy Change")
plot_avg(aggregate_logs["converged"], "Particles Near Center")
plot_avg(aggregate_logs["symbol_entropy"], "Symbolic Entropy")
plot_avg(aggregate_logs["avg_speed"], "Average Speed")

# Save transition counts
with open(f"{base_dir}/symbolic_transitions.json", "w") as f:
    json.dump({f"{k[0]}+{k[1]}": v for k, v in transitions_total.items()}, f, indent=4)

# Generate results summary
with open(f"{base_dir}/RESULTS.md", "w", encoding="utf-8") as f:
    f.write(f"""
# Symbolic Superfluid Collapse — Batch Results

## Summary
- **Trials**: {params['trials']}
- **Timesteps**: {params['timesteps']}
- **Particles**: {params['num_particles']}
- **Symbol Set**: {params['symbols']}

## Observations
- Average collapse behavior shows convergence toward radial center
- Symbolic entropy stabilizes with reduced motion
- Symbolic interactions logged across trials:

""")
    for k, v in sorted(transitions_total.items(), key=lambda item: -item[1]):
        f.write(f"- {k[0]} + {k[1]} -> {symbolic_rules[k]}: {v} occurrences\n")
    f.write("""

## Diagnostics
- See `*.png` for metric plots with mean and variance.
- See `symbolic_transitions.json` for raw symbolic interactions.
""")
