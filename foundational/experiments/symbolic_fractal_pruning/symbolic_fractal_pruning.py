# filename: symbolic_fractal_pruning.py

"""
Simulates how calculus (e.g. gradient, divergence, Laplacian) can prune and stabilize a fractal symbolic field.
Enhanced with entropy metrics, symbol tracking, and visual overlays for scientific diagnostics.
Includes line chart visualizing entropy change over recursion steps.
Adds active symbol percentage and symbol persistence lifetime diagnostics.
Now uses fractured symbolic geometry as initialization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace, sobel, binary_dilation
import os
from datetime import datetime
import json

# Parameters
params = {
    "grid_size": 256,
    "recursions": 20,
    "symbols": ['A', 'B', 'C', 'D'],
    "fracture_seeds": 10,
    "fracture_iterations": 80,
    "prune_threshold": 0.15,
    "laplacian_weight": 0.3,
    "gradient_weight": 0.3,
    "smoothing_sigma": 1.5
}

# Symbolic fractal field generator (fractured shards)
import hashlib

def generate_symbolic_field(params):
    # Generate deterministic seed from hash
    seed_hash = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    seed = int(seed_hash[:8], 16) % (2**32)
    np.random.seed(seed)

    grid_size = params['grid_size']
    symbol_field = np.full((grid_size, grid_size), '', dtype=object)
    seed_coords = [(np.random.randint(0, grid_size), np.random.randint(0, grid_size)) for _ in range(params['fracture_seeds'])]

    masks = []
    for i, (x, y) in enumerate(seed_coords):
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        mask[x, y] = True
        symbol = params['symbols'][i % len(params['symbols'])]

        for _ in range(params['fracture_iterations']):
            mask = binary_dilation(mask)
            noise_mask = np.random.rand(grid_size, grid_size) > 0.5
            mask = mask & noise_mask

        masks.append((mask, symbol))

    for mask, symbol in masks:
        symbol_field[mask] = symbol

    field = np.random.rand(grid_size, grid_size)
    return field, symbol_field

# Calculate entropy-like measure of symbol diversity in neighborhood
def compute_symbolic_entropy(symbol_field):
    from collections import Counter
    from math import log2
    size = symbol_field.shape[0]
    entropy_map = np.zeros_like(symbol_field, dtype=float)
    entropy_score = 0
    for i in range(1, size-1):
        for j in range(1, size-1):
            neighborhood = symbol_field[i-1:i+2, j-1:j+2].flatten()
            counts = Counter([s for s in neighborhood if s != ''])
            total = sum(counts.values())
            if total > 0:
                shannon = -sum((v/total) * log2(v/total) for v in counts.values())
                entropy_map[i, j] = shannon
                entropy_score += shannon
    return entropy_map, entropy_score

# Apply calculus-based pruning over recursions
def prune_symbolic_field(field, symbol_field, params):
    history = []
    stats = []
    lifetimes = np.zeros_like(symbol_field, dtype=int)
    resistance_field = np.random.rand(*symbol_field.shape)

    for step in range(params['recursions']):
        lap_weight = params['laplacian_weight'] * (1 + step / params['recursions'])
        grad_weight = params['gradient_weight'] * (1 + step / params['recursions'])
        dynamic_threshold = params['prune_threshold'] * np.exp(-step / 10)

        lap = laplace(field)
        grad_x = sobel(field, axis=0)
        grad_y = sobel(field, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        entropy_map, _ = compute_symbolic_entropy(symbol_field)
        resistance_mask = resistance_field > 0.5  # softened
        entropy_threshold = np.percentile(entropy_map, 25)
        entropy_resistance = np.random.rand(*entropy_map.shape) < (entropy_map < entropy_threshold) * 0.5
        resistive_cells = resistance_mask | entropy_resistance
        resistance_field *= 0.98  # decay resistance slightly

        from scipy.spatial.distance import cdist

        # RBF decay from symbolic center of mass
        coords = np.indices(symbol_field.shape).reshape(2, -1).T
        flat_entropy = entropy_map.flatten().reshape(-1, 1)
        if np.sum(flat_entropy) > 0:
            center = np.average(coords, axis=0, weights=flat_entropy.flatten())
        else:
            center = np.array(symbol_field.shape) // 2
        dists = np.linalg.norm(coords - center, axis=1)
        rbf_weights = np.exp(- (dists ** 2) / (2 * (params['grid_size'] * 0.3) ** 2))
        rbf_map = rbf_weights.reshape(symbol_field.shape)

        balance_map = entropy_map - np.mean(entropy_map)
        max_entropy = np.max(entropy_map)
        if max_entropy == 0:
            max_entropy = 1e-6
        entropy_penalty = 1 + 0.3 * (entropy_map / max_entropy) * rbf_map * np.exp(-balance_map * 0.05)
        adjusted_threshold = dynamic_threshold / entropy_penalty
        mask = ((np.abs(lap) * lap_weight + gradient_magnitude * grad_weight) < adjusted_threshold) & (~resistive_cells)

        field[mask] = gaussian_filter(field, sigma=params['smoothing_sigma'])[mask]
                # Simulate symbol drift: small chance to shift each symbol into a neighbor cell
        drift_mask = (symbol_field != '') & (np.random.rand(*symbol_field.shape) < 0.02)
        shift_x = np.random.choice([-1, 0, 1], size=drift_mask.shape)
        shift_y = np.random.choice([-1, 0, 1], size=drift_mask.shape)
        coords = np.indices(symbol_field.shape)
        target_x = np.clip(coords[0] + shift_x, 0, symbol_field.shape[0]-1)
        target_y = np.clip(coords[1] + shift_y, 0, symbol_field.shape[1]-1)
        symbol_field[target_x, target_y] = np.where(drift_mask, symbol_field, '')

        symbol_field[mask] = ''

        history.append(symbol_field.copy())

        symbol_counts = {s: int(np.sum(symbol_field == s)) for s in params['symbols']}
        _, entropy_score = compute_symbolic_entropy(symbol_field)
        active_cells = int(np.sum(symbol_field != ''))
        total_cells = symbol_field.size
        active_ratio = active_cells / total_cells

        lifetimes += (symbol_field != '').astype(int)

        stats.append({
            "step": int(step+1),
            "symbol_counts": symbol_counts,
            "entropy_score": float(entropy_score),
            "active_ratio": float(active_ratio),
            "laplacian_weight": float(lap_weight),
            "gradient_weight": float(grad_weight),
            "prune_threshold": float(dynamic_threshold)
        })

    return history, stats, lifetimes

# Visualize symbolic field with optional entropy overlay
def plot_symbol_field(symbol_field, entropy_map, title, save_path):
    color_map = { 'A': 'red', 'B': 'green', 'C': 'blue', 'D': 'purple', '': 'white' }
    rgb_image = np.zeros(symbol_field.shape + (3,), dtype=np.float32)
    for sym, color in color_map.items():
        mask = (symbol_field == sym)
        if color == 'red': rgb_image[mask] = [1, 0, 0]
        elif color == 'green': rgb_image[mask] = [0, 1, 0]
        elif color == 'blue': rgb_image[mask] = [0, 0, 1]
        elif color == 'purple': rgb_image[mask] = [0.5, 0, 0.5]
        elif color == 'white': rgb_image[mask] = [1, 1, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    if entropy_map is not None:
        plt.imshow(entropy_map, cmap='hot', alpha=0.3)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot entropy change over time
def plot_entropy_over_time(stats, save_path):
    entropy_vals = [s['entropy_score'] for s in stats]
    steps = [s['step'] for s in stats]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, entropy_vals, marker='o')
    plt.title("Entropy Score Over Recursions")
    plt.xlabel("Recursion Step")
    plt.ylabel("Entropy Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot active symbol ratio over time
def plot_active_ratio(stats, save_path):
    active_vals = [s['active_ratio'] for s in stats]
    steps = [s['step'] for s in stats]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, active_vals, marker='o', color='orange')
    plt.title("Active Symbol Ratio Over Recursions")
    plt.xlabel("Recursion Step")
    plt.ylabel("Active Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot symbol lifetimes
def plot_lifetimes(lifetime_matrix, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(lifetime_matrix, cmap='viridis')
    plt.title("Symbol Persistence Lifetime")
    plt.colorbar(label='Steps Alive')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Run experiment
base_dir = f"reference_material/pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(base_dir, exist_ok=True)

field, symbol_field = generate_symbolic_field(params)
history, stats, lifetimes = prune_symbolic_field(field, symbol_field, params)

for i, sym_field in enumerate(history):
    entropy_map, _ = compute_symbolic_entropy(sym_field)
    plot_symbol_field(sym_field, entropy_map, f"Step {i+1}", f"{base_dir}/step_{i+1:02d}.png")

plot_entropy_over_time(stats, f"{base_dir}/entropy_over_time.png")
plot_active_ratio(stats, f"{base_dir}/active_ratio_over_time.png")
plot_lifetimes(lifetimes, f"{base_dir}/symbol_lifetimes.png")

with open(f"{base_dir}/pruning_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"Pruning simulation complete. Output saved to: {base_dir}")
