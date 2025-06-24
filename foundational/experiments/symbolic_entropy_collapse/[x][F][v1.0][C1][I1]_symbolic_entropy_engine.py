# Informational Geometry Collapse — Clean Restart (with Classical Physics Variant + Multi-Run Support + Full Metrics + Higher Dimensional Support + Advanced Visualizations + CIP Extensions)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
import json

GRID_SHAPE = (20, 20, 20)  # 3D support
MAX_STEPS = 100
TARGET_ENTROPY = 0.45
SYMBOLS = ['A', 'B', 'C', 'D']
NOISE_PROB = 0.01
SEEDS = [42, 101, 202, 303, 404]

ALPHA = {
    'A': {'B': 0.5, 'C': 0.2, 'D': 0.3},
    'B': {'A': 0.5, 'C': 0.4, 'D': 0.1},
    'C': {'A': 0.3, 'B': 0.4, 'D': 0.3},
    'D': {'A': 0.2, 'B': 0.1, 'C': 0.7},
}

def initialize_field(seed=None):
    rng = np.random.default_rng(seed)
    return rng.choice(SYMBOLS, size=GRID_SHAPE)

def entropy(field):
    flat = field.flatten()
    counts = Counter(flat)
    total = sum(counts.values())
    probs = np.array([c/total for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-12))

def laplacian_entropy(field, outdir=None, label='', step=0):
    from scipy.ndimage import laplace
    numeric_field = np.vectorize(SYMBOLS.index)(field)
    ent_map = np.zeros_like(numeric_field, dtype=float)
    for idx, _ in np.ndenumerate(numeric_field):
        slices = tuple(slice(max(0, i-1), min(s, i+2)) for i, s in zip(idx, field.shape))
        region = field[slices]
        ent_map[idx] = entropy(region)

    curvature_map = laplace(ent_map)
    if outdir:
        for axis in range(curvature_map.ndim):
            mid = curvature_map.shape[axis] // 2
            plt.imshow(np.take(curvature_map, mid, axis=axis), cmap='viridis')
            plt.title(f"Curvature Slice Axis {axis} at Midpoint")
            plt.colorbar(label='Curvature')
            plt.savefig(f"{outdir}/curvature_slice_axis{axis}_step{step}.png")
            plt.close()

    return curvature_map.mean()

def symbol_distribution(field):
    flat = field.flatten()
    counts = Counter(flat)
    total = len(flat)
    dist = np.array([counts.get(s, 0) / total for s in SYMBOLS])
    return dist

def symbolic_diversity(field):
    probs = symbol_distribution(field)
    return -np.sum(probs * np.log2(probs + 1e-12))

def visualize_symbol_distribution(field, step, label, outdir):
    probs = symbol_distribution(field)
    plt.figure()
    plt.bar(SYMBOLS, probs)
    plt.title(f"Symbol Distribution at Step {step} [{label}]")
    plt.xlabel("Symbol")
    plt.ylabel("Proportion")
    plt.savefig(f"{outdir}/symbol_distribution_step_{step}.png")
    plt.close()
    np.savetxt(f"{outdir}/symbol_distribution_step_{step}.csv", probs, delimiter=",", header=','.join(SYMBOLS), comments='')

def apply_noise(field, rng):
    noisy = field.copy()
    for idx, _ in np.ndenumerate(noisy):
        if rng.random() < NOISE_PROB:
            noisy[idx] = rng.choice(SYMBOLS)
    return noisy

def emergent_step(field, t, rng):
    field = apply_noise(field, rng)
    new_field = field.copy()
    for idx, _ in np.ndenumerate(field):
        slices = tuple(slice(max(0, i-1), min(s, i+2)) for i, s in zip(idx, field.shape))
        window = field[slices]
        min_e = float('inf')
        best = field[idx]
        for s in SYMBOLS:
            temp = window.copy()
            center = tuple(i - max(0, i-1) for i in idx)
            temp[center] = s
            e = entropy(temp)
            if e < min_e:
                min_e = e
                best = s
        new_field[idx] = best
    return new_field

def reaction_potential(field, idx):
    slices = tuple(slice(max(0, i-1), min(s, i+2)) for i, s in zip(idx, field.shape))
    region = field[slices].flatten()
    current = field[idx]
    reactivity = {s: sum(ALPHA[s].get(s2, 0) for s2 in region if s2 != s) for s in SYMBOLS}
    return max(reactivity, key=reactivity.get)

def classical_step_factory():
    def step(field, t, rng):
        field = apply_noise(field, rng)
        new_field = field.copy()
        for idx, _ in np.ndenumerate(field):
            new_field[idx] = reaction_potential(field, idx)
        return new_field
    return step

def simulate(model_fn, seed=42, label='Model', timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f"reference_material/{timestamp}/{label}/seed_{seed}"
    os.makedirs(outdir, exist_ok=True)

    metadata = {
        "grid_shape": GRID_SHAPE,
        "noise_prob": NOISE_PROB,
        "target_entropy": TARGET_ENTROPY,
        "max_steps": MAX_STEPS,
        "timestamp": timestamp,
        "seed": seed,
        "symbols": SYMBOLS
    }
    with open(f"reference_material/{timestamp}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    rng = np.random.default_rng(seed)
    field = initialize_field(seed)
    trace, diversity, curvature = [], [], []
    t = 0
    with open(f"{outdir}/log.txt", "w") as logf:
        while t < MAX_STEPS:
            e = entropy(field)
            d = symbolic_diversity(field)
            c = laplacian_entropy(field, outdir, label, t)
            trace.append(e)
            diversity.append(d)
            curvature.append(c)
            logf.write(f"[{label}][Step {t}] Entropy: {e:.4f} Curvature: {c:.4f}\n")
            print(f"[{label}][Step {t}] Entropy: {e:.4f} Curvature: {c:.4f}")
            visualize_symbol_distribution(field, t, label, outdir)
            if e <= TARGET_ENTROPY:
                logf.write(f"[{label}] Stopping at Step {t} — Reached entropy threshold {e:.4f}\n")
                print(f"[{label}] Stopping at Step {t} — Reached entropy threshold {e:.4f}")
                break
            field = model_fn(field, t, rng)
            t += 1

    np.savetxt(f"{outdir}/entropy_trace.csv", trace, delimiter=",")
    np.savetxt(f"{outdir}/diversity_trace.csv", diversity, delimiter=",")
    np.savetxt(f"{outdir}/curvature_trace.csv", curvature, delimiter=",")
    return trace, diversity, curvature, outdir

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    classical_step = classical_step_factory()

    for seed in SEEDS:
        simulate(emergent_step, seed=seed, label='Emergent', timestamp=timestamp)
        simulate(classical_step, seed=seed, label='Classical', timestamp=timestamp)

    print(f"Experiments complete. All data saved to reference_material/{timestamp}/")
