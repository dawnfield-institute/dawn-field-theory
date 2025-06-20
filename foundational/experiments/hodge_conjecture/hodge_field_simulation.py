import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from math import sqrt, pi, e

# Simulation parameters
grid_size = 256
steps = 100
tau_c = 0.65
gamma_energy = 0.95
gamma_symbolic = 0.96
lambda_reinforce = 0.05

# Field setup
x = np.linspace(-1.0, 1.0, grid_size)
y = np.linspace(-1.0, 1.0, grid_size)
X, Y = np.meshgrid(x, y)
theta = np.arctan2(Y, X)

def run_crystallization_sim(n):
    angular_bias = np.sin(n * theta)
    np.random.seed(42)
    energy = np.random.normal(0.5, 0.1, size=(grid_size, grid_size))
    symbolic = np.random.normal(0.5, 0.1, size=(grid_size, grid_size))
    energy = np.clip(energy, 0, 1)
    symbolic = np.clip(symbolic, 0, 1)

    symbolic_history = [symbolic.copy()]
    for _ in range(steps):
        symbolic_mod = symbolic * (1 + 0.3 * angular_bias)
        crystallized = ((symbolic_mod + energy) / 2 > tau_c).astype(float)
        energy = gamma_energy * energy + lambda_reinforce * crystallized
        symbolic = gamma_symbolic * symbolic + lambda_reinforce * crystallized
        energy = np.clip(energy, 0, 1)
        symbolic = np.clip(symbolic, 0, 1)
        symbolic_history.append(symbolic.copy())
    return symbolic_history

def compute_density_map(history, threshold=0.7):
    density_map = np.zeros_like(history[0])
    for field in history:
        density_map += (field > threshold).astype(int)
    return density_map

def compute_persistence_map(history, threshold=0.7):
    persistence = np.zeros_like(history[0])
    current_streak = np.zeros_like(history[0])
    for field in history:
        active = (field > threshold).astype(int)
        current_streak = (current_streak + 1) * active
        persistence = np.maximum(persistence, current_streak)
    return persistence

def compute_entropy_map(history, bins=10):
    h, w = history[0].shape
    stack = np.stack(history, axis=-1)
    entropy_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            hist, _ = np.histogram(stack[i, j, :], bins=bins, range=(0, 1), density=True)
            hist = hist[hist > 0]
            entropy_map[i, j] = -np.sum(hist * np.log2(hist))
    return entropy_map

# Run and visualize simulations
n_values = [pi, sqrt(2) * pi, e]
titles = [r"Modulation: $\pi$", r"Modulation: $\sqrt{2}\pi$", r"Modulation: $e$"]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for idx, (n, title) in enumerate(zip(n_values, titles)):
    history = run_crystallization_sim(n=n)
    density_map = compute_density_map(history)
    persistence_map = compute_persistence_map(history)
    entropy_map = compute_entropy_map(history)

    im1 = axes[0, idx].imshow(density_map, cmap='inferno')
    axes[0, idx].set_title(f"Density Map: {title}")
    plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)

    im2 = axes[1, idx].imshow(persistence_map, cmap='viridis')
    axes[1, idx].set_title(f"Persistence Map: {title}")
    plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)

    im3 = axes[2, idx].imshow(entropy_map, cmap='magma')
    axes[2, idx].set_title(f"Entropy Map: {title}")
    plt.colorbar(im3, ax=axes[2, idx], fraction=0.046, pad=0.04)

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.suptitle("Symbolic Field Analysis Across Modulation Constants", fontsize=18, y=1.02)
plt.show()