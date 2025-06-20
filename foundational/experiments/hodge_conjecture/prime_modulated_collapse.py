import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Define prime-modulated constants
primes = [2, 3, 5, 7, 11]
modulations = [p * pi for p in primes]

def run_simulation(n, grid_size=256, steps=100, tau_c=0.65, gamma_energy=0.95, gamma_symbolic=0.96, lambda_reinforce=0.05):
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
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

def compute_density(history, threshold=0.7):
    return sum((frame > threshold).astype(int) for frame in history)

# Run simulations and collect density maps
density_maps = {}
for n in modulations:
    history = run_simulation(n)
    density = compute_density(history)
    density_maps[n] = density

# Visualization
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, (n, density) in zip(axes, density_maps.items()):
    im = ax.imshow(density, cmap='inferno')
    prime = round(n / pi)
    ax.set_title(f"Prime {prime} (n = {round(n, 2)})")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Prime-Modulated Symbolic Collapse (Density Maps)", fontsize=20)
plt.tight_layout()
plt.show()
