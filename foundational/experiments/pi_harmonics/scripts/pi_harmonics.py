# Pi-Harmonic Symbolic Collapse Validation Framework

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from math import sqrt, pi

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

# Function to run field interaction with given harmonic multiplier
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

# Compute attractor lifespan and density
def compute_attractor_lifespan_and_density(history, threshold=0.7):
    lifespan_map = np.zeros_like(history[0])
    density_map = np.zeros_like(history[0])
    active = np.zeros_like(history[0], dtype=bool)
    for field in history:
        active_now = field > threshold
        lifespan_map += active & active_now
        active = active_now
        density_map += active_now.astype(int)
    return lifespan_map, density_map

# Run Pi-harmonic and irrational harmonic cases
history_pi = run_crystallization_sim(n=pi)
history_irr = run_crystallization_sim(n=sqrt(2) * pi)

lifespan_pi, density_pi = compute_attractor_lifespan_and_density(history_pi)
lifespan_irr, density_irr = compute_attractor_lifespan_and_density(history_irr)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for ax, data, title in zip(axs.flat,
                           [lifespan_pi, density_pi, lifespan_irr, density_irr],
                           ["Pi-Harmonic Lifespan", "Pi-Harmonic Density",
                            "Irrational Harmonic Lifespan", "Irrational Harmonic Density"]):
    im = ax.imshow(data, cmap='inferno', extent=(-1, 1, -1, 1))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
