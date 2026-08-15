
import numpy as np
import matplotlib.pyplot as plt

def simulate_reflective_containment_2d(
    grid_size=150,
    t_steps=400,
    dt=0.1,
    dx=1.0,
    c=1.0,
    damping=0.01,
    nullifier_gain=3.5,
    containment_radius=25,
    initial_blast_amplitude=50.0,
    noise_level=0.01
):
    """
    Simulates a high-yield 2D collapse event with recursive containment and a reflective boundary.
    """
    field = np.zeros((t_steps, grid_size, grid_size))
    nullifier = np.zeros_like(field)
    cumulative_energy = np.zeros((grid_size, grid_size))

    center = grid_size // 2
    field[1, center, center] = initial_blast_amplitude

    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    dist_matrix = np.sqrt((xx - center)**2 + (yy - center)**2)
    containment_mask = dist_matrix <= containment_radius
    outer_damp_mask = dist_matrix > (containment_radius + 8)

    for t in range(2, t_steps):
        laplacian = (
            np.roll(field[t - 1], 1, axis=0) + np.roll(field[t - 1], -1, axis=0) +
            np.roll(field[t - 1], 1, axis=1) + np.roll(field[t - 1], -1, axis=1) -
            4 * field[t - 1]
        )
        wave = (
            2 * field[t - 1] - field[t - 2] +
            (c * dt / dx)**2 * laplacian -
            damping * field[t - 1]
        )
        n_energy = np.zeros_like(field[t])
        n_energy[containment_mask] = -nullifier_gain * field[t - 1][containment_mask]
        n_energy[outer_damp_mask] = -0.5 * field[t - 1][outer_damp_mask]
        nullifier[t] = n_energy
        new_field = wave + n_energy + np.random.normal(0, noise_level, size=(grid_size, grid_size))

        # Reflective boundary
        new_field[0, :] = new_field[1, :]
        new_field[-1, :] = new_field[-2, :]
        new_field[:, 0] = new_field[:, 1]
        new_field[:, -1] = new_field[:, -2]

        field[t] = new_field
        cumulative_energy += new_field**2

    plt.figure(figsize=(8, 8))
    plt.imshow(cumulative_energy, cmap='hot', origin='lower')
    plt.colorbar(label='Cumulative Energy Density')
    plt.title('Reflective High-Yield Blast Containment (2D)')
    plt.tight_layout()
    plt.show()
