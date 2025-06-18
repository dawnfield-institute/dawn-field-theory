import numpy as np
import matplotlib.pyplot as plt

# Core simulation function
def run_test(model="QBE", mode="standard"):
    grid_size = 100
    timesteps = 200
    pulse_strength = 1000.0
    pulse_radius = 6
    pulse_centers = [(50, 50)]

    if mode == "multi_zone":
        pulse_centers = [(30, 30), (70, 70), (50, 50)]

    pulse_interval = 10 if mode == "high_freq" else 40
    thermal_threshold = 150.0
    pulse_efficiency = 0.90
    harvest_efficiency = 0.80
    graphite_efficiency = 0.98

    # Constants
    lambda_0 = 1.0
    alpha = 0.05
    gamma = 0.3
    eta = 0.2
    thermal_dissipation = 0.99
    pi_draw = 0.5 / 3600

    # Fields
    E = np.zeros((grid_size, grid_size))
    I = np.zeros((grid_size, grid_size))
    M = np.zeros((grid_size, grid_size))
    T = np.zeros((grid_size, grid_size))
    Phi = np.ones((grid_size, grid_size))
    B = np.zeros((grid_size, grid_size))

    harvested_energy, net_energy = [], []

    for t in range(timesteps):
        if t % pulse_interval == 10:
            for cx, cy in pulse_centers:
                for x in range(grid_size):
                    for y in range(grid_size):
                        if np.sqrt((x - cx)**2 + (y - cy)**2) < pulse_radius:
                            if T[x, y] < thermal_threshold:
                                E[x, y] += pulse_strength * pulse_efficiency * graphite_efficiency

        # Chaotic entropy input
        if mode == "chaotic":
            noise = 0.1 * (np.random.rand(grid_size, grid_size) - 0.5)
            I += noise
        else:
            I += 0.01 * (np.random.rand(grid_size, grid_size) - 0.5)

        M += np.abs(E - I) * 0.01
        T += 0.1 * (B**2)
        T *= thermal_dissipation

        if model == "QBE":
            B = lambda_0 * ((E - I) / (1 + alpha * M)) * Phi
        elif model == "RBF":
            R = np.exp(-gamma * M) * np.exp(-eta * T)
            B = lambda_0 * R * ((E - I) / (1 + alpha * M)) * Phi

        total_field_energy = np.sum(B**2)
        harvested = total_field_energy * harvest_efficiency
        net = harvested - (pi_draw * grid_size * grid_size)

        harvested_energy.append(harvested)
        net_energy.append(net)

    return harvested_energy, net_energy, B, T


# Run and plot all 6 combinations
tests = [("QBE", "high_freq"), ("RBF", "high_freq"),
         ("QBE", "chaotic"), ("RBF", "chaotic"),
         ("QBE", "multi_zone"), ("RBF", "multi_zone")]

fig, axs = plt.subplots(6, 3, figsize=(18, 30))

for i, (model, mode) in enumerate(tests):
    harvested, net, B, T = run_test(model, mode)
    axs[i, 0].plot(harvested, label="Harvested")
    axs[i, 0].plot(net, label="Net")
    axs[i, 0].set_title(f"{model} - {mode} - Energy")
    axs[i, 0].legend()
    axs[i, 0].grid(True)

    im1 = axs[i, 1].imshow(B, cmap='inferno', origin='lower')
    axs[i, 1].set_title(f"{model} - {mode} - Collapse")
    plt.colorbar(im1, ax=axs[i, 1], label="Field")

    im2 = axs[i, 2].imshow(T, cmap='hot', origin='lower')
    axs[i, 2].set_title(f"{model} - {mode} - Thermal")
    plt.colorbar(im2, ax=axs[i, 2], label="Temp")

plt.tight_layout()
plt.show()
