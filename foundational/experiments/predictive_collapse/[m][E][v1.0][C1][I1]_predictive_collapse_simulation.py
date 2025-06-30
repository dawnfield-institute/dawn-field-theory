import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

# Setup output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"reference_material/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Parameters for the predictive actualization simulation
grid_size = 64
generations = 30
threshold_kappa = 0.12
threshold_stability = 0.25

eta = 0.05  # Entropy decay rate
alpha = 0.1  # Symbolic reinforcement rate
beta = 0.05  # Semantic memory rate
theta = 0.25  # Crystallization threshold

# Fields: symbolic potential, entropy, stability momentum, semantic tagging, thermodynamic and causal fields
phi = np.random.normal(loc=0.05, scale=0.4, size=(grid_size, grid_size))
entropy_field = np.abs(np.random.normal(loc=0.3, scale=0.1, size=(grid_size, grid_size)))
stability_momentum = np.zeros((grid_size, grid_size))
symbolic_field = np.zeros((grid_size, grid_size))
semantic_memory = np.zeros((grid_size, grid_size))
thermo_gradient = np.zeros((grid_size, grid_size))
potential_landscape = np.random.normal(loc=0, scale=0.2, size=(grid_size, grid_size))
causal_threading = np.zeros((grid_size, grid_size))

collapse_forecast = []
snapshots = {}

# Simulation loop
for t in range(generations):
    crystallized = phi > theta
    symbolic_field[crystallized] += alpha
    stability_momentum += alpha * crystallized
    semantic_memory += beta * (symbolic_field > 0.2)

    noise = np.random.normal(loc=0, scale=0.08, size=(grid_size, grid_size))
    entropy_field = np.clip(entropy_field - eta * symbolic_field + noise, 0, 1)

    thermo_gradient = np.gradient(entropy_field)[0] ** 2 + np.gradient(entropy_field)[1] ** 2
    causal_threading += 0.02 * (symbolic_field * stability_momentum)

    laplacian_phi = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 4 * phi
    )

    phi += (
        0.2 * laplacian_phi
        - 0.08 * entropy_field
        + 0.07 * symbolic_field
        + 0.03 * semantic_memory
        - 0.04 * thermo_gradient
        + 0.05 * potential_landscape
        + 0.02 * causal_threading
    )

    curvature_xx = np.gradient(np.gradient(entropy_field, axis=0), axis=0)
    curvature_yy = np.gradient(np.gradient(entropy_field, axis=1), axis=1)
    corrected_curvature = curvature_xx + curvature_yy

    forecasted_collapse = (corrected_curvature > threshold_kappa) & (stability_momentum > threshold_stability)
    collapse_forecast.append((t, np.sum(forecasted_collapse)))

    if t in [10, 20, 29]:
        snapshots[t] = {
            "phi": phi.copy(),
            "entropy": entropy_field.copy(),
            "stability": stability_momentum.copy(),
            "curvature": corrected_curvature.copy(),
            "semantic": semantic_memory.copy(),
            "thermo": thermo_gradient.copy(),
            "potential": potential_landscape.copy(),
            "causal": causal_threading.copy()
        }

# Plot forecasted collapse count
times, forecast_counts = zip(*collapse_forecast)
plt.figure(figsize=(6, 4))
plt.plot(times, forecast_counts, marker='o', color='blue')
plt.title("Collapse Forecast Over Time (Extended Run)")
plt.xlabel("Generation")
plt.ylabel("Predicted Collapse Zones")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/forecasted_collapse_plot.png")
plt.close()

# Save forecast to CSV
with open(f"{output_dir}/forecasted_collapse_data.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Generation", "Predicted Collapse Zones"])
    writer.writerows(collapse_forecast)

# Visualize snapshot at final generation
fields = snapshots[29]
fig, axs = plt.subplots(3, 3, figsize=(14, 10))
axs[0, 0].imshow(fields["phi"], cmap='viridis')
axs[0, 0].set_title("Phi Field")
axs[0, 1].imshow(fields["entropy"], cmap='cividis')
axs[0, 1].set_title("Entropy Field")
axs[0, 2].imshow(fields["semantic"], cmap='magma')
axs[0, 2].set_title("Semantic Memory")
axs[1, 0].imshow(fields["stability"], cmap='plasma')
axs[1, 0].set_title("Stability Momentum")
axs[1, 1].imshow(fields["curvature"], cmap='seismic')
axs[1, 1].set_title("Entropy Curvature")
axs[1, 2].imshow(fields["thermo"], cmap='inferno')
axs[1, 2].set_title("Thermo Gradient")
axs[2, 0].imshow(fields["potential"], cmap='coolwarm')
axs[2, 0].set_title("Potential Landscape")
axs[2, 1].imshow(fields["causal"], cmap='spring')
axs[2, 1].set_title("Causal Threading")

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/snapshot_generation_29.png")
plt.close()

# Save simulation log
with open(f"{output_dir}/simulation_log.txt", "w") as f:
    f.write("Simulation Metadata\n")
    f.write(f"Grid Size: {grid_size}\nGenerations: {generations}\n")
    f.write(f"alpha: {alpha}, beta: {beta}, eta: {eta}, theta: {theta}\n")
    f.write(f"threshold_kappa: {threshold_kappa}, threshold_stability: {threshold_stability}\n")
    f.write("\nCollapse Forecast (per generation):\n")
    for t, count in collapse_forecast:
        f.write(f"Generation {t}: {count} collapse zones\n")