import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
grid_size = 100
timesteps = 200
pulse_count = 10
pulse_interval = 20
pulse_duration = 5
lambda_0 = 1.0
alpha = 0.05
thermal_dissipation = 0.99
pi_draw = 0.5 / 3600
pulse_strength = 1000.0
pulse_radius = 6
pulse_center = (grid_size // 2, grid_size // 2)
reduced_pulse_strength = pulse_strength / pulse_count

# Material properties
materials_tuned = {
    "Graphite": {"efficiency": 0.65, "conversion": 0.75, "harvest": 0.40, "thermal_threshold": 75.0, "gamma": 0.1, "eta": 0.05},
    "Graphene": {"efficiency": 0.95, "conversion": 0.90, "harvest": 0.80, "thermal_threshold": 150.0, "gamma": 0.2, "eta": 0.1},
    "CNT": {"efficiency": 0.99, "conversion": 0.95, "harvest": 0.90, "thermal_threshold": 180.0, "gamma": 0.05, "eta": 0.02}
}

# Run experiments
experiment_results = []
fig, axs = plt.subplots(6, 2, figsize=(14, 24))

for idx, (material, props) in enumerate(materials_tuned.items()):
    for j, mode in enumerate(["Multi-Pulse Spike", "RBF-Controlled"]):
        E = np.zeros((grid_size, grid_size))
        I = np.zeros((grid_size, grid_size))
        M = np.zeros((grid_size, grid_size))
        T = np.zeros((grid_size, grid_size))
        Phi = np.ones((grid_size, grid_size))
        B = np.zeros((grid_size, grid_size))
        rbf_state = np.zeros((grid_size, grid_size))

        harvested_energy = []
        net_energy = []

        for t in range(timesteps):
            for i in range(pulse_count):
                pulse_start = 10 + pulse_interval * i
                if pulse_start <= t < pulse_start + pulse_duration:
                    for x in range(grid_size):
                        for y in range(grid_size):
                            if np.sqrt((x - pulse_center[0])**2 + (y - pulse_center[1])**2) < pulse_radius:
                                if T[x, y] < props["thermal_threshold"]:
                                    if mode == "RBF-Controlled":
                                        thermal_mod = max(0.0, 1 - (T[x, y] / props["thermal_threshold"]))
                                        memory_mod = max(0.0, 1 - (M[x, y] / 10.0))
                                        rbf_state[x, y] = 0.95 * rbf_state[x, y] + 0.05 * (thermal_mod * memory_mod)
                                        adaptive_strength = (reduced_pulse_strength *
                                                             props["conversion"] *
                                                             props["efficiency"] *
                                                             rbf_state[x, y]) / pulse_duration
                                        E[x, y] += adaptive_strength
                                    else:
                                        E[x, y] += (reduced_pulse_strength *
                                                    props["conversion"] *
                                                    props["efficiency"]) / pulse_duration

            I += 0.01 * (np.random.rand(grid_size, grid_size) - 0.5)
            M += np.abs(E - I) * 0.01
            T += 0.1 * (B**2)
            T *= thermal_dissipation

            R = np.exp(-props["gamma"] * M) * np.exp(-props["eta"] * T)
            B = lambda_0 * R * ((E - I) / (1 + alpha * M)) * Phi

            total_field_energy = np.sum(B**2)
            harvested = total_field_energy * props["harvest"]
            net = harvested - (pi_draw * grid_size * grid_size)

            harvested_energy.append(harvested)
            net_energy.append(net)

        gross_energy = sum(harvested_energy)
        net_energy_total = sum(net_energy)

        experiment_results.append({
            "Material": material,
            "Control Strategy": mode,
            "Gross Energy (J)": gross_energy,
            "Net Energy (J)": net_energy_total
        })

        row = 2 * idx + j
        axs[row, 0].plot(harvested_energy, label="Harvested")
        axs[row, 0].plot(net_energy, label="Net")
        axs[row, 0].set_title(f"{material} - {mode}")
        axs[row, 0].legend()
        axs[row, 0].grid(True)

        im = axs[row, 1].imshow(T, cmap='hot', origin='lower')
        axs[row, 1].set_title("Final Thermal Field")

plt.tight_layout()
plt.show()

# Show energy log
experiment_df = pd.DataFrame(experiment_results)
print(experiment_df)
