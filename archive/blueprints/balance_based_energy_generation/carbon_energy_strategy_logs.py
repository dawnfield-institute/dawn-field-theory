
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt

# Constants
grid_size = 100
timesteps = 1200
pulse_duration = 3
reduced_pulse_strength = 2.5
thermal_dissipation = 0.96
lambda_0 = 1.0
alpha = 0.01
pi_draw = 0.004
pulse_center = (grid_size // 2, grid_size // 2)
pulse_radius = 10

materials_tuned = {
    "Graphite": {"conversion": 0.18, "efficiency": 0.3, "harvest": 0.28, "thermal_threshold": 4.0, "gamma": 0.007, "eta": 0.004},
    "Graphene": {"conversion": 0.35, "efficiency": 0.6, "harvest": 0.4, "thermal_threshold": 5.5, "gamma": 0.006, "eta": 0.003},
    "CNT": {"conversion": 0.9, "efficiency": 0.85, "harvest": 0.55, "thermal_threshold": 6.5, "gamma": 0.005, "eta": 0.002}
}

X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
distance_from_center = np.sqrt((X - pulse_center[0])**2 + (Y - pulse_center[1])**2)
core_mask = distance_from_center < pulse_radius

def simulate_material(material, props, wave):
    E = np.zeros((grid_size, grid_size))
    I = np.zeros((grid_size, grid_size))
    M = np.zeros((grid_size, grid_size))
    T = np.zeros((grid_size, grid_size))
    Phi = np.ones((grid_size, grid_size))
    B = np.zeros((grid_size, grid_size))
    rbf_state = np.zeros((grid_size, grid_size))
    efficiency_map = np.ones((grid_size, grid_size)) * props["efficiency"]

    harvested_energy = []
    net_energy = []

    for t in range(timesteps):
        thermal_mod = np.maximum(0.0, 1 - (T / props["thermal_threshold"]))
        memory_mod = np.maximum(0.0, 1 - (M / 10.0))
        rbf_state = 0.95 * rbf_state + 0.05 * (thermal_mod * memory_mod)

        energy_input = (wave[t] * reduced_pulse_strength *
                        props["conversion"] * efficiency_map * rbf_state) / pulse_duration

        E += core_mask * (T < props["thermal_threshold"]) * energy_input

        I += 0.01 * (np.random.rand(grid_size, grid_size) - 0.5)
        M += np.abs(E - I) * 0.01
        T += 0.1 * (B**2)
        T *= thermal_dissipation
        efficiency_map *= np.exp(-0.0001 * M)

        R = np.exp(-props["gamma"] * M) * np.exp(-props["eta"] * T)
        B = lambda_0 * R * ((E - I) / (1 + alpha * M)) * Phi

        harvested = np.sum(B**2) * props["harvest"]
        net = harvested - (pi_draw * grid_size * grid_size)
        harvested_energy.append(harvested)
        net_energy.append(net)

    return sum(harvested_energy), sum(net_energy)

# Define waves
sigmoid_wave = expit(np.linspace(-6, 6, timesteps))
gentler_wave = expit(np.linspace(-8, 8, timesteps))

results = []
for material, props in materials_tuned.items():
    gross1, net1 = simulate_material(material, props, sigmoid_wave)
    gross2, net2 = simulate_material(material, props, gentler_wave)
    results.append({"Material": material, "Control Strategy": "Sigmoid", "Gross Energy (J)": gross1, "Net Energy (J)": net1})
    results.append({"Material": material, "Control Strategy": "Gentle Sigmoid", "Gross Energy (J)": gross2, "Net Energy (J)": net2})

df = pd.DataFrame(results)
print(df)
