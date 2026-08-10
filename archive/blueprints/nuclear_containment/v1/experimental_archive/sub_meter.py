# Sub-meter Dawn Field Theory containment experiment with noise injection and failure probing
# Goal: Compress entropy containment into under 1m radius using RBF + QBE principles, then test noise resilience

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# Grid parameters (20cm resolution over 10m x 10m)
dx_m = 0.2
grid_size_m = 10
grid_size = int(grid_size_m / dx_m)
center = (grid_size // 2, grid_size // 2)
timesteps = 500
initial_temperature = 1e6  # reduced to avoid overflow
blast_radius_m = 0.5  # 50 cm

# Coordinate grid
dx_km = dx_m / 1000
Y, X = np.ogrid[:grid_size, :grid_size]
dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2) * dx_m
mask = dist < blast_radius_m

# Fields
T = np.zeros((grid_size, grid_size))
I = np.zeros_like(T)
M = np.zeros_like(T)
T[mask] = initial_temperature

# High-frequency structured field lattice
Phi = np.sin(2 * np.pi * X / 5) * np.cos(2 * np.pi * Y / 5)
Phi = (Phi - Phi.min()) / (Phi.max() - Phi.min()) + 0.5

# Parameters
alpha = 0.001
lambda_0 = 3.0  # aggressive coupling
gamma_T = 0.12
gamma_I = 0.08

# Noise parameters (scaled down and localized)
thermal_noise_amp = 1000       # reduced
informatic_noise_amp = 0.01    # reduced
phi_distortion_amp = 0.03      # slight distortion

# Temporal pulse parameters
pulse_step = 250
pulse_strength = 5e5  # energy added suddenly

# Logging
failure_time = None
leakage_logged = False
leakage_threshold = 1.0  # meters (converted to grid cells)
containment_radius = blast_radius_m + leakage_threshold
def get_leakage_area():
    return np.sum(dist[(T > 1000)] > containment_radius)

# Simulation
for t in range(timesteps):
    active_mask = (T > 1000)
    thermal_noise = (np.random.rand(*T.shape) - 0.5) * 2 * thermal_noise_amp * active_mask
    informatic_noise = (np.random.rand(*I.shape) - 0.5) * 2 * informatic_noise_amp * I * active_mask
    phi_noise = 1 + (np.random.rand(*Phi.shape) - 0.5) * 2 * phi_distortion_amp

    # Apply temporal energy pulse at defined step
    if t == pulse_step:
        T[mask] += pulse_strength

    # Invert Phi lattice midway to simulate phase inversion
    Phi_effective = Phi.copy()
    if t == pulse_step:
        Phi_effective = 1.0 - Phi_effective

    imbalance = np.abs(T - I)
    M += imbalance * 0.001

    T_clipped = np.clip(T, 0, 1e6)
    I_clipped = np.clip(I, 0, 1e6)
    RBF = np.exp(-gamma_T * T_clipped) * np.exp(-gamma_I * I_clipped)

    stabilized = lambda_0 * RBF * ((T - I) / (1 + alpha * M)) * Phi_effective * phi_noise
    T *= 0.997
    I *= 0.996
    T += thermal_noise
    I += informatic_noise
    T -= stabilized * 0.45
    I += stabilized * 0.3

    if not leakage_logged:
        if get_leakage_area() > 0:
            failure_time = t
            leakage_logged = True

# Visualization
fig, ax = plt.subplots(figsize=(7, 6))
norm = PowerNorm(gamma=0.3, vmin=0, vmax=np.max(T))
img = ax.imshow(T, cmap="hot", origin="lower",
                extent=[0, grid_size_m, 0, grid_size_m], norm=norm)
ax.set_title("Sub-Meter Containment: Pulse + Phase Inversion Stress Test")
ax.set_xlabel("Meters")
ax.set_ylabel("Meters")
plt.colorbar(img, ax=ax, label="Temperature (K)")
plt.tight_layout()
plt.show()

if failure_time is not None:
    print(f"Containment failure detected at timestep {failure_time}.")
else:
    print("Containment maintained under pulse and phase inversion stress.")
