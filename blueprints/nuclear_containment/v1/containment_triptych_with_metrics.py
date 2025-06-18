import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colors as mcolors
import pandas as pd

# Grid and simulation parameters
dx_km = 0.005  # 5 meters
grid_size_km = 2.0  # 2 kilometers
grid_size = int(grid_size_km / dx_km)
timesteps = 500
center = (grid_size // 2, grid_size // 2)
initial_temperature = 1e7

# Set up grid
Y, X = np.ogrid[:grid_size, :grid_size]
dist_km = np.sqrt((X - center[0])**2 + (Y - center[1])**2) * dx_km

# Initialize base fields
def blast_init():
    T = np.zeros((grid_size, grid_size))
    T[dist_km < 0.05] = initial_temperature  # 50m radius blast
    return T

T_control = blast_init()
T_late = blast_init()
T_primed = blast_init()

# Dynamic shockwave radius calculation function
def dynamic_radius(T, threshold=1000):
    cell_area = dx_km**2
    area = np.sum(T > threshold) * cell_area
    return np.sqrt(area / np.pi)

# Uncontained control simulation
radii_control = []
for _ in range(timesteps):
    T_new = T_control.copy()
    T_new[1:-1,1:-1] += 0.05 * (
        T_control[2:,1:-1] + T_control[:-2,1:-1] +
        T_control[1:-1,2:] + T_control[1:-1,:-2] - 4 * T_control[1:-1,1:-1]
    )
    T_control = T_new
    radii_control.append(dynamic_radius(T_control))

# Late containment simulation
I_late = np.zeros_like(T_late)
M_late = np.zeros_like(T_late)
Phi = np.sin(2 * np.pi * X / 20) * np.cos(2 * np.pi * Y / 20)
Phi = (Phi - Phi.min()) / (Phi.max() - Phi.min()) + 0.5
alpha, gamma_T, gamma_I, lambda_0 = 0.001, 0.12, 0.06, 2.5
radii_late = []

for t in range(timesteps):
    if t < 250:
        T_new = T_late.copy()
        T_new[1:-1,1:-1] += 0.05 * (
            T_late[2:,1:-1] + T_late[:-2,1:-1] +
            T_late[1:-1,2:] + T_late[1:-1,:-2] - 4 * T_late[1:-1,1:-1]
        )
        T_late = T_new
    else:
        imbalance = np.abs(T_late - I_late)
        M_late += imbalance * 0.001
        RBF = np.exp(-gamma_T * T_late.clip(0, 1e6)) * np.exp(-gamma_I * I_late.clip(0, 1e6))
        stabilized = lambda_0 * RBF * ((T_late - I_late) / (1 + alpha * M_late)) * Phi
        T_late *= 0.998
        I_late *= 0.997
        T_late -= stabilized * 0.35
        I_late += stabilized * 0.2
    radii_late.append(dynamic_radius(T_late))

# Pre-primed containment simulation
I_primed = np.zeros_like(T_primed)
M_primed = np.zeros_like(T_primed)
pulse_step = 250
pulse_strength = 5e5
radii_primed = []

for t in range(timesteps):
    active_mask = (T_primed > 1000)
    thermal_noise = (np.random.rand(*T_primed.shape) - 0.5) * 2000 * active_mask
    informatic_noise = (np.random.rand(*I_primed.shape) - 0.5) * 0.02 * I_primed * active_mask
    phi_noise = 1 + (np.random.rand(*Phi.shape) - 0.5) * 0.04

    if t == pulse_step:
        T_primed[dist_km < 0.05] += pulse_strength
        Phi = 1.0 - Phi

    imbalance = np.abs(T_primed - I_primed)
    M_primed += imbalance * 0.001
    RBF = np.exp(-gamma_T * T_primed.clip(0, 1e6)) * np.exp(-gamma_I * I_primed.clip(0, 1e6))
    stabilized = lambda_0 * RBF * ((T_primed - I_primed) / (1 + alpha * M_primed)) * Phi * phi_noise
    T_primed *= 0.997
    I_primed *= 0.996
    T_primed += thermal_noise
    I_primed += informatic_noise
    T_primed -= stabilized * 0.45
    I_primed += stabilized * 0.3
    radii_primed.append(dynamic_radius(T_primed))

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
norm = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=np.max(T_control))

ax1.imshow(T_control, cmap="hot", origin="lower", extent=[0, grid_size_km, 0, grid_size_km], norm=norm)
ax1.add_patch(Circle((grid_size_km/2, grid_size_km/2), radii_control[-1], edgecolor='cyan',
                     facecolor='none', linewidth=2, linestyle='--'))
ax1.set_title("Uncontained Nuclear Blast")
ax1.set_xlabel("Kilometers")
ax1.set_ylabel("Kilometers")

ax2.imshow(T_late, cmap="hot", origin="lower", extent=[0, grid_size_km, 0, grid_size_km], norm=norm)
ax2.add_patch(Circle((grid_size_km/2, grid_size_km/2), radii_late[-1], edgecolor='cyan',
                     facecolor='none', linewidth=2, linestyle='--'))
ax2.set_title("Late Containment Attempt")
ax2.set_xlabel("Kilometers")
ax2.set_ylabel("Kilometers")

img3 = ax3.imshow(T_primed, cmap="hot", origin="lower", extent=[0, grid_size_km, 0, grid_size_km], norm=norm)
ax3.add_patch(Circle((grid_size_km/2, grid_size_km/2), radii_primed[-1], edgecolor='cyan',
                     facecolor='none', linewidth=2, linestyle='--'))
ax3.set_title("Pre-Primed Containment (Pulse + Phase Flip)")
ax3.set_xlabel("Kilometers")
ax3.set_ylabel("Kilometers")

plt.colorbar(img3, ax=[ax1, ax2, ax3], label="Temperature (K)", shrink=0.8)
plt.savefig("containment_triptych.png")
plt.show()
