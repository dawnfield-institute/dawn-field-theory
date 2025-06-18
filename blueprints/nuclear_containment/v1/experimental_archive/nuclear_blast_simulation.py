
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import pandas as pd

# Simulation parameters
dx_km = 0.1  # 100 meters per grid cell
grid_size_km = 100  # 100 km x 100 km domain
grid_size = int(grid_size_km / dx_km)
timesteps = 500
center = (grid_size // 2, grid_size // 2)
initial_temperature = 1e7
blast_radius_km = 1.5

# Initialize fields
T = np.zeros((grid_size, grid_size))
I = np.zeros((grid_size, grid_size))
logs = []

# Setup blast core
Y, X = np.ogrid[:grid_size, :grid_size]
dist_km = np.sqrt((X - center[0])**2 + (Y - center[1])**2) * dx_km
mask = dist_km < blast_radius_km
T[mask] = initial_temperature

# Run simulation
for t in range(timesteps):
    T_new = T.copy()
    T_new[1:-1,1:-1] += 0.05 * (
        T[2:,1:-1] + T[:-2,1:-1] + T[1:-1,2:] + T[1:-1,:-2] - 4 * T[1:-1,1:-1]
    )
    grad_x = np.abs(T[2:,1:-1] - T[:-2,1:-1])
    grad_y = np.abs(T[1:-1,2:] - T[1:-1,:-2])
    I[1:-1,1:-1] += 0.01 * (grad_x + grad_y)
    T = T_new

    if t % 100 == 0:
        active_area_km2 = np.sum(T > 1000) * (dx_km**2)
        max_temp = np.max(T)
        logs.append((t, max_temp, active_area_km2))

# Calculate shockwave radius
shock_speed_kms = 0.343
time_elapsed_s = timesteps * 0.1
shock_radius_km = shock_speed_kms * time_elapsed_s

# Plot final state
fig, ax = plt.subplots(figsize=(10, 8))
norm = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=np.max(T))
img = ax.imshow(T, cmap="hot", origin="lower", extent=[0, grid_size_km, 0, grid_size_km], norm=norm)

# Shockwave radius
shock_circle = Circle((grid_size_km/2, grid_size_km/2), shock_radius_km,
                      edgecolor='cyan', facecolor='none', linewidth=2, linestyle='--',
                      label=f"Shockwave (~{shock_radius_km:.1f} km)")
ax.add_patch(shock_circle)

ax.set_title(f"Nuclear Thermal Field + Shockwave Radius (Timestep {timesteps})")
ax.set_xlabel("Kilometers")
ax.set_ylabel("Kilometers")
ax.legend(loc="upper right")
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Temperature (K)")
plt.tight_layout()
plt.show()

# Log data output
log_df = pd.DataFrame(logs, columns=["Timestep", "Max_Temperature(K)", "Active_Area(km²)"])
print(log_df)
