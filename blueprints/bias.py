import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from matplotlib.colors import Normalize

# Simulation parameters
GRID_SIZE = 100
TIMESTEPS = 50
COLLAPSE_THRESHOLD = 0.6
MEASUREMENT_INTENSITY = 1.5  # Attention pressure factor

# Initialize fields
np.random.seed(42)
base_field = np.random.rand(GRID_SIZE, GRID_SIZE)
memory_field = np.zeros((GRID_SIZE, GRID_SIZE))
collapse_field_control = np.zeros((GRID_SIZE, GRID_SIZE))
collapse_field_bias = np.zeros((GRID_SIZE, GRID_SIZE))

# Bias zone definition
bias_zone = (slice(40, 60), slice(40, 60))

# Control group simulation (no measurement bias)
for t in range(TIMESTEPS):
    pressure = np.gradient(base_field)[0] + np.gradient(base_field)[1]
    collapse = (np.abs(pressure) > COLLAPSE_THRESHOLD).astype(float)
    collapse_field_control += collapse
    base_field += 0.01 * np.random.randn(GRID_SIZE, GRID_SIZE)

# Reset fields for bias group simulation
np.random.seed(42)
base_field = np.random.rand(GRID_SIZE, GRID_SIZE)
memory_field = np.zeros((GRID_SIZE, GRID_SIZE))

# Apply measurement bias at t=0
base_field[bias_zone] *= MEASUREMENT_INTENSITY

# Biased group simulation
for t in range(TIMESTEPS):
    pressure = np.gradient(base_field)[0] + np.gradient(base_field)[1]
    collapse = (np.abs(pressure) > COLLAPSE_THRESHOLD).astype(float)
    collapse_field_bias += collapse
    base_field += 0.01 * np.random.randn(GRID_SIZE, GRID_SIZE)
    memory_field += collapse

# KL divergence between histograms
bins = 20
hist_control, _ = np.histogram(collapse_field_control, bins=bins, range=(0, 1), density=True)
hist_bias, _ = np.histogram(collapse_field_bias, bins=bins, range=(0, 1), density=True)
kl_divergence = entropy(hist_bias + 1e-9, hist_control + 1e-9)

# Normalize fields for plotting
norm = Normalize(vmin=0, vmax=max(np.max(collapse_field_control), np.max(collapse_field_bias)))

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(collapse_field_control, ax=axes[0], cmap='Blues', cbar=False)
axes[0].set_title("Control Collapse Field")

sns.heatmap(collapse_field_bias, ax=axes[1], cmap='Reds', cbar=False)
axes[1].set_title("Biased Collapse Field")

sns.heatmap(collapse_field_bias - collapse_field_control, ax=axes[2], cmap='coolwarm', center=0, cbar=False)
axes[2].set_title("Anomaly Field (Bias - Control)")

plt.suptitle(f"KL Divergence (Bias vs Control): {kl_divergence:.4f}")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
