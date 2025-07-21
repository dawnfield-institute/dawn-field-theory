import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
import csv

# Configuration
FIELD_SIZE = 200
TRIALS = 30  # fewer trials for sharper pattern
PATH_POINTS = 40  # fewer path points per trial
SOURCE_POINTS = [50, 150]  # slit positions
PATH_SPREAD = 7  # smaller spread for less blurring
REINFORCEMENT_DECAY = 0.95
REINFORCEMENT_BOOST = 0.1
EDGE_MASK = 15  # number of points to mask at each edge

# --- Analytic quantum double-slit overlay ---
def analytic_double_slit(x, slit1, slit2, wavelength):
    x = np.asarray(x)
    d1 = np.abs(x - slit1)
    d2 = np.abs(x - slit2)
    phase_diff = 2 * np.pi * (d2 - d1) / wavelength
    intensity = np.cos(phase_diff / 2) ** 2
    return intensity / np.max(intensity)

x = np.arange(FIELD_SIZE)
slit1, slit2 = SOURCE_POINTS
wavelength = 2 * (slit2 - slit1) / 8  # set so there are 8 main fringes between slits
analytic_intensity = analytic_double_slit(x, slit1, slit2, wavelength)

# --- Parameter sweep for phase noise and prime modulus ---
PHASE_NOISE_STD_VALUES = np.linspace(0, 1.0, 9)  # e.g., [0, 0.125, ..., 1.0]
PRIME_MODULUS_VALUES = [1, 2, 3, 5, 7, 11]

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"interference_test_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

results = []
for PHASE_NOISE_STD in PHASE_NOISE_STD_VALUES:
    for PRIME_MODULUS in PRIME_MODULUS_VALUES:
        field = np.zeros(FIELD_SIZE)
        reinforcement_map = np.zeros(FIELD_SIZE)
        x = np.arange(FIELD_SIZE)
        wavelength = 2 * (SOURCE_POINTS[1] - SOURCE_POINTS[0]) / 8
        k = 2 * np.pi / wavelength
        for xi in range(FIELD_SIZE):
            amp_sum = 0j
            for src in SOURCE_POINTS:
                path_length = abs(xi - src)
                mod_angle = PRIME_MODULUS * np.pi
                phase = k * path_length + mod_angle
                phase += np.random.normal(0, PHASE_NOISE_STD)
                amp = np.exp(1j * phase)
                amp_sum += amp
            intensity = np.abs(amp_sum) ** 2
            field[xi] = intensity
            reinforcement_map[xi] = np.abs(amp_sum)
        # Mask out edge flatlines
        mask = np.ones(FIELD_SIZE, dtype=bool)
        mask[:EDGE_MASK] = False
        mask[-EDGE_MASK:] = False
        field_masked = field[mask]
        analytic_masked = analytic_intensity[mask]
        # Normalize
        field_masked_norm = field_masked / np.max(field_masked)
        analytic_masked_norm = analytic_masked / np.max(analytic_masked)
        # Metrics
        mse = np.mean((field_masked_norm - analytic_masked_norm) ** 2)
        correlation, _ = pearsonr(field_masked_norm, analytic_masked_norm)
        results.append({
            "phase_noise_std": PHASE_NOISE_STD,
            "prime_modulus": PRIME_MODULUS,
            "mse": mse,
            "pearson_corr": correlation
        })

# Save results to CSV
csv_path = os.path.join(output_dir, "parameter_sweep_results.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["phase_noise_std", "prime_modulus", "mse", "pearson_corr"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f"Parameter sweep results saved to: {csv_path}")

# --- Save representative plots for key parameter regimes ---
def save_overlay_plot(field, reinforcement_map, analytic_intensity, mask, label, output_dir):
    field_masked = field[mask]
    reinforcement_masked = reinforcement_map[mask]
    analytic_masked = analytic_intensity[mask]
    field_masked_norm = field_masked / np.max(field_masked)
    reinforcement_masked_norm = reinforcement_masked / np.max(reinforcement_masked)
    analytic_masked_norm = analytic_masked / np.max(analytic_masked)
    plt.figure()
    plt.plot(np.arange(EDGE_MASK, FIELD_SIZE-EDGE_MASK), field_masked_norm, label='Symbolic Path Intensity')
    plt.plot(np.arange(EDGE_MASK, FIELD_SIZE-EDGE_MASK), reinforcement_masked_norm, label='Reinforcement Map', linestyle='--')
    plt.plot(np.arange(EDGE_MASK, FIELD_SIZE-EDGE_MASK), analytic_masked_norm, label='Analytic Quantum Interference', linestyle=':')
    plt.title(f"Symbolic vs Quantum Interference Pattern\n{label}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"symbolic_vs_quantum_interference_{label}.png"))
    plt.close()

# Save overlays for min, mid, max phase noise and prime modulus=1,7
for PHASE_NOISE_STD in [0.0, 0.5, 1.0]:
    for PRIME_MODULUS in [1, 7]:
        field = np.zeros(FIELD_SIZE)
        reinforcement_map = np.zeros(FIELD_SIZE)
        x = np.arange(FIELD_SIZE)
        wavelength = 2 * (SOURCE_POINTS[1] - SOURCE_POINTS[0]) / 8
        k = 2 * np.pi / wavelength
        for xi in range(FIELD_SIZE):
            amp_sum = 0j
            for src in SOURCE_POINTS:
                path_length = abs(xi - src)
                mod_angle = PRIME_MODULUS * np.pi
                phase = k * path_length + mod_angle
                phase += np.random.normal(0, PHASE_NOISE_STD)
                amp = np.exp(1j * phase)
                amp_sum += amp
            intensity = np.abs(amp_sum) ** 2
            field[xi] = intensity
            reinforcement_map[xi] = np.abs(amp_sum)
        mask = np.ones(FIELD_SIZE, dtype=bool)
        mask[:EDGE_MASK] = False
        mask[-EDGE_MASK:] = False
        label = f"noise{PHASE_NOISE_STD}_prime{PRIME_MODULUS}"
        save_overlay_plot(field, reinforcement_map, analytic_intensity, mask, label, output_dir)

# --- Generate heatmaps for publication ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(csv_path)
pivot_corr = df.pivot(index='phase_noise_std', columns='prime_modulus', values='pearson_corr')
pivot_mse = df.pivot(index='phase_noise_std', columns='prime_modulus', values='mse')

plt.figure(figsize=(8,6))
sns.heatmap(pivot_corr, annot=True, cmap='viridis', fmt='.2f')
plt.title('Pearson Correlation vs Analytic\n(symbolic vs quantum)')
plt.ylabel('Phase Noise Std')
plt.xlabel('Prime Modulus')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_pearson_corr.png'))
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(pivot_mse, annot=True, cmap='magma_r', fmt='.2f')
plt.title('MSE vs Analytic\n(symbolic vs quantum)')
plt.ylabel('Phase Noise Std')
plt.xlabel('Prime Modulus')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_mse.png'))
plt.close()

print(f"All summary plots and overlays saved to: {output_dir}")
