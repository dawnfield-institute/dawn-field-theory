import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import find_peaks
import csv
from numpy.fft import fft2, fftshift

# Define prime-modulated constants
primes = [2, 3, 5, 7, 11]
modulations = [p * pi for p in primes]


def run_simulation(n, grid_size=256, steps=100, tau_c=0.65, gamma_energy=0.95, gamma_symbolic=0.96, lambda_reinforce=0.05):
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    angular_bias = np.sin(n * theta)

    np.random.seed(42)
    energy = np.random.normal(0.5, 0.1, size=(grid_size, grid_size))
    symbolic = np.random.normal(0.5, 0.1, size=(grid_size, grid_size))
    energy = np.clip(energy, 0, 1)
    symbolic = np.clip(symbolic, 0, 1)

    symbolic_history = [symbolic.copy()]
    for _ in range(steps):
        symbolic_mod = symbolic * (1 + 0.3 * angular_bias)
        crystallized = ((symbolic_mod + energy) / 2 > tau_c).astype(float)
        energy = gamma_energy * energy + lambda_reinforce * crystallized
        symbolic = gamma_symbolic * symbolic + lambda_reinforce * crystallized
        energy = np.clip(energy, 0, 1)
        symbolic = np.clip(symbolic, 0, 1)
        symbolic_history.append(symbolic.copy())

    return symbolic_history


def compute_density(history, threshold=0.7):
    return sum((frame > threshold).astype(int) for frame in history)


def radial_profile(density_map, bins=50):
    h, w = density_map.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_bin = np.linspace(0, r.max(), bins)
    radial_density = np.zeros(bins - 1)

    for i in range(len(r_bin) - 1):
        mask = (r >= r_bin[i]) & (r < r_bin[i+1])
        radial_density[i] = density_map[mask].mean() if np.any(mask) else 0

    return r_bin[:-1], radial_density


def count_radial_peaks(profile):
    peaks, _ = find_peaks(profile)
    return len(peaks), peaks


def compute_fft_symmetry(density_map):
    fft_map = fftshift(np.abs(fft2(density_map)))
    symmetry_score = np.mean(fft_map)
    return fft_map, symmetry_score

# Run simulations and collect metrics
density_maps = {}
radial_profiles = {}
peak_counts = {}
symmetry_maps = {}
symmetry_scores = {}

for n in modulations:
    history = run_simulation(n)
    density = compute_density(history)
    density_maps[n] = density
    r, profile = radial_profile(density)
    radial_profiles[n] = (r, profile)
    peak_counts[n] = count_radial_peaks(profile)
    fft_map, score = compute_fft_symmetry(density)
    symmetry_maps[n] = fft_map
    symmetry_scores[n] = score

# Export peak and symmetry data to CSV
with open("peak_symmetry_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Prime", "Modulation", "Peak Count", "FFT Symmetry Score"])
    for n in modulations:
        prime = round(n / pi)
        peak_count = peak_counts[n][0]
        score = symmetry_scores[n]
        writer.writerow([prime, n, peak_count, score])

# Visualization
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
for i, (n, density) in enumerate(density_maps.items()):
    prime = round(n / pi)

    # Density map
    ax_img = axes[0, i]
    im = ax_img.imshow(density, cmap='inferno')
    ax_img.set_title(f"Density: Prime {prime}")
    ax_img.axis('off')
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    # Radial profile
    ax_plot = axes[1, i]
    r, profile = radial_profiles[n]
    ax_plot.plot(r, profile)
    ax_plot.set_title(f"Radial Profile (Peaks: {peak_counts[n][0]})")

    # FFT symmetry
    ax_fft = axes[2, i]
    im_fft = ax_fft.imshow(np.log1p(symmetry_maps[n]), cmap='plasma')
    ax_fft.set_title(f"FFT Symmetry Score: {symmetry_scores[n]:.2f}")
    ax_fft.axis('off')
    fig.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

plt.suptitle("Prime-Modulated Symbolic Collapse: Density, Radial Symmetry, and FFT Symmetry Score", fontsize=20)
plt.tight_layout()
plt.show()
