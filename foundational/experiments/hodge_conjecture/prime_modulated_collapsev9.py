import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import find_peaks
import csv
from numpy.fft import fft2, fftshift
from statistics import mean, stdev

# Define both prime and non-prime modulated constants
primes = [2, 3, 5, 7, 11, 4, 6, 8, 9, 10]  # includes non-primes for comparison
modulations = [p * pi for p in primes]

# Parameters for robustness testing
noise_runs = 5
parameter_noise = 0.02


# Storage for extended analysis
symmetry_scores = {}
density_maps = {}
radial_profiles = {}
peak_counts = {}
symmetry_maps = {}
mod_p_fields = {}
all_scores = {p: [] for p in primes}


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

    return symbolic_history[-1], symbolic_history


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


def apply_mod_p_transform(symbolic_field, p):
    return np.mod(symbolic_field, p)


def extract_stable_cycles(density_map, threshold=80):
    return (density_map > threshold).astype(int)

# Run simulations with robustness testing
for p, n in zip(primes, modulations):
    for run in range(noise_runs):
        tau_noise = 0.65 + np.random.normal(0, parameter_noise)
        gamma_e_noise = 0.95 + np.random.normal(0, parameter_noise)
        gamma_s_noise = 0.96 + np.random.normal(0, parameter_noise)

        final_field, history = run_simulation(n, tau_c=tau_noise, gamma_energy=gamma_e_noise, gamma_symbolic=gamma_s_noise)
        density = compute_density(history)
        r, profile = radial_profile(density)
        count, positions = count_radial_peaks(profile)
        fft_map, score = compute_fft_symmetry(density)
        mod_p = apply_mod_p_transform(final_field, p)
        cycles = extract_stable_cycles(density)

        all_scores[p].append(score)

        if run == 0:
            symmetry_scores[n] = score
            density_maps[n] = density
            radial_profiles[n] = (r, profile)
            peak_counts[n] = (count, positions)
            symmetry_maps[n] = fft_map
            mod_p_fields[n] = mod_p

        with open("robustness_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([p, n, run, count, score, ";".join(map(str, positions))])

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(density, cmap='inferno')
        axs[0, 0].set_title(f"Density Map (p={p}, run={run})")

        axs[0, 1].plot(r, profile)
        axs[0, 1].set_title("Radial Profile")

        axs[1, 0].imshow(np.log1p(fft_map), cmap='plasma')
        axs[1, 0].set_title("FFT Symmetry")

        axs[1, 1].imshow(cycles, cmap='gray')
        axs[1, 1].set_title("Stable Symbolic Cycles")

        plt.tight_layout()
        plt.savefig(f"analysis_plot_p{p}_run{run}.png")
        plt.close()

# Summary analysis: compare prime vs non-prime
prime_scores = [mean(all_scores[p]) for p in [2, 3, 5, 7, 11]]
nonprime_scores = [mean(all_scores[p]) for p in [4, 6, 8, 9, 10]]

print("Prime Mean Symmetry:", mean(prime_scores), "+/-", stdev(prime_scores))
print("Non-Prime Mean Symmetry:", mean(nonprime_scores), "+/-", stdev(nonprime_scores))

# Plot symmetry score vs index
plt.figure(figsize=(10, 6))
plt.plot(primes, [symmetry_scores[n] for n in modulations], marker='o')
plt.title("FFT Symmetry Score vs Index")
plt.xlabel("Modulation Index (Prime & Non-Prime)")
plt.ylabel("FFT Symmetry Score")
plt.grid(True)
plt.savefig("fft_symmetry_vs_index.png")
plt.show()

# Visualization
fig, axes = plt.subplots(4, len(primes), figsize=(5 * len(primes), 20))
for i, (n, density) in enumerate(density_maps.items()):
    index = primes[i]

    ax_img = axes[0, i]
    im = ax_img.imshow(density, cmap='inferno')
    ax_img.set_title(f"Density: Index {index}")
    ax_img.axis('off')
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    ax_plot = axes[1, i]
    r, profile = radial_profiles[n]
    ax_plot.plot(r, profile)
    ax_plot.set_title(f"Radial Profile (Peaks: {peak_counts[n][0]})")

    ax_fft = axes[2, i]
    im_fft = ax_fft.imshow(np.log1p(symmetry_maps[n]), cmap='plasma')
    ax_fft.set_title(f"FFT Symmetry Score: {symmetry_scores[n]:.2f}")
    ax_fft.axis('off')
    fig.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

    ax_mod = axes[3, i]
    im_mod = ax_mod.imshow(mod_p_fields[n], cmap='cividis')
    ax_mod.set_title(f"Symbolic Field mod {index}")
    ax_mod.axis('off')
    fig.colorbar(im_mod, ax=ax_mod, fraction=0.046, pad=0.04)

plt.suptitle("Modulated Symbolic Collapse: Full Analysis Suite (Primes & Non-Primes)", fontsize=20)
plt.tight_layout()
plt.savefig("modulated_full_analysis.png")
plt.show()
