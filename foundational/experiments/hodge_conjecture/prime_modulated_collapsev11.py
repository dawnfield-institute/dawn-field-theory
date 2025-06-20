import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import find_peaks
import csv
from numpy.fft import fft2, fftshift
from statistics import mean, stdev
from scipy.stats import ttest_ind
import seaborn as sns

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
all_peaks = {p: [] for p in primes}

symbolic_cycles_by_index = {}


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


def extract_symbolic_cycle_shapes(binary_map):
    from scipy.ndimage import label
    structure = np.ones((3, 3))
    labeled, num_features = label(binary_map, structure)
    return labeled, num_features

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
        all_peaks[p].append(count)

        if run == 0:
            symmetry_scores[n] = score
            density_maps[n] = density
            radial_profiles[n] = (r, profile)
            peak_counts[n] = (count, positions)
            symmetry_maps[n] = fft_map
            mod_p_fields[n] = mod_p

            cycle_shapes, num = extract_symbolic_cycle_shapes(cycles)
            symbolic_cycles_by_index[p] = (cycle_shapes, num)

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

# Cycle summary output for next analysis phase
for p, (shapes, num) in symbolic_cycles_by_index.items():
    print(f"Index {p}: {num} symbolic cycles extracted")
