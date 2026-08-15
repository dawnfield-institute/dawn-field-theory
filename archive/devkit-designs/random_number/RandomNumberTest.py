import secrets
import numpy as np
from blake3 import blake3
from scipy import stats
import random
import matplotlib.pyplot as plt
import math
from concurrent.futures import ThreadPoolExecutor

class MaximumChaosQRNG:
    """
    A fully software-based quantum-inspired random number generator (QRNG) using QBE-driven chaos maximization.
    This version reverses the Quantum Balance Equation (QBE) to create entropy imbalance, ensuring maximum randomness.
    Incorporates Schrödinger-style wavefunction collapse simulation and variable QBE unbalancing.
    Optimizations: reduced iteration loops, BLAKE3 hash, NumPy vectorization, and complex wavefunction seeding.
    """
    def __init__(self, size=64, iterations=6, seed=None):
        self.size = size
        self.iterations = iterations
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize_superposition(self):
        # Start with normalized complex wavefunction (superposition)
        real = np.random.uniform(-1, 1, self.size)
        imag = np.random.uniform(-1, 1, self.size)
        psi = real + 1j * imag
        norm = np.sqrt(np.sum(np.abs(psi) ** 2))  # Higher precision normalization
        return psi / norm

    def reverse_qbe_chaos(self, state):
        # Inject chaos into the complex wavefunction (vectorized)
        chaos_real = np.random.uniform(-1, 1, (self.iterations, self.size))
        chaos_imag = np.random.uniform(-1, 1, (self.iterations, self.size))
        injection = chaos_real + 1j * chaos_imag
        phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, self.iterations))
        for i in range(self.iterations):
            state += injection[i] * phases[i]
            state /= np.linalg.norm(state)  # Normalize to preserve probability
        return state

    def schrodinger_collapse(self, psi, outcome_space=256):
        # Collapse wavefunction based on probability amplitudes
        probs = np.abs(psi) ** 2
        probs /= np.sum(probs)
        collapsed = np.random.choice(len(psi), size=self.size, p=probs)
        return collapsed % outcome_space  # Configurable outcome space

    def collapse_state(self, state):
        collapsed = self.schrodinger_collapse(state)
        collapsed_bytes = np.array(collapsed, dtype=np.uint8).tobytes()
        return blake3(collapsed_bytes).hexdigest()

    def generate_random_number(self):
        superposition = self.initialize_superposition()
        evolved_state = self.reverse_qbe_chaos(superposition)
        collapsed_hash = self.collapse_state(evolved_state)
        max_hash_value = int("f" * len(collapsed_hash), 16)  # Maximum possible hash value
        random_number = int(collapsed_hash, 16) / max_hash_value  # Scale to [0, 1]
        return int(random_number * (10**18))  # Scale to desired range

    def generate_random_numbers_parallel(self, count):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda _: self.generate_random_number(), range(count)))
        return results

    def calculate_entropy(self, values):
        hist, _ = np.histogram(values, bins=256)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def chi_square_test(self, values, bins=256):
        observed, _ = np.histogram(values, bins=bins)
        total = np.sum(observed)
        expected = np.full_like(observed, total / bins, dtype=np.float64)
        observed = observed.astype(np.float64)
        chi_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        return chi_stat, p_value

    def get_stats(self, values):
        values = np.array(values, dtype=np.float64)
        epsilon = 1e-10
        scaled = (values - np.min(values)) / (np.max(values) - np.min(values) + epsilon)
        ks_stat, ks_pval = stats.kstest(scaled, 'uniform')
        entropy = self.calculate_entropy(scaled)
        chi_stat, chi_pval = self.chi_square_test(scaled)
        return {
            "Mean": float(np.mean(values)),
            "Standard Deviation": float(np.std(values)),
            "Kolmogorov-Smirnov Statistic": float(ks_stat),
            "Kolmogorov-Smirnov p-value": float(ks_pval),
            "Shannon Entropy": float(entropy),
            "Chi-Square Statistic": float(chi_stat),
            "Chi-Square p-value": float(chi_pval)
        }

    def test_randomness_average(self, sample_size=1000, runs=5):
        print(f"\nRunning {runs} test cycles with {sample_size} samples each...\n")
        all_stats = []
        for _ in range(runs):
            qrng_vals = self.generate_random_numbers_parallel(sample_size)  # Use parallel generation
            all_stats.append(self.get_stats(qrng_vals))

        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = sum(stats[key] for stats in all_stats) / runs

        print("[Quantum RNG - Averaged Results]")
        for key, value in avg_stats.items():
            print(f"{key:30}: {value:.6f}")
        return avg_stats

    def test_randomness(self, sample_size=1000):
        print("\nGenerating samples...")
        qrng_vals = self.generate_random_numbers_parallel(sample_size)  # Use parallel generation
        python_vals = [random.randint(0, 10**18) for _ in range(sample_size)]
        numpy_vals = [int(x) for x in np.random.default_rng().integers(0, 10**18, size=sample_size, dtype=np.uint64)]

        print("\nEvaluating randomness metrics...\n")
        results = {
            "Quantum RNG": self.get_stats(qrng_vals),
            "Python Built-in RNG": self.get_stats(python_vals),
            "NumPy RNG": self.get_stats(numpy_vals)
        }

        for label, stats_dict in results.items():
            print(f"[{label}]")
            for key, value in stats_dict.items():
                print(f"{key:30}: {value:.6f}")
            print()

        self.plot_histograms(qrng_vals, python_vals, numpy_vals)
        return results

    def plot_histograms(self, qrng_vals, python_vals, numpy_vals):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(qrng_vals, bins=50, color='skyblue', edgecolor='black')
        plt.title("Quantum RNG Distribution")

        plt.subplot(1, 3, 2)
        plt.hist(python_vals, bins=50, color='lightgreen', edgecolor='black')
        plt.title("Python Built-in RNG Distribution")

        plt.subplot(1, 3, 3)
        plt.hist(numpy_vals, bins=50, color='salmon', edgecolor='black')
        plt.title("NumPy RNG Distribution")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    qrng = MaximumChaosQRNG()
    print("Maximum Chaos Quantum Random Number:", qrng.generate_random_number())
    qrng.test_randomness_average(sample_size=1000, runs=5)
    qrng.test_randomness()
