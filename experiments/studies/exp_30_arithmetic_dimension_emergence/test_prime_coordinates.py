"""
Test 2: Prime Statistics Across Coordinate Systems
Compute gap statistics for first 10,000 primes in additive, multiplicative,
and exponential coordinate systems.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Generate first 10,000 primes via sieve ---
def sieve_primes(n):
    """Return first n primes."""
    # Estimate upper bound using prime number theorem
    if n < 6:
        limit = 15
    else:
        limit = int(n * (np.log(n) + np.log(np.log(n))) * 1.2) + 100

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    if len(primes) < n:
        # Extend if needed
        return sieve_primes_extended(n)
    return primes[:n]

def sieve_primes_extended(n):
    limit = int(n * (np.log(n) + np.log(np.log(n))) * 2) + 1000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    return primes[:n]

print("Generating first 10,000 primes...")
primes = sieve_primes(10000).astype(float)
print(f"  Range: {primes[0]:.0f} to {primes[-1]:.0f}")

# --- Compute gaps in three coordinate systems ---
print("\nComputing gaps in three coordinate systems...")

# Additive: g_n = p_{n+1} - p_n
gaps_add = np.diff(primes)

# Multiplicative: r_n = p_{n+1} / p_n
gaps_mul = primes[1:] / primes[:-1]

# Exponential: l_n = log(p_{n+1}) - log(p_n)
gaps_exp = np.diff(np.log(primes))

def compute_entropy(data, n_bins=100):
    """Compute Shannon entropy of histogram."""
    counts, _ = np.histogram(data, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def compute_stats(data, name):
    """Compute comprehensive statistics."""
    mean = float(np.mean(data))
    var = float(np.var(data))
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))
    cv = float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else float('inf')
    entropy = compute_entropy(data)

    result = {
        "mean": mean,
        "variance": var,
        "std": float(np.sqrt(var)),
        "skewness": skew,
        "kurtosis": kurt,
        "coefficient_of_variation": cv,
        "entropy_bits": entropy,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data))
    }

    print(f"\n  {name}:")
    for k, v in result.items():
        print(f"    {k}: {v:.6f}")

    return result

all_stats = {}
all_stats["additive"] = compute_stats(gaps_add, "Additive (p_{n+1} - p_n)")
all_stats["multiplicative"] = compute_stats(gaps_mul, "Multiplicative (p_{n+1} / p_n)")
all_stats["exponential"] = compute_stats(gaps_exp, "Exponential (log p_{n+1} - log p_n)")

# --- Which is most uniform? ---
print("\n" + "=" * 60)
print("UNIFORMITY COMPARISON")
print("=" * 60)

cvs = {k: v["coefficient_of_variation"] for k, v in all_stats.items()}
ents = {k: v["entropy_bits"] for k, v in all_stats.items()}

best_cv = min(cvs, key=cvs.get)
best_ent = max(ents, key=ents.get)

print(f"\nLowest CV  (most uniform): {best_cv} (CV = {cvs[best_cv]:.6f})")
print(f"Highest entropy (most uniform): {best_ent} (H = {ents[best_ent]:.4f} bits)")

for name in ["additive", "multiplicative", "exponential"]:
    print(f"  {name}: CV={cvs[name]:.6f}, H={ents[name]:.4f}")

# --- Autocorrelation ---
print("\nComputing autocorrelations (lags 1-20)...")

def autocorrelation(data, max_lag=20):
    """Compute autocorrelation for lags 1..max_lag."""
    data_centered = data - np.mean(data)
    var = np.var(data)
    if var == 0:
        return [0.0] * max_lag
    n = len(data)
    acf = []
    for lag in range(1, max_lag + 1):
        c = np.sum(data_centered[:n-lag] * data_centered[lag:]) / (n * var)
        acf.append(float(c))
    return acf

acf_add = autocorrelation(gaps_add)
acf_mul = autocorrelation(gaps_mul)
acf_exp = autocorrelation(gaps_exp)

all_stats["additive"]["autocorrelation_lags_1_20"] = acf_add
all_stats["multiplicative"]["autocorrelation_lags_1_20"] = acf_mul
all_stats["exponential"]["autocorrelation_lags_1_20"] = acf_exp

# --- Plot: 3x2 (histogram + autocorrelation for each) ---
print("\nGenerating figure...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Prime Gap Statistics in Three Coordinate Systems (N=10,000 primes)",
             fontsize=14, fontweight="bold")

coord_data = [
    ("Additive: $g_n = p_{n+1} - p_n$", gaps_add, acf_add, "steelblue"),
    ("Multiplicative: $r_n = p_{n+1}/p_n$", gaps_mul, acf_mul, "darkorange"),
    ("Exponential: $\\ell_n = \\ln p_{n+1} - \\ln p_n$", gaps_exp, acf_exp, "seagreen"),
]

for i, (title, data, acf, color) in enumerate(coord_data):
    # Histogram
    ax = axes[i, 0]
    ax.hist(data, bins=80, density=True, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_title(f"{title}\nHistogram", fontsize=11)
    ax.set_ylabel("Density")
    cv_val = np.std(data) / np.mean(data)
    ent_val = compute_entropy(data)
    ax.text(0.97, 0.95, f"CV={cv_val:.4f}\nH={ent_val:.2f} bits",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Autocorrelation
    ax = axes[i, 1]
    lags = np.arange(1, 21)
    ax.bar(lags, acf, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    # Significance bounds (approximate 95% CI for white noise)
    n = len(data)
    sig = 1.96 / np.sqrt(n)
    ax.axhline(sig, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-sig, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title}\nAutocorrelation", fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_xlim(0.5, 20.5)

plt.tight_layout()
figpath = os.path.join(OUTPUT_DIR, "test2_prime_coordinates.png")
plt.savefig(figpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to {figpath}")

# Save stats
all_stats["uniformity_winner_cv"] = best_cv
all_stats["uniformity_winner_entropy"] = best_ent

statpath = os.path.join(OUTPUT_DIR, "test2_prime_stats.json")
with open(statpath, "w") as f:
    json.dump(all_stats, f, indent=2)
print(f"Stats saved to {statpath}")
