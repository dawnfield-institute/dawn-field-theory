"""
Test 4: Feigenbaum Dimensional Decomposition
Compute the logistic map period-doubling cascade, find bifurcation points,
verify Feigenbaum ratio, and characterize arithmetic dimension at each level.
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}

# ============================================================
# Logistic map: x_{n+1} = r * x_n * (1 - x_n)
# ============================================================

def logistic(x, r):
    return r * x * (1 - x)

def iterate_logistic(r, x0=0.5, n_transient=10000, n_sample=1000):
    """Iterate logistic map, discard transient, return samples."""
    x = x0
    for _ in range(n_transient):
        x = logistic(x, r)
    samples = []
    for _ in range(n_sample):
        x = logistic(x, r)
        samples.append(x)
    return np.array(samples)

def find_period(r, tol=1e-8, max_period=512):
    """Find the period of the logistic map at parameter r."""
    x = 0.5
    for _ in range(50000):
        x = logistic(x, r)

    # Record orbit
    orbit = [x]
    for _ in range(max_period):
        x = logistic(x, r)
        orbit.append(x)
        # Check if we've returned
        for p in range(1, len(orbit)):
            if abs(orbit[-1] - orbit[-1-p]) < tol:
                return p
    return -1  # Chaotic or period > max_period

def lyapunov_exponent(r, n_iter=100000):
    """Compute Lyapunov exponent of logistic map at r."""
    x = 0.5
    lyap = 0.0
    for _ in range(1000):  # transient
        x = logistic(x, r)
    for _ in range(n_iter):
        deriv = abs(r * (1 - 2*x))
        if deriv > 0:
            lyap += np.log(deriv)
        x = logistic(x, r)
    return lyap / n_iter

# ============================================================
# Find bifurcation points
# ============================================================
print("=" * 60)
print("FINDING BIFURCATION POINTS")
print("=" * 60)

# Known approximate bifurcation points for period doubling:
# Period 1->2: r1 = 3.0
# Period 2->4: r2 ≈ 3.449490
# etc.

def find_bifurcation(r_low, r_high, target_period, n_steps=10000):
    """Binary search for bifurcation point where period doubles."""
    for _ in range(100):  # binary search iterations
        r_mid = (r_low + r_high) / 2
        p = find_period(r_mid)
        if p <= target_period:
            r_low = r_mid
        else:
            r_high = r_mid
        if r_high - r_low < 1e-12:
            break
    return (r_low + r_high) / 2

# Find bifurcation points sequentially
print("\nSearching for bifurcation points...")

# Start with known ranges
bif_ranges = [
    (2.9, 3.1, 1),      # period 1->2
    (3.4, 3.5, 2),      # period 2->4
    (3.54, 3.56, 4),    # period 4->8
    (3.564, 3.570, 8),  # period 8->16
    (3.5687, 3.5697, 16),  # period 16->32
    (3.56965, 3.56980, 32),  # period 32->64
    (3.569890, 3.569910, 64),  # period 64->128
    (3.569934, 3.569940, 128),  # period 128->256
]

bifurcation_points = []
for r_lo, r_hi, period in bif_ranges:
    try:
        r_bif = find_bifurcation(r_lo, r_hi, period)
        actual_period_before = find_period(r_bif - 0.0001)
        actual_period_after = find_period(r_bif + 0.0001)
        bifurcation_points.append({
            "r": r_bif,
            "period_before": actual_period_before,
            "period_after": actual_period_after,
            "target_period": period
        })
        print(f"  Period {period}->{2*period}: r = {r_bif:.10f}")
    except Exception as e:
        print(f"  Period {period}->{2*period}: FAILED ({e})")
        bifurcation_points.append({"r": None, "target_period": period, "error": str(e)})

results["bifurcation_points"] = bifurcation_points

# ============================================================
# Feigenbaum ratio
# ============================================================
print("\n" + "=" * 60)
print("FEIGENBAUM RATIO")
print("=" * 60)

r_values = [bp["r"] for bp in bifurcation_points if bp["r"] is not None]
deltas = []
for i in range(2, len(r_values)):
    dr_prev = r_values[i-1] - r_values[i-2]
    dr_curr = r_values[i] - r_values[i-1]
    if abs(dr_curr) > 1e-15:
        delta = dr_prev / dr_curr
        deltas.append(delta)
        print(f"  delta_{i} = {delta:.6f}")

if deltas:
    print(f"\n  Feigenbaum delta (theoretical): 4.669201...")
    print(f"  Mean computed delta: {np.mean(deltas):.6f}")
    print(f"  Last computed delta: {deltas[-1]:.6f}")

results["feigenbaum_ratios"] = deltas
results["feigenbaum_delta_theoretical"] = 4.669201609102990

# ============================================================
# Lyapunov exponents at bifurcation points
# ============================================================
print("\n" + "=" * 60)
print("LYAPUNOV EXPONENTS AT BIFURCATIONS")
print("=" * 60)

lyap_at_bif = []
for bp in bifurcation_points:
    if bp["r"] is not None:
        le = lyapunov_exponent(bp["r"])
        lyap_at_bif.append(le)
        print(f"  r = {bp['r']:.8f}: lambda = {le:.6f}")
    else:
        lyap_at_bif.append(None)

results["lyapunov_at_bifurcations"] = [float(x) if x is not None else None for x in lyap_at_bif]

# ============================================================
# Arithmetic dimension characterization
# ============================================================
print("\n" + "=" * 60)
print("ARITHMETIC DIMENSION AT EACH BIFURCATION")
print("=" * 60)

def fit_models(orbit):
    """
    Fit orbit to three models:
    (a) Linear: y = ax + b  [additive]
    (b) Power-law: y = a*x^b [multiplicative] -> log y = log a + b*log x
    (c) Exponential: y = a*e^{bx} [exponential] -> log y = log a + b*x

    Return R^2 for each.
    """
    x = orbit[:-1]
    y = orbit[1:]

    results_fit = {}

    # (a) Linear fit
    try:
        slope, intercept, r_val, _, _ = linregress(x, y)
        results_fit["linear"] = {"R2": r_val**2, "a": slope, "b": intercept}
    except:
        results_fit["linear"] = {"R2": 0.0}

    # (b) Power-law fit (on positive values only)
    try:
        mask = (x > 0) & (y > 0)
        if mask.sum() > 10:
            lx, ly = np.log(x[mask]), np.log(y[mask])
            slope, intercept, r_val, _, _ = linregress(lx, ly)
            results_fit["power_law"] = {"R2": r_val**2, "b_exponent": slope, "a_coeff": np.exp(intercept)}
        else:
            results_fit["power_law"] = {"R2": 0.0}
    except:
        results_fit["power_law"] = {"R2": 0.0}

    # (c) Exponential fit
    try:
        mask = y > 0
        if mask.sum() > 10:
            ly = np.log(y[mask])
            slope, intercept, r_val, _, _ = linregress(x[mask], ly)
            results_fit["exponential"] = {"R2": r_val**2, "b_rate": slope, "a_coeff": np.exp(intercept)}
        else:
            results_fit["exponential"] = {"R2": 0.0}
    except:
        results_fit["exponential"] = {"R2": 0.0}

    # Determine dominant
    r2s = {k: v["R2"] for k, v in results_fit.items()}
    dominant = max(r2s, key=r2s.get)
    results_fit["dominant"] = dominant

    return results_fit

arith_dim = []
for i, bp in enumerate(bifurcation_points):
    if bp["r"] is None:
        arith_dim.append(None)
        continue

    r = bp["r"]
    orbit = iterate_logistic(r + 0.001, n_transient=50000, n_sample=5000)
    fits = fit_models(orbit)

    print(f"\n  Bifurcation {i+1} (r={r:.8f}):")
    for model in ["linear", "power_law", "exponential"]:
        r2 = fits[model]["R2"]
        print(f"    {model:15s}: R^2 = {r2:.6f}")
    print(f"    => Dominant: {fits['dominant']}")

    arith_dim.append({
        "r": r,
        "linear_R2": fits["linear"]["R2"],
        "power_law_R2": fits["power_law"]["R2"],
        "exponential_R2": fits["exponential"]["R2"],
        "dominant": fits["dominant"]
    })

results["arithmetic_dimension"] = arith_dim

# ============================================================
# Bifurcation diagram with color coding
# ============================================================
print("\n" + "=" * 60)
print("GENERATING BIFURCATION DIAGRAM")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

# Top: bifurcation diagram
ax = axes[0]
r_range = np.linspace(2.5, 4.0, 2000)
for r in r_range:
    orbit = iterate_logistic(r, n_transient=1000, n_sample=200)

    # Color by dominant arithmetic dimension
    fits = fit_models(orbit)
    color_map = {"linear": "steelblue", "power_law": "darkorange", "exponential": "seagreen"}
    color = color_map.get(fits["dominant"], "gray")

    ax.plot([r]*len(orbit), orbit, ",", color=color, alpha=0.15, markersize=0.5)

# Mark bifurcation points
for bp in bifurcation_points:
    if bp["r"] is not None:
        ax.axvline(bp["r"], color="red", alpha=0.3, linewidth=0.5)

ax.set_xlim(2.5, 4.0)
ax.set_ylim(0, 1)
ax.set_ylabel("x", fontsize=12)
ax.set_title("Logistic Map Bifurcation Diagram\nColored by Dominant Arithmetic Level: "
             "Blue=Linear(Additive), Orange=Power-law(Multiplicative), Green=Exponential",
             fontsize=12)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="steelblue", label="Linear (Additive)"),
    Patch(facecolor="darkorange", label="Power-law (Multiplicative)"),
    Patch(facecolor="seagreen", label="Exponential"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

# Bottom: Lyapunov exponent
ax2 = axes[1]
r_lyap = np.linspace(2.5, 4.0, 1000)
lyap_vals = [lyapunov_exponent(r, n_iter=10000) for r in r_lyap]
ax2.plot(r_lyap, lyap_vals, "k-", linewidth=0.5, alpha=0.7)
ax2.axhline(0, color="red", linewidth=0.8, linestyle="--")
ax2.set_xlim(2.5, 4.0)
ax2.set_xlabel("r", fontsize=12)
ax2.set_ylabel("Lyapunov exp.", fontsize=12)
ax2.set_title("Lyapunov Exponent", fontsize=11)

for bp in bifurcation_points:
    if bp["r"] is not None:
        ax2.axvline(bp["r"], color="red", alpha=0.3, linewidth=0.5)

plt.tight_layout()
figpath = os.path.join(OUTPUT_DIR, "test4_feigenbaum.png")
plt.savefig(figpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to {figpath}")

# ============================================================
# Check systematic shift
# ============================================================
print("\n--- Arithmetic Dimension Shift Analysis ---")
dominant_sequence = [ad["dominant"] for ad in arith_dim if ad is not None]
print(f"  Dominant sequence across bifurcations: {dominant_sequence}")

# Count transitions
shift_analysis = {
    "dominant_sequence": dominant_sequence,
    "observation": "The logistic map's return map is inherently quadratic (power-law/multiplicative), "
                   "so power-law fits tend to dominate throughout the cascade. "
                   "Near bifurcation points, linearization (additive) becomes more relevant. "
                   "Deep in chaos, exponential sensitivity (positive Lyapunov exponent) emerges."
}
results["shift_analysis"] = shift_analysis
print(f"  {shift_analysis['observation']}")

# Save
outpath = os.path.join(OUTPUT_DIR, "test4_feigenbaum.json")

def make_serializable(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj

with open(outpath, "w") as f:
    json.dump(make_serializable(results), f, indent=2, default=str)
print(f"Data saved to {outpath}")
