"""
Experiment 28: Finding the Optimal λ
====================================

exp_27 revealed that frac(E>0) is closest to 1/φ around λ ≈ 0.98,
NOT as λ → 1. Let's find the exact optimal λ and understand why.
"""

import numpy as np
import json
from datetime import datetime
from scipy.optimize import minimize_scalar

# Parameters
N_MAX = 100_000
WINDOW = 101
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
K = len(PRIMES)
PHI = (1 + np.sqrt(5)) / 2
XI = 1 / (2 * K)

def compute_S(n, primes):
    if n == 0:
        return 0
    count = sum(1 for p in primes if n % p == 0)
    return count / len(primes)

# Precompute S for efficiency
S_precomputed = np.array([compute_S(n, PRIMES) for n in range(N_MAX + 1)])
S_hat_precomputed = np.convolve(S_precomputed, np.ones(WINDOW)/WINDOW, mode='same')
I_precomputed = S_hat_precomputed - S_precomputed

def compute_E(lam):
    """Compute E given precomputed I."""
    E = np.zeros(N_MAX + 1)
    for n in range(1, N_MAX + 1):
        E[n] = lam * E[n-1] + I_precomputed[n]
    return E

def compute_frac(lam):
    """Compute frac(E>0) on odds."""
    E = compute_E(lam)
    odds = np.arange(1, N_MAX + 1, 2)
    return np.mean(E[odds] > 0)

def error_from_phi_inv(lam):
    """Error from 1/φ."""
    return abs(compute_frac(lam) - 1/PHI)

print("=" * 70)
print("EXPERIMENT 28: FINDING THE OPTIMAL λ")
print("=" * 70)
print()

# Fine sweep to find optimal
print("Fine sweep around λ ≈ 0.98:")
print()
print(f"{'λ':>8} {'frac':>10} {'error from 1/φ':>15}")
print("-" * 40)

lambda_fine = np.linspace(0.95, 0.995, 19)
results = []

for lam in lambda_fine:
    frac = compute_frac(lam)
    err = abs(frac - 1/PHI)
    results.append({'lambda': lam, 'frac': frac, 'error': err})
    print(f"{lam:>8.4f} {frac:>10.6f} {err:>15.6f}")

# Find minimum
min_result = min(results, key=lambda x: x['error'])
print()
print(f"Optimal λ in sweep: {min_result['lambda']:.4f}")
print(f"  frac = {min_result['frac']:.6f}")
print(f"  error = {min_result['error']:.6f}")
print()

# Use optimization to find exact minimum
print("=" * 70)
print("OPTIMIZATION: Finding exact optimal λ")
print("=" * 70)
print()

result = minimize_scalar(error_from_phi_inv, bounds=(0.9, 0.999), method='bounded')
lambda_opt = result.x
frac_opt = compute_frac(lambda_opt)
error_opt = error_from_phi_inv(lambda_opt)

print(f"Optimal λ = {lambda_opt:.6f}")
print(f"frac(E>0) at optimal = {frac_opt:.6f}")
print(f"1/φ = {1/PHI:.6f}")
print(f"Error at optimal = {error_opt:.6f}")
print()

# What's special about this λ?
print("=" * 70)
print("WHAT'S SPECIAL ABOUT THE OPTIMAL λ?")
print("=" * 70)
print()

e_fold_opt = 1 / (1 - lambda_opt)
print(f"E-folding time at optimal λ: {e_fold_opt:.2f}")
print(f"Window size: {WINDOW}")
print(f"Ratio E-fold/window: {e_fold_opt/WINDOW:.4f}")
print()

# Check relationships
print("Checking relationships:")
print(f"  1 - λ_opt = {1 - lambda_opt:.6f}")
print(f"  1/k = {1/K:.6f}")
print(f"  ξ = 1/(2k) = {XI:.6f}")
print(f"  1/(window) = {1/WINDOW:.6f}")
print()

# Is λ_opt related to window?
print(f"  λ_opt vs 1 - 1/window = {1 - 1/WINDOW:.6f}")
print(f"  λ_opt vs 1 - 2/window = {1 - 2/WINDOW:.6f}")
print()

# Check if optimal varies with window
print("=" * 70)
print("DOES OPTIMAL λ DEPEND ON WINDOW?")
print("=" * 70)
print()

windows_to_test = [51, 101, 151, 201]
optimal_lambdas = []

for win in windows_to_test:
    # Recompute S_hat and I for this window
    S_hat_w = np.convolve(S_precomputed, np.ones(win)/win, mode='same')
    I_w = S_hat_w - S_precomputed
    
    def compute_E_w(lam):
        E = np.zeros(N_MAX + 1)
        for n in range(1, N_MAX + 1):
            E[n] = lam * E[n-1] + I_w[n]
        return E
    
    def compute_frac_w(lam):
        E = compute_E_w(lam)
        odds = np.arange(1, N_MAX + 1, 2)
        return np.mean(E[odds] > 0)
    
    def error_w(lam):
        return abs(compute_frac_w(lam) - 1/PHI)
    
    res = minimize_scalar(error_w, bounds=(0.9, 0.999), method='bounded')
    opt_lam = res.x
    opt_frac = compute_frac_w(opt_lam)
    opt_err = error_w(opt_lam)
    e_fold = 1 / (1 - opt_lam)
    
    optimal_lambdas.append({
        'window': win,
        'lambda_opt': opt_lam,
        'frac': opt_frac,
        'error': opt_err,
        'e_fold': e_fold,
        'e_fold_over_window': e_fold / win
    })
    
    print(f"Window = {win}:")
    print(f"  Optimal λ = {opt_lam:.6f}")
    print(f"  E-fold = {e_fold:.2f}")
    print(f"  E-fold/window = {e_fold/win:.4f}")
    print(f"  frac = {opt_frac:.6f}, error = {opt_err:.6f}")
    print()

# Look for pattern
print("=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)
print()

print("E-fold/window ratios:")
for r in optimal_lambdas:
    print(f"  Window {r['window']}: E-fold/window = {r['e_fold_over_window']:.4f}")

print()

# Is there a formula?
# E-fold_opt / window = constant?
ratios = [r['e_fold_over_window'] for r in optimal_lambdas]
print(f"Mean ratio: {np.mean(ratios):.4f}")
print(f"Std ratio: {np.std(ratios):.4f}")
print()

# Check if λ_opt ≈ 1 - c/window for some constant c
print("Checking λ_opt = 1 - c/window:")
c_values = [(1 - r['lambda_opt']) * r['window'] for r in optimal_lambdas]
for r, c in zip(optimal_lambdas, c_values):
    print(f"  Window {r['window']}: c = {c:.4f}")
print(f"Mean c: {np.mean(c_values):.4f}")
print()

# ===================================================================
# THE KEY INSIGHT
# ===================================================================
print("=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print()

mean_c = np.mean(c_values)
print(f"""
The optimal λ appears to follow:

  λ_opt ≈ 1 - c/window

where c ≈ {mean_c:.2f}

This means the optimal E-folding time is about {mean_c:.1f}× the window size.

Physical interpretation:
- The window averages over ~w steps
- The optimal memory (E-fold) is ~{mean_c:.1f}w steps
- This balance gives frac(E>0) closest to 1/φ

At window = 101, λ_opt = 1 - {mean_c:.2f}/101 ≈ {1 - mean_c/101:.4f}
(Actual optimal: {lambda_opt:.4f})
""")

# ===================================================================
# WHY 1/φ AT THE OPTIMAL?
# ===================================================================
print("=" * 70)
print("WHY 1/φ AT THE OPTIMAL?")
print("=" * 70)
print()

# Compute run lengths at optimal
E_opt = compute_E(lambda_opt)
odds = np.arange(1, N_MAX + 1, 2)

def compute_run_lengths(E, indices):
    signs = np.sign(E[indices])
    positive_runs, negative_runs = [], []
    current_sign = signs[0]
    current_length = 1
    
    for i in range(1, len(signs)):
        if signs[i] == current_sign:
            current_length += 1
        else:
            if current_sign > 0:
                positive_runs.append(current_length)
            elif current_sign < 0:
                negative_runs.append(current_length)
            current_sign = signs[i]
            current_length = 1
    
    if current_sign > 0:
        positive_runs.append(current_length)
    elif current_sign < 0:
        negative_runs.append(current_length)
    
    return positive_runs, negative_runs

pos_runs, neg_runs = compute_run_lengths(E_opt, odds)
L_plus = np.mean(pos_runs)
L_minus = np.mean(neg_runs)
ratio = L_plus / L_minus

print(f"At optimal λ = {lambda_opt:.4f}:")
print(f"  L+ = {L_plus:.4f}")
print(f"  L- = {L_minus:.4f}")
print(f"  Run ratio = {ratio:.4f}")
print(f"  φ = {PHI:.4f}")
print(f"  Error from φ: {abs(ratio - PHI):.4f} ({100*abs(ratio - PHI)/PHI:.2f}%)")
print()
print(f"  frac from runs = {L_plus/(L_plus + L_minus):.6f}")
print(f"  1/φ = {1/PHI:.6f}")
print()

# Save results
output = {
    "experiment": "exp_28_optimal_lambda",
    "timestamp": datetime.now().isoformat(),
    "optimal_lambda": float(lambda_opt),
    "frac_at_optimal": float(frac_opt),
    "error_at_optimal": float(error_opt),
    "e_fold_at_optimal": float(e_fold_opt),
    "window_dependence": optimal_lambdas,
    "formula": f"λ_opt ≈ 1 - {mean_c:.2f}/window",
    "run_analysis_at_optimal": {
        "L_plus": float(L_plus),
        "L_minus": float(L_minus),
        "ratio": float(ratio),
        "error_from_phi": float(abs(ratio - PHI))
    }
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"exp_28_optimal_lambda_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {filename}")
