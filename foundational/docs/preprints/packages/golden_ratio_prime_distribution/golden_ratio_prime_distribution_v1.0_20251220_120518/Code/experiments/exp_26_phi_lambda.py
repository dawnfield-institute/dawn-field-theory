"""
Experiment 26: Is the Run-Length Ratio = φλ?
============================================

DISCOVERY from exp_25:
  run_ratio = 1.601837
  φ * λ     = 1.601854

These match to 4 decimal places! This suggests:

  L+/L- = φλ

If true, then:
  frac(E>0) = L+/(L+ + L-) = φλ/(φλ + 1)

Let's verify this relationship and understand why it might be true.
"""

import numpy as np
import json
from datetime import datetime

# Parameters
N_MAX = 100_000
LAMBDA = 0.99
WINDOW = 101
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
K = len(PRIMES)
PHI = (1 + np.sqrt(5)) / 2

def compute_S(n, primes):
    if n == 0:
        return 0
    count = sum(1 for p in primes if n % p == 0)
    return count / len(primes)

def compute_all(N, primes, lam, window):
    S = np.array([compute_S(n, primes) for n in range(N + 1)])
    S_hat = np.convolve(S, np.ones(window)/window, mode='same')
    I = S_hat - S
    E = np.zeros(N + 1)
    for n in range(1, N + 1):
        E[n] = lam * E[n-1] + I[n]
    return S, S_hat, I, E

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

print("=" * 70)
print("EXPERIMENT 26: IS RUN-LENGTH RATIO = φλ?")
print("=" * 70)
print()

# Compute
S, S_hat, I, E = compute_all(N_MAX, PRIMES, LAMBDA, WINDOW)
odds = np.arange(1, N_MAX + 1, 2)
E_odd = E[odds]

pos_runs, neg_runs = compute_run_lengths(E, odds)
L_plus = np.mean(pos_runs)
L_minus = np.mean(neg_runs)
run_ratio = L_plus / L_minus

# The hypothesis
phi_lambda = PHI * LAMBDA

print("OBSERVED:")
print(f"  L+ = {L_plus:.6f}")
print(f"  L- = {L_minus:.6f}")
print(f"  Run ratio = {run_ratio:.6f}")
print()

print("HYPOTHESIS: run_ratio = φλ")
print(f"  φ = {PHI:.6f}")
print(f"  λ = {LAMBDA:.6f}")
print(f"  φλ = {phi_lambda:.6f}")
print(f"  Error from φλ: {abs(run_ratio - phi_lambda):.6f} ({100*abs(run_ratio - phi_lambda)/phi_lambda:.4f}%)")
print()

# Compare to pure φ
print("Compare to pure φ:")
print(f"  Error from φ: {abs(run_ratio - PHI):.6f} ({100*abs(run_ratio - PHI)/PHI:.4f}%)")
print()

# If run_ratio = φλ, then what is frac(E>0)?
frac_from_phi_lambda = phi_lambda / (phi_lambda + 1)
actual_frac = np.mean(E_odd > 0)

print("IMPLICATION FOR frac(E>0):")
print(f"  If L+/L- = φλ, then frac = φλ/(φλ+1)")
print(f"  Predicted: {frac_from_phi_lambda:.6f}")
print(f"  Actual:    {actual_frac:.6f}")
print(f"  Error:     {abs(actual_frac - frac_from_phi_lambda):.6f}")
print()

# Compare to 1/φ prediction
print("Compare to 1/φ:")
print(f"  1/φ = {1/PHI:.6f}")
print(f"  Error from 1/φ: {abs(actual_frac - 1/PHI):.6f}")
print()

# ===================================================================
# TEST WITH DIFFERENT λ VALUES
# ===================================================================
print("=" * 70)
print("TEST: Does run_ratio = φλ hold for different λ?")
print("=" * 70)
print()

lambda_values = [0.95, 0.97, 0.99, 0.995, 0.999]
results = []

for lam in lambda_values:
    _, _, _, E_test = compute_all(N_MAX, PRIMES, lam, WINDOW)
    pos, neg = compute_run_lengths(E_test, odds)
    if len(pos) > 0 and len(neg) > 0:
        L_p = np.mean(pos)
        L_n = np.mean(neg)
        ratio = L_p / L_n
        predicted = PHI * lam
        error = abs(ratio - predicted)
        frac = np.mean(E_test[odds] > 0)
        
        results.append({
            'lambda': lam,
            'ratio': ratio,
            'phi_lambda': predicted,
            'error': error,
            'frac': frac
        })
        
        print(f"λ = {lam}:")
        print(f"  Run ratio = {ratio:.4f}")
        print(f"  φλ = {predicted:.4f}")
        print(f"  Error = {error:.4f} ({100*error/predicted:.2f}%)")
        print(f"  frac(E>0) = {frac:.4f}")
        print()

# ===================================================================
# ANALYTICAL INVESTIGATION
# ===================================================================
print("=" * 70)
print("WHY MIGHT L+/L- = φλ?")
print("=" * 70)
print()

print("""
If L+/L- = φλ, then:

  frac = φλ/(φλ + 1)

For λ = 0.99:
  frac = 1.602/(1.602 + 1) = 1.602/2.602 = 0.6156

For λ → 1:
  frac → φ/(φ + 1) = 1/φ²  (since φ+1 = φ²)
  = 0.382...

Wait, that's NOT 1/φ = 0.618!

Let me recalculate...
""")

# Check the algebra
print("Algebra check:")
print(f"  φ + 1 = {PHI + 1:.6f}")
print(f"  φ² = {PHI**2:.6f}")
print(f"  Indeed φ + 1 = φ² (golden ratio property)")
print()
print(f"  So if L+/L- = φ, then frac = φ/(φ+1) = φ/φ² = 1/φ = {1/PHI:.6f}")
print(f"  But if L+/L- = φλ, then frac = φλ/(φλ+1)")
print()

# What's the relationship?
print("For various λ:")
for lam in [0.99, 0.999, 1.0]:
    pl = PHI * lam
    frac_pred = pl / (pl + 1)
    print(f"  λ = {lam}: frac = {frac_pred:.6f}")

print()
print(f"Observation: frac → 1/φ only as λ → 1")
print(f"At λ = 0.99, frac should be {PHI*0.99/(PHI*0.99+1):.6f} if L+/L- = φλ")
print(f"Actual frac = {actual_frac:.6f}")
print()

# ===================================================================
# THE KEY TEST
# ===================================================================
print("=" * 70)
print("KEY TEST: Which model fits better?")
print("=" * 70)
print()

# Model A: L+/L- = φ (independent of λ)
# Model B: L+/L- = φλ (depends on λ)

print("Model A: L+/L- = φ")
print("  Predicts: frac = φ/(φ+1) = 1/φ = 0.6180 for all λ")
print()

print("Model B: L+/L- = φλ")
print("  Predicts: frac = φλ/(φλ+1) which varies with λ")
print()

print("Observed at λ=0.99:")
print(f"  frac = {actual_frac:.6f}")
print(f"  Model A prediction (1/φ): {1/PHI:.6f}, error = {abs(actual_frac - 1/PHI):.6f}")
print(f"  Model B prediction (φλ/(φλ+1)): {frac_from_phi_lambda:.6f}, error = {abs(actual_frac - frac_from_phi_lambda):.6f}")
print()

# The winner
if abs(actual_frac - frac_from_phi_lambda) < abs(actual_frac - 1/PHI):
    print(">>> Model B (L+/L- = φλ) fits better!")
else:
    print(">>> Model A (L+/L- = φ) fits better!")

print()

# Save results
results_dict = {
    "experiment": "exp_26_phi_lambda",
    "timestamp": datetime.now().isoformat(),
    "hypothesis": "run_ratio = phi * lambda",
    "parameters": {"N_MAX": N_MAX, "LAMBDA": LAMBDA, "K": K},
    "observed": {
        "L_plus": float(L_plus),
        "L_minus": float(L_minus),
        "run_ratio": float(run_ratio),
        "frac": float(actual_frac)
    },
    "predictions": {
        "phi": float(PHI),
        "phi_lambda": float(phi_lambda),
        "error_from_phi": float(abs(run_ratio - PHI)),
        "error_from_phi_lambda": float(abs(run_ratio - phi_lambda))
    },
    "lambda_sweep": results
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"exp_26_phi_lambda_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(results_dict, f, indent=2, default=float)

print(f"Results saved: {filename}")
