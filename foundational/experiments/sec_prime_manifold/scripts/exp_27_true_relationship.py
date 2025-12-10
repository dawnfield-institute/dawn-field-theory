"""
Experiment 27: Finding the True Relationship
============================================

exp_26 showed L+/L- = φλ works perfectly at λ=0.99, but fails at other λ.
This suggests either:
  A) λ=0.99 is special (coincidence)
  B) The true relationship is more complex

Let's find what actually varies with λ.
"""

import numpy as np
import json
from datetime import datetime

# Parameters
N_MAX = 100_000
WINDOW = 101
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
K = len(PRIMES)
PHI = (1 + np.sqrt(5)) / 2
XI = 1 / (2 * K)  # The 2-component amplitude

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
print("EXPERIMENT 27: FINDING THE TRUE RELATIONSHIP")
print("=" * 70)
print()

print(f"Fixed parameters:")
print(f"  k = {K}")
print(f"  ξ = 1/(2k) = {XI:.6f}")
print(f"  φ = {PHI:.6f}")
print(f"  window = {WINDOW}")
print()

# Sweep over λ
lambda_values = [0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
odds = np.arange(1, N_MAX + 1, 2)

data = []

print("=" * 70)
print("DATA COLLECTION")
print("=" * 70)
print()
print(f"{'λ':>6} {'L+':>8} {'L-':>8} {'ratio':>8} {'frac':>8}")
print("-" * 45)

for lam in lambda_values:
    _, _, I, E = compute_all(N_MAX, PRIMES, lam, WINDOW)
    pos, neg = compute_run_lengths(E, odds)
    
    if len(pos) > 0 and len(neg) > 0:
        L_plus = np.mean(pos)
        L_minus = np.mean(neg)
        ratio = L_plus / L_minus
        frac = np.mean(E[odds] > 0)
        
        data.append({
            'lambda': lam,
            'L_plus': L_plus,
            'L_minus': L_minus,
            'ratio': ratio,
            'frac': frac
        })
        
        print(f"{lam:>6.3f} {L_plus:>8.3f} {L_minus:>8.3f} {ratio:>8.4f} {frac:>8.4f}")

print()
print("=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)
print()

# What varies how with λ?
lambdas = np.array([d['lambda'] for d in data])
ratios = np.array([d['ratio'] for d in data])
fracs = np.array([d['frac'] for d in data])
L_pluses = np.array([d['L_plus'] for d in data])
L_minuses = np.array([d['L_minus'] for d in data])

print("Testing relationships:")
print()

# Model 1: ratio = φ * λ
pred1 = PHI * lambdas
err1 = np.abs(ratios - pred1)
print(f"Model 1: ratio = φλ")
print(f"  Errors: {[f'{e:.4f}' for e in err1]}")
print(f"  Mean error: {np.mean(err1):.4f}")
print()

# Model 2: ratio = φ * (something else)
# What if it's φ * (1 - (1-λ)^something)?
print("Model 2: ratio = φ * f(λ)")
print("  If ratio = φ * f(λ), then f(λ) = ratio/φ:")
f_lambda = ratios / PHI
for lam, f in zip(lambdas, f_lambda):
    print(f"    λ = {lam:.3f}: f(λ) = {f:.4f}")
print()

# Is f(λ) related to 1/(1-λ) or something similar?
print("Checking if f(λ) relates to 1-λ:")
one_minus_lambda = 1 - lambdas
for lam, f, oml in zip(lambdas, f_lambda, one_minus_lambda):
    print(f"    λ = {lam:.3f}: f = {f:.4f}, 1-λ = {oml:.4f}, f*(1-λ) = {f*oml:.6f}")
print()

# Model 3: What about the actual observed frac?
# frac = 1/φ would mean ratio = φ
# But frac varies with λ
print("Observed frac vs predictions:")
print()
print(f"{'λ':>6} {'frac':>8} {'1/φ':>8} {'φλ/(φλ+1)':>10}")
print("-" * 40)
for lam, frac in zip(lambdas, fracs):
    phi_lam = PHI * lam
    frac_pred = phi_lam / (phi_lam + 1)
    print(f"{lam:>6.3f} {frac:>8.4f} {1/PHI:>8.4f} {frac_pred:>10.4f}")

print()
print("=" * 70)
print("INSIGHT: The relationship might not be about λ alone")
print("=" * 70)
print()

# What if the key is the effective time scale?
# E-folding time = 1/(1-λ)
# At λ=0.99, E-folding = 100

print("E-folding analysis:")
print()
for lam, ratio, frac in zip(lambdas, ratios, fracs):
    e_fold = 1 / (1 - lam)
    # What's the "natural" unit? 
    # The prime gap on odds is about 5.2
    prime_gap = 5.2
    ratio_to_gap = e_fold / prime_gap
    print(f"  λ = {lam:.3f}: E-fold = {e_fold:>6.1f}, ratio_to_gap = {ratio_to_gap:.1f}, run_ratio = {ratio:.3f}, frac = {frac:.4f}")

print()

# What if λ=0.99 is just the sweet spot where things align?
print("=" * 70)
print("HYPOTHESIS: λ=0.99 is a resonance point")
print("=" * 70)
print()

# At λ=0.99, E-folding time = 100
# Window = 101
# These are almost equal!
print(f"At λ = 0.99:")
print(f"  E-folding time = {1/0.01:.0f}")
print(f"  Window size = {WINDOW}")
print(f"  Ratio = {100/WINDOW:.3f} (almost 1!)")
print()
print("This might explain why λ=0.99 gives the cleanest φ signal!")
print()

# Let's test: what λ would make E-folding = window exactly?
lambda_resonant = 1 - 1/WINDOW
print(f"Resonant λ (where E-fold = window): {lambda_resonant:.6f}")
print(f"We tested λ = 0.99, which gives E-fold = {1/0.01:.0f}")
print()

# Test at the exact resonant λ
print("Testing at resonant λ:")
_, _, I, E = compute_all(N_MAX, PRIMES, lambda_resonant, WINDOW)
pos, neg = compute_run_lengths(E, odds)
L_plus_res = np.mean(pos)
L_minus_res = np.mean(neg)
ratio_res = L_plus_res / L_minus_res
frac_res = np.mean(E[odds] > 0)

print(f"  λ = {lambda_resonant:.6f}")
print(f"  L+ = {L_plus_res:.4f}, L- = {L_minus_res:.4f}")
print(f"  Run ratio = {ratio_res:.6f}")
print(f"  φ = {PHI:.6f}")
print(f"  Error from φ: {abs(ratio_res - PHI):.6f} ({100*abs(ratio_res - PHI)/PHI:.4f}%)")
print(f"  frac(E>0) = {frac_res:.6f}")
print(f"  1/φ = {1/PHI:.6f}")
print(f"  Error from 1/φ: {abs(frac_res - 1/PHI):.6f}")
print()

# Save
results = {
    "experiment": "exp_27_true_relationship",
    "timestamp": datetime.now().isoformat(),
    "parameters": {"N_MAX": N_MAX, "K": K, "XI": XI, "WINDOW": WINDOW},
    "lambda_sweep": data,
    "resonant_lambda": {
        "value": float(lambda_resonant),
        "L_plus": float(L_plus_res),
        "L_minus": float(L_minus_res),
        "ratio": float(ratio_res),
        "frac": float(frac_res)
    }
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"exp_27_true_relationship_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {filename}")
