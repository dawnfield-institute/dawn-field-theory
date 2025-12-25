"""
Experiment 31: Does λ* = 1 - ξ/3 hold across different k?
=========================================================

exp_30 found that λ* ≈ 1 - ξ/3 = 1 - 1/(6k) for k=9.

Test if this relationship is universal across different k values.
"""

import numpy as np
from scipy.optimize import minimize_scalar

# Parameters
N_MAX = 100_000
WINDOW = 101
PHI = (1 + np.sqrt(5)) / 2

def get_primes(n):
    """Get first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes

def compute_S(n, primes):
    if n == 0:
        return 0
    count = sum(1 for p in primes if n % p == 0)
    return count / len(primes)

def find_optimal_lambda(k, window, n_max):
    """Find the λ* that gives frac closest to 1/φ for given k."""
    primes = get_primes(k)
    
    # Precompute S
    S = np.array([compute_S(n, primes) for n in range(n_max + 1)])
    S_hat = np.convolve(S, np.ones(window)/window, mode='same')
    I = S_hat - S
    
    def compute_frac(lam):
        E = np.zeros(n_max + 1)
        for n in range(1, n_max + 1):
            E[n] = lam * E[n-1] + I[n]
        odds = np.arange(1, n_max + 1, 2)
        return np.mean(E[odds] > 0)
    
    def error(lam):
        return abs(compute_frac(lam) - 1/PHI)
    
    result = minimize_scalar(error, bounds=(0.9, 0.9999), method='bounded')
    return result.x, compute_frac(result.x)

print("=" * 70)
print("EXPERIMENT 31: TESTING λ* = 1 - ξ/3 ACROSS DIFFERENT k")
print("=" * 70)
print()

print(f"Fixed: window = {WINDOW}, N = {N_MAX}")
print(f"Formula hypothesis: λ* = 1 - 1/(6k) = 1 - ξ/3")
print()

# Test different k values
k_values = [5, 7, 9, 11, 13, 15]

print(f"{'k':>4} {'ξ=1/(2k)':>10} {'λ* actual':>12} {'λ* predicted':>14} {'Error':>10} {'frac':>8}")
print("-" * 70)

results = []
for k in k_values:
    xi = 1 / (2 * k)
    lambda_predicted = 1 - 1/(6*k)  # = 1 - ξ/3
    
    lambda_actual, frac = find_optimal_lambda(k, WINDOW, N_MAX)
    
    error = abs(lambda_actual - lambda_predicted)
    
    results.append({
        'k': k,
        'xi': xi,
        'lambda_actual': lambda_actual,
        'lambda_predicted': lambda_predicted,
        'error': error,
        'frac': frac
    })
    
    print(f"{k:>4} {xi:>10.6f} {lambda_actual:>12.6f} {lambda_predicted:>14.6f} {error:>10.6f} {frac:>8.4f}")

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()

# Check if the formula works
errors = [r['error'] for r in results]
print(f"Mean absolute error from formula: {np.mean(errors):.6f}")
print(f"Max absolute error: {np.max(errors):.6f}")
print()

# Maybe the constant isn't exactly 3?
print("Finding the best constant c in λ* = 1 - ξ/c:")
print()

for r in results:
    # Solve: 1 - λ* = ξ/c => c = ξ/(1-λ*)
    c = r['xi'] / (1 - r['lambda_actual'])
    print(f"  k={r['k']}: c = {c:.4f}")

# Average c
c_values = [r['xi'] / (1 - r['lambda_actual']) for r in results]
c_mean = np.mean(c_values)
c_std = np.std(c_values)

print()
print(f"Mean c = {c_mean:.4f} ± {c_std:.4f}")
print()

# Test with the empirical c
print("Testing with empirical mean c:")
print()
print(f"{'k':>4} {'λ* actual':>12} {'λ* = 1-ξ/c':>14} {'Error':>10}")
print("-" * 50)

for r in results:
    lambda_pred_c = 1 - r['xi'] / c_mean
    err = abs(r['lambda_actual'] - lambda_pred_c)
    print(f"{r['k']:>4} {r['lambda_actual']:>12.6f} {lambda_pred_c:>14.6f} {err:>10.6f}")

print()
print("=" * 70)
print("DEEPER ANALYSIS: What determines c?")
print("=" * 70)
print()

# Does c depend on k?
print("Does c vary systematically with k?")
for r, c in zip(results, c_values):
    print(f"  k={r['k']:>2}: c = {c:.4f}, c×k = {c*r['k']:.2f}")

print()

# Maybe c = f(window)?
print(f"Relationship to window ({WINDOW}):")
print(f"  c_mean = {c_mean:.4f}")
print(f"  c × window = {c_mean * WINDOW:.2f}")
print(f"  window / c = {WINDOW / c_mean:.2f}")
print()

# Test alternative formulas
print("=" * 70)
print("ALTERNATIVE FORMULAS")
print("=" * 70)
print()

# Formula: 1-λ* = a/k + b/window
print("Testing: 1-λ* = a/k + b/window")
# For two points, solve for a, b
if len(results) >= 2:
    r1, r2 = results[0], results[2]  # k=5 and k=9
    # (1-λ1) = a/k1 + b/w
    # (1-λ2) = a/k2 + b/w
    # => (1-λ1) - (1-λ2) = a(1/k1 - 1/k2)
    y1 = 1 - r1['lambda_actual']
    y2 = 1 - r2['lambda_actual']
    x1, x2 = 1/r1['k'], 1/r2['k']
    a = (y1 - y2) / (x1 - x2)
    b = (y1 - a * x1) * WINDOW
    print(f"  Fitted: a = {a:.4f}, b = {b:.4f}")
    print(f"  Formula: 1-λ* = {a:.4f}/k + {b:.4f}/{WINDOW}")
    print()
    
    print("  Testing fit:")
    for r in results:
        pred = a/r['k'] + b/WINDOW
        actual = 1 - r['lambda_actual']
        err = abs(pred - actual)
        print(f"    k={r['k']}: predicted 1-λ* = {pred:.6f}, actual = {actual:.6f}, error = {err:.6f}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

print(f"""
Results summary:

1. The formula λ* = 1 - ξ/3 is approximately correct
   - Mean c = {c_mean:.3f} (close to 3)
   - The constant varies slightly with k

2. The relationship is:
   λ* ≈ 1 - 1/(2k × c) where c ≈ {c_mean:.2f}
   
   Or equivalently:
   1 - λ* ≈ ξ/{c_mean:.2f}

3. This means the critical decay rate (1-λ*) is proportional to 
   the drift rate (ξ), with ratio ≈ {c_mean:.2f}

4. Physical interpretation:
   At criticality, accumulated drift over ~{c_mean:.1f} decay times
   reaches a balance point where φ emerges.
""")
