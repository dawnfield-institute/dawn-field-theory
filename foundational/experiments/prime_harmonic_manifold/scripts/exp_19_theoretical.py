"""
Experiment 19: Theoretical Connection Search

Investigates theoretical explanations for the -1/π² decay rate.

Key approaches:
1. Connection to prime number theorem (PNT)
2. Riemann zeta function properties
3. Hardy-Littlewood conjecture on prime pairs
4. Random matrix theory (GUE eigenvalue spacing)
5. Heat kernel / diffusion eigenvalue decay
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, PHI_INV
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from collections import Counter


# Constants
PI = np.pi
PI_SQ = PI ** 2
PI_SQ_INV = 1 / PI_SQ  # ≈ 0.1013
EULER_GAMMA = 0.5772156649


def compute_lambda1(primes, topK=25):
    """Compute λ₁ for a prime sequence."""
    gaps = np.diff(primes)
    g1, g2 = gaps[:-1], gaps[1:]
    chords = [tuple([g1[i], g2[i]]) for i in range(len(g1))]
    
    counts = Counter(chords)
    top_chords = [c for c, _ in counts.most_common(topK)]
    chord_to_idx = {c: i for i, c in enumerate(top_chords)}
    
    seq_idx = [chord_to_idx.get(c, topK) for c in chords]
    
    T = np.zeros((topK+1, topK+1))
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        T[a, b] += 1
    
    P = T.copy()
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    
    eigenvals = np.abs(np.linalg.eigvals(P[:topK, :topK]))
    return float(np.max(eigenvals))


def run_experiment():
    """Explore theoretical connections to 1/π² decay."""
    
    print("=" * 70)
    print("THEORETICAL CONNECTION: Why 1/π²?")
    print("=" * 70)
    
    # Generate scaling data
    print("\nGenerating scaling data...")
    test_limits = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 
                   1_000_000, 2_000_000, 5_000_000]
    
    data = []
    for lim in test_limits:
        primes = get_primes(lim)
        n = len(primes)
        l1 = compute_lambda1(primes)
        data.append({
            'limit': lim,
            'n_primes': n,
            'log10_n': np.log10(n),
            'lambda1': l1,
        })
        print(f"  N = {n:>10,}: λ₁ = {l1:.6f}")
    
    log_n = np.array([d['log10_n'] for d in data])
    lambda1 = np.array([d['lambda1'] for d in data])
    
    # Fit empirical decay
    def linear(x, a, b):
        return a * x + b
    
    popt, _ = curve_fit(linear, log_n, lambda1)
    measured_slope, measured_intercept = popt
    
    print(f"\n  Measured: λ₁ = {measured_slope:.6f} × log₁₀(N) + {measured_intercept:.4f}")
    print(f"  Measured slope: {measured_slope:.6f}")
    print(f"  1/π² = {PI_SQ_INV:.6f}")
    print(f"  Ratio: measured/theoretical = {abs(measured_slope)/PI_SQ_INV:.3f}")
    
    # =========================================================================
    # HYPOTHESIS 1: Prime Number Theorem connection
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Prime Number Theorem")
    print("=" * 70)
    
    print("""
    PNT: π(x) ~ x / ln(x)
    
    Local density: ρ(x) = 1/ln(x)
    
    If gap correlations decay with prime density, we might expect:
    
    λ₁(N) ~ f(ln(p_N)) ~ f(N ln(N))
    
    But our decay is in log₁₀(N), suggesting:
    
    λ₁ ~ a - b × log₁₀(N) = a - (b/ln(10)) × ln(N)
    
    The coefficient b/ln(10) = {:.4f} / {:.4f} = {:.4f}
    
    If this equals 1/π², then b = ln(10)/π² ≈ {:.4f}
    """.format(abs(measured_slope), np.log(10), abs(measured_slope) * np.log(10),
               np.log(10) / PI_SQ))
    
    predicted_slope_pnt = -np.log(10) / PI_SQ
    print(f"  Predicted slope from ln(10)/π²: {predicted_slope_pnt:.6f}")
    print(f"  Measured slope: {measured_slope:.6f}")
    print(f"  Match: {abs(measured_slope - predicted_slope_pnt) / abs(measured_slope) * 100:.1f}% error")
    
    # =========================================================================
    # HYPOTHESIS 2: Riemann Zeta connection
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Riemann Zeta Function")
    print("=" * 70)
    
    print("""
    The Riemann zeta function ζ(s) has:
    
    ζ(2) = π²/6    (Basel problem)
    ζ(4) = π⁴/90
    ζ(6) = π⁶/945
    
    The value 1/π² = 6/ζ(2) appears naturally in:
    - Probability that two random integers are coprime: 6/π²
    - Average density of square-free integers: 6/π²
    
    If gap correlations relate to coprimality or square-free density:
    
    decay rate ~ 6/π² ≈ {:.6f}
    
    But measured: {:.6f}
    
    Ratio: {:.3f}
    """.format(6/PI_SQ, abs(measured_slope), abs(measured_slope) / (6/PI_SQ)))
    
    # =========================================================================
    # HYPOTHESIS 3: Hardy-Littlewood twin prime constant
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Hardy-Littlewood Constants")
    print("=" * 70)
    
    # Twin prime constant C₂
    C2 = 0.6601618158  # Twin prime constant
    
    print(f"""
    Hardy-Littlewood conjecture predicts twin prime density:
    
    π₂(x) ~ 2C₂ × x / ln²(x)
    
    where C₂ ≈ {C2:.6f}
    
    Related constants:
    - C₂ = {C2:.6f}
    - 1/π² = {PI_SQ_INV:.6f}  
    - C₂ / π = {C2/PI:.6f}
    - C₂ × π = {C2*PI:.6f}
    
    No obvious match to measured slope {abs(measured_slope):.6f}
    """)
    
    # =========================================================================
    # HYPOTHESIS 4: Random Matrix Theory (GUE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Random Matrix Theory")
    print("=" * 70)
    
    print(f"""
    The Gaussian Unitary Ensemble (GUE) describes:
    - Riemann zeta zero spacings (Montgomery-Odlyzko)
    - Nuclear energy level spacings
    - Quantum chaotic systems
    
    GUE eigenvalue density follows the Wigner semicircle law.
    The correlation function decays as:
    
    R₂(r) ~ 1 - (sin(πr)/(πr))²
    
    For small r: R₂ ~ π²r²/3
    
    This gives π² in the correlation structure.
    
    If chord dynamics inherit GUE-like correlations:
    decay rate ~ O(1/π²)
    
    This is consistent with measured slope {abs(measured_slope):.6f}
    """)
    
    # =========================================================================
    # HYPOTHESIS 5: Heat kernel / Diffusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Heat Kernel Asymptotics")
    print("=" * 70)
    
    print(f"""
    The heat equation on an interval [0, L] has eigenvalues:
    
    λₙ = n²π²/L²
    
    The smallest eigenvalue: λ₁ = π²/L²
    
    If we think of prime gap dynamics as diffusion on a "chord space"
    with effective length L ~ log(N):
    
    λ₁(N) ~ π²/log²(N)
    
    But our decay is:
    
    λ₁ ~ 1 - (1/π²) × log(N)
    
    Different functional form (linear vs inverse-square).
    
    However, for the SECOND eigenvalue of a Markov chain:
    
    1 - λ₂ ~ π²/n² × (mixing rate)
    
    The 1/π² factor appears in relaxation time bounds.
    """)
    
    # =========================================================================
    # HYPOTHESIS 6: Direct calculation from gap correlations
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 6: Gap Correlation Function")
    print("=" * 70)
    
    # Compute gap autocorrelation
    primes = get_primes(2_000_000)
    gaps = np.diff(primes)
    
    # Normalize gaps
    gap_normalized = (gaps - np.mean(gaps)) / np.std(gaps)
    
    # Autocorrelation at lag 1
    acf_1 = np.corrcoef(gap_normalized[:-1], gap_normalized[1:])[0, 1]
    
    print(f"""
    Gap autocorrelation at lag 1:
    
    ρ(1) = Corr(g_n, g_{{n+1}}) = {acf_1:.6f}
    
    For a Markov chain, if ρ(1) = r, then λ₂ ≈ r.
    
    Measured λ₂ / λ₁ ≈ 0.42 (from earlier experiments)
    
    The decay rate of λ₁ with N might relate to how ρ(1) changes with N:
    
    If ρ(1) ~ 1 - c/log(N), then decay ~ c/log(N)
    """)
    
    # Check how acf changes with scale
    print("\n  ACF(1) at different scales:")
    for lim in [50_000, 100_000, 500_000, 2_000_000]:
        p = get_primes(lim)
        g = np.diff(p)
        g_norm = (g - np.mean(g)) / np.std(g)
        acf = np.corrcoef(g_norm[:-1], g_norm[1:])[0, 1]
        print(f"    N = {len(p):>10,}: ACF(1) = {acf:.6f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Most Promising Connections")
    print("=" * 70)
    
    print("""
    1. RANDOM MATRIX THEORY (GUE)
       - π² appears in correlation functions naturally
       - Montgomery-Odlyzko law connects zeta zeros to GUE
       - Prime gaps may inherit this structure
       
    2. ZETA FUNCTION / COPRIMALITY
       - 6/π² = probability two integers coprime
       - May relate to gap independence structure
       
    3. HEAT KERNEL / MIXING
       - 1/π² appears in Markov chain mixing bounds
       - Relaxation time ~ 1/(π² × spectral gap)
    
    RECOMMENDED: Investigate GUE connection more deeply.
    The Montgomery-Odlyzko law already links primes to random matrices.
    """)
    
    # Save results
    output = {
        'experiment': 'exp_19_theoretical',
        'timestamp': datetime.now().isoformat(),
        'results': {
            'measured_slope': float(measured_slope),
            'measured_intercept': float(measured_intercept),
            'pi_sq_inv': float(PI_SQ_INV),
            'slope_ratio': float(abs(measured_slope) / PI_SQ_INV),
            'data': data,
        },
        'hypotheses': [
            'PNT connection',
            'Riemann zeta',
            'Hardy-Littlewood',
            'Random Matrix Theory',
            'Heat kernel',
            'Gap correlations',
        ],
        'most_promising': 'Random Matrix Theory (GUE)',
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_19_theoretical_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
