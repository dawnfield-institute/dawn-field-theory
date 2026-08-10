"""
Script 23b: Refined Riemann Zero Detection

GOAL: Reduce average error from 0.26 to <0.1 for zeros 61-100

REFINEMENTS:
1. Increase N (sum length) for better resolution
2. Use finer initial grid search
3. Apply iterative refinement with golden section
4. Use quadratic interpolation for peak finding

THE TOPOLOGY INSIGHT:
The user noted "recursive topologies should not be linear" - this is why
the LOG-phase formula works: exp(iγ log n) is logarithmic, not linear.
The Möbius function lives on a recursive structure (square-free integers),
so the phase must also be recursive (logarithmic).
"""

import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Known Riemann zeros (Odlyzko tables)
KNOWN_ZEROS_61_100 = [
    165.537069, 167.184439, 169.094515, 169.911977, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265397, 196.876481, 198.015309, 201.264751,
    202.493594, 204.189671, 205.394697, 207.906259, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714919, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250189, 231.987235, 233.693404, 236.524230
]


def mobius(n: int) -> int:
    """Compute Möbius function μ(n)"""
    if n == 1:
        return 1
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            if count > 1:
                return 0
            factors.append(d)
        d += 1
    if temp > 1:
        factors.append(temp)
    return (-1) ** len(factors)


# Precompute Möbius values
MOBIUS_CACHE = {}
def get_mobius(n: int) -> int:
    if n not in MOBIUS_CACHE:
        MOBIUS_CACHE[n] = mobius(n)
    return MOBIUS_CACHE[n]

# Precompute for large N
def precompute_mobius(N: int):
    for n in range(1, N + 1):
        get_mobius(n)

precompute_mobius(3000)  # Precompute for larger sums


def mobius_coherence(gamma: float, N: int = 2000) -> float:
    """
    Möbius coherence at height γ (LOG-PHASE formula).
    
    Z_μ(γ) = |Σ_{n=1}^{N} μ(n) exp(iγ log n) n^(-1/2)|
    
    Peaks at Riemann zeros.
    """
    total = 0.0 + 0.0j
    for n in range(1, N + 1):
        mu = get_mobius(n)
        if mu != 0:
            phase = gamma * np.log(n)
            weight = n ** (-0.5)
            total += mu * np.exp(1j * phase) * weight
    return abs(total)


def golden_section_maximize(f, a: float, b: float, tol: float = 0.001) -> float:
    """
    Golden section search for maximum in [a, b].
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    while abs(b - a) > tol:
        if f1 > f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
    
    return (a + b) / 2


def quadratic_peak_refine(gamma_rough: float, delta: float = 0.5) -> float:
    """
    Refine peak using quadratic interpolation.
    """
    g1 = gamma_rough - delta
    g2 = gamma_rough
    g3 = gamma_rough + delta
    
    f1 = mobius_coherence(g1)
    f2 = mobius_coherence(g2)
    f3 = mobius_coherence(g3)
    
    # Quadratic interpolation for peak
    denom = (g1 - g2) * (g1 - g3) * (g2 - g3)
    if abs(denom) < 1e-10:
        return g2
    
    a = (g3 * (f2 - f1) + g2 * (f1 - f3) + g1 * (f3 - f2)) / denom
    b = (g3*g3 * (f1 - f2) + g2*g2 * (f3 - f1) + g1*g1 * (f2 - f3)) / denom
    
    if abs(a) < 1e-10:
        return g2
    
    peak = -b / (2 * a)
    
    # Sanity check
    if abs(peak - g2) > 2 * delta:
        return g2
    
    return peak


def find_zeros_refined(start: float, end: float, N: int = 2000) -> List[Tuple[float, float]]:
    """
    Find zeros with refined search.
    
    1. Coarse grid search
    2. Golden section refinement
    3. Quadratic interpolation polish
    """
    # Step 1: Coarse grid (resolution 0.05)
    resolution = 0.05
    gammas = np.arange(start, end, resolution)
    coherences = [mobius_coherence(g, N) for g in gammas]
    
    # Step 2: Find local maxima
    peaks = []
    for i in range(1, len(coherences) - 1):
        if coherences[i] > coherences[i-1] and coherences[i] > coherences[i+1]:
            # Coarse peak found
            gamma_coarse = gammas[i]
            
            # Step 3: Golden section refinement
            gamma_refined = golden_section_maximize(
                lambda g: mobius_coherence(g, N),
                gammas[i-1], gammas[i+1],
                tol=0.01
            )
            
            # Step 4: Quadratic polish
            gamma_polished = quadratic_peak_refine(gamma_refined, delta=0.02)
            
            coherence = mobius_coherence(gamma_polished, N)
            peaks.append((gamma_polished, coherence))
    
    return peaks


def detect_zeros_61_to_100_refined() -> Dict:
    """
    Main experiment: Refined zero detection.
    """
    print("=" * 70)
    print("EXPERIMENT 23b: Refined Riemann Zero Detection")
    print("Target: <0.1 average error on zeros 61-100")
    print("=" * 70)
    
    print(f"\nUsing N=2000 (increased from 1000)")
    print(f"Using golden section + quadratic refinement")
    print(f"Resolution: 0.05 (doubled from 0.1)\n")
    
    # Search range
    start_gamma = 164.0
    end_gamma = 238.0
    
    print(f"Searching [{start_gamma}, {end_gamma}]...\n")
    
    # Find all peaks
    detected_peaks = find_zeros_refined(start_gamma, end_gamma, N=2000)
    
    # Sort by gamma
    detected_peaks.sort(key=lambda x: x[0])
    detected_gammas = [p[0] for p in detected_peaks]
    
    # Match to known zeros
    results = []
    used_peaks = set()
    
    for i, known_zero in enumerate(KNOWN_ZEROS_61_100):
        zero_num = 61 + i
        
        # Find closest unused peak
        best_match = None
        best_error = float('inf')
        best_idx = None
        
        for j, detected in enumerate(detected_gammas):
            if j in used_peaks:
                continue
            error = abs(detected - known_zero)
            if error < best_error:
                best_error = error
                best_match = detected
                best_idx = j
        
        if best_match is not None and best_error < 2.0:
            used_peaks.add(best_idx)
            matched = True
        else:
            matched = False
            best_match = None
            best_error = None
        
        results.append({
            "zero_number": zero_num,
            "known_value": known_zero,
            "detected_value": best_match,
            "error": best_error,
            "matched": matched
        })
    
    # Print results
    print(f"{'Zero':<8} {'Known γ':<15} {'Detected γ':<15} {'Error':<10} {'Status'}")
    print("-" * 60)
    
    total_error = 0
    matched_count = 0
    errors = []
    
    for r in results:
        if r["matched"]:
            matched_count += 1
            total_error += r["error"]
            errors.append(r["error"])
            status = "✓"
            print(f"{r['zero_number']:<8} {r['known_value']:<15.6f} {r['detected_value']:<15.6f} {r['error']:<10.6f} {status}")
        else:
            status = "✗ MISSED"
            print(f"{r['zero_number']:<8} {r['known_value']:<15.6f} {'N/A':<15} {'N/A':<10} {status}")
    
    avg_error = total_error / matched_count if matched_count > 0 else float('inf')
    median_error = np.median(errors) if errors else float('inf')
    max_error = max(errors) if errors else float('inf')
    
    print("-" * 60)
    print(f"\nSUMMARY:")
    print(f"  Detected: {matched_count}/40 ({matched_count/40*100:.1f}%)")
    print(f"  Average error: {avg_error:.4f}")
    print(f"  Median error: {median_error:.4f}")
    print(f"  Max error: {max_error:.4f}")
    print(f"  Target: <0.1 average error")
    
    success = avg_error < 0.1 and matched_count >= 35
    
    if success:
        print(f"\n✅ SUCCESS: Refined detection meets criterion!")
    elif avg_error < 0.15:
        print(f"\n⚠️ CLOSE: Error {avg_error:.4f} approaching target 0.1")
    else:
        print(f"\n❌ NEEDS MORE WORK: Error {avg_error:.4f} > 0.1")
    
    # Error distribution
    if errors:
        print(f"\nError distribution:")
        bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        for i in range(len(bins) - 1):
            count = sum(1 for e in errors if bins[i] <= e < bins[i+1])
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} zeros")
    
    output = {
        "experiment": "23b_refined_zero_detection",
        "timestamp": datetime.now().isoformat(),
        "parameters": {"N": 2000, "resolution": 0.05, "refinement": "golden+quadratic"},
        "results": {
            "matched": matched_count,
            "total": 40,
            "average_error": avg_error,
            "median_error": median_error,
            "max_error": max_error,
            "success": bool(success)
        },
        "topology_note": "Log-phase matches recursive Möbius structure"
    }
    
    return output


if __name__ == "__main__":
    results = detect_zeros_61_to_100_refined()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/23b_refined_zero_detection_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
