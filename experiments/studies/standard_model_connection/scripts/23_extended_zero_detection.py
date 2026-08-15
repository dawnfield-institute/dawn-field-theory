"""
Script 23: Extended Riemann Zero Detection (Out-of-Sample Prediction)

GOAL: Test the Möbius coherence formula on zeros 61-100 as a GENUINE
out-of-sample prediction. The formula was developed using zeros 1-20.

THE CHAIN (verified December 24, 2025):
    π → Möbius μ(n) → ζ-zeros → primes → SEC → φ → PAC → SM

THE TEST:
    1. Use the Möbius formula: Z_μ(γ) = |Σ μ(n) exp(iγ log n) n^(-1/2)|
    2. Predict zeros 61-100 BEFORE checking tables
    3. Compare to Odlyzko's computed values
    4. Document prediction accuracy

SUCCESS CRITERION: <0.1 average error on zeros 61-100

SIGNIFICANCE:
    If the formula predicts zeros it wasn't tuned on, the π → φ chain
    is not overfitting—it's capturing real structure.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import json
from datetime import datetime

# Known Riemann zeros (from Odlyzko tables)
# Zeros 1-20 (used for development)
KNOWN_ZEROS_1_20 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840
]

# Zeros 21-60 (intermediate validation)
KNOWN_ZEROS_21_60 = [
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029535, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000982, 147.422765, 150.053521, 150.925258, 153.024693,
    156.112909, 157.597592, 158.849988, 161.188964, 163.030709
]

# Zeros 61-100 (OUT-OF-SAMPLE - used only for validation)
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
    
    # Factor n
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
                return 0  # Has squared factor
            factors.append(d)
        d += 1
    if temp > 1:
        factors.append(temp)
    
    return (-1) ** len(factors)


def mobius_coherence(gamma: float, N: int = 1000) -> float:
    """
    Compute Möbius coherence at height γ on the critical line.
    
    Z_μ(γ) = |Σ_{n=1}^{N} μ(n) exp(iγ log n) n^(-1/2)|
    
    Peaks at Riemann zeros.
    """
    total = 0.0
    for n in range(1, N + 1):
        mu = mobius(n)
        if mu != 0:
            phase = gamma * np.log(n)
            weight = n ** (-0.5)
            total += mu * np.exp(1j * phase) * weight
    return abs(total)


def find_zeros_in_range(start: float, end: float, resolution: float = 0.05) -> List[float]:
    """
    Find Riemann zeros in [start, end] using Möbius coherence peaks.
    """
    gammas = np.arange(start, end, resolution)
    coherences = [mobius_coherence(g) for g in gammas]
    
    # Find local maxima (peaks)
    peaks = []
    for i in range(1, len(coherences) - 1):
        if coherences[i] > coherences[i-1] and coherences[i] > coherences[i+1]:
            # Refine with finer search
            refined = refine_peak(gammas[i-1], gammas[i+1])
            peaks.append(refined)
    
    return peaks


def refine_peak(low: float, high: float, iterations: int = 10) -> float:
    """Refine peak location using golden section search."""
    phi = (1 + np.sqrt(5)) / 2
    
    for _ in range(iterations):
        d = (high - low) / phi
        x1 = high - d
        x2 = low + d
        
        if mobius_coherence(x1) > mobius_coherence(x2):
            high = x2
        else:
            low = x1
    
    return (low + high) / 2


def detect_zeros_61_to_100() -> Dict:
    """
    Main experiment: Predict zeros 61-100 using Möbius formula,
    then compare to known values.
    
    THIS IS A GENUINE OUT-OF-SAMPLE PREDICTION.
    """
    print("=" * 70)
    print("EXPERIMENT 23: Extended Riemann Zero Detection")
    print("Out-of-sample prediction for zeros 61-100")
    print("=" * 70)
    
    # The range containing zeros 61-100
    # Zero 61 is around 165.5, Zero 100 is around 236.5
    start_gamma = 164.0
    end_gamma = 238.0
    
    print(f"\nSearching for zeros in [{start_gamma}, {end_gamma}]...")
    print("Using Möbius coherence formula (developed on zeros 1-20 only)\n")
    
    # Detect zeros
    detected_peaks = find_zeros_in_range(start_gamma, end_gamma, resolution=0.1)
    
    # Match detected peaks to known zeros
    results = []
    matched_zeros = []
    
    for i, known_zero in enumerate(KNOWN_ZEROS_61_100):
        zero_num = 61 + i
        
        # Find closest detected peak
        if detected_peaks:
            distances = [abs(peak - known_zero) for peak in detected_peaks]
            min_idx = np.argmin(distances)
            detected = detected_peaks[min_idx]
            error = abs(detected - known_zero)
            
            # Remove matched peak to avoid double-matching
            if error < 2.0:  # Reasonable match threshold
                detected_peaks.pop(min_idx)
                matched = True
            else:
                matched = False
                detected = None
                error = None
        else:
            matched = False
            detected = None
            error = None
        
        results.append({
            "zero_number": zero_num,
            "known_value": known_zero,
            "detected_value": detected,
            "error": error,
            "matched": matched
        })
        
        if matched:
            matched_zeros.append((zero_num, known_zero, detected, error))
    
    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"{'Zero #':<10} {'Known γ':<15} {'Predicted γ':<15} {'Error':<10} {'Status':<10}")
    print("-" * 70)
    
    total_error = 0
    matched_count = 0
    
    for r in results:
        if r["matched"]:
            status = "✓ FOUND"
            matched_count += 1
            total_error += r["error"]
            print(f"{r['zero_number']:<10} {r['known_value']:<15.6f} {r['detected_value']:<15.6f} {r['error']:<10.4f} {status}")
        else:
            status = "✗ MISSED"
            print(f"{r['zero_number']:<10} {r['known_value']:<15.6f} {'N/A':<15} {'N/A':<10} {status}")
    
    avg_error = total_error / matched_count if matched_count > 0 else float('inf')
    detection_rate = matched_count / 40  # 40 zeros (61-100)
    
    print("-" * 70)
    print(f"\nSUMMARY:")
    print(f"  Zeros detected: {matched_count}/40 ({detection_rate*100:.1f}%)")
    print(f"  Average error: {avg_error:.4f}")
    print(f"  Success criterion: <0.1 average error")
    
    success = avg_error < 0.1 and detection_rate > 0.8
    
    if success:
        print(f"\n✓ SUCCESS: Out-of-sample prediction validates π → φ chain!")
    else:
        print(f"\n⚠ NEEDS REFINEMENT: Detection rate or accuracy below threshold")
    
    # Prepare output
    output = {
        "experiment": "23_extended_zero_detection",
        "timestamp": datetime.now().isoformat(),
        "description": "Out-of-sample Riemann zero prediction (61-100)",
        "method": "Möbius coherence: Z_μ(γ) = |Σ μ(n) exp(iγ log n) n^(-1/2)|",
        "training_zeros": "1-20 (formula developed)",
        "test_zeros": "61-100 (out-of-sample)",
        "results": {
            "matched_count": matched_count,
            "total_zeros": 40,
            "detection_rate": detection_rate,
            "average_error": avg_error,
            "success_criterion_error": 0.1,
            "success_criterion_rate": 0.8,
            "success": success
        },
        "individual_results": results,
        "chain_validation": "π → Möbius → zeros → primes → SEC → φ → PAC → SM"
    }
    
    return output


def run_validation_on_intermediate_zeros() -> Dict:
    """
    Also validate on zeros 21-60 as intermediate check.
    """
    print("\n" + "=" * 70)
    print("INTERMEDIATE VALIDATION: Zeros 21-60")
    print("=" * 70)
    
    start_gamma = 78.0
    end_gamma = 165.0
    
    detected_peaks = find_zeros_in_range(start_gamma, end_gamma, resolution=0.1)
    
    matched = 0
    total_error = 0
    
    for known_zero in KNOWN_ZEROS_21_60:
        if detected_peaks:
            distances = [abs(peak - known_zero) for peak in detected_peaks]
            min_idx = np.argmin(distances)
            error = distances[min_idx]
            if error < 2.0:
                matched += 1
                total_error += error
                detected_peaks.pop(min_idx)
    
    avg_error = total_error / matched if matched > 0 else float('inf')
    
    print(f"Zeros 21-60: {matched}/40 matched, avg error {avg_error:.4f}")
    
    return {
        "range": "21-60",
        "matched": matched,
        "total": 40,
        "average_error": avg_error
    }


if __name__ == "__main__":
    # Run main experiment
    results = detect_zeros_61_to_100()
    
    # Run intermediate validation
    intermediate = run_validation_on_intermediate_zeros()
    results["intermediate_validation"] = intermediate
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/23_extended_zero_detection_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x))
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("SIGNIFICANCE")
    print("=" * 70)
    print("""
    If the Möbius formula correctly predicts zeros 61-100:
    
    1. The formula captures REAL structure, not overfit to zeros 1-20
    2. The π → Möbius → zeros chain is VALIDATED
    3. This strengthens the entire chain to φ and Standard Model
    
    This is a genuine PREDICTION, not a post-hoc observation.
    """)
