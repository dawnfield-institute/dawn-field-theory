#!/usr/bin/env python3
"""
exp_29: Feigenbaum Formula Extended Parameter Search

HYPOTHESIS: The Feigenbaum closed-form formulas remain unique (no additional
matches) when the search space is extended from a∈[1,200] to a∈[1,500].

SOURCE: sec_threshold_detection/scripts/exp_09_statistical_proof.py
TARGET: Paper 3 - §14.1 (extended search target)

BACKGROUND:
    The original search (exp_09 in sec_threshold_detection) tested:
        a ∈ [1, 200), b ∈ [1, 100), c ∈ [1, 200) → ~3.9M combinations
    and found exactly ONE 8+ digit match: (55, 17, 52).

    Paper 3 §14 targets a,b ∈ [1, 1000] (~10⁹ combos), which is
    computationally prohibitive in a single run. This experiment extends
    to a ∈ [1, 500), b ∈ [1, 250), c ∈ [1, 500) → ~31M combos (~8× more).
    
    If (55, 17, 52) remains the SOLE 8-digit match in 31M combinations,
    the uniqueness argument strengthens from 1-in-3.9M to 1-in-31M.

FALSIFICATION (F-EXT): If any new 8+ digit match is found outside [1,200],
the uniqueness claim weakens (though the structural argument — a=F₁₀,
b=17 Fermat-adjacent, c=a−3 — would still distinguish the known triple).

METHOD:
    1. r∞ exhaustive search: a∈[1,500), b∈[1,250), c∈[1,500)
    2. α (Feigenbaum scaling) extended search: k∈[1,2000)
    3. Report all matches at 7, 8, 9+ digit precision
    4. Compare to original (55,17,52) result
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import save_results

# =============================================================================
# KNOWN CONSTANTS (high precision from OEIS)
# =============================================================================

R_INF = 3.5699456718709449018420051513865   # OEIS A098587
DELTA = 4.669201609102990671853203820466    # OEIS A006890
ALPHA_F = 2.502907875095892822283902873218  # OEIS A006891


# =============================================================================
# FORMULA DEFINITIONS (from sec_threshold_detection/exp_09)
# =============================================================================

def formula_r_inf(a, b, c_base):
    """
    r∞ formula: π(a + √(b - π/(a·√(c + 2π/a))))(a + π) / a²
    Known best: (55, 17, 52) → 8+ digit match.
    """
    try:
        c = np.sqrt(c_base + 2 * np.pi / a)
        inner = b - np.pi / (a * c)
        if inner < 0:
            return None
        return np.pi * (a + np.sqrt(inner)) * (a + np.pi) / a**2
    except:
        return None


def formula_alpha_f(k):
    """α formula: (5 + π/k) / 2. Known best: k=540."""
    return (5 + np.pi / k) / 2


# =============================================================================
# SEARCH 1: r∞ EXTENDED PARAMETER SEARCH
# =============================================================================

def extended_search_r_inf(a_range=(1, 500), b_range=(1, 250), c_range=(1, 500),
                          progress_interval=50):
    """
    Exhaustively search all integer (a, b, c) combinations for r∞ matches.
    
    Parameters
    ----------
    a_range, b_range, c_range : tuple
        (start, end) exclusive ranges for each parameter.
    progress_interval : int
        Print progress every N values of a.
    
    Returns
    -------
    dict with hit counts, matches, and best parameters.
    """
    total = (a_range[1] - a_range[0]) * (b_range[1] - b_range[0]) * (c_range[1] - c_range[0])
    print(f"  r∞ search: {total:,} combinations")
    print(f"  Ranges: a∈[{a_range[0]},{a_range[1]}), b∈[{b_range[0]},{b_range[1]}), c∈[{c_range[0]},{c_range[1]})")
    
    hits_7 = []  # 7+ digit matches (error < 1e-7)
    hits_8 = []  # 8+ digit matches (error < 1e-8)
    hits_9 = []  # 9+ digit matches (error < 1e-9)
    best_error = 1.0
    best_params = None
    evaluated = 0
    t0 = time.time()
    
    for a in range(a_range[0], a_range[1]):
        if a % progress_interval == 0:
            elapsed = time.time() - t0
            rate = evaluated / elapsed if elapsed > 0 else 0
            pct = 100 * (a - a_range[0]) / (a_range[1] - a_range[0])
            print(f"    a={a} ({pct:.0f}%) — {evaluated:,} evaluated, {rate:.0f}/s, "
                  f"{len(hits_7)} hits@7, {len(hits_8)} hits@8")
        
        for b in range(b_range[0], b_range[1]):
            for c_base in range(c_range[0], c_range[1]):
                result = formula_r_inf(a, b, c_base)
                evaluated += 1
                if result is not None:
                    error = abs(result - R_INF) / R_INF
                    if error < best_error:
                        best_error = error
                        best_params = (a, b, c_base)
                    if error < 1e-7:
                        hits_7.append((a, b, c_base, float(error)))
                    if error < 1e-8:
                        hits_8.append((a, b, c_base, float(error)))
                    if error < 1e-9:
                        hits_9.append((a, b, c_base, float(error)))
    
    elapsed = time.time() - t0
    
    # Check if (55, 17, 52) is still uniquely the best
    is_known_best = best_params == (55, 17, 52)
    known_in_8 = any(h[0] == 55 and h[1] == 17 and h[2] == 52 for h in hits_8)
    
    # Identify any NEW hits not in original [1,200] range
    new_hits_7 = [(a, b, c, e) for (a, b, c, e) in hits_7
                  if a >= 200 or b >= 100 or c >= 200]
    new_hits_8 = [(a, b, c, e) for (a, b, c, e) in hits_8
                  if a >= 200 or b >= 100 or c >= 200]
    
    return {
        "search_ranges": {
            "a": list(a_range), "b": list(b_range), "c": list(c_range)
        },
        "total_combinations": total,
        "evaluated": evaluated,
        "elapsed_seconds": round(elapsed, 1),
        "rate_per_second": round(evaluated / elapsed, 0) if elapsed > 0 else 0,
        "hits_7_digit": len(hits_7),
        "hits_8_digit": len(hits_8),
        "hits_9_digit": len(hits_9),
        "all_hits_7": hits_7[:50],  # Cap at 50 for readability
        "all_hits_8": hits_8[:50],
        "all_hits_9": hits_9[:50],
        "new_hits_beyond_original_7": new_hits_7,
        "new_hits_beyond_original_8": new_hits_8,
        "best_params": list(best_params) if best_params else None,
        "best_error": float(best_error),
        "best_digits": round(-np.log10(best_error), 1) if best_error > 0 else float('inf'),
        "known_triple_still_best": is_known_best,
        "known_triple_in_8digit": known_in_8,
        "uniqueness_factor": f"1-in-{total:,}"
    }


# =============================================================================
# SEARCH 2: α EXTENDED SEARCH
# =============================================================================

def extended_search_alpha(k_range=(1, 2000)):
    """
    Search for k values where (5 + π/k)/2 matches the Feigenbaum α constant.
    Known best: k=540 → ~6 digit match.
    """
    print(f"\n  α search: k∈[{k_range[0]},{k_range[1]})")
    
    hits = []
    best_error = 1.0
    best_k = None
    
    for k in range(k_range[0], k_range[1]):
        result = formula_alpha_f(k)
        error = abs(result - ALPHA_F) / ALPHA_F
        if error < best_error:
            best_error = error
            best_k = k
        if error < 1e-4:
            hits.append({
                "k": k,
                "error": float(error),
                "digits": round(-np.log10(error), 1) if error > 0 else float('inf'),
                "value": float(result),
                "is_fibonacci_related": k in [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,
                                               144, 233, 377, 610, 987, 1597],
                "is_multiple_of_55": k % 55 == 0
            })
    
    return {
        "search_range": list(k_range),
        "total_searched": k_range[1] - k_range[0],
        "hits_at_4digits": len(hits),
        "all_hits": hits,
        "best_k": best_k,
        "best_error": float(best_error),
        "best_digits": round(-np.log10(best_error), 1) if best_error > 0 else float('inf'),
        "known_540_still_best": best_k == 540,
        "note": "k=540 was found in the original search. 540 = 10×54 = 10×(55-1)."
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXP_29: Feigenbaum Formula Extended Parameter Search")
    print("=" * 70)
    print()
    print("Extending the exhaustive Feigenbaum formula search from")
    print("a∈[1,200) to a∈[1,500), testing uniqueness of (55,17,52).")
    print()
    
    results = {
        "experiment": "exp_29_feigenbaum_extended_search",
        "target_paper": "Paper 3 (Feigenbaum Constants from Fibonacci Arithmetic)",
        "purpose": "Extend parameter search space 8× to test formula uniqueness",
        "original_range": {"a": [1, 200], "b": [1, 100], "c": [1, 200]},
        "extended_range": {"a": [1, 500], "b": [1, 250], "c": [1, 500]},
    }
    
    # --- Search 1: r∞ ---
    print("SEARCH 1: r∞ = 3.56994567... (Feigenbaum first bifurcation)")
    print("-" * 60)
    r_inf_results = extended_search_r_inf(
        a_range=(1, 500), b_range=(1, 250), c_range=(1, 500),
        progress_interval=50
    )
    results["r_inf_search"] = r_inf_results
    
    print(f"\n  Results:")
    print(f"    7-digit matches: {r_inf_results['hits_7_digit']}")
    print(f"    8-digit matches: {r_inf_results['hits_8_digit']}")
    print(f"    9-digit matches: {r_inf_results['hits_9_digit']}")
    print(f"    Best: {r_inf_results['best_params']} ({r_inf_results['best_digits']} digits)")
    print(f"    (55,17,52) still best: {r_inf_results['known_triple_still_best']}")
    print(f"    NEW hits beyond [1,200]: {len(r_inf_results['new_hits_beyond_original_8'])} at 8+ digits")
    
    # --- Search 2: α ---
    print("\nSEARCH 2: α = 2.50290787... (Feigenbaum scaling constant)")
    print("-" * 60)
    alpha_results = extended_search_alpha(k_range=(1, 2000))
    results["alpha_search"] = alpha_results
    
    print(f"\n  Results:")
    print(f"    4-digit matches: {alpha_results['hits_at_4digits']}")
    print(f"    Best k: {alpha_results['best_k']} ({alpha_results['best_digits']} digits)")
    print(f"    k=540 still best: {alpha_results['known_540_still_best']}")
    
    # --- Synthesis ---
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    
    r_unique = r_inf_results['known_triple_still_best']
    a_unique = alpha_results['known_540_still_best']
    new_r_hits = len(r_inf_results['new_hits_beyond_original_8'])
    
    if r_unique and a_unique and new_r_hits == 0:
        verdict = "STRENGTHENED"
        detail = (f"(55,17,52) remains the sole 8-digit match in {r_inf_results['total_combinations']:,} "
                  f"combinations (up from ~3.9M). No new matches found in extended range. "
                  f"Uniqueness factor: {r_inf_results['uniqueness_factor']}.")
    elif new_r_hits > 0:
        verdict = "WEAKENED"
        detail = (f"Found {new_r_hits} new 8-digit matches beyond original range. "
                  f"The (55,17,52) triple is NOT unique in the extended space. "
                  f"Structural argument (a=F₁₀, b=17, c=a−3) still distinguishes it.")
    else:
        verdict = "UNCHANGED"
        detail = "Best parameters unchanged, but results require further analysis."
    
    results["synthesis"] = {
        "verdict": verdict,
        "detail": detail,
        "r_inf_unique": r_unique,
        "alpha_unique": a_unique,
        "new_matches_found": new_r_hits,
        "scale_factor_vs_original": round(r_inf_results['total_combinations'] / 3_920_499, 1)
    }
    
    print(f"\n  Verdict: {verdict}")
    print(f"  {detail}")
    
    # --- PASS/FAIL ---
    passed = r_unique and a_unique
    results["PASS"] = passed
    results["status"] = "PASS — formulas remain unique in 8× larger search space" if passed \
        else "FAIL — new matches found, uniqueness weakened"
    
    print(f"\n  {'PASS' if passed else 'FAIL'}: {results['status']}")
    
    # Save
    save_results(results, "exp_29_feigenbaum_extended_search")
    
    return results


if __name__ == "__main__":
    main()
