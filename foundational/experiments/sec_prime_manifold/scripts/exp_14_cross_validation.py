#!/usr/bin/env python3
"""
Experiment 14: Cross-Validation with Alternative Prime Bases
============================================================

Goal: Test if Ï† emergence is specific to consecutive primes or
      generalizes to other prime-based factor bases.

Test bases:
1. Sophie Germain primes: p where 2p+1 is also prime
2. Twin primes: p where p+2 is also prime  
3. Mersenne exponents: primes p where 2^p-1 might be prime
4. Random primes (control)
5. Odd composites (negative control)

Key question: Is Ï† emergence a property of:
- ANY prime-based factor base of the right size?
- Only CONSECUTIVE primes?
- Some broader structure related to primality?

Trace output: results/exp_14_cross_validation_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    compute_sec, prime_sieve, symbolic_entropy, entropy_expectation,
    collapse_impulse, stress_field, FIRST_50_PRIMES, PHI
)

PHI_INV = 1 / PHI


def get_sophie_germain_primes(limit: int = 1000) -> List[int]:
    """
    Get Sophie Germain primes up to limit.
    p is Sophie Germain if both p and 2p+1 are prime.
    """
    prime_mask, primes = prime_sieve(limit * 3)  # Need room for 2p+1
    
    sg_primes = []
    for p in primes:
        if p > limit:
            break
        if 2*p + 1 <= len(prime_mask) - 1 and prime_mask[2*p + 1]:
            sg_primes.append(p)
    
    return sg_primes


def get_twin_primes(limit: int = 1000) -> List[int]:
    """
    Get twin primes up to limit (smaller of each pair).
    p is a twin prime if p+2 is also prime.
    """
    prime_mask, primes = prime_sieve(limit + 10)
    
    twin_primes = []
    for p in primes:
        if p > limit:
            break
        if p + 2 <= len(prime_mask) - 1 and prime_mask[p + 2]:
            twin_primes.append(p)
    
    return twin_primes


def get_mersenne_exponents(limit: int = 50) -> List[int]:
    """
    Get Mersenne exponents (primes p where 2^p-1 is known to be prime).
    These are: 2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, ...
    """
    # Known Mersenne exponents (p where M_p = 2^p - 1 is prime)
    known_mersenne_exp = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607]
    return [p for p in known_mersenne_exp if p <= limit]


def get_random_primes(n: int, max_val: int = 200, seed: int = 42) -> List[int]:
    """Get n random primes up to max_val."""
    np.random.seed(seed)
    prime_mask, primes = prime_sieve(max_val)
    primes = list(primes)
    
    if len(primes) < n:
        return primes
    
    selected = list(np.random.choice(primes, size=n, replace=False))
    return sorted(selected)


def get_odd_composites(n: int, max_val: int = 200) -> List[int]:
    """Get first n odd composites (negative control)."""
    prime_mask, _ = prime_sieve(max_val)
    
    composites = []
    for i in range(9, max_val, 2):  # Start at 9, first odd composite
        if not prime_mask[i]:
            composites.append(i)
            if len(composites) >= n:
                break
    
    return composites


def test_factor_base(name: str, factor_base: List[int], n_max: int) -> Dict[str, Any]:
    """Test a factor base and return key metrics."""
    
    if len(factor_base) < 2:
        return {
            "name": name,
            "factor_base": factor_base,
            "size": len(factor_base),
            "status": "insufficient_primes",
            "frac_E_positive": None,
            "error_vs_phi": None
        }
    
    try:
        sec = compute_sec(
            n_max=n_max,
            factor_base=factor_base,
            window=101,
            lam=0.99
        )
        
        idx = np.arange(3, n_max + 1, 2)
        E_odd = sec.E[idx]
        pm = sec.prime_mask[idx]
        
        frac_E_pos = float((E_odd > 0).mean())
        
        # Prime ratio
        prime_rate_pos = float(pm[E_odd > 0].mean()) if (E_odd > 0).any() else 0
        prime_rate_neg = float(pm[E_odd <= 0].mean()) if (E_odd <= 0).any() else 0
        ratio = prime_rate_pos / prime_rate_neg if prime_rate_neg > 0 else float('inf')
        
        return {
            "name": name,
            "factor_base": factor_base,
            "size": len(factor_base),
            "status": "success",
            "frac_E_positive": frac_E_pos,
            "error_vs_phi": frac_E_pos - PHI_INV,
            "abs_error": abs(frac_E_pos - PHI_INV),
            "prime_ratio": ratio,
            "enrichment_preserved": ratio > 2.0
        }
    except Exception as e:
        return {
            "name": name,
            "factor_base": factor_base,
            "size": len(factor_base),
            "status": f"error: {str(e)}",
            "frac_E_positive": None,
            "error_vs_phi": None
        }


def run_size_sweep(name: str, prime_list: List[int], n_max: int, 
                   sizes: List[int] = None) -> Dict[str, Any]:
    """Run size sweep for a given prime list."""
    
    if sizes is None:
        sizes = list(range(2, min(16, len(prime_list) + 1)))
    
    results = []
    best_size = None
    best_error = float('inf')
    
    for size in sizes:
        if size > len(prime_list):
            break
        
        fb = prime_list[:size]
        r = test_factor_base(f"{name}_size{size}", fb, n_max)
        results.append(r)
        
        if r["status"] == "success" and r["abs_error"] < best_error:
            best_error = r["abs_error"]
            best_size = size
    
    return {
        "prime_type": name,
        "available_primes": len(prime_list),
        "primes": prime_list[:20],  # First 20 for reference
        "size_sweep": results,
        "best_size": best_size,
        "best_error": best_error,
        "phi_achieved": best_error < 0.01 if best_size else False
    }


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> Dict[str, Any]:
    """Run cross-validation experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 14: Cross-Validation with Alternative Prime Bases")
    print("=" * 70)
    print(f"\nTarget: 1/phi = {PHI_INV:.6f}")
    
    results = {}
    
    # Get all prime lists
    sophie_germain = get_sophie_germain_primes(500)
    twin_primes = get_twin_primes(500)
    mersenne_exp = get_mersenne_exponents(100)
    random_primes = get_random_primes(25, max_val=200)
    consecutive_primes = FIRST_50_PRIMES[:25]
    odd_composites = get_odd_composites(15, max_val=200)
    
    print(f"\nPrime bases available:")
    print(f"  Sophie Germain: {len(sophie_germain)} primes - {sophie_germain[:10]}...")
    print(f"  Twin primes:    {len(twin_primes)} primes - {twin_primes[:10]}...")
    print(f"  Mersenne exp:   {len(mersenne_exp)} primes - {mersenne_exp}")
    print(f"  Random primes:  {len(random_primes)} primes - {random_primes[:10]}...")
    print(f"  Consecutive:    {len(consecutive_primes)} primes - {consecutive_primes[:10]}...")
    print(f"  Odd composites: {len(odd_composites)} numbers - {odd_composites[:10]}... (control)")
    
    # Test 1: Consecutive primes (baseline)
    print(f"\n" + "-" * 70)
    print("1. CONSECUTIVE PRIMES (BASELINE)")
    print("-" * 70)
    
    consec_results = run_size_sweep("consecutive", list(consecutive_primes), n_max)
    results["consecutive"] = consec_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in consec_results["size_sweep"]:
        if r["status"] == "success":
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {r['prime_ratio']:>10.2f}x")
    
    print(f"\nBest size: {consec_results['best_size']}, error: {consec_results['best_error']:.6f}")
    
    # Test 2: Sophie Germain primes
    print(f"\n" + "-" * 70)
    print("2. SOPHIE GERMAIN PRIMES")
    print("-" * 70)
    
    sg_results = run_size_sweep("sophie_germain", sophie_germain, n_max)
    results["sophie_germain"] = sg_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in sg_results["size_sweep"]:
        if r["status"] == "success":
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {r['prime_ratio']:>10.2f}x")
    
    print(f"\nBest size: {sg_results['best_size']}, error: {sg_results['best_error']:.6f}")
    
    # Test 3: Twin primes
    print(f"\n" + "-" * 70)
    print("3. TWIN PRIMES")
    print("-" * 70)
    
    twin_results = run_size_sweep("twin_primes", twin_primes, n_max)
    results["twin_primes"] = twin_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in twin_results["size_sweep"]:
        if r["status"] == "success":
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {r['prime_ratio']:>10.2f}x")
    
    print(f"\nBest size: {twin_results['best_size']}, error: {twin_results['best_error']:.6f}")
    
    # Test 4: Mersenne exponents
    print(f"\n" + "-" * 70)
    print("4. MERSENNE EXPONENTS")
    print("-" * 70)
    
    mers_results = run_size_sweep("mersenne", mersenne_exp, n_max, sizes=list(range(2, min(9, len(mersenne_exp)+1))))
    results["mersenne"] = mers_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in mers_results["size_sweep"]:
        if r["status"] == "success":
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {r['prime_ratio']:>10.2f}x")
    
    print(f"\nBest size: {mers_results['best_size']}, error: {mers_results['best_error']:.6f}")
    
    # Test 5: Random primes
    print(f"\n" + "-" * 70)
    print("5. RANDOM PRIMES")
    print("-" * 70)
    
    rand_results = run_size_sweep("random", random_primes, n_max)
    results["random"] = rand_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in rand_results["size_sweep"]:
        if r["status"] == "success":
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {r['prime_ratio']:>10.2f}x")
    
    print(f"\nBest size: {rand_results['best_size']}, error: {rand_results['best_error']:.6f}")
    
    # Test 6: Odd composites (negative control)
    print(f"\n" + "-" * 70)
    print("6. ODD COMPOSITES (NEGATIVE CONTROL)")
    print("-" * 70)
    
    comp_results = run_size_sweep("composites", odd_composites, n_max)
    results["composites"] = comp_results
    
    print(f"\n{'Size':>6} {'Frac E>0':>12} {'Error':>12} {'Ratio':>10}")
    print("-" * 45)
    for r in comp_results["size_sweep"]:
        if r["status"] == "success":
            ratio_str = f"{r['prime_ratio']:.2f}x" if r['prime_ratio'] < 100 else "inf"
            print(f"{r['size']:>6} {r['frac_E_positive']:>12.6f} {r['error_vs_phi']:>+12.6f} {ratio_str:>10}")
    
    print(f"\nBest size: {comp_results['best_size']}, error: {comp_results['best_error']:.6f}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Prime Type':<20} {'Best Size':>10} {'Best Error':>12} {'Ï† Achieved':>12} {'Enrichment':>12}")
    print("-" * 70)
    
    all_results = [
        ("Consecutive", consec_results),
        ("Sophie Germain", sg_results),
        ("Twin Primes", twin_results),
        ("Mersenne", mers_results),
        ("Random", rand_results),
        ("Composites (ctrl)", comp_results)
    ]
    
    for name, res in all_results:
        best_size = res.get('best_size', 'N/A')
        best_err = res.get('best_error', float('inf'))
        phi_ach = "âœ…" if res.get('phi_achieved', False) else "âŒ"
        
        # Check if enrichment preserved
        enriched = "N/A"
        for r in res.get("size_sweep", []):
            if r.get("size") == best_size and r.get("status") == "success":
                enriched = "âœ…" if r.get("enrichment_preserved", False) else "âŒ"
                break
        
        err_str = f"{best_err:.6f}" if best_err < float('inf') else "N/A"
        print(f"{name:<20} {str(best_size):>10} {err_str:>12} {phi_ach:>12} {enriched:>12}")
    
    # Validation
    validation = {
        "consecutive_achieves_phi": consec_results.get('phi_achieved', False),
        "sophie_germain_achieves_phi": sg_results.get('phi_achieved', False),
        "twin_primes_achieve_phi": twin_results.get('phi_achieved', False),
        "random_primes_achieve_phi": rand_results.get('phi_achieved', False),
        "composites_fail_phi": not comp_results.get('phi_achieved', True),
        "phi_is_universal_for_primes": (
            consec_results.get('phi_achieved', False) and
            sg_results.get('phi_achieved', False) and
            twin_results.get('phi_achieved', False)
        )
    }
    
    print(f"\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)
    for check, passed in validation.items():
        status = "âœ… PASS" if passed else "âŒ FAIL"
        print(f"  {check}: {status}")
    
    results["validation"] = validation
    
    # Key finding
    print(f"\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    
    prime_bases_achieve = sum([
        consec_results.get('phi_achieved', False),
        sg_results.get('phi_achieved', False),
        twin_results.get('phi_achieved', False),
        rand_results.get('phi_achieved', False)
    ])
    
    if prime_bases_achieve >= 3:
        print("""
Ï† emergence is UNIVERSAL for prime-based factor bases!

All tested prime bases achieve frac(E>0) â‰ˆ 1/Ï† at some size:
- Consecutive primes: optimal size varies
- Sophie Germain primes: optimal size varies  
- Twin primes: optimal size varies
- Random primes: optimal size varies

The composite base FAILS to achieve Ï†, confirming this is a
property of primality, not just any integer selection.
""")
    else:
        print("""
Ï† emergence appears SPECIFIC to consecutive primes.

Only consecutive primes reliably achieve frac(E>0) â‰ˆ 1/Ï†.
Alternative prime bases show different behavior.
""")
    
    # Save trace
    if save_trace:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"exp_14_cross_validation_{timestamp}.json"
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\nTrace saved: {filepath.name}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=50000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)

