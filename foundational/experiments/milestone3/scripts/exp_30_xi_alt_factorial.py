#!/usr/bin/env python3
"""
exp_30: Ξ Alternative Decomposition — 2×2 Factorial Design

HYPOTHESIS: The balance constant Ξ = γ + ln(φ) ≈ 1.0584 appears specifically
in systems with BOTH conservation AND self-similarity, not in systems
with only one condition.

SOURCE: Conditional Attractor Hypothesis (CAH) from cellular_automata_pac_attractors
TARGET: Paper 2 - sharpening Ξ uniqueness argument

BACKGROUND:
    The CAH tests 4 conditions: closed, recursive, conserving, saturated.
    Fisher exact p = 3.5×10⁻¹⁰ for the conjunction.
    
    Paper 2 notes γ + ln(φ) is rank #1 among 21 decompositions within 5% of Ξ,
    but the alternative Ξ_alt = 1/√3 + ln(φ) ≈ 1.0586 (0.02% different) has
    never been tested across all four convergence domains.
    
    The SCBF finding: conservation + self-similarity → φ (Fibonacci),
    conservation − self-similarity → 1 (Primes). But nobody has
    implemented the 2×2 factorial that isolates this pair.

METHOD:
    1. Define a minimal dynamical system (coupled oscillator network)
    2. Toggle two conditions independently:
       - Conservation: sum of node values is preserved (ON/OFF)
       - Self-similarity: hierarchical recursive coupling (ON/OFF)
    3. For each of the 4 quadrants, run N_trials and measure:
       - Dominant ratio (P/A attractor)
       - Distance to φ, γ, Ξ, 1/√3
    4. Test whether Ξ emerges preferentially in the BOTH-ON quadrant

FALSIFICATION: If Ξ ≈ 1.058 appears equally in all quadrants, the
conditional attractor hypothesis is falsified — Ξ would be an artifact
of the dynamical system itself, not of the condition pair.

ADDITIONALLY: Tests Ξ_alt = 1/√3 + ln(φ) as a competing decomposition
across 4 analytic domains (Navier-Stokes, Rule 110, primes, Landauer).
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE
from core.utils import save_results

# =============================================================================
# ALTERNATIVE DECOMPOSITIONS
# =============================================================================

XI_CANONICAL = GAMMA_EM + LN_PHI                    # γ + ln(φ) ≈ 1.0584
XI_ALT_SQRT3 = 1 / np.sqrt(3) + LN_PHI             # 1/√3 + ln(φ) ≈ 1.0586
XI_ALT_LN2_LN3 = np.log(2) / np.log(3) + LN_PHI   # ln2/ln3 + ln(φ) ≈ 1.1122
XI_ALT_1_MINUS_LNPHI = (1 - LN_PHI) + LN_PHI       # Trivially = 1.0
XI_PI_55 = 1 + np.pi / 55                           # 1 + π/55 ≈ 1.0571

TARGETS = {
    "γ + ln(φ)": XI_CANONICAL,
    "1/√3 + ln(φ)": XI_ALT_SQRT3,
    "1 + π/55": XI_PI_55,
}

# Reference values from four convergence domains
DOMAIN_MEASUREMENTS = {
    "navier_stokes": {"value": 1.0571, "source": "Symbolic engine threshold"},
    "rule_110":      {"value": 1.058,  "source": "P/A ratio at edge of chaos"},
    "primes":        {"value": 1.058,  "source": "Discrete-continuous interface"},
    "landauer":      {"value": 1.0584, "source": "γ + ln(φ) decomposition"},
}


# =============================================================================
# PART 1: ANALYTIC DOMAIN COMPARISON — Ξ vs Ξ_alt
# =============================================================================

def domain_comparison():
    """
    Compare γ + ln(φ) vs 1/√3 + ln(φ) across 4 convergence domains.
    
    The key question: does γ outperform 1/√3 in reproducing the
    measured domain values, given they differ by only 0.023%?
    """
    print("PART 1: Domain Comparison — γ + ln(φ) vs 1/√3 + ln(φ)")
    print("-" * 60)
    
    results = []
    for domain, data in DOMAIN_MEASUREMENTS.items():
        measured = data["value"]
        err_canonical = abs(measured - XI_CANONICAL) / measured * 100
        err_alt = abs(measured - XI_ALT_SQRT3) / measured * 100
        err_pi55 = abs(measured - XI_PI_55) / measured * 100
        
        winner = "γ + ln(φ)" if err_canonical < err_alt else "1/√3 + ln(φ)"
        
        results.append({
            "domain": domain,
            "measured": measured,
            "err_gamma_lnphi_pct": round(err_canonical, 4),
            "err_sqrt3_lnphi_pct": round(err_alt, 4),
            "err_pi_55_pct": round(err_pi55, 4),
            "winner": winner
        })
        
        print(f"  {domain:20s}: measured={measured:.4f}  "
              f"γ+ln(φ)={err_canonical:.4f}%  1/√3+ln(φ)={err_alt:.4f}%  "
              f"→ {winner}")
    
    # Aggregate: which decomposition has lower total error?
    total_canonical = sum(r["err_gamma_lnphi_pct"] for r in results)
    total_alt = sum(r["err_sqrt3_lnphi_pct"] for r in results)
    
    overall_winner = "γ + ln(φ)" if total_canonical < total_alt else "1/√3 + ln(φ)"
    
    print(f"\n  Total error: γ+ln(φ) = {total_canonical:.4f}%, "
          f"1/√3+ln(φ) = {total_alt:.4f}%")
    print(f"  Overall winner: {overall_winner}")
    print(f"  Margin: {abs(total_canonical - total_alt):.4f}%")
    
    return {
        "domain_results": results,
        "total_err_canonical": round(total_canonical, 4),
        "total_err_alt_sqrt3": round(total_alt, 4),
        "overall_winner": overall_winner,
        "margin_pct": round(abs(total_canonical - total_alt), 4),
        "note": ("Both decompositions are within 0.15% of all domains. "
                 "The difference is extremely small — 1/√3 differs from γ by only 0.023%.")
    }


# =============================================================================
# PART 2: 2×2 FACTORIAL — Conservation × Self-Similarity
# =============================================================================

def create_system(n_nodes=32, conservation=True, self_similarity=True, seed=None):
    """
    Create and evolve a coupled node network with toggled conditions.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    conservation : bool
        If True, total node values are preserved at each step.
    self_similarity : bool  
        If True, coupling is hierarchical (binary tree); if False, flat random.
    seed : int or None
        Random seed.
    
    Returns
    -------
    dict with attractor measurements after evolution.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random positive values
    values = rng.exponential(1.0, n_nodes)
    initial_total = values.sum()
    
    # Build coupling matrix
    coupling = np.zeros((n_nodes, n_nodes))
    
    if self_similarity:
        # Hierarchical coupling: binary tree structure
        # Level k couples nodes at distance 2^k with strength φ^(-k)
        for k in range(int(np.log2(n_nodes))):
            stride = 2**k
            strength = PHI**(-k)
            for i in range(n_nodes):
                j = (i + stride) % n_nodes
                coupling[i, j] += strength
                coupling[j, i] += strength
    else:
        # Flat random coupling: each node couples to ~4 random others
        for i in range(n_nodes):
            partners = rng.choice(n_nodes, size=4, replace=False)
            for j in partners:
                if i != j:
                    strength = rng.uniform(0.1, 1.0)
                    coupling[i, j] = strength
                    coupling[j, i] = strength
    
    # Normalize coupling rows
    row_sums = coupling.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    coupling = coupling / row_sums * 0.3  # 30% mixing per step
    
    # Evolve
    n_steps = 500
    history = [values.copy()]
    
    for step in range(n_steps):
        # Diffusion via coupling
        delta = coupling @ values - coupling.sum(axis=1) * values
        new_values = values + 0.1 * delta
        
        # Add small noise
        new_values += rng.normal(0, 0.01, n_nodes)
        new_values = np.maximum(new_values, 1e-10)  # Keep positive
        
        if conservation:
            # Enforce conservation: rescale to preserve total
            new_values = new_values * (initial_total / new_values.sum())
        
        values = new_values
        history.append(values.copy())
    
    history = np.array(history)
    
    # --- Measure attractor properties ---
    late = history[300:]  # Last 200 steps = steady state
    
    # 1. Dominant eigenvalue ratio of late correlation matrix
    corr = np.corrcoef(late.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(corr)))[::-1]
    
    if eigvals[1] > 1e-10:
        lambda_ratio = eigvals[0] / eigvals[1]
    else:
        lambda_ratio = float('inf')
    
    # 2. Potential/Actual ratio: std of late means vs mean of late stds
    node_means = late.mean(axis=0)
    node_stds = late.std(axis=0)
    potential = node_means.std()  # Variation across nodes
    actual = node_stds.mean()    # Variation within nodes
    pa_ratio = potential / actual if actual > 1e-10 else float('inf')
    
    # 3. Total conservation error
    totals = history.sum(axis=1)
    conservation_err = totals.std() / totals.mean() if totals.mean() > 0 else float('inf')
    
    # 4. Self-similarity measure: is the eigenvalue spectrum fractal?
    # Use participation ratio as proxy
    eigvals_norm = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
    participation = 1.0 / (eigvals_norm**2).sum() if (eigvals_norm**2).sum() > 0 else 0
    
    return {
        "lambda_ratio": float(lambda_ratio),
        "pa_ratio": float(pa_ratio),
        "conservation_error": float(conservation_err),
        "participation_ratio": float(participation),
        "final_total": float(values.sum()),
        "initial_total": float(initial_total),
    }


def factorial_experiment(n_trials=50, n_nodes=32):
    """
    Run the 2×2 factorial: {conservation ON/OFF} × {self-similarity ON/OFF}.
    """
    print("\nPART 2: 2×2 Factorial — Conservation × Self-Similarity")
    print("-" * 60)
    
    conditions = [
        ("BOTH_ON",   True,  True),
        ("CONS_ONLY", True,  False),
        ("SS_ONLY",   False, True),
        ("NEITHER",   False, False),
    ]
    
    all_results = {}
    
    for name, cons, ss in conditions:
        pa_ratios = []
        lambda_ratios = []
        
        for trial in range(n_trials):
            r = create_system(n_nodes=n_nodes, conservation=cons,
                              self_similarity=ss, seed=trial * 1000 + hash(name) % 1000)
            pa_ratios.append(r["pa_ratio"])
            lambda_ratios.append(r["lambda_ratio"])
        
        pa_arr = np.array(pa_ratios)
        pa_arr = pa_arr[np.isfinite(pa_arr)]
        
        mean_pa = float(np.mean(pa_arr)) if len(pa_arr) > 0 else float('nan')
        std_pa = float(np.std(pa_arr)) if len(pa_arr) > 0 else float('nan')
        
        # Distances to known constants
        dist_xi = abs(mean_pa - XI_CANONICAL) / XI_CANONICAL * 100 if np.isfinite(mean_pa) else float('inf')
        dist_phi = abs(mean_pa - PHI) / PHI * 100 if np.isfinite(mean_pa) else float('inf')
        dist_1 = abs(mean_pa - 1.0) * 100 if np.isfinite(mean_pa) else float('inf')
        dist_inv_phi = abs(mean_pa - INV_PHI) / INV_PHI * 100 if np.isfinite(mean_pa) else float('inf')
        
        all_results[name] = {
            "conservation": cons,
            "self_similarity": ss,
            "n_trials": n_trials,
            "n_valid": len(pa_arr),
            "mean_pa_ratio": round(mean_pa, 6),
            "std_pa_ratio": round(std_pa, 6),
            "dist_to_xi_pct": round(dist_xi, 2),
            "dist_to_phi_pct": round(dist_phi, 2),
            "dist_to_1_pct": round(dist_1, 2),
            "dist_to_inv_phi_pct": round(dist_inv_phi, 2),
            "pa_ratios": [round(float(x), 6) for x in pa_arr[:10]],  # Sample
        }
        
        print(f"  {name:12s}: P/A = {mean_pa:.4f} ± {std_pa:.4f}  "
              f"(dist Ξ: {dist_xi:.1f}%, dist φ: {dist_phi:.1f}%, dist 1: {dist_1:.1f}%)")
    
    # --- Statistical tests ---
    both = np.array([r for r in pa_ratios if np.isfinite(r)])
    
    # Test: is BOTH_ON closer to Ξ than other quadrants?
    both_pa = all_results["BOTH_ON"]["mean_pa_ratio"]
    cons_pa = all_results["CONS_ONLY"]["mean_pa_ratio"]
    ss_pa = all_results["SS_ONLY"]["mean_pa_ratio"]
    neither_pa = all_results["NEITHER"]["mean_pa_ratio"]
    
    # One-sided t-test: BOTH_ON P/A closer to Ξ than CONS_ONLY
    # (We test whether the distance is smaller for BOTH_ON)
    
    print(f"\n  Ξ target: {XI_CANONICAL:.6f}")
    print(f"  Nearest quadrant to Ξ: {min(all_results.items(), key=lambda x: x[1]['dist_to_xi_pct'])[0]}")
    
    return all_results


# =============================================================================
# PART 3: INFORMATION-THEORETIC ARGUMENT
# =============================================================================

def information_argument():
    """
    Why γ over 1/√3?
    
    γ = lim(Σ1/k - ln(n)) — the cost of bridging discrete and continuous
    1/√3 — appears in 3D geometry (equilateral triangle altitude)
    
    The PAC framework predicts: the additive component of Ξ should be
    the discrete-continuous interface cost. γ has this meaning. 1/√3 does not.
    """
    print("\nPART 3: Information-Theoretic Argument")
    print("-" * 60)
    
    # Compute partial sums of Σ1/k - ln(n) to show γ convergence
    partial_sums = []
    for n in [10, 100, 1000, 10000, 100000]:
        harmonic = sum(1.0 / k for k in range(1, n + 1))
        approx_gamma = harmonic - np.log(n)
        error = abs(approx_gamma - GAMMA_EM) / GAMMA_EM * 100
        partial_sums.append({
            "n": n,
            "harmonic_minus_ln": round(approx_gamma, 8),
            "error_pct": round(error, 6)
        })
    
    # Why this matters: γ appears whenever you bridge discrete counting (Σ1/k)
    # with continuous integration (ln n). In the PAC framework:
    # - Pure continuous collapse → ln(φ) (Paper 1)
    # - Discrete-continuous interface → adds γ
    # - Total balance constant → γ + ln(φ) = Ξ
    
    # 1/√3 has no such interpretation in the information/entropy framework
    
    argument = {
        "gamma_meaning": "Cost of bridging discrete counting with continuous integration",
        "sqrt3_meaning": "Geometric constant (equilateral triangle, 3D geometry)",
        "pac_prediction": "Additive component should be discrete-continuous interface cost",
        "gamma_convergence": partial_sums,
        "discrimination": (
            "Numerically, γ ≈ 0.57722 and 1/√3 ≈ 0.57735 differ by 0.023%. "
            "No numerical measurement can distinguish them at current precision. "
            "The argument is THEORETICAL: γ has the right meaning within the PAC "
            "framework (discrete-continuous bridge), while 1/√3 does not. "
            "This is a principled selection, not a numerical one."
        )
    }
    
    print(f"  γ  = {GAMMA_EM:.10f} (discrete-continuous bridge)")
    print(f"  1/√3 = {1/np.sqrt(3):.10f} (geometric constant)")
    print(f"  Difference: {abs(GAMMA_EM - 1/np.sqrt(3)):.10f} ({abs(GAMMA_EM - 1/np.sqrt(3))/GAMMA_EM*100:.4f}%)")
    print(f"  Discrimination: theoretical (meaning), not numerical (measurement)")
    
    return argument


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXP_30: Ξ Alternative Decomposition — 2×2 Factorial Design")
    print("=" * 70)
    print()
    
    results = {
        "experiment": "exp_30_xi_alt_factorial",
        "target_paper": "Paper 2 (The Balance Constant and Its Decomposition)",
        "purpose": "Test whether Ξ = γ + ln(φ) appears preferentially with conservation + self-similarity",
        "alternatives_tested": {
            "canonical": f"γ + ln(φ) = {XI_CANONICAL:.8f}",
            "alt_sqrt3": f"1/√3 + ln(φ) = {XI_ALT_SQRT3:.8f}",
            "pi_55": f"1 + π/55 = {XI_PI_55:.8f}",
            "difference_canonical_vs_alt": f"{abs(XI_CANONICAL - XI_ALT_SQRT3):.8f} ({abs(XI_CANONICAL - XI_ALT_SQRT3)/XI_CANONICAL*100:.4f}%)"
        }
    }
    
    # Part 1: Domain comparison
    results["domain_comparison"] = domain_comparison()
    
    # Part 2: 2×2 factorial
    results["factorial"] = factorial_experiment(n_trials=50, n_nodes=32)
    
    # Part 3: Information-theoretic argument
    results["information_argument"] = information_argument()
    
    # --- Synthesis ---
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    
    domain_winner = results["domain_comparison"]["overall_winner"]
    domain_margin = results["domain_comparison"]["margin_pct"]
    
    # Which quadrant is closest to Ξ?
    fact = results["factorial"]
    closest_to_xi = min(fact.items(), key=lambda x: x[1]["dist_to_xi_pct"])
    
    synthesis = {
        "domain_winner": domain_winner,
        "domain_margin_pct": domain_margin,
        "domain_verdict": (
            f"Across 4 domains, {domain_winner} fits better by {domain_margin:.4f}%. "
            f"This margin is too small for numerical discrimination."
        ),
        "factorial_closest_quadrant": closest_to_xi[0],
        "factorial_closest_dist_pct": closest_to_xi[1]["dist_to_xi_pct"],
        "factorial_verdict": (
            f"The {closest_to_xi[0]} quadrant has P/A ratio closest to Ξ "
            f"({closest_to_xi[1]['dist_to_xi_pct']:.1f}% distance)."
        ),
        "overall_verdict": (
            "γ and 1/√3 are NUMERICALLY indistinguishable at current domain precisions "
            "(0.023% difference). The selection of γ over 1/√3 depends on THEORETICAL "
            "meaning (discrete-continuous bridge), not on empirical fit. "
            "The 2×2 factorial tests whether Ξ-like attractors emerge preferentially "
            "in conservation + self-similarity systems."
        )
    }
    
    results["synthesis"] = synthesis
    print(f"\n  Domain comparison: {domain_winner} (by {domain_margin:.4f}%)")
    print(f"  Closest quadrant to Ξ: {closest_to_xi[0]} ({closest_to_xi[1]['dist_to_xi_pct']:.1f}%)")
    print(f"  Verdict: {synthesis['overall_verdict'][:100]}...")
    
    # PASS/FAIL
    # PASS if BOTH_ON is the closest quadrant to Ξ
    passed = closest_to_xi[0] == "BOTH_ON"
    results["PASS"] = passed
    results["status"] = (
        "PASS — conservation + self-similarity quadrant closest to Ξ" if passed
        else f"FAIL — {closest_to_xi[0]} quadrant is closer to Ξ than BOTH_ON"
    )
    
    print(f"\n  {'PASS' if passed else 'FAIL'}: {results['status']}")
    
    save_results(results, "exp_30_xi_alt_factorial")
    return results


if __name__ == "__main__":
    main()
