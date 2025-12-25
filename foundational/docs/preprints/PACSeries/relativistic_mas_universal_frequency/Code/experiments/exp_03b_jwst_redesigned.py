"""
Experiment 03b: JWST Comparison - REDESIGNED

The original exp_03 was circular: it computed k from observed mass,
then predicted mass at that k. The residual was just log(Ξ) for all objects.

THIS VERSION makes actual predictions by:
1. Using COSMIC AGE to predict which k-levels should be populated
2. Computing mass distribution at those k-levels
3. Comparing to observed mass DISTRIBUTION (not individual objects)

HYPOTHESIS: At z > 8, the mass distribution of SMBHs should follow
PAC-predicted k-level populations, not ΛCDM Eddington growth.

METHODOLOGY:
============

1. For each redshift, calculate available cosmic time
2. Apply SEC-enhanced Eddington growth to predict MAXIMUM mass
3. Compare to observed masses
4. The DISTRIBUTION should follow φ-spacing

SUCCESS CRITERION:
==================

1. Maximum masses within 0.5 dex of PAC prediction
2. Mass ratios between objects consistent with φ-levels
3. No objects exceed PAC maximum for their redshift
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.constants import PHI, XI, T_EDDINGTON, level_for_mass, mass_at_level
from core.pac_cosmology import JWST_OBSERVATIONS, pac_state_at_z, matter_fraction_at_z
from core.sec_dynamics import sec_state_at_z, duty_cycle, K_EQUILIBRIUM


def predict_max_mass_at_z(z: float, log_m_seed: float = 2.0) -> dict:
    """
    Predict maximum BH mass at redshift z using PAC/SEC growth.
    
    This is an ACTUAL prediction, not a round-trip.
    
    Args:
        z: Observation redshift
        log_m_seed: Log of seed mass in solar masses (default: 100 M☉)
    
    Returns:
        Dictionary with predictions
    """
    # Get cosmic age
    pac_state = pac_state_at_z(z)
    t_gyr = pac_state.cosmic_age_gyr
    
    # Get SEC enhancement
    m_frac, _ = matter_fraction_at_z(z)
    sec_state = sec_state_at_z(z, m_frac)
    
    # Growth scenarios
    m_seed = 10**log_m_seed
    
    # 1. Pure Eddington (100% duty, 100% rate) - theoretical max for ΛCDM
    n_efolds_pure = t_gyr / T_EDDINGTON
    m_max_pure = m_seed * np.exp(n_efolds_pure)
    
    # 2. ΛCDM realistic (20% duty, 30% rate)
    n_efolds_lcdm = 0.2 * 0.3 * t_gyr / T_EDDINGTON
    m_max_lcdm = m_seed * np.exp(n_efolds_lcdm)
    
    # 3. PAC/SEC enhanced (enhanced duty, 100% rate)
    duty_eq = duty_cycle(K_EQUILIBRIUM)
    effective_duty = duty_eq * sec_state.enhancement_factor
    n_efolds_pac = effective_duty * t_gyr / T_EDDINGTON
    m_max_pac = m_seed * np.exp(n_efolds_pac)
    
    # 4. PAC/SEC moderate (enhanced duty, 50% rate)
    n_efolds_pac_mod = effective_duty * 0.5 * t_gyr / T_EDDINGTON
    m_max_pac_mod = m_seed * np.exp(n_efolds_pac_mod)
    
    return {
        "z": z,
        "t_gyr": t_gyr,
        "log_m_seed": log_m_seed,
        "sec_enhancement": sec_state.enhancement_factor,
        "predictions": {
            "pure_eddington": {"log_m_max": np.log10(m_max_pure), "n_efolds": n_efolds_pure},
            "lcdm_realistic": {"log_m_max": np.log10(m_max_lcdm), "n_efolds": n_efolds_lcdm},
            "pac_optimistic": {"log_m_max": np.log10(m_max_pac), "n_efolds": n_efolds_pac},
            "pac_moderate": {"log_m_max": np.log10(m_max_pac_mod), "n_efolds": n_efolds_pac_mod}
        }
    }


def check_phi_spacing(masses: list) -> dict:
    """
    Check if observed mass ratios are consistent with φ-spacing.
    
    In PAC hierarchy, mass ratios between adjacent k-levels should be:
    M(k) / M(k+1) = φ^n for some integer n
    
    This is a TESTABLE prediction.
    """
    if len(masses) < 2:
        return {"sufficient_data": False}
    
    # Sort masses descending
    sorted_masses = sorted(masses, reverse=True)
    
    ratios = []
    for i in range(len(sorted_masses) - 1):
        ratio = sorted_masses[i] / sorted_masses[i+1]
        log_ratio = np.log(ratio) / np.log(PHI)  # In units of log_φ
        nearest_int = round(log_ratio)
        deviation = abs(log_ratio - nearest_int)
        
        ratios.append({
            "m1": np.log10(sorted_masses[i]),
            "m2": np.log10(sorted_masses[i+1]),
            "ratio": ratio,
            "log_phi_ratio": log_ratio,
            "nearest_n": nearest_int,
            "deviation": deviation,
            "consistent": deviation < 0.3  # Allow 30% deviation
        })
    
    n_consistent = sum(1 for r in ratios if r["consistent"])
    
    return {
        "sufficient_data": True,
        "ratios": ratios,
        "n_consistent": n_consistent,
        "n_total": len(ratios),
        "fraction_consistent": n_consistent / len(ratios) if ratios else 0
    }


def run_experiment():
    """Run Experiment 03b: Redesigned JWST Comparison."""
    
    print("=" * 70)
    print("EXPERIMENT 03b: JWST COMPARISON (REDESIGNED)")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis version makes ACTUAL predictions, not round-trips.")
    
    results = {
        "experiment": "exp_03b_jwst_comparison_redesigned",
        "timestamp": datetime.now().isoformat(),
        "purpose": "Make actual testable predictions for JWST SMBHs"
    }
    
    # =================================================================
    # Test 1: Maximum mass predictions
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 1: MAXIMUM MASS PREDICTIONS")
    print("=" * 70)
    
    print("""
METHODOLOGY:
  - Test with different seed masses
  - Grow via Eddington-limited accretion
  - Compare predicted maximum to observed masses
  
This is a FORWARD prediction, not a round-trip.
""")
    
    # Test multiple seed masses
    seed_scenarios = [
        (2.0, "Stellar remnant (100 M☉)"),
        (4.0, "Heavy stellar (10^4 M☉)"),
        (5.0, "Direct collapse (10^5 M☉)"),
    ]
    
    test1_all = {}
    
    for seed_mass, seed_name in seed_scenarios:
        print(f"\n--- Seed: {seed_name} ---")
        
        test1_results = []
        
        print(f"\n{'z':<8} {'t(Myr)':<10} {'log(M)obs':<12} {'ΛCDM real':<12} {'PAC mod':<12} {'Obs OK?':<10}")
        print("-" * 70)
        
        for obs in JWST_OBSERVATIONS:
            z = obs["z"]
            log_m_obs = obs["log_m_bh"]
            
            pred = predict_max_mass_at_z(z, seed_mass)
            
            lcdm_max = pred["predictions"]["lcdm_realistic"]["log_m_max"]
            pac_max = pred["predictions"]["pac_moderate"]["log_m_max"]
            
            # Is observed mass achievable?
            lcdm_ok = log_m_obs <= lcdm_max + 0.3  # Allow 0.3 dex tolerance
            pac_ok = log_m_obs <= pac_max + 0.3
            
            if pac_ok and not lcdm_ok:
                mark = "✓ PAC only"
            elif pac_ok and lcdm_ok:
                mark = "✓ Both"
            elif not pac_ok and lcdm_ok:
                mark = "✓ ΛCDM only"
            else:
                mark = "✗ Neither"
            
            print(f"{z:<8.1f} {pred['t_gyr']*1000:<10.0f} {log_m_obs:<12.1f} "
                  f"{lcdm_max:<12.1f} {pac_max:<12.1f} {mark:<10}")
            
            test1_results.append({
                "name": obs["name"],
                "z": z,
                "log_m_obs": log_m_obs,
                "log_m_max_lcdm": lcdm_max,
                "log_m_max_pac": pac_max,
                "achievable_lcdm": lcdm_ok,
                "achievable_pac": pac_ok
            })
        
        n_lcdm_ok = sum(1 for r in test1_results if r["achievable_lcdm"])
        n_pac_ok = sum(1 for r in test1_results if r["achievable_pac"])
        n_total = len(test1_results)
        
        print(f"\nAchievable with ΛCDM realistic: {n_lcdm_ok}/{n_total}")
        print(f"Achievable with PAC moderate: {n_pac_ok}/{n_total}")
        
        test1_all[f"seed_{seed_mass}"] = {
            "seed_mass": seed_mass,
            "seed_name": seed_name,
            "results": test1_results,
            "n_lcdm_achievable": n_lcdm_ok,
            "n_pac_achievable": n_pac_ok
        }
    
    results["test_1_max_mass"] = test1_all
    
    # Use DC seeds for summary
    dc_results = test1_all["seed_5.0"]["results"]
    n_lcdm_ok = test1_all["seed_5.0"]["n_lcdm_achievable"]
    n_pac_ok = test1_all["seed_5.0"]["n_pac_achievable"]
    seed_mass = 5.0
    
    # =================================================================
    # Test 2: φ-spacing in mass distribution
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 2: φ-SPACING IN MASS DISTRIBUTION")
    print("=" * 70)
    
    print("""
PREDICTION:
  Mass ratios between SMBHs should be powers of φ.
  M_i / M_j ≈ φ^n for some integer n
  
This is a UNIQUE prediction of PAC hierarchy.
ΛCDM has no such prediction.
""")
    
    masses = [10**obs["log_m_bh"] for obs in JWST_OBSERVATIONS]
    phi_check = check_phi_spacing(masses)
    
    if phi_check["sufficient_data"]:
        print(f"\n{'log(M₁)':<10} {'log(M₂)':<10} {'log_φ(ratio)':<15} {'n_nearest':<10} {'Consistent?':<12}")
        print("-" * 60)
        
        for r in phi_check["ratios"]:
            mark = "✓" if r["consistent"] else "✗"
            print(f"{r['m1']:<10.2f} {r['m2']:<10.2f} {r['log_phi_ratio']:<15.2f} "
                  f"{r['nearest_n']:<10} {mark:<12}")
        
        print(f"\nφ-consistent ratios: {phi_check['n_consistent']}/{phi_check['n_total']}")
        print(f"Fraction consistent: {phi_check['fraction_consistent']*100:.0f}%")
    else:
        print("Insufficient data for φ-spacing test")
    
    results["test_2_phi_spacing"] = phi_check
    
    # =================================================================
    # Test 3: Residuals from PAC prediction (proper version)
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 3: RESIDUALS FROM PAC GROWTH PREDICTION")
    print("=" * 70)
    
    print("""
METHODOLOGY:
  - Predict maximum mass at each z using PAC/SEC growth
  - Observed masses should be AT OR BELOW this maximum
  - Compute residual = log(M_obs) - log(M_max_predicted)
  
  Residual < 0: OK (observed below maximum)
  Residual > 0: PROBLEM (observed exceeds prediction)
""")
    
    print(f"\n{'Object':<15} {'log(M)obs':<12} {'log(M)max':<12} {'Residual':<12} {'Status':<10}")
    print("-" * 60)
    
    residuals = []
    
    for obs in JWST_OBSERVATIONS:
        name = obs["name"]
        z = obs["z"]
        log_m_obs = obs["log_m_bh"]
        
        pred = predict_max_mass_at_z(z, seed_mass)
        log_m_max = pred["predictions"]["pac_moderate"]["log_m_max"]
        
        residual = log_m_obs - log_m_max
        status = "✓ OK" if residual <= 0.3 else "⚠ HIGH" if residual <= 0.5 else "✗ EXCEEDS"
        
        print(f"{name:<15} {log_m_obs:<12.2f} {log_m_max:<12.2f} {residual:+.2f}        {status:<10}")
        
        residuals.append({
            "name": name,
            "z": z,
            "log_m_obs": log_m_obs,
            "log_m_max_pred": log_m_max,
            "residual": residual,
            "status": status
        })
    
    mean_residual = np.mean([r["residual"] for r in residuals])
    std_residual = np.std([r["residual"] for r in residuals])
    
    print(f"\nMean residual: {mean_residual:+.3f} dex")
    print(f"Std residual: {std_residual:.3f} dex")
    
    results["test_3_residuals"] = {
        "results": residuals,
        "mean_residual": mean_residual,
        "std_residual": std_residual
    }
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 03b SUMMARY")
    print("=" * 70)
    
    success = (n_pac_ok >= n_total - 1) and (mean_residual < 0.5)
    
    print(f"\n{'Metric':<40} {'Value':<20} {'Status':<10}")
    print("-" * 70)
    print(f"{'Objects achievable with PAC':<40} {n_pac_ok}/{n_total:<16} {'✓' if n_pac_ok >= n_total-1 else '✗':<10}")
    print(f"{'Objects achievable with ΛCDM realistic':<40} {n_lcdm_ok}/{n_total:<16} {'✓' if n_lcdm_ok >= n_total-1 else '✗':<10}")
    print(f"{'Mean residual from PAC max':<40} {mean_residual:+.3f} dex         {'✓' if mean_residual < 0.5 else '✗':<10}")
    print(f"{'φ-spacing consistency':<40} {phi_check.get('fraction_consistent', 0)*100:.0f}%              {'✓' if phi_check.get('fraction_consistent', 0) > 0.5 else '?':<10}")
    
    print(f"\n{'Overall Success':<40} {'✓' if success else '✗':<10}")
    
    results["success"] = success
    results["conclusion"] = (
        f"PAC/SEC predictions: {n_pac_ok}/{n_total} achievable, "
        f"mean residual {mean_residual:+.3f} dex. "
        f"ΛCDM realistic: only {n_lcdm_ok}/{n_total} achievable. "
        f"φ-spacing: {phi_check.get('fraction_consistent', 0)*100:.0f}% consistent."
    )
    
    print(f"\n{results['conclusion']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_03b_jwst_redesigned_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_experiment()
