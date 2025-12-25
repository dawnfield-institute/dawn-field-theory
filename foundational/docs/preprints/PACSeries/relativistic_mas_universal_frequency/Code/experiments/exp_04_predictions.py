"""
Experiment 04: Future Predictions

Generates falsifiable predictions for future JWST/high-z observations.

HYPOTHESIS: PAC cosmology makes specific, testable predictions about
SMBHs at z > 15 that can be falsified by future observations.

METHODOLOGY:
1. Extrapolate PAC framework to z = 15, 17, 20
2. Predict maximum SMBH mass at each redshift
3. Predict MBH/M* ratio evolution
4. Define clear falsification criteria

SUCCESS CRITERION: Predictions are well-defined and falsifiable.

PREDICTIONS STRUCTURE:
- UPPER LIMITS: Maximum mass possible at given z (violating means PAC fails)
- EXPECTED RANGE: Where most SMBHs should be found
- RATIOS: Expected MBH/M* relationships
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.constants import (
    PHI, XI, CONTEXT_VARIANCE, M_GALAXY_SCALE,
    PAC_FRACTION as PAC, SEC_FRACTION as SEC, 
    level_for_mass, mass_at_level
)
from core.pac_cosmology import (
    pac_state_at_z,
    pac_mbh_mstar_ratio,
    pac_rate_enhancement,
    relativistic_time_dilation,
    cosmic_age_at_z
)

# Derived constant
LOG_M0 = np.log10(M_GALAXY_SCALE * PHI**(-8))


def predict_max_mass_at_z(z: float) -> dict:
    """
    Predict maximum SMBH mass possible at redshift z.
    
    Uses PAC recursion + time constraints to determine
    what mass could physically form.
    """
    state = pac_state_at_z(z)
    
    # Time dilation gives effective age
    time_dilation = relativistic_time_dilation(z)
    effective_age = state.cosmic_age_gyr * time_dilation
    
    # PAC enhancement factor for accretion
    enhancement = pac_rate_enhancement(z)
    
    # The most massive object that could form:
    # Starting from seed mass, growing at enhanced Eddington
    # for effective_age with PAC enhancement
    
    # Eddington e-folding time ~ 45 Myr
    t_edd = 0.045  # Gyr
    
    # Number of e-foldings possible
    n_efolds = (effective_age / t_edd) * enhancement * PAC
    
    # Maximum mass
    log_m_max = LOG_M0 + n_efolds * np.log10(np.e)
    
    # Cap at physical limits
    log_m_max = min(log_m_max, 11.0)  # 10^11 M_sun absolute cap
    
    # Convert to k-level
    k_min = level_for_mass(log_m_max)
    
    return {
        "z": z,
        "cosmic_age_gyr": state.cosmic_age_gyr,
        "effective_age_gyr": effective_age,
        "time_dilation": time_dilation,
        "pac_enhancement": enhancement,
        "n_efolds": n_efolds,
        "log_m_max": log_m_max,
        "k_min": k_min
    }


def predict_typical_mass_range(z: float) -> dict:
    """
    Predict typical SMBH mass range at redshift z.
    
    Most SMBHs should fall within 1-2 dex of this range.
    """
    max_pred = predict_max_mass_at_z(z)
    
    # Typical objects are 1-2 levels below maximum
    log_m_typical_high = max_pred["log_m_max"] - 0.5
    log_m_typical_low = max_pred["log_m_max"] - 2.0
    
    # Floor at seed mass
    log_m_typical_low = max(log_m_typical_low, LOG_M0)
    
    return {
        "z": z,
        "log_m_typical_low": log_m_typical_low,
        "log_m_typical_high": log_m_typical_high,
        "log_m_max": max_pred["log_m_max"],
        "k_range": [level_for_mass(log_m_typical_high), level_for_mass(log_m_typical_low)]
    }


def predict_mbh_mstar_evolution() -> list:
    """Predict MBH/M* ratio as function of z."""
    results = []
    
    for z in [0, 2, 5, 8, 10, 12, 15, 17, 20]:
        ratio = pac_mbh_mstar_ratio(z)
        state = pac_state_at_z(z)
        
        results.append({
            "z": z,
            "mbh_mstar_ratio": ratio,
            "log_ratio": np.log10(ratio),
            "cosmic_age_gyr": state.cosmic_age_gyr,
            "matter_fraction": state.matter_fraction
        })
    
    return results


def generate_falsification_criteria() -> dict:
    """
    Generate specific falsification criteria for PAC cosmology.
    
    If these are violated, PAC is falsified.
    """
    
    criteria = []
    
    # Criterion 1: Maximum mass limits
    for z in [15, 17, 20]:
        pred = predict_max_mass_at_z(z)
        
        criteria.append({
            "id": f"MAX_MASS_z{z}",
            "statement": f"No SMBH at z > {z} should exceed 10^{pred['log_m_max']:.1f} M_sun",
            "falsified_if": f"Any confirmed SMBH at z > {z} with M > 10^{pred['log_m_max'] + 0.5:.1f} M_sun",
            "predicted_limit_log_m": pred["log_m_max"]
        })
    
    # Criterion 2: Minimum cosmic age for mass
    criteria.append({
        "id": "AGE_MASS_RELATION",
        "statement": "SMBHs require minimum cosmic age proportional to log(M) - log(M_seed)",
        "falsified_if": "SMBH with mass requiring growth time > cosmic age even with PAC enhancement"
    })
    
    # Criterion 3: MBH/M* ratio evolution
    criteria.append({
        "id": "RATIO_EVOLUTION",
        "statement": "MBH/M* ratio should increase with z (from ~0.001 at z=0 to ~0.1 at z=15)",
        "falsified_if": "Systematic evidence of MBH/M* DECREASING with z at z > 8"
    })
    
    # Criterion 4: K-level quantization
    criteria.append({
        "id": "K_QUANTIZATION",
        "statement": "SMBH masses should cluster near φ-spaced mass levels",
        "falsified_if": "Large sample showing uniform mass distribution (no φ-clustering)"
    })
    
    # Criterion 5: PAC/SEC ratio
    criteria.append({
        "id": "PAC_SEC_BALANCE",
        "statement": "~80% of SMBHs should show net mass growth (PAC > SEC)",
        "falsified_if": "Majority of high-z SMBHs showing mass loss signatures"
    })
    
    return {"falsification_criteria": criteria}


def run_experiment():
    """Run Experiment 04: Future Predictions."""
    
    print("=" * 70)
    print("EXPERIMENT 04: FUTURE PREDICTIONS")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "experiment": "exp_04_predictions",
        "timestamp": datetime.now().isoformat(),
        "purpose": "Generate falsifiable predictions for z > 15 observations"
    }
    
    # =================================================================
    # Prediction Set 1: Maximum masses at high z
    # =================================================================
    print("\n" + "-" * 50)
    print("PREDICTION SET 1: Maximum SMBH masses at high z")
    print("-" * 50)
    
    print("\nUsing PAC cosmology with fixed constants:")
    print(f"  φ = {PHI:.6f}")
    print(f"  Ξ = {XI:.6f}")
    print(f"  Context variance = {CONTEXT_VARIANCE}")
    
    print(f"\n{'z':<6} {'Age(Gyr)':<10} {'Eff.Age':<10} {'Enhance':<10} {'log(M)max':<12} {'k_min':<8}")
    print("-" * 60)
    
    max_mass_preds = []
    for z in [10, 12, 15, 17, 20, 25, 30]:
        pred = predict_max_mass_at_z(z)
        max_mass_preds.append(pred)
        
        print(f"{z:<6} {pred['cosmic_age_gyr']:<10.3f} {pred['effective_age_gyr']:<10.3f} "
              f"{pred['pac_enhancement']:<10.2f} {pred['log_m_max']:<12.2f} {pred['k_min']:<8.2f}")
    
    results["max_mass_predictions"] = max_mass_preds
    
    # =================================================================
    # Prediction Set 2: Typical mass ranges
    # =================================================================
    print("\n" + "-" * 50)
    print("PREDICTION SET 2: Expected mass ranges at high z")
    print("-" * 50)
    
    print(f"\n{'z':<6} {'Typical Range (log M)':<25} {'Max (log M)':<15}")
    print("-" * 50)
    
    range_preds = []
    for z in [10, 12, 15, 17, 20]:
        pred = predict_typical_mass_range(z)
        range_preds.append(pred)
        
        print(f"{z:<6} [{pred['log_m_typical_low']:.2f}, {pred['log_m_typical_high']:.2f}]"
              f"          {pred['log_m_max']:.2f}")
    
    results["range_predictions"] = range_preds
    
    # =================================================================
    # Prediction Set 3: MBH/M* ratio evolution
    # =================================================================
    print("\n" + "-" * 50)
    print("PREDICTION SET 3: MBH/M* ratio evolution")
    print("-" * 50)
    
    ratio_preds = predict_mbh_mstar_evolution()
    
    print(f"\n{'z':<6} {'MBH/M*':<15} {'log(ratio)':<12}")
    print("-" * 35)
    
    for r in ratio_preds:
        print(f"{r['z']:<6} {r['mbh_mstar_ratio']:<15.4f} {r['log_ratio']:<12.2f}")
    
    results["ratio_predictions"] = ratio_preds
    
    # =================================================================
    # Falsification Criteria
    # =================================================================
    print("\n" + "-" * 50)
    print("FALSIFICATION CRITERIA")
    print("-" * 50)
    print("\nPAC cosmology would be FALSIFIED if:")
    
    falsif = generate_falsification_criteria()
    
    for i, criterion in enumerate(falsif["falsification_criteria"], 1):
        print(f"\n{i}. [{criterion['id']}]")
        print(f"   Statement: {criterion['statement']}")
        print(f"   Falsified if: {criterion['falsified_if']}")
    
    results["falsification_criteria"] = falsif["falsification_criteria"]
    
    # =================================================================
    # Specific Testable Predictions
    # =================================================================
    print("\n" + "-" * 50)
    print("SPECIFIC TESTABLE PREDICTIONS")
    print("-" * 50)
    
    specific = [
        {
            "id": "PRED_01",
            "prediction": f"At z=15, max SMBH mass ≈ 10^{predict_max_mass_at_z(15)['log_m_max']:.1f} M_sun",
            "uncertainty": "±0.5 dex"
        },
        {
            "id": "PRED_02", 
            "prediction": f"At z=20, max SMBH mass ≈ 10^{predict_max_mass_at_z(20)['log_m_max']:.1f} M_sun",
            "uncertainty": "±0.5 dex"
        },
        {
            "id": "PRED_03",
            "prediction": "MBH/M* ratio at z=15 should be ~10x higher than z=0",
            "uncertainty": "Factor of 3"
        },
        {
            "id": "PRED_04",
            "prediction": "SMBH mass function should show φ-spaced peaks",
            "uncertainty": "Requires sample of N>20 SMBHs"
        },
        {
            "id": "PRED_05",
            "prediction": "Time dilation factor 7.42 should affect AGN variability timescales",
            "uncertainty": "Needs long-baseline monitoring"
        }
    ]
    
    for pred in specific:
        print(f"\n[{pred['id']}] {pred['prediction']}")
        print(f"  Uncertainty: {pred['uncertainty']}")
    
    results["specific_predictions"] = specific
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 04 SUMMARY")
    print("=" * 70)
    
    print("\nPAC cosmology generates falsifiable predictions for future observations.")
    print("Key predictions are bounded by φ, Ξ, and 7.42 context variance.")
    print("\nNext steps:")
    print("1. Monitor JWST announcements for z > 12 SMBHs")
    print("2. Compare new discoveries against maximum mass limits")
    print("3. Track MBH/M* ratios as sample grows")
    print("4. Test φ-clustering when N > 20 SMBHs available")
    
    results["success"] = True  # Predictions successfully generated
    results["conclusion"] = (
        "PAC cosmology makes specific, falsifiable predictions for z > 15 SMBHs. "
        f"Maximum mass at z=15: 10^{predict_max_mass_at_z(15)['log_m_max']:.1f} M_sun. "
        f"Maximum mass at z=20: 10^{predict_max_mass_at_z(20)['log_m_max']:.1f} M_sun. "
        "Framework is testable with future JWST observations."
    )
    
    print(f"\n{results['conclusion']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_04_predictions_{timestamp}.json"
    
    with open(results_dir / filename, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: results/{filename}")
    
    return results


if __name__ == "__main__":
    run_experiment()
