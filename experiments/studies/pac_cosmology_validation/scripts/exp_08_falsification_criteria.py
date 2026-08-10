"""
Experiment 08: Quantitative Falsification Criteria

For PAC/SEC cosmology to be scientific, it must make predictions that could be wrong.
This experiment establishes concrete, testable criteria.

WHAT MAKES A GOOD FALSIFICATION CRITERION:
==========================================

1. SPECIFIC: Numerical prediction, not directional ("higher" vs "10.5 ± 0.3")
2. DISTINGUISHING: Different from ΛCDM prediction by >2σ
3. OBSERVABLE: Can be measured with current/near-future instruments
4. DERIVED: Comes from theory, not fitted to data

PAC/SEC PREDICTIONS TO TEST:
============================

1. Seed mass requirements (from exp_07)
   - PAC/SEC: allows stellar seeds (10-100 M☉) at z>10
   - ΛCDM realistic: requires 10^6+ M☉ seeds at z>10
   
2. Duty cycle evolution
   - PAC/SEC: duty increases with z (61.8% → 72.3%)
   - ΛCDM: no prediction (free parameter)

3. Enhancement factor
   - PAC/SEC: exactly 1.17× at z=10 (derived from φ)
   - ΛCDM: any value can be fitted

4. Transition redshift
   - PAC/SEC: enhancement saturates around z_eq where Ω_m = 1/φ²
   - Should see discontinuity in inferred accretion rates

FALSIFICATION CONDITIONS:
=========================

IF we observe:
- Seed masses inconsistent with stellar/DC at z>10 → PAC/SEC wrong about enhancement
- Enhancement ≠ 1.17× (outside 1.10-1.25 range) → SEC dynamics wrong
- No φ-scaling in any observable → PAC cosmology wrong
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.constants import PHI, T_EDDINGTON
from core.pac_cosmology import JWST_OBSERVATIONS, pac_state_at_z, matter_fraction_at_z
from core.sec_dynamics import sec_state_at_z, duty_cycle, K_EQUILIBRIUM


def run_experiment():
    """Run Experiment 08: Quantitative Falsification Criteria."""
    
    print("=" * 70)
    print("EXPERIMENT 08: QUANTITATIVE FALSIFICATION CRITERIA")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "experiment": "exp_08_falsification_criteria",
        "timestamp": datetime.now().isoformat(),
        "purpose": "Establish concrete falsification criteria for PAC/SEC cosmology"
    }
    
    # =================================================================
    # Criterion 1: Enhancement Factor
    # =================================================================
    print("\n" + "=" * 70)
    print("CRITERION 1: SEC ENHANCEMENT FACTOR")
    print("=" * 70)
    
    print("""
PREDICTION:
  At z > 8, the growth enhancement from SEC dynamics is 1.17× ± 0.05
  
DERIVATION:
  - Run-length ratio R = φ^(1 + (k_eq - k)/2)
  - At high z: k → 0, so R → φ^2 = 2.618
  - Duty cycle = R/(R+1) = 72.3%
  - Enhancement = 72.3% / 61.8% = 1.170
  
UNCERTAINTY:
  - k_eq = 2.0 (could be 1.8-2.2 based on Ω_m uncertainty)
  - This gives enhancement range: 1.15 - 1.20
  - Conservative: 1.17 ± 0.05

FALSIFICATION:
  If inferred enhancement from future observations is:
  - < 1.10: PAC/SEC dynamics WRONG (insufficient enhancement)
  - > 1.25: PAC/SEC dynamics WRONG (too much enhancement)
  - Between 1.10 and 1.25: CONSISTENT with theory
""")
    
    # Calculate theoretical range
    k_eq_nominal = 2.0
    k_eq_low, k_eq_high = 1.8, 2.2
    
    def enhancement_at_k_eq(k_eq):
        R_eq = PHI
        duty_eq = R_eq / (R_eq + 1)
        R_high_z = PHI ** (1 + k_eq/2)
        duty_high = R_high_z / (R_high_z + 1)
        return duty_high / duty_eq
    
    enh_nominal = enhancement_at_k_eq(k_eq_nominal)
    enh_low = enhancement_at_k_eq(k_eq_low)
    enh_high = enhancement_at_k_eq(k_eq_high)
    
    print(f"Theoretical predictions:")
    print(f"  k_eq = {k_eq_low}: enhancement = {enh_low:.3f}")
    print(f"  k_eq = {k_eq_nominal}: enhancement = {enh_nominal:.3f}")
    print(f"  k_eq = {k_eq_high}: enhancement = {enh_high:.3f}")
    print(f"\n  Allowed range: {min(enh_low, enh_high):.2f} - {max(enh_low, enh_high):.2f}")
    
    criterion_1 = {
        "name": "SEC Enhancement Factor",
        "prediction": enh_nominal,
        "uncertainty": 0.05,
        "range_low": 1.10,
        "range_high": 1.25,
        "falsification": "Enhancement outside [1.10, 1.25] falsifies SEC dynamics"
    }
    
    results["criterion_1"] = criterion_1
    
    # =================================================================
    # Criterion 2: Maximum Viable Seed Mass
    # =================================================================
    print("\n" + "=" * 70)
    print("CRITERION 2: MAXIMUM VIABLE SEED MASS AT z > 10")
    print("=" * 70)
    
    print("""
PREDICTION:
  For objects observed at z > 10 with M_BH > 10^6 M☉:
  - PAC/SEC allows seeds < 10^5 M☉ (direct collapse regime)
  - ΛCDM realistic requires seeds > 10^6 M☉ (problematic)
  
DERIVATION:
  From exp_07, with PAC/SEC moderate assumptions:
  - UHZ-1: seed ~ 10^5.9 M☉ (DC regime)
  - GLASS-z12: seed ~ 10^4.8 M☉ (DC regime)
  
FALSIFICATION:
  If independent measurements show these objects required seeds > 10^6 M☉:
  - This would falsify PAC/SEC (enhancement insufficient)
  
  If measurements show seeds were < 10^3 M☉:
  - This would SUPPORT PAC/SEC (or require even more enhancement)
""")
    
    # Calculate seed mass constraints for z>10 objects
    high_z_objects = [obs for obs in JWST_OBSERVATIONS if obs["z"] > 10]
    
    print(f"\nHigh-z objects (z > 10):")
    print(f"{'Object':<15} {'z':<8} {'log(M_obs)':<12} {'PAC seed':<12} {'ΛCDM seed':<12}")
    print("-" * 60)
    
    seed_predictions = []
    for obs in high_z_objects:
        name = obs["name"]
        z = obs["z"]
        log_m = obs["log_m_bh"]
        m_obs = 10**log_m
        
        pac_state = pac_state_at_z(z)
        t_gyr = pac_state.cosmic_age_gyr
        
        m_frac, _ = matter_fraction_at_z(z)
        sec_state = sec_state_at_z(z, m_frac)
        
        duty_eq = duty_cycle(K_EQUILIBRIUM)
        effective_duty = duty_eq * sec_state.enhancement_factor
        
        # PAC/SEC moderate (enhanced duty, 50% Eddington)
        n_efolds_pac = effective_duty * 0.5 * t_gyr / T_EDDINGTON
        log_seed_pac = log_m - n_efolds_pac / np.log(10)
        
        # ΛCDM realistic (20% duty, 30% Eddington)
        n_efolds_lcdm = 0.2 * 0.3 * t_gyr / T_EDDINGTON
        log_seed_lcdm = log_m - n_efolds_lcdm / np.log(10)
        
        print(f"{name:<15} {z:<8.1f} {log_m:<12.1f} {log_seed_pac:<12.1f} {log_seed_lcdm:<12.1f}")
        
        seed_predictions.append({
            "name": name,
            "z": z,
            "log_m_obs": log_m,
            "log_seed_pac": log_seed_pac,
            "log_seed_lcdm": log_seed_lcdm
        })
    
    criterion_2 = {
        "name": "Maximum Seed Mass",
        "prediction": "PAC/SEC allows seeds in 10^4-10^6 M☉ range for z>10 SMBHs",
        "threshold": 6.0,  # log M☉
        "falsification": "If seeds must exceed 10^6 M☉, PAC/SEC is insufficient",
        "predictions": seed_predictions
    }
    
    results["criterion_2"] = criterion_2
    
    # =================================================================
    # Criterion 3: Duty Cycle Evolution with Redshift
    # =================================================================
    print("\n" + "=" * 70)
    print("CRITERION 3: DUTY CYCLE EVOLUTION")
    print("=" * 70)
    
    print("""
PREDICTION:
  The effective accretion duty cycle should increase with redshift:
  - z = 0: duty ~ 59.5% (below equilibrium)
  - z = 2: duty ~ 71.2%
  - z > 8: duty saturates at ~ 72.3%
  
OBSERVATIONAL TEST:
  Compare AGN duty cycles at different redshifts
  - Low z: measure from AGN luminosity functions
  - High z: infer from SMBH demographics
  
  If duty DECREASES with z, or stays constant: PAC/SEC WRONG
  If duty INCREASES with z: CONSISTENT
""")
    
    z_range = [0, 1, 2, 4, 6, 8, 10, 12, 15]
    duty_evolution = []
    
    print(f"\n{'z':<8} {'k_level':<12} {'duty (%)':<12} {'enhancement':<12}")
    print("-" * 50)
    
    for z in z_range:
        m_frac, _ = matter_fraction_at_z(z)
        state = sec_state_at_z(z, m_frac)
        
        print(f"{z:<8} {state.k_level:<12.3f} {state.duty_cycle*100:<12.1f} {state.enhancement_factor:<12.3f}")
        
        duty_evolution.append({
            "z": z,
            "k_level": state.k_level,
            "duty_cycle": state.duty_cycle,
            "enhancement": state.enhancement_factor
        })
    
    criterion_3 = {
        "name": "Duty Cycle Evolution",
        "prediction": "Duty cycle increases from ~60% at z=0 to ~72% at z>8",
        "z_low_duty": 0.595,
        "z_high_duty": 0.723,
        "falsification": "If observed duty cycle decreases or stays constant with z",
        "evolution": duty_evolution
    }
    
    results["criterion_3"] = criterion_3
    
    # =================================================================
    # Criterion 4: φ-Scaling Signature
    # =================================================================
    print("\n" + "=" * 70)
    print("CRITERION 4: φ-SCALING SIGNATURE")
    print("=" * 70)
    
    print(f"""
PREDICTION:
  The enhancement factor should involve φ = {PHI:.4f} specifically:
  - Not 1.5, not 1.7, but ~1.618
  - Run-length ratio at high z should be φ² = {PHI**2:.4f}
  
OBSERVATIONAL TEST:
  If we can measure the run-length ratio of accretion variability:
  - ratio of "on" to "off" times
  - Should approach φ² at high z, φ at low z
  
FALSIFICATION:
  If the observed ratio is significantly different from φ or φ²:
  - e.g., ratio = 2.0 (not 1.618) at z=0
  - e.g., ratio = 3.0 (not 2.618) at z=10
  This would falsify the SEC mechanism
""")
    
    print(f"\nPredicted run-length ratios:")
    print(f"  z = 0 (near equilibrium): R = {PHI:.4f}")
    print(f"  z = 10 (high z): R = φ² = {PHI**2:.4f}")
    print(f"  z → ∞ (limit): R → φ² = {PHI**2:.4f}")
    
    criterion_4 = {
        "name": "φ-Scaling Signature",
        "prediction": f"Run-length ratio = φ = {PHI:.4f} at z~0, φ² = {PHI**2:.4f} at z>8",
        "phi": PHI,
        "phi_squared": PHI**2,
        "falsification": "If observed ratios differ significantly from φ or φ²"
    }
    
    results["criterion_4"] = criterion_4
    
    # =================================================================
    # Summary Table
    # =================================================================
    print("\n" + "=" * 70)
    print("FALSIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ CRITERION                │ PREDICTION        │ FALSIFICATION        │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Enhancement factor    │ 1.17 ± 0.05       │ Outside [1.10, 1.25] │
│ 2. Seed mass at z>10     │ < 10^6 M☉         │ Requires > 10^6 M☉   │
│ 3. Duty cycle evolution  │ Increases with z  │ Decreases or flat    │
│ 4. Run-length ratio      │ φ at z=0, φ² high │ ≠ φ or φ²           │
└─────────────────────────────────────────────────────────────────────┘

IMPORTANT: These are DERIVED predictions, not fits.

If ANY criterion is falsified, it indicates:
- Criterion 1 failure: SEC phase transition dynamics wrong
- Criterion 2 failure: Enhancement insufficient, need different mechanism
- Criterion 3 failure: PAC-to-k mapping wrong
- Criterion 4 failure: φ not fundamental to the dynamics
""")
    
    results["summary"] = {
        "n_criteria": 4,
        "all_derived": True,
        "distinguishing": True,
        "note": "These predictions differ from ΛCDM which has no specific duty cycle or enhancement predictions"
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_08_falsification_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_experiment()
