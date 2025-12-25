"""
Experiment 05: Eddington vs PAC Comparison

This is the REAL test: Can standard Eddington growth explain JWST SMBHs?
If not, does PAC enhancement fix it?

BACKGROUND:
Standard Eddington-limited growth: M(t) = M_seed × exp(t/t_Edd)
  - t_Edd = 45 Myr (Salpeter time at efficiency η=0.1)
  - Starting from stellar seed: M_seed ~ 100 M☉
  - Or direct collapse seed: M_seed ~ 10^4-10^5 M☉

PROBLEM: 
At z=10, cosmic age ≈ 470 Myr = 10.4 × t_Edd
M_max = 100 × exp(10.4) = 100 × 33,000 = 3.3 × 10^6 M☉
But UHZ-1 has 10^7.5 = 3 × 10^7 M☉ → 10× too massive!

PAC CLAIM: Enhancement factor allows faster growth.

⚠️ CRITICAL NOTE:
The PAC enhancement (2.6× at z=10) reduces seed requirements by 7 dex
due to exponential amplification. This seems too powerful.

The mechanism for enhancement is: "unactualized potential enables faster
accretion." This needs explicit physical justification - currently it's
a mathematical consequence of the PAC → 1 limit, not derived from physics.

OPEN QUESTION: Is there a physical mechanism for enhanced accretion in
the attraction-dominated phase, or is this just a fitting parameter?
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.constants import (
    PHI, XI, T_EDDINGTON, PAC_FRACTION,
    M_SUN
)
from core.pac_cosmology import (
    JWST_OBSERVATIONS, 
    pac_state_at_z,
    pac_rate_enhancement
)


def eddington_growth(m_seed: float, time_gyr: float, 
                      efficiency: float = 0.1,
                      duty_cycle: float = 1.0) -> float:
    """
    Standard Eddington-limited growth.
    
    M(t) = M_seed × exp(t / t_Edd)
    
    where t_Edd = 0.45 Gyr × (η / 0.1)
    """
    t_edd = T_EDDINGTON / efficiency * 0.1  # Scale with efficiency
    n_efolds = (time_gyr / t_edd) * duty_cycle
    return m_seed * np.exp(n_efolds)


def required_seed_eddington(m_final: float, time_gyr: float,
                             efficiency: float = 0.1,
                             duty_cycle: float = 1.0) -> float:
    """
    What seed mass is needed to reach M_final in time_gyr?
    """
    t_edd = T_EDDINGTON / efficiency * 0.1
    n_efolds = (time_gyr / t_edd) * duty_cycle
    return m_final / np.exp(n_efolds)


def pac_enhanced_growth(m_seed: float, time_gyr: float, z: float) -> float:
    """
    PAC-enhanced growth: faster due to attraction-dominated early universe.
    """
    enhancement = pac_rate_enhancement(z)
    t_edd_eff = T_EDDINGTON / enhancement
    n_efolds = time_gyr / t_edd_eff
    return m_seed * np.exp(n_efolds)


def required_seed_pac(m_final: float, time_gyr: float, z: float) -> float:
    """
    What seed mass is needed with PAC enhancement?
    """
    enhancement = pac_rate_enhancement(z)
    t_edd_eff = T_EDDINGTON / enhancement
    n_efolds = time_gyr / t_edd_eff
    return m_final / np.exp(n_efolds)


def run_experiment():
    """Run Experiment 05: Eddington vs PAC Comparison."""
    
    print("=" * 70)
    print("EXPERIMENT 05: EDDINGTON vs PAC COMPARISON")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "experiment": "exp_05_eddington_comparison",
        "timestamp": datetime.now().isoformat(),
        "purpose": "Compare Eddington growth to PAC-enhanced growth"
    }
    
    # =================================================================
    # Physical parameters
    # =================================================================
    print("\n" + "-" * 50)
    print("Physical Parameters")
    print("-" * 50)
    
    print(f"\nEddington parameters:")
    print(f"  t_Edd (Salpeter) = {T_EDDINGTON * 1000:.0f} Myr")
    print(f"  Radiative efficiency η = 0.1")
    
    print(f"\nSeed mass scenarios:")
    print(f"  Stellar remnant: 10^2 M☉")
    print(f"  Direct collapse: 10^4-10^5 M☉")
    print(f"  Heavy seed: 10^6 M☉")
    
    results["parameters"] = {
        "t_edd_myr": T_EDDINGTON * 1000,
        "seeds": {"stellar": 100, "dc_low": 1e4, "dc_high": 1e5, "heavy": 1e6}
    }
    
    # =================================================================
    # Test 1: What does Eddington predict for each JWST object?
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 1: Eddington predictions (no PAC)")
    print("-" * 50)
    
    print(f"\nStarting from M_seed = 100 M☉ (Pop III remnant):")
    print(f"\n{'Object':<15} {'z':<6} {'Age(Myr)':<10} {'Edd pred':<12} {'Observed':<12} {'Deficit':<10}")
    print("-" * 70)
    
    eddington_results = []
    
    for obs in JWST_OBSERVATIONS:
        name = obs["name"]
        z = obs["z"]
        log_m_obs = obs["log_m_bh"]
        m_obs = 10**log_m_obs
        
        state = pac_state_at_z(z)
        age_myr = state.cosmic_age_gyr * 1000
        
        # Eddington prediction from stellar seed
        m_edd = eddington_growth(100, state.cosmic_age_gyr)
        log_m_edd = np.log10(m_edd)
        
        # How many dex short?
        deficit = log_m_obs - log_m_edd
        
        print(f"{name:<15} {z:<6.1f} {age_myr:<10.0f} {log_m_edd:<12.2f} {log_m_obs:<12.2f} {deficit:+.2f} dex")
        
        eddington_results.append({
            "name": name,
            "z": z,
            "age_myr": age_myr,
            "log_m_eddington": log_m_edd,
            "log_m_observed": log_m_obs,
            "deficit_dex": deficit
        })
    
    results["test_1_eddington"] = eddington_results
    
    # =================================================================
    # Test 2: What seed masses does Eddington require?
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 2: Required seed masses (Eddington only)")
    print("-" * 50)
    
    print(f"\n{'Object':<15} {'z':<6} {'M_obs':<12} {'Seed needed':<15} {'Physical?':<12}")
    print("-" * 60)
    
    seed_results = []
    
    for obs in JWST_OBSERVATIONS:
        name = obs["name"]
        z = obs["z"]
        log_m_obs = obs["log_m_bh"]
        m_obs = 10**log_m_obs
        
        state = pac_state_at_z(z)
        
        m_seed = required_seed_eddington(m_obs, state.cosmic_age_gyr)
        log_m_seed = np.log10(m_seed)
        
        # Is this seed physically reasonable?
        if m_seed <= 100:
            physical = "Stellar OK"
        elif m_seed <= 1e5:
            physical = "Direct collapse"
        elif m_seed <= 1e6:
            physical = "Heavy seed"
        else:
            physical = "IMPOSSIBLE"
        
        print(f"{name:<15} {z:<6.1f} 10^{log_m_obs:.1f}       10^{log_m_seed:.1f} M☉       {physical:<12}")
        
        seed_results.append({
            "name": name,
            "z": z,
            "log_m_observed": log_m_obs,
            "log_m_seed_required": log_m_seed,
            "physical": physical
        })
    
    results["test_2_seeds"] = seed_results
    
    # =================================================================
    # Test 3: PAC enhancement factor
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 3: PAC enhancement analysis")
    print("-" * 50)
    
    print(f"\n{'Object':<15} {'z':<6} {'PAC enhance':<12} {'Eff t_Edd':<12} {'PAC seed':<12} {'Reduction':<10}")
    print("-" * 70)
    
    pac_results = []
    
    for obs in JWST_OBSERVATIONS:
        name = obs["name"]
        z = obs["z"]
        log_m_obs = obs["log_m_bh"]
        m_obs = 10**log_m_obs
        
        state = pac_state_at_z(z)
        
        # PAC enhancement
        enhancement = pac_rate_enhancement(z)
        t_edd_eff = T_EDDINGTON * 1000 / enhancement  # Myr
        
        # Required seed with PAC
        m_seed_pac = required_seed_pac(m_obs, state.cosmic_age_gyr, z)
        log_m_seed_pac = np.log10(m_seed_pac)
        
        # Required seed without PAC
        m_seed_edd = required_seed_eddington(m_obs, state.cosmic_age_gyr)
        log_m_seed_edd = np.log10(m_seed_edd)
        
        reduction = log_m_seed_edd - log_m_seed_pac
        
        print(f"{name:<15} {z:<6.1f} {enhancement:<12.2f} {t_edd_eff:<12.1f} 10^{log_m_seed_pac:<8.1f} {reduction:+.1f} dex")
        
        pac_results.append({
            "name": name,
            "z": z,
            "enhancement": enhancement,
            "t_edd_effective_myr": t_edd_eff,
            "log_m_seed_pac": log_m_seed_pac,
            "log_m_seed_eddington": log_m_seed_edd,
            "seed_reduction_dex": reduction
        })
    
    results["test_3_pac"] = pac_results
    
    # =================================================================
    # Test 4: What's the critical question?
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 4: Critical Assessment")
    print("-" * 50)
    
    # Average enhancement
    avg_enhancement = np.mean([r["enhancement"] for r in pac_results])
    avg_reduction = np.mean([r["seed_reduction_dex"] for r in pac_results])
    
    print(f"\nAverage PAC enhancement: {avg_enhancement:.2f}×")
    print(f"Average seed mass reduction: {avg_reduction:.1f} dex")
    
    # What seed does PAC require on average?
    avg_pac_seed = np.mean([r["log_m_seed_pac"] for r in pac_results])
    print(f"Average PAC seed requirement: 10^{avg_pac_seed:.1f} M☉")
    
    # Is this physically reasonable?
    print("\n" + "-" * 30)
    print("PHYSICAL ASSESSMENT:")
    print("-" * 30)
    
    if avg_pac_seed <= 2:
        assessment = "PAC allows stellar seeds (100 M☉) - TESTABLE"
        pac_works = True
    elif avg_pac_seed <= 5:
        assessment = "PAC requires direct collapse seeds - POSSIBLE"
        pac_works = True
    elif avg_pac_seed <= 6:
        assessment = "PAC requires heavy seeds - TENSION"
        pac_works = False
    else:
        assessment = "PAC still requires impossible seeds - FAILS"
        pac_works = False
    
    print(f"\n{assessment}")
    
    results["assessment"] = {
        "avg_enhancement": avg_enhancement,
        "avg_seed_reduction_dex": avg_reduction,
        "avg_pac_seed_log": avg_pac_seed,
        "assessment": assessment,
        "pac_helps": pac_works
    }
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 60)
    print(f"{'Eddington alone works?':<40} {'NO - seeds 10^3-10^5 needed':<20}")
    print(f"{'PAC enhancement factor':<40} {f'{avg_enhancement:.2f}×':<20}")
    print(f"{'PAC seed requirement':<40} {f'10^{avg_pac_seed:.1f} M☉':<20}")
    print(f"{'PAC helps?':<40} {'YES' if pac_works else 'PARTIALLY':<20}")
    
    results["success"] = True
    results["conclusion"] = (
        f"Standard Eddington requires 10^3-10^5 M☉ seeds. "
        f"PAC enhancement ({avg_enhancement:.2f}×) reduces requirement to 10^{avg_pac_seed:.1f} M☉. "
        f"{assessment}"
    )
    
    print(f"\n{results['conclusion']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_05_eddington_comparison_{timestamp}.json"
    
    with open(results_dir / filename, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: results/{filename}")
    
    return results


if __name__ == "__main__":
    run_experiment()
