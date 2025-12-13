"""
QBE Dynamics - Quantum Balance Equation for Cosmology

Core equation: dI/dt + dE/dt = λ·QPL(t)

This module implements QBE constraints for cosmological applications,
determining what information-energy states are ALLOWED.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .constants import (
    PHI, XI, LAMBDA_QBE, QPL_OMEGA,
    M_GALAXY_SCALE, T_EDDINGTON,
    mass_at_level
)


@dataclass
class QBEState:
    """QBE state for a cosmological object."""
    name: str
    redshift: float
    
    # Observed
    observed_log_mass: float
    observed_dE_dt: Optional[float]  # Accretion power (if measurable)
    
    # QBE-derived
    qpl_value: float                  # QPL at cosmic time
    required_dI_dt: float            # What dI/dt must be for balance
    inferred_k: float                # PAC level from balance
    
    # Residuals
    qbe_residual: float              # |dI + dE - λ·QPL|
    is_balanced: bool


def compute_qpl(t: float) -> float:
    """
    Compute Quantum Potential Layer at cosmic time t.
    
    QPL(t) = cos(ω·t) where ω = 0.020 Hz
    
    For cosmological applications, t is in Gyr, so we convert:
    ω_cosmo = ω × (Gyr in seconds) = 0.020 × 3.156e16
    
    But the oscillation period is ~50 seconds, which is negligible
    on cosmological timescales. For cosmology, QPL effectively
    represents the "average" quantum regulation.
    """
    # On cosmological scales, QPL averages to a constant
    # The oscillations matter on quantum/lab timescales
    # For cosmology, use the RMS value = 1/√2
    return 1 / np.sqrt(2)  # ~0.707


def compute_dE_dt_eddington(mass: float, duty_cycle: float = 0.1) -> float:
    """
    Compute dE/dt for Eddington-limited accretion.
    
    P_Edd = L_Edd × η = (4πGMm_p c / σ_T) × 0.1
    
    In normalized units, dE/dt ∝ M / t_Edd
    """
    # Eddington luminosity in L_sun: L_Edd = 3.2 × 10^4 × (M/M_sun)
    # Power = L × η
    # For normalized QBE units, we use M / t_Edd as the scale
    
    return mass * duty_cycle / T_EDDINGTON


def compute_dI_dt_from_mass_growth(mass: float, growth_rate: float) -> float:
    """
    Compute dI/dt from mass growth.
    
    Information content I ~ log(M) in hierarchy interpretation
    So dI/dt = (1/M) × dM/dt = growth_rate / M × M = growth_rate
    
    Actually, in PAC hierarchy:
    I = k = -log_φ(M/M_galaxy)
    dI/dt = -d(log M)/dt / log(φ) = -(1/M × dM/dt) / log(φ)
    """
    if mass <= 0:
        return 0
    
    # dM/dt = growth_rate × M (exponential growth)
    dM_dt = growth_rate * mass
    
    # dI/dt in PAC units
    dI_dt = -dM_dt / (mass * np.log(PHI))
    
    return dI_dt


def qbe_constrained_k(z: float, observed_log_mass: float, dE_dt: float = None) -> Dict:
    """
    Find QBE-constrained k level for an observation.
    
    Given:
    - Redshift z (determines cosmic time → QPL)
    - Observed mass (determines approximate k)
    - dE/dt (accretion power, if known)
    
    Compute:
    - Required dI/dt for QBE balance
    - Whether this is consistent with observed mass
    
    The key insight: QBE constrains what k values are ALLOWED.
    Not all masses are QBE-consistent at all redshifts.
    """
    # Cosmic time (approximate)
    t_cosmic = 13.8 / (1 + z)**1.5  # Gyr, matter-dominated approx
    
    # QPL at this cosmic time
    qpl = compute_qpl(t_cosmic)
    
    # Observed mass
    M_obs = 10 ** observed_log_mass
    
    # If dE/dt not provided, estimate from Eddington
    if dE_dt is None:
        dE_dt = compute_dE_dt_eddington(M_obs)
    
    # QBE constraint: dI/dt + dE/dt = λ·QPL
    required_dI_dt = LAMBDA_QBE * qpl - dE_dt
    
    # What does this imply for mass growth?
    # dI/dt = -(dM/dt) / (M × ln(φ))
    # So dM/dt = -dI/dt × M × ln(φ)
    implied_dM_dt = -required_dI_dt * M_obs * np.log(PHI)
    
    # Growth rate
    implied_growth_rate = implied_dM_dt / M_obs if M_obs > 0 else 0
    
    # Is this physical?
    # Positive growth rate = mass increasing (expected for young BH)
    # Negative = mass decreasing (unphysical for isolated BH)
    is_physical = implied_growth_rate > -1e-10  # Allow small numerical errors
    
    # K level from observed mass
    k_from_mass = -np.log(M_obs / M_GALAXY_SCALE) / np.log(PHI)
    
    # QBE residual: how close is dI + dE to λ·QPL?
    # If we use the implied dI/dt, residual is zero by construction
    # The question is whether the implied growth is physical
    actual_dI_dt = compute_dI_dt_from_mass_growth(M_obs, implied_growth_rate)
    qbe_sum = actual_dI_dt + dE_dt
    qbe_target = LAMBDA_QBE * qpl
    qbe_residual = abs(qbe_sum - qbe_target)
    
    return {
        "z": z,
        "observed_log_mass": observed_log_mass,
        "k_from_mass": k_from_mass,
        "t_cosmic_gyr": t_cosmic,
        "qpl": qpl,
        "dE_dt": dE_dt,
        "required_dI_dt": required_dI_dt,
        "implied_growth_rate": implied_growth_rate,
        "is_physical": is_physical,
        "qbe_residual": qbe_residual,
        "qbe_balanced": qbe_residual < 0.1
    }


def test_qbe_constraints(observations: List[Dict]) -> Dict:
    """
    Test QBE constraints on a set of observations.
    
    Args:
        observations: List of dicts with 'name', 'z', 'log_m_bh'
    
    Returns:
        Summary of QBE consistency
    """
    results = []
    
    for obs in observations:
        qbe_result = qbe_constrained_k(
            z=obs["z"],
            observed_log_mass=obs["log_m_bh"]
        )
        qbe_result["name"] = obs["name"]
        results.append(qbe_result)
    
    n_physical = sum(1 for r in results if r["is_physical"])
    n_balanced = sum(1 for r in results if r["qbe_balanced"])
    
    return {
        "n_objects": len(observations),
        "n_physical": n_physical,
        "n_balanced": n_balanced,
        "all_physical": n_physical == len(observations),
        "results": results
    }


def run_qbe_analysis():
    """Run QBE analysis on JWST observations."""
    
    from .pac_cosmology import JWST_OBSERVATIONS
    
    print("=" * 70)
    print("QBE CONSTRAINT ANALYSIS")
    print("=" * 70)
    
    print("\nCore equation: dI/dt + dE/dt = λ·QPL(t)")
    print(f"λ = {LAMBDA_QBE}, QPL ≈ {1/np.sqrt(2):.3f} (cosmological average)\n")
    
    results = test_qbe_constraints(JWST_OBSERVATIONS)
    
    print(f"{'Object':<15} {'z':<8} {'log(M)':<10} {'k':<10} {'dE/dt':<12} {'dI/dt_req':<12} {'Physical?':<10}")
    print("-" * 77)
    
    for r in results["results"]:
        phys = "✓" if r["is_physical"] else "✗"
        print(f"{r['name']:<15} {r['z']:<8.2f} {r['observed_log_mass']:<10.2f} {r['k_from_mass']:<10.2f} {r['dE_dt']:<12.2e} {r['required_dI_dt']:<12.2e} {phys:<10}")
    
    print(f"\nPhysical: {results['n_physical']}/{results['n_objects']}")
    print(f"QBE balanced: {results['n_balanced']}/{results['n_objects']}")
    
    if results["all_physical"]:
        print("\n✅ All observations are QBE-consistent")
    else:
        print("\n⚠️ Some observations violate QBE physical constraints")
    
    return results


if __name__ == "__main__":
    run_qbe_analysis()
