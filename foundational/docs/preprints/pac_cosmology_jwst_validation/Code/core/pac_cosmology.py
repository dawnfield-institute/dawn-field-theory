"""
PAC Cosmology Core Module

Apply PAC/SEC framework to cosmological predictions.

Key principles:
1. PAC (attraction) = 4/5, SEC (repulsion) = 1/5 at equilibrium
2. Early universe was ATTRACTION-DOMINATED (PAC → 1)
3. Ξ = 1 + π/F₁₀ = 1.0571 is the balance operator
4. Cosmological equilibrium: DE = 1/φ ≈ 61.8%, Matter = 1/φ² ≈ 38.2%
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy import integrate
from scipy.optimize import minimize_scalar

from .constants import (
    PHI, PHI_SQUARED, XI, 
    PAC_FRACTION, SEC_FRACTION,
    MATTER_EQUILIBRIUM, DE_EQUILIBRIUM,
    OMEGA_M_TODAY, OMEGA_DE_TODAY, H0, T_HUBBLE, T_EDDINGTON,
    M_GALAXY_SCALE, CONTEXT_VARIANCE,
    mass_at_level, level_for_mass
)


@dataclass
class PACCosmologyState:
    """State of PAC cosmology at a given redshift."""
    redshift: float
    cosmic_age_gyr: float
    matter_fraction: float
    de_fraction: float
    pac_fraction: float          # Effective attraction fraction
    sec_fraction: float          # Effective repulsion fraction
    xi_effective: float          # Effective balance operator
    phase: str                   # "attraction_dominated", "near_equilibrium", or "repulsion_dominated"
    k_level: float               # Approximate PAC hierarchy level


def cosmic_age_at_z(z: float) -> float:
    """
    Calculate cosmic age at redshift z (Gyr).
    Flat ΛCDM approximation.
    """
    def integrand(z_prime):
        E_z = np.sqrt(OMEGA_M_TODAY * (1 + z_prime)**3 + OMEGA_DE_TODAY)
        return 1 / ((1 + z_prime) * E_z)
    
    result, _ = integrate.quad(integrand, z, 1100)
    t_H = 1 / (H0 * 1e3 / 3.086e22) / (3.156e7 * 1e9)
    
    return t_H * result


def matter_fraction_at_z(z: float) -> Tuple[float, float]:
    """
    Calculate matter and dark energy fractions at redshift z.
    
    Returns:
        (matter_fraction, de_fraction)
    """
    rho_m = OMEGA_M_TODAY * (1 + z)**3
    rho_de = OMEGA_DE_TODAY
    
    total = rho_m + rho_de
    
    return rho_m / total, rho_de / total


def pac_state_at_z(z: float) -> PACCosmologyState:
    """
    Calculate the PAC cosmological state at redshift z.
    
    The PAC tree actualization is NON-LINEAR:
    - Early universe (z→∞): U → 1 (all potential, no actualization)
    - φ equilibrium: U = 1/φ ≈ 0.618 (golden balance)
    - Today (z=0): U = 1/φ² ≈ 0.382 (mostly actualized)
    - Heat death: U → 0 (fully actualized)
    
    The PAC fraction (attraction) tracks the UNACTUALIZED potential.
    """
    age = cosmic_age_at_z(z)
    m_frac, de_frac = matter_fraction_at_z(z)
    
    # Map matter fraction to PAC level
    if m_frac > MATTER_EQUILIBRIUM:
        # Early universe: between k=0 and k=1
        frac_to_equil = (m_frac - MATTER_EQUILIBRIUM) / (1 - MATTER_EQUILIBRIUM)
        k_level = 1 - frac_to_equil
    else:
        # Late universe: k > 1
        k_level = 1 + np.log(MATTER_EQUILIBRIUM / max(m_frac, 0.001)) / np.log(PHI)
    
    # Unactualized fraction = φ^(-k)
    unactualized = PHI ** (-k_level)
    
    pac_eff = unactualized
    sec_eff = 1 - pac_eff
    
    # Phase determination
    if pac_eff > 1/PHI:
        phase = "attraction_dominated"
    elif pac_eff > 1/PHI_SQUARED:
        phase = "near_equilibrium"
    else:
        phase = "repulsion_dominated"
    
    # Effective Ξ: enhanced when more potential remains
    xi_eff = XI * (1 + (pac_eff - 1/PHI) * (XI - 1) / (1 - 1/PHI))
    
    return PACCosmologyState(
        redshift=z,
        cosmic_age_gyr=age,
        matter_fraction=m_frac,
        de_fraction=de_frac,
        pac_fraction=pac_eff,
        sec_fraction=sec_eff,
        xi_effective=xi_eff,
        phase=phase,
        k_level=k_level
    )


def pac_hierarchy_mass(k: float) -> float:
    """
    Get mass at PAC hierarchy level k.
    
    M(k) = M_galaxy × φ^(-k)
    """
    return mass_at_level(k)


def pac_hierarchy_level(mass: float) -> float:
    """
    Get PAC level for a given mass.
    
    k = -log_φ(M / M_galaxy)
    """
    return level_for_mass(mass)


def pac_rate_enhancement(z: float) -> float:
    """
    Calculate PAC enhancement to accretion rate at redshift z.
    
    In attraction-dominated phase, hierarchical enhancement applies:
    - At equilibrium (k=2): no enhancement
    - At PAC→1 (k=0): enhancement = φ²
    """
    state = pac_state_at_z(z)
    
    equilibrium_pac = 1 / PHI_SQUARED
    
    if state.pac_fraction > equilibrium_pac:
        k_current = -np.log(state.pac_fraction) / np.log(PHI)
        k_equilibrium = 2
        delta_k = k_equilibrium - k_current
        return PHI ** delta_k
    else:
        return 1.0


def pac_mbh_mstar_ratio(z: float) -> float:
    """
    Predict M_BH/M* ratio from PAC hierarchy.
    
    The level difference between BH and galaxy formation determines the ratio.
    """
    state = pac_state_at_z(z)
    
    local_ratio = 1e-3  # Local M_BH/M*
    local_pac = 1 / PHI_SQUARED
    
    if state.pac_fraction > local_pac:
        frac = (state.pac_fraction - local_pac) / (1 - local_pac)
        primordial_ratio = 0.1
        log_ratio = np.log10(local_ratio) + frac * (np.log10(primordial_ratio) - np.log10(local_ratio))
        return 10 ** log_ratio
    else:
        return local_ratio


def relativistic_time_dilation(z: float) -> float:
    """
    Compute relativistic time dilation factor from EDV framework.
    
    At high z (attraction-dominated): γ up to √7.42 ≈ 2.72
    At equilibrium: γ = 1
    """
    state = pac_state_at_z(z)
    
    pac_excess = max(0, state.pac_fraction - 1/PHI_SQUARED)
    normalized_excess = pac_excess / (1 - 1/PHI_SQUARED)
    
    gamma = 1 + normalized_excess * (np.sqrt(CONTEXT_VARIANCE) - 1)
    
    return gamma


# =============================================================================
# JWST OBSERVATIONS - EXPANDED CATALOG (December 2025)
# =============================================================================

# Sources:
# - Goulding et al. 2023 (arXiv:2308.02750) - UHZ-1 spectroscopic confirmation
# - Maiolino et al. 2023 (arXiv:2305.12492) - GN-z11 BH detection
# - Harikane et al. 2023 (arXiv:2303.11946) - z=4-7 AGN census
# - Larson et al. 2023 - CEERS AGN sample
# - Various GLASS/UNCOVER papers

JWST_OBSERVATIONS = [
    # Highest redshift objects (z > 10) - most constraining
    {"name": "UHZ-1", "z": 10.073, "log_m_bh": 7.5, "log_m_star": 8.15, 
     "notes": "X-ray detected, Compton-thick, MBH/M* extremely high"},
    {"name": "GN-z11", "z": 10.603, "log_m_bh": 6.2, "log_m_star": 9.0,
     "notes": "Super-Eddington (5x), outflow detected"},
    {"name": "GLASS-z12", "z": 12.5, "log_m_bh": 6.0, "log_m_star": 8.0,
     "notes": "Photometric z, highest redshift candidate"},
    
    # z ~ 8-10 regime
    {"name": "CEERS-1019", "z": 8.68, "log_m_bh": 6.95, "log_m_star": 9.5,
     "notes": "Spectroscopically confirmed AGN"},
    {"name": "CEERS-746", "z": 8.0, "log_m_bh": 6.8, "log_m_star": 9.3,
     "notes": "Broad H-alpha detection"},
    
    # z ~ 5-7 regime (Harikane census)
    {"name": "CEERS-2782", "z": 5.242, "log_m_bh": 7.2, "log_m_star": 9.8,
     "notes": "Harikane et al. census"},
    {"name": "CEERS-1670", "z": 4.483, "log_m_bh": 7.5, "log_m_star": 10.1,
     "notes": "Harikane et al. census"},
    {"name": "GLASS-38108", "z": 6.936, "log_m_bh": 6.5, "log_m_star": 8.5,
     "notes": "Harikane et al. census"},
    {"name": "GLASS-160133", "z": 6.232, "log_m_bh": 7.8, "log_m_star": 9.2,
     "notes": "Harikane et al. census"},
    {"name": "GLASS-150029", "z": 4.015, "log_m_bh": 6.3, "log_m_star": 9.5,
     "notes": "Harikane et al. census"},
]

# Subsets for different analyses
HIGH_Z_OBJECTS = [obs for obs in JWST_OBSERVATIONS if obs["z"] > 8]
MID_Z_OBJECTS = [obs for obs in JWST_OBSERVATIONS if 5 <= obs["z"] <= 8]
LOW_Z_OBJECTS = [obs for obs in JWST_OBSERVATIONS if obs["z"] < 5]


def get_observations():
    """Return JWST observations as list of dicts."""
    return JWST_OBSERVATIONS.copy()


if __name__ == "__main__":
    print("PAC Cosmology States")
    print("=" * 70)
    
    for z in [0, 2, 5, 8, 10, 12, 15, 20]:
        state = pac_state_at_z(z)
        gamma = relativistic_time_dilation(z)
        ratio = pac_mbh_mstar_ratio(z)
        
        print(f"\nz = {z}")
        print(f"  Age: {state.cosmic_age_gyr:.3f} Gyr")
        print(f"  Matter: {state.matter_fraction:.3f}")
        print(f"  PAC: {state.pac_fraction:.4f}")
        print(f"  Phase: {state.phase}")
        print(f"  k-level: {state.k_level:.2f}")
        print(f"  Time dilation γ: {gamma:.2f}")
        print(f"  M_BH/M*: {ratio:.4f}")
