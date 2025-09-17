"""
Scale-Dependent Infodynamic Gravity Arithmetic

Implements smooth transition from galaxy-scale gravity dominance 
to cosmic web-scale dark matter emergence using characteristic length scaling.

Key insight: Same physics, different regimes based on system scale.
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

# Physical constants
KPC_TO_METERS = 3.086e19
MPC_TO_METERS = 3.086e22

@dataclass
class ScaleRegimes:
    """Physical scale regime definitions"""
    # Galaxy regime: κ=1e4, λ_c=30 kpc, β_floor=0.1 (10% dark matter)
    κ_galaxy: float = 1e4
    λ_galaxy: float = 30.0  # kpc
    β_galaxy: float = 0.1
    
    # Cosmic web regime: κ=1e2, λ_c=2 Mpc, β_floor=0.6 (60% dark matter)  
    κ_cosmic: float = 1e2
    λ_cosmic: float = 2000.0  # kpc (2 Mpc)
    β_cosmic: float = 0.6
    
    # Transition parameters
    L_transition: float = 1000.0  # kpc (1 Mpc transition point)
    L_smooth: float = 200.0       # kpc (0.2 Mpc smoothness)

def calculate_characteristic_length(positions: np.ndarray, masses: np.ndarray) -> float:
    """
    Calculate characteristic length scale of the system.
    
    L_characteristic = sqrt(Σ m_i × r_i²) / M_total
    
    This measures the "spread" of the system - small for compact galaxies,
    large for extended cosmic web structures.
    
    Args:
        positions: Array of particle positions (N, 3) in kpc
        masses: Array of particle masses (N,) in solar masses
        
    Returns:
        Characteristic length in kpc
    """
    # Calculate center of mass using proper weighted average
    M_total = np.sum(masses)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / M_total
    
    # Calculate distance squared from center of mass
    r_vec = positions - com
    r_squared = np.sum(r_vec**2, axis=1)
    
    # Weighted RMS radius
    L_char = np.sqrt(np.sum(masses * r_squared) / M_total)
    
    return L_char

def scale_transition_function(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> float:
    """
    Smooth transition function between galaxy and cosmic regimes.
    
    σ(L) = 1 / (1 + exp((L - L_transition)/L_smooth))
    
    Returns:
        σ = 1.0 for galaxy scale (L << L_transition)
        σ = 0.5 at transition (L = L_transition) 
        σ = 0.0 for cosmic scale (L >> L_transition)
    """
    exponent = (L - regimes.L_transition) / regimes.L_smooth
    # Prevent overflow
    exponent = np.clip(exponent, -50, 50)
    return 1.0 / (1.0 + np.exp(exponent))

def get_scale_dependent_parameters(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> Dict[str, float]:
    """
    Calculate scale-dependent parameters based on system characteristic length.
    
    Args:
        L: Characteristic length scale in kpc
        regimes: Scale regime definitions
        
    Returns:
        Dictionary with scale-adapted parameters:
        - κ: Force coupling strength
        - λ_c: Coherence length (kpc)
        - β_floor: Quantum floor fraction
        - scale_regime: String describing dominant regime
    """
    σ = scale_transition_function(L, regimes)
    
    # Smooth interpolation between regimes
    κ = regimes.κ_galaxy * σ + regimes.κ_cosmic * (1 - σ)
    λ_c = regimes.λ_galaxy * σ + regimes.λ_cosmic * (1 - σ)
    β_floor = regimes.β_galaxy * σ + regimes.β_cosmic * (1 - σ)
    
    # Determine dominant regime
    if σ > 0.8:
        scale_regime = "galaxy"
    elif σ > 0.2:
        scale_regime = "transition"
    else:
        scale_regime = "cosmic_web"
    
    return {
        "κ": κ,
        "λ_c": λ_c,
        "β_floor": β_floor,
        "sigma": σ,
        "scale_regime": scale_regime,
        "L_characteristic": L
    }

def calculate_expected_dark_matter_fraction(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> float:
    """
    Calculate expected dark matter fraction based on system scale.
    
    At galaxy scale (L < 100 kpc): ~10% dark matter (gravity dominates)
    At cosmic scale (L > 10 Mpc): ~60% dark matter (quantum floor dominates)
    
    Args:
        L: Characteristic length scale in kpc
        
    Returns:
        Expected dark matter fraction (0.0 to 1.0)
    """
    σ = scale_transition_function(L, regimes)
    expected_dm = regimes.β_galaxy * σ + regimes.β_cosmic * (1 - σ)
    return expected_dm

def analyze_system_scale(positions: np.ndarray, masses: np.ndarray) -> Dict[str, Any]:
    """
    Complete scale analysis of a particle system.
    
    Args:
        positions: Particle positions (N, 3) in kpc
        masses: Particle masses (N,) in solar masses
        
    Returns:
        Complete scale analysis including parameters and predictions
    """
    L_char = calculate_characteristic_length(positions, masses)
    params = get_scale_dependent_parameters(L_char)
    expected_dm = calculate_expected_dark_matter_fraction(L_char)
    
    # Additional metrics
    box_size = np.max(positions) - np.min(positions)
    particle_separation = np.median(np.diff(np.sort(np.linalg.norm(positions, axis=1))))
    
    analysis = {
        **params,
        "expected_dark_matter_fraction": expected_dm,
        "box_size_kpc": box_size,
        "median_particle_separation_kpc": particle_separation,
        "n_particles": len(positions),
        "total_mass_solar": np.sum(masses),
        "scale_ratios": {
            "L_char_to_transition": L_char / 1000.0,  # Relative to 1 Mpc
            "coherence_to_separation": params["λ_c"] / particle_separation,
            "quantum_enhancement": params["β_floor"] / 0.1  # Relative to galaxy floor
        }
    }
    
    return analysis

def validate_scale_regime(analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate that system parameters are appropriate for the detected scale regime.
    
    Returns:
        Dictionary of validation results and recommendations
    """
    L = analysis["L_characteristic"]
    regime = analysis["scale_regime"]
    ratios = analysis["scale_ratios"]
    
    recommendations = {}
    
    if regime == "galaxy":
        if L > 500:  # kpc
            recommendations["warning"] = f"Large galaxy system (L={L:.0f} kpc) may show transitional effects"
        if ratios["coherence_to_separation"] < 2:
            recommendations["resolution"] = "Consider smaller particle separation for galaxy dynamics"
            
    elif regime == "cosmic_web":
        if L < 5000:  # kpc (5 Mpc)
            recommendations["warning"] = f"Small cosmic system (L={L:.0f} kpc) may not show full filamentary structure"
        if ratios["coherence_to_separation"] > 10:
            recommendations["resolution"] = "Consider larger particle separation for cosmic web dynamics"
            
    elif regime == "transition":
        recommendations["note"] = "Transition regime - expect mixed galaxy cluster and filamentary dynamics"
        
    # General recommendations
    if analysis["quantum_enhancement"] < 2:
        recommendations["dark_matter"] = "Low quantum floor may not produce visible dark matter effects"
    elif analysis["quantum_enhancement"] > 10:
        recommendations["dark_matter"] = "Very high quantum floor - dark matter may dominate completely"
        
    return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Test scale transition
    print("=== Scale-Dependent Parameter Testing ===")
    
    test_scales = [10, 50, 200, 1000, 5000, 20000]  # kpc
    
    for L in test_scales:
        params = get_scale_dependent_parameters(L)
        expected_dm = calculate_expected_dark_matter_fraction(L)
        
        print(f"\nL = {L:5.0f} kpc ({L/1000:.1f} Mpc)")
        print(f"  Regime: {params['scale_regime']}")
        print(f"  κ = {params['κ']:.1e}")
        print(f"  λ_c = {params['λ_c']:.0f} kpc")
        print(f"  β_floor = {params['β_floor']:.2f}")
        print(f"  Expected DM: {expected_dm:.1%}")
        print(f"  σ(L) = {params['sigma']:.3f}")
    
    print("\n=== Critical Scale Ratios ===")
    regimes = ScaleRegimes()
    print(f"κ_galaxy/κ_cosmic = {regimes.κ_galaxy/regimes.κ_cosmic:.0f}")
    print(f"λ_cosmic/λ_galaxy = {regimes.λ_cosmic/regimes.λ_galaxy:.0f}")
    print(f"β_cosmic/β_galaxy = {regimes.β_cosmic/regimes.β_galaxy:.0f}")
