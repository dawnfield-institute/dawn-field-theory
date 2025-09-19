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
    if len(positions) == 0 or len(masses) == 0:
        return 0.0
        
    # Calculate center of mass
    total_mass = np.sum(masses)
    if total_mass == 0:
        return 0.0
    
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    
    # Calculate weighted RMS distance from center of mass
    distances_squared = np.sum((positions - com) ** 2, axis=1)
    weighted_rms = np.sqrt(np.sum(masses * distances_squared) / total_mass)
    
    return weighted_rms

def scale_transition_function(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> float:
    """
    Smooth transition function between galaxy and cosmic web regimes.
    
    σ(L) = 0.5 * (1 + tanh((L - L_transition) / L_smooth))
    
    Returns:
        0.0 for galaxy regime (L << L_transition)
        1.0 for cosmic web regime (L >> L_transition)
        Smooth transition between
    """
    return 0.5 * (1 + np.tanh((L - regimes.L_transition) / regimes.L_smooth))

def get_scale_dependent_parameters(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> Dict[str, float]:
    """
    Calculate scale-dependent parameters for infodynamic gravity.
    
    Uses smooth interpolation between galaxy and cosmic web regimes
    based on characteristic length L.
    
    Args:
        L: Characteristic length in kpc
        regimes: Scale regime configuration
        
    Returns:
        Dictionary with parameters: κ, λ_c, β_floor, sigma, scale_regime
    """
    σ = scale_transition_function(L, regimes)
    
    # Interpolate parameters
    κ = regimes.κ_galaxy * (1 - σ) + regimes.κ_cosmic * σ
    λ_c = regimes.λ_galaxy * (1 - σ) + regimes.λ_cosmic * σ
    β_floor = regimes.β_galaxy * (1 - σ) + regimes.β_cosmic * σ
    
    # Determine regime classification
    if σ < 0.2:
        scale_regime = "galaxy"
    elif σ > 0.8:
        scale_regime = "cosmic_web"
    else:
        scale_regime = "transition"
    
    return {
        'κ': κ,
        'λ_c': λ_c,
        'β_floor': β_floor,
        'sigma': σ,
        'scale_regime': scale_regime
    }

def calculate_expected_dark_matter_fraction(L: float, regimes: ScaleRegimes = ScaleRegimes()) -> float:
    """
    Calculate expected dark matter fraction based on scale.
    
    This is the theoretical prediction for β (dark matter fraction)
    based purely on the characteristic length scale.
    
    Args:
        L: Characteristic length in kpc
        
    Returns:
        Expected dark matter fraction (0.0 to 1.0)
    """
    params = get_scale_dependent_parameters(L, regimes)
    return params['β_floor']

def analyze_system_scale(positions: np.ndarray, masses: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive analysis of system scale properties.
    
    Args:
        positions: Array of particle positions (N, 3) in kpc
        masses: Array of particle masses (N,) in solar masses
        
    Returns:
        Analysis dictionary with scale properties
    """
    # Calculate characteristic length
    L = calculate_characteristic_length(positions, masses)
    
    # Get scale-dependent parameters
    params = get_scale_dependent_parameters(L)
    
    # Calculate additional metrics
    total_mass = np.sum(masses)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass if total_mass > 0 else np.zeros(3)
    
    # Maximum extent
    if len(positions) > 0:
        max_distance = np.max(np.linalg.norm(positions - com, axis=1))
    else:
        max_distance = 0.0
    
    # Mass distribution metrics
    mass_variance = np.var(masses) if len(masses) > 0 else 0.0
    
    # Quantum enhancement factor
    quantum_enhancement = params['κ'] / 1e3  # Normalized to typical scale
    
    return {
        'characteristic_length': L,
        'total_mass': total_mass,
        'max_extent': max_distance,
        'mass_variance': mass_variance,
        'quantum_enhancement': quantum_enhancement,
        'scale_regime': params['scale_regime'],
        'transition_sigma': params['sigma'],
        'expected_dark_matter': params['β_floor'],
        'kappa': params['κ'],
        'lambda_c': params['λ_c']
    }

def validate_scale_regime(analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate if the system is in the expected scale regime.
    
    Provides recommendations and warnings about scale-dependent behavior.
    
    Args:
        analysis: Output from analyze_system_scale()
        
    Returns:
        Dictionary with validation results and recommendations
    """
    L = analysis["characteristic_length"]
    regime = analysis["scale_regime"]
    
    recommendations = {}
    
    # Scale regime validation
    if regime == "galaxy" and L > 500:
        recommendations["scale"] = "Large galaxy system - consider cosmic web effects"
    elif regime == "cosmic_web" and L < 500:
        recommendations["scale"] = "Small cosmic system - galaxy dynamics may dominate"
    else:
        recommendations["scale"] = f"System correctly classified as {regime} regime"
    
    # Mass distribution validation
    if analysis["mass_variance"] > analysis["total_mass"] ** 2 / 100:
        recommendations["mass"] = "High mass variance - check for outliers or substructure"
    
    # Quantum enhancement validation
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
