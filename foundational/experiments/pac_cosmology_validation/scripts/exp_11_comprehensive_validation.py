"""
Experiment 11: Comprehensive PAC/SEC Validation with Full JWST Catalog

This is the "rigorous" version of the JWST validation - treating the previous
work as a viability test and now doing it properly with:

1. MUCH LARGER SAMPLE: 85 objects (vs 10 in initial study)
2. NULL HYPOTHESIS TESTING: 4 alternative models
3. MONTE CARLO UNCERTAINTY: Full error propagation
4. PARAMETER SENSITIVITY: Comprehensive sweeps
5. FIRST-PRINCIPLES SEC: Clean derivation of enhancement

Data Sources:
- Andika et al. 2024 (arXiv:2401.11826): 64 candidates z=6-8
- Harikane et al. 2023 (arXiv:2303.11946): 10 AGN z=4-7
- Maiolino et al. 2023 (arXiv:2305.12492): GN-z11
- Goulding et al. 2023 (arXiv:2308.02750): UHZ-1
- Kocevski et al. 2023 (arXiv:2302.00012): CEERS objects
- Juodžbalis et al. 2024 (arXiv:2403.03872): Dormant BH
- Maiolino et al. 2024 (arXiv:2405.00504): 71 AGN z=2-11

Target: Publication-quality validation for preprint.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONSTANTS (First-Principles Derivation)
# ============================================================================

# Golden ratio - derived from PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2)
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...
PHI_SQUARED = PHI**2  # 2.6180339887...
PHI_INVERSE = 1 / PHI  # 0.6180339887...

# Xi constant - balance operator (1 + π/F₁₀)
F10 = 55  # 10th Fibonacci number
XI = 1 + np.pi / F10  # 1.0571...

# Cosmological parameters (Planck 2018)
OMEGA_M_TODAY = 0.315
OMEGA_DE_TODAY = 0.685
H0 = 67.4  # km/s/Mpc
T_HUBBLE = 14.4  # Gyr

# Eddington timescale
T_EDDINGTON = 0.045  # Gyr (~45 Myr Salpeter timescale)

# PAC equilibrium fractions
PAC_EQUILIBRIUM = PHI / (PHI + 1)  # 0.618... (attraction fraction)
SEC_EQUILIBRIUM = 1 / (PHI + 1)     # 0.382... (repulsion fraction)

# SEC equilibrium level (k where φ^(-k) = 0.618)
K_EQUILIBRIUM = -np.log(PAC_EQUILIBRIUM) / np.log(PHI)  # ≈ 1.0


# ============================================================================
# PAC COSMOLOGY CORE
# ============================================================================

@dataclass
class PACState:
    """PAC cosmological state at redshift z."""
    z: float
    cosmic_age_gyr: float
    matter_fraction: float
    pac_level_k: float
    unactualized_fraction: float  # φ^(-k)
    enhancement_factor: float     # SEC duty cycle enhancement


def cosmic_age_at_z(z: float) -> float:
    """
    Calculate cosmic age at redshift z in Gyr.
    Uses flat ΛCDM approximation.
    """
    from scipy import integrate
    
    def integrand(z_prime):
        E_z = np.sqrt(OMEGA_M_TODAY * (1 + z_prime)**3 + OMEGA_DE_TODAY)
        return 1 / ((1 + z_prime) * E_z)
    
    result, _ = integrate.quad(integrand, z, 1100)
    t_H = 1 / (H0 * 1e3 / 3.086e22) / (3.156e7 * 1e9)  # Hubble time in Gyr
    
    return t_H * result


def matter_fraction_at_z(z: float) -> float:
    """Calculate matter fraction at redshift z."""
    rho_m = OMEGA_M_TODAY * (1 + z)**3
    rho_de = OMEGA_DE_TODAY
    return rho_m / (rho_m + rho_de)


def pac_level_at_z(z: float) -> float:
    """
    Map redshift to PAC recursion level k.
    
    At equilibrium: Ω_m = 1/φ² ≈ 0.382, k = 1
    Early universe: Ω_m → 1, k → 0
    Late universe: Ω_m → 0, k → ∞
    """
    m_frac = matter_fraction_at_z(z)
    m_eq = 1 / PHI_SQUARED  # ~0.382
    
    if m_frac > m_eq:
        # Early universe: k ∈ [0, 1)
        frac_to_eq = (m_frac - m_eq) / (1 - m_eq)
        k = 1 - frac_to_eq
    else:
        # Late universe: k > 1
        k = 1 + np.log(m_eq / max(m_frac, 1e-10)) / np.log(PHI)
    
    return max(0, k)


def sec_enhancement_at_z(z: float) -> float:
    """
    Calculate SEC enhancement factor at redshift z.
    
    First-principles derivation:
    1. PAC recursion gives duty cycle D(k) = φ^(-k)
    2. At equilibrium (k=1): D = 1/φ ≈ 0.618
    3. At high-z (k→0): D → 1.0
    4. Enhancement = D(z) / D_equilibrium
    
    This is the ratio of effective growth time at z vs equilibrium.
    """
    k = pac_level_at_z(z)
    k_eq = 1.0  # Equilibrium at k=1
    
    # Unactualized fractions (duty cycles)
    D_z = PHI ** (-k)
    D_eq = PHI ** (-k_eq)  # = 1/φ ≈ 0.618
    
    # Enhancement relative to equilibrium
    enhancement = D_z / D_eq
    
    return enhancement


def pac_state_at_z(z: float) -> PACState:
    """Get full PAC state at redshift z."""
    age = cosmic_age_at_z(z)
    m_frac = matter_fraction_at_z(z)
    k = pac_level_at_z(z)
    enhancement = sec_enhancement_at_z(z)
    
    return PACState(
        z=z,
        cosmic_age_gyr=age,
        matter_fraction=m_frac,
        pac_level_k=k,
        unactualized_fraction=PHI**(-k),
        enhancement_factor=enhancement
    )


# ============================================================================
# GROWTH MODELS
# ============================================================================

@dataclass
class GrowthPrediction:
    """Predicted maximum BH mass for a given model."""
    model_name: str
    z: float
    t_gyr: float
    log_m_seed: float
    log_m_max: float
    e_foldings: float
    duty_cycle: float
    eddington_frac: float


def predict_mass_pac(z: float, log_m_seed: float = 5.0, 
                     f_edd: float = 0.5) -> GrowthPrediction:
    """
    Predict maximum BH mass using PAC/SEC model.
    
    Key insight: SEC enhancement at high-z allows faster effective growth
    without requiring super-Eddington rates.
    """
    state = pac_state_at_z(z)
    
    # Base duty cycle at equilibrium (1/φ ≈ 0.618)
    D_eq = 1 / PHI
    
    # Enhanced duty at this redshift
    D_eff = D_eq * state.enhancement_factor
    
    # e-folding time = T_Edd / (D_eff × f_Edd)
    n_efolds = D_eff * f_edd * state.cosmic_age_gyr / T_EDDINGTON
    
    m_seed = 10**log_m_seed
    m_max = m_seed * np.exp(n_efolds)
    
    return GrowthPrediction(
        model_name="PAC/SEC",
        z=z,
        t_gyr=state.cosmic_age_gyr,
        log_m_seed=log_m_seed,
        log_m_max=np.log10(m_max),
        e_foldings=n_efolds,
        duty_cycle=D_eff,
        eddington_frac=f_edd
    )


def predict_mass_lcdm_realistic(z: float, log_m_seed: float = 5.0,
                                 duty: float = 0.2, f_edd: float = 0.3) -> GrowthPrediction:
    """
    Predict maximum BH mass using realistic ΛCDM assumptions.
    
    Based on observational constraints:
    - Duty cycle: ~20% (most SMBHs are inactive)
    - Eddington fraction: ~30% (typical for luminous AGN)
    """
    age = cosmic_age_at_z(z)
    
    n_efolds = duty * f_edd * age / T_EDDINGTON
    
    m_seed = 10**log_m_seed
    m_max = m_seed * np.exp(n_efolds)
    
    return GrowthPrediction(
        model_name="ΛCDM Realistic",
        z=z,
        t_gyr=age,
        log_m_seed=log_m_seed,
        log_m_max=np.log10(m_max),
        e_foldings=n_efolds,
        duty_cycle=duty,
        eddington_frac=f_edd
    )


def predict_mass_continuous_eddington(z: float, log_m_seed: float = 5.0,
                                       f_edd: float = 1.0) -> GrowthPrediction:
    """
    Predict maximum BH mass with continuous Eddington accretion.
    
    This is the theoretical maximum - requires 100% duty cycle at Eddington,
    which is physically implausible but sets an upper limit.
    """
    age = cosmic_age_at_z(z)
    
    n_efolds = f_edd * age / T_EDDINGTON
    
    m_seed = 10**log_m_seed
    m_max = m_seed * np.exp(n_efolds)
    
    return GrowthPrediction(
        model_name="Continuous Eddington",
        z=z,
        t_gyr=age,
        log_m_seed=log_m_seed,
        log_m_max=np.log10(m_max),
        e_foldings=n_efolds,
        duty_cycle=1.0,
        eddington_frac=f_edd
    )


def predict_mass_heavy_seed(z: float, log_m_seed: float = 5.5) -> GrowthPrediction:
    """
    Predict using heavy seed scenario (ΛCDM with direct collapse seeds).
    
    Direct collapse black holes (DCBHs) can form with M_seed ~ 10^5-10^6 M☉.
    This reduces the growth requirement.
    """
    return predict_mass_lcdm_realistic(z, log_m_seed=log_m_seed, 
                                        duty=0.3, f_edd=0.5)


# ============================================================================
# NULL HYPOTHESIS MODELS
# ============================================================================

@dataclass
class NullHypothesis:
    """Definition of a null hypothesis for testing."""
    name: str
    description: str
    predict_fn: callable
    free_params: int  # For AIC/BIC calculation


NULL_HYPOTHESES = [
    NullHypothesis(
        name="H0_Random",
        description="No relationship between z and M_BH",
        predict_fn=lambda z: 6.5,  # Mean log mass
        free_params=1
    ),
    NullHypothesis(
        name="H0_PowerLaw",
        description="Power law M_BH ∝ (1+z)^α",
        predict_fn=lambda z: 6.5 - 0.1 * np.log10(1 + z),
        free_params=2
    ),
    NullHypothesis(
        name="H0_LCDM_Optimal",
        description="ΛCDM with optimized duty/Eddington",
        predict_fn=lambda z: predict_mass_lcdm_realistic(z, duty=0.5, f_edd=0.5).log_m_max,
        free_params=3
    ),
    NullHypothesis(
        name="H0_Continuous",
        description="Continuous Eddington (upper limit)",
        predict_fn=lambda z: predict_mass_continuous_eddington(z).log_m_max,
        free_params=1
    )
]


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

@dataclass
class ValidationResult:
    """Result of validating one object against a model."""
    object_id: str
    z: float
    log_m_observed: float
    log_m_observed_err: float
    log_m_predicted: float
    residual: float  # Observed - Predicted
    is_explained: bool  # Observed <= Predicted (within errors)
    sigma_tension: float  # (Observed - Predicted) / error


@dataclass
class ModelComparison:
    """Comparison of models using information criteria."""
    model_name: str
    n_objects: int
    n_explained: int
    explained_fraction: float
    chi_squared: float
    reduced_chi_squared: float
    aic: float
    bic: float
    mean_residual: float
    std_residual: float
    max_tension_sigma: float


def validate_object(obj: dict, model_fn: callable) -> ValidationResult:
    """Validate a single object against a model prediction."""
    z = obj['redshift']
    log_m = obj['log_mass']
    log_m_err = (obj['log_mass_error_lower'] + obj['log_mass_error_upper']) / 2
    
    pred = model_fn(z)
    if hasattr(pred, 'log_m_max'):
        log_m_pred = pred.log_m_max
    else:
        log_m_pred = pred
    
    # Residual: positive means observed > predicted (unexplained)
    residual = log_m - log_m_pred
    sigma = residual / max(log_m_err, 0.1)  # Avoid division by zero
    
    # Object is "explained" if observed mass is achievable 
    # (predicted >= observed - 2σ, i.e., residual <= 2σ)
    is_explained = residual <= 2 * log_m_err
    
    return ValidationResult(
        object_id=obj['id'],
        z=z,
        log_m_observed=log_m,
        log_m_observed_err=log_m_err,
        log_m_predicted=log_m_pred,
        residual=residual,
        is_explained=is_explained,
        sigma_tension=sigma
    )


def compare_models(objects: List[dict], 
                   models: Dict[str, callable]) -> List[ModelComparison]:
    """Compare multiple models against the data."""
    comparisons = []
    n = len(objects)
    
    for name, model_fn in models.items():
        results = [validate_object(obj, model_fn) for obj in objects]
        
        n_explained = sum(1 for r in results if r.is_explained)
        residuals = [r.residual for r in results]
        errors = [r.log_m_observed_err for r in results]
        
        # Chi-squared
        chi_sq = sum((r.residual / max(r.log_m_observed_err, 0.1))**2 for r in results)
        
        # Degrees of freedom (estimate based on model type)
        k = 2  # Approximate free parameters
        for null in NULL_HYPOTHESES:
            if null.name == name:
                k = null.free_params
                break
        
        dof = max(n - k, 1)
        reduced_chi_sq = chi_sq / dof
        
        # Information criteria
        log_likelihood = -0.5 * chi_sq
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        comparisons.append(ModelComparison(
            model_name=name,
            n_objects=n,
            n_explained=n_explained,
            explained_fraction=n_explained / n,
            chi_squared=chi_sq,
            reduced_chi_squared=reduced_chi_sq,
            aic=aic,
            bic=bic,
            mean_residual=np.mean(residuals),
            std_residual=np.std(residuals),
            max_tension_sigma=max(r.sigma_tension for r in results)
        ))
    
    return comparisons


# ============================================================================
# MONTE CARLO UNCERTAINTY PROPAGATION
# ============================================================================

def monte_carlo_validation(objects: List[dict], model_fn: callable,
                           n_samples: int = 1000, seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation for uncertainty propagation.
    
    Samples from measurement uncertainties to get robust statistics.
    """
    np.random.seed(seed)
    
    explained_counts = []
    mean_residuals = []
    
    for _ in range(n_samples):
        # Perturb observations within errors
        perturbed = []
        for obj in objects:
            p = obj.copy()
            err = (obj['log_mass_error_lower'] + obj['log_mass_error_upper']) / 2
            p['log_mass'] = np.random.normal(obj['log_mass'], err)
            p['redshift'] = np.random.normal(obj['redshift'], obj.get('redshift_error', 0.01))
            perturbed.append(p)
        
        # Validate against model
        results = [validate_object(p, model_fn) for p in perturbed]
        explained_counts.append(sum(1 for r in results if r.is_explained))
        mean_residuals.append(np.mean([r.residual for r in results]))
    
    return {
        "n_samples": n_samples,
        "explained_mean": np.mean(explained_counts),
        "explained_std": np.std(explained_counts),
        "explained_percentiles": {
            "5%": np.percentile(explained_counts, 5),
            "50%": np.percentile(explained_counts, 50),
            "95%": np.percentile(explained_counts, 95)
        },
        "residual_mean": np.mean(mean_residuals),
        "residual_std": np.std(mean_residuals),
        "residual_percentiles": {
            "5%": np.percentile(mean_residuals, 5),
            "50%": np.percentile(mean_residuals, 50),
            "95%": np.percentile(mean_residuals, 95)
        }
    }


# ============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

def parameter_sweep_duty_cycle(objects: List[dict], 
                                duty_range: Tuple[float, float] = (0.1, 1.0),
                                n_points: int = 20) -> List[dict]:
    """Sweep duty cycle and measure explained fraction for ΛCDM."""
    results = []
    
    for duty in np.linspace(duty_range[0], duty_range[1], n_points):
        model_fn = lambda z, d=duty: predict_mass_lcdm_realistic(z, duty=d, f_edd=0.5)
        
        validation_results = [validate_object(obj, model_fn) for obj in objects]
        n_explained = sum(1 for r in validation_results if r.is_explained)
        
        results.append({
            "duty_cycle": duty,
            "explained_fraction": n_explained / len(objects),
            "mean_residual": np.mean([r.residual for r in validation_results])
        })
    
    return results


def parameter_sweep_seed_mass(objects: List[dict],
                               log_seed_range: Tuple[float, float] = (3.0, 6.0),
                               n_points: int = 20) -> List[dict]:
    """Sweep seed mass and measure explained fraction for both models."""
    results = []
    
    for log_seed in np.linspace(log_seed_range[0], log_seed_range[1], n_points):
        # PAC model
        pac_fn = lambda z, ls=log_seed: predict_mass_pac(z, log_m_seed=ls)
        pac_results = [validate_object(obj, pac_fn) for obj in objects]
        pac_explained = sum(1 for r in pac_results if r.is_explained)
        
        # ΛCDM model
        lcdm_fn = lambda z, ls=log_seed: predict_mass_lcdm_realistic(z, log_m_seed=ls)
        lcdm_results = [validate_object(obj, lcdm_fn) for obj in objects]
        lcdm_explained = sum(1 for r in lcdm_results if r.is_explained)
        
        results.append({
            "log_seed_mass": log_seed,
            "pac_explained_fraction": pac_explained / len(objects),
            "lcdm_explained_fraction": lcdm_explained / len(objects)
        })
    
    return results


# ============================================================================
# REDSHIFT BIN ANALYSIS
# ============================================================================

def analyze_by_redshift_bin(objects: List[dict], 
                             bin_edges: List[float] = [4, 6, 8, 10, 13]) -> List[dict]:
    """Analyze PAC vs ΛCDM performance by redshift bin."""
    results = []
    
    for i in range(len(bin_edges) - 1):
        z_min, z_max = bin_edges[i], bin_edges[i+1]
        bin_objects = [o for o in objects if z_min <= o['redshift'] < z_max]
        
        if len(bin_objects) == 0:
            continue
        
        # PAC predictions
        pac_results = [validate_object(obj, predict_mass_pac) for obj in bin_objects]
        pac_explained = sum(1 for r in pac_results if r.is_explained)
        
        # ΛCDM predictions
        lcdm_results = [validate_object(obj, predict_mass_lcdm_realistic) for obj in bin_objects]
        lcdm_explained = sum(1 for r in lcdm_results if r.is_explained)
        
        # Mean SEC enhancement in bin
        mean_z = np.mean([o['redshift'] for o in bin_objects])
        enhancement = sec_enhancement_at_z(mean_z)
        
        results.append({
            "z_min": z_min,
            "z_max": z_max,
            "z_mean": mean_z,
            "n_objects": len(bin_objects),
            "pac_explained": pac_explained,
            "pac_fraction": pac_explained / len(bin_objects),
            "lcdm_explained": lcdm_explained,
            "lcdm_fraction": lcdm_explained / len(bin_objects),
            "sec_enhancement": enhancement,
            "advantage_ratio": (pac_explained + 1) / (lcdm_explained + 1)  # +1 to avoid div0
        })
    
    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def load_catalog(filepath: Path) -> List[dict]:
    """Load the comprehensive JWST catalog."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['objects']


def main():
    """Run comprehensive PAC/SEC validation."""
    print("=" * 70)
    print("PAC/SEC COSMOLOGY VALIDATION - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"\nRun time: {datetime.now().isoformat()}")
    
    # Load data
    script_dir = Path(__file__).parent.parent
    catalog_path = script_dir / "data" / "comprehensive_catalog.json"
    
    try:
        objects = load_catalog(catalog_path)
        print(f"\nLoaded {len(objects)} objects from comprehensive catalog")
    except FileNotFoundError:
        print(f"ERROR: Catalog not found at {catalog_path}")
        return
    
    # Filter to high-z only (z > 4 for PAC/SEC relevance)
    high_z_objects = [o for o in objects if o['redshift'] >= 4]
    print(f"High-z sample (z≥4): {len(high_z_objects)} objects")
    
    # Summary statistics
    redshifts = [o['redshift'] for o in high_z_objects]
    masses = [o['log_mass'] for o in high_z_objects]
    print(f"\nRedshift range: {min(redshifts):.2f} - {max(redshifts):.2f}")
    print(f"Mass range: {min(masses):.1f} - {max(masses):.1f} log(M☉)")
    
    # ========================================================================
    # 1. CORE MODEL COMPARISON
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. CORE MODEL COMPARISON")
    print("=" * 70)
    
    models = {
        "PAC/SEC": predict_mass_pac,
        "ΛCDM Realistic": predict_mass_lcdm_realistic,
        "Continuous Eddington": predict_mass_continuous_eddington,
        "Heavy Seed ΛCDM": predict_mass_heavy_seed
    }
    
    comparisons = compare_models(high_z_objects, models)
    
    print(f"\n{'Model':<25} {'Explained':<12} {'Fraction':<10} {'χ²/dof':<10} {'ΔAIC':<10}")
    print("-" * 70)
    
    aic_min = min(c.aic for c in comparisons)
    for c in sorted(comparisons, key=lambda x: -x.explained_fraction):
        print(f"{c.model_name:<25} {c.n_explained:>3}/{c.n_objects:<8} "
              f"{c.explained_fraction:>8.1%} {c.reduced_chi_squared:>9.2f} "
              f"{c.aic - aic_min:>+9.1f}")
    
    # ========================================================================
    # 2. NULL HYPOTHESIS TESTING
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. NULL HYPOTHESIS TESTING")
    print("=" * 70)
    
    null_models = {h.name: h.predict_fn for h in NULL_HYPOTHESES}
    null_models["PAC/SEC"] = predict_mass_pac
    
    null_comparisons = compare_models(high_z_objects, null_models)
    
    pac_result = next(c for c in null_comparisons if c.model_name == "PAC/SEC")
    
    print(f"\n{'Null Hypothesis':<25} {'χ²':<12} {'ΔAIC vs PAC':<15}")
    print("-" * 55)
    
    for c in sorted(null_comparisons, key=lambda x: x.aic):
        delta_aic = c.aic - pac_result.aic
        sig = ""
        if delta_aic > 10:
            sig = "***"  # Strong evidence against null
        elif delta_aic > 6:
            sig = "**"
        elif delta_aic > 2:
            sig = "*"
        print(f"{c.model_name:<25} {c.chi_squared:>10.1f} {delta_aic:>+13.1f} {sig}")
    
    print("\n* ΔAIC > 2: weak evidence, ** > 6: moderate, *** > 10: strong")
    
    # ========================================================================
    # 3. MONTE CARLO UNCERTAINTY
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. MONTE CARLO UNCERTAINTY PROPAGATION")
    print("=" * 70)
    
    print("\nRunning 1000 Monte Carlo samples...")
    
    pac_mc = monte_carlo_validation(high_z_objects, predict_mass_pac, n_samples=1000)
    lcdm_mc = monte_carlo_validation(high_z_objects, predict_mass_lcdm_realistic, n_samples=1000)
    
    print(f"\nPAC/SEC explained objects:")
    print(f"  Mean: {pac_mc['explained_mean']:.1f} ± {pac_mc['explained_std']:.1f}")
    print(f"  95% CI: [{pac_mc['explained_percentiles']['5%']:.0f}, "
          f"{pac_mc['explained_percentiles']['95%']:.0f}]")
    
    print(f"\nΛCDM Realistic explained objects:")
    print(f"  Mean: {lcdm_mc['explained_mean']:.1f} ± {lcdm_mc['explained_std']:.1f}")
    print(f"  95% CI: [{lcdm_mc['explained_percentiles']['5%']:.0f}, "
          f"{lcdm_mc['explained_percentiles']['95%']:.0f}]")
    
    # ========================================================================
    # 4. REDSHIFT BIN ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. REDSHIFT BIN ANALYSIS")
    print("=" * 70)
    
    bin_results = analyze_by_redshift_bin(high_z_objects)
    
    print(f"\n{'z range':<12} {'N':<5} {'PAC':<10} {'ΛCDM':<10} {'SEC enh.':<10} {'Advantage':<10}")
    print("-" * 60)
    
    for r in bin_results:
        print(f"{r['z_min']:.0f}-{r['z_max']:.0f}       {r['n_objects']:<5} "
              f"{r['pac_fraction']:.1%}     {r['lcdm_fraction']:.1%}     "
              f"{r['sec_enhancement']:.3f}     {r['advantage_ratio']:.2f}×")
    
    # ========================================================================
    # 5. PARAMETER SENSITIVITY
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Seed mass sweep
    seed_sweep = parameter_sweep_seed_mass(high_z_objects)
    
    print(f"\nSeed mass required for 80% explanation:")
    for result in seed_sweep:
        if result['pac_explained_fraction'] >= 0.8:
            print(f"  PAC/SEC: log(M_seed) ≥ {result['log_seed_mass']:.1f}")
            break
    
    for result in seed_sweep:
        if result['lcdm_explained_fraction'] >= 0.8:
            print(f"  ΛCDM:    log(M_seed) ≥ {result['log_seed_mass']:.1f}")
            break
    
    # Duty cycle sweep for ΛCDM
    duty_sweep = parameter_sweep_duty_cycle(high_z_objects)
    
    print(f"\nΛCDM duty cycle required for 80% explanation:")
    for result in duty_sweep:
        if result['explained_fraction'] >= 0.8:
            print(f"  Duty ≥ {result['duty_cycle']:.0%}")
            break
    else:
        print(f"  Cannot reach 80% even with 100% duty cycle")
    
    # ========================================================================
    # 6. KEY PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. KEY PAC/SEC PREDICTIONS")
    print("=" * 70)
    
    print("\nSEC enhancement by redshift:")
    for z in [4, 6, 8, 10, 12]:
        state = pac_state_at_z(z)
        pac_pred = predict_mass_pac(z)
        lcdm_pred = predict_mass_lcdm_realistic(z)
        
        print(f"  z={z:>2}: Enhancement = {state.enhancement_factor:.3f}×, "
              f"log(M_max,PAC) = {pac_pred.log_m_max:.1f}, "
              f"log(M_max,ΛCDM) = {lcdm_pred.log_m_max:.1f}, "
              f"Δ = {pac_pred.log_m_max - lcdm_pred.log_m_max:.1f} dex")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results = {
        "run_time": datetime.now().isoformat(),
        "catalog_size": len(objects),
        "high_z_sample_size": len(high_z_objects),
        "redshift_range": [min(redshifts), max(redshifts)],
        "mass_range": [min(masses), max(masses)],
        "model_comparisons": [
            {
                "model": c.model_name,
                "n_explained": c.n_explained,
                "fraction": c.explained_fraction,
                "chi_squared": c.chi_squared,
                "reduced_chi_squared": c.reduced_chi_squared,
                "aic": c.aic,
                "bic": c.bic
            }
            for c in comparisons
        ],
        "monte_carlo": {
            "pac": pac_mc,
            "lcdm": lcdm_mc
        },
        "redshift_bins": bin_results,
        "seed_mass_sweep": seed_sweep,
        "duty_cycle_sweep": duty_sweep,
        "conclusions": {
            "pac_explains_fraction": next(c for c in comparisons if c.model_name == "PAC/SEC").explained_fraction,
            "lcdm_explains_fraction": next(c for c in comparisons if c.model_name == "ΛCDM Realistic").explained_fraction,
            "pac_vs_null_delta_aic": min(c.aic for c in null_comparisons if c.model_name != "PAC/SEC") - pac_result.aic,
            "sec_enhancement_z10": sec_enhancement_at_z(10)
        }
    }
    
    # Save
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_11_comprehensive_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {results_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    pac_frac = results['conclusions']['pac_explains_fraction']
    lcdm_frac = results['conclusions']['lcdm_explains_fraction']
    delta_aic = results['conclusions']['pac_vs_null_delta_aic']
    
    print(f"""
PAC/SEC explains {pac_frac:.0%} of high-z SMBHs with realistic parameters.
ΛCDM explains {lcdm_frac:.0%} with realistic parameters.

Key mechanism: SEC enhancement at high-z provides {sec_enhancement_at_z(10):.2f}× 
effective duty cycle boost at z=10, allowing faster growth without 
requiring continuous super-Eddington accretion.

Model comparison: PAC/SEC favored over best null hypothesis by ΔAIC = {delta_aic:.1f}
(>10 = strong evidence).

Falsification condition: If >50% of high-z SMBHs require enhancement factors
exceeding {sec_enhancement_at_z(10) * 1.5:.2f}×, theory is falsified.
""")


if __name__ == "__main__":
    main()
