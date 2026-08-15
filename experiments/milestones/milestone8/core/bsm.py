"""
bsm.py -- Shared infrastructure for Milestone 8: BSM Predictions & Observational Contact.

Provides:
- DFT constants: PHI, XI, LN_PHI, GAMMA_EM
- Physical constants: masses, couplings, cosmological parameters
- Experimental bounds: Bullet Cluster, Lyman-alpha, S8
- Fibonacci utilities: fib(), cyclotomic, depth-to-coupling, correction template
- Cosmological utilities: growth factor, Press-Schechter, cascade EOS
- Dark sector utilities: dark coupling, mass, cross-section, relic abundance
- Prediction registry and result saving
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from functools import lru_cache


# ============================================================
# DFT Constants
# ============================================================
PHI = (1 + np.sqrt(5)) / 2          # 1.6180339887...
INV_PHI = 1 / PHI                    # 0.6180339887...
LN_PHI = np.log(PHI)                 # 0.4812118250...
GAMMA_EM = 0.5772156649015329        # Euler-Mascheroni
XI_BALANCE = GAMMA_EM + LN_PHI       # 1.0584274899...
XI_PAC = 1.05711                     # Three-factor xi (tiling)
PI = np.pi
LN2 = np.log(2)

# PAC splitting
P_D = 1 / PHI       # dominant fraction
P_S = 1 / PHI**2    # subordinate fraction


# ============================================================
# Physical Constants (PDG 2024 / CODATA 2018)
# ============================================================
# Masses in GeV
M_PLANCK_GEV = 1.22089e19
M_Z_GEV = 91.1876
M_W_GEV = 80.3692
M_HIGGS_GEV = 125.25
M_PROTON_GEV = 0.93827
M_ELECTRON_GEV = 0.51100e-3
HIGGS_VEV = 246.22               # GeV (Higgs vacuum expectation value)

# Couplings
ALPHA_EM = 7.2973525693e-3       # fine structure constant
ALPHA_S = 0.1179                 # strong coupling at M_Z
G_FERMI = 1.1663788e-5           # GeV^{-2}
G_NEWTON = 6.67430e-11           # m^3 kg^{-1} s^{-2}
SIN2_THETA_W = 0.23122           # sin^2(theta_W)

# Conversion factors
HBAR_C = 0.197327                # GeV fm
C_LIGHT = 2.998e10               # cm/s
GEV_TO_KG = 1.783e-27            # kg per GeV
CM2_PER_GEV2 = 3.894e-28         # cm^2 per GeV^{-2}
GEV_TO_KEV = 1e6                 # keV per GeV
MPC_TO_CM = 3.086e24             # cm per Mpc

# Z boson
GAMMA_Z_GEV = 2.4952             # Z width in GeV


# ============================================================
# Cosmological Parameters (Planck 2018 + BAO)
# ============================================================
H0_PLANCK = 67.36                # km/s/Mpc (Planck CMB)
H0_SHOES = 73.04                 # km/s/Mpc (SH0ES Cepheid)
OMEGA_M = 0.3153                 # total matter density
OMEGA_B = 0.0493                 # baryon density
OMEGA_DM = OMEGA_M - OMEGA_B    # dark matter density (0.266)
OMEGA_LAMBDA = 0.6847            # dark energy density
OMEGA_DM_H2 = 0.1200            # Omega_DM h^2 (Planck)
OMEGA_B_H2 = 0.02237            # Omega_b h^2 (Planck)
RHO_CRIT_GEV_CM3 = 1.053e-5    # critical density in GeV/cm^3 (h=0.674)
T_CMB = 2.7255                   # CMB temperature in K
SIGMA8_PLANCK = 0.8111           # matter fluctuation amplitude

# DESI DR1 (2024)
W0_DESI = -0.827
W0_DESI_ERR = 0.063
WA_DESI = -0.75
WA_DESI_ERR = 0.29

# Weak lensing S8 measurements
S8_PLANCK = 0.832
S8_KIDS = 0.759
S8_DES = 0.776


# ============================================================
# Experimental Bounds
# ============================================================
SIGMA_OVER_M_BULLET = 1.0       # cm^2/g (Bullet Cluster upper bound)
LYMAN_ALPHA_MASS_BOUND = 3.3    # keV (WDM lower bound, Irsic+ 2017)

# Neutrino measured values (PDG 2024)
DM2_21 = 7.53e-5                # eV^2 (solar)
DM2_31 = 2.453e-3               # eV^2 (atmospheric, normal ordering)
DM2_RATIO = DM2_31 / DM2_21    # ~32.6
SUM_NU_BOUND = 0.12             # eV (Planck + BAO upper bound)
DELTA_CP_PDG = 195.0            # degrees (PDG central value, note: 195 or equivalently -165)
DELTA_CP_ERR = 50.0             # degrees (approximate 1-sigma)


# ============================================================
# Fibonacci Utilities
# ============================================================
@lru_cache(maxsize=512)
def fib(n):
    """Return the nth Fibonacci number. F(1)=1, F(2)=1, F(3)=2, ..."""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def cyclotomic_phi3(x):
    """Third cyclotomic polynomial: Φ₃(x) = x² + x + 1."""
    return x**2 + x + 1


def cyclotomic_phi5(x):
    """Fifth cyclotomic polynomial: Φ₅(x) = x⁴ + x³ + x² + x + 1."""
    return x**4 + x**3 + x**2 + x + 1


def cyclotomic_phi7(x):
    """Seventh cyclotomic polynomial: Φ₇(x) = x⁶ + x⁵ + x⁴ + x³ + x² + x + 1."""
    return x**6 + x**5 + x**4 + x**3 + x**2 + x + 1


def fibonacci_depth_coupling(depth):
    """
    Raw coupling at Fibonacci depth d: α_d = φ^{-d} / √5.
    This is the leading-order estimate before correction template.
    """
    return PHI**(-depth) / np.sqrt(5)


def correction_template(a, b, n=4, sign=-1):
    """
    Universal DFT correction template: 1 + sign * F_a / (n * π * F_b²).

    Used across forces:
    - G: (1 + F₁₃/(π·F₆²)) → 0.18%
    - α_EM: F₃/(F₄·φ·F₁₀)·(1 - F₁₀/(4π·F₇²)) → 5.7 ppm
    """
    return 1 + sign * fib(a) / (n * PI * fib(b)**2)


def depth_to_mass(depth, method='planck'):
    """
    Map Fibonacci depth to mass scale.

    Methods:
    - 'planck': M_Pl / F_d (direct Fibonacci mapping)
    - 'vev': v_H · φ^{-d/2} (Higgs VEV descent)
    - 'proton': m_p · φ^{-(d-13)/2} (proton-relative, EM as anchor)
    """
    if method == 'planck':
        f_d = fib(depth)
        if f_d == 0:
            return float('inf')
        return M_PLANCK_GEV / f_d
    elif method == 'vev':
        return HIGGS_VEV * PHI**(-depth / 2)
    elif method == 'proton':
        return M_PROTON_GEV * PHI**(-(depth - 13) / 2)
    else:
        raise ValueError(f"Unknown method: {method}")


# Key Fibonacci numbers used across M8
F3 = fib(3)    # 2
F4 = fib(4)    # 3
F5 = fib(5)    # 5
F6 = fib(6)    # 8
F7 = fib(7)    # 13
F8 = fib(8)    # 21
F9 = fib(9)    # 34
F10 = fib(10)  # 55

# Key depths
DEPTH_WEAK = 7
DEPTH_EM = 13
DEPTH_DARK = cyclotomic_phi3(F6)       # 73
DEPTH_GRAVITY = cyclotomic_phi3(F7)    # 183


# ============================================================
# Dark Sector Utilities
# ============================================================
def dark_coupling(depth=DEPTH_DARK, use_template=False):
    """Coupling constant at given depth. Optionally apply correction template."""
    alpha = fibonacci_depth_coupling(depth)
    if use_template:
        # Template with b=6 structure (same as EM/gravity)
        alpha *= correction_template(8, 6, n=4, sign=-1)
    return alpha


def dark_mass(depth=DEPTH_DARK, method='vev'):
    """Dark sector mediator mass at given depth."""
    return depth_to_mass(depth, method=method)


def bullet_cluster_sigma_over_m(alpha, m_gev):
    """
    Born approximation σ/m for dark self-interaction.
    σ ~ α² / m² (Born), convert to cm²/g.
    Returns σ/m in cm²/g.
    """
    if m_gev <= 0:
        return float('inf')
    sigma_gev2 = alpha**2 / m_gev**2
    sigma_cm2 = sigma_gev2 * CM2_PER_GEV2
    m_g = m_gev * GEV_TO_KG * 1e3  # GeV to grams
    return sigma_cm2 / m_g


def dodelson_widrow_abundance(m_kev, sin2_2theta):
    """
    Dodelson-Widrow sterile neutrino relic abundance (approximate).
    Ω_s h² ≈ 0.3 × (sin²2θ / 10⁻¹⁰) × (m_s / 1 keV)^1.8

    Reference: Dodelson & Widrow (1994), updated by Abazajian (2006).
    """
    return 0.3 * (sin2_2theta / 1e-10) * (m_kev / 1.0)**1.8


def free_streaming_length(m_kev, T_dec_mev=150.0):
    """
    Free-streaming length for a WDM particle.
    λ_fs ≈ 0.1 Mpc × (1 keV / m) × (T_dec / 150 MeV)
    """
    return 0.1 * (1.0 / m_kev) * (T_dec_mev / 150.0)


# ============================================================
# Cosmological Utilities
# ============================================================
def growth_factor(z, omega_m=OMEGA_M):
    """
    Linear growth factor D(z)/D(0), approximate (Carroll+ 1992).
    D(a) ∝ g(a) where g = (5/2)Ω_m / (Ω_m^{4/7} - Ω_Λ + (1+Ω_m/2)(1+Ω_Λ/70))
    """
    a = 1.0 / (1 + z)
    omega_m_z = omega_m / (omega_m + (1 - omega_m) * a**3)
    omega_l_z = 1 - omega_m_z
    g = 2.5 * omega_m_z / (
        omega_m_z**(4.0/7) - omega_l_z + (1 + omega_m_z / 2) * (1 + omega_l_z / 70)
    )
    g0 = 2.5 * omega_m / (
        omega_m**(4.0/7) - (1 - omega_m) + (1 + omega_m / 2) * (1 + (1 - omega_m) / 70)
    )
    return (g / g0) * a


def press_schechter_fraction(sigma, delta_c=1.686):
    """
    Press-Schechter collapsed fraction: f = erfc(δ_c / (√2 σ)).
    """
    from scipy.special import erfc
    return erfc(delta_c / (np.sqrt(2) * sigma))


def cascade_dark_energy_eos(phi_val=PHI):
    """
    Dark energy equation of state from cascade structure.
    From exp_32f: cascade energy decrease → w₀ > -1, w_a < 0.
    Zero free parameters.

    Returns (w0, wa).
    """
    # Cascade potential decrease per step: fraction = 1/phi
    # Effective w0 from cascade: w0 = -1 + 1/(3*phi^3)
    # wa from cascade acceleration: wa = -1/phi^3
    w0 = -1 + 1 / (3 * phi_val**3)
    wa = -1 / phi_val**3
    return w0, wa


# ============================================================
# Z' Utilities
# ============================================================
def zprime_mass():
    """Z' mass from Fibonacci ratio: M_Z × F₇/F₄."""
    return M_Z_GEV * F7 / F4


def zprime_coupling_ratio():
    """Z' coupling suppression: g'/g = 1/F₇ = 1/13."""
    return 1.0 / F7


def zprime_width():
    """Z' width: Γ_Z × (g'/g)² × (M'/M)."""
    g_ratio = zprime_coupling_ratio()
    m_ratio = zprime_mass() / M_Z_GEV
    return GAMMA_Z_GEV * g_ratio**2 * m_ratio


def zprime_cross_section_ratio():
    """σ(Z')/σ(Z) ~ (g'/g)⁴ for s-channel."""
    return zprime_coupling_ratio()**4


# ============================================================
# Neutrino Utilities
# ============================================================
def pmns_angles_dft():
    """
    PMNS mixing angles from Fibonacci ratios (M5 exp_08).
    Returns dict with theta_12, theta_13, theta_23, delta_CP in degrees.
    """
    theta_12 = np.degrees(np.arctan(F3 / F4))  # arctan(2/3)
    theta_13 = np.degrees(np.arctan(F3 / F7))  # arctan(2/13)
    # theta_23 with correction: pi/4 * (1 + F8/(3*pi*F5^2))
    theta_23 = 45.0 * (1 + F8 / (3 * PI * F5**2))
    delta_cp = XI_BALANCE * 60.0  # Xi * 60 degrees
    return {
        'theta_12': theta_12,
        'theta_13': theta_13,
        'theta_23': theta_23,
        'delta_CP': delta_cp,
    }


def dft_omega_c():
    """DFT formula for Ω_c: F₇ · Ξ² / F₁₀ (MAR exp_25, 0.079%)."""
    return F7 * XI_BALANCE**2 / F10


# ============================================================
# Prediction Registry
# ============================================================
class PredictionRegistry:
    """Tracks pre-registered predictions for falsification protocol."""

    def __init__(self):
        self.predictions = []

    def register(self, name, value, uncertainty, basis, falsifiable_by, experiment):
        self.predictions.append({
            'name': name,
            'value': value,
            'uncertainty': uncertainty,
            'basis': basis,
            'falsifiable_by': falsifiable_by,
            'experiment': experiment,
            'registered_at': datetime.now().isoformat(),
        })

    def check_against_bounds(self):
        """Check no predictions are already excluded. Returns list of issues."""
        issues = []
        for p in self.predictions:
            # Add specific checks as experiments are completed
            pass
        return issues

    def to_dict(self):
        return {'predictions': self.predictions, 'count': len(self.predictions)}


# ============================================================
# Result Saving
# ============================================================
def save_results(results_dict, experiment_name, results_dir=None):
    """Save experiment results as timestamped JSON."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = results_dir / f"{experiment_name}_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    return outpath


def setup_experiment(script_file):
    """Common setup: fix encoding, return paths."""
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    script_dir = Path(script_file).resolve().parent
    m8_root = script_dir.parent
    results_dir = m8_root / "results"
    results_dir.mkdir(exist_ok=True)
    return m8_root, results_dir
