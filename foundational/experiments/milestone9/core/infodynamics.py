"""
infodynamics.py -- Shared infrastructure for Milestone 9: The Infodynamic Mechanism.

Extends M8's bsm.py with cascade clock, information-time nexus, and
scale-dependent prediction utilities.

Provides:
- Cascade clock: N(t), level timing, clock fitting
- Information cost: Xi decomposition, survival fraction
- Cosmological predictions: S8(z), H0(z), w(z) from cascade clock
- SEC dynamics: field equation, entropy rate
- PAC cascade: energy ratios, temporal self-similarity
- Friedmann integration: z <-> lookback time conversion
"""

import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.integrate import quad

# Import M8 infrastructure
M9_ROOT = Path(__file__).resolve().parent.parent
M8_ROOT = M9_ROOT.parent / "milestone8"
sys.path.insert(0, str(M8_ROOT / "core"))

from bsm import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI,
    P_D, P_S, LN2,
    # Physical constants
    M_PLANCK_GEV, M_Z_GEV, HIGGS_VEV, ALPHA_EM, G_NEWTON,
    # Cosmological parameters
    H0_PLANCK, H0_SHOES, OMEGA_M, OMEGA_B, OMEGA_DM, OMEGA_LAMBDA,
    OMEGA_DM_H2, SIGMA8_PLANCK, T_CMB,
    S8_PLANCK, S8_KIDS, S8_DES,
    W0_DESI, W0_DESI_ERR, WA_DESI, WA_DESI_ERR,
    # Fibonacci utilities
    fib, fibonacci_depth_coupling, correction_template,
    F3, F4, F5, F6, F7, F8, F9, F10,
    DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    # Cosmological utilities
    growth_factor, press_schechter_fraction, cascade_dark_energy_eos,
    # Dark sector
    dark_coupling, dark_mass,
    # Result infrastructure
    save_results, setup_experiment, PredictionRegistry,
)


# ============================================================
# Cascade Clock Constants
# ============================================================
B_DFT = 1.0 / LN_PHI             # 2.0781 -- DFT-constrained slope
B_FREE = 2.264                     # free-fit slope from 3 M8 data points

# Observational data from M8 exp_11 cross-consistency
N_DATA = {
    's8':     {'N': 4.16, 't_lookback_gyr': 4.0,  'z_eff': 0.4},
    'hubble': {'N': 5.94, 't_lookback_gyr': 9.5,  'z_eff': 1.5},
    'jwst':   {'N': 6.90, 't_lookback_gyr': 13.2, 'z_eff': 10.0},
}

# Cosmological time scales
T_UNIVERSE = 13.8                  # Gyr (age of universe)
T_RECOMBINATION = 0.000380         # Gyr (380 kyr)

# Derived: S8 lensing mean
S8_LENSING = (S8_KIDS + S8_DES) / 2  # 0.7675

# H0 measurement data (probe, H0, error, effective lookback Gyr)
H0_PROBES = {
    'shoes':    {'H0': 73.04, 'err': 1.04, 't_look': 0.14, 'z_eff': 0.01},
    'trgb':     {'H0': 69.8,  'err': 1.7,  't_look': 0.07, 'z_eff': 0.005},
    'tdsl':     {'H0': 73.3,  'err': 1.7,  't_look': 5.1,  'z_eff': 0.5},
    'planck':   {'H0': 67.36, 'err': 0.54, 't_look': 13.8, 'z_eff': 1100.0},
}

# DESI BAO effective redshifts
DESI_Z_EFF = [0.15, 0.38, 0.51, 0.70, 1.48]


# ============================================================
# Cascade Clock Functions
# ============================================================
def cascade_clock(t_lookback_gyr, a, slope=B_DFT):
    """
    Cascade clock: N(t) = a + slope * ln(t_lookback).

    The cascade clock uses LOOKBACK TIME as the independent variable.
    Deeper lookback = higher N = more cascade levels seen through.

    Parameters:
        t_lookback_gyr: lookback time in Gyr
        a: intercept (fit parameter)
        slope: default 1/ln(phi) = 2.0781

    Returns: cascade level N at lookback time t
    """
    return a + slope * np.log(np.asarray(t_lookback_gyr, dtype=float))


def cascade_clock_fit(constrained=True):
    """
    Fit cascade clock to the 3 M8 data points.

    If constrained: fix slope to 1/ln(phi), fit only intercept a.
    If unconstrained: fit both a and slope.

    Returns: (a, slope, rms_residual)
    """
    t_look = np.array([d['t_lookback_gyr'] for d in N_DATA.values()])
    n_obs = np.array([d['N'] for d in N_DATA.values()])

    if constrained:
        # Fixed slope, fit intercept only
        ln_t = np.log(t_look)
        a_fit = np.mean(n_obs - B_DFT * ln_t)
        residuals = n_obs - cascade_clock(t_look, a_fit, B_DFT)
        rms = np.sqrt(np.mean(residuals**2))
        return a_fit, B_DFT, rms
    else:
        def model(t, a, b):
            return a + b * np.log(t)
        popt, _ = curve_fit(model, t_look, n_obs)
        residuals = n_obs - model(t_look, *popt)
        rms = np.sqrt(np.mean(residuals**2))
        return popt[0], popt[1], rms


def cascade_level_time(level, t1_gyr):
    """
    Time at which cascade level n completes: t_n = t1 * phi^n.

    Parameters:
        level: cascade level number
        t1_gyr: first level completion time in Gyr

    Returns: completion time in Gyr
    """
    return t1_gyr * PHI**level


def cascade_level_duration(level, t1_gyr):
    """Duration of level n: t1 * phi^n * (phi - 1) / phi = t1 * phi^{n-1} * (phi-1)."""
    if level == 0:
        return t1_gyr
    return t1_gyr * PHI**(level - 1) * (PHI - 1)


def current_cascade_level(a_clock):
    """What cascade level at the full age of the universe (max lookback)?"""
    return cascade_clock(T_UNIVERSE, a_clock, B_DFT)


def fraction_through_current_level(a_clock):
    """How far through the current level are we (0 to 1)?"""
    n_now = current_cascade_level(a_clock)
    n_floor = int(np.floor(n_now))
    return n_now - n_floor


def n_at_lookback(t_lookback_gyr, a_clock):
    """Cascade level at a given lookback time."""
    return cascade_clock(t_lookback_gyr, a_clock, B_DFT)


# ============================================================
# Friedmann Integration (z <-> lookback time)
# ============================================================
def _friedmann_integrand(z, Om=OMEGA_M, Ol=OMEGA_LAMBDA):
    """1/((1+z)*E(z)) where E(z) = sqrt(Om*(1+z)^3 + Ol)."""
    Ez = np.sqrt(Om * (1 + z)**3 + Ol)
    return 1.0 / ((1 + z) * Ez)


def z_to_lookback(z, H0=H0_PLANCK, Om=OMEGA_M, Ol=OMEGA_LAMBDA):
    """
    Convert redshift to lookback time in Gyr.

    Uses Friedmann integration: t_look = (1/H0) * integral_0^z dz'/((1+z')*E(z'))
    """
    H0_per_gyr = H0 * 1.022e-3  # km/s/Mpc -> 1/Gyr (approx)
    result, _ = quad(_friedmann_integrand, 0, z, args=(Om, Ol))
    return result / H0_per_gyr


def lookback_to_cosmic(t_lookback_gyr):
    """Convert lookback time to cosmic time: t_cosmic = T_UNIVERSE - t_lookback."""
    return T_UNIVERSE - t_lookback_gyr


def z_to_cosmic(z, H0=H0_PLANCK):
    """Convert redshift to cosmic time in Gyr."""
    return lookback_to_cosmic(z_to_lookback(z, H0))


# ============================================================
# PAC Cascade Dynamics
# ============================================================
def pac_cascade_ratios(n_levels, g_in=INV_PHI):
    """
    Build a PAC cascade with retention fraction g_in per level.

    Returns dict with:
    - energies: array of energy at each level (starting from 1.0)
    - ratios: consecutive energy ratios E_{n+1}/E_n
    - cumulative_time: cumulative time (energy-proportional)
    - time_ratios: consecutive time ratios t_{n+1}/t_n
    """
    energies = np.array([g_in**n for n in range(n_levels)])
    ratios = energies[1:] / energies[:-1]

    # Energy-proportional timing: each level takes time proportional to its energy
    cumulative_time = np.cumsum(energies)
    time_ratios = cumulative_time[1:] / cumulative_time[:-1]

    return {
        'energies': energies,
        'ratios': ratios,
        'cumulative_time': cumulative_time,
        'time_ratios': time_ratios,
        'g_in': g_in,
    }


def pac_cascade_conservation_error(alpha):
    """
    Conservation error for a cascade with timing ratio alpha.
    Conservation: g_in + g_out = 1, where g_in = 1/alpha.
    Returns absolute error from exact conservation.
    """
    g_in = 1.0 / alpha
    g_out = 1.0 - g_in
    # Scale invariance: at each level, D_{n+1}/S_n should equal the global ratio
    # For PAC: g_out should equal g_in^2 (from exp_32e duality)
    scale_inv_error = abs(g_out - g_in**2)
    conservation_error = abs(g_in + g_out - 1.0)  # identically 0 by construction
    return conservation_error, scale_inv_error


def pac_cascade_convergence(alpha, n_levels=100):
    """Check if cascade with ratio alpha converges (finite total time)."""
    g_in = 1.0 / alpha
    if g_in >= 1.0:
        return False, float('inf')
    total = sum(g_in**n for n in range(n_levels))
    theoretical = 1.0 / (1.0 - g_in)
    return True, abs(total / theoretical - 1.0)


# ============================================================
# Information Cost Functions
# ============================================================
def xi_info_cost():
    """Xi = gamma + ln(phi). The information cost per cascade boundary crossing."""
    return XI_BALANCE


def xi_survival_fraction():
    """Survival fraction per boundary crossing: e^{-Xi}."""
    return np.exp(-XI_BALANCE)


def xi_decomposition():
    """
    Decompose Xi into its two components.

    Returns:
        gamma: Euler-Mascheroni (counting/discreteness cost)
        ln_phi: ln(phi) (branching/splitting cost)
        total: gamma + ln(phi) = Xi
    """
    return {
        'gamma': GAMMA_EM,
        'ln_phi': LN_PHI,
        'total': GAMMA_EM + LN_PHI,
        'survival': np.exp(-(GAMMA_EM + LN_PHI)),
    }


def cascade_info_loss(n_levels, include_stochastic=True, seed=42):
    """
    Compute information loss through a PAC cascade.

    At each level, the phi-split creates dominant (P/phi) and subordinate (P/phi^2).
    Information loss = entropy of the split distribution.

    If include_stochastic: add small perturbations around phi-ratio to make
    the cascade non-trivially irreversible.
    """
    rng = np.random.default_rng(seed)
    levels = []
    P = 1.0

    for n in range(n_levels):
        if include_stochastic:
            noise = 1.0 + rng.normal(0, 0.01)
            ratio = INV_PHI * noise
        else:
            ratio = INV_PHI

        D = P * ratio
        S = P - D

        # Shannon entropy of the split
        p_d = D / P
        p_s = S / P
        H_split = -(p_d * np.log(p_d) + p_s * np.log(p_s)) if p_d > 0 and p_s > 0 else 0

        # Mutual information: I(parent; dominant) / H(parent)
        # For a deterministic split, I = H(parent). For stochastic, I < H(parent).
        mi_ratio = 1.0 - H_split / np.log(2)  # normalized

        levels.append({
            'level': n,
            'P': P,
            'D': D,
            'S': S,
            'H_split': H_split,
            'mi_ratio': mi_ratio,
        })

        P = D  # cascade continues with dominant branch

    return levels


# ============================================================
# SEC Dynamics
# ============================================================
def sec_entropy_rate(I_field, H_field, alpha=1.0, beta=1.0):
    """
    SEC field equation: dS/dt = alpha * grad(I) - beta * grad(H).

    For discrete nodes: dS_i/dt = alpha * (I_neighbors - I_i) - beta * (H_i - H_mean)

    Parameters:
        I_field: information density array
        H_field: entropy density array
        alpha, beta: coupling constants

    Returns: dS/dt array
    """
    I_mean = np.mean(I_field)
    H_mean = np.mean(H_field)
    grad_I = I_mean - I_field  # simplified: gradient toward mean
    grad_H = H_field - H_mean
    return alpha * grad_I - beta * grad_H


def run_sec_dynamics(n_nodes, n_steps, alpha=1.0, beta=0.5, seed=42):
    """
    Run SEC dynamics on a network of PAC nodes.

    Each node has potential P_i (conserved total). SEC drives entropy
    changes that reorganize the distribution.

    Returns: history of node states at each step.
    """
    rng = np.random.default_rng(seed)

    # Initialize: random potentials summing to 1
    P = rng.dirichlet(np.ones(n_nodes))
    total_P = np.sum(P)

    history = [P.copy()]

    for step in range(n_steps):
        # Information density: -p*log(p) for each node
        I = np.where(P > 1e-15, -P * np.log(P + 1e-30), 0)
        # Entropy: disorder measure (how far from phi-ratio)
        H = np.abs(P - INV_PHI * np.max(P))

        # SEC update
        dS = sec_entropy_rate(I, H, alpha, beta)
        dt = 0.01

        # Apply: redistribute potential based on entropy flow
        P_new = P + dt * dS * P
        P_new = np.maximum(P_new, 1e-15)
        P_new *= total_P / np.sum(P_new)  # conserve total (PAC)

        P = P_new
        history.append(P.copy())

    return np.array(history)


# ============================================================
# Scale-Dependent Cosmological Predictions
# ============================================================

def N_physical(z, a_clock):
    """
    Physical cascade level at redshift z with proper boundary handling.

    The cascade is discrete: N counts completed levels. The continuous
    clock N(t) = a + slope*ln(t) diverges to -inf as t->0. Physical
    constraints:
      - z=0 (now): we are at the current epoch with N = N_max
      - t < t1 (very local): cascade hasn't started at that lookback, use N_max
      - t >= t1: clock formula applies, floored at 1.0

    The floor at N=1 means: the minimum meaningful cascade depth is
    one completed level. Below that, the cascade hasn't established itself.
    """
    if z <= 0:
        # z=0 is the present epoch -- use N at full universe age
        return cascade_clock(T_UNIVERSE, a_clock, B_DFT)

    t_look = z_to_lookback(z)
    if t_look <= 0.001:
        t_look = 0.001

    # t1 boundary: for lookback times shorter than t1, we are in the
    # present epoch where all completed levels exist
    t1 = np.exp(-a_clock / B_DFT)
    if t_look < t1:
        return cascade_clock(T_UNIVERSE, a_clock, B_DFT)

    N_raw = cascade_clock(t_look, a_clock, B_DFT)
    return max(N_raw, 1.0)


def s8_at_z(z, a_clock):
    """
    S8 at redshift z using cascade clock.

    S8(z) = S8_Planck * (1 - f_dissipation(z))
    f_dissipation(z) = (1/phi^2) * (Omega_DM/Omega_M) / N(t_lookback(z))
    """
    N_z = N_physical(z, a_clock)
    f_diss = (1.0 / PHI**2) * (OMEGA_DM / OMEGA_M) / N_z
    return S8_PLANCK * (1.0 - f_diss)


def h0_at_z(z, a_clock):
    """
    H0 ratio at redshift z using cascade clock.

    H0(z) = H0_Planck * phi^{1/N(t_lookback(z))}
    """
    N_z = N_physical(z, a_clock)
    return H0_PLANCK * PHI**(1.0 / N_z)


def w_at_z(z, a_clock):
    """
    Dark energy equation of state at redshift z from cascade clock.

    w(z) = -1 + 1/(3 * phi^{N(t_lookback(z))})
    """
    N_z = N_physical(z, a_clock)
    return -1.0 + 1.0 / (3.0 * PHI**N_z)


# ============================================================
# CascadeClock Class
# ============================================================
class CascadeClock:
    """
    Central M9 object. Encapsulates N(t) = a + log_phi(t/t1) with all
    derived quantities.
    """

    def __init__(self, constrained=True):
        """Fit the cascade clock to M8 data."""
        self.a, self.slope, self.rms = cascade_clock_fit(constrained)
        self.constrained = constrained

        # Derived quantities
        self.t1_gyr = np.exp(-self.a / self.slope)  # lookback time where N=0
        self.n_max = self.N(T_UNIVERSE)  # N at max lookback (full age)
        self.n_floor = int(np.floor(self.n_max))
        self.frac_through = self.n_max - self.n_floor

        # Level completion lookback times: t where N(t) = level
        self.level_times = {}
        for lev in range(1, 10):
            # N(t) = a + slope*ln(t) = lev => t = exp((lev - a)/slope)
            t_lev = np.exp((lev - self.a) / self.slope)
            self.level_times[lev] = t_lev

    def N(self, t_lookback_gyr):
        """Cascade level at lookback time t."""
        return cascade_clock(t_lookback_gyr, self.a, self.slope)

    def N_at_z(self, z):
        """Cascade level at redshift z."""
        t_look = z_to_lookback(z)
        return self.N(max(t_look, 0.001))

    def s8(self, z):
        """S8 at redshift z."""
        return s8_at_z(z, self.a)

    def h0(self, z):
        """H0 at redshift z."""
        return h0_at_z(z, self.a)

    def w(self, z):
        """Dark energy EOS at redshift z."""
        return w_at_z(z, self.a)

    def summary(self):
        """Print clock summary."""
        return {
            'a': self.a,
            'slope': self.slope,
            'slope_label': '1/ln(phi)' if self.constrained else 'free',
            'rms': self.rms,
            't1_gyr': self.t1_gyr,
            'N_now': self.n_max,
            'level_floor': self.n_floor,
            'fraction_through': self.frac_through,
            'level_times': self.level_times,
        }
