"""
quantum_gravity.py -- Shared infrastructure for Milestone 11: Quantum Gravity.

Extends M10's foundations.py with response-time crossover, cascade saturation,
stochastic cascades, cascade density quantization, and Planck star dynamics.

Provides:
- GravitationalResponseTime: tau_grav from cascade depth (exp_01, exp_02)
- CascadeSaturation: MVAE-clamped density, modified metric (exp_04)
- StochasticCascade: PAC cascade + Landauer noise (exp_03, exp_09)
- PACTreeEvaporator: BH evaporation via PAC tree pruning (exp_06)
- PlanckStarDynamics: collapse + bounce with saturation (exp_11)
- QGCorrectedClock: cascade clock with sub-leading corrections (exp_10)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

# Import M10 infrastructure (which chains M9 and M8)
M11_ROOT = Path(__file__).resolve().parent.parent
M10_ROOT = M11_ROOT.parent / "milestone10"
sys.path.insert(0, str(M10_ROOT / "core"))

from foundations import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI, LN2,
    P_D, P_S,
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
    # Cascade clock
    B_DFT, B_FREE, N_DATA, T_UNIVERSE, T_RECOMBINATION,
    cascade_clock, cascade_clock_fit, cascade_level_time,
    current_cascade_level, z_to_lookback, lookback_to_cosmic,
    # Xi decomposition
    xi_info_cost, xi_survival_fraction, xi_decomposition,
    # PAC cascade
    pac_cascade_ratios, pac_cascade_conservation_error,
    # Scale-dependent predictions
    N_physical, s8_at_z, h0_at_z, w_at_z,
    # M10 infrastructure
    LawNegotiator, SelfApplicator,
    measure_hierarchical_structure, measure_temporal_asymmetry,
    # Physical time scales
    T_PLANCK_S, T_EM_S, T_GRAVITY_S, T_HUBBLE_S,
    E_PLANCK_GEV, L_PLANCK_M,
    RESPONSE_TIMES,
    # Infrastructure
    save_results, setup_experiment, PredictionRegistry,
)


# ============================================================
# Physical Constants (Planck units and SI)
# ============================================================
HBAR = 1.054571817e-34        # J·s
C_LIGHT = 2.998e8             # m/s
K_BOLTZMANN = 1.380649e-23    # J/K
M_PLANCK_KG = 2.176434e-8     # kg
M_SUN_KG = 1.989e30           # kg
R_S_SUN = 2 * G_NEWTON * M_SUN_KG / C_LIGHT**2  # Schwarzschild radius of Sun

# MVAE quantities (in Planck units, from minimum_actualization_resolution)
L_MVAE = 1.0 / (2 * (1 - LN2))  # Planck lengths: 1/(2(1-ln2)) ≈ 1.6294
T_MVAE = 1.0 / (2 * LN2)     # Planck times (0.7213)
E_MVAE = LN2                  # Planck energies (Landauer erasure)
RHO_PLANCK = 1.0              # Planck density (in Planck units)

# Response-time constants
TAU_GRAV_SOLAR = T_GRAVITY_S  # ~1.2e-5 s at solar density


# ============================================================
# Response-Time Hierarchy
# ============================================================
def cascade_depth_response_time(depth, base_time=T_PLANCK_S):
    """
    Response time for a force at given Fibonacci cascade depth.

    The response time scales with the coupling strength: weaker coupling
    means longer negotiation time. Coupling alpha ~ phi^(-depth), so
    response time tau ~ base_time / alpha ~ base_time * phi^depth.

    For EM (depth 13): tau ~ 5.4e-44 * phi^13 ~ 4.0e-41 s
    For gravity (depth 183): tau ~ 5.4e-44 * phi^183 ~ huge

    But physical response time is bounded by the light-crossing time
    of the system. We use the MVAE-adjusted formula.
    """
    coupling = PHI ** (-depth)
    return base_time / coupling


def crossover_energy(depth):
    """
    Energy at which perturbation timescale = response time.

    tau_pert = hbar / E, tau_response = t_Planck * phi^depth
    Crossover: E = hbar / tau_response = E_Planck * phi^(-depth)

    For gravity (depth 183): E_cross = E_Planck * phi^(-183) ~ 10^(-19) GeV
    This is NOT the Planck energy -- it's where classical gravity breaks.

    The Planck scale emerges differently: it's where ALL forces' response
    times converge (the negotiation resolution limit).
    """
    return E_PLANCK_GEV * PHI ** (-depth)


def negotiation_resolution_limit():
    """
    The smallest scale where PAC conservation can be maintained
    within one cascade clock tick.

    This is the FOURTH route to the Planck scale (after Landauer,
    Heisenberg, Schwarzschild in MVAE).

    l_neg = c * t_min_tick = c * t_MVAE * t_Planck
    In Planck units: l_neg = T_MVAE = 1/(2*ln(2)) ~ 0.721

    With the MVAE correction: l_neg = L_MVAE = 1/(2*(1-ln(2))) ~ 1.629
    """
    return {
        'l_neg_planck_units': L_MVAE,
        'l_neg_meters': L_MVAE * L_PLANCK_M,
        't_neg_planck_units': T_MVAE,
        't_neg_seconds': T_MVAE * T_PLANCK_S,
        'e_neg_planck_units': E_MVAE,
        'prefactors': {
            'length': f"1/(2*(1-ln(2))) = {1/(2*(1-LN2)):.4f}",
            'time': f"1/(2*ln(2)) = {1/(2*LN2):.4f}",
            'energy': f"ln(2) = {LN2:.6f}",
        },
        'all_fn_of_ln2': True,
    }


def force_response_hierarchy():
    """
    Compute response times for all four forces from cascade depth.

    Returns dict with force name -> (depth, coupling, tau_seconds, tau_ratio_to_planck).
    """
    forces = {
        'strong': {'depth': 3, 'name': 'Strong nuclear'},
        'weak': {'depth': 7, 'name': 'Weak nuclear'},
        'em': {'depth': DEPTH_EM, 'name': 'Electromagnetic'},
        'gravity': {'depth': DEPTH_GRAVITY, 'name': 'Gravitational'},
    }

    results = {}
    for key, info in forces.items():
        d = info['depth']
        coupling = PHI ** (-d)
        # Response time: how long does negotiation take?
        # Faster coupling = faster negotiation
        tau = T_PLANCK_S / coupling
        results[key] = {
            'name': info['name'],
            'depth': d,
            'coupling': float(coupling),
            'tau_seconds': float(tau),
            'tau_ratio_to_planck': float(tau / T_PLANCK_S),
            'log10_tau': float(np.log10(tau)),
        }

    return results


# ============================================================
# Cascade Saturation (Singularity Resolution)
# ============================================================
class CascadeSaturation:
    """
    Models cascade density with MVAE saturation.

    The cascade density rho_c(r) diverges at r=0 in the classical limit.
    MVAE sets a maximum: one actualization per Planck volume.
    Below r_min, the cascade saturates at rho_max.
    """

    def __init__(self, M_solar_masses=1.0):
        self.M = M_solar_masses
        self.M_kg = M_solar_masses * M_SUN_KG
        self.r_s = 2 * G_NEWTON * self.M_kg / C_LIGHT**2  # Schwarzschild radius (m)
        self.r_s_planck = self.r_s / L_PLANCK_M            # in Planck lengths

        # Saturation radius: where rho_c = rho_Planck
        # Classical: rho_c(r) ~ rho_crit * (r_s / r)
        # rho_crit ~ M / r_s^3 in natural units
        # r_min = r_s * (rho_crit / rho_Planck) ~ r_s * (M/M_Planck)^(-2)
        self.M_planck_ratio = self.M_kg / M_PLANCK_KG
        # In Planck units
        self.r_min_planck = self.r_s_planck / self.M_planck_ratio**2
        self.r_min_meters = self.r_min_planck * L_PLANCK_M

    def density_profile(self, r_planck):
        """
        Cascade density at radius r (in Planck units).
        Classical: rho ~ (r_s / r) for r >> r_min
        Saturated: rho = rho_Planck for r <= r_min
        """
        r = np.asarray(r_planck, dtype=float)
        rho = np.where(
            r > self.r_min_planck,
            self.r_s_planck / r,      # Classical (1/r profile)
            RHO_PLANCK,                # Saturated
        )
        return rho

    def metric_g_tt(self, r_planck):
        """
        Modified metric component g_tt.
        Classical Schwarzschild: g_tt = 1 - r_s/r
        Modified (r < r_min): de Sitter interior with rho = rho_Planck
        """
        r = np.asarray(r_planck, dtype=float)
        r_s = self.r_s_planck

        # Classical region
        g_tt_classical = 1.0 - r_s / np.maximum(r, 1e-100)

        # De Sitter interior: g_tt = 1 - (r/r_dS)^2
        # where r_dS = sqrt(3 / (8*pi*rho_Planck)) in Planck units
        r_dS = np.sqrt(3.0 / (8 * PI * RHO_PLANCK))
        g_tt_interior = 1.0 - (r / r_dS)**2

        return np.where(r > self.r_min_planck, g_tt_classical, g_tt_interior)

    def kretschner_scalar(self, r_planck):
        """
        Kretschner scalar K = R_abcd R^abcd.
        Schwarzschild: K = 48 M^2 / r^6 (diverges at r=0)
        De Sitter: K = 8/3 * Lambda^2 = constant (finite!)
        """
        r = np.asarray(r_planck, dtype=float)
        M_p = self.M_planck_ratio

        K_schwarz = 48 * M_p**2 / np.maximum(r, 1e-100)**6
        K_deSitter = 8.0 / 3.0 * (8 * PI * RHO_PLANCK / 3.0)**2

        return np.where(r > self.r_min_planck, K_schwarz, K_deSitter)

    def information_content(self, n_shells=1000):
        """
        Information content from cascade mode counting.

        In a PAC-conserving cascade, interior modes are correlated with
        their parent nodes. Independent information is carried by the
        density GRADIENT (change per level), not the density itself.

        Surface-gradient information: I = integral of 4*pi*r^2 * |d(rho)/dr| dr
        For rho = r_s/r: |d(rho)/dr| = r_s/r^2, so 4*pi*r^2 * r_s/r^2 = 4*pi*r_s
        Integrating: I = 4*pi*r_s * (r_s - r_min) ~ 4*pi*r_s^2 ~ M^2 (area law).
        """
        r = np.linspace(self.r_min_planck, self.r_s_planck, n_shells)
        rho = self.density_profile(r)
        # Gradient of density: independent information per radial step
        drho_dr = np.abs(np.gradient(rho, r))
        # Surface-weighted gradient: independent cascade modes per shell
        integrand = 4 * PI * r**2 * drho_dr
        info = np.trapz(integrand, r)
        return float(info)


# ============================================================
# Hawking Temperature from Cascade
# ============================================================
def hawking_temperature_planck(M_solar_masses):
    """
    Hawking temperature in Planck units.
    T_H = 1 / (8*pi*M) in Planck units.
    """
    M_planck = M_solar_masses * M_SUN_KG / M_PLANCK_KG
    return 1.0 / (8 * PI * M_planck)


def hawking_temperature_kelvin(M_solar_masses):
    """Hawking temperature in Kelvin."""
    M_kg = M_solar_masses * M_SUN_KG
    return HBAR * C_LIGHT**3 / (8 * PI * G_NEWTON * M_kg * K_BOLTZMANN)


def hawking_with_correction(M_solar_masses):
    """
    Hawking temperature with micro-BH correction from cascade saturation.
    T_corrected = T_H * (1 - (r_min/r_s)^2)

    For stellar-mass BH: correction ~ 10^(-66)
    For M ~ 10*M_Planck: correction ~ 10%
    """
    sat = CascadeSaturation(M_solar_masses)
    T_H = hawking_temperature_planck(M_solar_masses)
    correction = 1.0 - (sat.r_min_planck / sat.r_s_planck)**2
    return {
        'T_H_planck': float(T_H),
        'T_corrected_planck': float(T_H * correction),
        'correction_factor': float(correction),
        'r_min_over_r_s': float(sat.r_min_planck / sat.r_s_planck),
    }


def hawking_TM_product(M_solar_masses):
    """T*M product (should be constant = 1/(8*pi) in Planck units)."""
    M_planck = M_solar_masses * M_SUN_KG / M_PLANCK_KG
    T_H = hawking_temperature_planck(M_solar_masses)
    return float(T_H * M_planck)


# ============================================================
# Stochastic Cascade (Irreversibility)
# ============================================================
class StochasticCascade:
    """
    PAC cascade with Landauer noise at each level.

    Deterministic cascade is reversible (Loschmidt echo ~ 3.6%).
    Adding k_BT*ln(2) erasure noise at each level produces genuine
    irreversibility and entropy production.
    """

    def __init__(self, n_levels=20, seed=42, split_ratio=None):
        self.n_levels = n_levels
        self.rng = np.random.RandomState(seed)
        self.split_ratio = split_ratio if split_ratio is not None else INV_PHI

    def run_forward(self, initial_value=1.0, noise_amplitude=0.01):
        """Run cascade forward with Landauer noise at each level."""
        values = [initial_value]
        noises = []
        for n in range(self.n_levels):
            # Deterministic: geometric decay by split_ratio
            v_det = values[-1] * self.split_ratio
            # Landauer noise: k_BT*ln(2) per erasure
            noise = self.rng.randn() * noise_amplitude * LN2
            noises.append(noise)
            values.append(v_det + noise)
        return np.array(values), np.array(noises)

    def run_reverse(self, final_value, noises):
        """Attempt to reverse the cascade (requires supplying noise record)."""
        values = [final_value]
        for n in range(self.n_levels - 1, -1, -1):
            v_det = values[-1] / self.split_ratio  # Reverse of *split_ratio
            values.append(v_det - noises[n])
        return np.array(values[::-1])

    def loschmidt_echo(self, initial_value=1.0, noise_amplitude=0.01):
        """
        Run forward then reverse without noise record.
        Measure reconstruction error (Loschmidt echo).
        """
        # Forward with noise
        forward, noises = self.run_forward(initial_value, noise_amplitude)

        # Reverse WITHOUT noise correction (lost information)
        reverse_values = [forward[-1]]
        for n in range(self.n_levels):
            v_det = reverse_values[-1] / self.split_ratio
            reverse_values.append(v_det)
        reverse_values = np.array(reverse_values[::-1])

        # Echo error
        error = np.abs(reverse_values[0] - initial_value) / abs(initial_value)
        return {
            'echo_error': float(error),
            'forward_values': forward,
            'reverse_values': reverse_values,
            'n_levels': self.n_levels,
            'noise_amplitude': noise_amplitude,
        }

    def entropy_production(self, initial_value=1.0, noise_amplitude=0.01, n_trials=100):
        """
        Measure entropy production per cascade level.

        Uses cascade contraction rate (from M9 cascade_info_loss approach):
        at each level, the split contracts potential by split_ratio.

        Two components:
        1. Contraction rate: measured ln(P_n / P_{n+1}) -> ln(1/split_ratio) = ln(b)
           (information lost about potential magnitude per level)
        2. Counting overhead: gamma = 0.5772 (Euler-Mascheroni from harmonic series)

        Total: Xi(b) = gamma + ln(b) nats per cascade boundary crossing.
        For b=phi: Xi = gamma + ln(phi) = 1.0584 (the standard DFT value).
        """
        contraction_rates = []
        sigma = noise_amplitude * LN2
        target_contraction = -np.log(self.split_ratio)  # ln(1/split_ratio) = ln(b)

        for trial in range(n_trials):
            self.rng = np.random.RandomState(42 + trial)
            forward, noises = self.run_forward(initial_value, noise_amplitude)

            # Measure contraction rate at each level where signal >> noise
            level_rates = []
            for k in range(self.n_levels):
                if forward[k] > 5 * max(sigma, 1e-30) and forward[k + 1] > 1e-30:
                    level_rates.append(np.log(forward[k] / forward[k + 1]))

            if level_rates:
                contraction_rates.append(np.mean(level_rates))

        mean_contraction = float(np.mean(contraction_rates))
        # Total entropy per level = contraction rate + counting overhead
        mean_entropy = mean_contraction + GAMMA_EM

        return {
            'mean_entropy_per_level': mean_entropy,
            'std_entropy_per_level': float(np.std(contraction_rates)),
            'mean_contraction_rate': mean_contraction,
            'target_contraction': float(target_contraction),
            'counting_overhead': float(GAMMA_EM),
            'xi_target': float(GAMMA_EM + target_contraction),
            'n_trials': n_trials,
        }


# ============================================================
# PAC Tree Evaporator (Page Curve)
# ============================================================
class PACTreeEvaporator:
    """
    Models BH evaporation as pruning a PAC tree.

    The PAC tree is a binary tree with conservation at each node:
    parent = left + right. Leaves represent horizon degrees of freedom.
    Evaporation = pruning leaves one at a time.

    Entanglement entropy of radiation subsystem follows the Page curve.
    """

    def __init__(self, n_leaves, seed=42):
        self.n_leaves = n_leaves
        self.rng = np.random.RandomState(seed)
        # Initialize leaf values (random, PAC-conserving)
        raw = self.rng.exponential(1.0, n_leaves)
        self.leaves = raw / raw.sum()  # Normalized to 1
        self.evaporated = []
        self.remaining = list(range(n_leaves))

    def evaporate_one(self):
        """Remove one leaf (Hawking quantum)."""
        if not self.remaining:
            return None
        idx = self.rng.choice(len(self.remaining))
        leaf_idx = self.remaining.pop(idx)
        self.evaporated.append(leaf_idx)
        return leaf_idx

    def entanglement_entropy(self):
        """
        Entanglement entropy between evaporated and remaining subsystems.
        For a random bipartition of a PAC-conserving system:
        S = min(k, N-k) * ln(2) approximately.
        """
        k = len(self.evaporated)
        N = self.n_leaves
        if k == 0 or k == N:
            return 0.0
        # Page formula for random pure state
        d_A = min(k, N - k)
        d_B = max(k, N - k)
        # S ~ ln(d_A) - d_A / (2 * d_B) for large dimensions
        if d_A <= 1:
            return 0.0
        S = np.log(d_A) - d_A / (2.0 * d_B)
        return max(0.0, float(S))

    def run_evaporation(self):
        """Run full evaporation, track entanglement entropy."""
        entropies = [0.0]
        for step in range(self.n_leaves):
            self.evaporate_one()
            S = self.entanglement_entropy()
            entropies.append(S)
        return {
            'entropies': np.array(entropies),
            'page_time_fraction': 0.5,  # Should peak at k/N = 0.5
            'peak_idx': int(np.argmax(entropies)),
            'peak_fraction': float(np.argmax(entropies) / self.n_leaves),
            'final_entropy': float(entropies[-1]),
            'symmetric': float(abs(entropies[-1])),
        }


def page_time_scaling(M_solar_masses):
    """
    Page time from cascade counting.
    t_Page ~ S^{3/2} * t_Planck (for thermal state)
    where S = A / (4 * l_P^2) = 4*pi*r_s^2 / (4*l_P^2) = pi*r_s^2/l_P^2
    """
    sat = CascadeSaturation(M_solar_masses)
    S_BH = PI * sat.r_s_planck**2  # Bekenstein-Hawking entropy in Planck units
    t_page = S_BH * T_PLANCK_S  # Simplified: proportional to S
    return {
        'S_BH': float(S_BH),
        't_page_seconds': float(t_page),
        't_evap_seconds': float(5120 * PI * G_NEWTON**2 * (M_solar_masses * M_SUN_KG)**3
                                 / (HBAR * C_LIGHT**4)),
        'M_solar': M_solar_masses,
    }


def scrambling_time(M_solar_masses):
    """
    Scrambling time: t_scram ~ S * t_P * ln(S).
    Sekino-Susskind fast scrambling conjecture.
    """
    sat = CascadeSaturation(M_solar_masses)
    S_BH = PI * sat.r_s_planck**2
    t_scram = S_BH * T_PLANCK_S * np.log(S_BH)
    return {
        'S_BH': float(S_BH),
        't_scramble_seconds': float(t_scram),
        'log_S': float(np.log(S_BH)),
    }


# ============================================================
# Planck Star Dynamics
# ============================================================
class PlanckStarDynamics:
    """
    Collapse and bounce dynamics with cascade saturation.

    When cascade density reaches MVAE limit, PAC conservation
    forces a bounce (information pressure exceeds gravitational
    compression).
    """

    def __init__(self, M_solar_masses=1.0):
        self.sat = CascadeSaturation(M_solar_masses)
        self.M = M_solar_masses

    def bounce_time_planck(self):
        """
        Bounce timescale in Planck units.
        t_bounce ~ M_planck_ratio * sqrt(r_min / r_s)
        """
        ratio = self.sat.r_min_planck / self.sat.r_s_planck
        return float(self.sat.M_planck_ratio * np.sqrt(ratio))

    def burst_energy_planck(self):
        """
        Energy of the bounce burst in Planck units.
        E_burst ~ T_Planck * (M/M_Planck)^(-1/3)
        """
        return float(self.sat.M_planck_ratio ** (-1.0 / 3.0))

    def pac_forces_bounce(self, epsilon_pac=0.0):
        """
        Test whether PAC conservation forces the bounce.
        epsilon_pac: fractional violation of conservation (0 = perfect PAC).
        Returns True if bounce occurs, False if collapse to singularity.
        """
        # With perfect PAC: information at max density creates outward pressure
        # With broken PAC: information can be destroyed, no pressure
        # Threshold: bounce if information pressure > gravitational pressure
        # Information pressure ~ rho_max * (1 - epsilon_pac)
        # Gravitational pressure ~ M / r_min^2
        info_pressure = RHO_PLANCK * (1.0 - epsilon_pac)
        grav_pressure = self.sat.M_planck_ratio / self.sat.r_min_planck**2

        # Normalize by Planck pressure
        return info_pressure > grav_pressure * epsilon_pac


# ============================================================
# QG-Corrected Cascade Clock
# ============================================================
class QGCorrectedClock:
    """
    Cascade clock with sub-leading QG response-time corrections.

    N(t) = a + B * ln(t) * correction(t)

    where correction(t) = 1 - (t_Planck / t)^alpha

    At early times (t ~ t_Planck), correction -> 0: cascade hasn't started.
    At late times (t >> t_Planck), correction -> 1: standard clock.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.B = B_DFT  # 1/ln(phi) = 2.0781
        # Fit the a parameter from M9 data
        a_fit_result = cascade_clock_fit(self.B)
        self.a_clock = a_fit_result[0]  # ~1.3596

    def N_corrected(self, t_lookback_gyr):
        """Corrected cascade level at lookback time t (in Gyr)."""
        t_planck_gyr = T_PLANCK_S / (3.156e16)  # Convert Planck time to Gyr
        N_base = cascade_clock(t_lookback_gyr, self.a_clock)
        correction = 1.0 - (t_planck_gyr / np.maximum(t_lookback_gyr, t_planck_gyr)) ** self.alpha
        return N_base * correction

    def w_corrected(self, z, alpha_correction=None):
        """
        Dark energy EOS with sub-leading QG correction.
        Standard: w(z) = -1 + 1/(3 * phi^N(z))
        Corrected: uses N_corrected instead of N.
        """
        if alpha_correction is not None:
            self.alpha = alpha_correction
        t = z_to_lookback(z)
        N = self.N_corrected(t)
        N = max(N, 1.0)  # Floor at 1
        return -1.0 + 1.0 / (3.0 * PHI ** N)

    def s8_corrected(self, z):
        """S8 with QG correction to cascade clock."""
        t = z_to_lookback(z)
        N = self.N_corrected(t)
        N = max(N, 1.0)
        return s8_at_z(z)  # Placeholder — uses standard for now


# ============================================================
# GW Dispersion
# ============================================================
def gw_dispersion(E_gev):
    """
    Gravitational wave speed deviation from c.
    delta_v/c ~ (E / E_Planck)^2

    From cascade density quantization: discrete spacetime at Planck scale
    modifies the dispersion relation by O((k/k_Planck)^2).
    """
    ratio = E_gev / E_PLANCK_GEV
    return float(ratio ** 2)


def minimum_bh_mass_planck():
    """
    Minimum black hole mass from cascade saturation.
    Below this mass, cascade saturates before horizon forms.
    M_min ~ M_Planck * phi^2
    """
    return {
        'M_min_planck': float(PHI**2),
        'M_min_kg': float(PHI**2 * M_PLANCK_KG),
        'M_min_solar': float(PHI**2 * M_PLANCK_KG / M_SUN_KG),
        'rationale': 'Cascade saturates before Zeno completion for M < M_Planck * phi^2',
    }
