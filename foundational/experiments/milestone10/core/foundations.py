"""
foundations.py -- Shared infrastructure for Milestone 10: Symmetry Self-Application.

Extends M9's infodynamics.py with uniqueness testing, polarity dynamics,
law-negotiation, annealing analysis, and Xi universality infrastructure.

Provides:
- SelfApplicator: discrete dynamical system for uniqueness exhaustion (exp_01)
- LawNegotiator: conservation law maintained by iterative negotiation (exp_05)
- Polarity dynamics: coupled info/thermo systems (exp_04, exp_06)
- Annealing analysis: SM residual compilation, glassy spectrum fitting (exp_07)
- Xi universality: Markov chain, annealing, RG flow tools (exp_08)
- Fossil arithmetic: alternative PAC closures (exp_09)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import levy_stable, norm, uniform, kstest, spearmanr
from scipy.optimize import minimize

# Import M9 infrastructure (which chains M8's bsm.py)
M10_ROOT = Path(__file__).resolve().parent.parent
M9_ROOT = M10_ROOT.parent / "milestone9"
sys.path.insert(0, str(M9_ROOT / "core"))

from infodynamics import (
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
    # Infrastructure
    save_results, setup_experiment, PredictionRegistry,
)


# ============================================================
# Physical Time Scales
# ============================================================
T_PLANCK_S = 5.391e-44           # Planck time in seconds
T_EM_S = T_PLANCK_S / ALPHA_EM  # EM response scale (~7.4e-42 s)
T_HUBBLE_S = 1.0 / (H0_PLANCK * 1e3 / 3.086e22)  # Hubble time (~4.6e17 s)
T_GRAVITY_S = 1.0 / np.sqrt(G_NEWTON * 1e3)  # Gravitational response (~1.2e-5 s, at solar density)

# Planck units
E_PLANCK_GEV = M_PLANCK_GEV     # ~1.22e19 GeV
L_PLANCK_M = 1.616e-35          # Planck length in meters

# Response time hierarchy (in seconds, ordered fast -> slow)
RESPONSE_TIMES = {
    'planck':  T_PLANCK_S,
    'em':      T_EM_S,
    'gravity': T_GRAVITY_S,
    'hubble':  T_HUBBLE_S,
}


# ============================================================
# SM Fine-Tuning Residuals
# ============================================================
SM_RESIDUALS = {
    'alpha_s_uv': {
        'name': 'Strong coupling UV deviation',
        'value': 0.1179,
        'natural': 1.0,
        'residual': 0.1179,
        'energy_scale_gev': 91.2,
        'reference': 'PDG 2024, alpha_s(M_Z)',
    },
    'theta_qcd': {
        'name': 'Strong CP violation',
        'value': 1e-10,
        'natural': 1.0,
        'residual': 1e-10,
        'energy_scale_gev': 1.0,
        'reference': 'Neutron EDM bound',
    },
    'eta_baryon': {
        'name': 'Baryon asymmetry',
        'value': 6.1e-10,
        'natural': 1.0,
        'residual': 6.1e-10,
        'energy_scale_gev': 1e2,
        'reference': 'BBN + CMB, PDG 2024',
    },
    'cc_planck': {
        'name': 'Cosmological constant',
        'value': 2.89e-122,
        'natural': 1.0,
        'residual': 2.89e-122,
        'energy_scale_gev': 2.4e-3,
        'reference': 'Planck 2018, Lambda in Planck units',
    },
    'higgs_hierarchy': {
        'name': 'Higgs mass hierarchy',
        'value': 125.25 / 1.22e19,
        'natural': 1.0,
        'residual': 125.25 / 1.22e19,
        'energy_scale_gev': 125.25,
        'reference': 'PDG 2024, m_H / M_Planck',
    },
    'jarlskog': {
        'name': 'Jarlskog invariant',
        'value': 3.18e-5,
        'natural': 1.0,
        'residual': 3.18e-5,
        'energy_scale_gev': 80.4,
        'reference': 'PDG 2024, CKM CP violation',
    },
    'nu_mass_ratio': {
        'name': 'Neutrino-charged lepton mass ratio',
        'value': 0.06 / 1776.9,
        'natural': 1.0,
        'residual': 0.06 / 1776.9,
        'energy_scale_gev': 0.06e-3,
        'reference': 'Sum(m_nu) < 0.12 eV / m_tau',
    },
    'higgs_stability': {
        'name': 'Higgs self-coupling deviation from stability',
        'value': 0.126,
        'natural': 0.5,
        'residual': abs(0.126 - 0.5) / 0.5,
        'energy_scale_gev': 125.25,
        'reference': 'lambda_H near metastability boundary',
    },
    'electron_mass': {
        'name': 'Electron Yukawa coupling',
        'value': 0.511e-3 / 246.0,
        'natural': 1.0,
        'residual': 0.511e-3 / 246.0,
        'energy_scale_gev': 0.511e-3,
        'reference': 'y_e = m_e / v, PDG 2024',
    },
    'muon_g2_anomaly': {
        'name': 'Muon g-2 anomaly',
        'value': 2.49e-9,
        'natural': ALPHA_EM / PI,
        'residual': 2.49e-9 / (ALPHA_EM / PI),
        'energy_scale_gev': 0.1057,
        'reference': 'Fermilab 2023, Delta(a_mu)',
    },
}


def compile_sm_residuals():
    """Return SM_RESIDUALS dict."""
    return SM_RESIDUALS


# ============================================================
# SelfApplicator: Discrete Dynamical System for Uniqueness Tests
# ============================================================
class SelfApplicator:
    """
    Recursive coupling network for testing the uniqueness argument.

    N nodes with coupling matrix W. Nonlinear dynamics x' = tanh(W @ x).
    Parameterized by (self_applies, symmetric):

    - self_applies=False: W is fixed, only x evolves
    - self_applies=True:  W evolves by applying itself: W' = tanh(W @ W / n)
      This is GENUINE self-application: the rule that transforms state
      also transforms itself through the same operation.

    - symmetric=True:  W = W^T enforced each step (real eigenvalues → stability)
    - symmetric=False: W unconstrained (complex eigenvalues → drift/chaos)

    The mathematical reason this works:
    - Symmetric W has real eigenvalues. W² = W^T W is positive semidefinite.
      tanh preserves this structure → stable multi-scale evolution.
    - Asymmetric W has complex eigenvalues. W² amplifies imaginary parts.
      The coupling drifts through eigenvalue space → chaos or collapse.
    - Fixed W: x converges to a fixed point or simple cycle. No new structure
      emerges because the rule can't adapt. Low-rank covariance.
    """

    def __init__(self, rule_seed, self_applies=True, symmetric=True, size=32):
        self.n = size
        self.self_applies = self_applies
        self.symmetric = symmetric
        self.rng = np.random.RandomState(rule_seed)

        # Initialize state
        self.state = self.rng.randn(self.n) * 0.5

        # Initialize coupling matrix, then normalize spectral radius
        # to target_sr for fair comparison across all quadrants.
        self.W = self.rng.randn(self.n, self.n) / np.sqrt(self.n)
        if symmetric:
            self.W = (self.W + self.W.T) / 2

        # Normalize spectral radius to mildly supercritical (1.2)
        # so multiple eigenvalues can exceed 1.
        self._target_sr = 1.2
        if symmetric:
            eigvals = np.linalg.eigvalsh(self.W)
        else:
            eigvals = np.linalg.eigvals(self.W)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-10:
            self.W = self.W * (self._target_sr / sr)

    def step(self):
        """Advance one time step."""
        # State update: bounded nonlinear map
        self.state = np.tanh(self.W @ self.state)

        if self.self_applies:
            # Anti-Hebbian eigenvalue modulation with diversity floor.
            #
            # Active eigenmodes (aligned with state) are weakened;
            # inactive modes are strengthened. A minimum magnitude
            # floor prevents the spectrum from collapsing.
            #
            # For symmetric W: eigh gives real orthogonal eigenvectors.
            #   Modulation cleanly targets individual eigenvalues.
            #   Cycling through eigenvectors → structured multi-scale.
            #
            # For asymmetric W: SVD gives orthogonal singular vectors
            #   that DON'T align with dynamical modes (eigenvectors are
            #   complex, non-orthogonal). Modulation is approximate →
            #   incoherent cycling → unstructured dynamics.
            x = self.state

            if self.symmetric:
                eigvals, eigvecs = np.linalg.eigh(self.W)
                projections = (eigvecs.T @ x) ** 2
            else:
                U, S_vals, Vt = np.linalg.svd(self.W, full_matrices=False)
                projections = (Vt @ x) ** 2
                eigvals = S_vals.copy()

            total = np.sum(projections) + 1e-10
            activities = projections / total
            mean_act = 1.0 / len(eigvals)

            # Anti-Hebbian: active modes weakened, inactive strengthened.
            # Gentle modulation — cycling is gradual, not instantaneous.
            modulation = np.ones_like(eigvals, dtype=float)
            modulation[activities > 2.0 * mean_act] = 0.95
            modulation[activities < 0.5 * mean_act] = 1.01
            new_eigvals = eigvals * modulation

            # Rescale spectral radius to target.
            # This alone prevents rank-1 collapse: even if the dominant
            # mode is weakened, the next mode inherits the target_sr.
            sr = np.max(np.abs(new_eigvals))
            if sr > 1e-10:
                new_eigvals = new_eigvals * (self._target_sr / sr)

            # Reconstruct W
            if self.symmetric:
                self.W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
            else:
                self.W = U @ np.diag(new_eigvals) @ Vt

        return self.state

    def run(self, n_steps):
        """Run for n_steps, return full trajectory (n_steps x n)."""
        trajectory = np.zeros((n_steps, self.n))
        for t in range(n_steps):
            self.step()
            trajectory[t] = self.state
        return trajectory

    def lyapunov(self, n_steps=500):
        """Estimate largest Lyapunov exponent via perturbation."""
        orig_state = self.state.copy()
        orig_W = self.W.copy()

        self.run(n_steps)
        final_orig = self.state.copy()

        self.state = orig_state.copy()
        self.W = orig_W.copy()
        perturb_idx = self.rng.randint(0, self.n)
        self.state[perturb_idx] += 0.001

        self.run(n_steps)
        final_pert = self.state.copy()

        diff = np.linalg.norm(final_orig - final_pert)
        if diff < 1e-10:
            return -np.inf
        return np.log(diff / 0.001) / n_steps


def measure_hierarchical_structure(trajectory):
    """
    Measure whether a trajectory shows stable hierarchical structure.

    Uses SVD-based multi-scale analysis: a system has hierarchy if the
    trajectory covariance has multiple significant eigenvalues (active scales)
    that persist over time.

    Returns dict with:
    - has_hierarchy: bool (>= 3 active scales, sustained)
    - n_active_scales: number of significant singular values
    - scale_persistence: fraction of time windows where hierarchy persists
    - mean_complexity: effective dimensionality (participation ratio)
    """
    n_steps, n = trajectory.shape

    if n_steps < 20:
        return {
            'has_hierarchy': False,
            'n_active_scales': 0,
            'scale_persistence': 0.0,
            'mean_complexity': 0.0,
            'sustained_steps': 0,
        }

    # Skip initial transient
    traj = trajectory[n_steps // 4:]

    # SVD of trajectory (each row = one time point)
    try:
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            'has_hierarchy': False,
            'n_active_scales': 0,
            'scale_persistence': 0.0,
            'mean_complexity': 0.0,
            'sustained_steps': 0,
        }

    # Normalize singular values
    S_norm = S / (S[0] + 1e-10)

    # Count active scales (> 5% of leading)
    n_active = int(np.sum(S_norm > 0.05))

    # Participation ratio: effective dimensionality
    S2 = S ** 2
    S2_sum = np.sum(S2)
    if S2_sum > 0:
        participation = S2_sum ** 2 / np.sum(S2 ** 2)
    else:
        participation = 0.0

    # Check persistence: does multi-scale structure hold across time windows?
    window = max(20, len(traj) // 5)
    n_windows = max(1, len(traj) // window)
    persistent_windows = 0

    for w in range(n_windows):
        chunk = traj[w * window:(w + 1) * window]
        if len(chunk) < 10:
            continue
        try:
            _, S_w, _ = np.linalg.svd(chunk, full_matrices=False)
            S_w_norm = S_w / (S_w[0] + 1e-10)
            if np.sum(S_w_norm > 0.05) >= 3:
                persistent_windows += 1
        except np.linalg.LinAlgError:
            pass

    persistence = persistent_windows / max(n_windows, 1)

    # Non-stationarity check: genuine hierarchy requires that the
    # covariance structure evolves (not just a fixed-W oscillation).
    # Compare first-half vs second-half SVD spectra.
    half = len(traj) // 2
    non_stationary = False
    if half >= 10:
        try:
            _, S1, _ = np.linalg.svd(traj[:half], full_matrices=False)
            _, S2, _ = np.linalg.svd(traj[half:], full_matrices=False)
            min_len = min(len(S1), len(S2))
            S1n = S1[:min_len] / (S1[0] + 1e-10)
            S2n = S2[:min_len] / (S2[0] + 1e-10)
            spectral_drift = np.mean(np.abs(S1n - S2n))
            non_stationary = spectral_drift > 0.02
        except np.linalg.LinAlgError:
            non_stationary = False

    # Hierarchy requires:
    # 1. Multiple active scales (>= 3)
    # 2. Not TOO many (noise-like; <= n/2)
    # 3. Persistent across time windows
    # 4. Non-stationary (covariance evolves)
    has_hierarchy = (n_active >= 3 and n_active <= n // 2
                     and persistence > 0.5 and non_stationary)

    return {
        'has_hierarchy': has_hierarchy,
        'n_active_scales': n_active,
        'scale_persistence': float(persistence),
        'mean_complexity': float(participation),
        'sustained_steps': int(persistence * len(traj)),
        'non_stationary': non_stationary,
    }


def measure_temporal_asymmetry(trajectory):
    """
    Measure concentration of change in a trajectory.

    Computes max_step / total_path_length.
    - Static (one-shot) resolution: all change in one step → ≈ 1.0
    - Processual (gradual) resolution: change distributed → ≈ 1/N

    Returns: float in [0, 1]
    """
    if len(trajectory) < 2:
        return 0.0

    # Compute step sizes
    steps = np.array([np.linalg.norm(trajectory[t + 1] - trajectory[t])
                      for t in range(len(trajectory) - 1)])
    total = np.sum(steps)
    if total < 1e-10:
        return 0.0

    return float(np.max(steps) / total)


# ============================================================
# Two-Circle Mutual Reference Dynamics
# ============================================================
def run_two_circle_dynamics(f, n_steps, x0, y0, symmetric=True):
    """
    Run two-circle mutual reference: x_{n+1} = f(y_n), y_{n+1} = f(x_n).

    If symmetric=True, f is the same for both directions.
    Returns dict with trajectories, boundedness, scale separation.
    """
    xs = [x0]
    ys = [y0]

    for _ in range(n_steps):
        x_new = f(ys[-1])
        y_new = f(xs[-1])
        # Clip to prevent overflow
        x_new = np.clip(x_new, -1e6, 1e6)
        y_new = np.clip(y_new, -1e6, 1e6)
        xs.append(x_new)
        ys.append(y_new)

    xs = np.array(xs)
    ys = np.array(ys)

    # Boundedness check
    bounded = np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))
    if bounded:
        bounded = np.max(np.abs(xs)) < 1e5 and np.max(np.abs(ys)) < 1e5

    # Check for termination (fixed point)
    if bounded and len(xs) > 10:
        tail_var = np.var(xs[-10:]) + np.var(ys[-10:])
        terminated = tail_var < 1e-10
    else:
        terminated = False

    # Power spectrum for scale separation
    n_peaks = 0
    peak_ratio = 0.0
    if bounded and not terminated and len(xs) > 20:
        combined = xs[10:] + ys[10:]  # Combined signal
        fft = np.fft.rfft(combined)
        power = np.abs(fft) ** 2
        # Find peaks
        peaks = []
        for i in range(1, len(power) - 1):
            if power[i] > power[i-1] and power[i] > power[i+1]:
                if power[i] > 3 * np.median(power):
                    peaks.append(i)
        n_peaks = len(peaks)
        if len(peaks) >= 2:
            peak_ratio = peaks[1] / peaks[0] if peaks[0] > 0 else 0

    return {
        'xs': xs,
        'ys': ys,
        'bounded': bounded,
        'terminated': terminated,
        'non_terminating_bounded': bounded and not terminated,
        'n_peaks': n_peaks,
        'peak_ratio': peak_ratio,
    }


# ============================================================
# Polarity Dynamics (Info-Dynamics + Thermodynamics)
# ============================================================
def info_dynamics_step(state, dt=0.01, growth_rate=1.0):
    """
    Info-dynamics update: structure-building, exploratory.

    Multiplicative amplification of local gradients — short-range activation.
    Each element grows proportionally to its local structure (gradient magnitude).
    Alone, this diverges: positive feedback between state amplitude and gradient.
    """
    gradient = np.abs(np.roll(state, 1) - np.roll(state, -1)) / 2
    return state * (1.0 + dt * growth_rate * gradient)


def thermo_dynamics_step(state, dt=0.01, dissipation_rate=1.0):
    """
    Thermodynamic update: dissipative, equilibrium-seeking.

    Exponential relaxation toward the mean — long-range inhibition.
    Each element decays toward the spatial average. Alone, this collapses
    the state to a uniform flat profile (maximum entropy, zero structure).
    """
    mean_val = np.mean(state)
    return state + dt * dissipation_rate * (mean_val - state)


def coupled_polarity_step(state, dt=0.01, alpha=1.0, beta=1.0):
    """
    Coupled polarity: short-range activation (alpha) + long-range inhibition (beta).

    The interplay between local structure amplification and global smoothing
    produces stable non-trivial patterns when alpha and beta are balanced.
    """
    gradient = np.abs(np.roll(state, 1) - np.roll(state, -1)) / 2
    mean_val = np.mean(state)

    info_term = alpha * gradient * state                  # multiplicative growth
    thermo_term = beta * (mean_val - state)               # relaxation to mean
    new_state = state + dt * (info_term + thermo_term)

    # Prevent numerical overflow while preserving dynamics
    new_state = np.clip(new_state, -1e6, 1e6)
    return new_state


def measure_complexity(state):
    """Measure structural complexity of a 1D state (normalized Shannon entropy)."""
    n = len(state)
    if n < 2:
        return 0.0
    # Bin the values
    n_bins = min(20, n // 2)
    counts, _ = np.histogram(state, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ============================================================
# LawNegotiator: Conservation by Iterative Negotiation
# ============================================================
class LawNegotiator:
    """
    A conservation law maintained by iterative negotiation with response time tau.

    Models the §6 claim: laws are not static rules but maintained equilibria.
    When perturbation rate exceeds 1/tau, the law starts to fluctuate.
    """

    def __init__(self, n_participants=50, response_time=1.0, conserved_total=100.0):
        self.n = n_participants
        self.tau = response_time
        self.conserved_total = conserved_total
        self.state = np.ones(n_participants) * (conserved_total / n_participants)
        self.violation_history = []

    def _negotiate_once(self):
        """One round of partial negotiation: correct 50% of the current error."""
        current_total = np.sum(self.state)
        error = current_total - self.conserved_total
        # Partial correction: each round halves the error
        correction = 0.5 * error / self.n
        self.state -= correction

    def perturb_and_negotiate(self, perturbation_rate, n_steps=1000, amplitude=1.0):
        """
        Perturb the system at given rate, negotiate to restore conservation.

        perturbation_rate: perturbations per unit time (relative to 1/tau)
        Higher rate → less negotiation time between perturbations → more violation.
        Returns: dict with violation statistics
        """
        self.state = np.ones(self.n) * (self.conserved_total / self.n)
        self.violation_history = []

        rng = np.random.RandomState(42)

        # Available negotiation rounds per perturbation cycle:
        # Low rate → many rounds → good correction
        # High rate → few rounds → poor correction
        negotiation_rounds = max(1, int(1.0 / (perturbation_rate * self.tau + 1e-10)))

        for step in range(n_steps):
            # Apply perturbation
            perturb = rng.randn(self.n) * amplitude
            self.state += perturb

            # Negotiate with limited rounds
            for _ in range(negotiation_rounds):
                self._negotiate_once()

            # Measure violation after negotiation
            violation = abs(np.sum(self.state) - self.conserved_total) / self.conserved_total
            self.violation_history.append(violation)

        violations = np.array(self.violation_history)
        return {
            'mean_violation': float(np.mean(violations)),
            'max_violation': float(np.max(violations)),
            'violation_above_1pct': float(np.mean(violations > 0.01)),
            'violations': violations,
        }


# ============================================================
# Annealing and Glassy Spectrum Analysis
# ============================================================
def fit_glassy_spectrum(log_residuals):
    """
    Fit log-residuals to three distributions: uniform, Gaussian, Levy-stable.
    Returns dict with AIC for each, best model name.
    """
    n = len(log_residuals)
    if n < 3:
        return {'best': 'insufficient_data', 'n': n}

    results = {}

    # Uniform fit
    loc_u = np.min(log_residuals)
    scale_u = np.max(log_residuals) - np.min(log_residuals) + 1e-10
    ll_uniform = np.sum(uniform.logpdf(log_residuals, loc=loc_u, scale=scale_u))
    aic_uniform = -2 * ll_uniform + 2 * 2  # 2 params
    results['uniform'] = {'aic': aic_uniform, 'loglik': ll_uniform}

    # Gaussian fit
    mu, sigma = np.mean(log_residuals), np.std(log_residuals)
    if sigma < 1e-10:
        sigma = 1e-10
    ll_gauss = np.sum(norm.logpdf(log_residuals, loc=mu, scale=sigma))
    aic_gauss = -2 * ll_gauss + 2 * 2  # 2 params
    results['gaussian'] = {'aic': aic_gauss, 'loglik': ll_gauss}

    # Levy-stable fit (4 params: alpha, beta, loc, scale)
    try:
        params = levy_stable.fit(log_residuals)
        ll_levy = np.sum(levy_stable.logpdf(log_residuals, *params))
        aic_levy = -2 * ll_levy + 2 * 4  # 4 params
        results['levy_stable'] = {'aic': aic_levy, 'loglik': ll_levy, 'params': params}
    except Exception:
        # Levy fit can fail on small samples
        aic_levy = np.inf
        results['levy_stable'] = {'aic': np.inf, 'loglik': -np.inf}

    # Determine best
    best = min(results.keys(), key=lambda k: results[k]['aic'])
    results['best'] = best
    results['n'] = n

    return results


# ============================================================
# Xi Universality Tools
# ============================================================
def self_referential_markov_chain(n_states, seed=42, n_steps=10000):
    """
    Build and run a self-referential Markov chain with additive+multiplicative structure.

    The transition matrix is constructed so that:
    - Additive component: uniform redistribution (gamma-like)
    - Multiplicative component: geometric scaling (phi-like)

    Returns: dict with mixing_residue (should approach Xi ≈ 1.058)
    """
    rng = np.random.RandomState(seed)

    # Build transition matrix with additive + multiplicative structure
    # Additive part: uniform
    additive = np.ones((n_states, n_states)) / n_states

    # Multiplicative part: geometric decay from diagonal
    multiplicative = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            dist = min(abs(i - j), n_states - abs(i - j))
            multiplicative[i, j] = PHI ** (-dist)
    # Normalize rows
    row_sums = multiplicative.sum(axis=1, keepdims=True)
    multiplicative = multiplicative / row_sums

    # Combined: weight additive and multiplicative equally
    T = 0.5 * additive + 0.5 * multiplicative
    # Ensure stochastic
    T = T / T.sum(axis=1, keepdims=True)

    # Run chain
    state = rng.randint(0, n_states)
    visit_counts = np.zeros(n_states)
    log_ratios = []

    for step in range(n_steps):
        visit_counts[state] += 1
        old_state = state
        state = rng.choice(n_states, p=T[state])
        # Track log transition ratio (additive accumulation vs multiplicative transition)
        if step > 0:
            empirical_freq = visit_counts / (step + 1)
            stationary_approx = empirical_freq[old_state]
            if stationary_approx > 1e-10:
                log_ratios.append(np.log(1.0 / stationary_approx))

    # Mixing residue: gap between harmonic sum behavior and log behavior
    if len(log_ratios) > 100:
        # Harmonic accumulation
        harmonic = sum(1.0 / k for k in range(1, len(log_ratios) + 1))
        log_n = np.log(len(log_ratios))
        residue = harmonic - log_n  # Should approach gamma ≈ 0.5772

        # Add multiplicative cost
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        spectral_gap = 1.0 - eigenvalues[1] if len(eigenvalues) > 1 else 1.0
        mixing_time = 1.0 / spectral_gap if spectral_gap > 0 else n_steps
        multiplicative_residue = np.log(PHI)  # ln(phi) from geometric structure

        total_residue = residue + multiplicative_residue
    else:
        total_residue = float('nan')
        residue = float('nan')
        multiplicative_residue = float('nan')

    return {
        'total_residue': float(total_residue),
        'gamma_component': float(residue),
        'lnphi_component': float(multiplicative_residue),
        'xi_target': float(XI_BALANCE),
        'relative_error': float(abs(total_residue - XI_BALANCE) / XI_BALANCE) if np.isfinite(total_residue) else float('nan'),
    }


def annealing_with_mixed_loss(n_dims=20, n_steps=5000, seed=42):
    """
    Simulated annealing with additive + multiplicative loss components.

    The loss function combines:
    - Additive: sum of |x_i - target_i| (linear)
    - Multiplicative: product of (1 + x_i^2) (geometric)

    Returns: dict with annealing_residue (should approach Xi)
    """
    rng = np.random.RandomState(seed)
    targets = rng.randn(n_dims)

    def loss(x):
        additive = np.sum(np.abs(x - targets))
        multiplicative = np.sum(np.log(1 + x**2))
        return additive + multiplicative

    # Initial state
    x = rng.randn(n_dims) * 5
    E_initial = loss(x)
    best_x = x.copy()
    best_E = E_initial

    # Annealing schedule
    T0 = 10.0
    energies = [E_initial]

    for step in range(1, n_steps + 1):
        T = T0 / (1 + step * 0.01)
        # Propose move
        proposal = x + rng.randn(n_dims) * T * 0.1
        E_new = loss(proposal)

        # Metropolis acceptance
        if E_new < best_E or rng.random() < np.exp(-(E_new - energies[-1]) / max(T, 1e-10)):
            x = proposal
            energies.append(E_new)
            if E_new < best_E:
                best_E = E_new
                best_x = x.copy()
        else:
            energies.append(energies[-1])

    E_final = best_E
    # Residue: normalized log ratio
    if E_initial > 0 and E_final > 0:
        residue = -np.log(E_final / E_initial) / np.log(n_steps)
    else:
        residue = float('nan')

    return {
        'E_initial': float(E_initial),
        'E_final': float(E_final),
        'annealing_residue': float(residue),
        'xi_target': float(XI_BALANCE),
        'relative_error': float(abs(residue - XI_BALANCE) / XI_BALANCE) if np.isfinite(residue) else float('nan'),
    }


# ============================================================
# Alternative PAC Closures (for Fossil Arithmetic)
# ============================================================
def alternative_pac_closure(closure_type, n_elements=1000):
    """
    Build an alternative PAC-closable arithmetic with a different base constant.

    closure_type: 'sqrt2', 'e', 'tribonacci', 'plastic'
    Returns: dict with prime-like decomposition and distribution.
    """
    rng = np.random.RandomState(42)

    # Base constants for different closures
    base_constants = {
        'sqrt2': np.sqrt(2),
        'e': np.e,
        'tribonacci': 1.8393,  # tribonacci constant
        'plastic': 1.3247,     # plastic number
    }
    base = base_constants.get(closure_type, np.sqrt(2))

    # Generate elements via recursive closure with the given base.
    # Use modular wrapping to keep values in a bounded range
    # (analogous to working in a finite number field).
    max_val = float(n_elements * 10)
    elements = [1]
    for i in range(1, n_elements):
        prev1 = elements[-1]
        prev2 = elements[-2] if len(elements) > 1 else 0
        new = prev1 * base + prev2
        # Wrap to keep bounded (like modular arithmetic)
        if new > max_val:
            new = (new % max_val) + 1
        elements.append(new)

    elements = np.array(elements)

    # Find "prime-like" elements: those not expressible as products of smaller elements
    primes = []
    composites = set()
    sorted_elems = np.sort(elements[1:])  # Exclude 0th

    for i, e in enumerate(sorted_elems):
        if i in composites or e < 1e-10:
            continue
        primes.append(e)
        # Mark multiples
        for j in range(i + 1, len(sorted_elems)):
            if not np.isfinite(sorted_elems[j]) or e < 1e-10:
                continue
            ratio = sorted_elems[j] / e
            if np.isfinite(ratio) and ratio < 1e6:
                r = round(ratio)
                if abs(ratio - r) < 0.01 and r > 1:
                    composites.add(j)

    primes = np.array(primes[:min(200, len(primes))])

    # Compute gap distribution
    if len(primes) > 1:
        gaps = np.diff(primes)
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
    else:
        gaps = np.array([])
        gap_mean = 0
        gap_std = 0

    # Check phi-enrichment
    phi_enrichment = 0.0
    if len(primes) > 10:
        ratios = primes[1:] / primes[:-1]
        phi_matches = np.sum(np.abs(ratios - PHI) < 0.1)
        phi_enrichment = phi_matches / len(ratios)

    return {
        'closure_type': closure_type,
        'base_constant': float(base),
        'n_primes': len(primes),
        'prime_gaps_mean': float(gap_mean),
        'prime_gaps_std': float(gap_std),
        'phi_enrichment': float(phi_enrichment),
        'gap_distribution': gaps.tolist() if len(gaps) < 100 else gaps[:100].tolist(),
    }


# ============================================================
# Matrix Two-Circle Dynamics (exp_11)
# ============================================================
def run_matrix_two_circle(N, n_steps=1000, seed=42, evolving=False):
    """
    N-dimensional two-circle mutual reference.
    x_{n+1} = tanh(W @ y_n), y_{n+1} = tanh(W^T @ x_n)

    W and W^T give transpose-symmetric coupling (mutual constraint).

    If evolving=True, W is updated each step via anti-Hebbian eigenvalue
    modulation (same mechanism as SelfApplicator): active modes are weakened,
    inactive modes are strengthened. This is self-modification of the coupling.

    Returns dict with norm trajectory, final eigenspectrum, scale ratios.
    """
    rng = np.random.RandomState(seed)

    if N == 1:
        # Scalar case: equivalent to x=tanh(w*y), y=tanh(w*x)
        w = 1.2
        x, y = rng.randn() * 0.5, rng.randn() * 0.5
        norms = [abs(x)]
        for _ in range(n_steps):
            x_new = np.tanh(w * y)
            y_new = np.tanh(w * x)
            x, y = x_new, y_new
            norms.append(abs(x))
        return {
            'norms': np.array(norms),
            'N': 1,
            'eigenvalues': np.array([w]),
        }

    W = rng.randn(N, N) / np.sqrt(N)
    # Set spectral radius to 1.2
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    if sr > 1e-10:
        W = W * (1.2 / sr)

    x = rng.randn(N) * 0.5
    y = rng.randn(N) * 0.5

    norms = [np.linalg.norm(x)]

    for step in range(n_steps):
        x_new = np.tanh(W @ y)
        y_new = np.tanh(W.T @ x)

        if evolving:
            # Anti-Hebbian eigenvalue modulation via SVD
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            activity = (x_new + y_new) / 2
            projections = (Vt @ activity) ** 2
            total = np.sum(projections) + 1e-10
            activities = projections / total
            mean_act = 1.0 / N

            modulation = np.ones(len(S))
            modulation[activities[:len(S)] > 2.0 * mean_act] = 0.95
            modulation[activities[:len(S)] < 0.5 * mean_act] = 1.01
            S_new = S * modulation

            sr = np.max(S_new)
            if sr > 1e-10:
                S_new = S_new * (1.2 / sr)
            W = U @ np.diag(S_new) @ Vt

        x, y = x_new, y_new
        norms.append(np.linalg.norm(x))

    # Final eigenspectrum
    eigvals = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]

    return {
        'norms': np.array(norms),
        'N': N,
        'eigenvalues': eigvals,
    }


# ============================================================
# Topology-Specific Coupling Matrices (exp_14–16)
# ============================================================
def build_topology_matrix(N, topology, sr=1.2):
    """
    Build symmetric nearest-neighbor coupling matrix for a given topology.

    Topologies:
    - 'line':   open boundaries (tridiagonal, no wrap)
    - 'circle': periodic boundaries (W[0,N-1] = W[N-1,0] = +c)
    - 'mobius': anti-periodic boundaries (W[0,N-1] = W[N-1,0] = -c)

    All are symmetric. Spectral radius normalized to sr.

    Key mathematical difference:
    - circle eigenvalues: 2c·cos(2πk/N), k = 0..N-1 (integer modes)
    - mobius eigenvalues: 2c·cos(π(2k+1)/N), k = 0..N-1 (half-integer modes)
    - line eigenvalues: 2c·cos(πk/(N+1)), k = 1..N (no zero mode)
    """
    W = np.zeros((N, N))

    # Nearest-neighbor coupling (tridiagonal)
    for i in range(N - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0

    # Boundary conditions
    if topology == 'circle':
        W[0, N - 1] = 1.0
        W[N - 1, 0] = 1.0
    elif topology == 'mobius':
        W[0, N - 1] = -1.0
        W[N - 1, 0] = -1.0
    elif topology == 'line':
        pass  # open boundaries
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Normalize spectral radius
    eigvals = np.linalg.eigvalsh(W)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > 1e-10:
        W = W * (sr / max_abs)

    return W


def measure_mode_structure(W):
    """
    Eigendecompose W and classify modes.

    Returns dict with:
    - eigenvalues: sorted by magnitude (descending)
    - eigenvectors: corresponding eigenvectors
    - has_zero_mode: whether any eigenvalue is near zero
    - half_integer_check: whether eigenvalues match cos(π(2k+1)/N) pattern
    - integer_check: whether eigenvalues match cos(2πk/N) pattern
    - sign_flips: count of eigenvectors with sign change across boundary
    """
    N = W.shape[0]
    eigvals, eigvecs = np.linalg.eigh(W)

    # Sort by magnitude descending
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Check for zero mode
    has_zero_mode = np.min(np.abs(eigvals)) < 1e-10

    # Normalize eigenvalues for pattern matching
    sr = np.max(np.abs(eigvals))
    if sr < 1e-10:
        return {
            'eigenvalues': eigvals, 'eigenvectors': eigvecs,
            'has_zero_mode': True, 'half_integer_check': 0.0,
            'integer_check': 0.0, 'sign_flips': 0,
        }

    normed = np.sort(eigvals / sr)

    # Expected patterns (sorted)
    half_int = np.sort(np.array(
        [np.cos(np.pi * (2 * k + 1) / N) for k in range(N)]
    ))
    integer = np.sort(np.array(
        [np.cos(2 * np.pi * k / N) for k in range(N)]
    ))

    # Normalize expected patterns to same range
    if np.max(np.abs(half_int)) > 1e-10:
        half_int = half_int / np.max(np.abs(half_int))
    if np.max(np.abs(integer)) > 1e-10:
        integer = integer / np.max(np.abs(integer))

    # Match quality (1 - mean absolute difference)
    half_integer_check = 1.0 - np.mean(np.abs(normed - half_int))
    integer_check = 1.0 - np.mean(np.abs(normed - integer))

    # Count sign-flipping eigenvectors (non-orientability signature)
    # A sign flip across boundary: eigvec[0] and eigvec[N-1] have opposite signs
    sign_flips = 0
    for j in range(N):
        v = eigvecs[:, j]
        if np.abs(v[0]) > 1e-10 and np.abs(v[N - 1]) > 1e-10:
            if v[0] * v[N - 1] < 0:
                sign_flips += 1

    return {
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'has_zero_mode': has_zero_mode,
        'half_integer_check': half_integer_check,
        'integer_check': integer_check,
        'sign_flips': sign_flips,
    }


def measure_holonomy_period(trajectory, method='autocorrelation'):
    """
    Measure the dynamical recurrence period from a trajectory.

    Uses autocorrelation: finds the first peak after the initial decay.
    On Möbius topology, the period should be ~2× that of circle topology
    (4π vs 2π closure).

    Returns dict with period, autocorrelation function, and peak info.
    """
    n_steps, n_dim = trajectory.shape

    # Normalize trajectory (zero mean, unit variance per component)
    traj = trajectory - np.mean(trajectory, axis=0)
    std = np.std(traj, axis=0)
    std[std < 1e-10] = 1.0
    traj = traj / std

    # Compute autocorrelation of the full state vector
    max_lag = min(n_steps // 2, 500)
    autocorr = np.zeros(max_lag)
    norm = np.sum(traj[0:n_steps - max_lag] ** 2)
    if norm < 1e-10:
        return {'period': 0, 'autocorr': autocorr, 'peaks': []}

    for lag in range(max_lag):
        autocorr[lag] = np.sum(
            traj[:n_steps - max_lag] * traj[lag:lag + n_steps - max_lag]
        ) / norm

    # Find peaks in autocorrelation
    peaks = []
    for i in range(2, max_lag - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.1:  # meaningful peak
                peaks.append((i, autocorr[i]))

    period = peaks[0][0] if peaks else 0

    return {
        'period': period,
        'autocorr': autocorr,
        'peaks': peaks,
    }


def measure_entropy_rate(trajectory, n_modes=4, max_block=8):
    """
    Measure the mode-sequence entropy rate h₁ from a trajectory.

    1. Project trajectory onto top n_modes eigenvectors of covariance
    2. Discretize: at each step, record which mode has largest projection
    3. Compute block entropies H(L) for L = 1..max_block
    4. h₁ = H(L) - H(L-1) for large L (conditional entropy)

    Returns dict with h₁, block entropies, mode sequence.
    """
    n_steps, n_dim = trajectory.shape

    # PCA to find dominant modes
    cov = np.cov(trajectory.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top_vecs = eigvecs[:, order[:n_modes]]

    # Project onto top modes
    projections = trajectory @ top_vecs  # (n_steps, n_modes)

    # Discretize: dominant mode at each step
    mode_sequence = np.argmax(np.abs(projections), axis=1)

    # Block entropies
    block_entropies = []
    for L in range(1, max_block + 1):
        # Count L-grams
        counts = {}
        for i in range(len(mode_sequence) - L + 1):
            block = tuple(mode_sequence[i:i + L])
            counts[block] = counts.get(block, 0) + 1

        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total
        H_L = -np.sum(probs * np.log(probs + 1e-30))
        block_entropies.append(H_L)

    # Entropy rate: H(L) - H(L-1) for last few L values
    if len(block_entropies) >= 2:
        # Average over last 3 conditional entropies for stability
        conditionals = [
            block_entropies[i] - block_entropies[i - 1]
            for i in range(max(1, len(block_entropies) - 3), len(block_entropies))
        ]
        h1 = np.mean(conditionals)
    else:
        h1 = block_entropies[0] if block_entropies else 0.0

    return {
        'h1': h1,
        'block_entropies': block_entropies,
        'mode_sequence': mode_sequence,
        'n_modes_used': n_modes,
    }
