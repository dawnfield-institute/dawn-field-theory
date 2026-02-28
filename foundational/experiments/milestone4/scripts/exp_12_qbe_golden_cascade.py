#!/usr/bin/env python3
"""
Experiment 12: QBE-Constrained Dynamics -> Golden Cascade Potential
====================================================================

The DERIVATION GAP: The herniation simulations (sims 1-6) prescribed the
golden cascade potential V(x) = -g * Sum phi^(-n) * Gauss(x; w0*phi^n).
The legacy GPU sims (brain.py, cosmo.py, vcpu.py) showed QPL reinforcement
self-organizing. But nobody proved the QPL landscape converges to phi-scaling.

The QUANTUM BALANCE EQUATION (QBE) bridges them:

  dI/dt + dE/dt = lambda * QPL(t)

With QPL carrying Fibonacci harmonics:

  QPL(t) = cos(wt) + (1/phi)*cos(phi*w*t) + (1/phi^2)*cos(phi^2*w*t)

This experiment tests: when we run 1D field dynamics CONSTRAINED by the QBE,
does the emergent reinforcement landscape converge to the golden cascade?

  Part A: QPL EMERGENCE FROM QBE-CONSTRAINED DYNAMICS
    --------------------------------------------------
    Two coupled fields (I, E) on a 1D lattice with:
    - Diffusion (order slow, chaos fast — from herniation_sim.py)
    - Threshold collapse (from legacy: I>0.4, E>0.05)
    - QPL reinforcement at collapse sites (QPL *= 1.05)
    - QPL damping (0.02, from legacy)
    - QBE constraint enforced: dI + dE regulated by QPL
    Measure: QPL(x) spatial profile at steady state.

  Part B: GOLDEN CASCADE CORRELATION
    ---------------------------------
    Compare the emergent QPL(x) profile to analytic golden cascades:
      V(x) = -g * Sum_{n=0}^{N-1} phi^(-n) * exp(-x^2 / (2*(w0*phi^n)^2))
    Vary g, w0, N to find best match. Measure correlation coefficient.
    
  Part C: SPECTRAL COMPARISON
    --------------------------
    Solve Schrodinger with the EMERGENT potential (QPL landscape).
    Compare mass ratios to those from the PRESCRIBED golden cascade.
    If they match, the QBE closes the derivation chain.
    
  Part D: FIBONACCI HARMONIC FINGERPRINT
    --------------------------------------
    Test whether the QPL temporal spectrum is key by comparing:
    1. QBE with Fibonacci QPL -> emergent landscape
    2. QBE with sinusoidal QPL (no phi ratios) -> emergent landscape
    3. QBE with random QPL -> emergent landscape
    Only Fibonacci should produce golden cascade. This is the null test.

Dawn Field Institute, 2026-02-25
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, minimize
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE
from core.utils import save_results, timer

# ============================================================
# Target mass ratios from exp_07-09
# ============================================================
LEPTON_RATIOS = np.array([1.0, 206.768, 3477.2])   # e : mu : tau
DOWN_QUARK_RATIOS = np.array([1.0, 20.0, 893.6])    # d : s : b

print("=" * 80)
print("EXPERIMENT 12: QBE-Constrained Dynamics -> Golden Cascade Potential")
print("=" * 80)
print(f"  phi = {PHI:.10f}")
print(f"  1/phi = {INV_PHI:.10f}")
print(f"  ln(phi) = {LN_PHI:.10f}")
print(f"  Xi = {XI_BALANCE:.10f}")
sys.stdout.flush()


# ============================================================
# PART A: QBE-Constrained 1D Field Dynamics
# ============================================================

def qpl_fibonacci(t, omega=0.020):
    """
    QPL(t) with Fibonacci harmonics — the form from the vCPU architecture.
    QPL(t) = cos(wt) + (1/phi)*cos(phi*w*t) + (1/phi^2)*cos(phi^2*w*t)
    
    Three harmonics at golden-ratio frequency ratios create a 
    quasi-periodic signal that never exactly repeats.
    """
    return (np.cos(omega * t) + 
            INV_PHI * np.cos(PHI * omega * t) + 
            INV_PHI**2 * np.cos(PHI**2 * omega * t))


def qpl_sinusoidal(t, omega=0.020):
    """Control: simple sinusoidal QPL (no phi structure)."""
    return (np.cos(omega * t) + 
            0.5 * np.cos(2.0 * omega * t) + 
            0.25 * np.cos(3.0 * omega * t))


def qpl_random(t, omega=0.020, seed=42):
    """Control: random quasi-periodic QPL."""
    rng = np.random.default_rng(seed)
    r1, r2 = rng.uniform(1.2, 2.5, 2)
    return (np.cos(omega * t) + 
            0.5 * np.cos(r1 * omega * t) + 
            0.25 * np.cos(r2 * omega * t))


def run_qbe_dynamics(
    N=512,                # lattice sites
    T=50000,              # time steps 
    dt=0.01,              # time step size
    dx=0.5,               # spatial step
    D_order=0.1,          # order field diffusion (slow — from herniation_sim.py)
    D_chaos=0.5,          # chaos field diffusion (fast — from herniation_sim.py)
    collapse_threshold_I=0.4,   # from legacy cosmo.py
    collapse_threshold_E=0.05,  # from legacy cosmo.py
    qpl_reinforce=1.05,         # from legacy brain.py QPL *= 1.05
    qpl_damping=0.02,           # from legacy QPL_damping = 0.02
    lambda_qbe=0.020,           # QBE coupling constant (matches 0.020 Hz)
    qpl_func=None,              # QPL(t) function
    boundary_width=20,          # width of interaction zone
    seed=42,
    label="fibonacci"
):
    """
    1D field dynamics constrained by the Quantum Balance Equation.
    
    Two fields: I(x,t) [information/order] and E(x,t) [energy/chaos]
    QPL(x) evolves via reinforcement at collapse sites.
    QBE constrains: at each point, dI/dt + dE/dt ~ lambda * QPL_temporal(t)
    
    Returns the steady-state QPL(x) spatial profile.
    """
    if qpl_func is None:
        qpl_func = qpl_fibonacci
    
    rng = np.random.default_rng(seed)
    
    x = np.arange(N) * dx
    center = N // 2
    x_centered = x - x[center]
    
    # Initialize fields — spatially heterogeneous like legacy SHA-256 seeding
    # Order field: Gaussian envelope * random fluctuations (not smooth!)
    # This creates random patches that cross threshold at different times
    envelope = np.exp(-x_centered**2 / (2 * (boundary_width * 2)**2))
    I_field = 0.35 * envelope + 0.15 * rng.standard_normal(N) * envelope
    I_field = np.clip(I_field, 0.0, 1.0)
    
    # Energy field: uniform background + random fluctuations
    # (from legacy: energy starts random via SHA-256 seeding)
    E_field = 0.15 + 0.1 * rng.standard_normal(N)
    E_field = np.clip(E_field, 0.01, 1.0)
    
    # QPL field: starts uniform at 1.0 (no memory yet)
    # In legacy: QPL starts at 1.0, only grows at collapse sites, capped at 2.0
    QPL_spatial = np.ones(N, dtype=np.float64)
    
    # Boundary mask (from herniation_sim.py) — where fields can interact
    boundary_mask = np.exp(-x_centered**2 / (2 * boundary_width**2))
    
    # Track collapse counts per site
    collapse_counts = np.zeros(N, dtype=np.int64)
    
    # Track I+E conservation for QBE check
    ie_history = []
    qpl_history = []
    qpl_snapshots = []
    snapshot_times = [T//10, T//5, T//3, T//2, 2*T//3, T-1]
    
    # Stability: CFL condition
    max_D = max(D_order, D_chaos)
    cfl = max_D * dt / dx**2
    if cfl > 0.4:
        print(f"  WARNING: CFL = {cfl:.3f} > 0.4, reducing dt")
        dt = 0.4 * dx**2 / max_D
        cfl = max_D * dt / dx**2
        print(f"  New dt = {dt:.6f}, CFL = {cfl:.3f}")
    
    for t_step in range(T):
        t = t_step * dt
        
        # Laplacians (reflecting boundary conditions)
        I_padded = np.pad(I_field, 1, mode='reflect')
        E_padded = np.pad(E_field, 1, mode='reflect')
        lap_I = (I_padded[2:] + I_padded[:-2] - 2 * I_padded[1:-1]) / dx**2
        lap_E = (E_padded[2:] + E_padded[:-2] - 2 * E_padded[1:-1]) / dx**2
        
        # Self-interaction terms (from herniation_sim.py)
        # Order field: tends toward structure (Ginzburg-Landau-like)
        self_I = I_field - I_field**3
        # Chaos field: dispersive  
        self_E = -0.1 * E_field**3
        
        # Coupling at boundary
        coupling_term = boundary_mask * I_field * E_field
        
        # QPL temporal value — the QBE driving term
        qpl_t = qpl_func(t)
        
        # === QBE-CONSTRAINED FIELD EVOLUTION ===
        # From legacy: QPL damps the I field, not itself
        # val_info -= QPL[x,y,z] * QPL_damping  (brain.py line 37)
        # This means QPL acts as a POTENTIAL that suppresses info growth
        # The QBE constraint: dI/dt + dE/dt = lambda * QPL_temporal(t)
        # QPL_spatial modulates WHERE dynamics are strongest
        
        # Info growth rate modulated by temporal QPL and spatial QPL
        # Higher QPL_spatial = stronger attractor = more info exchange allowed
        info_growth = 0.05 * (E_field / (E_field.max() + 1e-10))  # from legacy info_growth_rate
        
        # QBE: the total dI+dE budget at each point
        qbe_local = lambda_qbe * qpl_t * QPL_spatial
        
        # Compute field updates
        dI = dt * (D_order * lap_I + 0.1 * self_I + 0.5 * coupling_term + info_growth)
        dE = dt * (D_chaos * lap_E + 0.1 * self_E - 0.3 * coupling_term)
        
        # QPL damping on info field (from legacy: val_info -= QPL * QPL_damping)
        dI -= dt * QPL_spatial * qpl_damping
        
        # QBE soft constraint: scale total (dI+dE) toward QBE budget
        dI_plus_dE = dI + dE
        scale = np.ones(N)
        nonzero = np.abs(dI_plus_dE) > 1e-12
        scale[nonzero] = (dt * qbe_local[nonzero]) / dI_plus_dE[nonzero]
        # Soft blend: 30% QBE-constrained, 70% free dynamics
        alpha_qbe = 0.3
        effective_scale = 1.0 + alpha_qbe * (scale - 1.0)
        effective_scale = np.clip(effective_scale, 0.1, 5.0)
        
        I_field += dI * effective_scale
        E_field += dE * effective_scale
        
        # Clamp fields
        I_field = np.clip(I_field, 0.0, 2.0)
        E_field = np.clip(E_field, 0.01, 2.0)
        
        # === COLLAPSE / CRYSTALLIZATION (from legacy) ===
        collapse_sites = (I_field > collapse_threshold_I) & (E_field > collapse_threshold_E)
        
        if np.any(collapse_sites):
            # QPL reinforcement at collapse sites (from brain.py: *= 1.05)
            QPL_spatial[collapse_sites] *= qpl_reinforce
            
            # Energy consumed in collapse (from legacy: energy *= 0.9)
            E_field[collapse_sites] *= 0.9
            
            # Information partially consumed
            I_field[collapse_sites] *= 0.95
            
            collapse_counts[collapse_sites] += 1
        
        # QPL does NOT decay — in legacy, QPL only grows at collapse sites
        # The 0.02 damping applies to the I field (already done above)
        # QPL max clamping (from legacy: min 1.0, max 2.0)
        QPL_spatial = np.clip(QPL_spatial, 1.0, 5.0)
        
        # Record
        if t_step % 100 == 0:
            ie_history.append(float(np.sum(I_field + E_field)))
            qpl_history.append(float(np.mean(QPL_spatial)))
        
        if t_step in snapshot_times:
            qpl_snapshots.append(QPL_spatial.copy())
    
    return {
        'QPL_final': QPL_spatial,
        'QPL_snapshots': qpl_snapshots,
        'I_field': I_field,
        'E_field': E_field,
        'collapse_counts': collapse_counts,
        'ie_history': ie_history,
        'qpl_history': qpl_history,
        'x': x_centered,
        'boundary_mask': boundary_mask,
        'params': {
            'N': N, 'T': T, 'dt': dt, 'dx': dx,
            'D_order': D_order, 'D_chaos': D_chaos,
            'lambda_qbe': lambda_qbe, 'label': label,
            'boundary_width': boundary_width
        }
    }


# ============================================================
# PART B: Golden Cascade Fit
# ============================================================

def golden_cascade(x, g, w0, n_levels):
    """Build the analytic golden cascade potential."""
    V = np.zeros_like(x)
    for n in range(n_levels):
        V += g * INV_PHI**n * np.exp(-x**2 / (2 * (w0 * PHI**n)**2))
    return V


def fit_golden_cascade(x, qpl_profile, n_levels_range=(3, 20)):
    """
    Fit a golden cascade to the emergent QPL profile.
    Returns best-fit parameters and correlation.
    """
    # Normalize QPL profile to [0, 1] for comparison
    qpl_norm = qpl_profile - qpl_profile.min()
    if qpl_norm.max() > 0:
        qpl_norm = qpl_norm / qpl_norm.max()
    
    best_corr = -1
    best_params = None
    
    for n_lev in range(n_levels_range[0], n_levels_range[1] + 1):
        def neg_corr(params):
            g, w0 = abs(params[0]) + 0.01, abs(params[1]) + 0.1
            cascade = golden_cascade(x, g, w0, n_lev)
            c_norm = cascade - cascade.min()
            if c_norm.max() > 0:
                c_norm = c_norm / c_norm.max()
            corr = np.corrcoef(qpl_norm, c_norm)[0, 1]
            return -corr if not np.isnan(corr) else 0
        
        # Multi-start
        best_trial = 1e10
        best_trial_params = None
        for g_init in [0.5, 1.0, 2.0, 5.0]:
            for w0_init in [0.5, 1.0, 2.0, 5.0, 10.0]:
                try:
                    res = minimize(neg_corr, [g_init, w0_init], 
                                   method='Nelder-Mead',
                                   options={'maxiter': 200})
                    if res.fun < best_trial:
                        best_trial = res.fun
                        best_trial_params = res.x
                except:
                    continue
        
        if best_trial_params is not None:
            corr = -best_trial
            if corr > best_corr:
                best_corr = corr
                best_params = {
                    'g': abs(best_trial_params[0]) + 0.01,
                    'w0': abs(best_trial_params[1]) + 0.1,
                    'n_levels': n_lev,
                    'correlation': corr
                }
    
    return best_params


def fit_generic_cascade(x, qpl_profile, base):
    """
    Fit a cascade with arbitrary base (not phi).
    V(x) = g * Sum base^(-n) * Gauss(x; w0 * base^n)
    """
    qpl_norm = qpl_profile - qpl_profile.min()
    if qpl_norm.max() > 0:
        qpl_norm = qpl_norm / qpl_norm.max()
    
    best_corr = -1
    best_params = None
    
    for n_lev in range(3, 16):
        def neg_corr(params):
            g, w0 = abs(params[0]) + 0.01, abs(params[1]) + 0.1
            V = np.zeros_like(x)
            for n in range(n_lev):
                V += g * base**(-n) * np.exp(-x**2 / (2 * (w0 * base**n)**2))
            c_norm = V - V.min()
            if c_norm.max() > 0:
                c_norm = c_norm / c_norm.max()
            corr = np.corrcoef(qpl_norm, c_norm)[0, 1]
            return -corr if not np.isnan(corr) else 0
        
        for g_init in [0.5, 2.0, 5.0]:
            for w0_init in [0.5, 2.0, 10.0]:
                try:
                    res = minimize(neg_corr, [g_init, w0_init],
                                   method='Nelder-Mead', options={'maxiter': 200})
                    corr = -res.fun
                    if corr > best_corr:
                        best_corr = corr
                        best_params = {
                            'base': base,
                            'g': abs(res.x[0]) + 0.01,
                            'w0': abs(res.x[1]) + 0.1,
                            'n_levels': n_lev,
                            'correlation': corr
                        }
                except:
                    continue
    
    return best_params


# ============================================================
# PART C: Spectral Comparison
# ============================================================

def solve_spectrum(V, x):
    """Solve 1D Schrodinger equation, return bound state energies."""
    N = len(x)
    dx = x[1] - x[0]
    T_diag = -0.5 / dx**2 * np.full(N, -2.0)
    T_off = -0.5 / dx**2 * np.ones(N - 1)
    H = np.diag(T_diag + V) + np.diag(T_off, 1) + np.diag(T_off, -1)
    eigenvalues, _ = eigh(H)
    return eigenvalues[eigenvalues < 0]


def score_mass_ratios(energies, target_ratios):
    """Score how well a triplet of bound state energies matches target ratios."""
    if len(energies) < 3:
        return 1e6, None
    
    masses = np.abs(energies)
    n = len(masses)
    
    best_score = 1e6
    best_triplet = None
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            k = n - 1  # always use shallowest
            sel = masses[[i, j, k]]
            ratios = np.sort(sel / sel.min())
            # Log-space distance
            log_err = np.sum((np.log(ratios) - np.log(target_ratios))**2)
            if log_err < best_score:
                best_score = log_err
                best_triplet = (i, j, k, ratios)
    
    return best_score, best_triplet


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    all_results = {}
    
    # ================================================================
    # PART A: Run QBE-constrained dynamics with Fibonacci QPL
    # ================================================================
    print("\n" + "=" * 80)
    print("PART A: QBE-CONSTRAINED DYNAMICS WITH FIBONACCI QPL")
    print("=" * 80)
    sys.stdout.flush()
    
    with timer() as t_a:
        result_fib = run_qbe_dynamics(
            N=512, T=50000, dt=0.01, dx=0.5,
            qpl_func=qpl_fibonacci,
            lambda_qbe=0.020,
            label="fibonacci"
        )
    
    qpl_fib = result_fib['QPL_final']
    x = result_fib['x']
    collapse_fib = result_fib['collapse_counts']
    
    print(f"\n  Runtime: {t_a.elapsed:.1f}s")
    print(f"  QPL range: [{qpl_fib.min():.4f}, {qpl_fib.max():.4f}]")
    print(f"  QPL center value: {qpl_fib[len(qpl_fib)//2]:.4f}")
    print(f"  Total collapses: {collapse_fib.sum()}")
    print(f"  Collapse sites: {(collapse_fib > 0).sum()} / {len(collapse_fib)}")
    print(f"  Max collapses at single site: {collapse_fib.max()}")
    
    # QPL profile shape
    peak_idx = np.argmax(qpl_fib)
    print(f"  QPL peak position: x = {x[peak_idx]:.2f}")
    
    # Check if QPL has structure (not flat)
    qpl_contrast = (qpl_fib.max() - qpl_fib.min()) / (qpl_fib.mean() + 1e-10)
    print(f"  QPL contrast: {qpl_contrast:.4f}")
    has_structure = qpl_contrast > 0.1
    print(f"  Has spatial structure: {has_structure}")
    
    all_results['part_a_fibonacci'] = {
        'qpl_range': [float(qpl_fib.min()), float(qpl_fib.max())],
        'qpl_center': float(qpl_fib[len(qpl_fib)//2]),
        'total_collapses': int(collapse_fib.sum()),
        'collapse_sites': int((collapse_fib > 0).sum()),
        'qpl_contrast': float(qpl_contrast),
        'has_structure': has_structure,
        'runtime': t_a.elapsed
    }
    sys.stdout.flush()
    
    # ================================================================
    # PART B: Fit Golden Cascade to Emergent QPL
    # ================================================================
    print("\n" + "=" * 80)
    print("PART B: GOLDEN CASCADE FIT TO EMERGENT QPL PROFILE")
    print("=" * 80)
    sys.stdout.flush()
    
    with timer() as t_b:
        # Fit golden cascade (phi-based)
        print("\n  Fitting phi-cascade to emergent QPL...")
        fit_phi = fit_golden_cascade(x, qpl_fib, n_levels_range=(3, 15))
        
        if fit_phi:
            print(f"  GOLDEN CASCADE FIT:")
            print(f"    g = {fit_phi['g']:.4f}")
            print(f"    w0 = {fit_phi['w0']:.4f}")
            print(f"    n_levels = {fit_phi['n_levels']}")
            print(f"    correlation = {fit_phi['correlation']:.6f}")
        else:
            print("  WARNING: Golden cascade fit failed")
        
        # Compare to other bases (null test)
        print("\n  Comparing alternative bases...")
        bases = {
            'phi': PHI,
            'sqrt2': np.sqrt(2),
            'e': np.e,
            '2.0': 2.0,
            '3.0': 3.0,
            'pi': np.pi,
        }
        
        base_results = {}
        for name, base in bases.items():
            fit = fit_generic_cascade(x, qpl_fib, base)
            if fit:
                print(f"    base={name} ({base:.4f}): corr={fit['correlation']:.6f}, "
                      f"n={fit['n_levels']}, g={fit['g']:.3f}, w0={fit['w0']:.3f}")
                base_results[name] = fit
        
        # Ranking
        if base_results:
            ranked = sorted(base_results.items(), key=lambda x: x[1]['correlation'], reverse=True)
            print(f"\n  RANKING:")
            for i, (name, fit) in enumerate(ranked):
                marker = " <-- GOLDEN" if name == 'phi' else ""
                print(f"    {i+1}. base={name}: corr={fit['correlation']:.6f}{marker}")
            
            phi_rank = [i+1 for i, (n, _) in enumerate(ranked) if n == 'phi'][0]
            print(f"\n  PHI RANK: {phi_rank} / {len(ranked)}")
    
    print(f"\n  Part B runtime: {t_b.elapsed:.1f}s")
    
    all_results['part_b_fit'] = {
        'golden_cascade': fit_phi,
        'base_comparison': {k: v for k, v in base_results.items()},
        'phi_rank': phi_rank if base_results else None,
        'runtime': t_b.elapsed
    }
    sys.stdout.flush()
    
    # ================================================================
    # PART C: Spectral Comparison — Emergent vs Prescribed
    # ================================================================
    print("\n" + "=" * 80)
    print("PART C: SPECTRAL COMPARISON (EMERGENT vs PRESCRIBED)")
    print("=" * 80)
    sys.stdout.flush()
    
    with timer() as t_c:
        # Build potential from emergent QPL
        # QPL landscape acts as potential well: V(x) = -k * (QPL(x) - QPL_min)
        # Deeper QPL = deeper well = stronger binding
        qpl_shifted = qpl_fib - qpl_fib.min()
        V_emergent = -qpl_shifted / (qpl_shifted.max() + 1e-10) * 5.0  # Normalize depth to ~5
        
        bound_emergent = solve_spectrum(V_emergent, x)
        print(f"\n  Emergent potential:")
        print(f"    Bound states: {len(bound_emergent)}")
        if len(bound_emergent) > 0:
            print(f"    Ground state: {bound_emergent[0]:.6f}")
            print(f"    First 5 energies: {[f'{e:.4f}' for e in bound_emergent[:5]]}")
        
        # Build prescribed golden cascade with best-fit params
        if fit_phi:
            V_prescribed = -golden_cascade(x, fit_phi['g'], fit_phi['w0'], fit_phi['n_levels'])
            # Scale to same depth as emergent
            V_prescribed = V_prescribed / (abs(V_prescribed.min()) + 1e-10) * abs(V_emergent.min())
            
            bound_prescribed = solve_spectrum(V_prescribed, x)
            print(f"\n  Prescribed golden cascade (best-fit params):")
            print(f"    Bound states: {len(bound_prescribed)}")
            if len(bound_prescribed) > 0:
                print(f"    Ground state: {bound_prescribed[0]:.6f}")
                print(f"    First 5 energies: {[f'{e:.4f}' for e in bound_prescribed[:5]]}")
        
        # Score both against lepton ratios
        if len(bound_emergent) >= 3:
            score_em, trip_em = score_mass_ratios(bound_emergent, LEPTON_RATIOS)
            print(f"\n  Emergent lepton match:")
            print(f"    Score (log-space dist): {score_em:.4f}")
            if trip_em:
                i, j, k, ratios = trip_em
                print(f"    Best triplet: ({i}, {j}, {k})")
                print(f"    Found ratios: {[f'{r:.1f}' for r in ratios]}")
                print(f"    Target:       {[f'{r:.1f}' for r in LEPTON_RATIOS]}")
        
        if fit_phi and len(bound_prescribed) >= 3:
            score_pr, trip_pr = score_mass_ratios(bound_prescribed, LEPTON_RATIOS)
            print(f"\n  Prescribed lepton match:")
            print(f"    Score (log-space dist): {score_pr:.4f}")
            if trip_pr:
                i, j, k, ratios = trip_pr
                print(f"    Best triplet: ({i}, {j}, {k})")
                print(f"    Found ratios: {[f'{r:.1f}' for r in ratios]}")
        
        # Compare spectra directly
        if fit_phi and len(bound_emergent) >= 3 and len(bound_prescribed) >= 3:
            n_common = min(len(bound_emergent), len(bound_prescribed))
            # Normalize both spectra by ground state
            em_normed = np.abs(bound_emergent[:n_common]) / np.abs(bound_emergent[0])
            pr_normed = np.abs(bound_prescribed[:n_common]) / np.abs(bound_prescribed[0])
            
            spectral_corr = np.corrcoef(em_normed, pr_normed)[0, 1]
            log_diff = np.mean(np.abs(np.log(em_normed + 1e-10) - np.log(pr_normed + 1e-10)))
            
            print(f"\n  SPECTRAL COMPARISON ({n_common} levels):")
            print(f"    Correlation (normalized energies): {spectral_corr:.6f}")
            print(f"    Mean log difference: {log_diff:.6f}")
            print(f"    Spectra match: {spectral_corr > 0.95 and log_diff < 0.3}")
        
    print(f"\n  Part C runtime: {t_c.elapsed:.1f}s")
    
    all_results['part_c_spectra'] = {
        'emergent_n_bound': len(bound_emergent),
        'emergent_energies_first5': bound_emergent[:5].tolist() if len(bound_emergent) >= 5 else bound_emergent.tolist(),
        'prescribed_n_bound': len(bound_prescribed) if fit_phi else 0,
        'spectral_correlation': float(spectral_corr) if fit_phi and 'spectral_corr' in dir() else None,
        'runtime': t_c.elapsed
    }
    sys.stdout.flush()
    
    # ================================================================
    # PART D: NULL TEST — Fibonacci vs Sinusoidal vs Random QPL
    # ================================================================
    print("\n" + "=" * 80)
    print("PART D: NULL TEST — FIBONACCI vs SINUSOIDAL vs RANDOM QPL")  
    print("=" * 80)
    print("\n  If QBE + Fibonacci harmonics produces golden cascade,")
    print("  then QBE + non-Fibonacci should NOT. This is the falsification test.")
    sys.stdout.flush()
    
    with timer() as t_d:
        qpl_functions = {
            'fibonacci': qpl_fibonacci,
            'sinusoidal': qpl_sinusoidal,
            'random': qpl_random,
        }
        
        null_results = {}
        
        for name, func in qpl_functions.items():
            print(f"\n  --- {name.upper()} QPL ---")
            sys.stdout.flush()
            
            result = run_qbe_dynamics(
                N=512, T=50000, dt=0.01, dx=0.5,
                qpl_func=func,
                lambda_qbe=0.020,
                label=name
            )
            
            qpl_profile = result['QPL_final']
            collapse = result['collapse_counts']
            
            contrast = (qpl_profile.max() - qpl_profile.min()) / (qpl_profile.mean() + 1e-10)
            print(f"    QPL contrast: {contrast:.4f}")
            print(f"    Total collapses: {collapse.sum()}")
            
            # Fit golden cascade
            fit = fit_golden_cascade(x, qpl_profile, n_levels_range=(3, 12))
            if fit:
                print(f"    Golden cascade corr: {fit['correlation']:.6f}")
            
            # Also fit with simple Gaussian (1-level)
            def neg_gauss_corr(params):
                w = abs(params[0]) + 0.1
                gauss = np.exp(-x**2 / (2 * w**2))
                g_norm = gauss / gauss.max()
                q_norm = qpl_profile - qpl_profile.min()
                if q_norm.max() > 0:
                    q_norm = q_norm / q_norm.max()
                c = np.corrcoef(g_norm, q_norm)[0, 1]
                return -c if not np.isnan(c) else 0
            
            res_gauss = minimize(neg_gauss_corr, [10.0], method='Nelder-Mead')
            gauss_corr = -res_gauss.fun
            print(f"    Simple Gaussian corr: {gauss_corr:.6f}")
            
            # Does cascade improve over Gaussian?
            if fit:
                improvement = fit['correlation'] - gauss_corr
                print(f"    Cascade improvement over Gaussian: {improvement:.6f}")
            
            null_results[name] = {
                'qpl_contrast': float(contrast),
                'total_collapses': int(collapse.sum()),
                'golden_cascade_corr': fit['correlation'] if fit else None,
                'gaussian_corr': float(gauss_corr),
                'cascade_params': fit,
            }
        
        # Analysis: does only Fibonacci produce golden cascade?
        print(f"\n  NULL TEST COMPARISON:")
        print(f"  {'QPL Type':<15} {'Cascade Corr':>15} {'Gaussian Corr':>15} {'Improvement':>15}")
        print(f"  {'-'*60}")
        
        for name, res in null_results.items():
            cc = res['golden_cascade_corr']
            gc = res['gaussian_corr']
            imp = cc - gc if cc is not None else None
            print(f"  {name:<15} {cc:>15.6f} {gc:>15.6f} {imp:>15.6f}" if imp is not None 
                  else f"  {name:<15} {'FAILED':>15} {gc:>15.6f}")
        
        # Key test: Fibonacci should have highest cascade correlation
        fib_corr = null_results['fibonacci']['golden_cascade_corr']
        sin_corr = null_results['sinusoidal']['golden_cascade_corr']
        rand_corr = null_results['random']['golden_cascade_corr']
        
        if fib_corr is not None:
            fib_best = True
            if sin_corr is not None and sin_corr >= fib_corr:
                fib_best = False
            if rand_corr is not None and rand_corr >= fib_corr:
                fib_best = False
            
            print(f"\n  Fibonacci produces best golden cascade fit: {fib_best}")
            print(f"  Fibonacci advantage over sinusoidal: {(fib_corr - (sin_corr or 0)):.6f}")
            print(f"  Fibonacci advantage over random: {(fib_corr - (rand_corr or 0)):.6f}")
    
    print(f"\n  Part D runtime: {t_d.elapsed:.1f}s")
    
    all_results['part_d_null_test'] = {
        'results': null_results,
        'fibonacci_best': fib_best if 'fib_best' in dir() else None,
        'runtime': t_d.elapsed
    }
    sys.stdout.flush()
    
    # ================================================================
    # PART E: PARAMETER SENSITIVITY — Does the QPL Attractor Hold?
    # ================================================================
    print("\n" + "=" * 80)
    print("PART E: PARAMETER SENSITIVITY")
    print("=" * 80)
    print("\n  Test whether the golden cascade emergence is robust across:")
    print("  - Different QBE coupling strengths (lambda)")
    print("  - Different boundary widths")
    print("  - Different initial conditions (seeds)")
    sys.stdout.flush()
    
    with timer() as t_e:
        sensitivity_results = {}
        
        # Lambda sweep
        print("\n  Lambda sweep:")
        for lam in [0.005, 0.010, 0.020, 0.040, 0.080]:
            result = run_qbe_dynamics(
                N=256, T=30000, dt=0.01, dx=0.5,
                qpl_func=qpl_fibonacci,
                lambda_qbe=lam,
                label=f"lambda_{lam}"
            )
            fit = fit_golden_cascade(result['x'], result['QPL_final'], n_levels_range=(3, 10))
            corr = fit['correlation'] if fit else None
            contrast = (result['QPL_final'].max() - result['QPL_final'].min()) / (result['QPL_final'].mean() + 1e-10)
            corr_str = f"{corr:.6f}" if corr is not None else "NONE"
            print(f"    lambda={lam:.3f}: cascade_corr={corr_str}, contrast={contrast:.4f}")
            sensitivity_results[f'lambda_{lam}'] = {
                'lambda': lam, 'cascade_corr': corr, 'contrast': float(contrast)
            }
        
        # Boundary width sweep
        print("\n  Boundary width sweep:")
        for bw in [5, 10, 20, 40, 80]:
            result = run_qbe_dynamics(
                N=256, T=30000, dt=0.01, dx=0.5,
                qpl_func=qpl_fibonacci,
                lambda_qbe=0.020,
                boundary_width=bw,
                label=f"bw_{bw}"
            )
            fit = fit_golden_cascade(result['x'], result['QPL_final'], n_levels_range=(3, 10))
            corr = fit['correlation'] if fit else None
            contrast = (result['QPL_final'].max() - result['QPL_final'].min()) / (result['QPL_final'].mean() + 1e-10)
            corr_str = f"{corr:.6f}" if corr is not None else "NONE"
            print(f"    bw={bw:3d}: cascade_corr={corr_str}, contrast={contrast:.4f}")
            sensitivity_results[f'bw_{bw}'] = {
                'boundary_width': bw, 'cascade_corr': corr, 'contrast': float(contrast)
            }
        
        # Seed sweep (reproducibility)
        print("\n  Seed sweep:")
        seed_corrs = []
        for seed in [1, 42, 137, 256, 999]:
            result = run_qbe_dynamics(
                N=256, T=30000, dt=0.01, dx=0.5,
                qpl_func=qpl_fibonacci,
                lambda_qbe=0.020,
                seed=seed,
                label=f"seed_{seed}"
            )
            fit = fit_golden_cascade(result['x'], result['QPL_final'], n_levels_range=(3, 10))
            corr = fit['correlation'] if fit else None
            if corr is not None:
                seed_corrs.append(corr)
            corr_str = f"{corr:.6f}" if corr is not None else "NONE"
            print(f"    seed={seed:4d}: cascade_corr={corr_str}")
            sensitivity_results[f'seed_{seed}'] = {
                'seed': seed, 'cascade_corr': corr
            }
        
        if seed_corrs:
            print(f"\n  Seed stats: mean={np.mean(seed_corrs):.6f}, "
                  f"std={np.std(seed_corrs):.6f}, "
                  f"min={min(seed_corrs):.6f}, max={max(seed_corrs):.6f}")
    
    print(f"\n  Part E runtime: {t_e.elapsed:.1f}s")
    
    all_results['part_e_sensitivity'] = sensitivity_results
    sys.stdout.flush()
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"""
  EXPERIMENT 12: QBE-Constrained Dynamics -> Golden Cascade Potential
  
  HYPOTHESIS: The QBE (dI/dt + dE/dt = lambda * QPL(t)) with Fibonacci
  harmonics in QPL produces an emergent spatial QPL landscape that
  converges to the golden cascade potential V(x) ~ Sum phi^(-n) * Gauss.
  
  PART A: QBE Dynamics
    QPL contrast (structure): {all_results['part_a_fibonacci']['qpl_contrast']:.4f}
    Has spatial structure: {all_results['part_a_fibonacci']['has_structure']}
    Total collapses: {all_results['part_a_fibonacci']['total_collapses']}
  
  PART B: Golden Cascade Fit""")
    
    if fit_phi:
        print(f"    Best fit: g={fit_phi['g']:.4f}, w0={fit_phi['w0']:.4f}, n={fit_phi['n_levels']}")
        print(f"    Correlation: {fit_phi['correlation']:.6f}")
        if base_results:
            print(f"    Phi rank among {len(base_results)} bases: #{phi_rank}")
    
    print(f"""
  PART C: Spectral Comparison
    Emergent bound states: {all_results['part_c_spectra']['emergent_n_bound']}
    Prescribed bound states: {all_results['part_c_spectra']['prescribed_n_bound']}""")
    
    if all_results['part_c_spectra'].get('spectral_correlation') is not None:
        print(f"    Spectral correlation: {all_results['part_c_spectra']['spectral_correlation']:.6f}")
    
    print(f"""
  PART D: Null Test (Fibonacci vs alternatives)""")
    if 'fib_best' in dir():
        print(f"    Fibonacci produces best golden cascade: {fib_best}")
    
    print(f"""
  PART E: Parameter Sensitivity""")
    if seed_corrs:
        print(f"    Across seeds: {np.mean(seed_corrs):.6f} +/- {np.std(seed_corrs):.6f}")
    
    # Overall verdict
    print(f"\n  {'='*60}")
    
    # Assess whether QBE closes the gap
    gap_closed = (
        all_results['part_a_fibonacci']['has_structure'] and
        fit_phi is not None and
        fit_phi['correlation'] > 0.9
    )
    
    print(f"  QBE CLOSES DERIVATION GAP: {'YES' if gap_closed else 'NOT YET'}")
    
    if gap_closed:
        print(f"  The emergent QPL from QBE-constrained dynamics matches the")
        print(f"  golden cascade with r = {fit_phi['correlation']:.4f}")
        print(f"  Legacy QPL -> QBE -> Fibonacci harmonics -> Golden cascade -> Mass ratios")
        print(f"  The chain is complete.")
    else:
        corr_val = f"{fit_phi['correlation']:.4f}" if fit_phi else "N/A"
        print(f"  The QBE dynamics produce spatial structure but correlation")
        print(f"  with golden cascade is {corr_val}")
        print(f"  ")
        print(f"  KEY INSIGHT: QBE Fibonacci harmonics modulate TEMPORAL rate,")
        print(f"  not SPATIAL pattern. The phi-cascade in space comes from PAC")
        print(f"  conservation (derivation chain links 1-3), while the QBE/QPL")
        print(f"  provides the STABILITY REGULATION that keeps dynamics far-")
        print(f"  from-equilibrium. They are complementary, not identical.")
        print(f"  ")
        print(f"  The derivation gap between 'PAC conservation' and 'cascade")
        print(f"  of Gaussians' is NOT closed by QBE alone.")
    
    all_results['summary'] = {
        'gap_closed': gap_closed,
        'golden_cascade_correlation': fit_phi['correlation'] if fit_phi else None,
        'phi_rank': phi_rank if base_results else None,
        'fibonacci_best_in_null': all_results['part_d_null_test'].get('fibonacci_best'),
    }
    
    # Save results
    save_results(all_results, 'exp_12_qbe_golden_cascade')
    
    return all_results


if __name__ == '__main__':
    main()
