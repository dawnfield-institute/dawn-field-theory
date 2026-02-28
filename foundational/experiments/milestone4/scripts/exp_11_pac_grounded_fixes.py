#!/usr/bin/env python3
"""
Experiment 11: PAC-Grounded Weakness Fixes
==========================================

Exp_10 tried to fix the three weaknesses with EXTERNAL ideas (cascade orders,
Landauer threshold, WKB). Two of three failed.

This experiment returns to FIRST PRINCIPLES — the original PAC conservation:

  f(Parent) = Sum f(Children)

Each weakness is re-addressed through the PAC lens:

  Part A: PAC COMPLEMENT POTENTIAL (up-type quarks)
    -----------------------------------------------
    The phi-cascade is the ACTUALIZED branch. PAC conservation demands:
    
      V_parent = V_actualized + V_residual
    
    The parent is the undifferentiated potential (single well with total
    depth = g * Sum phi^(-n) = g * phi^2). The actualized part is our
    phi-cascade. The RESIDUAL is what's left:
    
      V_residual(x) = V_parent(x) - V_cascade(x)
    
    Up-type quarks should live in this complement potential.
    This requires ZERO new parameters — everything is determined by PAC.

  Part B: PAC COMPLETENESS for n_levels
    ------------------------------------
    The cascade terminates when cumulative redistribution reaches a
    PAC-natural fraction of the total. The geometric series:
    
      S_n = Sum_{k=0}^{n} phi^(-k) = phi^2 * (1 - phi^(-(n+1)))
      S_inf = phi^2 = 2.618...
    
    Completeness fraction: C_n = S_n / S_inf = 1 - phi^(-(n+1))
    
    Test PAC-natural thresholds:
      - C = 1/phi = 0.618... (golden fraction)
      - C = 1 - 1/phi^3 = 0.764...
      - C = (phi-1)/phi = 1/phi^2 = 0.382... (inverse golden squared)
      - C = phi/phi^2 = 1/phi (same as first)
      
    For C = 1 - 1/phi^k: n_max = k - 1
    
    So PAC predicts discrete n_levels = {1, 2, 3, 4, ...} at completeness
    thresholds {0.382, 0.618, 0.764, 0.854, 0.910, ...}

  Part C: RICHARDSON EXTRAPOLATION for grid convergence
    ---------------------------------------------------
    Run numerical solver at 5+ grid resolutions, fit polynomial in dx^2,
    extrapolate to dx=0. Standard technique, no WKB needed.

Dawn Field Institute, 2026-02-25
"""

import numpy as np
from scipy.linalg import eigh
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE
from core.utils import save_results

# ============================================================
# Targets
# ============================================================
LEPTON_RATIOS = np.array([1.0, 206.77, 3477.2])
QUARK_DN_RATIOS = np.array([1.0, 20.0, 895.1])
QUARK_UP_RATIOS = np.array([1.0, 587.96, 79861.1])

PHI_SQ = PHI ** 2  # 2.618... = sum of infinite phi^(-n) series

# Sweep grids (shared across parts)
N_VALUES = list(range(5, 14))
G_VALUES = [5.0, 8.0, 12.0, 15.0, 20.0, 25.0]
W0_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]


# ============================================================
# Solver
# ============================================================
def solve_numerical(V, x):
    """Solve 1D TISE, return bound energies."""
    N = len(x)
    dx = x[1] - x[0]
    T_diag = -0.5 / dx**2 * np.full(N, -2.0)
    T_off = -0.5 / dx**2 * np.ones(N - 1)
    H = np.diag(T_diag + V) + np.diag(T_off, 1) + np.diag(T_off, -1)
    eigenvalues, _ = eigh(H)
    return eigenvalues[eigenvalues < 0]


def cascade_potential(x, n_levels, g_base, w0):
    """Standard phi-cascade (actualized branch)."""
    V = np.zeros_like(x)
    for n in range(n_levels):
        V -= g_base * INV_PHI**n * np.exp(-x**2 / (2 * (w0 * PHI**n)**2))
    return V


def score_ratios(found, target):
    """Log-space L2 distance."""
    found = np.maximum(np.abs(found), 1e-10)
    target = np.maximum(target, 1e-10)
    return float(np.sum((np.log(found) - np.log(target))**2))


def pct_errors(found, target):
    return list(np.abs(found - target) / target * 100)


def best_evenly_spaced(masses, target):
    """Find best evenly-spaced triplet (j, j+k, j+2k)."""
    n_bound = len(masses)
    best_s = float('inf')
    best_m = None
    for j in range(n_bound):
        for k in range(1, (n_bound - j) // 2 + 1):
            if j + 2*k >= n_bound:
                break
            sel = masses[[j, j+k, j+2*k]]
            ratios = np.sort(sel / sel.min())
            s = score_ratios(ratios, target)
            if s < best_s:
                best_s = s
                best_m = {
                    'j': j, 'k': k, 'ratios': ratios.tolist(),
                    'score': s, 'n_bound': n_bound,
                    'errors_pct': pct_errors(ratios, target),
                }
    return best_m


# ============================================================
# PART A: PAC Complement Potential for Up-Type Quarks
# ============================================================
def part_a_pac_complement(x):
    """
    PAC conservation: V_parent = V_actualized + V_residual
    
    The parent potential is the TOTAL well before cascade differentiation.
    Its depth at x=0 equals g * Sum_{n=0}^{N-1} phi^(-n) and its shape
    depends on how we model the undifferentiated state.
    
    Three models for V_parent:
    
    Model 1 (Equal-depth Gaussian): 
      V_parent = -g_total * exp(-x^2 / 2*sigma_parent^2)
      where g_total = g * S_N, sigma_parent = w0 * phi^((N-1)/2) (geometric mean)
    
    Model 2 (Flat-bottom well):
      V_parent = -g_total for |x| < L, 0 otherwise
      where L spans the cascade width
      
    Model 3 (Envelope):
      V_parent = envelope of the cascade (convex hull of all Gaussians)
      This is the most physically natural — the parent potential is 
      the OUTER BOUNDARY of the cascade
    """
    print("\n" + "=" * 70)
    print("PART A: PAC Complement Potential for Up-Type Quarks")
    print("=" * 70)
    
    results = {}
    
    # Use the best lepton config from exp_09 as reference
    ref_configs = [
        {'label': 'lepton_ref', 'n': 7, 'g': 12.0, 'w0': 0.5},
        {'label': 'quark_dn_ref', 'n': 11, 'g': 20.0, 'w0': 0.3},
        {'label': 'wide_cascade', 'n': 9, 'g': 15.0, 'w0': 1.5},
    ]
    
    for cfg in ref_configs:
        n, g, w0 = cfg['n'], cfg['g'], cfg['w0']
        label = cfg['label']
        
        print(f"\n  --- Config: {label} (n={n}, g={g}, w0={w0}) ---")
        
        # Build cascade (actualized branch)
        V_cascade = cascade_potential(x, n, g, w0)
        
        # Total cascade depth (partial sum of geometric series)
        S_N = sum(INV_PHI**k for k in range(n))
        g_total = g * S_N
        
        # === Model 1: Gaussian parent ===
        # Width = geometric mean of cascade widths
        sigma_parent = w0 * PHI**((n-1)/2)
        V_parent_gauss = -g_total * np.exp(-x**2 / (2 * sigma_parent**2))
        V_residual_gauss = V_parent_gauss - V_cascade
        
        # === Model 2: Envelope parent ===
        # For each x, parent = min(V_cascade(x)) across a smooth envelope
        # Simply: V_parent is the widest Gaussian scaled to total depth
        sigma_wide = w0 * PHI**(n-1)  # width of widest level
        V_parent_env = -g_total * np.exp(-x**2 / (2 * sigma_wide**2))
        V_residual_env = V_parent_env - V_cascade
        
        # === Model 3: Matched-depth parent ===
        # Parent has SAME depth as cascade at x=0, wider than any level
        # This ensures V_residual(0) = 0 exactly (all potential actualized at center)
        V_cascade_0 = V_cascade[len(x)//2]  # depth at x=0
        sigma_matched = sigma_wide * 1.5  # 50% wider than widest level
        V_parent_matched = V_cascade_0 * np.exp(-x**2 / (2 * sigma_matched**2))
        V_residual_matched = V_parent_matched - V_cascade
        
        models = {
            'gauss_parent': V_residual_gauss,
            'envelope_parent': V_residual_env,
            'matched_parent': V_residual_matched,
        }
        
        cfg_results = {}
        
        for model_name, V_res in models.items():
            # The residual might have positive regions — that's fine, 
            # they act as barriers. Only keep bound states in the wells.
            E = solve_numerical(V_res, x)
            n_bound = len(E)
            
            if n_bound < 3:
                print(f"    {model_name}: only {n_bound} bound states - skip")
                cfg_results[model_name] = {'n_bound': n_bound, 'skip': True}
                continue
            
            masses = np.abs(E)
            
            # Check all three families
            model_res = {'n_bound': n_bound}
            for fam, target in [('lepton', LEPTON_RATIOS), 
                                ('quark_dn', QUARK_DN_RATIOS),
                                ('quark_up', QUARK_UP_RATIOS)]:
                m = best_evenly_spaced(masses, target)
                if m:
                    model_res[fam] = m
                    print(f"    {model_name} -> {fam}: score={m['score']:.4f}, "
                          f"errors={[f'{e:.1f}%' for e in m['errors_pct']]}")
            
            cfg_results[model_name] = model_res
        
        # Also check: what does the CASCADE itself give for up quarks?
        E_cascade = solve_numerical(V_cascade, x)
        if len(E_cascade) >= 6:
            m_up_cascade = best_evenly_spaced(np.abs(E_cascade), QUARK_UP_RATIOS)
            if m_up_cascade:
                print(f"    [reference] cascade -> up_quark: score={m_up_cascade['score']:.4f}, "
                      f"errors={[f'{e:.1f}%' for e in m_up_cascade['errors_pct']]}")
                cfg_results['cascade_up_ref'] = m_up_cascade
        
        results[label] = cfg_results
        sys.stdout.flush()
    
    # A2: Wider parameter sweep for the best complement model
    print("\n  --- A2: Complement Potential Parameter Sweep ---")
    
    best_complement_up = None
    best_complement_config = None
    
    total_cfgs = len(N_VALUES) * len(G_VALUES) * len(W0_VALUES) * 2  # 2 models
    done = 0
    
    for n in N_VALUES:
        for g in G_VALUES:
            for w0 in W0_VALUES:
                V_cas = cascade_potential(x, n, g, w0)
                
                S_N = sum(INV_PHI**k for k in range(n))
                g_total = g * S_N
                
                # Only two most physical models (skip envelope — sigma too wide)
                for model_id, sigma_fn in [
                    ('gauss', lambda: w0 * PHI**((n-1)/2)),
                    ('matched', lambda: w0 * PHI**((n-1)/2) * 1.5),
                ]:
                    done += 1
                    sigma = sigma_fn()
                    
                    # Skip if sigma wider than grid domain
                    if sigma > 50:
                        continue
                    
                    try:
                        if model_id == 'matched':
                            V_parent = V_cas[len(x)//2] * np.exp(-x**2 / (2 * sigma**2))
                        else:
                            V_parent = -g_total * np.exp(-x**2 / (2 * sigma**2))
                        
                        V_res = V_parent - V_cas
                        E = solve_numerical(V_res, x)
                    except Exception:
                        continue
                    
                    if len(E) < 6:
                        continue
                    
                    masses = np.abs(E)
                    m = best_evenly_spaced(masses, QUARK_UP_RATIOS)
                    
                    if m and (best_complement_up is None or m['score'] < best_complement_up['score']):
                        best_complement_up = m
                        best_complement_config = {
                            'n': n, 'g': g, 'w0': w0, 'model': model_id,
                        }
                        
                        # Also check leptons and down quarks in same potential
                        m_lep = best_evenly_spaced(masses, LEPTON_RATIOS)
                        m_dn = best_evenly_spaced(masses, QUARK_DN_RATIOS)
                        best_complement_config['lepton'] = m_lep
                        best_complement_config['quark_dn'] = m_dn
        
        print(f"    A2 progress: n={n} done ({done}/{total_cfgs})")
        sys.stdout.flush()
    
    if best_complement_up:
        print(f"\n    Best complement up-quark: score={best_complement_up['score']:.4f}")
        print(f"      config: n={best_complement_config['n']}, "
              f"g={best_complement_config['g']}, w0={best_complement_config['w0']}, "
              f"model={best_complement_config['model']}")
        print(f"      ratios: {[f'{r:.1f}' for r in best_complement_up['ratios']]}")
        print(f"      errors: {[f'{e:.1f}%' for e in best_complement_up['errors_pct']]}")
        
        # Compare to exp_10's best cascade result (score=0.563)
        print(f"\n    Improvement over cascade-only: "
              f"{0.5631/best_complement_up['score']:.1f}x" 
              if best_complement_up['score'] > 0 else "inf")
    else:
        print("\n    No valid complement up-quark matches found")
    
    # A3: Simultaneous test — can cascade handle leptons+down quarks while
    #     complement handles up quarks from the SAME (n, g, w0)?
    print("\n  --- A3: Simultaneous Cascade + Complement ---")
    
    best_simultaneous = None
    best_sim_score = float('inf')
    done = 0
    
    for n in N_VALUES:
        for g in G_VALUES:
            for w0 in W0_VALUES:
                done += 1
                # Cascade spectrum (for leptons + down quarks)
                V_cas = cascade_potential(x, n, g, w0)
                try:
                    E_cas = solve_numerical(V_cas, x)
                except Exception:
                    continue
                if len(E_cas) < 6:
                    continue
                masses_cas = np.abs(E_cas)
                
                m_lep = best_evenly_spaced(masses_cas, LEPTON_RATIOS)
                m_dn = best_evenly_spaced(masses_cas, QUARK_DN_RATIOS)
                
                if not m_lep or not m_dn:
                    continue
                
                # Complement spectrum (for up quarks)
                S_N = sum(INV_PHI**k for k in range(n))
                g_total = g * S_N
                
                # Gaussian parent (geometric-mean width)
                sigma = w0 * PHI**((n-1)/2)
                if sigma > 50:
                    continue
                V_parent = -g_total * np.exp(-x**2 / (2 * sigma**2))
                V_res = V_parent - V_cas
                try:
                    E_res = solve_numerical(V_res, x)
                except Exception:
                    continue
                
                if len(E_res) < 6:
                    continue
                masses_res = np.abs(E_res)
                m_up = best_evenly_spaced(masses_res, QUARK_UP_RATIOS)
                
                if not m_up:
                    continue
                
                # Combined score: all three families
                combined = m_lep['score'] + m_dn['score'] + m_up['score']
                
                if combined < best_sim_score:
                    best_sim_score = combined
                    best_simultaneous = {
                        'n': n, 'g': g, 'w0': w0,
                        'lepton': m_lep,
                        'quark_dn': m_dn,
                        'quark_up': m_up,
                        'combined': combined,
                        'n_cascade_states': len(E_cas),
                        'n_complement_states': len(E_res),
                    }
        
        print(f"    A3 progress: n={n} done ({done} configs)")
        sys.stdout.flush()
    
    if best_simultaneous:
        s = best_simultaneous
        print(f"\n    Best simultaneous (n={s['n']}, g={s['g']}, w0={s['w0']}):")
        print(f"      Combined score: {s['combined']:.4f}")
        for fam in ['lepton', 'quark_dn', 'quark_up']:
            m = s[fam]
            src = "cascade" if fam != 'quark_up' else "complement"
            print(f"      {fam} ({src}): score={m['score']:.4f}, "
                  f"errors={[f'{e:.1f}%' for e in m['errors_pct']]}")
        print(f"      Cascade bound states: {s['n_cascade_states']}")
        print(f"      Complement bound states: {s['n_complement_states']}")
    else:
        print("\n    No valid simultaneous match found")
    
    results['sweep_best'] = best_complement_config
    results['simultaneous'] = best_simultaneous
    sys.stdout.flush()
    
    return results


# ============================================================
# PART B: PAC Completeness Threshold for n_levels
# ============================================================
def part_b_pac_completeness(x):
    """
    The cascade partial sum: S_n = Sum_{k=0}^{n} phi^(-k)
    Converges to: S_inf = 1/(1-1/phi) = phi^2 = 2.618...
    Completeness: C_n = S_n / S_inf = 1 - phi^(-(n+1))
    
    PAC-natural thresholds for cascade termination:
      C = 1/phi     = 0.618  -> n+1 = 2  -> n = 1
      C = phi/phi^2  = 1/phi  (same)
      C = 1-1/phi^3 = 0.764  -> n+1 = 3  -> n = 2
      C = 1-1/phi^4 = 0.854  -> n+1 = 4  -> n = 3
      ...
      C = 1-1/phi^k = ...    -> n = k-1
    
    So every integer n IS a PAC threshold! The question is: which one
    matches the PHYSICAL cascade?
    
    New idea: the cascade depth is determined by the GOLDEN CUT of the
    angular momentum space. In 1D, the system has a finite number of
    phase-space cells N_cells = L * p_max / (2*pi*hbar). The cascade
    runs for n levels until it has sampled 1/phi of phase space.
    
    n_max = floor(1/phi * total_phase_space_cells)
    
    Or simpler: n_max is where the cascade eigenvalue spacing crosses
    from ordered to disordered (edge-of-chaos). This is testable.
    """
    print("\n" + "=" * 70)
    print("PART B: PAC Completeness for n_levels")
    print("=" * 70)
    
    # B1: Completeness fractions
    print("\n  --- B1: PAC Completeness Series ---")
    for n in range(1, 16):
        C_n = 1 - INV_PHI**(n+1)
        S_n = sum(INV_PHI**k for k in range(n+1))
        print(f"    n={n:2d}: C_n = {C_n:.6f}, S_n = {S_n:.4f} / {PHI_SQ:.4f}")
    sys.stdout.flush()
    
    # B2: For each family, find optimal n and report its completeness
    print("\n  --- B2: Optimal n_levels and Completeness ---")
    
    families = {
        'lepton': LEPTON_RATIOS,
        'quark_dn': QUARK_DN_RATIOS,
    }
    
    b2_results = {}
    
    for fam_name, target in families.items():
        best_per_n = {}
        
        for n in range(3, 16):
            best_score = float('inf')
            best_cfg = None
            
            for g in G_VALUES:
                for w0 in W0_VALUES:
                    try:
                        V = cascade_potential(x, n, g, w0)
                        E = solve_numerical(V, x)
                    except Exception:
                        continue
                    if len(E) < 6:
                        continue
                    m = best_evenly_spaced(np.abs(E), target)
                    if m and m['score'] < best_score:
                        best_score = m['score']
                        best_cfg = {'n': n, 'g': g, 'w0': w0, **m}
            
            if best_cfg:
                C_n = 1 - INV_PHI**(n+1)
                best_per_n[n] = {**best_cfg, 'completeness': C_n}
        
        # Find optimal n
        optimal_n = min(best_per_n, key=lambda n: best_per_n[n]['score'])
        optimal = best_per_n[optimal_n]
        
        print(f"\n    --- {fam_name} ---")
        print(f"    Score by n_levels:")
        for n in sorted(best_per_n.keys()):
            d = best_per_n[n]
            marker = " <-- OPTIMAL" if n == optimal_n else ""
            print(f"      n={n:2d}: score={d['score']:.4f}, C={d['completeness']:.4f}{marker}")
        
        # Check: is there a THRESHOLD below which all n give good scores?
        good_threshold = optimal['score'] * 3  # within 3x of optimal
        good_n_range = [n for n, d in best_per_n.items() if d['score'] < good_threshold]
        
        if good_n_range:
            print(f"    Good range (score < {good_threshold:.3f}): "
                  f"n = {min(good_n_range)} to {max(good_n_range)}")
            print(f"    Completeness range: "
                  f"C = {1-INV_PHI**(min(good_n_range)+1):.4f} to "
                  f"{1-INV_PHI**(max(good_n_range)+1):.4f}")
        
        b2_results[fam_name] = {
            'optimal_n': optimal_n,
            'optimal_completeness': optimal['completeness'],
            'optimal_score': optimal['score'],
            'good_n_range': good_n_range,
            'scores_by_n': {n: d['score'] for n, d in best_per_n.items()},
        }
    
    # B3: Is there a UNIVERSAL completeness threshold?
    print("\n  --- B3: Universal Completeness Threshold ---")
    
    opt_completeness = [b2_results[f]['optimal_completeness'] for f in b2_results]
    mean_C = np.mean(opt_completeness)
    
    print(f"    Lepton optimal C: {b2_results['lepton']['optimal_completeness']:.4f} "
          f"(n={b2_results['lepton']['optimal_n']})")
    print(f"    Down-Q optimal C: {b2_results['quark_dn']['optimal_completeness']:.4f} "
          f"(n={b2_results['quark_dn']['optimal_n']})")
    print(f"    Mean: {mean_C:.4f}")
    
    # Check nearby phi-derived values
    phi_thresholds = {
        '1 - 1/phi^8': 1 - INV_PHI**8,
        '1 - 1/phi^9': 1 - INV_PHI**9,
        '1 - 1/phi^10': 1 - INV_PHI**10,
        '1 - 1/phi^11': 1 - INV_PHI**11,
        '1 - 1/phi^12': 1 - INV_PHI**12,
        '1 - 1/phi^13': 1 - INV_PHI**13,
    }
    
    print(f"\n    Nearby phi-thresholds:")
    for name, val in phi_thresholds.items():
        n_implied = round(np.log(1 - val) / np.log(INV_PHI)) - 1 if val < 1 else 99
        print(f"      {name} = {val:.6f} -> n = {n_implied}")
    
    b2_results['mean_completeness'] = mean_C
    sys.stdout.flush()
    
    return b2_results


# ============================================================
# PART C: Richardson Extrapolation
# ============================================================
def part_c_richardson(x_ref):
    """
    Run the solver at multiple grid spacings, extrapolate eigenvalue
    ratios to dx -> 0 using Richardson extrapolation.
    
    For a 2nd-order finite difference scheme, the error is O(dx^2).
    So we fit: ratio(dx) = ratio_true + a * dx^2 + b * dx^4 + ...
    """
    print("\n" + "=" * 70)
    print("PART C: Richardson Extrapolation for Grid Convergence")
    print("=" * 70)
    
    # Test on lepton reference config
    cfg = {'n': 7, 'g': 12.0, 'w0': 0.5}
    
    # Grid sizes: vary N while keeping domain fixed at [-60, 60]
    domain_half = 60.0
    grid_sizes = [
        ('coarse',   300, 2*domain_half/300),
        ('medium',   500, 2*domain_half/500),
        ('standard', 600, 2*domain_half/600),
        ('fine',     800, 2*domain_half/800),
        ('finer',   1000, 2*domain_half/1000),
        ('finest',  1500, 2*domain_half/1500),
        ('extreme', 2000, 2*domain_half/2000),
    ]
    
    print("\n  --- C1: Eigenvalue Convergence ---")
    
    all_energies = {}
    all_lepton_matches = {}
    all_quark_dn_matches = {}
    
    for name, N, dx in grid_sizes:
        x = np.linspace(-domain_half, domain_half, N)
        V = cascade_potential(x, cfg['n'], cfg['g'], cfg['w0'])
        E = solve_numerical(V, x)
        n_bound = len(E)
        
        all_energies[name] = {'N': N, 'dx': dx, 'n_bound': n_bound, 'E': E}
        
        if n_bound >= 6:
            masses = np.abs(E)
            m_lep = best_evenly_spaced(masses, LEPTON_RATIOS)
            m_dn = best_evenly_spaced(masses, QUARK_DN_RATIOS)
            
            all_lepton_matches[name] = m_lep
            all_quark_dn_matches[name] = m_dn
            
            lep_str = f"score={m_lep['score']:.6f}" if m_lep else "no match"
            dn_str = f"score={m_dn['score']:.6f}" if m_dn else "no match"
            
            print(f"    {name:10s} (N={N:5d}, dx={dx:.4f}): "
                  f"{n_bound} states, lepton {lep_str}, quark_dn {dn_str}")
    
    sys.stdout.flush()
    
    # C2: Richardson extrapolation on eigenvalue ratios
    print("\n  --- C2: Richardson Extrapolation ---")
    
    # Collect the lepton ratio data points for extrapolation
    dx_list = []
    ratio2_list = []  # mu/e ratio
    ratio3_list = []  # tau/e ratio
    
    for name, N, dx in grid_sizes:
        m = all_lepton_matches.get(name)
        if m and m['score'] < 5.0:  # reasonable match
            dx_list.append(dx)
            ratio2_list.append(m['ratios'][1])  # mu/e
            ratio3_list.append(m['ratios'][2])  # tau/e
    
    if len(dx_list) >= 4:
        dx_arr = np.array(dx_list)
        r2_arr = np.array(ratio2_list)
        r3_arr = np.array(ratio3_list)
        
        # Fit: ratio = a0 + a1*dx^2 + a2*dx^4
        dx2 = dx_arr**2
        
        # For mu/e ratio
        A = np.column_stack([np.ones_like(dx2), dx2, dx2**2])
        try:
            coeffs_2, _, _, _ = np.linalg.lstsq(A, r2_arr, rcond=None)
            r2_extrap = coeffs_2[0]
            r2_err = abs(r2_extrap - 206.77) / 206.77 * 100
            
            print(f"\n    mu/e ratio:")
            print(f"      Data points: {len(dx_arr)}")
            print(f"      Extrapolated (dx->0): {r2_extrap:.2f}")
            print(f"      Target: 206.77")
            print(f"      Error: {r2_err:.2f}%")
        except Exception as e:
            print(f"\n    mu/e extrapolation failed: {e}")
            r2_extrap = None
            r2_err = None
        
        # For tau/e ratio
        try:
            coeffs_3, _, _, _ = np.linalg.lstsq(A, r3_arr, rcond=None)
            r3_extrap = coeffs_3[0]
            r3_err = abs(r3_extrap - 3477.2) / 3477.2 * 100
            
            print(f"\n    tau/e ratio:")
            print(f"      Extrapolated (dx->0): {r3_extrap:.2f}")
            print(f"      Target: 3477.2")
            print(f"      Error: {r3_err:.2f}%")
        except Exception as e:
            print(f"\n    tau/e extrapolation failed: {e}")
            r3_extrap = None
            r3_err = None
        
        # Compare to raw values
        print(f"\n    Raw vs Extrapolated:")
        for name, N, dx in grid_sizes:
            m = all_lepton_matches.get(name)
            if m:
                print(f"      {name:10s} (dx={dx:.4f}): "
                      f"mu/e={m['ratios'][1]:.1f}, tau/e={m['ratios'][2]:.1f}")
        if r2_extrap:
            print(f"      {'EXTRAP':10s} (dx=0):     "
                  f"mu/e={r2_extrap:.1f}, tau/e={r3_extrap:.1f}")
        
        # Is the extrapolated value more stable than individual grid results?
        r2_cv = np.std(r2_arr) / np.mean(r2_arr) if np.mean(r2_arr) > 0 else 0
        
        return {
            'grid_data': {name: {'N': N, 'dx': dx, 
                                 'lepton': all_lepton_matches.get(name),
                                 'quark_dn': all_quark_dn_matches.get(name)}
                          for name, N, dx in grid_sizes},
            'extrapolation': {
                'mu_e_extrap': float(r2_extrap) if r2_extrap else None,
                'mu_e_error': float(r2_err) if r2_err else None,
                'tau_e_extrap': float(r3_extrap) if r3_extrap else None,
                'tau_e_error': float(r3_err) if r3_err else None,
                'n_points': len(dx_arr),
                'raw_cv': float(r2_cv),
            },
        }
    else:
        print("\n    Insufficient data points for extrapolation")
        return {'error': 'insufficient data', 'n_points': len(dx_list)}


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.perf_counter()
    
    print("=" * 70)
    print("EXP_11: PAC-Grounded Weakness Fixes")
    print("=" * 70)
    
    N = 600
    dx = 0.2
    x = np.arange(N) * dx - N * dx / 2
    print(f"  Grid: N={N}, dx={dx}")
    sys.stdout.flush()
    
    results = {}
    
    # Part A: PAC complement for up quarks
    results['part_a'] = part_a_pac_complement(x)
    
    # Part B: PAC completeness for n_levels
    results['part_b'] = part_b_pac_completeness(x)
    
    # Part C: Richardson extrapolation
    results['part_c'] = part_c_richardson(x)
    
    elapsed = time.perf_counter() - t0
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # W1 assessment
    sim = results['part_a'].get('simultaneous')
    if sim:
        up_score = sim['quark_up']['score']
        print(f"\n  W1 (Up Quarks via PAC Complement):")
        print(f"    Combined 3-family score: {sim['combined']:.4f}")
        print(f"    Up-quark score: {up_score:.4f}")
        print(f"    Up-quark errors: {[f'{e:.1f}%' for e in sim['quark_up']['errors_pct']]}")
        w1_improved = up_score < 0.56  # better than exp_10's phi^1 score
        print(f"    Improved over cascade-only (0.563): {w1_improved}")
    else:
        w1_improved = False
        print("\n  W1 (Up Quarks): No simultaneous match found")
    
    # W2 assessment
    b = results['part_b']
    print(f"\n  W2 (n_levels from PAC Completeness):")
    print(f"    Lepton optimal n={b['lepton']['optimal_n']}, "
          f"C={b['lepton']['optimal_completeness']:.4f}")
    print(f"    Down-Q optimal n={b['quark_dn']['optimal_n']}, "
          f"C={b['quark_dn']['optimal_completeness']:.4f}")
    print(f"    Mean completeness: {b['mean_completeness']:.4f}")
    
    # W3 assessment
    c = results['part_c']
    if isinstance(c, dict) and 'extrapolation' in c:
        ext = c['extrapolation']
        print(f"\n  W3 (Richardson Extrapolation):")
        if ext.get('mu_e_extrap'):
            print(f"    mu/e ratio: {ext['mu_e_extrap']:.1f} (target 206.8, "
                  f"error {ext['mu_e_error']:.1f}%)")
        if ext.get('tau_e_extrap'):
            print(f"    tau/e ratio: {ext['tau_e_extrap']:.1f} (target 3477.2, "
                  f"error {ext['tau_e_error']:.1f}%)")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    sys.stdout.flush()
    
    output = {
        'experiment': 'milestone4/exp_11_pac_grounded_fixes',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'milestone': 4,
        'hypothesis': 'PAC conservation provides principled solutions to all '
                     'three weaknesses: complement potential for up-type quarks, '
                     'completeness threshold for n_levels, Richardson '
                     'extrapolation for grid convergence.',
        'results': results,
        'elapsed_seconds': elapsed,
    }
    
    save_results(output, 'exp_11_pac_grounded_fixes')
    return output


if __name__ == '__main__':
    main()
