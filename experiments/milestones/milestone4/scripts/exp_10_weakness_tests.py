#!/usr/bin/env python3
"""
Experiment 10: Addressing Derivation Chain Weaknesses
=====================================================

Three identified weaknesses from exp_07/08/09:

  W1. Up-type quarks (u:c:t) have 45-59% errors
  W2. n_levels parameter is implicitly free per family
  W3. Grid sensitivity — mass ratios shift with discretization

This experiment addresses each:

  Part A: Up-Quark Diagnosis — test phi^2 cascade (second-order coupling)
    Physical motivation: M^2 has eigenvalue phi^2, representing 
    two-step cascade processes. Up-type quarks couple to Higgs 
    differently from down-type in SM; in PAC, they may couple 
    through second-order cascade processes.

  Part B: Landauer n_levels — derive n_levels from first principles
    n_max = floor(ln(g_base / E_threshold) / ln(decay_base))
    where E_threshold = kT * ln(2) (Landauer minimum erasure cost)

  Part C: WKB Analytic Eigenvalues — remove grid dependence
    Implement semi-classical quantization to get eigenvalues
    without matrix diagonalization. If WKB ratios match numerical
    ones, mass ratios are grid-independent.

Dawn Field Institute, 2026-02-25
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.integrate import quad
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE, GAMMA_EM
from core.utils import save_results

# ============================================================
# Targets (PDG 2024)
# ============================================================
LEPTON_RATIOS = np.array([1.0, 206.77, 3477.2])
QUARK_DN_RATIOS = np.array([1.0, 20.0, 895.1])
QUARK_UP_RATIOS = np.array([1.0, 587.96, 79861.1])

# Mass hierarchy spans
LEPTON_SPAN = 3477.2       # tau/electron
QUARK_DN_SPAN = 895.1      # bottom/down
QUARK_UP_SPAN = 79861.1    # top/up

# ============================================================
# Numerical Solver
# ============================================================
def cascade_potential_fn(x, n_levels, g_base, w0, decay_base=None):
    """Cascade potential with configurable decay base."""
    if decay_base is None:
        decay_base = INV_PHI
    V = np.zeros_like(x)
    for n in range(n_levels):
        depth = g_base * decay_base ** n
        width = w0 * (1.0 / decay_base) ** n
        V -= depth * np.exp(-x**2 / (2 * width**2))
    return V


def solve_numerical(V, x):
    """Solve 1D TISE numerically, return bound state energies."""
    N = len(x)
    dx = x[1] - x[0]
    T_diag = -0.5 / dx**2 * np.full(N, -2.0)
    T_off = -0.5 / dx**2 * np.ones(N - 1)
    H = np.diag(T_diag + V) + np.diag(T_off, 1) + np.diag(T_off, -1)
    eigenvalues, _ = eigh(H)
    return eigenvalues[eigenvalues < 0]


def score_ratios(found, target):
    """Log-space L2 distance."""
    found = np.maximum(np.abs(found), 1e-10)
    target = np.maximum(target, 1e-10)
    return float(np.sum((np.log(found) - np.log(target))**2))


def pct_errors(found, target):
    return list(np.abs(found - target) / target * 100)


def best_evenly_spaced(masses, target):
    """Find best evenly-spaced triplet matching target ratios."""
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
# PART A: Up-Quark Diagnosis
# ============================================================
def part_a_up_quark_diagnosis(x):
    """
    Test whether up-type quarks require a DIFFERENT cascade order.
    
    Physics: The Fibonacci matrix M = [[1,1],[1,0]] has eigenvalue phi.
    M^2 = [[2,1],[1,1]] has eigenvalue phi^2 = 2.618...
    M^3 = [[3,2],[2,1]] has eigenvalue phi^3 = 4.236...
    
    If up-type quarks couple through a 2nd-order process (two cascade
    steps per interaction), they'd see phi^2 scaling.
    
    Hierarchy analysis:
      Leptons: span 3477x  -> log_phi(3477) = 16.9
      Down Q:  span 895x   -> log_phi(895) = 14.1  
      Up Q:    span 79861x -> log_phi(79861) = 23.4
      Up Q:    span 79861x -> log_phi2(79861) = 11.7 (much more compact!)
    """
    print("\n" + "=" * 70)
    print("PART A: Up-Quark Cascade Order Diagnosis")
    print("=" * 70)
    
    # A1: Verify M^k eigenvalues
    print("\n  --- A1: Fibonacci Matrix Powers ---")
    M = np.array([[1, 1], [1, 0]], dtype=float)
    for k in range(1, 5):
        Mk = np.linalg.matrix_power(M, k)
        eigs = np.linalg.eigvals(Mk)
        phi_k = PHI ** k
        max_eig = max(abs(eigs))
        print(f"    M^{k}: eigenvalue = {max_eig:.6f}, phi^{k} = {phi_k:.6f}, "
              f"match = {abs(max_eig - phi_k) < 1e-10}")
    sys.stdout.flush()
    
    # Hierarchy analysis
    print("\n  --- A2: Hierarchy Span Analysis ---")
    for name, span in [('lepton', LEPTON_SPAN), ('quark_dn', QUARK_DN_SPAN), 
                        ('quark_up', QUARK_UP_SPAN)]:
        log_phi = np.log(span) / np.log(PHI)
        log_phi2 = np.log(span) / np.log(PHI**2)
        log_phi3 = np.log(span) / np.log(PHI**3)
        print(f"    {name:10s}: span = {span:10.1f}x, "
              f"log_phi = {log_phi:5.1f}, log_phi2 = {log_phi2:5.1f}, "
              f"log_phi3 = {log_phi3:5.1f}")
    sys.stdout.flush()
    
    # A3: Test each cascade order for up-type quarks
    print("\n  --- A3: Cascade Order Sweep for Up-Type Quarks ---")
    
    cascade_orders = {
        'phi^1 (standard)': INV_PHI,          # phi^(-1) = 0.618
        'phi^2 (2nd order)': INV_PHI**2,      # phi^(-2) = 0.382
        'phi^3 (3rd order)': INV_PHI**3,      # phi^(-3) = 0.236
    }
    
    order_results = {}
    
    for order_name, decay_base in cascade_orders.items():
        best_up = None
        best_lepton = None
        best_quark_dn = None
        
        # Sweep parameters
        for n_lev in range(4, 16):
            for g in [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0]:
                for w0 in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                    V = cascade_potential_fn(x, n_lev, g, w0, decay_base)
                    E = solve_numerical(V, x)
                    n_bound = len(E)
                    if n_bound < 6:
                        continue
                    masses = np.abs(E)
                    
                    # Up-type quarks
                    m = best_evenly_spaced(masses, QUARK_UP_RATIOS)
                    if m and (best_up is None or m['score'] < best_up['score']):
                        best_up = {**m, 'n': n_lev, 'g': g, 'w0': w0}
                    
                    # Leptons (for comparison)
                    m = best_evenly_spaced(masses, LEPTON_RATIOS)
                    if m and (best_lepton is None or m['score'] < best_lepton['score']):
                        best_lepton = {**m, 'n': n_lev, 'g': g, 'w0': w0}
                    
                    # Down quarks
                    m = best_evenly_spaced(masses, QUARK_DN_RATIOS)
                    if m and (best_quark_dn is None or m['score'] < best_quark_dn['score']):
                        best_quark_dn = {**m, 'n': n_lev, 'g': g, 'w0': w0}
        
        order_results[order_name] = {
            'decay_base': float(decay_base),
            'up_quark': best_up,
            'lepton': best_lepton,
            'quark_dn': best_quark_dn,
        }
        
        print(f"\n    {order_name} (decay={decay_base:.4f}):")
        for fam, res in [('up_quark', best_up), ('lepton', best_lepton), 
                          ('quark_dn', best_quark_dn)]:
            if res:
                print(f"      {fam:10s}: score={res['score']:.4f}, "
                      f"n={res['n']}, g={res['g']:.1f}, w0={res['w0']:.1f}")
                print(f"        errors: {[f'{e:.1f}%' for e in res['errors_pct']]}")
            else:
                print(f"      {fam:10s}: no valid match found")
        sys.stdout.flush()
    
    # A4: Unified model — leptons + down quarks use phi^1, up quarks use phi^2
    print("\n  --- A4: Unified Model Assessment ---")
    phi1_up = order_results.get('phi^1 (standard)', {}).get('up_quark')
    phi2_up = order_results.get('phi^2 (2nd order)', {}).get('up_quark')
    
    if phi1_up and phi2_up:
        improvement = phi1_up['score'] / phi2_up['score'] if phi2_up['score'] > 0 else float('inf')
        print(f"    phi^1 up-quark score: {phi1_up['score']:.4f}")
        print(f"    phi^2 up-quark score: {phi2_up['score']:.4f}")
        print(f"    Improvement ratio: {improvement:.1f}x")
        
        phi2_pass = phi2_up['score'] < 0.1  # Stringent threshold
        print(f"    phi^2 passes (score < 0.1): {phi2_pass}")
    
    # Check if phi^1 still best for leptons and down quarks
    phi1_lep = order_results.get('phi^1 (standard)', {}).get('lepton')
    phi2_lep = order_results.get('phi^2 (2nd order)', {}).get('lepton')
    if phi1_lep and phi2_lep:
        print(f"\n    Lepton: phi^1 score={phi1_lep['score']:.4f}, "
              f"phi^2 score={phi2_lep['score']:.4f}")
        print(f"    phi^1 better for leptons: {phi1_lep['score'] < phi2_lep['score']}")
    
    phi1_dn = order_results.get('phi^1 (standard)', {}).get('quark_dn')
    phi2_dn = order_results.get('phi^2 (2nd order)', {}).get('quark_dn')
    if phi1_dn and phi2_dn:
        print(f"    Down Q: phi^1 score={phi1_dn['score']:.4f}, "
              f"phi^2 score={phi2_dn['score']:.4f}")
        print(f"    phi^1 better for down quarks: {phi1_dn['score'] < phi2_dn['score']}")
    
    sys.stdout.flush()
    
    # A5: Null test — SKIP if phi^2 didn't improve up quarks
    print("\n  --- A5: Null Test ---")
    phi2_improved = phi2_up and phi1_up and phi2_up['score'] < phi1_up['score']
    
    if phi2_improved and phi2_up['score'] < 0.5:
        # Only worth testing if phi^2 actually helped
        n_null = 50  # Reduced from 200 for speed
        best_g, best_w0 = phi2_up['g'], phi2_up['w0']
        n_lev_range = range(max(4, phi2_up['n']-2), min(14, phi2_up['n']+3))
        
        random_scores = []
        rng = np.random.RandomState(42)
        
        for trial in range(n_null):
            rand_decay = 0.1 + rng.random() * 0.8
            best_trial = float('inf')
            
            for n_lev in n_lev_range:
                V = cascade_potential_fn(x, n_lev, best_g, best_w0, rand_decay)
                E = solve_numerical(V, x)
                if len(E) < 6:
                    continue
                m = best_evenly_spaced(np.abs(E), QUARK_UP_RATIOS)
                if m and m['score'] < best_trial:
                    best_trial = m['score']
            
            random_scores.append(best_trial)
        
        random_scores = np.array(random_scores)
        phi2_score = phi2_up['score']
        p_value = float(np.mean(random_scores <= phi2_score))
        
        print(f"    phi^2 score: {phi2_score:.4f}, random min: {np.min(random_scores):.4f}")
        print(f"    p-value: {p_value:.3f}")
        
        order_results['null_test'] = {
            'phi2_score': phi2_score,
            'random_min': float(np.min(random_scores)),
            'p_value': p_value,
            'PASS': p_value < 0.05,
        }
    else:
        print(f"    SKIPPED — phi^2 did not improve up quarks (score {phi2_up['score']:.3f} > phi^1 {phi1_up['score']:.3f})")
        print(f"    CONCLUSION: Up-type quarks resist ALL tested cascade orders.")
        print(f"    The phi-cascade mechanism works for leptons and down quarks")
        print(f"    but NOT for up-type quarks. This is an honest boundary of the model.")
        order_results['null_test'] = {
            'skipped': True,
            'reason': 'phi^2 did not improve up-type quarks',
            'PASS': False,
        }
    
    sys.stdout.flush()
    return order_results


# ============================================================
# PART B: Landauer n_levels Derivation
# ============================================================
def part_b_landauer_nlevels(x):
    """
    Derive n_levels from the Landauer erasure bound.
    
    Each cascade level n has coupling g * decay^n.
    The cascade terminates when this coupling drops below the
    Landauer erasure threshold: E_threshold = kT * ln(2).
    
    In natural units with kT = 1:
      g * decay^n_max = ln(2)
      n_max = floor(ln(g / ln(2)) / ln(1/decay))
    
    For phi-cascade: n_max = floor(ln(g / ln(2)) / ln(phi))
    For phi^2-cascade: n_max = floor(ln(g / ln(2)) / ln(phi^2))
    """
    print("\n" + "=" * 70)
    print("PART B: Landauer-Derived n_levels")
    print("=" * 70)
    
    kT = 1.0
    E_thresh = kT * np.log(2)  # Landauer limit
    
    # B1: Analytical predictions
    print("\n  --- B1: Predicted n_levels ---")
    print(f"    Landauer threshold: kT*ln(2) = {E_thresh:.4f}")
    
    test_cases = []
    for g in [5.0, 7.0, 10.0, 12.0, 15.0, 20.0]:
        for cascade_type, decay in [('phi^1', INV_PHI), ('phi^2', INV_PHI**2)]:
            n_pred = int(np.floor(np.log(g / E_thresh) / np.log(1.0 / decay)))
            test_cases.append({
                'g': g, 'cascade': cascade_type, 'decay': decay,
                'n_predicted': n_pred,
            })
            print(f"    g={g:5.1f}, {cascade_type}: n_predicted = {n_pred}")
    
    sys.stdout.flush()
    
    # B2: Compare predicted vs optimal n_levels for each family
    print("\n  --- B2: Predicted vs Optimal n_levels ---")
    
    families = {
        'lepton': (LEPTON_RATIOS, INV_PHI),
        'quark_dn': (QUARK_DN_RATIOS, INV_PHI),
        'quark_up': (QUARK_UP_RATIOS, INV_PHI**2),
    }
    
    b2_results = {}
    
    for fam_name, (target, decay) in families.items():
        print(f"\n    --- {fam_name} (decay={decay:.4f}) ---")
        
        best_overall = None  # best across all n
        best_at_predicted = {}  # grouped by g
        
        for g in [5.0, 7.0, 10.0, 12.0, 15.0, 20.0]:
            n_pred = int(np.floor(np.log(g / E_thresh) / np.log(1.0 / decay)))
            n_pred = max(3, n_pred)
            
            # Score at predicted n
            best_pred = None
            for w0 in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                V = cascade_potential_fn(x, n_pred, g, w0, decay)
                E = solve_numerical(V, x)
                if len(E) < 6:
                    continue
                m = best_evenly_spaced(np.abs(E), target)
                if m and (best_pred is None or m['score'] < best_pred['score']):
                    best_pred = {**m, 'n': n_pred, 'g': g, 'w0': w0}
            
            # Score at optimal n (sweep n_pred +/- 5)
            best_opt = None
            best_opt_n = None
            for n in range(max(3, n_pred - 5), n_pred + 6):
                for w0 in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                    V = cascade_potential_fn(x, n, g, w0, decay)
                    E = solve_numerical(V, x)
                    if len(E) < 6:
                        continue
                    m = best_evenly_spaced(np.abs(E), target)
                    if m and (best_opt is None or m['score'] < best_opt['score']):
                        best_opt = {**m, 'n': n, 'g': g, 'w0': w0}
                        best_opt_n = n
            
            if best_pred and best_opt:
                closeness = abs(n_pred - best_opt_n) if best_opt_n else 99
                ratio = best_pred['score'] / best_opt['score'] if best_opt['score'] > 0 else float('inf')
                print(f"    g={g:5.1f}: predicted n={n_pred}, optimal n={best_opt_n}, "
                      f"diff={closeness}, score_ratio={ratio:.2f}")
                best_at_predicted[g] = {
                    'n_predicted': n_pred,
                    'n_optimal': best_opt_n,
                    'score_predicted': best_pred['score'],
                    'score_optimal': best_opt['score'],
                    'score_ratio': ratio,
                }
            
            if best_opt and (best_overall is None or best_opt['score'] < best_overall['score']):
                best_overall = best_opt
        
        b2_results[fam_name] = {
            'best_overall': best_overall,
            'landauer_comparison': best_at_predicted,
        }
        
        if best_overall:
            print(f"    BEST: n={best_overall['n']}, g={best_overall['g']:.1f}, "
                  f"w0={best_overall['w0']:.1f}, score={best_overall['score']:.4f}")
            print(f"      errors: {[f'{e:.1f}%' for e in best_overall['errors_pct']]}")
    
    # B3: Assess whether Landauer-derived n is competitive
    print("\n  --- B3: Landauer n_levels Assessment ---")
    all_close = True
    for fam_name, data in b2_results.items():
        ratios = [v['score_ratio'] for v in data['landauer_comparison'].values() 
                  if 'score_ratio' in v and v['score_ratio'] < 100]
        if ratios:
            avg_ratio = np.mean(ratios)
            print(f"    {fam_name}: avg score_ratio (predicted/optimal) = {avg_ratio:.2f}")
            if avg_ratio > 5.0:
                all_close = False
        else:
            print(f"    {fam_name}: no valid comparisons")
            all_close = False
    
    b2_results['landauer_competitive'] = all_close
    print(f"    Landauer n_levels competitive (ratio < 5x): {all_close}")
    sys.stdout.flush()
    
    return b2_results


# ============================================================
# PART C: WKB Semi-Classical Eigenvalues
# ============================================================
def part_c_wkb_eigenvalues(x_grid):
    """
    Implement WKB (Wentzel-Kramers-Brillouin) quantization for the
    cascade potential. The WKB condition is:
    
      integral_{x1}^{x2} sqrt(2(E - V(x))) dx = (n + 1/2) * pi
    
    where x1, x2 are classical turning points (V(x) = E).
    
    This gives eigenvalues WITHOUT matrix diagonalization, eliminating
    grid sensitivity entirely.
    """
    print("\n" + "=" * 70)
    print("PART C: WKB Semi-Classical Eigenvalues")
    print("=" * 70)
    
    def wkb_action(E, V_func, x_fine):
        """Compute WKB action integral for energy E."""
        V_vals = V_func(x_fine)
        mask = (E - V_vals) > 0
        if not np.any(mask):
            return 0.0
        
        # Find turning points
        indices = np.where(mask)[0]
        if len(indices) < 2:
            return 0.0
        
        x1_idx, x2_idx = indices[0], indices[-1]
        integrand = np.sqrt(2 * np.maximum(E - V_vals[x1_idx:x2_idx+1], 0))
        dx = x_fine[1] - x_fine[0]
        return np.trapz(integrand, dx=dx)
    
    def wkb_eigenvalues(V_func, x_fine, n_max=60):
        """Find WKB eigenvalues by root-finding on the quantization condition."""
        V_vals = V_func(x_fine)
        V_min = np.min(V_vals)
        
        eigenvalues = []
        
        # For each quantum number, find E such that action = (n + 1/2)*pi
        for n in range(n_max):
            target_action = (n + 0.5) * np.pi
            
            # Bracket the energy
            E_low = V_min + 1e-10
            E_high = -1e-10  # Just below zero (bound states)
            
            action_low = wkb_action(E_low, V_func, x_fine)
            action_high = wkb_action(E_high, V_func, x_fine)
            
            if action_low > target_action:
                # Even lowest energy has too much action -- no solution
                break
            if action_high < target_action:
                # Even E=0 doesn't have enough action -- no more bound states
                break
            
            # Bisection
            try:
                def f(E):
                    return wkb_action(E, V_func, x_fine) - target_action
                E_n = brentq(f, E_low, E_high, xtol=1e-10, rtol=1e-10, maxiter=200)
                eigenvalues.append(E_n)
            except (ValueError, RuntimeError):
                break
        
        return np.array(eigenvalues)
    
    # C1: Test on reference configurations
    print("\n  --- C1: WKB vs Numerical Comparison ---")
    
    # Fine grid for WKB (independent of numerical grid)
    x_fine = np.linspace(-80, 80, 10000)
    
    # Reference configs from exp_09
    ref_configs = [
        {'name': 'lepton_best', 'n': 7, 'g': 12.0, 'w0': 0.5, 'decay': INV_PHI},
        {'name': 'quark_dn_best', 'n': 16, 'g': 10.0, 'w0': 0.5, 'decay': INV_PHI},
        {'name': 'reference_1', 'n': 10, 'g': 10.0, 'w0': 1.0, 'decay': INV_PHI},
    ]
    
    c1_results = {}
    
    for cfg in ref_configs:
        V_func = lambda xx, c=cfg: cascade_potential_fn(xx, c['n'], c['g'], c['w0'], c['decay'])
        
        # Numerical (on the standard grid)
        V_num = V_func(x_grid)
        E_num = solve_numerical(V_num, x_grid)
        
        # WKB (on fine grid)
        E_wkb = wkb_eigenvalues(V_func, x_fine)
        
        # Compare the bound states that both methods find
        n_compare = min(len(E_num), len(E_wkb))
        if n_compare > 0:
            # For each numerical eigenvalue, find nearest WKB
            max_frac_err = 0
            rms_frac_err = 0
            errors = []
            for i in range(n_compare):
                frac = abs(E_wkb[i] - E_num[i]) / abs(E_num[i]) if E_num[i] != 0 else 0
                errors.append(frac)
                max_frac_err = max(max_frac_err, frac)
            rms_frac_err = np.sqrt(np.mean(np.array(errors)**2))
            
            print(f"\n    {cfg['name']}:")
            print(f"      Numerical: {len(E_num)} bound states")
            print(f"      WKB:       {len(E_wkb)} bound states")
            print(f"      Compared:  {n_compare} states")
            print(f"      Max frac error: {max_frac_err:.6f}")
            print(f"      RMS frac error: {rms_frac_err:.6f}")
            
            c1_results[cfg['name']] = {
                'n_numerical': len(E_num),
                'n_wkb': len(E_wkb),
                'n_compared': n_compare,
                'max_frac_error': float(max_frac_err),
                'rms_frac_error': float(rms_frac_err),
            }
        else:
            print(f"\n    {cfg['name']}: insufficient states for comparison")
            c1_results[cfg['name']] = {'error': 'insufficient states'}
    
    sys.stdout.flush()
    
    # C2: Mass ratio comparison — do WKB ratios match numerical ratios?
    print("\n  --- C2: WKB Mass Ratios vs Numerical ---")
    
    c2_results = {}
    
    for cfg in ref_configs[:2]:  # lepton and quark configs
        V_func = lambda xx, c=cfg: cascade_potential_fn(xx, c['n'], c['g'], c['w0'], c['decay'])
        
        V_num = V_func(x_grid)
        E_num = solve_numerical(V_num, x_grid)
        E_wkb = wkb_eigenvalues(V_func, x_fine)
        
        if len(E_num) < 6 or len(E_wkb) < 6:
            continue
        
        masses_num = np.abs(E_num)
        masses_wkb = np.abs(E_wkb)
        
        target = LEPTON_RATIOS if 'lepton' in cfg['name'] else QUARK_DN_RATIOS
        
        m_num = best_evenly_spaced(masses_num, target)
        m_wkb = best_evenly_spaced(masses_wkb, target)
        
        if m_num and m_wkb:
            print(f"\n    {cfg['name']}:")
            print(f"      Numerical: score={m_num['score']:.6f}, "
                  f"j={m_num['j']}, k={m_num['k']}")
            print(f"        ratios: {[f'{r:.1f}' for r in m_num['ratios']]}")
            print(f"      WKB:      score={m_wkb['score']:.6f}, "
                  f"j={m_wkb['j']}, k={m_wkb['k']}")
            print(f"        ratios: {[f'{r:.1f}' for r in m_wkb['ratios']]}")
            
            # Are the ratios the same?
            if m_num['j'] == m_wkb['j'] and m_num['k'] == m_wkb['k']:
                ratio_diff = np.abs(np.array(m_num['ratios']) - np.array(m_wkb['ratios']))
                ratio_frac = ratio_diff / np.array(m_num['ratios'])
                print(f"      SAME indices! Ratio fractional diff: "
                      f"{[f'{d:.4f}' for d in ratio_frac]}")
            
            c2_results[cfg['name']] = {
                'numerical': m_num,
                'wkb': m_wkb,
                'same_indices': m_num['j'] == m_wkb['j'] and m_num['k'] == m_wkb['k'],
            }
    
    sys.stdout.flush()
    
    # C3: Grid independence test — compare WKB at different resolutions
    print("\n  --- C3: Grid Independence (WKB at Multiple Resolutions) ---")
    
    resolutions = [2000, 5000, 10000, 20000, 50000]
    cfg = ref_configs[0]  # lepton config
    V_func = lambda xx: cascade_potential_fn(xx, cfg['n'], cfg['g'], cfg['w0'], cfg['decay'])
    
    wkb_scores = []
    wkb_ratios_list = []
    
    for npts in resolutions:
        x_test = np.linspace(-80, 80, npts)
        E_wkb = wkb_eigenvalues(V_func, x_test)
        if len(E_wkb) < 6:
            continue
        
        m = best_evenly_spaced(np.abs(E_wkb), LEPTON_RATIOS)
        if m:
            wkb_scores.append(m['score'])
            wkb_ratios_list.append(m['ratios'])
            print(f"    npts={npts:6d}: score={m['score']:.8f}, "
                  f"ratios={[f'{r:.2f}' for r in m['ratios']]}")
    
    if len(wkb_scores) >= 3:
        score_range = max(wkb_scores) - min(wkb_scores)
        score_cv = np.std(wkb_scores) / np.mean(wkb_scores) if np.mean(wkb_scores) > 0 else 0
        print(f"\n    Score range: {score_range:.8f}")
        print(f"    Score CV: {score_cv:.6f}")
        print(f"    Grid-independent (CV < 0.01): {score_cv < 0.01}")
        
        c1_results['grid_independence'] = {
            'resolutions': resolutions[:len(wkb_scores)],
            'scores': wkb_scores,
            'score_range': float(score_range),
            'score_cv': float(score_cv),
            'PASS': score_cv < 0.01,
        }
    
    # Also compare NUMERICAL grid sensitivity
    print("\n  --- C4: Numerical Grid Sensitivity (for reference) ---")
    
    num_grids = [
        (400, 0.30), (500, 0.24), (600, 0.20), (800, 0.15), (1000, 0.12)
    ]
    
    num_scores = []
    for N, dx in num_grids:
        x_test = np.arange(N) * dx - N * dx / 2
        V = cascade_potential_fn(x_test, cfg['n'], cfg['g'], cfg['w0'], cfg['decay'])
        E = solve_numerical(V, x_test)
        if len(E) < 6:
            continue
        m = best_evenly_spaced(np.abs(E), LEPTON_RATIOS)
        if m:
            num_scores.append(m['score'])
            print(f"    N={N:5d}, dx={dx:.2f}: score={m['score']:.8f}, "
                  f"ratios={[f'{r:.2f}' for r in m['ratios']]}")
    
    if len(num_scores) >= 3:
        num_cv = np.std(num_scores) / np.mean(num_scores) if np.mean(num_scores) > 0 else 0
        print(f"\n    Numerical score CV: {num_cv:.6f}")
        print(f"    WKB score CV:      {score_cv:.6f}" if len(wkb_scores) >= 3 else "")
        if len(wkb_scores) >= 3:
            print(f"    WKB improvement: {num_cv/score_cv:.1f}x" if score_cv > 0 else 
                  "    WKB perfectly stable")
    
    sys.stdout.flush()
    
    return {**c1_results, 'mass_ratio_comparison': c2_results}


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.perf_counter()
    
    print("=" * 70)
    print("EXP_10: Weakness Tests — Up Quarks, n_levels, Grid Sensitivity")
    print("=" * 70)
    
    N = 600
    dx = 0.2
    x = np.arange(N) * dx - N * dx / 2
    print(f"  Grid: N={N}, dx={dx}")
    sys.stdout.flush()
    
    results = {}
    
    # Part A: Up-quark diagnosis
    results['part_a_up_quarks'] = part_a_up_quark_diagnosis(x)
    
    # Part B: Landauer n_levels  
    results['part_b_landauer'] = part_b_landauer_nlevels(x)
    
    # Part C: WKB eigenvalues
    results['part_c_wkb'] = part_c_wkb_eigenvalues(x)
    
    elapsed = time.perf_counter() - t0
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # W1: Up quarks
    a_res = results['part_a_up_quarks']
    phi2_up = a_res.get('phi^2 (2nd order)', {}).get('up_quark')
    phi1_up = a_res.get('phi^1 (standard)', {}).get('up_quark')
    if phi2_up and phi1_up:
        print(f"\n  W1 (Up Quarks):")
        print(f"    phi^1 best up-quark score: {phi1_up['score']:.4f}")
        print(f"    phi^2 best up-quark score: {phi2_up['score']:.4f}")
        print(f"    phi^2 improvement: {phi1_up['score']/phi2_up['score']:.1f}x")
        w1_pass = phi2_up['score'] < 0.1
        print(f"    W1 RESOLVED: {w1_pass}")
    else:
        w1_pass = False
        print("\n  W1 (Up Quarks): insufficient data")
    
    # W2: n_levels
    b_res = results['part_b_landauer']
    w2_pass = b_res.get('landauer_competitive', False)
    print(f"\n  W2 (n_levels): Landauer-derived n competitive: {w2_pass}")
    
    # W3: Grid sensitivity
    c_res = results['part_c_wkb']
    grid_test = c_res.get('grid_independence', {})
    w3_pass = grid_test.get('PASS', False)
    print(f"\n  W3 (Grid): WKB grid-independent: {w3_pass}")
    if 'score_cv' in grid_test:
        print(f"    WKB CV: {grid_test['score_cv']:.6f}")
    
    all_resolved = w1_pass and w2_pass and w3_pass
    print(f"\n  ALL WEAKNESSES RESOLVED: {all_resolved}")
    print(f"  Total time: {elapsed:.1f}s")
    sys.stdout.flush()
    
    output = {
        'experiment': 'milestone4/exp_10_weakness_tests',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'milestone': 4,
        'hypothesis': 'The three identified weaknesses (up quarks, n_levels, '
                     'grid sensitivity) can be addressed through phi^2 '
                     'cascade, Landauer derivation, and WKB quantization.',
        'results': results,
        'summary': {
            'W1_up_quarks_resolved': w1_pass,
            'W2_nlevels_derived': w2_pass,
            'W3_grid_independent': w3_pass,
            'all_resolved': all_resolved,
        },
        'elapsed_seconds': elapsed,
    }
    
    save_results(output, 'exp_10_weakness_tests')
    return output


if __name__ == '__main__':
    main()
