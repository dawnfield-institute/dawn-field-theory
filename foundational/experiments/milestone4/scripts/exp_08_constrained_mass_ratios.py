#!/usr/bin/env python3
"""
Experiment 08: Herniation Boundary — Constrained Mass Ratio Test
================================================================

Exp_07 showed the phi-scaled potential matches all three particle families
to remarkable accuracy (leptons 0.2%, down quarks 0.1%, up quarks 3.7%).
BUT the null test failed because triplet scanning over ~40-60 states gives
too many degrees of freedom — random potentials also score well.

This experiment fixes the methodology:

A. FIXED-INDEX TEST: Can we match masses using a SINGLE indexing rule?
   If electron = level j, muon = level j+k, tau = level j+2k
   (evenly spaced) — how many potentials match?

B. CONSECUTIVE STATE TEST: Do CONSECUTIVE states (j, j+1, j+2) ever
   match lepton ratios? This has no selection freedom.

C. FIBONACCI-INDEX TEST: States at Fibonacci indices (j, j+F_n, j+F_{n+1})
   — motivated by the cascade structure.

D. SIMULTANEOUS FAMILY TEST: Can one potential match BOTH leptons AND
   down quarks simultaneously? Random potentials almost never do this.

E. TOPOLOGY COMPARISON: phi-decay vs xi-decay vs linear-decay vs random-decay
   All with the same selection rule. Does phi win?

F. BOOTSTRAP CONFIDENCE: For the best match, bootstrap the eigenvalue
   uncertainties from grid resolution effects.

Dawn Field Institute, 2025-02-25
"""

import numpy as np
from scipy.linalg import eigh
import sys, os, time
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE, GAMMA_EM
from core.utils import save_results, bootstrap_ci

# ============================================================
# Target mass ratios (PDG 2024)
# ============================================================
LEPTON_RATIOS = np.array([1.0, 206.77, 3477.2])
QUARK_DN_RATIOS = np.array([1.0, 20.0, 895.1])
QUARK_UP_RATIOS = np.array([1.0, 587.96, 79861.1])

# ============================================================
# Solver (from exp_07)
# ============================================================
def cascade_potential(x, n_levels, g_base, w0, decay_base=None):
    """Cascade potential with configurable decay base (default: 1/phi)."""
    if decay_base is None:
        decay_base = INV_PHI
    V = np.zeros_like(x)
    for n in range(n_levels):
        depth = g_base * decay_base ** n
        width = w0 * (1.0 / decay_base) ** n
        V -= depth * np.exp(-x**2 / (2 * width**2))
    return V


def solve_schrodinger(V, x):
    """Solve 1D TISE, return bound state energies."""
    N = len(x)
    dx = x[1] - x[0]
    T_diag = -0.5 / dx**2 * np.full(N, -2.0)
    T_off = -0.5 / dx**2 * np.ones(N - 1)
    H = np.diag(T_diag + V) + np.diag(T_off, 1) + np.diag(T_off, -1)
    eigenvalues, eigenvectors = eigh(H)
    bound_mask = eigenvalues < 0
    return eigenvalues[bound_mask], eigenvectors[:, bound_mask]


def score_ratios(found, target):
    """Log-space L2 distance."""
    found = np.maximum(np.abs(found), 1e-10)
    target = np.maximum(target, 1e-10)
    return float(np.sum((np.log(found) - np.log(target))**2))


def pct_errors(found, target):
    """Percentage errors for each ratio."""
    return list(np.abs(found - target) / target * 100)


# ============================================================
# Part A: Fixed-Index (Evenly Spaced) Test
# ============================================================
def part_a_fixed_index(x, configs):
    """
    For each config, find the best evenly-spaced triplet (j, j+k, j+2k).
    This tests whether a SINGLE spacing rule can match mass ratios.
    """
    print("\n" + "=" * 70)
    print("PART A: Fixed-Index (Evenly Spaced) Test")
    print("=" * 70)
    
    results = {'lepton': [], 'quark_dn': []}
    
    for cfg in configs:
        V = cascade_potential(x, cfg['n'], cfg['g'], cfg['w0'], cfg.get('decay'))
        E, _ = solve_schrodinger(V, x)
        n_bound = len(E)
        if n_bound < 6:
            continue
        
        masses = np.abs(E)
        best_scores = {'lepton': float('inf'), 'quark_dn': float('inf')}
        best_matches = {'lepton': None, 'quark_dn': None}
        
        for j in range(n_bound):
            for k in range(1, (n_bound - j) // 2 + 1):
                if j + 2*k >= n_bound:
                    break
                sel = masses[[j, j+k, j+2*k]]
                ratios = sel / sel.min()
                ratios = np.sort(ratios)
                
                for family, target in [('lepton', LEPTON_RATIOS), ('quark_dn', QUARK_DN_RATIOS)]:
                    s = score_ratios(ratios, target)
                    if s < best_scores[family]:
                        best_scores[family] = s
                        best_matches[family] = {
                            'j': j, 'k': k, 'ratios': ratios.tolist(),
                            'score': s, 'n_bound': n_bound,
                            'errors_pct': pct_errors(ratios, target)
                        }
        
        for family in ['lepton', 'quark_dn']:
            if best_matches[family]:
                results[family].append({**cfg, **best_matches[family]})
    
    for family, target_name in [('lepton', 'LEPTON'), ('quark_dn', 'DOWN QUARK')]:
        results[family].sort(key=lambda r: r['score'])
        best = results[family][0] if results[family] else None
        print(f"\n  --- {target_name} (evenly spaced) ---")
        if best:
            print(f"  Best: score={best['score']:.4f}, j={best['j']}, k={best['k']}, "
                  f"n={best.get('n','?')}, g={best.get('g','?'):.2f}")
            print(f"  Ratios: {[f'{r:.1f}' for r in best['ratios']]}")
            print(f"  Errors: {[f'{e:.1f}%' for e in best['errors_pct']]}")
            sys.stdout.flush()
    
    return results


# ============================================================
# Part B: Consecutive State Test
# ============================================================
def part_b_consecutive(x, configs):
    """
    The strictest test: can 3 CONSECUTIVE states match mass ratios?
    Zero selection freedom.
    """
    print("\n" + "=" * 70)
    print("PART B: Consecutive State Test")
    print("=" * 70)
    
    results = {'lepton': [], 'quark_dn': []}
    
    for cfg in configs:
        V = cascade_potential(x, cfg['n'], cfg['g'], cfg['w0'], cfg.get('decay'))
        E, _ = solve_schrodinger(V, x)
        n_bound = len(E)
        if n_bound < 3:
            continue
        
        masses = np.abs(E)
        for j in range(n_bound - 2):
            sel = masses[j:j+3]
            ratios = np.sort(sel / sel.min())
            
            for family, target in [('lepton', LEPTON_RATIOS), ('quark_dn', QUARK_DN_RATIOS)]:
                s = score_ratios(ratios, target)
                results[family].append({
                    **cfg, 'j': j, 'ratios': ratios.tolist(),
                    'score': s, 'n_bound': n_bound,
                    'errors_pct': pct_errors(ratios, target)
                })
    
    for family, target_name in [('lepton', 'LEPTON'), ('quark_dn', 'DOWN QUARK')]:
        results[family].sort(key=lambda r: r['score'])
        best = results[family][0] if results[family] else None
        print(f"\n  --- {target_name} (consecutive states) ---")
        if best:
            print(f"  Best: score={best['score']:.4f}, j={best['j']}, "
                  f"n={best.get('n','?')}, g={best.get('g','?'):.2f}")
            print(f"  Ratios: {[f'{r:.1f}' for r in best['ratios']]}")
            print(f"  Errors: {[f'{e:.1f}%' for e in best['errors_pct']]}")
            sys.stdout.flush()
    
    return results


# ============================================================
# Part C: Fibonacci-Index Test
# ============================================================
def part_c_fibonacci_index(x, configs):
    """
    States at Fibonacci-spaced indices: (j, j+F_n, j+F_{n+1}).
    Motivated by cascade structure.
    """
    print("\n" + "=" * 70)
    print("PART C: Fibonacci-Index Test")
    print("=" * 70)
    
    # Fibonacci numbers
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    results = {'lepton': [], 'quark_dn': []}
    
    for cfg in configs:
        V = cascade_potential(x, cfg['n'], cfg['g'], cfg['w0'], cfg.get('decay'))
        E, _ = solve_schrodinger(V, x)
        n_bound = len(E)
        if n_bound < 6:
            continue
        
        masses = np.abs(E)
        
        for j in range(n_bound):
            for fi in range(len(fibs) - 1):
                f1, f2 = fibs[fi], fibs[fi + 1]
                if j + f2 >= n_bound:
                    break
                sel = masses[[j, j+f1, j+f2]]
                ratios = np.sort(sel / sel.min())
                
                for family, target in [('lepton', LEPTON_RATIOS), ('quark_dn', QUARK_DN_RATIOS)]:
                    s = score_ratios(ratios, target)
                    if s < 5.0:  # only track good matches
                        results[family].append({
                            **cfg, 'j': j, 'f1': f1, 'f2': f2,
                            'ratios': ratios.tolist(), 'score': s,
                            'n_bound': n_bound,
                            'errors_pct': pct_errors(ratios, target)
                        })
    
    for family, target_name in [('lepton', 'LEPTON'), ('quark_dn', 'DOWN QUARK')]:
        results[family].sort(key=lambda r: r['score'])
        best = results[family][0] if results[family] else None
        print(f"\n  --- {target_name} (Fibonacci-spaced) ---")
        if best:
            print(f"  Best: score={best['score']:.4f}, j={best['j']}, "
                  f"F=({best['f1']},{best['f2']})")
            print(f"  Ratios: {[f'{r:.1f}' for r in best['ratios']]}")
            print(f"  Errors: {[f'{e:.1f}%' for e in best['errors_pct']]}")
        else:
            print(f"  No good matches found (score < 5.0)")
        sys.stdout.flush()
    
    return results


# ============================================================
# Part D: Simultaneous Family Test
# ============================================================
def part_d_simultaneous(x, configs, n_random=500, seed=42):
    """
    Critical test: can one potential match BOTH leptons AND quarks?
    
    Score = lepton_score + quark_dn_score (lower is better).
    Compare phi-potential vs random potentials.
    """
    print("\n" + "=" * 70)
    print("PART D: Simultaneous Family Test (leptons + down quarks)")
    print("=" * 70)
    
    def compute_simultaneous_score(E):
        """Best simultaneous score using free triplet selection."""
        if len(E) < 6:
            return float('inf'), None, None
        masses = np.abs(E)
        n = len(masses)
        
        best_l = float('inf')
        best_d = float('inf')
        best_lt = None
        best_dt = None
        
        step = max(1, n // 25)
        for i in range(0, n-2, step):
            for j in range(i+1, n-1, step):
                for k in range(j+1, n, step):
                    sel = masses[[i, j, k]]
                    ratios = np.sort(sel / sel.min())
                    
                    sl = score_ratios(ratios, LEPTON_RATIOS)
                    if sl < best_l:
                        best_l = sl
                        best_lt = ratios.tolist()
                    
                    sd = score_ratios(ratios, QUARK_DN_RATIOS)
                    if sd < best_d:
                        best_d = sd
                        best_dt = ratios.tolist()
        
        return best_l + best_d, best_lt, best_dt
    
    # Evaluate phi-decay configs
    phi_scores = []
    for cfg in configs:
        V = cascade_potential(x, cfg['n'], cfg['g'], cfg['w0'])
        E, _ = solve_schrodinger(V, x)
        s, lt, dt = compute_simultaneous_score(E)
        phi_scores.append({
            **cfg, 'combined_score': s, 'lepton_ratios': lt, 'quark_ratios': dt
        })
    
    phi_scores.sort(key=lambda r: r['combined_score'])
    best_phi = phi_scores[0] if phi_scores else None
    
    print(f"\n  Best phi-decay combined score: {best_phi['combined_score']:.4f}" if best_phi else "  No phi-decay results")
    sys.stdout.flush()
    
    # Null test: random potentials
    rng = np.random.default_rng(seed)
    random_combined = []
    
    for trial in range(n_random):
        n_lev = rng.integers(5, 21)
        g = rng.uniform(0.5, 20.0)
        w0 = rng.uniform(0.3, 4.0)
        decay = rng.uniform(0.3, 0.95)
        
        V = cascade_potential(x, n_lev, g, w0, decay)
        E, _ = solve_schrodinger(V, x)
        s, _, _ = compute_simultaneous_score(E)
        random_combined.append(s)
    
    random_combined = np.array([s for s in random_combined if s < float('inf')])
    
    if best_phi and len(random_combined) > 0:
        our = best_phi['combined_score']
        p_value = np.mean(random_combined <= our)
        
        print(f"\n  Phi-decay combined: {our:.4f}")
        print(f"  Random distribution: mean={random_combined.mean():.4f}, "
              f"std={random_combined.std():.4f}")
        print(f"  Random min: {random_combined.min():.4f}")
        print(f"  P-value (fraction <= ours): {p_value:.4f}")
        print(f"  Significant (p < 0.05)? {p_value < 0.05}")
        sys.stdout.flush()
    
    return {
        'best_phi': best_phi,
        'n_random': len(random_combined),
        'random_mean': float(random_combined.mean()) if len(random_combined) > 0 else None,
        'random_std': float(random_combined.std()) if len(random_combined) > 0 else None,
        'random_min': float(random_combined.min()) if len(random_combined) > 0 else None,
        'p_value': float(p_value) if best_phi and len(random_combined) > 0 else None,
    }


# ============================================================
# Part E: Topology Comparison
# ============================================================
def part_e_topology_comparison(x, best_n, best_g, best_w0):
    """
    Compare phi-decay vs other decay bases:
    phi (0.618), xi (1/Ξ = 0.945), linear (0.5), sqrt(2) (0.707), e (0.368)
    
    All with the SAME g and w0. Does phi actually win?
    """
    print("\n" + "=" * 70)
    print("PART E: Topology Comparison (same g, w0, different decay)")
    print("=" * 70)
    
    topologies = {
        'phi (1/phi=0.618)': INV_PHI,
        'sqrt(2) (0.707)': 1/np.sqrt(2),
        'half (0.500)': 0.5,
        'xi-inv (1/Xi=0.945)': 1/XI_BALANCE,
        'e-inv (1/e=0.368)': 1/np.e,
        'linear (0.800)': 0.8,
        'cube-root (0.794)': 1/np.cbrt(2),
    }
    
    results = {}
    for name, decay in topologies.items():
        V = cascade_potential(x, best_n, best_g, best_w0, decay)
        E, _ = solve_schrodinger(V, x)
        n_bound = len(E)
        
        if n_bound < 6:
            results[name] = {'n_bound': n_bound, 'lepton_score': float('inf'),
                           'quark_score': float('inf')}
            continue
        
        masses = np.abs(E)
        # Use evenly-spaced test (less permissive than free triplet)
        best_ls = float('inf')
        best_ds = float('inf')
        best_lr = None
        best_dr = None
        
        for j in range(n_bound):
            for k in range(1, (n_bound - j) // 2 + 1):
                if j + 2*k >= n_bound:
                    break
                sel = masses[[j, j+k, j+2*k]]
                ratios = np.sort(sel / sel.min())
                
                sl = score_ratios(ratios, LEPTON_RATIOS)
                if sl < best_ls:
                    best_ls = sl
                    best_lr = ratios.tolist()
                
                sd = score_ratios(ratios, QUARK_DN_RATIOS)
                if sd < best_ds:
                    best_ds = sd
                    best_dr = ratios.tolist()
        
        results[name] = {
            'decay': float(decay),
            'n_bound': n_bound,
            'lepton_score': float(best_ls),
            'quark_score': float(best_ds),
            'combined': float(best_ls + best_ds),
            'lepton_ratios': best_lr,
            'quark_ratios': best_dr,
        }
    
    # Sort by combined score
    sorted_results = sorted(results.items(), key=lambda r: r[1].get('combined', float('inf')))
    
    print(f"\n  {'Topology':<25} {'Decay':>6} {'N_bound':>7} {'Lepton':>8} {'Quark':>8} {'Combined':>10}")
    print("  " + "-" * 70)
    for name, r in sorted_results:
        print(f"  {name:<25} {r.get('decay',0):>6.3f} {r['n_bound']:>7} "
              f"{r['lepton_score']:>8.4f} {r['quark_score']:>8.4f} "
              f"{r.get('combined', float('inf')):>10.4f}")
    
    winner = sorted_results[0][0]
    print(f"\n  Winner: {winner}")
    phi_rank = next(i for i, (n, _) in enumerate(sorted_results) if 'phi' in n.lower()) + 1
    print(f"  Phi rank: {phi_rank}/{len(sorted_results)}")
    sys.stdout.flush()
    
    return {
        'comparison': {k: v for k, v in results.items()},
        'winner': winner,
        'phi_rank': phi_rank,
    }


# ============================================================
# Part F: Grid Resolution Bootstrap
# ============================================================
def part_f_grid_bootstrap(best_n, best_g, best_w0, n_bootstrap=50):
    """
    Bootstrap eigenvalue stability against grid resolution.
    Run the same potential at different N and dx to quantify
    numerical uncertainty in the mass ratios.
    """
    print("\n" + "=" * 70)
    print("PART F: Grid Resolution Bootstrap")
    print("=" * 70)
    
    grid_configs = [
        (400, 0.25), (500, 0.20), (600, 0.20), (700, 0.18), 
        (800, 0.15), (500, 0.25), (600, 0.15), (400, 0.30),
    ]
    
    lepton_scores = []
    quark_scores = []
    
    for N, dx in grid_configs:
        x = np.arange(N) * dx - N * dx / 2
        V = cascade_potential(x, best_n, best_g, best_w0)
        E, _ = solve_schrodinger(V, x)
        
        if len(E) < 6:
            continue
        
        masses = np.abs(E)
        n = len(masses)
        
        best_ls = float('inf')
        best_ds = float('inf')
        
        for j in range(n):
            for k in range(1, (n - j) // 2 + 1):
                if j + 2*k >= n:
                    break
                sel = masses[[j, j+k, j+2*k]]
                ratios = np.sort(sel / sel.min())
                
                sl = score_ratios(ratios, LEPTON_RATIOS)
                if sl < best_ls:
                    best_ls = sl
                
                sd = score_ratios(ratios, QUARK_DN_RATIOS)
                if sd < best_ds:
                    best_ds = sd
        
        lepton_scores.append(best_ls)
        quark_scores.append(best_ds)
    
    lepton_scores = np.array(lepton_scores)
    quark_scores = np.array(quark_scores)
    
    print(f"  Grid configs tested: {len(grid_configs)}")
    print(f"  Lepton score range: [{lepton_scores.min():.6f}, {lepton_scores.max():.6f}]")
    print(f"  Lepton score std: {lepton_scores.std():.6f}")
    print(f"  Quark score range: [{quark_scores.min():.6f}, {quark_scores.max():.6f}]")
    print(f"  Quark score std: {quark_scores.std():.6f}")
    
    robust = lepton_scores.std() < 0.1 and quark_scores.std() < 0.1
    print(f"  Grid-robust? {robust}")
    sys.stdout.flush()
    
    return {
        'n_grids': len(grid_configs),
        'lepton_mean': float(lepton_scores.mean()),
        'lepton_std': float(lepton_scores.std()),
        'quark_mean': float(quark_scores.mean()),
        'quark_std': float(quark_scores.std()),
        'grid_robust': robust,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.perf_counter()
    
    print("=" * 70)
    print("EXP_08: Herniation Boundary -- Constrained Mass Ratio Test")
    print("=" * 70)
    
    # Grid
    N = 600
    dx = 0.2
    x = np.arange(N) * dx - N * dx / 2
    print(f"  Grid: N={N}, dx={dx}")
    sys.stdout.flush()
    
    # Config sweep (smaller but targeted, based on exp_07 best regions)
    configs = []
    for n in range(5, 22):
        for g in np.linspace(1.0, 20.0, 12):
            for w0 in np.linspace(0.3, 4.0, 10):
                configs.append({'n': n, 'g': float(g), 'w0': float(w0)})
    
    print(f"  Total configs: {len(configs)}")
    sys.stdout.flush()
    
    # --- Part A: Evenly spaced ---
    result_a = part_a_fixed_index(x, configs)
    
    # --- Part B: Consecutive states ---
    result_b = part_b_consecutive(x, configs)
    
    # --- Part C: Fibonacci indices ---
    result_c = part_c_fibonacci_index(x, configs)
    
    # --- Part D: Simultaneous family test ---
    # Use best configs from Part A (top 50)
    top_configs = []
    for family in ['lepton', 'quark_dn']:
        for r in result_a[family][:25]:
            top_configs.append({'n': r['n'], 'g': r['g'], 'w0': r['w0']})
    # Deduplicate
    seen = set()
    unique_configs = []
    for c in top_configs:
        key = (c['n'], round(c['g'], 2), round(c['w0'], 2))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    result_d = part_d_simultaneous(x, unique_configs, n_random=500)
    
    # --- Part E: Topology comparison ---
    # Use the best lepton config
    best_l = result_a['lepton'][0] if result_a['lepton'] else None
    result_e = None
    if best_l:
        result_e = part_e_topology_comparison(x, best_l['n'], best_l['g'], best_l['w0'])
    
    # --- Part F: Grid bootstrap ---
    result_f = None
    if best_l:
        result_f = part_f_grid_bootstrap(best_l['n'], best_l['g'], best_l['w0'])
    
    elapsed = time.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    tests = []
    
    # A: Do evenly-spaced states match? (score < 1.0)
    a_pass = (result_a['lepton'] and result_a['lepton'][0]['score'] < 1.0)
    tests.append(('Evenly-spaced lepton match (score < 1.0)', a_pass))
    
    # B: Consecutive states — these likely won't match (too strict)
    b_pass = (result_b['lepton'] and result_b['lepton'][0]['score'] < 0.5)
    tests.append(('Consecutive state match (score < 0.5)', b_pass))
    
    # D: Simultaneous family — phi beats random
    d_pass = result_d.get('p_value', 1.0) is not None and result_d.get('p_value', 1.0) < 0.05
    tests.append(('Simultaneous family beats random (p < 0.05)', d_pass))
    
    # E: Phi wins topology comparison
    e_pass = result_e is not None and result_e.get('phi_rank', 99) <= 2
    tests.append(('Phi in top-2 topologies', e_pass))
    
    # F: Grid robust
    f_pass = result_f is not None and result_f.get('grid_robust', False)
    tests.append(('Grid-resolution robust', f_pass))
    
    for desc, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
    
    n_pass = sum(1 for _, p in tests if p)
    print(f"\n  Score: {n_pass}/{len(tests)}")
    sys.stdout.flush()
    
    # Save
    results = {
        'experiment': 'milestone4/exp_08_constrained_mass_ratios',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'milestone': 4,
        'hypothesis': 'The phi-scaled potential matches particle mass ratios '
                     'even under constrained selection rules (evenly-spaced, '
                     'consecutive, Fibonacci indices) and beats random potentials '
                     'in simultaneous family matching.',
        'grid': {'N': N, 'dx': dx},
        'n_configs': len(configs),
        'part_a_fixed_index': {
            'lepton_top5': [r for r in result_a['lepton'][:5]],
            'quark_dn_top5': [r for r in result_a['quark_dn'][:5]],
        },
        'part_b_consecutive': {
            'lepton_top5': [r for r in result_b['lepton'][:5]],
            'quark_dn_top5': [r for r in result_b['quark_dn'][:5]],
        },
        'part_c_fibonacci': {
            'lepton_top5': result_c['lepton'][:5] if result_c['lepton'] else [],
            'quark_dn_top5': result_c['quark_dn'][:5] if result_c['quark_dn'] else [],
        },
        'part_d_simultaneous': result_d,
        'part_e_topology': result_e,
        'part_f_grid_bootstrap': result_f,
        'tests': [(d, p) for d, p in tests],
        'n_pass': n_pass,
        'n_total': len(tests),
        'elapsed_seconds': elapsed,
    }
    
    save_results(results, 'exp_08_constrained_mass_ratios')
    
    return results


if __name__ == '__main__':
    main()
