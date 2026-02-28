"""
Herniation Simulation — Experiment I
======================================
Can a PAC tree topology reproduce known particle mass ratios?

Strategy:
- Known particle masses define TARGET ratios
- The bound state spectrum depends on the potential shape
- The potential shape is determined by the PAC tree topology
- Sweep topological parameters to find spectra that match

We'll work with mass ratios rather than absolute masses since
our simulation is in arbitrary units. The key ratios:

LEPTON FAMILY:
  electron : muon : tau = 1 : 206.77 : 3477.2
  muon/electron = 206.77
  tau/muon = 16.82
  tau/electron = 3477.2

QUARK FAMILY (approximate current masses):
  up : down : strange : charm : bottom : top
  2.2 : 4.7 : 95 : 1275 : 4180 : 173000 MeV
  
  down/up = 2.14
  strange/down = 20.2
  charm/strange = 13.4
  bottom/charm = 3.28
  top/bottom = 41.4

The question: does ANY reasonable potential topology produce
bound state ratios that match these?

Peter McNally / Dawn Field Institute, 2026
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

# =============================================================================
# Target mass ratios
# =============================================================================

# Leptons (well-measured, clean targets)
LEPTON_MASSES = {
    'electron': 0.511,      # MeV
    'muon': 105.658,        # MeV  
    'tau': 1776.86          # MeV
}
LEPTON_RATIOS = np.array([1.0, 206.77, 3477.2])  # e, μ, τ

# Quarks (less clean due to confinement, but approximate)
QUARK_MASSES = {
    'up': 2.2, 'down': 4.7, 'strange': 95,
    'charm': 1275, 'bottom': 4180, 'top': 173000
}
QUARK_RATIOS = np.array([1.0, 2.14, 43.2, 579.5, 1900.0, 78636.4])

# Generation structure (most interesting pattern):
# Each generation is roughly 10-200x the previous
GEN_RATIOS = {
    'lepton_gen': [1.0, 206.77, 3477.2],      # e → μ → τ
    'quark_gen_up': [1.0, 579.5, 78636.4],     # u → c → t  
    'quark_gen_down': [1.0, 20.2, 889.4],      # d → s → b
}


# =============================================================================
# Potential builders
# =============================================================================

def build_potential(x, params, topology='cascade'):
    """
    Build potential from topology parameters.
    
    Parameters vary by topology type but generally include:
    - depths: depth of each level
    - widths: width of each level  
    - offsets: center position of each level
    """
    V = np.zeros_like(x)
    
    if topology == 'cascade':
        n_levels = len(params) // 2
        depths = params[:n_levels]
        widths = params[n_levels:2*n_levels]
        for i in range(n_levels):
            V -= abs(depths[i]) * np.exp(-x**2 / (2 * widths[i]**2))
    
    elif topology == 'fibonacci':
        coupling = params[0]
        base_width = params[1]
        n_levels = int(params[2]) if len(params) > 2 else 8
        exponent = params[3] if len(params) > 3 else 0.5
        
        fibs = [1, 1]
        for _ in range(max(0, n_levels - 2)):
            fibs.append(fibs[-1] + fibs[-2])
        
        for i, f in enumerate(fibs[:n_levels]):
            depth = coupling / (i + 1)**exponent
            width = base_width * f
            V -= abs(depth) * np.exp(-x**2 / (2 * width**2))
    
    elif topology == 'power_law':
        coupling = params[0]
        base_width = params[1]
        n_levels = int(params[2]) if len(params) > 2 else 8
        alpha = params[3] if len(params) > 3 else 1.0  # depth scaling
        beta = params[4] if len(params) > 4 else 1.0    # width scaling
        
        for i in range(n_levels):
            depth = coupling / (i + 1)**alpha
            width = base_width * (i + 1)**beta
            V -= abs(depth) * np.exp(-x**2 / (2 * width**2))
    
    elif topology == 'golden':
        coupling = params[0]
        base_width = params[1]
        n_levels = int(params[2]) if len(params) > 2 else 8
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(n_levels):
            depth = coupling * phi**(-i)
            width = base_width * phi**i
            V -= abs(depth) * np.exp(-x**2 / (2 * width**2))
    
    elif topology == 'xi_scaled':
        coupling = params[0]
        base_width = params[1]
        n_levels = int(params[2]) if len(params) > 2 else 8
        xi = 1.0571
        
        for i in range(n_levels):
            depth = coupling * xi**(-i)
            width = base_width * xi**i
            V -= abs(depth) * np.exp(-x**2 / (2 * width**2))
    
    elif topology == 'double_cascade':
        # Two interleaved cascades (order + chaos)
        coupling_o = params[0]
        coupling_c = params[1]
        base_width_o = params[2]
        base_width_c = params[3]
        n_levels = int(params[4]) if len(params) > 4 else 5
        separation = params[5] if len(params) > 5 else 0.0
        
        for i in range(n_levels):
            # Order cascade (steeper, narrower)
            V -= abs(coupling_o) / (i+1) * np.exp(-((x - separation))**2 / (2 * (base_width_o*(i+1))**2))
            # Chaos cascade (shallower, wider)
            V -= abs(coupling_c) / (i+1)**0.5 * np.exp(-((x + separation))**2 / (2 * (base_width_c*(i+1))**2))
    
    return V


def solve_spectrum(V, x):
    """Solve for bound states."""
    N = len(x)
    dx = x[1] - x[0]
    
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = -2.0
        if i + 1 < N: T[i, i+1] = 1.0
        if i - 1 >= 0: T[i, i-1] = 1.0
    T *= -0.5 / dx**2
    
    H = T + np.diag(V)
    eigenvalues = eigh(H, eigvals_only=True)
    
    bound_E = eigenvalues[eigenvalues < 0]
    return bound_E


def get_mass_ratios(bound_E, n_target):
    """
    Extract mass ratios from bound state energies.
    
    Mass ~ |binding energy|
    We want ratios relative to the SHALLOWEST state
    (since the shallowest massive particle is the electron/lightest).
    
    But actually — the deep states are the heavy ones.
    So mass_ratios = |E_deep| / |E_shallow|
    
    We pick n_target evenly-spaced states from the spectrum.
    """
    if len(bound_E) < n_target:
        return None
    
    masses = np.abs(bound_E)
    
    # Strategy: pick states at specific fractional positions
    # The idea is that different "particle types" correspond to
    # different regions of the spectrum, not adjacent levels
    indices = np.linspace(0, len(bound_E)-1, n_target).astype(int)
    selected = masses[indices]
    
    # Ratios relative to lightest (last = shallowest)
    ratios = selected / selected[-1]
    
    return ratios[::-1]  # lightest first


def score_against_targets(ratios, targets):
    """
    Score how well the extracted ratios match target mass ratios.
    Use log-space comparison since ratios span orders of magnitude.
    """
    if ratios is None or len(ratios) != len(targets):
        return float('inf')
    
    # Avoid log(0)
    ratios = np.maximum(ratios, 1e-10)
    targets = np.maximum(targets, 1e-10)
    
    log_diff = np.log(ratios) - np.log(targets)
    return np.sum(log_diff**2)


# =============================================================================
# MAIN EXPERIMENTS
# =============================================================================

if __name__ == '__main__':
    
    # Setup
    N = 800
    dx = 0.2
    x = np.arange(N) * dx - N * dx / 2
    
    print("=" * 80)
    print("EXPERIMENT I: MATCHING PARTICLE MASS RATIOS")
    print("=" * 80)
    
    # ================================================================
    # Part 1: Survey different topologies
    # ================================================================
    print("\n" + "-" * 80)
    print("PART 1: TOPOLOGY SURVEY — LEPTON RATIOS (1 : 206.8 : 3477.2)")
    print("-" * 80)
    
    topologies = {
        'cascade': [
            ([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1], [3, 5, 8, 12, 18, 25, 35, 50]),
            ([10.0, 5.0, 2.5, 1.25, 0.625], [2, 4, 8, 16, 32]),
            ([8.0, 4.0, 2.0, 1.0, 0.5, 0.25], [3, 6, 12, 24, 48, 96]),
        ],
        'fibonacci': [
            [5.0, 2.0, 8, 0.5],
            [10.0, 1.5, 10, 0.3],
            [3.0, 3.0, 8, 0.7],
            [8.0, 1.0, 12, 0.4],
        ],
        'power_law': [
            [5.0, 3.0, 8, 1.0, 1.0],
            [5.0, 3.0, 8, 0.5, 1.5],
            [5.0, 3.0, 8, 1.5, 0.5],
            [5.0, 3.0, 8, 2.0, 1.0],
            [10.0, 2.0, 10, 1.0, 2.0],
            [10.0, 2.0, 10, 0.3, 2.0],
        ],
        'golden': [
            [5.0, 2.0, 8],
            [10.0, 1.5, 10],
            [3.0, 3.0, 12],
            [8.0, 1.0, 15],
        ],
        'xi_scaled': [
            [5.0, 2.0, 8],
            [10.0, 1.5, 10],
            [3.0, 3.0, 12],
            [20.0, 1.0, 15],
        ],
        'double_cascade': [
            [5.0, 3.0, 3.0, 5.0, 5, 0.0],
            [5.0, 3.0, 3.0, 5.0, 5, 10.0],
            [8.0, 4.0, 2.0, 6.0, 7, 5.0],
            [10.0, 5.0, 2.0, 8.0, 8, 15.0],
        ],
    }
    
    best_score = float('inf')
    best_config = None
    
    all_results = []
    
    for topo_name, param_sets in topologies.items():
        for pi, params in enumerate(param_sets):
            if topo_name == 'cascade':
                depths, widths = params
                flat_params = depths + widths
                V = build_potential(x, flat_params, 'cascade')
            else:
                V = build_potential(x, params, topo_name)
            
            bound_E = solve_spectrum(V, x)
            
            if len(bound_E) < 3:
                continue
            
            # Try different ways to pick 3 states for lepton matching
            n_bound = len(bound_E)
            masses = np.abs(bound_E)
            
            # Strategy 1: evenly spaced
            ratios_even = get_mass_ratios(bound_E, 3)
            score_even = score_against_targets(ratios_even, LEPTON_RATIOS)
            
            # Strategy 2: ground, middle, shallowest
            if n_bound >= 3:
                mid = n_bound // 2
                sel = masses[[0, mid, -1]]
                ratios_gms = sel[::-1] / sel[-1]
                ratios_gms = ratios_gms[::-1]
                # Flip so lightest first
                ratios_gms = np.sort(sel / sel.min())
                score_gms = score_against_targets(ratios_gms, LEPTON_RATIOS)
            else:
                score_gms = float('inf')
                ratios_gms = None
            
            # Strategy 3: scan all triplets for best match
            best_triplet_score = float('inf')
            best_triplet = None
            if n_bound >= 10:
                # Sample triplets
                for i in range(0, n_bound-2, max(1, n_bound//20)):
                    for j in range(i+1, n_bound-1, max(1, n_bound//15)):
                        k = n_bound - 1
                        sel3 = masses[[i, j, k]]
                        r3 = np.sort(sel3 / sel3.min())
                        s3 = score_against_targets(r3, LEPTON_RATIOS)
                        if s3 < best_triplet_score:
                            best_triplet_score = s3
                            best_triplet = (i, j, k, r3)
            
            best_local = min(score_even, score_gms, best_triplet_score)
            
            result = {
                'topology': topo_name,
                'param_idx': pi,
                'n_bound': n_bound,
                'score_even': score_even,
                'score_gms': score_gms,
                'best_triplet_score': best_triplet_score,
                'best_score': best_local,
                'ratios_even': ratios_even.tolist() if ratios_even is not None else None,
                'best_triplet': best_triplet
            }
            all_results.append(result)
            
            if best_local < best_score:
                best_score = best_local
                best_config = result
    
    # Sort by score
    all_results.sort(key=lambda r: r['best_score'])
    
    print(f"\n  TOP 10 CONFIGURATIONS (lepton ratio matching):")
    print(f"  {'Rank':>4} {'Topology':>16} {'#Bound':>7} {'Score':>10} {'Ratios':>40}")
    print("  " + "-" * 80)
    
    for rank, r in enumerate(all_results[:10]):
        if r['best_triplet'] is not None and r['best_triplet_score'] <= r['score_even']:
            ratios_str = f"{r['best_triplet'][3]}"
        elif r['ratios_even'] is not None:
            ratios_str = f"{r['ratios_even']}"
        else:
            ratios_str = "N/A"
        
        print(f"  {rank+1:4d} {r['topology']:>16} {r['n_bound']:7d} "
              f"{r['best_score']:10.4f} {ratios_str[:40]}")
    
    print(f"\n  TARGET: {LEPTON_RATIOS}")
    
    # ================================================================
    # Part 2: Detailed analysis of best topology
    # ================================================================
    print("\n" + "-" * 80)
    print("PART 2: DETAILED ANALYSIS OF BEST MATCH")
    print("-" * 80)
    
    if best_config:
        topo = best_config['topology']
        pi = best_config['param_idx']
        print(f"\n  Best topology: {topo}, param set {pi}")
        print(f"  Score: {best_config['best_score']:.4f}")
        print(f"  Bound states: {best_config['n_bound']}")
        
        if best_config['best_triplet']:
            i, j, k, ratios = best_config['best_triplet']
            print(f"  Best triplet indices: ({i}, {j}, {k})")
            print(f"  Ratios: {ratios}")
            print(f"  Target: {LEPTON_RATIOS}")
            print(f"  Log errors: {np.log(ratios) - np.log(LEPTON_RATIOS)}")
    
    # ================================================================
    # Part 3: Optimization — find the BEST potential for leptons
    # ================================================================
    print("\n" + "-" * 80)
    print("PART 3: NUMERICAL OPTIMIZATION")
    print("-" * 80)
    
    def objective_lepton(params):
        """Optimize power-law potential to match lepton ratios."""
        try:
            coupling = abs(params[0]) + 0.1
            base_width = abs(params[1]) + 0.5
            n_levels = 8
            alpha = abs(params[2]) + 0.1
            beta = abs(params[3]) + 0.1
            
            V = build_potential(x, [coupling, base_width, n_levels, alpha, beta], 'power_law')
            bound_E = solve_spectrum(V, x)
            
            if len(bound_E) < 10:
                return 100.0
            
            masses = np.abs(bound_E)
            n_bound = len(bound_E)
            
            # Scan triplets
            best_s = 100.0
            for i in range(0, n_bound-2, max(1, n_bound//15)):
                for j in range(i+1, n_bound-1, max(1, n_bound//10)):
                    k = n_bound - 1
                    sel3 = masses[[i, j, k]]
                    r3 = np.sort(sel3 / sel3.min())
                    s = score_against_targets(r3, LEPTON_RATIOS)
                    if s < best_s:
                        best_s = s
            
            return best_s
        except:
            return 100.0
    
    print("\n  Optimizing power-law topology for lepton ratios...")
    
    # Multi-start optimization
    best_opt_score = float('inf')
    best_opt_params = None
    
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        x0 = rng.uniform([1, 0.5, 0.1, 0.1], [15, 5, 3, 3])
        
        try:
            result = minimize(objective_lepton, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01})
            
            if result.fun < best_opt_score:
                best_opt_score = result.fun
                best_opt_params = result.x
                
                if trial % 10 == 0 or result.fun < 1.0:
                    print(f"    Trial {trial}: score={result.fun:.4f}, "
                          f"params=[{', '.join(f'{p:.3f}' for p in result.x)}]")
        except:
            continue
    
    print(f"\n  Best optimized score: {best_opt_score:.4f}")
    if best_opt_params is not None:
        coupling = abs(best_opt_params[0]) + 0.1
        base_width = abs(best_opt_params[1]) + 0.5
        alpha = abs(best_opt_params[2]) + 0.1
        beta = abs(best_opt_params[3]) + 0.1
        
        print(f"  Parameters: coupling={coupling:.3f}, width={base_width:.3f}, "
              f"alpha={alpha:.3f}, beta={beta:.3f}")
        
        V = build_potential(x, [coupling, base_width, 8, alpha, beta], 'power_law')
        bound_E = solve_spectrum(V, x)
        masses = np.abs(bound_E)
        n_bound = len(bound_E)
        
        print(f"  Bound states: {n_bound}")
        
        # Find best triplet
        best_s = 100.0
        best_t = None
        for i in range(0, n_bound-2):
            for j in range(i+1, n_bound-1):
                k = n_bound - 1
                sel3 = masses[[i, j, k]]
                r3 = np.sort(sel3 / sel3.min())
                s = score_against_targets(r3, LEPTON_RATIOS)
                if s < best_s:
                    best_s = s
                    best_t = (i, j, k, r3)
        
        if best_t:
            i, j, k, ratios = best_t
            print(f"\n  Best triplet: levels ({i}, {j}, {k})")
            print(f"  Found ratios:  {ratios}")
            print(f"  Target ratios: {LEPTON_RATIOS}")
            
            # Percent errors
            pct_err = np.abs(ratios - LEPTON_RATIOS) / LEPTON_RATIOS * 100
            print(f"  Percent errors: {[f'{e:.1f}%' for e in pct_err]}")
            
            log_err = np.abs(np.log(ratios) - np.log(LEPTON_RATIOS))
            print(f"  Log errors: {[f'{e:.4f}' for e in log_err]}")
    
    # ================================================================
    # Part 4: Do generations emerge naturally?
    # ================================================================
    print("\n" + "-" * 80)
    print("PART 4: DO GENERATION STRUCTURES EMERGE?")
    print("-" * 80)
    
    print("\n  Testing whether the spectrum naturally groups into generations")
    print("  (clusters of 3 with large gaps between clusters)")
    
    # Use the best potential we found
    if best_opt_params is not None:
        V = build_potential(x, [coupling, base_width, 8, alpha, beta], 'power_law')
    else:
        V = build_potential(x, [5.0, 3.0, 8, 1.0, 1.0], 'power_law')
    
    bound_E = solve_spectrum(V, x)
    masses = np.abs(bound_E)
    n_bound = len(bound_E)
    
    gaps = np.diff(bound_E)
    
    # Look for large gaps (generation boundaries)
    mean_gap = np.mean(gaps)
    large_gap_threshold = 1.5 * mean_gap
    
    print(f"\n  Mean gap: {mean_gap:.4f}")
    print(f"  Large gap threshold (1.5x mean): {large_gap_threshold:.4f}")
    
    large_gaps = [(i, g) for i, g in enumerate(gaps) if g > large_gap_threshold]
    
    print(f"\n  Large gaps found: {len(large_gaps)}")
    for idx, gap in large_gaps[:10]:
        print(f"    Between levels {idx} and {idx+1}: gap = {gap:.4f} "
              f"({gap/mean_gap:.1f}x mean)")
    
    # Gap ratio analysis — do ratios cluster?
    gap_ratios = gaps[:-1] / gaps[1:]
    
    print(f"\n  Gap ratio distribution:")
    print(f"  Min: {gap_ratios.min():.4f}")
    print(f"  Max: {gap_ratios.max():.4f}")
    print(f"  Mean: {gap_ratios.mean():.4f}")
    print(f"  Std: {gap_ratios.std():.4f}")
    
    # ================================================================
    # Part 5: Mass ratio from consecutive states
    # ================================================================
    print("\n" + "-" * 80)
    print("PART 5: CONSECUTIVE MASS RATIOS")
    print("-" * 80)
    
    print("\n  If particles are specific levels in the spectrum,")
    print("  consecutive mass ratios = mass(n)/mass(n+1)")
    
    mass_ratios_consec = masses[:-1] / masses[1:]
    
    print(f"\n  {'Level':>5} {'Mass':>10} {'Ratio to next':>15} {'Known match':>20}")
    print("  " + "-" * 55)
    
    # Known ratios to look for
    known = [
        ('d/u', 2.14), ('s/d', 20.2), ('c/s', 13.4),
        ('b/c', 3.28), ('t/b', 41.4),
        ('μ/e', 206.77), ('τ/μ', 16.82),
        ('p/e', 1836.15), ('W/p', 85.7),
    ]
    
    for i in range(min(20, len(mass_ratios_consec))):
        r = mass_ratios_consec[i]
        
        # Find closest known ratio
        closest = min(known, key=lambda k: abs(np.log(r) - np.log(k[1])))
        log_diff = abs(np.log(r) - np.log(closest[1]))
        
        match_str = f"{closest[0]}={closest[1]:.1f}" if log_diff < 0.5 else ""
        
        print(f"  {i:5d} {masses[i]:10.4f} {r:15.4f} {match_str:>20}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    WHAT WE TESTED:
    - 6 topology types × multiple parameter sets
    - Numerical optimization of power-law topology
    - Lepton mass ratio matching (1 : 206.8 : 3477.2)
    
    KEY QUESTION: Can the bound state spectrum of a PAC-motivated
    potential reproduce known particle mass ratios?
    
    The answer determines whether the quantum-lock interpretation
    of particles is quantitatively viable, not just qualitatively
    suggestive.
    
    NEXT STEPS based on results:
    - If good match: derive the topology from first principles
    - If approximate: identify what additional physics is needed
    - If poor match: the 1D model may need to be extended to
      capture the full PAC tree structure
    """)
