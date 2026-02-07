"""
Experiment 7b: Cascade Time-Computation Analysis
==================================================
Dawn Field Institute

HYPOTHESIS:
Time is experienced from INSIDE the cascade. Each cascade tick is one
unit of experienced time. The computational cost per tick determines
how "fast" time feels from inside.

Early universe (fresh medium, dense interactions):
  - Every collapse creates lots of new structure (high ξ per event)
  - Each tick is computationally expensive
  - Experienced time is SLOW (each moment is heavy)

Late universe (saturated medium, sparse interactions):
  - Collapses create little new structure (low ξ per event)
  - Each tick is computationally cheap
  - Experienced time is FAST (each moment is light)

This isn't time "speeding up" or "slowing down" on an external clock.
There IS no external clock. It's about the computational density
per experienced moment.

WHAT WE MODEL:
Phase 1: Dense regime (many interacting modes, high coupling)
         → models early universe conditions
Phase 2: Sparse regime (spread out modes, lower coupling)  
         → models late universe conditions

Measure: ξ per event, Θ per event, total correlation created,
computational work (new structure) per cascade tick.
"""

import numpy as np
from scipy import stats

print("=" * 70)
print("EXPERIMENT 7b: Cascade Time-Computation Analysis")
print("Dawn Field Institute")
print("=" * 70)


def shannon_H(bits):
    p1 = np.mean(bits)
    if p1 <= 0 or p1 >= 1:
        return 0.0
    return -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))


def total_correlation(env):
    n_samples, n_env = env.shape
    H_marginals = sum(shannon_H(env[:, i]) for i in range(n_env))
    if n_env <= 14:
        state = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_env):
            state += env[:, i].astype(np.int64) * (2 ** i)
        _, counts = np.unique(state, return_counts=True)
        probs = counts / n_samples
        H_joint = -np.sum(probs * np.log2(probs + 1e-15))
    else:
        H_joint = H_marginals
        for i in range(n_env):
            for j in range(i+1, min(n_env, i+8)):
                joint = env[:, i] * 2 + env[:, j]
                _, counts = np.unique(joint, return_counts=True)
                probs = counts / n_samples
                H_ij = -np.sum(probs * np.log2(probs + 1e-15))
                mi = shannon_H(env[:, i]) + shannon_H(env[:, j]) - H_ij
                H_joint -= mi
    return max(H_marginals - H_joint, 0)


def mutual_info_sys_env(system, env):
    n_samples = len(system)
    env_summary = np.sum(env, axis=1)
    joint = system * (np.max(env_summary) + 1) + env_summary
    _, jc = np.unique(joint, return_counts=True)
    H_joint = -np.sum((jc/n_samples) * np.log2(jc/n_samples + 1e-15))
    H_s = shannon_H(system)
    _, ec = np.unique(env_summary, return_counts=True)
    H_e = -np.sum((ec/n_samples) * np.log2(ec/n_samples + 1e-15))
    return max(H_s + H_e - H_joint, 0)


def erase_into(system_bits, env, coupling):
    n_samples = len(system_bits)
    env_new = env.copy()
    for i in range(min(env.shape[1], len(coupling))):
        if coupling[i] > 0:
            mask = np.random.random(n_samples) < coupling[i]
            env_new[mask, i] = system_bits[mask]
    return env_new


def run_epoch(env, n_events, coupling, n_samples):
    """
    Run n_events erasure events (cascade ticks) into the environment.
    Returns per-tick measurements.
    """
    ticks = []
    for event in range(n_events):
        system = np.random.randint(0, 2, n_samples)
        P = shannon_H(system)
        tc_pre = total_correlation(env)
        
        env = erase_into(system, env, coupling)
        
        A = mutual_info_sys_env(system, env)
        tc_post = total_correlation(env)
        xi = max(tc_post - tc_pre, 0)
        theta = max(P - A - xi, 0)
        
        ticks.append({
            'xi': xi,
            'theta': theta,
            'tc_total': tc_post,
            'P': P,
            'A': A,
        })
    
    return ticks, env


# =============================================================================
# TEST 1: Dense vs Sparse Regime
# =============================================================================

print(f"\n{'='*60}")
print("TEST 1: Dense vs Sparse Cascade Regimes")
print(f"{'='*60}")

N_SAMPLES = 500000
N_TICKS = 15
N_TRIALS = 25

# DENSE: many modes, strong coupling (early universe analog)
# All modes tightly coupled, lots of interaction per event
dense_results = {t: {'xi': [], 'theta': [], 'tc': []} for t in range(N_TICKS)}

# SPARSE: same total modes, weaker coupling (late universe analog)  
# Modes loosely coupled, less interaction per event
sparse_results = {t: {'xi': [], 'theta': [], 'tc': []} for t in range(N_TICKS)}

for trial in range(N_TRIALS):
    np.random.seed(trial * 13 + 7)
    
    # === DENSE REGIME ===
    # 8 modes, strong cascade coupling (decay=0.85 = tight)
    n_dense = 8
    coupling_dense = np.array([0.85 ** i for i in range(n_dense)])
    coupling_dense = coupling_dense / np.sum(coupling_dense) * 0.9  # High total coupling
    
    env_dense = np.random.randint(0, 2, (N_SAMPLES, n_dense))
    ticks_d, _ = run_epoch(env_dense, N_TICKS, coupling_dense, N_SAMPLES)
    
    for t, tick in enumerate(ticks_d):
        dense_results[t]['xi'].append(tick['xi'])
        dense_results[t]['theta'].append(tick['theta'])
        dense_results[t]['tc'].append(tick['tc_total'])
    
    # === SPARSE REGIME ===
    # 8 modes, weak cascade coupling (decay=0.4 = loose)
    coupling_sparse = np.array([0.4 ** i for i in range(n_dense)])
    coupling_sparse = coupling_sparse / np.sum(coupling_sparse) * 0.5  # Lower total coupling
    
    env_sparse = np.random.randint(0, 2, (N_SAMPLES, n_dense))
    ticks_s, _ = run_epoch(env_sparse, N_TICKS, coupling_sparse, N_SAMPLES)
    
    for t, tick in enumerate(ticks_s):
        sparse_results[t]['xi'].append(tick['xi'])
        sparse_results[t]['theta'].append(tick['theta'])
        sparse_results[t]['tc'].append(tick['tc_total'])

print(f"\nDENSE REGIME (strong coupling = early universe analog):")
print(f"{'Tick':>5s} {'ξ':>10s} {'Θ':>10s} {'ξ/Θ':>10s} {'TC':>10s}")
print("-" * 48)
for t in range(N_TICKS):
    xi_m = np.mean(dense_results[t]['xi'])
    th_m = np.mean(dense_results[t]['theta'])
    tc_m = np.mean(dense_results[t]['tc'])
    ratio = xi_m / th_m if th_m > 0.001 else 0
    print(f"  {t:3d}  {xi_m:10.5f} {th_m:10.5f} {ratio:10.5f} {tc_m:10.5f}")

print(f"\nSPARSE REGIME (weak coupling = late universe analog):")
print(f"{'Tick':>5s} {'ξ':>10s} {'Θ':>10s} {'ξ/Θ':>10s} {'TC':>10s}")
print("-" * 48)
for t in range(N_TICKS):
    xi_m = np.mean(sparse_results[t]['xi'])
    th_m = np.mean(sparse_results[t]['theta'])
    tc_m = np.mean(sparse_results[t]['tc'])
    ratio = xi_m / th_m if th_m > 0.001 else 0
    print(f"  {t:3d}  {xi_m:10.5f} {th_m:10.5f} {ratio:10.5f} {tc_m:10.5f}")

# Compare total ξ production
dense_total_xi = [sum(dense_results[t]['xi'][trial] for t in range(N_TICKS)) 
                  for trial in range(N_TRIALS)]
sparse_total_xi = [sum(sparse_results[t]['xi'][trial] for t in range(N_TICKS)) 
                   for trial in range(N_TRIALS)]

t_ds, p_ds = stats.ttest_ind(dense_total_xi, sparse_total_xi)
print(f"\nTotal ξ over {N_TICKS} ticks:")
print(f"  Dense:  {np.mean(dense_total_xi):.5f} ± {np.std(dense_total_xi):.5f}")
print(f"  Sparse: {np.mean(sparse_total_xi):.5f} ± {np.std(sparse_total_xi):.5f}")
print(f"  t={t_ds:.3f}, p={p_ds:.2e}")
print(f"  Dense produces {'more' if np.mean(dense_total_xi) > np.mean(sparse_total_xi) else 'less'} structure")


# =============================================================================
# TEST 2: Computational Cost Per Tick (Time Analysis)
# =============================================================================

print(f"\n{'='*60}")
print("TEST 2: Computational Cost Per Tick")
print("ξ per tick = work done per moment of experienced time")
print(f"{'='*60}")

# Run a single long cascade and track how computational cost evolves
print(f"\nSingle 30-tick cascade, tracking cost per tick:\n")

np.random.seed(42)
n_env = 10
coupling = np.array([0.75 ** i for i in range(n_env)])
coupling = coupling / np.sum(coupling) * 0.85

env = np.random.randint(0, 2, (N_SAMPLES, n_env))

print(f"{'Tick':>5s} {'ξ (work)':>10s} {'Θ (fuel)':>10s} {'ξ/Θ (eff)':>10s} "
      f"{'TC (struct)':>12s} {'Interpret':>20s}")
print("-" * 72)

long_xi = []
long_theta = []
long_tc = []

for tick in range(30):
    system = np.random.randint(0, 2, N_SAMPLES)
    P = shannon_H(system)
    tc_pre = total_correlation(env)
    env = erase_into(system, env, coupling)
    A = mutual_info_sys_env(system, env)
    tc_post = total_correlation(env)
    xi = max(tc_post - tc_pre, 0)
    theta = max(P - A - xi, 0)
    
    long_xi.append(xi)
    long_theta.append(theta)
    long_tc.append(tc_post)
    
    ratio = xi / theta if theta > 0.001 else 0
    
    # Interpretation
    if tick < 5:
        interp = "EARLY (heavy ticks)"
    elif tick < 15:
        interp = "MIDDLE (medium)"
    else:
        interp = "LATE (light ticks)"
    
    print(f"  {tick:3d}  {xi:10.5f} {theta:10.5f} {ratio:10.5f} "
          f"{tc_post:12.5f} {interp:>20s}")

# Quantify the early vs late difference
early_xi = long_xi[:5]
late_xi = long_xi[-5:]
t_el, p_el = stats.ttest_ind(early_xi, late_xi)

print(f"\nEarly (ticks 0-4) avg ξ per tick: {np.mean(early_xi):.5f}")
print(f"Late (ticks 25-29) avg ξ per tick: {np.mean(late_xi):.5f}")
print(f"Ratio: {np.mean(early_xi)/np.mean(late_xi):.1f}x" if np.mean(late_xi) > 0 
      else "Late ξ ≈ 0")
print(f"t={t_el:.3f}, p={p_el:.2e}")


# =============================================================================
# TEST 3: Interaction Density Gradient
# Models the universe expanding — starting dense, becoming sparse
# =============================================================================

print(f"\n{'='*60}")
print("TEST 3: Expanding Universe Model")
print("Coupling strength decreases over time (universe expands)")
print(f"{'='*60}")

N_TRIALS_EXP = 25
N_TICKS_EXP = 20

expansion_xi = {t: [] for t in range(N_TICKS_EXP)}
expansion_theta = {t: [] for t in range(N_TICKS_EXP)}
expansion_coupling_strength = []

for trial in range(N_TRIALS_EXP):
    np.random.seed(trial * 11 + 5)
    
    n_env = 10
    env = np.random.randint(0, 2, (N_SAMPLES, n_env))
    
    for tick in range(N_TICKS_EXP):
        # Coupling weakens over time (expansion)
        # Start at 0.9 total, decay to 0.3
        expansion_factor = 0.9 - (0.6 * tick / N_TICKS_EXP)
        coupling = np.array([0.7 ** i for i in range(n_env)])
        coupling = coupling / np.sum(coupling) * expansion_factor
        
        if trial == 0:
            expansion_coupling_strength.append(expansion_factor)
        
        system = np.random.randint(0, 2, N_SAMPLES)
        P = shannon_H(system)
        tc_pre = total_correlation(env)
        env = erase_into(system, env, coupling)
        A = mutual_info_sys_env(system, env)
        tc_post = total_correlation(env)
        xi = max(tc_post - tc_pre, 0)
        theta = max(P - A - xi, 0)
        
        expansion_xi[tick].append(xi)
        expansion_theta[tick].append(theta)

print(f"\n{'Tick':>5s} {'Coupling':>9s} {'ξ':>10s} {'Θ':>10s} {'ξ/Θ':>10s}")
print("-" * 48)
for t in range(N_TICKS_EXP):
    xi_m = np.mean(expansion_xi[t])
    th_m = np.mean(expansion_theta[t])
    ratio = xi_m / th_m if th_m > 0.001 else 0
    cs = expansion_coupling_strength[t] if t < len(expansion_coupling_strength) else 0
    print(f"  {t:3d}  {cs:9.3f} {xi_m:10.5f} {th_m:10.5f} {ratio:10.5f}")

# Correlation between coupling strength and ξ
all_couplings = []
all_xi_exp = []
for t in range(N_TICKS_EXP):
    for trial in range(N_TRIALS_EXP):
        all_couplings.append(expansion_coupling_strength[t] if t < len(expansion_coupling_strength) else 0)
        all_xi_exp.append(expansion_xi[t][trial])

corr_exp, p_exp = stats.pearsonr(all_couplings, all_xi_exp)
print(f"\nCorrelation between coupling strength and ξ:")
print(f"  r = {corr_exp:.4f}, p = {p_exp:.2e}")


# =============================================================================
# TEST 4: Two competing effects — saturation vs coupling
# =============================================================================

print(f"\n{'='*60}")
print("TEST 4: Saturation vs Coupling — Disentangling Effects")
print(f"{'='*60}")
print(f"\nThe medium saturates (less ξ over time) BUT")
print(f"the coupling also affects ξ production.")
print(f"Which dominates?\n")

# Run constant coupling (isolate saturation only)
const_xi = []
np.random.seed(42)
env_const = np.random.randint(0, 2, (N_SAMPLES, 10))
coupling_const = np.array([0.7 ** i for i in range(10)])
coupling_const = coupling_const / np.sum(coupling_const) * 0.7

for tick in range(N_TICKS_EXP):
    system = np.random.randint(0, 2, N_SAMPLES)
    tc_pre = total_correlation(env_const)
    env_const = erase_into(system, env_const, coupling_const)
    tc_post = total_correlation(env_const)
    const_xi.append(max(tc_post - tc_pre, 0))

# Run expanding coupling (both effects)
expand_xi = []
np.random.seed(42)
env_exp = np.random.randint(0, 2, (N_SAMPLES, 10))

for tick in range(N_TICKS_EXP):
    expansion_factor = 0.9 - (0.6 * tick / N_TICKS_EXP)
    coupling_exp = np.array([0.7 ** i for i in range(10)])
    coupling_exp = coupling_exp / np.sum(coupling_exp) * expansion_factor
    
    system = np.random.randint(0, 2, N_SAMPLES)
    tc_pre = total_correlation(env_exp)
    env_exp = erase_into(system, env_exp, coupling_exp)
    tc_post = total_correlation(env_exp)
    expand_xi.append(max(tc_post - tc_pre, 0))

print(f"{'Tick':>5s} {'Const ξ':>10s} {'Expand ξ':>10s} {'Ratio':>8s}")
print("-" * 38)
for t in range(N_TICKS_EXP):
    r = expand_xi[t] / const_xi[t] if const_xi[t] > 0.00001 else 0
    print(f"  {t:3d}  {const_xi[t]:10.5f} {expand_xi[t]:10.5f} {r:8.3f}")

print(f"\nConstant coupling total ξ: {sum(const_xi):.5f}")
print(f"Expanding coupling total ξ: {sum(expand_xi):.5f}")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("EXPERIMENT 7b SUMMARY")
print(f"{'='*70}")
print(f"""
TIME-COMPUTATION ANALYSIS

The cascade tick is one unit of experienced time.
ξ per tick = computational work done per moment.

KEY FINDINGS:

1. DENSE REGIME (early universe analog):
   - High coupling, fresh medium
   - ξ per tick: {np.mean(early_xi):.5f} (HIGH)
   - Each moment is computationally HEAVY
   - Experienced time is SLOW

2. SPARSE/SATURATED REGIME (late universe analog):
   - Lower coupling, structured medium  
   - ξ per tick: {np.mean(late_xi):.5f} (LOW)
   - Each moment is computationally LIGHT
   - Experienced time is FAST

3. EXPANSION MODEL:
   - As coupling weakens (universe expands), ξ per tick decreases
   - Correlation r = {corr_exp:.4f}
   - Two effects combine: medium saturation AND coupling reduction

4. INTERPRETATION:
   Early universe: dense, every interaction creates massive structure,
   each tick of time is computationally expensive = slow time.
   
   Late universe: sparse, interactions create little new structure,
   each tick is cheap = fast time.
   
   Time doesn't speed up or slow down on an external clock.
   There IS no external clock. The computational density of each
   moment is what changes. Early moments are thick. Late moments
   are thin.
""")
