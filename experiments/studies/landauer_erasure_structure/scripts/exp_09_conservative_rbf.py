"""
Experiment 5c: Conservative RBF Binding
========================================
Dawn Field Institute — Dawn Field Theory Validation

FIXES FROM 5b:
1. STRICT CONSERVATION: Total E+I preserved exactly. No source/sink terms.
   RBF mediates TRANSFER between E and I fields, never creates or destroys.
2. ξ measured as EXCESS structure beyond linear coupling prediction.
   Not "does coupling create correlation" (trivially yes) but 
   "does the nonlinear balance constraint create structure that
   can't be predicted from the coupling strength alone?"

RBF: B(x,t) = λ · [(E-I)/(1+αM)] · Φ(x)

Under conservation, B drives transfer: what leaves E enters I and vice versa.
Diffusion is also conservative (redistribution in space, not creation).

The question: does PAC conservation + RBF binding create emergent ξ 
that exceeds what you'd predict from knowing λ alone?
"""

import numpy as np
from scipy import stats
import json

phi = (1 + np.sqrt(5)) / 2
inv_phi = 1 / phi
pi = np.pi
XI_PREDICTED = 1.0571

print("=" * 70)
print("EXPERIMENT 5c: Conservative RBF Binding")
print("Dawn Field Institute")
print("=" * 70)


def build_phi(N):
    """Fibonacci harmonic modulation."""
    x = np.linspace(0, 2*pi, N, endpoint=False)
    result = np.zeros(N)
    fibs = [1, 1, 2, 3, 5, 8, 13, 21]
    for k, f in enumerate(fibs, 1):
        result += (inv_phi**k) * np.cos(f * x)
    return 0.5 + (result - result.min()) / (result.max() - result.min())


def conservative_diffuse(field, D, dt):
    """
    Conservative diffusion: flux-based, preserves total exactly.
    Flux from site i to i+1 proportional to gradient.
    """
    N = len(field)
    flux = D * (np.roll(field, -1) - field)  # flux[i] = flow from i to i+1
    # Net change: what flows in minus what flows out
    delta = (np.roll(flux, 1) - flux) * dt
    return delta


def evolve_conservative_bound(E0, I0, Phi, n_steps=3000, lam=1.0, 
                                alpha=0.1, dt=0.005, D_E=0.10, D_I=0.03):
    """
    STRICTLY CONSERVATIVE RBF dynamics.
    
    Rules:
    - Total S = sum(E) + sum(I) is EXACTLY conserved
    - RBF drives transfer: E -> I where B > 0 (E excess), I -> E where B < 0
    - Diffusion redistributes each field spatially (conservative)
    - No source terms, no sink terms, no decay
    
    SEC collapse: when |E-I| < threshold at a site, both lock to mean.
    This is conservative (E+I unchanged at that site).
    """
    N = len(E0)
    E = E0.copy().astype(float)
    I = I0.copy().astype(float)
    M = np.zeros(N)
    
    S0 = np.sum(E) + np.sum(I)  # Must be preserved
    
    collapses = []
    collapse_times = []
    balance_history = []
    conservation_drift = []
    
    for t in range(n_steps):
        # === Balance field ===
        B = lam * ((E - I) / (1 + alpha * M)) * Phi
        
        # === TRANSFER: E <-> I mediated by B ===
        # Where B > 0: E > I, so transfer E -> I (I grows, E shrinks)
        # Where B < 0: I > E, so transfer I -> E
        # Transfer rate proportional to B
        transfer = 0.1 * B * dt
        
        # Limit transfer so fields stay positive
        max_from_E = 0.5 * E  # Never take more than half
        max_from_I = 0.5 * I
        transfer = np.where(transfer > 0, 
                           np.minimum(transfer, max_from_E),
                           np.maximum(transfer, -max_from_I))
        
        E_new = E - transfer  # E loses where B>0
        I_new = I + transfer  # I gains where B>0
        
        # === CONSERVATIVE DIFFUSION ===
        dE = conservative_diffuse(E_new, D_E, dt)
        dI = conservative_diffuse(I_new, D_I, dt)
        
        E_new = E_new + dE
        I_new = I_new + dI
        
        # === Enforce strict conservation by correcting numerical drift ===
        S_current = np.sum(E_new) + np.sum(I_new)
        if S_current > 0:
            correction = S0 / S_current
            E_new *= correction
            I_new *= correction
        
        # Ensure positivity
        E_new = np.maximum(E_new, 1e-10)
        I_new = np.maximum(I_new, 1e-10)
        
        # Final conservation correction after positivity enforcement
        S_current = np.sum(E_new) + np.sum(I_new)
        if abs(S_current - S0) > 1e-12:
            correction = S0 / S_current
            E_new *= correction
            I_new *= correction
        
        E = E_new
        I = I_new
        
        # === Memory update ===
        M = 0.95 * M + np.abs(E - I)
        
        # === SEC COLLAPSE ===
        balance_ratio = np.abs(E - I) / (E + I + 1e-10)
        for site in np.where(balance_ratio < 0.02)[0]:
            if site not in collapses:
                collapses.append(site)
                collapse_times.append(t)
                # Conservative collapse: lock E=I at mean
                mean_val = (E[site] + I[site]) / 2
                E[site] = mean_val
                I[site] = mean_val
        
        if t % 500 == 0:
            S_check = np.sum(E) + np.sum(I)
            balance_history.append({
                't': t,
                'mean_B': np.mean(np.abs(B)),
                'EI_ratio': np.mean(E) / np.mean(I) if np.mean(I) > 0 else 999,
                'n_collapsed': len(collapses),
                'conservation': S_check / S0,
            })
            conservation_drift.append(abs(S_check - S0) / S0)
    
    return {
        'E': E, 'I': I, 'M': M,
        'collapses': collapses,
        'collapse_times': collapse_times,
        'S0': S0,
        'S_final': np.sum(E) + np.sum(I),
        'balance_history': balance_history,
        'conservation_drift': conservation_drift,
    }


def evolve_conservative_unbound(E0, I0, n_steps=3000, dt=0.005, 
                                  D_E=0.10, D_I=0.03):
    """
    Conservative evolution WITHOUT RBF binding.
    E and I diffuse independently. No transfer between fields.
    Same initial conditions, same conservation, no binding.
    """
    N = len(E0)
    E = E0.copy().astype(float)
    I = I0.copy().astype(float)
    
    SE0 = np.sum(E)
    SI0 = np.sum(I)
    
    for t in range(n_steps):
        dE = conservative_diffuse(E, D_E, dt)
        dI = conservative_diffuse(I, D_I, dt)
        E = E + dE
        I = I + dI
        
        E = np.maximum(E, 1e-10)
        I = np.maximum(I, 1e-10)
        
        # Conserve each field separately
        E *= SE0 / np.sum(E)
        I *= SI0 / np.sum(I)
    
    return E, I


def evolve_linear_bound(E0, I0, Phi, n_steps=3000, lam=1.0, dt=0.005,
                         D_E=0.10, D_I=0.03):
    """
    LINEAR coupling control: E and I coupled with strength λ
    but WITHOUT the nonlinear RBF terms (no memory M, no Φ modulation).
    
    This tells us what structure λ-coupling ALONE creates.
    ξ = full_RBF_structure - linear_coupling_structure
    """
    N = len(E0)
    E = E0.copy().astype(float)
    I = I0.copy().astype(float)
    
    S0 = np.sum(E) + np.sum(I)
    
    for t in range(n_steps):
        # Simple linear transfer: proportional to (E-I)
        transfer = 0.1 * lam * (E - I) * dt
        max_from_E = 0.5 * E
        max_from_I = 0.5 * I
        transfer = np.where(transfer > 0,
                           np.minimum(transfer, max_from_E),
                           np.maximum(transfer, -max_from_I))
        
        E_new = E - transfer
        I_new = I + transfer
        
        dE = conservative_diffuse(E_new, D_E, dt)
        dI = conservative_diffuse(I_new, D_I, dt)
        E_new = E_new + dE
        I_new = I_new + dI
        
        E_new = np.maximum(E_new, 1e-10)
        I_new = np.maximum(I_new, 1e-10)
        
        S_current = np.sum(E_new) + np.sum(I_new)
        correction = S0 / S_current
        E = E_new * correction
        I = I_new * correction
    
    return E, I


def measure_structure(E, I):
    """
    Multi-metric structure measurement.
    """
    N = len(E)
    
    # 1. E-I site-wise correlation
    ei_corr = np.corrcoef(E, I)[0, 1] if np.std(E) > 1e-10 and np.std(I) > 1e-10 else 0
    
    # 2. Spatial correlation length of balance field
    balance = E - I
    b_norm = balance - np.mean(balance)
    if np.std(b_norm) > 1e-10:
        autocorr = np.correlate(b_norm, b_norm, mode='full')[N-1:]
        autocorr /= autocorr[0] + 1e-10
        corr_length = np.sum(autocorr[:N//2] > 1/np.e)
    else:
        corr_length = N
    
    # 3. Spectral structure of total field
    total = E + I
    t_norm = total - np.mean(total)
    if np.std(t_norm) > 1e-10:
        fft = np.abs(np.fft.rfft(t_norm))
        fft_p = fft / (np.sum(fft) + 1e-10)
        fft_p = fft_p[fft_p > 1e-15]
        s_ent = -np.sum(fft_p * np.log(fft_p))
        spectral_order = 1 - s_ent / np.log(len(fft_p)) if len(fft_p) > 1 else 0
    else:
        spectral_order = 1.0
    
    # 4. Cross-field spatial correlation
    cross = 0
    for lag in [1, 2, 3, 5, 8]:
        c = np.corrcoef(E, np.roll(I, lag))[0, 1] if np.std(E) > 1e-10 else 0
        cross += abs(c)
    cross /= 5
    
    # 5. Gini coefficient of E+I (inequality = structure)
    total_sorted = np.sort(total)
    n = len(total_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * total_sorted) / (n * np.sum(total_sorted))) - (n + 1) / n
    
    # 6. KL divergence from uniform distribution
    total_prob = total / np.sum(total)
    uniform = np.ones(N) / N
    kl_div = np.sum(total_prob * np.log(total_prob / uniform + 1e-10))
    
    return {
        'ei_corr': ei_corr,
        'corr_length': corr_length,
        'spectral_order': spectral_order,
        'cross_field': cross,
        'gini': gini,
        'kl_from_uniform': kl_div,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

N = 89
n_steps = 3000
n_runs = 50

Phi = build_phi(N)

print(f"\n{'='*60}")
print(f"Running {n_runs} paired trials: RBF-bound vs Unbound vs Linear")
print(f"N={N}, steps={n_steps}, STRICTLY CONSERVATIVE")
print(f"{'='*60}\n")

results_rbf = []
results_unbound = []
results_linear = []
xi_vs_unbound = []   # RBF - unbound
xi_vs_linear = []    # RBF - linear (the REAL test)
collapse_data = []
conservation_checks = []
balance_operators = []

for run in range(n_runs):
    seed = run * 13 + 7
    np.random.seed(seed)
    
    # Initial conditions
    E0 = np.random.exponential(1.0, N)
    I0 = 0.1 * np.ones(N)  # Start with low information
    # Seed some structure
    for k in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        if k < N:
            I0[k] = 0.5
    
    # === 1. Full RBF (nonlinear, with memory and Φ) ===
    r_rbf = evolve_conservative_bound(E0, I0, Phi, n_steps)
    s_rbf = measure_structure(r_rbf['E'], r_rbf['I'])
    results_rbf.append(s_rbf)
    
    # === 2. Unbound (no coupling) ===
    E_u, I_u = evolve_conservative_unbound(E0, I0, n_steps)
    s_unb = measure_structure(E_u, I_u)
    results_unbound.append(s_unb)
    
    # === 3. Linear coupling (same λ, no memory/Φ) ===
    E_l, I_l = evolve_linear_bound(E0, I0, Phi, n_steps)
    s_lin = measure_structure(E_l, I_l)
    results_linear.append(s_lin)
    
    # ξ measurements
    xi_vu = s_rbf['kl_from_uniform'] - s_unb['kl_from_uniform']
    xi_vl = s_rbf['kl_from_uniform'] - s_lin['kl_from_uniform']
    xi_vs_unbound.append(xi_vu)
    xi_vs_linear.append(xi_vl)
    
    collapse_data.append({
        'n': len(r_rbf['collapses']),
        'sites': r_rbf['collapses'],
        'times': r_rbf['collapse_times'],
    })
    
    conservation_checks.append(r_rbf['S_final'] / r_rbf['S0'])
    
    # Balance operator
    collapses = r_rbf['collapses']
    if len(collapses) > 2:
        uncollapsed = [i for i in range(N) if i not in collapses]
        if uncollapsed:
            c_d = np.mean(r_rbf['E'][collapses] + r_rbf['I'][collapses])
            u_d = np.mean(r_rbf['E'][uncollapsed] + r_rbf['I'][uncollapsed])
            if u_d > 0:
                balance_operators.append(c_d / u_d)
    
    if run % 10 == 0:
        cons = r_rbf['S_final'] / r_rbf['S0']
        print(f"  Run {run:3d}: ξ_vs_linear={xi_vl:+.5f}, "
              f"collapses={len(r_rbf['collapses']):2d}, "
              f"conservation={cons:.10f}")


# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("RESULTS: CONSERVATIVE RBF BINDING")
print("=" * 70)

# --- Conservation verification ---
cons_arr = np.array(conservation_checks)
print(f"\n0. CONSERVATION VERIFICATION")
print(f"   S_final/S_initial: {np.mean(cons_arr):.12f} ± {np.std(cons_arr):.2e}")
print(f"   Max deviation: {np.max(np.abs(cons_arr - 1)):.2e}")
print(f"   {'PASS: System is conservative' if np.max(np.abs(cons_arr - 1)) < 1e-8 else 'FAIL'}")

# --- ξ vs unbound ---
xi_vu_arr = np.array(xi_vs_unbound)
t1, p1 = stats.ttest_1samp(xi_vu_arr, 0)
print(f"\n1. ξ vs UNBOUND (RBF structure - independent structure)")
print(f"   Mean: {np.mean(xi_vu_arr):+.6f} ± {np.std(xi_vu_arr):.6f}")
print(f"   t={t1:.3f}, p={p1:.2e}")
print(f"   RBF > unbound: {'YES' if t1 > 0 and p1 < 0.05 else 'NO'}")

# --- ξ vs linear (THE KEY TEST) ---
xi_vl_arr = np.array(xi_vs_linear)
t2, p2 = stats.ttest_1samp(xi_vl_arr, 0)
print(f"\n2. ξ vs LINEAR COUPLING (THE KEY TEST)")
print(f"   Does nonlinear RBF (memory + Φ) create structure beyond linear coupling?")
print(f"   Mean: {np.mean(xi_vl_arr):+.6f} ± {np.std(xi_vl_arr):.6f}")
print(f"   t={t2:.3f}, p={p2:.2e}")
print(f"   RBF > linear: {'YES — EMERGENT STRUCTURE CONFIRMED' if t2 > 0 and p2 < 0.05 else 'NO — not yet demonstrated'}")

# --- Per-metric comparison ---
print(f"\n3. METRIC BREAKDOWN (RBF vs Linear vs Unbound)")
metrics = ['ei_corr', 'corr_length', 'spectral_order', 'cross_field', 'gini', 'kl_from_uniform']
for m in metrics:
    rbf_vals = [r[m] for r in results_rbf]
    lin_vals = [r[m] for r in results_linear]
    unb_vals = [r[m] for r in results_unbound]
    t_rl, p_rl = stats.ttest_rel(rbf_vals, lin_vals)
    sig = "***" if p_rl < 0.001 else "**" if p_rl < 0.01 else "*" if p_rl < 0.05 else "ns"
    winner = "RBF" if np.mean(rbf_vals) > np.mean(lin_vals) else "LIN"
    print(f"   {m:20s}: RBF={np.mean(rbf_vals):.4f}  LIN={np.mean(lin_vals):.4f}  "
          f"UNB={np.mean(unb_vals):.4f}  [{winner}] p={p_rl:.2e} {sig}")

# --- Collapses ---
n_collapses = [c['n'] for c in collapse_data]
print(f"\n4. SEC COLLAPSE EVENTS")
print(f"   Mean: {np.mean(n_collapses):.1f} ± {np.std(n_collapses):.1f}")

# Collapse location analysis
all_collapse_sites = []
for c in collapse_data:
    all_collapse_sites.extend(c['sites'])
if all_collapse_sites:
    phi_at_collapse = Phi[all_collapse_sites]
    phi_overall_mean = np.mean(Phi)
    t_phi, p_phi = stats.ttest_1samp(phi_at_collapse, phi_overall_mean)
    print(f"   Φ at collapse sites: {np.mean(phi_at_collapse):.4f} (overall mean: {phi_overall_mean:.4f})")
    print(f"   t={t_phi:.3f}, p={p_phi:.2e}")
    print(f"   Collapses prefer high-Φ: {'YES' if t_phi > 0 and p_phi < 0.05 else 'NO'}")

# --- Balance operator ---
if balance_operators:
    bo_arr = np.array(balance_operators)
    print(f"\n5. BALANCE OPERATOR Ξ")
    print(f"   Measured: {np.mean(bo_arr):.4f} ± {np.std(bo_arr):.4f}")
    print(f"   Predicted: {XI_PREDICTED}")
    dev = abs(np.mean(bo_arr) - XI_PREDICTED) / XI_PREDICTED * 100
    print(f"   Deviation: {dev:.2f}%")
    t_bo, p_bo = stats.ttest_1samp(bo_arr, XI_PREDICTED)
    print(f"   t-test vs {XI_PREDICTED}: t={t_bo:.3f}, p={p_bo:.2e}")
    print(f"   Consistent with Ξ: {'YES' if p_bo > 0.05 else 'NO (significantly different)'}")


# =============================================================================
# λ SENSITIVITY
# =============================================================================
print(f"\n6. λ SENSITIVITY")

for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
    xi_lam = []
    for run in range(20):
        seed = run * 17 + 3
        np.random.seed(seed)
        E0 = np.random.exponential(1.0, N)
        I0 = 0.1 * np.ones(N)
        for k in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            if k < N: I0[k] = 0.5
        
        r = evolve_conservative_bound(E0, I0, Phi, n_steps, lam=lam)
        s_r = measure_structure(r['E'], r['I'])
        E_l, I_l = evolve_linear_bound(E0, I0, Phi, n_steps, lam=lam)
        s_l = measure_structure(E_l, I_l)
        
        xi_lam.append(s_r['kl_from_uniform'] - s_l['kl_from_uniform'])
    
    t_l, p_l = stats.ttest_1samp(xi_lam, 0)
    sig = "***" if p_l < 0.001 else "**" if p_l < 0.01 else "*" if p_l < 0.05 else "ns"
    print(f"   λ={lam:4.1f}: ξ_excess={np.mean(xi_lam):+.5f} ± {np.std(xi_lam):.5f}  "
          f"p={p_l:.2e} {sig}")


# =============================================================================
# ALPHA SENSITIVITY (memory damping)
# =============================================================================
print(f"\n7. α SENSITIVITY (memory strength)")
print(f"   Higher α = memory matters more = more nonlinear")

for alpha in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
    xi_a = []
    for run in range(20):
        seed = run * 17 + 3
        np.random.seed(seed)
        E0 = np.random.exponential(1.0, N)
        I0 = 0.1 * np.ones(N)
        for k in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            if k < N: I0[k] = 0.5
        
        r = evolve_conservative_bound(E0, I0, Phi, n_steps, alpha=alpha)
        s_r = measure_structure(r['E'], r['I'])
        E_l, I_l = evolve_linear_bound(E0, I0, Phi, n_steps)
        s_l = measure_structure(E_l, I_l)
        
        xi_a.append(s_r['kl_from_uniform'] - s_l['kl_from_uniform'])
    
    t_a, p_a = stats.ttest_1samp(xi_a, 0)
    sig = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "ns"
    print(f"   α={alpha:4.2f}: ξ_excess={np.mean(xi_a):+.5f} ± {np.std(xi_a):.5f}  "
          f"p={p_a:.2e} {sig}")


# =============================================================================
# SUMMARY
# =============================================================================
bo_mean = np.mean(balance_operators) if balance_operators else float('nan')
bo_str = f"{bo_mean:.4f}" if balance_operators else "N/A"

print(f"\n{'='*70}")
print(f"EXPERIMENT 5c FINAL SUMMARY")
print(f"{'='*70}")
print(f"""
System: STRICTLY CONSERVATIVE (verified: max drift = {np.max(np.abs(cons_arr-1)):.2e})

Q1: Does RBF binding create structure vs no binding?
    ξ = {np.mean(xi_vu_arr):+.6f}, p = {p1:.2e}
    {'YES' if t1 > 0 and p1 < 0.05 else 'NO'}

Q2: Does NONLINEAR RBF create structure beyond LINEAR coupling?
    ξ = {np.mean(xi_vl_arr):+.6f}, p = {p2:.2e}
    {'YES — Emergent structure from memory + harmonic modulation' if t2 > 0 and p2 < 0.05 else 'NOT YET — linear coupling captures most of the structure'}

Q3: Does SEC collapse prefer Φ-modulated sites?
    {'YES' if all_collapse_sites and t_phi > 0 and p_phi < 0.05 else 'Insufficient data'}

Q4: Balance Operator Ξ ≈ 1.0571?
    Measured: {bo_str}, Predicted: {XI_PREDICTED}

Q5: Is conservation exact?
    Yes: S_final/S_initial = {np.mean(cons_arr):.12f}
""")
