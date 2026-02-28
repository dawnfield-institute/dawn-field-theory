"""
Turbulence as Landauer Cascade v3: Clean Energy-Based Model
=============================================================
Dawn Field Institute — PACSeries Exploration

KEY FIX: In v2, the ξ (bits) → energy conversion via Landauer minimum
was incommensurable. Here we work entirely in energy units.

MODEL:
- Energy P_k arrives at scale k
- It distributes across N interacting modes
- The eigenvalue structure of the mode coupling determines:
  - Organized fraction (top eigenvalue / total) → stays at this scale
  - Distributed fraction (remaining eigenvalues / total) → transfers down
- This IS the ξ/Θ split, but measured directly in energy
- The coupling matrix evolves with the cascade (nonlinearity)

The organized fraction is the PARTICIPATION RATIO of the energy
distribution across eigenvalues. High PR = energy spread evenly 
(transfers efficiently). Low PR = energy concentrated in dominant
mode (stays organized at this scale).
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

phi = (1 + np.sqrt(5)) / 2
kT = 1.0
LANDAUER_MIN = kT * np.log(2)

print("=" * 70)
print("TURBULENCE AS LANDAUER CASCADE v3")
print("Clean Energy-Based Partitioning")
print("Dawn Field Institute")
print("=" * 70)


def energy_cascade(
    injection_energy,
    n_scales,
    n_modes=8,
    n_samples=20000,
    coupling_decay=0.3,
    nonlinear_strength=0.3,
    verbose=False
):
    """
    Energy cascade where the eigenvalue structure determines energy transfer.
    
    At each scale:
    1. Energy P distributes across modes with coupling matrix C
    2. Covariance eigenvalues reveal: how much energy is ORGANIZED
    3. Organized energy = (λ_max / Σλ) × P  → stays at this scale
    4. Remaining energy = (1 - λ_max/Σλ) × P → transfers to next scale
    5. The coupling matrix for next scale inherits structure (nonlinearity)
    """
    results = []
    P = injection_energy
    cumulative_structure = 0.0
    prev_dominant = None
    
    for k_idx in range(n_scales):
        if P < 1e-18:
            results.append({
                'k_index': k_idx, 'wavenumber': 2**(k_idx+1),
                'P_input': 0, 'E_organized': 0, 'E_transfer': 0,
                'org_fraction': 0, 'alive': False,
                'cumul_structure': cumulative_structure
            })
            continue
        
        # Build coupling matrix
        C = np.zeros((n_modes, n_modes))
        for i in range(n_modes):
            for j in range(n_modes):
                C[i, j] = np.exp(-abs(i-j) * coupling_decay)
        
        # Nonlinear feedback: previous scale's dominant mode biases coupling
        if prev_dominant is not None:
            bias = np.outer(prev_dominant, prev_dominant)
            bias /= (np.max(np.abs(bias)) + 1e-15)
            C = C + bias * nonlinear_strength
        
        C = (C + C.T) / 2
        eigs_C = np.linalg.eigvalsh(C)
        if np.min(eigs_C) < 1e-10:
            C += np.eye(n_modes) * (abs(np.min(eigs_C)) + 1e-6)
        
        # Distribute energy across modes
        means = P * np.exp(-np.arange(n_modes) * coupling_decay)
        means *= P / np.sum(means)
        
        try:
            sf = P / (np.trace(C) / n_modes) * 0.2
            samples = np.abs(np.random.multivariate_normal(means, C * sf, size=n_samples))
        except:
            samples = np.random.exponential(P / n_modes, (n_samples, n_modes))
        
        # Eigenvalue analysis of the ENERGY distribution
        cov = np.cov(samples.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
        
        # Organized fraction = dominance of top eigenvalue
        total_variance = np.sum(eigenvalues)
        top_eigenvalue = eigenvalues[-1]
        organized_fraction = top_eigenvalue / total_variance
        
        # Energy partition
        E_organized = P * organized_fraction    # Stays at this scale (structure)
        E_transfer = P * (1 - organized_fraction)  # Goes to next scale (cascade)
        
        # Ensure minimum transfer (Landauer guarantee)
        if E_transfer < LANDAUER_MIN and P > LANDAUER_MIN:
            E_transfer = LANDAUER_MIN
            E_organized = P - E_transfer
        
        cumulative_structure += E_organized
        
        # Track dominant eigenvector for nonlinear feedback
        _, eigvecs = np.linalg.eigh(cov)
        prev_dominant = eigvecs[:, -1]
        
        results.append({
            'k_index': k_idx,
            'wavenumber': 2**(k_idx+1),
            'P_input': P,
            'E_organized': E_organized,
            'E_transfer': E_transfer,
            'org_fraction': organized_fraction,
            'transfer_fraction': 1 - organized_fraction,
            'participation_ratio': np.sum(eigenvalues)**2 / np.sum(eigenvalues**2),
            'alive': True,
            'cumul_structure': cumulative_structure
        })
        
        if verbose:
            print(f"  k={2**(k_idx+1):>8} | P={P:>10.6f} | org={organized_fraction:>6.4f} | "
                  f"E_stay={E_organized:>10.6f} | E_down={E_transfer:>10.6f}")
        
        # Transfer to next scale (small coupling loss)
        P = E_transfer * 0.98
    
    return results


# ============================================================
# EXPERIMENT 1: Basic Cascade
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Energy-Based Cascade Spectrum")
print("=" * 70)

print("\nBaseline cascade:")
res = energy_cascade(1.0, 25, verbose=True)

alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
if len(alive) > 6:
    k_arr = np.array([r['wavenumber'] for r in alive])
    e_arr = np.array([r['P_input'] for r in alive])
    lk = np.log10(k_arr[2:-2])
    le = np.log10(e_arr[2:-2])
    s, _, rv, _, _ = stats.linregress(lk, le)
    
    avg_org = np.mean([r['org_fraction'] for r in alive[2:-2]])
    avg_tf = np.mean([r['transfer_fraction'] for r in alive[2:-2]])
    predicted = np.log(avg_tf) / np.log(2)
    
    print(f"\n  Exponent: {s:.4f}  (target: -1.6667)")
    print(f"  R²: {rv**2:.6f}")
    print(f"  Avg organized fraction: {avg_org:.4f}")
    print(f"  Avg transfer fraction: {avg_tf:.4f}")
    print(f"  Predicted from ln(f)/ln(2): {predicted:.4f}")


# ============================================================
# EXPERIMENT 2: Parameter Sweep — coupling_decay × nonlinear_strength
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 2: 2D Parameter Sweep")
print("=" * 70)
print("Finding which (coupling_decay, nonlinear_strength) gives -5/3\n")

coupling_decays = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
nonlinear_strengths = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

best_match = {'diff': 999}
all_results = []

print(f"{'c_decay':>8} | {'nl_str':>8} | {'Exponent':>10} | {'Avg f_trans':>11} | "
      f"{'Predicted':>10} | {'Diff from -5/3':>15}")
print("-" * 75)

for cd in coupling_decays:
    for ns in nonlinear_strengths:
        res = energy_cascade(1.0, 25, coupling_decay=cd, nonlinear_strength=ns,
                           n_samples=12000)
        alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
        
        if len(alive) > 6:
            k_arr = np.array([r['wavenumber'] for r in alive])
            e_arr = np.array([r['P_input'] for r in alive])
            lk = np.log10(k_arr[2:-2])
            le = np.log10(e_arr[2:-2])
            s, _, rv, _, _ = stats.linregress(lk, le)
            
            avg_f = np.mean([r['transfer_fraction'] for r in alive[2:-2]])
            predicted = np.log(avg_f) / np.log(2)
            diff = abs(s - (-5/3))
            
            marker = " <<<" if diff < 0.15 else ""
            print(f"  {cd:>6.1f} | {ns:>6.1f} | {s:>10.4f} | {avg_f:>11.4f} | "
                  f"{predicted:>10.4f} | {diff:>15.4f}{marker}")
            
            all_results.append({
                'cd': cd, 'ns': ns, 'exponent': s, 'avg_f': avg_f,
                'predicted': predicted, 'diff': diff
            })
            
            if diff < best_match['diff']:
                best_match = {'cd': cd, 'ns': ns, 'exponent': s, 'diff': diff,
                             'avg_f': avg_f}

print(f"\n  BEST MATCH: coupling_decay={best_match['cd']}, "
      f"nonlinear={best_match['ns']}")
print(f"  Exponent: {best_match['exponent']:.4f} (target: -1.6667)")
print(f"  Transfer fraction: {best_match['avg_f']:.4f}")


# ============================================================
# EXPERIMENT 3: Mode count at best parameters
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 3: Mode Count at Best Parameters")
print("=" * 70)

if best_match['cd']:
    cd_best = best_match['cd']
    ns_best = best_match['ns']
    
    mode_counts = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    
    print(f"\nUsing coupling_decay={cd_best}, nonlinear={ns_best}")
    print(f"\n{'N_modes':>10} | {'Exponent':>10} | {'f_trans':>8} | {'Diff':>8}")
    print("-" * 45)
    
    for nm in mode_counts:
        res = energy_cascade(1.0, 25, n_modes=nm, coupling_decay=cd_best,
                           nonlinear_strength=ns_best, n_samples=12000)
        alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
        
        if len(alive) > 6:
            k_arr = np.array([r['wavenumber'] for r in alive])
            e_arr = np.array([r['P_input'] for r in alive])
            lk = np.log10(k_arr[2:-2])
            le = np.log10(e_arr[2:-2])
            s, _, _, _, _ = stats.linregress(lk, le)
            avg_f = np.mean([r['transfer_fraction'] for r in alive[2:-2]])
            
            marker = " <<<" if abs(s-(-5/3)) < 0.15 else ""
            print(f"  {nm:>8} | {s:>10.4f} | {avg_f:>8.4f} | "
                  f"{abs(s-(-5/3)):>8.4f}{marker}")


# ============================================================
# EXPERIMENT 4: Regularity — ξ bounded?
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 4: Regularity Under Extreme Injection")
print("=" * 70)

for E_inj in [1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]:
    res = energy_cascade(E_inj, 40, n_samples=8000)
    alive = [r for r in res if r['alive']]
    if alive:
        max_org = max(r['org_fraction'] for r in alive)
        min_org = min(r['org_fraction'] for r in alive)
        max_E = max(r['E_organized'] for r in alive)
        print(f"  E={E_inj:>8.0e} | org_frac: [{min_org:.4f}, {max_org:.4f}] | "
              f"max E_organized: {max_E:.4f} | steps: {len(alive)} | "
              f"{'BOUNDED' if max_org < 0.99 else 'SATURATED'}")


# ============================================================
# EXPERIMENT 5: What Transfer Fraction = -5/3 Analytically?
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 5: Analytical Check")
print("=" * 70)

target_f = 2**(-5/3)
print(f"""
For E(k) ∝ k^(-5/3) with k doubling each step:
  E(k+1)/E(k) = 2^(-5/3) = {target_f:.6f}

So the transfer fraction needed is {target_f:.6f} — about 31.5%.

This means ~68.5% of energy must stay organized at each scale.
That's the organized fraction needed: {1-target_f:.4f}

Compare to our measured organized fractions across the parameter sweep:
""")

# Show which parameter combos give org_fraction near 0.685
for r in sorted(all_results, key=lambda x: abs((1-x['avg_f']) - (1-target_f))):
    if abs((1-r['avg_f']) - (1-target_f)) < 0.05:
        print(f"  cd={r['cd']:.1f}, ns={r['ns']:.1f}: "
              f"org_frac={1-r['avg_f']:.4f} (target: {1-target_f:.4f}), "
              f"exponent={r['exponent']:.4f}")

print(f"""

PHYSICAL INTERPRETATION:
If -5/3 requires 68.5% of energy to stay organized at each scale,
that's saying: at every scale of the turbulent cascade, about 2/3
of the kinetic energy is locked into coherent vortical structure,
and about 1/3 cascades to smaller scales.

This is the energy partition that Kolmogorov's -5/3 IMPLIES but
never explicitly stated. Our framework makes it explicit:
  organized (ξ) ≈ 2/3 of energy at each scale
  transferring (Θ) ≈ 1/3 of energy at each scale

And the 2/3 : 1/3 split is suspiciously close to the 
φ decomposition: 1/φ ≈ 0.618 and 1/φ² ≈ 0.382...
or to 1 - 1/e ≈ 0.632 and 1/e ≈ 0.368.
""")


# ============================================================
# EXPERIMENT 6: Driven Steady State at Best Parameters
# ============================================================
print("=" * 70)
print("EXPERIMENT 6: Driven Steady State")
print("=" * 70)

n_scales = 20
n_drives = 200
cumul_E = np.zeros(n_scales)
cumul_org = np.zeros(n_scales)

for d in range(n_drives):
    np.random.seed(5000 + d)
    res = energy_cascade(1.0, n_scales, coupling_decay=best_match.get('cd', 0.3),
                        nonlinear_strength=best_match.get('ns', 0.3),
                        n_samples=5000)
    for r in res:
        if r['alive']:
            cumul_E[r['k_index']] += r['P_input']
            cumul_org[r['k_index']] += r['org_fraction']

avg_E = cumul_E / n_drives
avg_org = cumul_org / n_drives

k_all = np.array([2**(i+1) for i in range(n_scales)])
valid = avg_E > 1e-15

if np.sum(valid) > 6:
    lk = np.log10(k_all[valid][2:-2])
    le = np.log10(avg_E[valid][2:-2])
    s_d, _, r_d, _, _ = stats.linregress(lk, le)
    
    print(f"\n  Driven steady-state exponent: {s_d:.4f}")
    print(f"  R²: {r_d**2:.6f}")
    print(f"  Target: -1.6667")
    
    print(f"\n  {'k':>8} | {'Avg E':>12} | {'Avg org_frac':>12}")
    print(f"  {'-'*40}")
    for i in range(n_scales):
        if avg_E[i] > 1e-15:
            print(f"  {k_all[i]:>8} | {avg_E[i]:>12.6f} | {avg_org[i]:>12.4f}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY: Turbulence as Landauer Cascade")
print("=" * 70)
print(f"""
RESULTS ACROSS 3 VERSIONS:

✓ REGULARITY: ξ (organized fraction) stays bounded [0, 1] across
  8+ orders of magnitude of injection energy. The cascade CANNOT
  blow up because energy partitioning is a fraction. This is a
  genuine information-theoretic regularity argument.

? KOLMOGOROV -5/3: The exponent depends on the organized fraction.
  For exact -5/3, we need org_frac ≈ 0.685. The model produces
  various fractions depending on coupling geometry. The question
  is whether 0.685 is NATURAL or requires tuning.

? INTERMITTENCY: Weak evidence. The Monte Carlo approach averages
  out the extreme events that real turbulence produces.

? DISSIPATION SCALE: The cascade naturally terminates when energy
  drops below Landauer minimum, but the scaling needs work.

KEY INSIGHT: The Kolmogorov -5/3 exponent IS a statement about
energy partitioning: at each cascade step, fraction f ≈ 0.315
transfers to smaller scales. The PAC framework makes this explicit
as the ξ/Θ split. Whether the split ratio emerges naturally from
the coupling physics or requires the specific topology of 3D 
fluid dynamics — that's the open question.

WORTH PURSUING? YES, but the next step isn't more Monte Carlo.
The next step is ANALYTICAL: derive what organized fraction the
cascade coupling matrix must produce, and show it's constrained
to ≈ 0.685 for 3D systems. If you can show that N=3 spatial
dimensions + cascade topology → org_frac = 1 - 2^(-5/3), that's
a derivation of Kolmogorov from information theory.
""")
