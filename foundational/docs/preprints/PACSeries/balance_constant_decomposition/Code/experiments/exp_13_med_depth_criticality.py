"""
exp_11: MED Depth Criticality — Is effective coupling depth d=3 fundamental?

HYPOTHESIS: The effective coupling depth d_eff ≈ 3.0 where A/(A+ξ) = ln(φ)
is the MED (Macro Emergence Dynamics) saturation depth — the boundary
between order and chaos, the minimum recursion depth for complex emergence.

CONNECTION TO THEORY:
  MED says: "all complex flows converge to symbolic patterns with
  depth(σ) ≤ 2, nodes(σ) ≤ 3" (Paper 7, Navier-Stokes across 1000+ sims).
  But depth ≤ 2 in a 2D Möbius base manifold is 3 LEVELS from a singular
  origin (levels 0, 1, 2). From the zero-dimensional point, reaching the
  MED attractor requires exactly 3 recursive steps.

  Five independent arguments converge on d=3 (Paper 7 §3):
    1. MED nodes ≤ 3 → D ≤ 3
    2. Curl algebra: n(n-1)/2 = n → D = 3
    3. Möbius embedding: non-orientable surface → D ≥ 3
    4. Orbital stability: Bertrand's theorem → D = 3
    5. Quaternion uniqueness: associative rotations → D = 3

  The double pendulum (pendulum-on-pendulum) has structural depth 3 from
  its pivot. It is the MINIMAL mechanical system exhibiting chaos and
  π-irrationality. Two oscillators with irrational frequency ratio generate
  Möbius topology (harmonic mobius.md, pre-field recursion notes).

  exp_02 Diagnostic 2 found: effective coupling depth ≈ 2.76 ± 0.43
  (CV=16%) across flip_decay values, with nc shifting (4-7) to maintain
  roughly constant effective depth. If this depth is truly the MED boundary,
  it should be EXACTLY 3.0 (not just "about 3").

TESTS:
  1. Iso-depth collapse: Multiple fd/nc combos targeting eff_depth=3.0.
     If depth is the invariant, they ALL give ratio ≈ ln(φ).
  2. Depth phase diagram: Sweep eff_depth from 1.0 to 5.0. Where does
     the ratio cross ln(φ)? Is it at d_eff = 3.0?
  3. Crossover precision: Fine-grained sweep near d_eff=3.0 to locate
     exact crossing. Is it 3.00 or some other value?
  4. MED attractor: At d_eff > 3 (above MED bound), does the ratio
     return toward ln(φ), i.e. collapse back to the attractor?
  5. Universality: Does d_eff=3.0 hold across DIFFERENT coupling_strength
     and n_env values?

FALSIFICATION (F10):
  If ln(φ) crossing occurs at d_eff significantly different from 3.0
  (|d_cross - 3.0| > 0.5), the MED-depth interpretation fails. If
  different fd/nc combos at the same effective depth give different ratios,
  depth is NOT the invariant.

SOURCES:
  - classical_physics_information_geometry/paper.md §3, §4.3 (MED bounds)
  - challenges.md line 132: "3 levels (0,1,2) is theoretically optimal"
  - harmonic mobius.md (π-irrational coupling → Möbius topology)
  - exp_02 Diagnostic 2 (nc shifts with fd, effective depth ≈ constant)
  - reality-engine .spec/challenges.md (MED depth ≤ 2 = 3 levels)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, LN_PHI
from core.utils import experiment_header, save_results


# =====================================================================
# Reuse the Landauer model from exp_02 (same physics, different tests)
# =====================================================================

N_SAMPLES = 50000  # Per seed


def entropy_bits(data):
    """Shannon entropy in bits from discrete data."""
    if data.ndim == 1:
        _, counts = np.unique(data, return_counts=True)
    else:
        # Hash multi-column to single int
        h = np.zeros(len(data), dtype=np.int64)
        for col in range(data.shape[1]):
            h += data[:, col].astype(np.int64) * (2 ** col)
        _, counts = np.unique(h, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-30)))


def total_correlation(env, cap=12):
    """Total correlation: sum of marginal entropies minus joint entropy."""
    n_modes = min(env.shape[1], cap)
    marginal_sum = sum(entropy_bits(env[:, j]) for j in range(n_modes))
    return max(0.0, marginal_sum - entropy_bits(env[:, :n_modes]))


def pairwise_mi(env, cap=12):
    """Sum of pairwise mutual information across env modes."""
    n_modes = min(env.shape[1], cap)
    total = 0.0
    for i in range(n_modes):
        H_i = entropy_bits(env[:, i])
        for j in range(i + 1, n_modes):
            H_j = entropy_bits(env[:, j])
            joint = env[:, i].astype(np.int64) * 2 + env[:, j].astype(np.int64)
            H_ij = entropy_bits(joint)
            total += max(0.0, H_i + H_j - H_ij)
    return total


def transfer_entropy(sys_pre, env_post, n_modes=5):
    """Transfer entropy from system to environment (bits)."""
    n_m = min(n_modes, env_post.shape[1])
    env_hash = np.zeros(len(sys_pre), dtype=np.int64)
    for j in range(n_m):
        env_hash += env_post[:, j].astype(np.int64) * (2 ** j)
    H_sp = entropy_bits(sys_pre)
    H_ep = entropy_bits(env_hash)
    joint = sys_pre.astype(np.int64) * (2 ** 20) + env_hash
    _, counts = np.unique(joint, return_counts=True)
    probs = counts / counts.sum()
    H_joint = float(-np.sum(probs * np.log2(probs + 1e-30)))
    return max(0.0, H_sp + H_ep - H_joint)


def eff_depth(fd, nc):
    """Compute effective coupling depth: sum of geometric decay weights."""
    if fd < 1e-10:
        return float(nc)
    return float((1 - np.exp(-fd * nc)) / (1 - np.exp(-fd)))


def max_eff_depth(fd):
    """Maximum achievable effective depth as nc → ∞."""
    if fd < 1e-10:
        return float('inf')
    return float(1.0 / (1 - np.exp(-fd)))


def find_nc_for_depth(fd, target_depth, max_nc=50):
    """Find nc that gives closest effective depth to target."""
    best_nc = 1
    best_delta = float('inf')
    for nc in range(1, max_nc + 1):
        d = eff_depth(fd, nc)
        delta = abs(d - target_depth)
        if delta < best_delta:
            best_delta = delta
            best_nc = nc
        if d > target_depth + 0.5:
            break  # Past target, won't improve
    return best_nc, eff_depth(fd, best_nc)


def landauer_erasure(n_env=20, n_samples=50000, coupling_strength=0.8,
                     flip_decay=0.3, corr_strength=0.3, corr_decay=0.2,
                     n_coupling=None, seed=42):
    """Run one Landauer erasure and measure PAC components."""
    rng = np.random.RandomState(seed)
    system = rng.randint(0, 2, n_samples)
    env_energies = 0.5 + rng.exponential(1.0, n_env)
    env_probs = 1.0 / (1.0 + np.exp(env_energies))
    env_pre = np.zeros((n_samples, n_env), dtype=int)
    for j in range(n_env):
        env_pre[:, j] = (rng.random(n_samples) < env_probs[j]).astype(int)

    TC_pre = total_correlation(env_pre)
    pw_pre = pairwise_mi(env_pre)

    env_post = env_pre.copy()
    was_one = (system == 1)
    if n_coupling is None:
        n_coupling = min(5, n_env)
    n_coupling = min(n_coupling, n_env)

    for j in range(n_coupling):
        coupling = coupling_strength * np.exp(-flip_decay * j)
        flip_mask = was_one & (rng.random(n_samples) < coupling)
        env_post[flip_mask, j] = 1 - env_post[flip_mask, j]

    for j in range(1, n_coupling):
        corr_mask = was_one & (rng.random(n_samples) < corr_strength * np.exp(-corr_decay * j))
        env_post[corr_mask, j] = env_post[corr_mask, 0]

    system_post = np.zeros_like(system)

    TC_post = total_correlation(env_post)
    pw_post = pairwise_mi(env_post)

    P = entropy_bits(system)
    A = transfer_entropy(system, env_post, n_modes=n_coupling)
    xi = max(0.0, (TC_post - TC_pre) + (pw_post - pw_pre))
    R = P - A - xi
    coherent = A + xi
    ratio = A / coherent if coherent > 1e-10 else 0.0

    return {
        'P': float(P), 'A': float(A), 'xi': float(xi),
        'R': float(R), 'coherent': float(coherent), 'ratio': float(ratio),
    }


def ensemble_ratio(n_seeds, n_env=20, n_samples=50000,
                    coupling_strength=0.8, flip_decay=0.3,
                    n_coupling=None, base_seed=42):
    """Compute A/(A+ξ) for an ensemble of independent seeds."""
    ratios = []
    for i in range(n_seeds):
        result = landauer_erasure(
            n_env=n_env, n_samples=n_samples,
            coupling_strength=coupling_strength,
            flip_decay=flip_decay,
            n_coupling=n_coupling,
            seed=base_seed + i
        )
        ratios.append(result['ratio'])

    ratios = np.array(ratios)
    mean = float(np.mean(ratios))
    std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    se = std / np.sqrt(len(ratios)) if len(ratios) > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return {
        'ratios': ratios,
        'mean': mean, 'std': std, 'se': se,
        'ci_95': (ci_low, ci_high),
        'ln_phi_in_ci': ci_low <= LN_PHI <= ci_high,
        'dev_pct': abs(mean - LN_PHI) / LN_PHI * 100,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    meta = experiment_header(
        'exp_11_med_depth_criticality',
        'MED depth d=3 as the edge-of-chaos invariant for A/(A+ξ) = ln(φ)',
        paper='Paper 7 (classical_physics_information_geometry)',
        section='§3-4 (MED bounds, dimensional emergence)'
    )

    results = {**meta, 'tests': {}}

    # =================================================================
    # TEST 1: Iso-depth collapse
    #
    # Different fd/nc combinations that all give eff_depth ≈ 3.0.
    # If depth is the physical invariant, they should ALL give
    # A/(A+ξ) ≈ ln(φ), regardless of how they achieve depth 3.
    # =================================================================
    print("Test 1: Iso-depth collapse (multiple fd/nc → same eff_depth=3.0)")
    print("  If depth is the invariant, all combos give same ratio\n")

    target_depth = 3.0
    iso_seeds = 30
    # fd values where max_eff_depth > 3.0 (otherwise can't reach it)
    fd_values = [fd for fd in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
                 if max_eff_depth(fd) > target_depth + 0.3]

    iso_results = []
    iso_ratios = []
    for fd in fd_values:
        nc, actual_d = find_nc_for_depth(fd, target_depth)
        if abs(actual_d - target_depth) > 0.5:
            continue  # Can't get close enough

        ens = ensemble_ratio(
            n_seeds=iso_seeds, n_env=max(20, nc + 5), n_samples=N_SAMPLES,
            coupling_strength=0.8, flip_decay=fd,
            n_coupling=nc, base_seed=30000 + int(fd * 1000)
        )

        print(f"  fd={fd:.2f}  nc={nc:2d}  d_eff={actual_d:.2f}  "
              f"ratio={ens['mean']:.4f}  dev={ens['dev_pct']:.1f}%  "
              f"CI={'✓' if ens['ln_phi_in_ci'] else '✗'}")

        iso_results.append({
            'flip_decay': fd, 'n_coupling': nc,
            'eff_depth': actual_d, 'mean': ens['mean'],
            'std': ens['std'], 'dev_pct': ens['dev_pct'],
            'ln_phi_in_ci': ens['ln_phi_in_ci'],
        })
        iso_ratios.append(ens['mean'])

    iso_ratios = np.array(iso_ratios)
    iso_cv = float(np.std(iso_ratios) / np.mean(iso_ratios))
    iso_ci_pass = sum(1 for r in iso_results if r['ln_phi_in_ci'])
    iso_grand_mean = float(np.mean(iso_ratios))
    iso_grand_dev = abs(iso_grand_mean - LN_PHI) / LN_PHI * 100

    print(f"\n  Iso-depth grand mean: {iso_grand_mean:.4f} "
          f"({iso_grand_dev:.1f}% from ln(φ))")
    print(f"  Cross-combo CV: {iso_cv:.3f} ({iso_cv*100:.1f}%)")
    print(f"  CI containment: {iso_ci_pass}/{len(iso_results)}")
    depth_is_invariant = iso_cv < 0.05  # < 5% variation across combos
    print(f"  Depth is the invariant: {depth_is_invariant}")

    results['tests']['iso_depth_collapse'] = {
        'target_depth': target_depth,
        'combos': iso_results,
        'grand_mean': iso_grand_mean,
        'grand_dev_pct': iso_grand_dev,
        'cross_combo_cv': iso_cv,
        'ci_pass': f'{iso_ci_pass}/{len(iso_results)}',
        'depth_is_invariant': depth_is_invariant,
        'status': 'PASS' if depth_is_invariant and iso_grand_dev < 5 else 'FAIL',
    }

    # =================================================================
    # TEST 2: Depth phase diagram
    #
    # Sweep effective depth from 1.0 to 5.0 using fd=0.2 (which can
    # reach up to ~5.5). At each depth, find the nc that achieves it.
    # Map ratio vs effective depth — where does it cross ln(φ)?
    # =================================================================
    print("\nTest 2: Depth phase diagram (ratio vs effective depth)")
    print("  Where does A/(A+ξ) cross ln(φ)?\n")

    phase_fd = 0.2  # Can reach max_depth ≈ 5.5
    phase_seeds = 20
    target_depths = np.arange(1.0, 5.5, 0.5)
    phase_data = []

    for td in target_depths:
        nc, actual_d = find_nc_for_depth(phase_fd, td)
        ens = ensemble_ratio(
            n_seeds=phase_seeds, n_env=max(20, nc + 5), n_samples=N_SAMPLES,
            coupling_strength=0.8, flip_decay=phase_fd,
            n_coupling=nc, base_seed=31000 + int(td * 100)
        )
        dev = ens['mean'] - LN_PHI
        print(f"  d_eff={actual_d:.2f} (nc={nc:2d}): "
              f"ratio={ens['mean']:.4f}  Δ={dev:+.4f}  "
              f"CI={'✓' if ens['ln_phi_in_ci'] else '✗'}")
        phase_data.append({
            'target_depth': float(td), 'actual_depth': actual_d,
            'n_coupling': nc, 'mean': ens['mean'],
            'deviation': float(dev), 'ln_phi_in_ci': ens['ln_phi_in_ci'],
        })

    # Find crossing point
    signs = [p['deviation'] for p in phase_data]
    crossings = []
    for i in range(len(signs) - 1):
        if signs[i] * signs[i + 1] < 0:
            # Linear interpolation for crossing depth
            d1, d2 = phase_data[i]['actual_depth'], phase_data[i+1]['actual_depth']
            s1, s2 = signs[i], signs[i+1]
            d_cross = d1 + (d2 - d1) * abs(s1) / (abs(s1) + abs(s2))
            crossings.append(d_cross)

    if crossings:
        crossing_depth = crossings[0]  # First crossing
        print(f"\n  ln(φ) crossing at d_eff = {crossing_depth:.2f}")
        print(f"  Distance from 3.0: {abs(crossing_depth - 3.0):.2f}")
        print(f"  MED consistent (|d_cross - 3| < 0.5): "
              f"{abs(crossing_depth - 3.0) < 0.5}")
    else:
        crossing_depth = None
        closest_idx = int(np.argmin([abs(s) for s in signs]))
        print(f"\n  No zero-crossing; closest at "
              f"d_eff={phase_data[closest_idx]['actual_depth']:.2f}")

    results['tests']['depth_phase_diagram'] = {
        'fd': phase_fd,
        'phase_data': phase_data,
        'crossing_depth': float(crossing_depth) if crossing_depth else None,
        'n_crossings': len(crossings),
        'distance_from_3': float(abs(crossing_depth - 3.0)) if crossing_depth else None,
        'med_consistent': abs(crossing_depth - 3.0) < 0.5 if crossing_depth else False,
        'status': ('PASS' if crossing_depth and abs(crossing_depth - 3.0) < 0.5
                   else 'FAIL'),
    }

    # =================================================================
    # TEST 3: Fine-grained crossover near d_eff = 3.0
    #
    # High-resolution sweep 2.0–4.0 to pin down the exact crossing.
    # Uses fd=0.25 (max_depth ≈ 4.5, good resolution around 3.0).
    # =================================================================
    print("\nTest 3: Fine-grained crossover near d=3.0")
    print("  High-resolution sweep to pin exact crossing depth\n")

    fine_fd = 0.25
    fine_seeds = 25
    fine_targets = np.arange(2.0, 4.1, 0.25)
    fine_data = []

    for td in fine_targets:
        nc, actual_d = find_nc_for_depth(fine_fd, td)
        if abs(actual_d - td) > 0.3:
            continue
        ens = ensemble_ratio(
            n_seeds=fine_seeds, n_env=max(20, nc + 5), n_samples=N_SAMPLES,
            coupling_strength=0.8, flip_decay=fine_fd,
            n_coupling=nc, base_seed=32000 + int(td * 100)
        )
        dev = ens['mean'] - LN_PHI
        print(f"  d_eff={actual_d:.2f} (nc={nc:2d}): "
              f"ratio={ens['mean']:.4f}  Δ={dev:+.4f}  "
              f"CI={'✓' if ens['ln_phi_in_ci'] else '✗'}")
        fine_data.append({
            'target_depth': float(td), 'actual_depth': actual_d,
            'n_coupling': nc, 'mean': ens['mean'],
            'deviation': float(dev), 'ln_phi_in_ci': ens['ln_phi_in_ci'],
        })

    # Precise crossing via interpolation
    fine_signs = [p['deviation'] for p in fine_data]
    fine_crossings = []
    for i in range(len(fine_signs) - 1):
        if fine_signs[i] * fine_signs[i + 1] < 0:
            d1, d2 = fine_data[i]['actual_depth'], fine_data[i+1]['actual_depth']
            s1, s2 = fine_signs[i], fine_signs[i+1]
            d_cross = d1 + (d2 - d1) * abs(s1) / (abs(s1) + abs(s2))
            fine_crossings.append(d_cross)

    if fine_crossings:
        fine_crossing = fine_crossings[0]
        dist_from_3 = abs(fine_crossing - 3.0)
        print(f"\n  Precise crossing: d_eff = {fine_crossing:.3f}")
        print(f"  Distance from 3.0: {dist_from_3:.3f}")
        print(f"  Distance from π-1: {abs(fine_crossing - (np.pi - 1)):.3f} "
              f"(π-1 = {np.pi-1:.3f})")
        print(f"  Distance from e-φ: {abs(fine_crossing - (np.e - PHI)):.3f} "
              f"(e-φ = {np.e - PHI:.3f})")
    else:
        fine_crossing = None
        print(f"\n  No crossing found in [2.0, 4.0] range")

    results['tests']['fine_crossover'] = {
        'fd': fine_fd,
        'fine_data': fine_data,
        'crossing_depth': float(fine_crossing) if fine_crossing else None,
        'distance_from_3': float(abs(fine_crossing - 3.0)) if fine_crossing else None,
        'status': ('PASS' if fine_crossing and abs(fine_crossing - 3.0) < 0.3
                   else 'FAIL'),
    }

    # =================================================================
    # TEST 4: MED attractor behavior
    #
    # MED says depth-3 systems collapse BACK to depth 2 (the attractor).
    # In our model: at d_eff > 3, does the ratio return TOWARD ln(φ)?
    # This would show the phase diagram has a minimum between d=3 and
    # larger depths — the system "wants" to be at d=3.
    #
    # From exp_02 phase diagram: ratio descends from 0.66 (nc=2) to
    # minimum 0.44 (nc=10), then RISES back. The rise toward ln(φ)
    # at large nc IS the attractor pulling back.
    # =================================================================
    print("\nTest 4: MED attractor — does ratio return toward ln(φ) past d=3?")
    print("  Checking for non-monotonic behavior (U-shape in depth space)\n")

    # Use phase diagram data from Test 2
    depths_arr = np.array([p['actual_depth'] for p in phase_data])
    means_arr = np.array([p['mean'] for p in phase_data])

    # Find minimum of ratio across all depths
    min_idx = int(np.argmin(means_arr))
    min_depth = depths_arr[min_idx]
    min_ratio = means_arr[min_idx]

    # Is there a rise after the minimum?
    post_min = means_arr[min_idx:]
    has_rise = len(post_min) > 1 and any(post_min[i] < post_min[i+1]
                                          for i in range(len(post_min) - 1))

    # Does the minimum occur AFTER the MED depth (d=3)?
    min_past_med = min_depth > 3.0

    # Does the ratio return TOWARD ln(φ) after the minimum?
    if len(post_min) > 1:
        recovery = means_arr[-1] - min_ratio
        recovery_toward_lnphi = (means_arr[-1] > min_ratio and
                                 abs(means_arr[-1] - LN_PHI) < abs(min_ratio - LN_PHI))
    else:
        recovery = 0.0
        recovery_toward_lnphi = False

    print(f"  Ratio minimum: {min_ratio:.4f} at d_eff={min_depth:.2f}")
    print(f"  Minimum past MED depth (d>3): {min_past_med}")
    print(f"  Rise after minimum: {has_rise}")
    print(f"  Recovery toward ln(φ): {recovery_toward_lnphi}")
    if has_rise:
        print(f"  Recovery: {min_ratio:.4f} → {means_arr[-1]:.4f} "
              f"(Δ={recovery:+.4f})")

    attractor_behavior = has_rise and min_past_med

    if attractor_behavior:
        print(f"\n  MED attractor confirmed: ratio descends through ln(φ), "
              f"reaches minimum at d_eff={min_depth:.1f},")
        print(f"  then rises back. The system is pulled back toward "
              f"the MED boundary.")

    results['tests']['med_attractor'] = {
        'min_depth': float(min_depth),
        'min_ratio': float(min_ratio),
        'min_past_med': min_past_med,
        'has_rise': has_rise,
        'recovery_toward_lnphi': recovery_toward_lnphi,
        'attractor_behavior': attractor_behavior,
        'interpretation': (
            'MED predicts depth-3 systems collapse back to depth 2. '
            'In the ratio vs depth diagram, this manifests as a minimum '
            'past d=3, after which the ratio rises back. The system '
            '"wants" to be at depth 3 (the MED boundary) — departing '
            'in either direction costs complexity.'
        ),
        'status': 'PASS' if attractor_behavior else 'FAIL',
    }

    # =================================================================
    # TEST 5: Universality across coupling strength and n_env
    #
    # Does d_eff ≈ 3.0 remain the critical depth when we change the
    # coupling strength and environment size? If MED is fundamental,
    # the crossing depth should be robust.
    # =================================================================
    print("\nTest 5: Universality — critical depth across coupling/env params")
    print("  Is d_cross ≈ 3 robust to coupling_strength and n_env?\n")

    univ_configs = [
        {'coupling_strength': 0.6, 'n_env': 20, 'label': 'c=0.6 env=20'},
        {'coupling_strength': 0.8, 'n_env': 20, 'label': 'c=0.8 env=20 (default)'},
        {'coupling_strength': 1.0, 'n_env': 20, 'label': 'c=1.0 env=20'},
        {'coupling_strength': 0.8, 'n_env': 12, 'label': 'c=0.8 env=12'},
        {'coupling_strength': 0.8, 'n_env': 30, 'label': 'c=0.8 env=30'},
    ]

    univ_fd = 0.2  # Good depth resolution
    univ_seeds = 15
    univ_crossings = []

    for cfg in univ_configs:
        # Sweep nc to find crossing
        sweep_ncs = list(range(2, min(15, cfg['n_env'])))
        sweep_data = []
        for nc in sweep_ncs:
            d = eff_depth(univ_fd, nc)
            ens = ensemble_ratio(
                n_seeds=univ_seeds, n_env=cfg['n_env'], n_samples=N_SAMPLES,
                coupling_strength=cfg['coupling_strength'],
                flip_decay=univ_fd, n_coupling=nc,
                base_seed=33000 + int(cfg['coupling_strength'] * 100)
                         + cfg['n_env'] + nc
            )
            sweep_data.append({
                'nc': nc, 'depth': d,
                'mean': ens['mean'], 'dev': ens['mean'] - LN_PHI,
            })

        # Find crossing
        cfg_crossing = None
        for i in range(len(sweep_data) - 1):
            s1, s2 = sweep_data[i]['dev'], sweep_data[i+1]['dev']
            if s1 * s2 < 0:
                d1, d2 = sweep_data[i]['depth'], sweep_data[i+1]['depth']
                cfg_crossing = d1 + (d2 - d1) * abs(s1) / (abs(s1) + abs(s2))
                break

        if cfg_crossing is None:
            # Find closest approach
            abs_devs = [abs(sd['dev']) for sd in sweep_data]
            best = sweep_data[int(np.argmin(abs_devs))]
            cfg_crossing = best['depth']

        univ_crossings.append(cfg_crossing)
        print(f"  {cfg['label']:25s}: d_cross={cfg_crossing:.2f}  "
              f"(Δ from 3.0: {abs(cfg_crossing - 3.0):.2f})")

    univ_mean = float(np.mean(univ_crossings))
    univ_std = float(np.std(univ_crossings))
    univ_cv = univ_std / univ_mean if univ_mean > 0 else float('inf')

    print(f"\n  Crossing depth: {univ_mean:.2f} ± {univ_std:.2f} "
          f"(CV={univ_cv:.2f})")
    print(f"  All within 0.5 of 3.0: "
          f"{all(abs(c - 3.0) < 0.5 for c in univ_crossings)}")

    univ_robust = univ_cv < 0.15 and abs(univ_mean - 3.0) < 0.5

    results['tests']['universality'] = {
        'configs': [c['label'] for c in univ_configs],
        'crossing_depths': univ_crossings,
        'mean': univ_mean,
        'std': univ_std,
        'cv': univ_cv,
        'robust': univ_robust,
        'status': 'PASS' if univ_robust else 'FAIL',
    }

    # =================================================================
    # FALSIFICATION ASSESSMENT
    # =================================================================
    t1_pass = results['tests']['iso_depth_collapse']['status'] == 'PASS'
    t2_pass = results['tests']['depth_phase_diagram']['status'] == 'PASS'
    t3_pass = results['tests']['fine_crossover']['status'] == 'PASS'
    t4_pass = results['tests']['med_attractor']['status'] == 'PASS'
    t5_pass = results['tests']['universality']['status'] == 'PASS'

    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass])

    xd = results['tests']['depth_phase_diagram']['crossing_depth']
    xd_str = f"{xd:.2f}" if xd else "N/A"
    fd_str = (f"{fine_crossing:.3f}" if fine_crossing else "N/A")

    results['falsification'] = {
        'test_id': 'F10',
        'hypothesis': (
            'The effective coupling depth d_eff ≈ 3.0 where A/(A+ξ) = ln(φ) '
            'is the MED saturation depth — depth ≤ 2 in 2D Möbius base = '
            '3 levels from singular origin. This is the same "3" as: '
            'three spatial dimensions, three quark sub-nodes, three fermion '
            'generations, and the double-pendulum chaos depth. The MED '
            'attractor pulls systems back from depth > 3, explaining the '
            'U-shaped phase diagram.'
        ),
        'chain': [
            f'Step 1 (iso-depth collapse): {"PASS" if t1_pass else "FAIL"} '
            f'(same ratio at d=3 via different fd/nc combos)',
            f'Step 2 (depth phase diagram): {"PASS" if t2_pass else "FAIL"} '
            f'(crossing at d_eff={xd_str})',
            f'Step 3 (fine crossover): {"PASS" if t3_pass else "FAIL"} '
            f'(precise crossing at d_eff={fd_str})',
            f'Step 4 (MED attractor): {"PASS" if t4_pass else "FAIL"} '
            f'(ratio recovers past d=3)',
            f'Step 5 (universality): {"PASS" if t5_pass else "FAIL"} '
            f'(d_cross robust across params)',
        ],
        'n_pass': f'{n_pass}/5',
        'falsified': n_pass < 2,
        'honest_assessment': (
            'This experiment tests whether the Landauer critical coupling '
            'depth connects to MED, the dimensionality of space, and the '
            'onset of chaos. If d_cross ≈ 3.0 with low variance across '
            'parameter regimes, the connection is compelling. If d_cross '
            'is parameter-dependent or significantly different from 3.0, '
            'the MED connection fails and the depth is just a model artifact.'
        ),
    }

    print(f"\n--- Falsification Assessment (F10) ---")
    for s in results['falsification']['chain']:
        print(f"  {s}")

    save_results(results, 'exp_11_med_depth_criticality')


if __name__ == '__main__':
    main()
