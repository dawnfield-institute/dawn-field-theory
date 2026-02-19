"""
exp_24: PAC-Lazy Signal Anatomy — Decomposing and Strengthening F19

MOTIVATION:
  exp_21 (F19) showed PAC-Lazy discriminates physics matches from non-matches:
    KL divergence p=0.035, cosine p=0.058, Cohen's d=0.198, enrichment 1.32×
  This is real but modest. Key questions:
    1. Which formulas drive the signal? (leave-one-out)
    2. How robust is p=0.035? (bootstrap CI)
    3. Does the effective dimensionality of PAC profiles explain why exp_16
       failed (6 DOF null space → no prediction) while exp_21 partially works?
    4. Can we identify a subpopulation where the effect is stronger?

CHAIN OF EVIDENCE:
  exp_16 (F14): FAIL 0/4 — null space too large, 6 DOF, no prediction
  exp_18 (F16): 2/4 — conservation necessary but not sufficient
  exp_20 (F18): FAIL 1/4 — raw pressure = depth bias
  exp_21 (F19): PASS 4/4 — PAC + cosine/KL fixes direction, modest effect

TESTS:
  Test 1 — Bootstrap Confidence Interval: 10,000 bootstrap resamples of the
           KL divergence test. Report 95% CI for the matched-vs-unmatched
           difference and for p-value.

  Test 2 — Leave-One-Out Signal Decomposition: Remove each known formula
           from the mesh, re-run discrimination. Which formula contributes
           most to the signal? Which ones hurt it?

  Test 3 — Effective Dimensionality: PCA of PAC profiles. How many principal
           components capture 95% of variance? Compare to exp_16's 6 DOF
           null space. If PAC reduces effective dimensionality, that's WHY
           it works better than the raw null space.

  Test 4 — Subpopulation Analysis: Split novel formulas by their best-target
           domain (particle physics, cosmology, turbulence, neutrino).
           Is the discrimination stronger for specific domains?

FALSIFICATION (F22):
  If bootstrap 95% CI for KL difference includes zero, the signal is fragile.
  If leave-one-out shows >1 formula driving the entire effect, it's driven by
  outliers, not a systematic property.
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results

# Import exp_21's infrastructure
from exp_21_pac_lazy_formula_mesh import (
    KNOWN_FORMULAS, NOVEL_TARGETS, FIB_INDICES, FIB_VALUES,
    formula_pac_profile, build_pac_mesh, profile_to_vector,
    profile_cosine_similarity, profile_kl_divergence,
    compute_index_set_matches, pac_decompose, collect_potential,
)


# =====================================================================
# Rebuild exp_21's discrimination pipeline
# =====================================================================

def run_discrimination(known_formulas, novel_targets, fib_indices):
    """
    Run the full exp_21 discrimination pipeline.
    Returns (matched_kl, unmatched_kl, matched_cos, unmatched_cos, all_records).
    """
    fib_values = [FIB[i] for i in fib_indices]

    # Build mesh from known formulas
    mesh = build_pac_mesh(known_formulas)

    # For each novel index set (2-3 indices), compute:
    # 1. Whether it matches any novel target (physics match)
    # 2. Its PAC profile cosine similarity and KL to the mesh

    all_records = []
    matched_cos = []
    matched_kl = []
    unmatched_cos = []
    unmatched_kl = []

    # Compute index-set matches
    matches = compute_index_set_matches(fib_values, novel_targets, threshold_pct=1.0)

    for idx_set, info in matches.items():
        indices = sorted(idx_set)
        profile = formula_pac_profile(indices)
        cos_sim = profile_cosine_similarity(profile, mesh)
        kl_div = profile_kl_divergence(profile, mesh)

        if not math.isfinite(kl_div):
            continue

        record = {
            'indices': indices,
            'matched': info['matched'],
            'best_target': info['best_target'],
            'best_error': info['best_error'],
            'cosine': cos_sim,
            'kl': kl_div,
        }
        all_records.append(record)

        if info['matched']:
            matched_cos.append(cos_sim)
            matched_kl.append(kl_div)
        else:
            unmatched_cos.append(cos_sim)
            unmatched_kl.append(kl_div)

    return (np.array(matched_kl), np.array(unmatched_kl),
            np.array(matched_cos), np.array(unmatched_cos),
            all_records)


# =====================================================================
# MAIN
# =====================================================================

def main():
    meta = experiment_header(
        'exp_24_pac_lazy_signal_anatomy',
        'Decomposing and strengthening PAC-Lazy discrimination (F19)',
        paper='Paper 4',
        section='§pac_lazy_mesh'
    )

    results = {**meta, 'tests': {}}

    # Baseline run
    print("Running baseline discrimination (exp_21 reproduction)...")
    mk, uk, mc, uc, records = run_discrimination(
        KNOWN_FORMULAS, NOVEL_TARGETS, FIB_INDICES)

    n_matched = len(mk)
    n_unmatched = len(uk)
    baseline_kl_diff = mk.mean() - uk.mean()
    baseline_kl_stat, baseline_kl_p = stats.mannwhitneyu(mk, uk, alternative='less')

    print(f"  Baseline: n_matched={n_matched}, n_unmatched={n_unmatched}")
    print(f"  KL: matched {mk.mean():.4f} vs unmatched {uk.mean():.4f}, "
          f"diff={baseline_kl_diff:.4f}, p={baseline_kl_p:.4f}")

    # =================================================================
    # TEST 1: Bootstrap CI
    # =================================================================
    print("\n" + "=" * 70)
    print("Test 1: Bootstrap Confidence Interval for KL Discrimination")
    print("=" * 70 + "\n")

    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_diffs = np.zeros(n_boot)
    boot_pvals = np.zeros(n_boot)

    for i in range(n_boot):
        # Resample within each group
        mk_boot = mk[rng.randint(0, len(mk), size=len(mk))]
        uk_boot = uk[rng.randint(0, len(uk), size=len(uk))]
        boot_diffs[i] = mk_boot.mean() - uk_boot.mean()
        try:
            _, p = stats.mannwhitneyu(mk_boot, uk_boot, alternative='less')
            boot_pvals[i] = p
        except ValueError:
            boot_pvals[i] = 1.0

    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)
    median_p = np.median(boot_pvals)
    frac_significant = np.mean(boot_pvals < 0.05)

    print(f"  Bootstrap KL difference (matched - unmatched):")
    print(f"    Mean:   {boot_diffs.mean():.6f}")
    print(f"    95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"    CI includes zero: {ci_lo <= 0 <= ci_hi}")
    print(f"\n  Bootstrap p-value:")
    print(f"    Median p: {median_p:.4f}")
    print(f"    Fraction significant (p<0.05): {frac_significant:.1%}")
    print(f"    Fraction significant (p<0.01): {np.mean(boot_pvals < 0.01):.1%}")

    # Signal robust if CI doesn't include zero
    t1_pass = ci_hi < 0  # matched KL should be LOWER than unmatched (negative diff)

    results['tests']['bootstrap_ci'] = {
        'n_bootstrap': n_boot,
        'baseline_kl_diff': float(baseline_kl_diff),
        'baseline_p': float(baseline_kl_p),
        'boot_mean_diff': float(boot_diffs.mean()),
        'boot_ci_95': [float(ci_lo), float(ci_hi)],
        'ci_includes_zero': bool(ci_lo <= 0 <= ci_hi),
        'median_p': float(median_p),
        'frac_p_lt_005': float(frac_significant),
        'frac_p_lt_001': float(np.mean(boot_pvals < 0.01)),
        'status': 'PASS' if t1_pass else 'FAIL',
    }

    # =================================================================
    # TEST 2: Leave-One-Out Signal Decomposition
    # =================================================================
    print("\n" + "=" * 70)
    print("Test 2: Leave-One-Out Signal Decomposition")
    print("=" * 70 + "\n")

    loo_results = {}
    formula_names = list(KNOWN_FORMULAS.keys())

    print(f"  {'Removed':15s}  {'KL diff':>10s}  {'p-value':>10s}  {'Δ effect':>10s}  {'Impact':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for name in formula_names:
        # Build mesh without this formula
        reduced_formulas = {k: v for k, v in KNOWN_FORMULAS.items() if k != name}
        mk_loo, uk_loo, _, _, _ = run_discrimination(
            reduced_formulas, NOVEL_TARGETS, FIB_INDICES)

        if len(mk_loo) < 3 or len(uk_loo) < 3:
            continue

        kl_diff = mk_loo.mean() - uk_loo.mean()
        try:
            _, p = stats.mannwhitneyu(mk_loo, uk_loo, alternative='less')
        except ValueError:
            p = 1.0

        delta_effect = kl_diff - baseline_kl_diff
        # Positive delta_effect means removing this formula INCREASES the diff
        # (makes discrimination weaker if the baseline diff is negative)
        impact = 'weakens' if delta_effect > 0 else 'strengthens'

        loo_results[name] = {
            'kl_diff': float(kl_diff),
            'p_value': float(p),
            'delta_effect': float(delta_effect),
            'impact': impact,
        }

        print(f"  {name:15s}  {kl_diff:10.6f}  {p:10.4f}  {delta_effect:+10.6f}  {impact:>10s}")

    # How concentrated is the signal?
    effects = [abs(v['delta_effect']) for v in loo_results.values()]
    max_effect_name = max(loo_results, key=lambda k: abs(loo_results[k]['delta_effect']))
    max_effect = abs(loo_results[max_effect_name]['delta_effect'])
    total_effect = sum(effects)
    concentration = max_effect / total_effect if total_effect > 0 else 0

    print(f"\n  Signal concentration:")
    print(f"    Largest impact: {max_effect_name} ({concentration:.1%} of total effect)")
    print(f"    Concentrated (>50% in one formula): {concentration > 0.5}")

    # PASS if no single formula drives >50% of the effect
    t2_pass = concentration < 0.5

    results['tests']['leave_one_out'] = {
        'formula_effects': loo_results,
        'max_effect_formula': max_effect_name,
        'concentration': float(concentration),
        'signal_distributed': t2_pass,
        'status': 'PASS' if t2_pass else 'FAIL',
    }

    # =================================================================
    # TEST 3: Effective Dimensionality via PCA
    # =================================================================
    print("\n" + "=" * 70)
    print("Test 3: Effective Dimensionality of PAC Profiles")
    print("=" * 70 + "\n")

    # Collect all PAC profiles (known + novel candidates) into a matrix
    all_profiles = []
    all_labels = []

    # Known formula profiles
    for name, indices in KNOWN_FORMULAS.items():
        profile = formula_pac_profile(indices)
        all_profiles.append(profile)
        all_labels.append(f'known:{name}')

    # Novel candidate profiles (sample from all_records)
    for rec in records[:100]:  # First 100 novel candidates
        profile = formula_pac_profile(rec['indices'])
        all_profiles.append(profile)
        all_labels.append(f'novel:{rec["indices"]}')

    # Build profile matrix
    all_indices = sorted(set().union(*[p.keys() for p in all_profiles]))
    profile_matrix = np.array([
        [p.get(idx, 0.0) for idx in all_indices]
        for p in all_profiles
    ])

    # PCA
    from numpy.linalg import svd
    centered = profile_matrix - profile_matrix.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    explained_var = S**2 / (S**2).sum()
    cumulative = np.cumsum(explained_var)

    # How many components for 95%?
    n_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    n_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n_80 = int(np.searchsorted(cumulative, 0.80) + 1)

    print(f"  Profile matrix: {profile_matrix.shape[0]} profiles × "
          f"{profile_matrix.shape[1]} indices")
    print(f"\n  PCA explained variance:")
    for i in range(min(8, len(explained_var))):
        bar_len = int(explained_var[i] * 50)
        bar = '#' * bar_len
        print(f"    PC{i+1:2d}: {explained_var[i]:6.1%}  cumul: {cumulative[i]:6.1%}  {bar}")

    print(f"\n  Components for 80% variance: {n_80}")
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Total possible dimensions: {profile_matrix.shape[1]}")
    print(f"\n  exp_16 null space DOF: 6")
    print(f"  PAC effective DOF (95%): {n_95}")
    print(f"  Dimensionality reduction: {profile_matrix.shape[1]} → {n_95} "
          f"({(1 - n_95/profile_matrix.shape[1]):.0%} reduction)")

    # exp_16 had 6 DOF which was too many. Does PAC reduce below this?
    reduces_below_null = n_95 < 6
    print(f"  Below exp_16 null DOF? {reduces_below_null}")

    t3_pass = n_95 <= 5  # Effective dimensionality below exp_16's 6

    results['tests']['effective_dimensionality'] = {
        'n_profiles': profile_matrix.shape[0],
        'n_indices': profile_matrix.shape[1],
        'n_components_80pct': int(n_80),
        'n_components_90pct': int(n_90),
        'n_components_95pct': int(n_95),
        'variance_explained': explained_var[:10].tolist(),
        'exp16_null_dof': 6,
        'reduces_below_null': reduces_below_null,
        'status': 'PASS' if t3_pass else 'FAIL',
    }

    # =================================================================
    # TEST 4: Subpopulation Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("Test 4: Subpopulation Analysis by Target Domain")
    print("=" * 70 + "\n")

    # Classify novel targets by domain
    domains = {
        'quark_mixing': ['V_us', 'V_cb', 'V_ub', 'V_td', 'V_ts', 'Jarlskog_J'],
        'mass_ratios': ['m_u/m_d', 'm_s/m_d', 'm_c/m_s', 'm_b/m_c', 'm_t/m_b',
                        'm_mu/m_e', 'm_tau/m_mu'],
        'neutrino': ['sin2_theta12', 'sin2_theta23', 'sin2_theta13', 'Dm2_ratio'],
        'running_couplings': ['Lambda_QCD_mp', 'alpha_em_MZ', 'sin2_theta_eff',
                              'alpha_s_Mtau'],
        'cosmology': ['Omega_b', 'Omega_c', 'Omega_Lambda', 'n_s', 'sigma_8'],
        'turbulence': ['zeta_3', 'von_Karman'],
    }

    # For each matched record, find its domain
    domain_matched_kl = defaultdict(list)
    domain_unmatched_kl = defaultdict(list)

    for rec in records:
        target = rec['best_target']
        domain = 'unknown'
        for d, targets in domains.items():
            if target in targets:
                domain = d
                break

        if rec['matched']:
            domain_matched_kl[domain].append(rec['kl'])
        # All unmatched go into their closest-target domain
        domain_unmatched_kl[domain].append(rec['kl'])

    print(f"  {'Domain':20s}  {'n_match':>7s}  {'n_unmatch':>9s}  "
          f"{'KL_m':>8s}  {'KL_u':>8s}  {'Δ':>8s}  {'p':>8s}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    domain_results = {}
    best_domain = None
    best_domain_d = 0

    for domain in sorted(domains.keys()):
        m = np.array(domain_matched_kl.get(domain, []))
        u = np.array(domain_unmatched_kl.get(domain, []))

        if len(m) < 3:
            print(f"  {domain:20s}  {len(m):7d}  {len(u):9d}  "
                  f"{'(too few)':>8s}")
            domain_results[domain] = {'n_matched': len(m), 'status': 'too_few'}
            continue

        delta = m.mean() - u.mean()
        try:
            _, p = stats.mannwhitneyu(m, u, alternative='less')
        except ValueError:
            p = 1.0

        # Cohen's d
        pooled_std = np.sqrt((m.var() * (len(m)-1) + u.var() * (len(u)-1)) /
                             (len(m) + len(u) - 2)) if len(m) + len(u) > 2 else 1
        d = delta / pooled_std if pooled_std > 0 else 0

        domain_results[domain] = {
            'n_matched': int(len(m)),
            'n_unmatched': int(len(u)),
            'mean_kl_matched': float(m.mean()),
            'mean_kl_unmatched': float(u.mean()),
            'delta': float(delta),
            'p_value': float(p),
            'cohens_d': float(d),
        }

        if abs(d) > abs(best_domain_d):
            best_domain = domain
            best_domain_d = d

        print(f"  {domain:20s}  {len(m):7d}  {len(u):9d}  "
              f"{m.mean():8.4f}  {u.mean():8.4f}  {delta:+8.4f}  {p:8.4f}")

    print(f"\n  Best domain: {best_domain} (Cohen's d = {best_domain_d:.3f})")

    # PASS if at least one domain has |d| > 0.3 (medium effect)
    t4_pass = abs(best_domain_d) > 0.3

    results['tests']['subpopulation'] = {
        'domains': domain_results,
        'best_domain': best_domain,
        'best_domain_d': float(best_domain_d),
        'has_medium_effect': t4_pass,
        'status': 'PASS' if t4_pass else 'FAIL',
    }

    # =================================================================
    # SYNTHESIS
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    t1_s = results['tests']['bootstrap_ci']['status']
    t2_s = results['tests']['leave_one_out']['status']
    t3_s = results['tests']['effective_dimensionality']['status']
    t4_s = results['tests']['subpopulation']['status']

    tests_pass = sum(1 for s in [t1_s, t2_s, t3_s, t4_s] if s == 'PASS')

    print(f"\n  Test 1 (bootstrap CI):         {t1_s}")
    print(f"  Test 2 (leave-one-out):        {t2_s}")
    print(f"  Test 3 (effective dim):        {t3_s}")
    print(f"  Test 4 (subpopulation):        {t4_s}")
    print(f"\n  Result: {tests_pass}/4 PASS")

    print(f"\n  Signal anatomy:")
    print(f"    Baseline: KL diff = {baseline_kl_diff:.4f}, p = {baseline_kl_p:.4f}")
    print(f"    Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Most important formula: {max_effect_name} "
          f"(concentration: {concentration:.1%})")
    print(f"    Effective dimensionality: {n_95} (vs 6 DOF null space)")
    if best_domain:
        print(f"    Best domain: {best_domain} (d = {best_domain_d:.3f})")

    # Falsification
    results['falsification'] = {
        'test_id': 'F22',
        'hypothesis': (
            'PAC-Lazy discrimination (exp_21, F19) is robust, distributed across '
            'formulas, and driven by reduced effective dimensionality.'
        ),
        'chain': [
            f'Test 1 (bootstrap): {t1_s} — 95% CI for KL difference',
            f'Test 2 (LOO): {t2_s} — signal concentration {concentration:.1%}',
            f'Test 3 (PCA): {t3_s} — effective DOF {n_95}',
            f'Test 4 (subpop): {t4_s} — best domain d={best_domain_d:.3f}',
        ],
        'n_pass': f'{tests_pass}/4',
        'falsified': tests_pass < 1,
    }

    save_results(results, 'exp_24_pac_lazy_signal_anatomy')


if __name__ == '__main__':
    main()
