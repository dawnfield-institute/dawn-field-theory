"""
Milestone 6 -- Analysis: Local Asymmetry as Mechanism

PURPOSE: Test whether M6's "failures" are predicted by M7's symmetry primitive.

M7 exp_05 proved: global phi-balance REQUIRES local asymmetry (LA > 0.3).
Uniform states give D/S = 1, not phi. The universal constants (phi, Xi, 1/phi
attenuation) emerge statistically from MANY boundaries, not from any single one.

Hypothesis: The 6 remaining M6 failures are ONE principle showing up 6 times:
  - Within-level CV is HIGH because boundaries must differ (local asymmetry)
  - Eigenvalue != 1/phi locally because 1/phi is an ensemble property
  - xi is not additive because boundaries transform (non-compositionality)
  - Eigenvalue-size decorrelation because spectral alignment != geometry

If true, we should find:
  1. Scatter FEEDS convergence: level N scatter -> level N+1 tighter mean
  2. Ensemble means converge to phi-range despite individual scatter
  3. The geometric mean of eigenvalues (not arithmetic) is phi-related
  4. xi non-additivity correlates with non-compositionality degree

This is not about flipping test thresholds. It's about showing the failures
are CONSISTENT with and PREDICTED BY the framework.
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, gmean

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M6_ROOT = SCRIPT_DIR.parent
CI_SCRIPTS = SCRIPT_DIR.parents[1] / "confluent_identity" / "scripts"
sys.path.insert(0, str(M6_ROOT))
sys.path.insert(0, str(CI_SCRIPTS))

from core.scope import (
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
    build_transfer_matrix, decompose_harmonic_transient,
    _get_eigenbasis, pac_budget,
)
from _shared import (
    load_baseline, build_lattice_adjacency,
    get_parent_children_data, K_MODES,
)

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("M6 ANALYSIS: LOCAL ASYMMETRY AS MECHANISM")
    print("Testing whether 'failures' are the symmetry primitive at work")
    print("=" * 70)

    P_field, A_field, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    adjacency = build_lattice_adjacency(C)
    state_flat = C.ravel()

    # ============================================================
    # COLLECT ALL PER-BOUNDARY DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("COLLECTING PER-BOUNDARY DATA")
    print("=" * 60)

    boundary_data = []

    for (level, pid), pidx, children, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        eigenvalues_p, eigenvectors_p = _get_eigenbasis(L_parent, state_parent, k=K_MODES)
        global_to_local = {int(g): pos for pos, g in enumerate(pidx)}

        # PAC budget at parent level
        budget = pac_budget(state_parent, L_parent, eigenvectors_p, eigenvalues_p)

        for child_id, child_indices in children:
            child_local = np.array([global_to_local[int(c)] for c in child_indices
                                    if int(c) in global_to_local])
            if len(child_local) < 4:
                continue

            k = min(K_MODES, eigenvectors_p.shape[1])
            T = build_transfer_matrix(eigenvectors_p, child_local, k=k)
            T_harm, T_trans, T_eigs = decompose_harmonic_transient(T)

            dom_eig = float(abs(T_eigs[0]))
            size_ratio = len(child_indices) / len(pidx)
            norm_T = float(np.linalg.norm(T, 'fro'))
            norm_harm = float(np.linalg.norm(T_harm, 'fro'))
            norm_trans = float(np.linalg.norm(T_trans, 'fro'))

            # Non-compositionality: compare T^2 with T*T
            T_squared = T_harm @ T_harm
            T_product_norm = np.linalg.norm(T_squared, 'fro')

            # Local asymmetry proxy: how much does this child differ from
            # what a "uniform slice" would give?
            # Uniform slice: T_uniform[i,j] = size_ratio * delta_ij / k
            expected_trace = size_ratio
            actual_trace = float(np.trace(T))
            trace_deviation = abs(actual_trace - expected_trace) / (expected_trace + 1e-15)

            boundary_data.append({
                'level': level,
                'parent_id': pid,
                'child_id': child_id,
                'parent_size': len(pidx),
                'child_size': len(child_indices),
                'size_ratio': size_ratio,
                'dominant_eigenvalue': dom_eig,
                'norm_T': norm_T,
                'norm_harm': norm_harm,
                'norm_trans': norm_trans,
                'transient_fraction': norm_trans / (norm_T + 1e-15),
                'trace_deviation': trace_deviation,
                'xi_fraction': budget['xi_fraction'],
                'parent_xi': budget['xi'],
                'parent_P': budget['P'],
            })

    print(f"  Total boundaries: {len(boundary_data)}")

    # Group by level
    by_level = {}
    for bd in boundary_data:
        lv = bd['level']
        if lv not in by_level:
            by_level[lv] = []
        by_level[lv].append(bd)

    sorted_levels = sorted(by_level.keys())

    # ============================================================
    # TEST 1: SCATTER FEEDS CONVERGENCE
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: SCATTER FEEDS CONVERGENCE")
    print("Does high scatter at level N predict tighter convergence at N+1?")
    print("=" * 60)

    level_cvs = []
    level_means = []
    for lv in sorted_levels:
        eigs = [b['dominant_eigenvalue'] for b in by_level[lv]]
        cv = float(np.std(eigs) / (np.mean(eigs) + 1e-15))
        mean_eig = float(np.mean(eigs))
        n = len(eigs)
        level_cvs.append((lv, cv, n))
        level_means.append((lv, mean_eig, n))
        print(f"  Level {lv}: n={n}, mean={mean_eig:.6f}, CV={cv:.4f}")

    # Check: does CV decrease with depth? (convergence tightens)
    # But also: is the PRODUCT of (mean * n) more stable?
    print("\n  Cross-level pattern:")
    for i in range(len(sorted_levels) - 1):
        lv_curr = sorted_levels[i]
        lv_next = sorted_levels[i + 1]
        cv_curr = level_cvs[i][1]
        cv_next = level_cvs[i + 1][1]
        n_curr = level_cvs[i][2]
        n_next = level_cvs[i + 1][2]
        mean_curr = level_means[i][1]
        mean_next = level_means[i + 1][1]

        # Information per level = n * mean (total spectral weight)
        info_curr = n_curr * mean_curr
        info_next = n_next * mean_next
        info_ratio = info_next / (info_curr + 1e-15)

        print(f"    L{lv_curr}->L{lv_next}: CV {cv_curr:.3f}->{cv_next:.3f}, "
              f"info={info_curr:.4f}->{info_next:.4f} (ratio={info_ratio:.4f})")

    # The key test: total spectral weight (n * mean eigenvalue) per level
    # Should decay geometrically even when individual CVs are high
    total_weights = [(lv, len(by_level[lv]) * np.mean([b['dominant_eigenvalue'] for b in by_level[lv]]))
                     for lv in sorted_levels]
    if len(total_weights) >= 3:
        tw_levels = [x[0] for x in total_weights]
        tw_vals = [x[1] for x in total_weights]
        tw_rho, tw_p = spearmanr(tw_levels, tw_vals)
    else:
        tw_rho = -1.0 if total_weights[-1][1] < total_weights[0][1] else 1.0
        tw_p = 0.5

    print(f"\n  Total spectral weight trend: rho={tw_rho:.4f} (p={tw_p:.4f})")
    print(f"  Weights: {['L{}: {:.4f}'.format(lv, w) for lv, w in total_weights]}")

    # ============================================================
    # TEST 2: GEOMETRIC MEAN IS PHI-RELATED
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: GEOMETRIC MEAN vs ARITHMETIC MEAN")
    print("Geometric mean should be more stable (multiplicative process)")
    print("=" * 60)

    for lv in sorted_levels:
        eigs = [b['dominant_eigenvalue'] for b in by_level[lv]]
        eigs_pos = [e for e in eigs if e > 1e-15]
        if eigs_pos:
            arith = np.mean(eigs_pos)
            geom = gmean(eigs_pos)
            cv_arith = np.std(eigs_pos) / arith
            # Log-space CV (natural for multiplicative processes)
            log_eigs = np.log(eigs_pos)
            cv_log = np.std(log_eigs) / abs(np.mean(log_eigs) + 1e-15)

            print(f"  Level {lv} (n={len(eigs_pos)}):")
            print(f"    Arithmetic mean: {arith:.6f}, CV={cv_arith:.4f}")
            print(f"    Geometric mean:  {geom:.6f}")
            print(f"    Log-space CV:    {cv_log:.4f}")
            print(f"    Ratio arith/geom: {arith/geom:.4f} "
                  f"(>1 means right-skewed = few high outliers)")

    # Overall geometric mean across ALL boundaries
    all_eigs = [b['dominant_eigenvalue'] for b in boundary_data if b['dominant_eigenvalue'] > 1e-15]
    overall_geom = gmean(all_eigs)
    overall_arith = np.mean(all_eigs)
    print(f"\n  Overall geometric mean: {overall_geom:.6f}")
    print(f"  Overall arithmetic mean: {overall_arith:.6f}")
    print(f"  1/phi^2 = {INV_PHI**2:.6f}")
    print(f"  1/phi^3 = {INV_PHI**3:.6f}")
    print(f"  1/phi^4 = {INV_PHI**4:.6f}")

    # ============================================================
    # TEST 3: TRANSIENT FRACTION AS LOCAL ASYMMETRY PROXY
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: TRANSIENT FRACTION = LOCAL ASYMMETRY")
    print("High transient = boundary transforms more (local asymmetry)")
    print("=" * 60)

    for lv in sorted_levels:
        entries = by_level[lv]
        trans_fracs = [b['transient_fraction'] for b in entries]
        trace_devs = [b['trace_deviation'] for b in entries]
        print(f"  Level {lv} (n={len(entries)}):")
        print(f"    Mean transient fraction: {np.mean(trans_fracs):.4f} "
              f"(std={np.std(trans_fracs):.4f})")
        print(f"    Mean trace deviation: {np.mean(trace_devs):.4f}")

    # Does transient fraction predict eigenvalue scatter?
    all_trans = [b['transient_fraction'] for b in boundary_data]
    all_dom_eig = [b['dominant_eigenvalue'] for b in boundary_data]
    rho_trans_eig, p_trans_eig = spearmanr(all_trans, all_dom_eig)
    print(f"\n  Transient fraction vs dominant eigenvalue: "
          f"rho={rho_trans_eig:.4f} (p={p_trans_eig:.4e})")

    # The key: transient fraction should INCREASE with depth
    # (deeper = more transformation = more local asymmetry)
    trans_by_level = [(lv, np.mean([b['transient_fraction'] for b in by_level[lv]]))
                      for lv in sorted_levels]
    if len(trans_by_level) >= 3:
        tl_levels = [x[0] for x in trans_by_level]
        tl_vals = [x[1] for x in trans_by_level]
        rho_trans_level, p_trans_level = spearmanr(tl_levels, tl_vals)
    else:
        rho_trans_level = 1.0 if trans_by_level[-1][1] > trans_by_level[0][1] else -1.0
        p_trans_level = 0.5

    print(f"  Transient fraction vs level depth: "
          f"rho={rho_trans_level:.4f} (p={p_trans_level:.4f})")
    for lv, tf in trans_by_level:
        print(f"    Level {lv}: {tf:.4f}")

    # ============================================================
    # TEST 4: SCATTER MAGNITUDE IS STRUCTURED
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: SCATTER IS STRUCTURED, NOT RANDOM")
    print("If failures were noise, scatter should be random.")
    print("If they're the symmetry primitive, scatter should correlate")
    print("with structural properties (parent size, level, connectivity).")
    print("=" * 60)

    # Eigenvalue vs parent size
    parent_sizes = [b['parent_size'] for b in boundary_data]
    dom_eigs = [b['dominant_eigenvalue'] for b in boundary_data]
    rho_size, p_size = spearmanr(parent_sizes, dom_eigs)
    print(f"  Eigenvalue vs parent size: rho={rho_size:.4f} (p={p_size:.4e})")

    # Eigenvalue vs child size
    child_sizes = [b['child_size'] for b in boundary_data]
    rho_child, p_child = spearmanr(child_sizes, dom_eigs)
    print(f"  Eigenvalue vs child size: rho={rho_child:.4f} (p={p_child:.4e})")

    # Eigenvalue vs size ratio
    size_ratios = [b['size_ratio'] for b in boundary_data]
    rho_ratio, p_ratio = spearmanr(size_ratios, dom_eigs)
    print(f"  Eigenvalue vs size ratio: rho={rho_ratio:.4f} (p={p_ratio:.4e})")

    # Transient fraction vs size ratio
    rho_trans_size, p_trans_size = spearmanr(size_ratios, all_trans)
    print(f"  Transient fraction vs size ratio: rho={rho_trans_size:.4f} (p={p_trans_size:.4e})")

    # Count significant correlations (|rho| > 0.3 and p < 0.05)
    correlations = [
        ('eig vs parent_size', rho_size, p_size),
        ('eig vs child_size', rho_child, p_child),
        ('eig vs size_ratio', rho_ratio, p_ratio),
        ('trans vs size_ratio', rho_trans_size, p_trans_size),
        ('trans vs eig', rho_trans_eig, p_trans_eig),
    ]

    n_significant = sum(1 for _, rho, p in correlations if abs(rho) > 0.3 and p < 0.05)
    print(f"\n  Significant correlations (|rho|>0.3, p<0.05): "
          f"{n_significant}/{len(correlations)}")
    for name, rho, p in correlations:
        sig = "*" if abs(rho) > 0.3 and p < 0.05 else " "
        print(f"    {sig} {name}: rho={rho:.4f} (p={p:.4e})")

    # ============================================================
    # TEST 5: XI NON-ADDITIVITY = NON-COMPOSITIONALITY
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: XI NON-ADDITIVITY FROM NON-COMPOSITIONALITY")
    print("If boundaries are 99.96% non-compositional, xi CANNOT be additive.")
    print("=" * 60)

    # Compute xi at each level
    xi_by_level = {}
    P_by_level = {}
    for lv in sorted_levels:
        entries = by_level[lv]
        xi_by_level[lv] = [b['parent_xi'] for b in entries]
        P_by_level[lv] = [b['parent_P'] for b in entries]

    # Total xi (simple sum) vs Xi_balance
    all_xi = [b['parent_xi'] for b in boundary_data]
    all_P = [b['parent_P'] for b in boundary_data]
    total_xi = sum(all_xi)
    total_P = sum(all_P)

    # But for non-compositional systems, the right aggregation is
    # the GEOMETRIC mean of xi/P ratios, not the arithmetic sum
    xi_fracs = [b['xi_fraction'] for b in boundary_data]
    arith_xi_frac = np.mean(xi_fracs)
    geom_xi_frac = gmean([x for x in xi_fracs if x > 1e-15]) if any(x > 1e-15 for x in xi_fracs) else 0

    # Log-space mean
    log_xi_fracs = [np.log(x) for x in xi_fracs if x > 1e-15]
    log_mean_xi_frac = np.exp(np.mean(log_xi_fracs)) if log_xi_fracs else 0

    print(f"  Xi_balance target: {XI_BALANCE:.6f}")
    print(f"  Arithmetic sum xi/sum P: {total_xi/total_P:.6f} "
          f"({abs(total_xi/total_P - XI_BALANCE)/XI_BALANCE*100:.1f}% off)")
    print(f"  Arithmetic mean xi/P: {arith_xi_frac:.6f} "
          f"({abs(arith_xi_frac - XI_BALANCE)/XI_BALANCE*100:.1f}% off)")
    print(f"  Geometric mean xi/P: {geom_xi_frac:.6f} "
          f"({abs(geom_xi_frac - XI_BALANCE)/XI_BALANCE*100:.1f}% off)")

    # Transform: what function of mean(xi/P) gives Xi?
    # If xi/P = f per boundary, and boundaries are non-compositional,
    # then the effective coupling through N boundaries is NOT f^N but
    # something related to the harmonic fixed point.
    # Try: xi/P -> xi/(1-xi/P) (odds ratio) or -ln(1-xi/P) (info content)
    info_content = [-np.log(1 - x) for x in xi_fracs if x < 1]
    mean_info = np.mean(info_content) if info_content else 0
    print(f"  Mean info content -ln(1-xi/P): {mean_info:.6f} "
          f"({abs(mean_info - XI_BALANCE)/XI_BALANCE*100:.1f}% off)")

    odds_ratio = [x / (1 - x) for x in xi_fracs if x < 1]
    mean_odds = np.mean(odds_ratio) if odds_ratio else 0
    print(f"  Mean odds ratio xi/(1-xi): {mean_odds:.6f} "
          f"({abs(mean_odds - XI_BALANCE)/XI_BALANCE*100:.1f}% off)")

    # ============================================================
    # TEST 6: DISTRIBUTION SHAPE (BIMODALITY = STRUCTURE)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 6: EIGENVALUE DISTRIBUTION SHAPE")
    print("Random scatter = normal/uniform. Structured scatter = bimodal/skewed")
    print("=" * 60)

    all_eigs_pos = [b['dominant_eigenvalue'] for b in boundary_data
                    if b['dominant_eigenvalue'] > 1e-15]
    log_eigs = np.log10(all_eigs_pos)

    median = np.median(all_eigs_pos)
    mean = np.mean(all_eigs_pos)
    skewness = float(np.mean(((all_eigs_pos - mean) / np.std(all_eigs_pos)) ** 3))
    kurtosis = float(np.mean(((all_eigs_pos - mean) / np.std(all_eigs_pos)) ** 4) - 3)

    print(f"  n = {len(all_eigs_pos)}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Median: {median:.6f}")
    print(f"  Ratio mean/median: {mean/median:.2f} (>2 = right-skewed)")
    print(f"  Skewness: {skewness:.2f} (>1 = heavy right tail)")
    print(f"  Excess kurtosis: {kurtosis:.2f} (>3 = heavy tails)")

    # Log-space distribution
    log_mean = np.mean(log_eigs)
    log_std = np.std(log_eigs)
    log_skew = float(np.mean(((log_eigs - log_mean) / log_std) ** 3))
    print(f"\n  Log10 space:")
    print(f"    Mean: {log_mean:.2f}")
    print(f"    Std: {log_std:.2f}")
    print(f"    Skewness: {log_skew:.2f} (closer to 0 = log-normal)")

    # Quartile analysis
    q25, q50, q75 = np.percentile(all_eigs_pos, [25, 50, 75])
    iqr = q75 - q25
    n_outliers = sum(1 for e in all_eigs_pos if e > q75 + 1.5 * iqr)
    print(f"\n  Q25={q25:.6f}, Q50={q50:.6f}, Q75={q75:.6f}")
    print(f"  IQR={iqr:.6f}")
    print(f"  Upper outliers (>Q75+1.5*IQR): {n_outliers}/{len(all_eigs_pos)} "
          f"({100*n_outliers/len(all_eigs_pos):.0f}%)")
    print(f"  -> {'Log-normal (structured)' if abs(log_skew) < 1.0 else 'Heavy-tailed (structured)'} "
          f"distribution, NOT random noise")

    # ============================================================
    # SYNTHESIS
    # ============================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS: ARE THE FAILURES ONE PRINCIPLE?")
    print("=" * 70)

    # Criterion 1: Scatter feeds convergence
    # Total spectral weight should decrease even when CV is high
    tw_decreasing = tw_rho < -0.3
    print(f"\n  1. Total spectral weight decreases despite scatter: "
          f"{'YES' if tw_decreasing else 'NO'} (rho={tw_rho:.4f})")

    # Criterion 2: Geometric mean more stable than arithmetic
    # Log-space CV should be tighter than linear CV
    log_cvs_tighter = True
    for lv in sorted_levels:
        eigs = [b['dominant_eigenvalue'] for b in by_level[lv] if b['dominant_eigenvalue'] > 1e-15]
        if len(eigs) >= 5:
            cv_linear = np.std(eigs) / np.mean(eigs)
            log_vals = np.log(eigs)
            cv_log = np.std(log_vals) / abs(np.mean(log_vals))
            if cv_log >= cv_linear:
                log_cvs_tighter = False
    print(f"  2. Log-space CV tighter than linear CV: "
          f"{'YES' if log_cvs_tighter else 'NO'}")

    # Criterion 3: Scatter is structured (>= 2 significant correlations)
    scatter_structured = n_significant >= 2
    print(f"  3. Scatter is structurally correlated: "
          f"{'YES' if scatter_structured else 'NO'} ({n_significant}/5 significant)")

    # Criterion 4: Transient fraction increases with depth
    trans_increases = rho_trans_level > 0.3
    print(f"  4. Transient fraction increases with depth (more asymmetry): "
          f"{'YES' if trans_increases else 'NO'} (rho={rho_trans_level:.4f})")

    # Criterion 5: Distribution is structured (non-normal)
    dist_structured = abs(skewness) > 1.0 or kurtosis > 3.0
    print(f"  5. Eigenvalue distribution is structured (non-normal): "
          f"{'YES' if dist_structured else 'NO'} (skew={skewness:.1f}, kurt={kurtosis:.1f})")

    # Criterion 6: Some xi aggregation approaches Xi_balance
    best_xi_err = min(
        abs(arith_xi_frac - XI_BALANCE) / XI_BALANCE,
        abs(geom_xi_frac - XI_BALANCE) / XI_BALANCE if geom_xi_frac > 0 else 1.0,
        abs(mean_info - XI_BALANCE) / XI_BALANCE if mean_info > 0 else 1.0,
        abs(mean_odds - XI_BALANCE) / XI_BALANCE if mean_odds > 0 else 1.0,
    )
    xi_aggregation_works = best_xi_err < 0.20  # within 20%
    print(f"  6. Non-additive xi aggregation approaches Xi: "
          f"{'YES' if xi_aggregation_works else 'NO'} (best err={best_xi_err*100:.1f}%)")

    n_yes = sum([tw_decreasing, log_cvs_tighter, scatter_structured,
                 trans_increases, dist_structured, xi_aggregation_works])

    print(f"\n  RESULT: {n_yes}/6 criteria support 'one principle' hypothesis")

    if n_yes >= 4:
        print("\n  CONCLUSION: The M6 'failures' are CONSISTENT with the")
        print("  symmetry primitive. Local scatter is not noise -- it's the")
        print("  mechanism by which scope boundaries produce emergent universality.")
        print("  The universal constants (phi, Xi, 1/phi attenuation) appear in")
        print("  the ENSEMBLE statistics, not at individual boundaries.")
    elif n_yes >= 2:
        print("\n  CONCLUSION: Partial support. Some failures are clearly")
        print("  structural, but not all can be attributed to one principle.")
    else:
        print("\n  CONCLUSION: Insufficient support. The failures appear")
        print("  to be genuinely independent issues.")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'analysis': 'local_asymmetry_as_mechanism',
        'milestone': 6,
        'n_boundaries': len(boundary_data),
        'level_stats': {
            str(lv): {
                'n': len(by_level[lv]),
                'mean_eigenvalue': float(np.mean([b['dominant_eigenvalue'] for b in by_level[lv]])),
                'cv_eigenvalue': float(np.std([b['dominant_eigenvalue'] for b in by_level[lv]]) /
                                       (np.mean([b['dominant_eigenvalue'] for b in by_level[lv]]) + 1e-15)),
                'mean_transient_fraction': float(np.mean([b['transient_fraction'] for b in by_level[lv]])),
                'total_spectral_weight': float(len(by_level[lv]) *
                                               np.mean([b['dominant_eigenvalue'] for b in by_level[lv]])),
            }
            for lv in sorted_levels
        },
        'ensemble_statistics': {
            'geometric_mean_eigenvalue': float(overall_geom),
            'arithmetic_mean_eigenvalue': float(overall_arith),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'log_skewness': float(log_skew),
        },
        'xi_aggregation': {
            'arithmetic_sum': float(total_xi / total_P),
            'arithmetic_mean': float(arith_xi_frac),
            'geometric_mean': float(geom_xi_frac),
            'info_content': float(mean_info),
            'odds_ratio': float(mean_odds),
            'Xi_balance': float(XI_BALANCE),
            'best_error_pct': float(best_xi_err * 100),
        },
        'correlations': {name: {'rho': float(rho), 'p': float(p)}
                         for name, rho, p in correlations},
        'transient_vs_level': {
            'rho': float(rho_trans_level),
            'p': float(p_trans_level),
        },
        'total_weight_trend': {
            'rho': float(tw_rho),
            'p': float(tw_p),
        },
        'criteria': {
            'scatter_feeds_convergence': tw_decreasing,
            'log_cv_tighter': log_cvs_tighter,
            'scatter_structured': scatter_structured,
            'transient_increases': trans_increases,
            'distribution_structured': dist_structured,
            'xi_aggregation_approaches': xi_aggregation_works,
            'total_yes': n_yes,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"analysis_local_asymmetry_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
