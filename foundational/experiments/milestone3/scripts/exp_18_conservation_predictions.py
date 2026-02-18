"""
exp_18: Conservation-Required Predictions

INSIGHT (from exp_16 honest failure):
  Brute-force scanning asks "what can Fibonacci construct?" — enrichment
  0.92× against random integer sequences. ANY increasing sequence matches
  ~26/30 physics targets at 1% tolerance. The scan approach is backwards:
  it treats Fibonacci as a parts bin for formula construction.

  PAC says something different: potential EXPANDS into possibilities,
  conservation BALANCES the budget, actualization COLLAPSES possibilities
  into structure. Each collapse constrains what MUST follow.

  Predictions aren't "what can we build?" but "what MUST happen for the
  system to conserve?"

MECHANISM:
  The stoichiometric null space (d dims out of 11D) defines conservation-
  compatible "reactions." Known formulas occupy specific null-space
  directions. With enough actualizations, the remaining space becomes
  DETERMINED — new formulas aren't "discovered," they're REQUIRED for
  the conservation budget to close.

  The key metric: CONSERVATION FRACTION. For each novel formula, how much
  of its null-space projection is explained by the already-actualized
  formulas? If CF ≈ 1.0, the formula is conservation-required. If CF ≈ 0,
  it's conservation-compatible but not forced.

  exp_16 showed: Fibonacci matches the same number of targets as random.
  exp_18 asks: are Fibonacci's matches CONSERVATION-REQUIRED, while
  random matches are mere coincidences?

WHY THIS IS DIFFERENT:
  exp_16: "How many targets does this sequence match?" → Same for any sequence
  exp_18: "Are the matches structurally forced by conservation?" → Fibonacci-specific

TESTS:
  Test 1 — Budget Exhaustion: Known formulas span r/d of the null space.
           If r ≈ d, new formulas are fully conservation-determined.
           Also: is exhaustion higher for Fibonacci than alternatives?

  Test 2 — Conservation Cascade: Add formulas one by one, track how the
           residual freedom shrinks. Show that actualization progressively
           constrains what can follow — possibilities collapse.

  Test 3 — Conservation Fraction vs Physics: Among formulas that match
           physics (<1%), do high-CF formulas dominate? Compare mean CF
           of physics-matching vs non-matching. This is the core test:
           conservation should SELECT physics over noise.

  Test 4 — Sequence Specificity: Repeat Test 3 for Lucas, Primes,
           Tribonacci, Random. Only Fibonacci should show elevated CF
           for physics-matching formulas.

SOURCES:
  - exp_13, exp_14 (stoichiometric matrix)
  - exp_16 (brute-force baseline: 0/4 PASS — the failure this corrects)
  - PDG 2024, CODATA 2018
"""

import sys
import os
import math
import numpy as np
from itertools import combinations
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results


# =====================================================================
# Constants
# =====================================================================
FIB_INDICES = list(range(2, 13))  # indices 2..12
N_SPECIES = len(FIB_INDICES)       # 11

FIB_VALUES = [FIB[i] for i in FIB_INDICES]  # [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Known formulas: actualized relationships
KNOWN_FORMULAS = {
    'sin2_thetaW': [4, 7],
    'Koide':       [2, 3],
    'She_Lev':     [3, 4],
    'nu_WF':       [4],
    'alpha_s':     [4, 6],
    'Cabibbo':     [4, 7],
    'mu_e':        [4, 6, 7],
    'alpha_em':    [3, 4, 7, 10],
    'p_e':         [4, 6, 9, 12],
    'tau_e':       [4, 5, 7, 11],
}
KNOWN_INDEX_SETS = [frozenset(v) for v in KNOWN_FORMULAS.values()]

# Novel targets — non-trivial only (exclude things near 1.0, φ, 1/φ, 1.5, 0.5)
TRIVIAL_PROXIMITY = 0.02
TRIVIAL_VALUES = [1.0, PHI, INV_PHI, 2.0, 0.5, 3.0, 1.5]

NOVEL_TARGETS = {
    'V_us':               0.2243,
    'V_cb':               0.0422,
    'V_ub':               0.00394,
    'V_td':               0.00814,
    'V_ts':               0.0400,
    'Jarlskog_J':         3.08e-5,
    'm_u/m_d':            0.474,
    'm_s/m_d':            20.2,
    'm_c/m_s':            11.7,
    'm_b/m_c':            3.41,
    'm_t/m_b':            41.3,
    'm_mu/m_e':           206.7682830,
    'm_tau/m_mu':         16.8170,
    'sin2_theta12':       0.307,
    'sin2_theta23':       0.546,
    'sin2_theta13':       0.0220,
    'Dm2_ratio':          0.0297,
    'Lambda_QCD_mp':      0.217,
    'alpha_em_MZ':        1.0/127.951,
    'sin2_theta_eff':     0.23155,
    'alpha_s_Mtau':       0.332,
    'Omega_b':            0.0493,
    'Omega_c':            0.265,
    'Omega_Lambda':       0.685,
    'n_s':                0.965,
    'sigma_8':            0.811,
    'zeta_3':             1.2020569,
    'von_Karman':         0.41,
}

# Filter to non-trivial only
NOVEL_TARGETS = {k: v for k, v in NOVEL_TARGETS.items()
                 if not any(abs(v - tv) / max(abs(tv), 1e-12) < TRIVIAL_PROXIMITY
                            for tv in TRIVIAL_VALUES)}


# =====================================================================
# Alternative sequences for Test 4
# =====================================================================
def lucas_sequence(n=13):
    """Lucas numbers L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2)."""
    L = [2, 1]
    for i in range(2, n):
        L.append(L[-1] + L[-2])
    return L

def tribonacci_sequence(n=13):
    """Tribonacci: T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3)."""
    T = [0, 0, 1]
    for i in range(3, n):
        T.append(T[-1] + T[-2] + T[-3])
    # Avoid zeros — use from index 2 onward
    return T

def primes_sequence(n=13):
    """First n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes

ALTERNATIVE_SEQUENCES = {
    'Lucas': [lucas_sequence()[i] for i in FIB_INDICES],
    'Primes': primes_sequence()[1:12],  # skip 2 to get 11 values
    'Tribonacci': [tribonacci_sequence()[i] for i in FIB_INDICES],
}
# Ensure all have length 11 — pad/trim if needed
for name in list(ALTERNATIVE_SEQUENCES):
    seq = ALTERNATIVE_SEQUENCES[name]
    if len(seq) < N_SPECIES:
        # Extend with growth-rate-matched values
        while len(seq) < N_SPECIES:
            seq.append(seq[-1] + seq[-2])
    ALTERNATIVE_SEQUENCES[name] = seq[:N_SPECIES]
    # Replace zeros with 1 to avoid division errors
    ALTERNATIVE_SEQUENCES[name] = [max(v, 1) for v in ALTERNATIVE_SEQUENCES[name]]


# =====================================================================
# Matrix construction
# =====================================================================
def build_stoichiometric_matrix(values):
    """
    Build 5-constraint stoichiometric matrix for a given integer sequence.
    Row 0 (magnitude) depends on values; rows 1-4 on structural indices only.
    """
    S = np.zeros((5, N_SPECIES))
    idx_map = {n: FIB_INDICES.index(n) for n in FIB_INDICES}

    S[0] = values                                       # Magnitude conservation
    S[1] = FIB_INDICES                                  # Hierarchy depth
    S[2] = [n % 3 for n in FIB_INDICES]                 # E-I-S cycle
    S[3] = [n % 2 for n in FIB_INDICES]                 # Parity
    S[4, idx_map[5]] = -1
    S[4, idx_map[6]] = -1
    S[4, idx_map[7]] = 1                                # Gauge closure: F₅+F₆=F₇
    return S


def get_null_space(S):
    """Return orthonormal null space basis (d × n), rank, null dimension."""
    U, sigma, Vt = np.linalg.svd(S)
    tol = 1e-10
    rank = int(np.sum(sigma > tol * sigma[0]))
    null_dim = S.shape[1] - rank
    null_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, S.shape[1]))
    return null_basis, rank, null_dim


def formula_indicator(indices):
    """Unit indicator vector for given Fibonacci indices."""
    vec = np.zeros(N_SPECIES)
    for i in indices:
        if i in FIB_INDICES:
            vec[FIB_INDICES.index(i)] = 1.0
    return vec


def null_alignment(vec, null_basis):
    """Fraction of vector that lies in null space (0–1)."""
    n = np.linalg.norm(vec)
    if n < 1e-12 or null_basis.shape[0] == 0:
        return 0.0
    proj = null_basis @ vec
    return float(np.linalg.norm(proj) / n)


def conservation_fraction(vec, null_basis, known_null_coords):
    """
    Fraction of vec's null-space projection explained by known formulas.

    Parameters
    ----------
    vec : array (n,)
        Formula indicator vector.
    null_basis : array (d, n)
        Orthonormal null space basis.
    known_null_coords : array (k, d)
        Null-space coordinates of k known formulas.

    Returns
    -------
    float
        0.0 = formula is in the residual (conservation-compatible, not forced)
        1.0 = formula is fully conservation-determined (forced by known formulas)
    """
    # Project vec into null space
    c = null_basis @ vec  # (d,) coordinates
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-12:
        return 0.0  # vec orthogonal to null space

    if known_null_coords.shape[0] == 0:
        return 0.0

    # Project c onto the column space of known_null_coords^T
    # K = (d × k) matrix whose columns are known formula directions
    K = known_null_coords.T
    try:
        alpha, residuals, rank_K, sv = np.linalg.lstsq(K, c, rcond=None)
        c_explained = K @ alpha
        explained_frac = min(np.linalg.norm(c_explained) / c_norm, 1.0)
        return float(explained_frac)
    except Exception:
        return 0.0


# =====================================================================
# Formula scanning
# =====================================================================
def scan_formulas(values, null_basis, known_null_coords, targets,
                  threshold_pct=1.0, include_triples=True):
    """
    Scan ratio and product templates, scoring each by conservation fraction.

    Returns list of dicts with: expression, value, target, error_pct,
    conservation_fraction, matched.
    """
    results = []
    n = len(FIB_INDICES)

    # --- Ratio templates (pairs) ---
    for ai in range(n):
        for bi in range(n):
            if ai == bi or values[bi] == 0:
                continue

            indicator = formula_indicator([FIB_INDICES[ai], FIB_INDICES[bi]])
            cf = conservation_fraction(indicator, null_basis, known_null_coords)

            va, vb = values[ai], values[bi]
            templates = {
                f'V{FIB_INDICES[ai]}/V{FIB_INDICES[bi]}':
                    va / vb,
                f'V{FIB_INDICES[ai]}/(V{FIB_INDICES[bi]}*Xi)':
                    va / (vb * XI_BALANCE),
                f'V{FIB_INDICES[ai]}/(V{FIB_INDICES[bi]}*phi)':
                    va / (vb * PHI),
                f'V{FIB_INDICES[ai]}*Xi/V{FIB_INDICES[bi]}':
                    va * XI_BALANCE / vb,
            }

            for expr, val in templates.items():
                if not math.isfinite(val) or val <= 0:
                    continue
                for tname, tval in targets.items():
                    if tval == 0:
                        continue
                    err = abs(val - tval) / abs(tval) * 100
                    results.append({
                        'expression': expr,
                        'value': float(val),
                        'target': tname,
                        'error_pct': float(err),
                        'matched': err < threshold_pct,
                        'cf': float(cf),
                    })

    # --- Product templates (triples) ---
    if include_triples:
        for ai in range(n):
            for bi in range(n):
                for ci in range(n):
                    if len({ai, bi, ci}) < 3 or values[ci] == 0:
                        continue

                    indicator = formula_indicator([
                        FIB_INDICES[ai], FIB_INDICES[bi], FIB_INDICES[ci]])
                    cf = conservation_fraction(
                        indicator, null_basis, known_null_coords)

                    va, vb, vc = values[ai], values[bi], values[ci]
                    templates = {}
                    templates[f'V{FIB_INDICES[ai]}*V{FIB_INDICES[bi]}/V{FIB_INDICES[ci]}'] = \
                        va * vb / vc
                    if vb * vc > 0:
                        templates[f'V{FIB_INDICES[ai]}^2/(V{FIB_INDICES[bi]}*V{FIB_INDICES[ci]})'] = \
                            va**2 / (vb * vc)
                    templates[f'V{FIB_INDICES[ai]}*V{FIB_INDICES[bi]}/(V{FIB_INDICES[ci]}*Xi)'] = \
                        va * vb / (vc * XI_BALANCE)
                    if vb * vc > 0:
                        templates[f'V{FIB_INDICES[ai]}/(V{FIB_INDICES[bi]}*V{FIB_INDICES[ci]})'] = \
                            va / (vb * vc)

                    for expr, val in templates.items():
                        if not math.isfinite(val) or val <= 0:
                            continue
                        for tname, tval in targets.items():
                            if tval == 0:
                                continue
                            err = abs(val - tval) / abs(tval) * 100
                            results.append({
                                'expression': expr,
                                'value': float(val),
                                'target': tname,
                                'error_pct': float(err),
                                'matched': err < threshold_pct,
                                'cf': float(cf),
                            })

    return results


# =====================================================================
# Test 1: Budget Exhaustion
# =====================================================================
def test_1_budget_exhaustion():
    """
    How much of the stoichiometric null space is spanned by known
    formula indicator vectors?

    If exhaustion ≈ 100%, every new formula is a LINEAR CONSEQUENCE
    of the actualized ones. The null space is "full" — conservation
    determines everything.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Budget Exhaustion — How Determined Is the System?")
    print("=" * 70)

    S = build_stoichiometric_matrix(FIB_VALUES)
    null_basis, rank, null_dim = get_null_space(S)

    print(f"\n  Stoichiometric matrix: {S.shape[0]}×{S.shape[1]}, "
          f"rank={rank}, null dim={null_dim}")

    # Project each known formula into the null space
    known_coords = []
    print(f"\n  Known formulas — null-space projections:")
    print(f"  {'Formula':15s}  {'Indices':20s}  {'Align':>6s}  {'Coord norm':>10s}")
    for name, indices in KNOWN_FORMULAS.items():
        vec = formula_indicator(indices)
        c = null_basis @ vec  # d-dimensional coordinates
        align = null_alignment(vec, null_basis)
        known_coords.append(c)
        print(f"  {name:15s}  {str(indices):20s}  {align:6.4f}  {np.linalg.norm(c):10.4f}")

    known_matrix = np.array(known_coords)  # (k × d)

    # Rank = how many independent directions known formulas span
    _, sv, _ = np.linalg.svd(known_matrix)
    significant = sv > 1e-10 * sv[0]
    known_rank = int(np.sum(significant))
    exhaustion = known_rank / null_dim

    print(f"\n  Known formula projections: {known_matrix.shape}")
    print(f"  Singular values: {', '.join(f'{s:.4f}' for s in sv)}")
    print(f"  Rank of known projections: {known_rank}")
    print(f"  Null space exhaustion: {known_rank}/{null_dim} = {exhaustion:.1%}")

    if exhaustion >= 1.0:
        print(f"\n  ★ NULL SPACE FULLY EXHAUSTED")
        print(f"  Every conservation-compatible formula is a linear combination")
        print(f"  of the known formulas. New relationships are DETERMINED,")
        print(f"  not discovered.")
    else:
        residual_dim = null_dim - known_rank
        print(f"\n  Residual freedom: {residual_dim} independent direction(s)")
        print(f"  The system allows {residual_dim} more independent formulas")
        print(f"  before conservation fully determines everything.")

    passed = exhaustion >= 0.70
    print(f"\n  PASS: {passed} (exhaustion ≥ 70%)")

    return {
        'null_dim': null_dim,
        'known_rank': known_rank,
        'exhaustion': float(exhaustion),
        'singular_values': [float(s) for s in sv],
        'passed': passed,
    }, null_basis, known_matrix


# =====================================================================
# Test 2: Conservation Cascade
# =====================================================================
def test_2_conservation_cascade(null_basis, full_known_matrix):
    """
    Add known formulas one at a time. After each actualization, how
    much of the null space is consumed?

    This shows the PROGRESSIVE TIGHTENING: each collapse of potential
    into actuality reduces the remaining freedom.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Conservation Cascade — Progressive Tightening")
    print("=" * 70)

    null_dim = null_basis.shape[0]
    formula_names = list(KNOWN_FORMULAS.keys())

    # Order by information contribution: add in greedy order
    # (each time, add the formula that increases rank the most)
    remaining = list(range(len(formula_names)))
    order = []
    cumulative_matrix = np.empty((0, null_dim))
    cascade = []

    while remaining:
        best_idx = None
        best_rank_increase = -1
        for idx in remaining:
            test_matrix = np.vstack([cumulative_matrix,
                                     full_known_matrix[idx:idx+1]])
            _, sv, _ = np.linalg.svd(test_matrix)
            test_rank = int(np.sum(sv > 1e-10 * sv[0])) if len(sv) > 0 else 0
            current_rank = cumulative_matrix.shape[0]
            # Use actual rank, not just row count
            if cumulative_matrix.shape[0] > 0:
                _, sv_curr, _ = np.linalg.svd(cumulative_matrix)
                current_rank = int(np.sum(sv_curr > 1e-10 * sv_curr[0]))
            else:
                current_rank = 0
            increase = test_rank - current_rank
            if increase > best_rank_increase:
                best_rank_increase = increase
                best_idx = idx

        order.append(best_idx)
        remaining.remove(best_idx)
        cumulative_matrix = np.vstack([cumulative_matrix,
                                       full_known_matrix[best_idx:best_idx+1]])
        if cumulative_matrix.shape[0] > 0:
            _, sv, _ = np.linalg.svd(cumulative_matrix)
            r = int(np.sum(sv > 1e-10 * sv[0]))
        else:
            r = 0
        exhaustion = r / null_dim
        cascade.append({
            'step': len(order),
            'formula': formula_names[best_idx],
            'rank': r,
            'exhaustion': float(exhaustion),
            'rank_increase': best_rank_increase,
        })

    print(f"\n  {'Step':>4s}  {'Formula':15s}  {'Rank':>4s}  {'Exhaust':>8s}  {'ΔRank':>5s}")
    print(f"  {'─'*4}  {'─'*15}  {'─'*4}  {'─'*8}  {'─'*5}")
    for step in cascade:
        bar = '█' * int(step['exhaustion'] * 20) + '░' * (20 - int(step['exhaustion'] * 20))
        print(f"  {step['step']:4d}  {step['formula']:15s}  "
              f"{step['rank']:4d}  {step['exhaustion']:7.1%}  "
              f"+{step['rank_increase']:4d}  {bar}")

    # Check for monotonicity
    exhaustions = [s['exhaustion'] for s in cascade]
    monotonic = all(exhaustions[i] <= exhaustions[i+1]
                    for i in range(len(exhaustions) - 1))

    # Key finding: at what formula does system become >80% exhausted?
    threshold_step = None
    for s in cascade:
        if s['exhaustion'] >= 0.80 and threshold_step is None:
            threshold_step = s['step']

    print(f"\n  Monotonic tightening: {monotonic}")
    if threshold_step:
        print(f"  System >80% determined after {threshold_step} formulas")
    print(f"  Final exhaustion: {cascade[-1]['exhaustion']:.1%}")

    passed = monotonic and cascade[-1]['exhaustion'] >= 0.70
    print(f"\n  PASS: {passed} (monotonic AND final exhaustion ≥ 70%)")

    return {
        'cascade': cascade,
        'monotonic': monotonic,
        'threshold_80pct': threshold_step,
        'final_exhaustion': cascade[-1]['exhaustion'],
        'passed': passed,
    }


# =====================================================================
# Test 3: Conservation Fraction vs Physics Matching
# =====================================================================
def test_3_conservation_vs_physics(null_basis, known_matrix):
    """
    The core test: among all formula/target comparisons, do physics-
    matching formulas have HIGHER conservation fractions than non-matching?

    If yes: conservation SELECTS physics. The matches aren't coincidences —
    they're conservation requirements.

    If no: conservation doesn't distinguish physics from noise. Matches
    are just as coincidental as exp_16 showed.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Conservation Fraction vs Physics Matching")
    print("=" * 70)

    # Scan all formulas against all targets
    print(f"\n  Scanning Fibonacci formulas against {len(NOVEL_TARGETS)} "
          f"non-trivial targets...")
    all_results = scan_formulas(
        FIB_VALUES, null_basis, known_matrix, NOVEL_TARGETS,
        threshold_pct=1.0, include_triples=True)

    matched = [r for r in all_results if r['matched']]
    unmatched = [r for r in all_results if not r['matched']]

    print(f"  Total formula-target comparisons: {len(all_results)}")
    print(f"  Matched (<1%): {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}")

    if not matched:
        print(f"\n  No matches found — cannot test")
        return {'passed': False, 'reason': 'no matches'}

    # Conservation fractions
    cf_matched = [r['cf'] for r in matched]
    cf_unmatched = [r['cf'] for r in unmatched]

    mean_cf_matched = float(np.mean(cf_matched))
    mean_cf_unmatched = float(np.mean(cf_unmatched))
    std_cf_matched = float(np.std(cf_matched))
    std_cf_unmatched = float(np.std(cf_unmatched))

    print(f"\n  Conservation Fraction Statistics:")
    print(f"  {'':30s}  {'Mean CF':>8s}  {'Std CF':>8s}  {'N':>6s}")
    print(f"  {'Physics matches (<1%)':30s}  {mean_cf_matched:8.4f}  "
          f"{std_cf_matched:8.4f}  {len(cf_matched):6d}")
    print(f"  {'Non-matches':30s}  {mean_cf_unmatched:8.4f}  "
          f"{std_cf_unmatched:8.4f}  {len(cf_unmatched):6d}")
    print(f"  Difference: {mean_cf_matched - mean_cf_unmatched:+.4f}")

    # Statistical test (Mann-Whitney U)
    from scipy import stats
    if len(cf_matched) >= 5 and len(cf_unmatched) >= 5:
        stat, p_val = stats.mannwhitneyu(
            cf_matched, cf_unmatched, alternative='greater')
        print(f"\n  Mann-Whitney U (matched > unmatched): "
              f"U={stat:.0f}, p={p_val:.6f}")
    else:
        p_val = 1.0
        print(f"\n  Too few matches for statistical test")

    # Show top physics matches with their CF
    matched_sorted = sorted(matched, key=lambda x: x['error_pct'])
    print(f"\n  Top physics matches by conservation fraction:")
    print(f"  {'Expression':35s}  {'Target':15s}  {'Err%':>7s}  {'CF':>6s}")
    print(f"  {'─'*35}  {'─'*15}  {'─'*7}  {'─'*6}")
    for r in matched_sorted[:20]:
        marker = '★' if r['cf'] > 0.8 else '·'
        print(f"  {marker} {r['expression']:33s}  {r['target']:15s}  "
              f"{r['error_pct']:7.4f}  {r['cf']:6.3f}")

    # Conservation fraction distribution of matches
    cf_bins = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n  Conservation fraction distribution:")
    for lo, hi in cf_bins:
        n_in_bin = sum(1 for cf in cf_matched if lo <= cf < hi)
        pct = n_in_bin / len(cf_matched) * 100 if cf_matched else 0
        bar = '█' * int(pct / 2)
        print(f"    CF [{lo:.1f}–{hi:.1f}): {n_in_bin:4d} ({pct:5.1f}%)  {bar}")

    # How many matches are conservation-determined (CF > 0.8)?
    n_determined = sum(1 for cf in cf_matched if cf >= 0.8)
    n_undetermined = sum(1 for cf in cf_matched if cf < 0.3)
    det_fraction = n_determined / len(cf_matched) if cf_matched else 0

    print(f"\n  Conservation-determined matches (CF ≥ 0.8): "
          f"{n_determined}/{len(cf_matched)} = {det_fraction:.1%}")
    print(f"  Residual matches (CF < 0.3): "
          f"{n_undetermined}/{len(cf_matched)}")

    # PASS: physics matches have significantly higher CF than non-matches
    # AND majority of matches are conservation-determined
    cf_elevated = mean_cf_matched > mean_cf_unmatched and p_val < 0.05
    majority_determined = det_fraction > 0.5

    passed = cf_elevated and majority_determined
    print(f"\n  CF elevated (p < 0.05): {cf_elevated}")
    print(f"  Majority conservation-determined (>50%): {majority_determined}")
    print(f"  PASS: {passed}")

    return {
        'n_matched': len(matched),
        'n_unmatched': len(unmatched),
        'mean_cf_matched': mean_cf_matched,
        'mean_cf_unmatched': mean_cf_unmatched,
        'difference': float(mean_cf_matched - mean_cf_unmatched),
        'mann_whitney_p': float(p_val),
        'n_determined': n_determined,
        'det_fraction': float(det_fraction),
        'top_matches': [{
            'expression': r['expression'],
            'target': r['target'],
            'error_pct': r['error_pct'],
            'cf': r['cf'],
        } for r in matched_sorted[:15]],
        'passed': passed,
    }


# =====================================================================
# Test 4: Sequence Specificity
# =====================================================================
def test_4_sequence_specificity(fib_cf_matched_mean, fib_det_fraction):
    """
    Repeat the conservation analysis for alternative sequences.
    Only Fibonacci should show elevated conservation fractions
    for physics-matching formulas.

    This is the definitive control: same matrix structure, same formula
    indices, same targets — but different VALUES. If conservation
    fractions are equally high for Lucas/Primes, then CF doesn't
    distinguish Fibonacci. If only Fibonacci has high CF for matches,
    conservation is sequence-specific.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Sequence Specificity — Is Conservation Fibonacci-Specific?")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Add random sequences to the alternatives
    all_alternatives = dict(ALTERNATIVE_SEQUENCES)
    for seed in range(5):
        r = np.random.default_rng(seed + 100)
        vals = sorted(r.choice(np.arange(1, 201), size=N_SPECIES, replace=False))
        all_alternatives[f'Random_{seed}'] = [int(v) for v in vals]

    results_by_seq = {}
    results_by_seq['Fibonacci'] = {
        'mean_cf_matched': fib_cf_matched_mean,
        'det_fraction': fib_det_fraction,
    }

    print(f"\n  {'Sequence':15s}  {'Null dim':>8s}  {'Exhaust':>8s}  "
          f"{'Matches':>8s}  {'Mean CF':>8s}  {'Det%':>6s}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

    # Fibonacci reference line
    print(f"  {'Fibonacci':15s}  {'─':>8s}  {'─':>8s}  "
          f"{'─':>8s}  {fib_cf_matched_mean:8.4f}  "
          f"{fib_det_fraction:5.1%}")

    for seq_name, seq_values in all_alternatives.items():
        try:
            S = build_stoichiometric_matrix(seq_values)
            null_basis, rank, null_dim = get_null_space(S)

            if null_dim == 0:
                results_by_seq[seq_name] = {
                    'mean_cf_matched': 0, 'det_fraction': 0,
                    'n_matched': 0, 'note': 'null dim = 0',
                }
                print(f"  {seq_name:15s}  {null_dim:8d}  {'N/A':>8s}  "
                      f"{'N/A':>8s}  {'N/A':>8s}  {'N/A':>6s}")
                continue

            # Compute known formula projections for this sequence's null space
            known_coords = []
            for name, indices in KNOWN_FORMULAS.items():
                vec = formula_indicator(indices)
                c = null_basis @ vec
                known_coords.append(c)
            known_matrix = np.array(known_coords)

            # Null space exhaustion
            _, sv, _ = np.linalg.svd(known_matrix)
            kr = int(np.sum(sv > 1e-10 * sv[0]))
            exh = kr / null_dim

            # Scan formulas using THIS sequence's values
            all_res = scan_formulas(
                seq_values, null_basis, known_matrix, NOVEL_TARGETS,
                threshold_pct=1.0, include_triples=True)

            matched = [r for r in all_res if r['matched']]
            n_matched = len(matched)

            if n_matched > 0:
                cf_vals = [r['cf'] for r in matched]
                mean_cf = float(np.mean(cf_vals))
                n_det = sum(1 for cf in cf_vals if cf >= 0.8)
                det_frac = n_det / n_matched
            else:
                mean_cf = 0.0
                det_frac = 0.0

            results_by_seq[seq_name] = {
                'null_dim': null_dim,
                'exhaustion': float(exh),
                'n_matched': n_matched,
                'mean_cf_matched': mean_cf,
                'det_fraction': det_frac,
            }

            print(f"  {seq_name:15s}  {null_dim:8d}  {exh:7.1%}  "
                  f"{n_matched:8d}  {mean_cf:8.4f}  {det_frac:5.1%}")

        except Exception as e:
            results_by_seq[seq_name] = {'error': str(e)}
            print(f"  {seq_name:15s}  ERROR: {e}")

    # Analysis: is Fibonacci's mean CF significantly above alternatives?
    alt_cfs = [v['mean_cf_matched'] for k, v in results_by_seq.items()
               if k != 'Fibonacci' and isinstance(v.get('mean_cf_matched'), (int, float))
               and v.get('n_matched', 0) > 0]

    if alt_cfs:
        alt_mean = float(np.mean(alt_cfs))
        alt_std = float(np.std(alt_cfs))
        fib_z = ((fib_cf_matched_mean - alt_mean) / alt_std
                 if alt_std > 0 else 0)

        print(f"\n  Alternative mean CF: {alt_mean:.4f} ± {alt_std:.4f}")
        print(f"  Fibonacci mean CF:   {fib_cf_matched_mean:.4f}")
        print(f"  z-score: {fib_z:.2f}")

        # Also compare det_fractions
        alt_dets = [v['det_fraction'] for k, v in results_by_seq.items()
                    if k != 'Fibonacci'
                    and isinstance(v.get('det_fraction'), (int, float))
                    and v.get('n_matched', 0) > 0]
        alt_det_mean = float(np.mean(alt_dets)) if alt_dets else 0

        print(f"  Alternative det%:    {alt_det_mean:.1%}")
        print(f"  Fibonacci det%:      {fib_det_fraction:.1%}")
    else:
        fib_z = 0
        alt_mean = 0

    # PASS: Fibonacci CF significantly above alternatives (z > 1.5)
    passed = fib_z > 1.5
    print(f"\n  PASS: {passed} (Fibonacci z > 1.5 above alternatives)")
    if not passed and fib_z > 0:
        print(f"  → Fibonacci is ABOVE average but not significantly")
    elif fib_z <= 0:
        print(f"  → Fibonacci conservation fractions are NOT elevated")
        print(f"    This means conservation doesn't distinguish Fibonacci")
        print(f"    from alternatives — the structure is generic, not special.")

    return {
        'results_by_sequence': {k: v for k, v in results_by_seq.items()
                                 if k != 'Fibonacci'},
        'fib_cf': fib_cf_matched_mean,
        'alt_mean_cf': float(alt_mean) if alt_cfs else None,
        'fib_z_score': float(fib_z),
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_18_conservation_predictions',
        'Conservation-required predictions — what MUST happen for PAC to balance',
        paper='Paper 4',
        section='§predictions'
    )

    results = {'metadata': meta, 'tests': {}}

    # Test 1
    t1_results, null_basis, known_matrix = test_1_budget_exhaustion()
    results['tests']['test_1_exhaustion'] = t1_results

    # Test 2
    results['tests']['test_2_cascade'] = test_2_conservation_cascade(
        null_basis, known_matrix)

    # Test 3
    t3_results = test_3_conservation_vs_physics(null_basis, known_matrix)
    results['tests']['test_3_cf_vs_physics'] = t3_results

    # Test 4
    fib_cf = t3_results.get('mean_cf_matched', 0)
    fib_det = t3_results.get('det_fraction', 0)
    results['tests']['test_4_specificity'] = test_4_sequence_specificity(
        fib_cf, fib_det)

    # --- Final synthesis ---
    print("\n" + "=" * 70)
    print("  SYNTHESIS: Conservation-Required Predictions")
    print("=" * 70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:35s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  INTERPRETATION                                             │")
    print(f"  │                                                              │")
    print(f"  │  exp_16 showed: Fibonacci matches ≈ random matches (0.92×)  │")
    print(f"  │  exp_18 asks:   Are Fibonacci's matches CONSERVATION-FORCED? │")
    print(f"  │                                                              │")
    print(f"  │  If Tests 3+4 pass:                                         │")
    print(f"  │    → Conservation SELECTS physics matches                    │")
    print(f"  │    → The matches aren't coincidental — they're required      │")
    print(f"  │    → Fibonacci is specifically distinguished from alternatives│")
    print(f"  │                                                              │")
    print(f"  │  If Tests 3+4 fail:                                         │")
    print(f"  │    → Conservation fractions are generic (not Fibonacci-specific)│")
    print(f"  │    → The stoichiometric matrix is too permissive to constrain│")
    print(f"  │    → More physics-derived constraints needed (see exp_17)    │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    results['summary'] = {
        'total': total, 'passed': pass_count,
        'score': f"{pass_count}/{total}",
    }

    # Falsification block
    results['falsification'] = {
        'test_id': 'experimental (not in registry)',
        'hypothesis': (
            'PAC conservation structure selects physics-matching formulas: '
            'Fibonacci matches have higher conservation fractions than random, '
            'and this property is Fibonacci-specific.'
        ),
        'falsified_if': (
            'Conservation fractions of physics matches are indistinguishable '
            'from non-matches, OR alternative sequences show identical pattern.'
        ),
        'falsified': pass_count < 2,
        'assessment': f"{pass_count}/{total} tests pass.",
    }

    save_results(results, 'exp_18_conservation_predictions')
    return results


if __name__ == '__main__':
    main()
