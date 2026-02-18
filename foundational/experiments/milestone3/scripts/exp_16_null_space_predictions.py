"""
exp_16: Null Space Predictions — Making the Framework Predictive

HYPOTHESIS: The stoichiometric null space contains UNUSED "allowed reactions"
that correspond to physics relationships not yet identified in our catalog.
If the framework is genuinely predictive (not just retrodictive), these null-
space-preferred Fibonacci expressions should match measurable quantities.

MOTIVATION:
  exp_13 established a 5-constraint stoichiometric matrix S with a 6-dim null
  space. exp_14 showed the Fibonacci sequence at the 99.98th percentile for
  multi-target matching. exp_15 confirmed the SEC cost hierarchy (r=0.86).

  But EVERY result so far is retrodiction: we know the formula, then check
  it against the framework. The honest assessment identified "no predictions"
  as the critical gap.

  This experiment inverts the workflow:
    1. Extract null space basis vectors
    2. Identify the simplest UNUSED Fibonacci combinations they prefer
    3. Compute what physical quantity each combination produces
    4. Check against PDG/CODATA values NOT in our catalog

  If even ONE novel prediction matches at <1%, it transforms the framework
  from "interesting pattern" to "predictive theory."

TESTS:
  Test 1 — Null Space Mining: Extract all preferred Fibonacci index pairs from
           the null space. Rank by null-space alignment. Filter out known
           formulas. List the top-10 "predicted" expressions.

  Test 2 — Novel Ratio Scan: For each predicted F_a/F_b ratio, search against
           a comprehensive set of PDG/CODATA constants to find matches
           within 1%. These are GENUINE predictions (not fitted).

  Test 3 — Novel Product Scan: For predicted multi-index combinations
           (F_a·F_b/F_c etc.), search against measured ratios and
           dimensionless constants. Look for <1% matches.

  Test 4 — SEC Cost Prediction: The framework predicts that any new formula's
           SEC cost should follow the established linear relationship
           (~55.7 units per Fibonacci index). Verify: do the newly found
           formulas obey this relationship?

SOURCES:
  - exp_13 (stoichiometric matrix, null space)
  - exp_14 (atomic decomposition matrix)
  - exp_15 (SEC cost = 55.7 per index, r=0.86)
  - PDG 2024 (https://pdg.lbl.gov)
  - CODATA 2018 fundamental constants
"""

import sys
import os
import math
import numpy as np
from itertools import combinations, product as iterproduct
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results

# =====================================================================
# Reference values — KNOWN formulas (these don't count as predictions)
# =====================================================================
KNOWN_TARGETS = {
    'sin²θ_W':    0.23122,
    'Koide Q':    0.666661,
    'She-Lev β':  2.0 / 3.0,
    'ν_WF':       0.6299709,
    'α_s':        0.1180,
    'Cabibbo':    13.04,       # degrees
    'μ/e':        206.7682830,
    'α_em':       0.0072973525693,
    'p/e':        1836.15267343,
    'τ/e':        3477.48,
    '1/φ':        INV_PHI,
    'φ':          PHI,
}

# =====================================================================
# NOVEL targets — these are NOT in our catalog. Predictions checked here.
# All dimensionless or in natural units.
# =====================================================================
NOVEL_TARGETS = {
    # Electroweak
    'ρ_parameter':        1.00038,        # electroweak rho (PDG)
    'G_F_natural':        1.1663788e-5,   # Fermi constant (GeV^-2) — dimensionless proxy
    # CKM matrix elements
    'V_us':               0.2243,         # |V_us| (Cabibbo element)
    'V_cb':               0.0422,         # |V_cb|
    'V_ub':               0.00394,        # |V_ub|
    'V_td':               0.00814,        # |V_td|
    'V_ts':               0.0400,         # |V_ts|
    'Jarlskog_J':         3.08e-5,        # Jarlskog invariant
    # Quark mass ratios (MS-bar at 2 GeV, PDG 2024)
    'm_u/m_d':            0.474,          # up/down mass ratio
    'm_s/m_d':            20.2,           # strange/down mass ratio
    'm_c/m_s':            11.7,           # charm/strange mass ratio
    'm_b/m_c':            3.41,           # bottom/charm mass ratio
    'm_t/m_b':            41.3,           # top/bottom mass ratio (pole masses)
    # Lepton mass ratios
    'm_mu/m_e':           206.7682830,    # (already known — include for completeness)
    'm_tau/m_mu':         16.8170,        # tau/muon mass ratio
    # Neutrino mixing (PMNS)
    'sin²θ_12':           0.307,          # solar angle
    'sin²θ_23':           0.546,          # atmospheric angle
    'sin²θ_13':           0.0220,         # reactor angle
    'Δm²_21/Δm²_31':     0.0297,         # mass squared ratio (solar/atm)
    # QCD
    'Λ_QCD/m_p':          0.217,          # QCD scale / proton mass
    # Mathematical/physics constants
    'α_em(M_Z)':          1.0/127.951,    # running α at M_Z
    'sin²θ_eff':          0.23155,        # effective weak mixing (leptonic)
    'α_s(M_τ)':           0.332,          # strong coupling at tau mass
    # Cosmological (dimensionless)
    'Ω_b':                0.0493,         # baryon density parameter
    'Ω_c':                0.265,          # cold dark matter density
    'Ω_Λ':                0.685,          # dark energy density
    'n_s':                0.965,          # scalar spectral index
    'σ_8':                0.811,          # matter fluctuation amplitude
    # Thermodynamic
    'ζ(3)':               1.2020569,      # Apéry's constant (appears in QFT)
    'ζ(5)':               1.0369278,      # Riemann zeta at 5
    # Turbulence
    'von_Karman':         0.41,           # von Kármán constant
    'Kolmogorov_C_K':     1.5,            # Kolmogorov constant
}


def pct_error(pred, meas):
    """Percentage error."""
    if meas == 0:
        return float('inf')
    return abs(pred - meas) / abs(meas) * 100


# =====================================================================
# Matrix builders (from exp_13 and exp_14)
# =====================================================================
FIB_INDICES = list(range(2, 13))  # F₂=1 ... F₁₂=144
FIB_VALUES = [FIB[i] for i in FIB_INDICES]
N_SPECIES = len(FIB_INDICES)


def build_exp13_matrix():
    """5-constraint matrix from exp_13."""
    S = np.zeros((5, N_SPECIES))
    idx = {n: FIB_INDICES.index(n) for n in FIB_INDICES}
    S[0] = FIB_VALUES                                # PAC magnitude
    S[1] = FIB_INDICES                               # Hierarchy depth
    S[2] = [n % 3 for n in FIB_INDICES]              # E-I-S cycle
    S[3] = [n % 2 for n in FIB_INDICES]              # Parity
    S[4, idx[5]] = -1; S[4, idx[6]] = -1; S[4, idx[7]] = 1  # Gauge closure
    return S


def build_exp14_matrix():
    """6-constraint matrix from exp_14 (atomic decomposition)."""
    atom_a = {2: 1, 3: 0}
    atom_b = {2: 0, 3: 1}
    for n in range(4, 13):
        atom_a[n] = atom_a[n-1] + atom_a[n-2]
        atom_b[n] = atom_b[n-1] + atom_b[n-2]

    S = np.zeros((6, N_SPECIES))
    idx = {n: FIB_INDICES.index(n) for n in FIB_INDICES}
    S[0] = [atom_a[n] for n in FIB_INDICES]
    S[1] = [atom_b[n] for n in FIB_INDICES]
    S[2, idx[4]] = -1; S[2, idx[5]] = -1; S[2, idx[6]] = 1
    S[3, idx[5]] = -1; S[3, idx[6]] = -1; S[3, idx[7]] = 1
    S[4, idx[4]] = FIB[4]; S[4, idx[6]] = -1
    S[5] = FIB_INDICES
    return S


def get_null_space(S):
    """Return null space basis from SVD."""
    U, sigma, Vt = np.linalg.svd(S)
    tol = 1e-10
    rank = int(np.sum(sigma > tol * sigma[0]))
    null_dim = N_SPECIES - rank
    null_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, N_SPECIES))
    return null_basis, rank, null_dim


def formula_vector(indices):
    """Create unit indicator vector for given Fibonacci indices."""
    vec = np.zeros(N_SPECIES)
    for i in indices:
        if i in FIB_INDICES:
            vec[FIB_INDICES.index(i)] = 1.0
    return vec


def null_alignment(vec, null_basis):
    """Fraction of vector that lies in null space (0 = orthogonal, 1 = fully in)."""
    n = np.linalg.norm(vec)
    if n < 1e-12 or null_basis.shape[0] == 0:
        return 0.0
    proj = null_basis @ vec
    return float(np.linalg.norm(proj) / n)


# =====================================================================
# Known formula index sets (to exclude from predictions)
# =====================================================================
KNOWN_INDEX_SETS = [
    frozenset([4, 7]),       # sin²θ_W, Cabibbo
    frozenset([2, 3]),       # Koide
    frozenset([3, 4]),       # She-Lev, ν_WF
    frozenset([4, 6]),       # α_s
    frozenset([4, 6, 7]),    # μ/e
    frozenset([3, 4, 7, 10]),# α_em
    frozenset([4, 6, 9, 12]),# p/e
    frozenset([4, 5, 7, 11]),# τ/e
]


def is_known_set(indices):
    """Check if this index combination is already used in a known formula."""
    return frozenset(indices) in KNOWN_INDEX_SETS


# =====================================================================
# Look-Elsewhere Monte Carlo Infrastructure
# =====================================================================

# Targets near simple constants give trivial matches — flag them
TRIVIAL_VALUES = [1.0, PHI, INV_PHI, 2.0, 0.5, 3.0, 1.5]
TRIVIAL_PROXIMITY = 0.02  # within 2% of trivial value


def is_trivial_target(target_val):
    """Check if target value is trivially close to a simple constant."""
    for tv in TRIVIAL_VALUES:
        if tv == 0:
            continue
        if abs(target_val - tv) / abs(tv) < TRIVIAL_PROXIMITY:
            return True
    return False


def generate_random_sequence(rng, n=11, max_val=200):
    """Generate random strictly increasing integer sequence."""
    vals = sorted(rng.choice(np.arange(1, max_val + 1), size=n, replace=False))
    return [int(v) for v in vals]


def scan_ratio_targets(seq, targets, threshold_pct=1.0):
    """Return set of target names matched by ratio templates for a sequence."""
    matched = set()
    n = len(seq)
    for ai in range(n):
        for bi in range(n):
            if ai == bi or seq[bi] == 0:
                continue
            values = [
                seq[ai] / seq[bi],
                seq[ai] / (seq[bi] * XI_BALANCE),
                seq[ai] / (seq[bi] * PHI),
                seq[ai] * XI_BALANCE / seq[bi],
            ]
            for val in values:
                for tname, tval in targets.items():
                    if tval == 0:
                        continue
                    if abs(val - tval) / abs(tval) * 100 < threshold_pct:
                        matched.add(tname)
    return matched


def scan_product_targets(seq, targets, threshold_pct=1.0):
    """Return set of target names matched by product templates."""
    matched = set()
    n = len(seq)
    for ai in range(n):
        for bi in range(n):
            for ci in range(n):
                if len({ai, bi, ci}) < 3 or seq[ci] == 0:
                    continue
                vals = [seq[ai] * seq[bi] / seq[ci]]
                if seq[bi] * seq[ci] > 0:
                    vals.append(seq[ai]**2 / (seq[bi] * seq[ci]))
                vals.append(seq[ai] * seq[bi] / (seq[ci] * XI_BALANCE))
                if seq[bi] * seq[ci] > 0:
                    vals.append(seq[ai] / (seq[bi] * seq[ci]))
                for val in vals:
                    if not math.isfinite(val) or val <= 0:
                        continue
                    for tname, tval in targets.items():
                        if tval == 0:
                            continue
                        if abs(val - tval) / abs(tval) * 100 < threshold_pct:
                            matched.add(tname)
    return matched


# =====================================================================
# Test 1: Null Space Mining
# =====================================================================
def test_1_null_space_mining():
    """
    Extract all Fibonacci index pairs/triples from the null space and rank
    by alignment. Filter out known formulas. These are the framework's
    PREDICTIONS: Fibonacci combinations it prefers but we haven't used.
    """
    print("\n" + "="*70)
    print("TEST 1: Null Space Mining — Extracting Predictions")
    print("="*70)

    S13 = build_exp13_matrix()
    S14 = build_exp14_matrix()
    null13, rank13, ndim13 = get_null_space(S13)
    null14, rank14, ndim14 = get_null_space(S14)

    print(f"\n  exp_13 matrix: rank={rank13}, null dim={ndim13}")
    print(f"  exp_14 matrix: rank={rank14}, null dim={ndim14}")

    # --- Score all index combinations ---
    candidates = []

    # Pairs
    for a, b in combinations(FIB_INDICES, 2):
        vec = formula_vector([a, b])
        align13 = null_alignment(vec, null13)
        align14 = null_alignment(vec, null14)
        avg_align = (align13 + align14) / 2
        known = is_known_set([a, b])
        candidates.append({
            'indices': [a, b], 'type': 'pair',
            'align_13': align13, 'align_14': align14,
            'avg_align': avg_align, 'known': known,
        })

    # Triples
    for a, b, c in combinations(FIB_INDICES, 3):
        vec = formula_vector([a, b, c])
        align13 = null_alignment(vec, null13)
        align14 = null_alignment(vec, null14)
        avg_align = (align13 + align14) / 2
        known = is_known_set([a, b, c])
        candidates.append({
            'indices': [a, b, c], 'type': 'triple',
            'align_13': align13, 'align_14': align14,
            'avg_align': avg_align, 'known': known,
        })

    # Quadruples (only include highest-alignment ones)
    for combo in combinations(FIB_INDICES, 4):
        vec = formula_vector(combo)
        align13 = null_alignment(vec, null13)
        align14 = null_alignment(vec, null14)
        avg_align = (align13 + align14) / 2
        if avg_align > 0.6:  # only include well-aligned quadruples
            known = is_known_set(combo)
            candidates.append({
                'indices': list(combo), 'type': 'quad',
                'align_13': align13, 'align_14': align14,
                'avg_align': avg_align, 'known': known,
            })

    # Sort by average alignment (descending)
    candidates.sort(key=lambda x: x['avg_align'], reverse=True)

    # --- Show known formulas and their ranking ---
    print(f"\n  Known formulas (validation — should rank high):")
    known_ranks = []
    for i, c in enumerate(candidates):
        if c['known']:
            known_ranks.append(i + 1)
            print(f"    Rank {i+1:3d}: {c['indices']}  align={c['avg_align']:.4f}  ({c['type']})")

    # --- Top novel predictions ---
    novel = [c for c in candidates if not c['known']]
    print(f"\n  Top 20 NOVEL predictions (not in known catalog):")
    print(f"  {'Rank':>4s}  {'Indices':20s}  {'Type':6s}  {'Align_13':>8s}  {'Align_14':>8s}  {'Avg':>6s}")
    top_novel = novel[:20]
    for i, c in enumerate(top_novel):
        idx_str = str(c['indices'])
        fib_str = '×'.join(f'F_{n}({FIB[n]})' for n in c['indices'])
        print(f"  {i+1:4d}  {idx_str:20s}  {c['type']:6s}  {c['align_13']:8.4f}  {c['align_14']:8.4f}  {c['avg_align']:6.4f}")
        if i < 5:
            print(f"        = {fib_str}")

    # --- Dimensionality correction ---
    # With null dim = d out of N_SPECIES, a random indicator vector has
    # expected alignment ≈ √(d/N). alignment > 0.5 is EXPECTED for most
    # combinations when the null space is > 25% of the vector space.
    n_total = len(candidates)
    avg_null_dim = (ndim13 + ndim14) / 2
    expected_align = math.sqrt(avg_null_dim / N_SPECIES)
    print(f"\n  --- Dimensionality Correction ---")
    print(f"  Null space covers {avg_null_dim:.0f}/{N_SPECIES} = "
          f"{avg_null_dim/N_SPECIES*100:.0f}% of vector space")
    print(f"  Expected random alignment: ~{expected_align:.3f}")
    print(f"  → alignment > 0.5 is BELOW random expectation!")

    # MC: random indicator baseline (5000 samples)
    rng_t1 = np.random.default_rng(42)
    random_aligns = []
    for _ in range(5000):
        k = int(rng_t1.choice([2, 3, 4]))
        idx_sample = rng_t1.choice(FIB_INDICES, size=k, replace=False).tolist()
        vec = formula_vector(idx_sample)
        a13 = null_alignment(vec, null13)
        a14 = null_alignment(vec, null14)
        random_aligns.append((a13 + a14) / 2)

    rand_mean = float(np.mean(random_aligns))
    rand_std = float(np.std(random_aligns))

    known_aligns = [c['avg_align'] for c in candidates if c['known']]
    known_mean_align = float(np.mean(known_aligns)) if known_aligns else 0
    z_score = float((known_mean_align - rand_mean) / rand_std) if rand_std > 0 else 0

    known_avg_rank = float(np.mean(known_ranks)) if known_ranks else n_total
    n_novel_high = sum(1 for c in novel if c['avg_align'] > 0.5)

    print(f"  Random indicator mean alignment: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  Known formula mean alignment: {known_mean_align:.4f}")
    print(f"  z-score (known vs random): {z_score:.2f}")

    # REVISED PASS: known formulas above random baseline (z > 1.5)
    # Original threshold (top 30%) was uninformative — null space is ~55% of space
    passed = z_score > 1.5
    print(f"\n  Known formulas avg rank: {known_avg_rank:.1f}/{n_total} "
          f"(top {known_avg_rank/n_total*100:.1f}%)")
    print(f"  Novel combinations with align > 0.5: {n_novel_high}")
    print(f"  PASS: {passed} (z > 1.5: known formulas significantly above random indicators)")
    if not passed:
        print(f"  NOTE: The null space is too large ({avg_null_dim:.0f}/{N_SPECIES} dims) to")
        print(f"  be a useful filter. Most combinations project well onto it.")
        print(f"  This is a limitation of the stoichiometric matrix, not a surprise.")

    return {
        'n_total_candidates': n_total,
        'n_novel': len(novel),
        'n_novel_high_align': n_novel_high,
        'known_avg_rank': float(known_avg_rank),
        'known_mean_alignment': float(known_mean_align),
        'random_mean_alignment': float(rand_mean),
        'random_std_alignment': float(rand_std),
        'z_score_vs_random': float(z_score),
        'expected_random_alignment': float(expected_align),
        'top_10_novel': [{
            'indices': c['indices'], 'avg_align': c['avg_align']
        } for c in top_novel[:10]],
        'passed': passed,
    }


# =====================================================================
# Test 2: Novel Ratio Scan
# =====================================================================
def test_2_novel_ratio_scan():
    """
    For every NOVEL Fibonacci pair (not in known catalog), compute:
      - F_a/F_b (simple ratio)
      - F_a/(F_b·Ξ)  (E-I-S decomposed ratio)
      - F_a/(F_b·φ)  (golden-scaled ratio)
    Then check each against the NOVEL_TARGETS dictionary.

    Any match at <1% is a GENUINE PREDICTION.
    """
    print("\n" + "="*70)
    print("TEST 2: Novel Ratio Scan — Searching for Predictions")
    print("="*70)

    S13 = build_exp13_matrix()
    null13, _, ndim13 = get_null_space(S13)

    predictions = []
    n_checked = 0

    # All ordered pairs
    for a in FIB_INDICES:
        for b in FIB_INDICES:
            if a == b or FIB[b] == 0:
                continue

            pair_set = frozenset([a, b])
            if pair_set in KNOWN_INDEX_SETS:
                continue

            vec = formula_vector([a, b])
            align = null_alignment(vec, null13)

            # Compute candidate values
            ratio_plain = FIB[a] / FIB[b]
            ratio_xi = FIB[a] / (FIB[b] * XI_BALANCE)
            ratio_phi = FIB[a] / (FIB[b] * PHI)
            ratio_xi_inv = FIB[a] * XI_BALANCE / FIB[b]

            templates = {
                f'F_{a}/F_{b}':     ratio_plain,
                f'F_{a}/(F_{b}·Ξ)': ratio_xi,
                f'F_{a}/(F_{b}·φ)': ratio_phi,
                f'F_{a}·Ξ/F_{b}':   ratio_xi_inv,
            }

            for template_name, value in templates.items():
                n_checked += 1
                for target_name, target_val in NOVEL_TARGETS.items():
                    if target_val == 0:
                        continue
                    err = pct_error(value, target_val)
                    if err < 1.0:
                        predictions.append({
                            'expression': template_name,
                            'indices': [a, b],
                            'value': value,
                            'target': target_name,
                            'target_val': target_val,
                            'error_pct': err,
                            'null_align': align,
                            'type': 'ratio',
                        })

    # Sort by error
    predictions.sort(key=lambda x: x['error_pct'])

    print(f"\n  Checked {n_checked} ratio expressions against {len(NOVEL_TARGETS)} targets")
    print(f"  Found {len(predictions)} matches at <1%")

    if predictions:
        print(f"\n  {'Expression':25s}  {'Value':>12s}  {'Target':20s}  {'Measured':>12s}  {'Err%':>8s}  {'Align':>6s}")
        print(f"  {'─'*25}  {'─'*12}  {'─'*20}  {'─'*12}  {'─'*8}  {'─'*6}")
        seen = set()
        unique_preds = []
        for p in predictions:
            key = (p['target'], p['expression'])
            if key not in seen:
                seen.add(key)
                unique_preds.append(p)
                print(f"  {p['expression']:25s}  {p['value']:12.8f}  {p['target']:20s}  "
                      f"{p['target_val']:12.8f}  {p['error_pct']:8.4f}  {p['null_align']:6.3f}")
                if len(unique_preds) >= 20:
                    break
    else:
        unique_preds = []

    # --- Categorize by quality ---
    high_align = [p for p in unique_preds if p['null_align'] > 0.5]
    print(f"\n  High-alignment predictions (null align > 0.5): {len(high_align)}")
    for p in high_align:
        print(f"    ★ {p['expression']} = {p['value']:.8f} → {p['target']} ({p['error_pct']:.4f}%)")

    # --- Trivial match filtering ---
    non_trivial = [p for p in unique_preds if not is_trivial_target(p['target_val'])]
    trivial = [p for p in unique_preds if is_trivial_target(p['target_val'])]

    print(f"\n  --- Trivial Match Filter ---")
    print(f"  Non-trivial predictions: {len(non_trivial)}")
    print(f"  Trivial (near 1.0, φ, 1/φ, 0.5, 2.0, 1.5, 3.0): {len(trivial)}")
    if trivial:
        for p in trivial[:5]:
            print(f"    ⚠ {p['expression']} → {p['target']} = {p['target_val']}")

    # --- Look-elsewhere Monte Carlo ---
    print(f"\n  --- Look-Elsewhere Correction (Monte Carlo) ---")
    non_trivial_targets = {k: v for k, v in NOVEL_TARGETS.items()
                           if not is_trivial_target(v)}
    fib_nt_hits = len(scan_ratio_targets(FIB_VALUES, non_trivial_targets))

    N_MC_RATIO = 500
    rng_t2 = np.random.default_rng(42)
    mc_hits = []
    for _ in range(N_MC_RATIO):
        rand_seq = generate_random_sequence(rng_t2)
        mc_hits.append(len(scan_ratio_targets(rand_seq, non_trivial_targets)))

    mc_mean = float(np.mean(mc_hits))
    mc_std = float(np.std(mc_hits))
    mc_p = float(np.mean([h >= fib_nt_hits for h in mc_hits]))
    enrichment = fib_nt_hits / mc_mean if mc_mean > 0 else float('inf')

    print(f"  Fibonacci non-trivial target hits: {fib_nt_hits}")
    print(f"  Random sequence hits: {mc_mean:.1f} ± {mc_std:.1f}")
    print(f"  Enrichment: {enrichment:.2f}×")
    print(f"  p-value: {mc_p:.4f}")
    print(f"  Look-elsewhere: {'SURVIVES' if mc_p < 0.10 else 'DOES NOT SURVIVE'}")

    # PASS requires: ≥1 non-trivial high-align prediction AND survives look-elsewhere
    n_genuine_nt = sum(1 for p in non_trivial if p['null_align'] > 0.5)
    passed_original = n_genuine_nt >= 1
    passed_mc = mc_p < 0.10
    passed = passed_original and passed_mc

    print(f"\n  Non-trivial high-align predictions: {n_genuine_nt}")
    print(f"  Original criterion (≥1 genuine): {passed_original}")
    print(f"  Look-elsewhere (p < 0.10): {passed_mc}")
    print(f"  PASS: {passed}")

    return {
        'n_checked': n_checked,
        'n_matches': len(predictions),
        'n_unique': len(unique_preds),
        'n_non_trivial': len(non_trivial),
        'n_trivial': len(trivial),
        'n_high_align': len(high_align),
        'mc_fib_hits': fib_nt_hits,
        'mc_random_mean': mc_mean,
        'mc_random_std': mc_std,
        'mc_enrichment': float(enrichment) if math.isfinite(enrichment) else None,
        'mc_p_value': mc_p,
        'predictions': [{
            'expression': p['expression'],
            'value': float(p['value']),
            'target': p['target'],
            'target_val': float(p['target_val']),
            'error_pct': float(p['error_pct']),
            'null_align': float(p['null_align']),
            'trivial': is_trivial_target(p['target_val']),
        } for p in unique_preds[:15]],
        'passed': passed,
    }


# =====================================================================
# Test 3: Novel Product/Composite Scan
# =====================================================================
def test_3_novel_product_scan():
    """
    For novel index triples and quads, compute multi-index expressions
    and check against NOVEL_TARGETS. Templates:
      - F_a · F_b / F_c
      - F_a · F_b · F_c  (for large targets)
      - F_a² / (F_b · F_c)
      - F_a · F_b / (F_c · Ξ)
    """
    print("\n" + "="*70)
    print("TEST 3: Novel Product Scan — Multi-Index Predictions")
    print("="*70)

    S13 = build_exp13_matrix()
    null13, _, _ = get_null_space(S13)

    predictions = []
    n_checked = 0

    # Three-index combinations: F_a · F_b / F_c
    for a in FIB_INDICES:
        for b in FIB_INDICES:
            for c in FIB_INDICES:
                if len(set([a, b, c])) < 3 or FIB[c] == 0:
                    continue

                idx_set = frozenset([a, b, c])
                if idx_set in KNOWN_INDEX_SETS:
                    continue

                vec = formula_vector([a, b, c])
                align = null_alignment(vec, null13)

                # Only check higher-alignment combinations (saves time)
                if align < 0.3:
                    continue

                templates = {}
                templates[f'F_{a}·F_{b}/F_{c}'] = FIB[a] * FIB[b] / FIB[c]
                if FIB[b] * FIB[c] > 0:
                    templates[f'F_{a}²/(F_{b}·F_{c})'] = FIB[a]**2 / (FIB[b] * FIB[c])
                templates[f'F_{a}·F_{b}/(F_{c}·Ξ)'] = FIB[a] * FIB[b] / (FIB[c] * XI_BALANCE)
                templates[f'F_{a}/(F_{b}·F_{c})'] = FIB[a] / (FIB[b] * FIB[c])

                for template_name, value in templates.items():
                    n_checked += 1
                    if not math.isfinite(value) or value <= 0:
                        continue
                    for target_name, target_val in NOVEL_TARGETS.items():
                        if target_val == 0:
                            continue
                        err = pct_error(value, target_val)
                        if err < 1.0:
                            predictions.append({
                                'expression': template_name,
                                'indices': sorted(set([a, b, c])),
                                'value': value,
                                'target': target_name,
                                'target_val': target_val,
                                'error_pct': err,
                                'null_align': align,
                                'type': 'product',
                            })

    predictions.sort(key=lambda x: x['error_pct'])

    print(f"\n  Checked {n_checked} product expressions")
    print(f"  Found {len(predictions)} matches at <1%")

    # Deduplicate by (target, indices)
    seen = set()
    unique_preds = []
    for p in predictions:
        key = (p['target'], tuple(p['indices']), p['expression'])
        if key not in seen:
            seen.add(key)
            unique_preds.append(p)

    if unique_preds:
        print(f"\n  Unique predictions: {len(unique_preds)}")
        print(f"\n  {'Expression':30s}  {'Value':>12s}  {'Target':20s}  {'Measured':>12s}  {'Err%':>8s}  {'Align':>6s}")
        print(f"  {'─'*30}  {'─'*12}  {'─'*20}  {'─'*12}  {'─'*8}  {'─'*6}")
        for p in unique_preds[:25]:
            print(f"  {p['expression']:30s}  {p['value']:12.6f}  {p['target']:20s}  "
                  f"{p['target_val']:12.6f}  {p['error_pct']:8.4f}  {p['null_align']:6.3f}")

    high_align = [p for p in unique_preds if p['null_align'] > 0.5]
    print(f"\n  High-alignment product predictions: {len(high_align)}")
    for p in high_align[:10]:
        print(f"    ★ {p['expression']} = {p['value']:.6f} → {p['target']} ({p['error_pct']:.4f}%)")

    # --- Trivial match filtering ---
    non_trivial = [p for p in unique_preds if not is_trivial_target(p['target_val'])]
    trivial = [p for p in unique_preds if is_trivial_target(p['target_val'])]

    print(f"\n  --- Trivial Match Filter ---")
    print(f"  Non-trivial: {len(non_trivial)}, Trivial: {len(trivial)}")
    if trivial:
        for p in trivial[:5]:
            print(f"    ⚠ {p['expression']} → {p['target']} = {p['target_val']}")

    # --- Look-elsewhere Monte Carlo ---
    print(f"\n  --- Look-Elsewhere Correction (Monte Carlo) ---")
    non_trivial_targets = {k: v for k, v in NOVEL_TARGETS.items()
                           if not is_trivial_target(v)}
    fib_nt_hits = len(scan_product_targets(FIB_VALUES, non_trivial_targets))

    N_MC_PROD = 200  # fewer iterations (O(n³) is slower)
    rng_t3 = np.random.default_rng(42)
    mc_hits = []
    for i in range(N_MC_PROD):
        rand_seq = generate_random_sequence(rng_t3)
        mc_hits.append(len(scan_product_targets(rand_seq, non_trivial_targets)))
        if (i + 1) % 50 == 0:
            print(f"    MC progress: {i+1}/{N_MC_PROD}")

    mc_mean = float(np.mean(mc_hits))
    mc_std = float(np.std(mc_hits))
    mc_p = float(np.mean([h >= fib_nt_hits for h in mc_hits]))
    enrichment = fib_nt_hits / mc_mean if mc_mean > 0 else float('inf')

    print(f"  Fibonacci non-trivial target hits: {fib_nt_hits}")
    print(f"  Random sequence hits: {mc_mean:.1f} ± {mc_std:.1f}")
    print(f"  Enrichment: {enrichment:.2f}×")
    print(f"  p-value: {mc_p:.4f}")

    n_genuine_nt = sum(1 for p in non_trivial if p['null_align'] > 0.5)
    passed_original = n_genuine_nt >= 1
    passed_mc = mc_p < 0.10
    passed = passed_original and passed_mc

    print(f"\n  Non-trivial high-align: {n_genuine_nt}")
    print(f"  PASS: {passed} (≥1 non-trivial high-align AND look-elsewhere p < 0.10)")

    return {
        'n_checked': n_checked,
        'n_matches': len(predictions),
        'n_unique': len(unique_preds),
        'n_non_trivial': len(non_trivial),
        'n_trivial': len(trivial),
        'n_high_align': len(high_align),
        'mc_fib_hits': fib_nt_hits,
        'mc_random_mean': mc_mean,
        'mc_random_std': mc_std,
        'mc_enrichment': float(enrichment) if math.isfinite(enrichment) else None,
        'mc_p_value': mc_p,
        'top_predictions': [{
            'expression': p['expression'],
            'indices': p['indices'],
            'value': float(p['value']),
            'target': p['target'],
            'target_val': float(p['target_val']),
            'error_pct': float(p['error_pct']),
            'null_align': float(p['null_align']),
            'trivial': is_trivial_target(p['target_val']),
        } for p in unique_preds[:15]],
        'passed': passed,
    }


# =====================================================================
# Test 4: Combined Look-Elsewhere Enrichment
# =====================================================================
def test_4_sec_cost_prediction():
    """
    REVISED: The original SEC cost law test failed because ‖S·v‖
    (stoichiometric violation norm) ≠ SEC symbolic cost (exp_15).
    The two measure fundamentally different things.

    Replaced with the definitive look-elsewhere test: how many
    non-trivial physics constants does Fibonacci match compared to
    random integer sequences using the SAME templates?

    This is the headline number. If Fibonacci matches significantly
    more targets than random, the predictions are non-trivial.
    If not, the matches are numerological artifacts.
    """
    print("\n" + "="*70)
    print("TEST 4: Combined Look-Elsewhere Enrichment")
    print("="*70)

    non_trivial_targets = {k: v for k, v in NOVEL_TARGETS.items()
                           if not is_trivial_target(v)}
    n_nt = len(non_trivial_targets)
    n_trivial_excluded = len(NOVEL_TARGETS) - n_nt

    print(f"\n  Non-trivial targets: {n_nt} (excluded {n_trivial_excluded} trivial)")
    print(f"  Trivial exclusions (within 2% of {TRIVIAL_VALUES}):")
    for k, v in NOVEL_TARGETS.items():
        if is_trivial_target(v):
            print(f"    ⚠ {k} = {v}")

    # --- Fibonacci combined scan ---
    print(f"\n  Fibonacci ratio scan...")
    fib_ratio = scan_ratio_targets(FIB_VALUES, non_trivial_targets)
    print(f"  Fibonacci product scan...")
    fib_product = scan_product_targets(FIB_VALUES, non_trivial_targets)
    fib_combined = fib_ratio | fib_product
    fib_ratio_only = fib_ratio - fib_product
    fib_product_only = fib_product - fib_ratio
    fib_both = fib_ratio & fib_product

    print(f"\n  Fibonacci results:")
    print(f"    Ratio matches: {len(fib_ratio)} unique targets")
    print(f"    Product matches: {len(fib_product)} unique targets")
    print(f"    Combined (union): {len(fib_combined)} unique targets")
    print(f"    Overlap: {len(fib_both)} in both, "
          f"{len(fib_ratio_only)} ratio-only, {len(fib_product_only)} product-only")

    for tname in sorted(fib_combined):
        source = ("ratio+product" if tname in fib_both
                  else ("ratio" if tname in fib_ratio else "product"))
        print(f"      {tname}: {non_trivial_targets[tname]} [{source}]")

    # --- Monte Carlo: random integer sequences ---
    N_MC = 200
    rng = np.random.default_rng(42)
    mc_combined = []
    mc_ratio_ct = []
    mc_product_ct = []

    print(f"\n  Monte Carlo ({N_MC} random integer sequences, range 1-200)...")
    for i in range(N_MC):
        rand_seq = generate_random_sequence(rng)
        r_match = scan_ratio_targets(rand_seq, non_trivial_targets)
        p_match = scan_product_targets(rand_seq, non_trivial_targets)
        mc_ratio_ct.append(len(r_match))
        mc_product_ct.append(len(p_match))
        mc_combined.append(len(r_match | p_match))
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{N_MC} "
                  f"(avg combined: {np.mean(mc_combined):.1f})")

    mc_mean = float(np.mean(mc_combined))
    mc_std = float(np.std(mc_combined))
    mc_p = float(np.mean([h >= len(fib_combined) for h in mc_combined]))
    enrichment = len(fib_combined) / mc_mean if mc_mean > 0 else float('inf')

    mc_r_mean = float(np.mean(mc_ratio_ct))
    mc_p_mean = float(np.mean(mc_product_ct))
    mc_max = max(mc_combined)

    print(f"\n  {'='*58}")
    print(f"  LOOK-ELSEWHERE RESULTS")
    print(f"  {'='*58}")
    print(f"  Fibonacci matched:   {len(fib_combined):2d} / {n_nt} non-trivial targets")
    print(f"  Random matched:      {mc_mean:.1f} ± {mc_std:.1f} targets "
          f"(max {mc_max})")
    print(f"  Enrichment:          {enrichment:.2f}×")
    print(f"  p-value:             {mc_p:.4f}")
    print(f"  {'='*58}")
    print(f"  Breakdown:")
    print(f"    Ratio:   Fib {len(fib_ratio):2d} vs random {mc_r_mean:.1f}")
    print(f"    Product: Fib {len(fib_product):2d} vs random {mc_p_mean:.1f}")

    # PASS: enrichment > 1.5 AND p < 0.05
    passed = enrichment > 1.5 and mc_p < 0.05
    print(f"\n  PASS: {passed} (enrichment > 1.5× AND p < 0.05)")
    if passed:
        print(f"  → Fibonacci matches significantly more physics constants than random")
    else:
        partially = mc_p < 0.10
        print(f"  → Fibonacci matches are "
              f"{'partially' if partially else 'NOT'} "
              f"distinguishable from random integer sequences")

    return {
        'n_non_trivial_targets': n_nt,
        'fib_ratio_hits': len(fib_ratio),
        'fib_product_hits': len(fib_product),
        'fib_combined_hits': len(fib_combined),
        'fib_matched_targets': sorted(list(fib_combined)),
        'mc_n': N_MC,
        'mc_combined_mean': mc_mean,
        'mc_combined_std': mc_std,
        'mc_combined_max': mc_max,
        'mc_ratio_mean': mc_r_mean,
        'mc_product_mean': mc_p_mean,
        'enrichment': float(enrichment) if math.isfinite(enrichment) else None,
        'mc_p_value': mc_p,
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_16_null_space_predictions',
        'Null space predictions — making the framework predictive',
        paper='Paper 4',
        section='§predictions'
    )

    results = {'metadata': meta, 'tests': {}}

    results['tests']['test_1_mining']       = test_1_null_space_mining()
    results['tests']['test_2_ratio_scan']   = test_2_novel_ratio_scan()
    results['tests']['test_3_product_scan'] = test_3_novel_product_scan()
    results['tests']['test_4_sec_cost']     = test_4_sec_cost_prediction()

    # --- Final synthesis ---
    print("\n" + "="*70)
    print("  SYNTHESIS: Null Space Predictions")
    print("="*70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:35s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    # Collect all predictions
    all_preds = []
    for key in ['test_2_ratio_scan', 'test_3_product_scan']:
        test = results['tests'].get(key, {})
        preds = test.get('predictions', test.get('top_predictions', []))
        all_preds.extend(preds)

    n_predictions = len(all_preds)
    n_sub_half = sum(1 for p in all_preds if p.get('error_pct', 100) < 0.5)

    print(f"\n  Total novel predictions found: {n_predictions}")
    print(f"  Predictions with <0.5% error: {n_sub_half}")

    if all_preds:
        print(f"\n  TOP PREDICTIONS (best novel matches):")
        all_preds.sort(key=lambda x: x.get('error_pct', 100))
        for i, p in enumerate(all_preds[:10]):
            print(f"    {i+1}. {p.get('expression','?')} = {p.get('value',0):.8f} "
                  f"→ {p.get('target','?')} = {p.get('target_val',0):.8f} "
                  f"({p.get('error_pct',0):.4f}%)")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  INTERPRETATION                                             │")
    print(f"  │                                                              │")
    print(f"  │  If predictions match at <1%:                                │")
    print(f"  │    → Framework is genuinely PREDICTIVE                       │")
    print(f"  │    → Null space captures physics structure                   │")
    print(f"  │    → SEC cost law is a constraint, not a fit                 │")
    print(f"  │                                                              │")
    print(f"  │  If predictions fail:                                        │")
    print(f"  │    → Framework captures known patterns but doesn't extend    │")
    print(f"  │    → Stoichiometric matrix needs physics-derived constraints │")
    print(f"  │    → Current constraints are necessary but insufficient      │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    results['summary'] = {
        'total': total, 'passed': pass_count,
        'score': f"{pass_count}/{total}",
        'n_total_predictions': n_predictions,
        'n_sub_half_pct': n_sub_half,
    }
    save_results(results, 'exp_16_null_space_predictions')
    return results


if __name__ == '__main__':
    main()
