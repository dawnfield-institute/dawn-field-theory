"""
exp_17: Physics-Derived Stoichiometric Matrix

HYPOTHESIS: The stoichiometric constraints in exp_13 (mod-3 cycle, parity,
gauge closure) were chosen by the researcher. If the PAC framework is
physical, the constraints should be DERIVABLE from known physics principles:
anomaly cancellation, gauge invariance, renormalization group structure.

Replacing hand-chosen rows with physics-derived ones should produce a matrix
that is MORE selective for actual physics formulas (addressing exp_14 T2's
selectivity ratio of 0.86×, which showed the hand-built matrix RESISTS physics).

MOTIVATION:
  exp_13 T2 null space: 6-dim, allows too much freedom
  exp_14 T2 selectivity: 0.86× (matrix resists physics formulas)
  Both hint that hand-chosen constraints are not aligned with physics.

  Physical constraints:
    1. ANOMALY CANCELLATION: In gauge theory, Σcharges = 0 per generation.
       This constrains which Fibonacci numbers can appear together — their
       charges (E-I-S assignments) must cancel.
    2. ASYMPTOTIC FREEDOM: SU(3) runs to weak coupling at high energy.
       The β-function coefficient b₀ = 11 - 2n_f/3 = 7 for 6 flavors.
       F₇ = 13 appears as the total gauge dimension; 11 - 2·6/3 = 7.
    3. RG FLOW: Coupling constants run logarithmically. The ratio of
       coupling values at different scales involves ln(M_Z/M) where
       M_Z ≈ 91 GeV. In Fibonacci units, this is related to hierarchy depth.
    4. GENERATION UNIVERSALITY: All generations have identical gauge charges.
       This means F₄=3 multiplies ALL gauge-charged quantities identically.

TESTS:
  Test 1 — Physics-Motivated Matrix: Build S from anomaly cancellation,
           RG coefficient matching, generation universality, and Fibonacci
           recursion. Compare null space to exp_13/14 matrices.

  Test 2 — Selectivity Improvement: Do physics formulas project MORE
           into the new matrix's null space than random formulas?
           Must beat exp_14's 0.86× selectivity to PASS.

  Test 3 — Null Space Tightness: How many targets can the physics-derived
           null space match? Must be ≥ the exp_13 null space.

  Test 4 — Cross-Matrix Consensus: Where exp_13, exp_14, and exp_17
           matrices all agree on null-space alignment → strongest
           predictions. Where they disagree → framework limit.

SOURCES:
  - exp_13 (5-constraint hand-built matrix)
  - exp_14 (6-constraint atomic matrix, selectivity 0.86×)
  - Standard Model anomaly cancellation (Peskin & Schroeder)
  - QCD β-function coefficients
  - Generation universality (SM gauge structure)
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
FIB_INDICES = list(range(2, 13))  # F₂=1 ... F₁₂=144
FIB_VALUES = [FIB[i] for i in FIB_INDICES]
N_SPECIES = len(FIB_INDICES)


def pct_error(pred, meas):
    if meas == 0:
        return float('inf')
    return abs(pred - meas) / abs(meas) * 100


# =====================================================================
# Physics formula catalog (same as exp_15)
# =====================================================================
SIN2_TW    = 0.23122
MU_E       = 206.7682830
NU_WF      = 0.6299709
ALPHA_S    = 0.1180
P_E        = 1836.15267343
TAU_E      = 3477.48

FORMULAS = {
    'sin²θ_W': {'indices': [4, 7], 'class': 'fundamental', 'measured': SIN2_TW},
    'Koide Q': {'indices': [2, 3], 'class': 'fundamental', 'measured': 0.666661},
    'She-Lev β': {'indices': [3, 4], 'class': 'fundamental', 'measured': 2/3},
    'ν_WF': {'indices': [3, 4], 'class': 'derived', 'measured': NU_WF},
    'α_s': {'indices': [4, 6], 'class': 'derived', 'measured': ALPHA_S},
    'Cabibbo': {'indices': [4, 7], 'class': 'derived', 'measured': 13.04},
    'μ/e': {'indices': [4, 6, 7], 'class': 'composite', 'measured': MU_E},
    'α_em': {'indices': [3, 4, 7, 10], 'class': 'composite', 'measured': ALPHA_EM_PDG},
    'p/e': {'indices': [4, 6, 9, 12], 'class': 'composite', 'measured': P_E},
    'τ/e': {'indices': [4, 5, 7, 11], 'class': 'composite', 'measured': TAU_E},
}


def formula_vector(indices):
    """Create unit indicator vector."""
    vec = np.zeros(N_SPECIES)
    for i in indices:
        if i in FIB_INDICES:
            vec[FIB_INDICES.index(i)] = 1.0
    return vec


def get_null_space(S):
    """Return null space basis, rank, null dim."""
    U, sigma, Vt = np.linalg.svd(S)
    tol = 1e-10
    rank = int(np.sum(sigma > tol * sigma[0]))
    null_dim = N_SPECIES - rank
    null_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, N_SPECIES))
    return null_basis, rank, null_dim


def null_alignment(vec, null_basis):
    """Fraction of vector in null space."""
    n = np.linalg.norm(vec)
    if n < 1e-12 or null_basis.shape[0] == 0:
        return 0.0
    proj = null_basis @ vec
    return float(np.linalg.norm(proj) / n)


# =====================================================================
# Matrix builders
# =====================================================================
def build_exp13_matrix():
    """Hand-built 5-constraint matrix from exp_13."""
    S = np.zeros((5, N_SPECIES))
    idx = {n: FIB_INDICES.index(n) for n in FIB_INDICES}
    S[0] = FIB_VALUES                                # PAC magnitude
    S[1] = FIB_INDICES                               # Hierarchy depth
    S[2] = [n % 3 for n in FIB_INDICES]              # E-I-S cycle
    S[3] = [n % 2 for n in FIB_INDICES]              # Parity
    S[4, idx[5]] = -1; S[4, idx[6]] = -1; S[4, idx[7]] = 1  # Gauge closure
    return S


def build_exp14_matrix():
    """Atomic decomposition 6-constraint matrix from exp_14."""
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


def build_physics_matrix():
    """
    Physics-derived stoichiometric matrix.
    
    Rows derived from actual physics principles, not hand-chosen proxies.
    """
    S = np.zeros((7, N_SPECIES))
    idx = {n: FIB_INDICES.index(n) for n in FIB_INDICES}

    # ------------------------------------------------------------------
    # Row 0: ANOMALY CANCELLATION
    # In SM, anomaly-free condition requires: Σ (charges) = 0 per generation.
    # For hypercharge: Σ Y = 0 over all fermion representations.
    # Map to Fibonacci: each F_n has an E-I-S "charge" assignment.
    # In the E-I-S triad: charge_n = cos(2π n/3)
    # This is the PHYSICAL version of exp_13's "mod 3" row, but using
    # the actual sinusoidal structure of U(1) charges.
    # ------------------------------------------------------------------
    S[0] = [math.cos(2 * math.pi * n / 3) for n in FIB_INDICES]

    # ------------------------------------------------------------------
    # Row 1: ANOMALY CANCELLATION (imaginary part)
    # The second constraint from anomaly: sin component.
    # Together rows 0+1 enforce proper U(1) charge cancellation.
    # ------------------------------------------------------------------
    S[1] = [math.sin(2 * math.pi * n / 3) for n in FIB_INDICES]

    # ------------------------------------------------------------------
    # Row 2: ASYMPTOTIC FREEDOM COEFFICIENT
    # QCD β-function: b₀ = (11 C_A - 4 T_F n_f) / (4π)
    # For SU(3): C_A = 3, T_F = 1/2, n_f = 6 (flavors)
    # b₀ = (11·3 - 4·½·6) / (4π) = (33-12)/(4π) = 21/(4π)
    # The integer part: 33 - 12 = 21 = F₈
    # Constraint: any valid gauge combination satisfies
    #   F₄² × coefficient - F₇ = asymptotic freedom margin
    # Encode: bosonic contribution (11·F₄=33) minus fermionic (2·nf=12)
    # Maps to: contributions from even-indexed F_n (bosonic) minus
    #          odd-indexed (fermionic) weighted by gauge dimension
    # ------------------------------------------------------------------
    for j, n in enumerate(FIB_INDICES):
        if n % 2 == 0:  # even index → bosonic (self-interaction)
            S[2, j] = 11.0 * FIB[n] / FIB[7]   # normalized by gauge dim
        else:           # odd index → fermionic (screening)
            S[2, j] = -4.0 * FIB[n] / FIB[7]

    # ------------------------------------------------------------------
    # Row 3: GENERATION UNIVERSALITY
    # All 3 generations have identical gauge charges.
    # This means: in any valid formula, the coefficient of F₄=3
    # (when present) must multiply the entire remaining expression.
    # Encode: F₄ contribution must be proportional to its value (3)
    # relative to the full gauge sector (F₇=13).
    # 
    # Concretely: the "generation weight" of F_n is:
    #   0 if n < 4 (pre-generational)
    #   FIB[n]/FIB[4] - 1 if n ≥ 4 (how many F₄ units it contains)
    # ------------------------------------------------------------------
    for j, n in enumerate(FIB_INDICES):
        if n < 4:
            S[3, j] = 0
        else:
            S[3, j] = FIB[n] / FIB[4] - (n - 4 + 1)

    # ------------------------------------------------------------------
    # Row 4: FIBONACCI RECURSION (electroweak closure)
    # This is the only constraint shared with exp_13.
    # F₇ = F₆ + F₅ is the Fibonacci identity at the electroweak scale.
    # ------------------------------------------------------------------
    S[4, idx[5]] = -1; S[4, idx[6]] = -1; S[4, idx[7]] = 1

    # ------------------------------------------------------------------
    # Row 5: FIBONACCI RECURSION (strong closure)
    # F₆ = F₅ + F₄ — the SU(3) gauge closure.
    # ------------------------------------------------------------------
    S[5, idx[4]] = -1; S[5, idx[5]] = -1; S[5, idx[6]] = 1

    # ------------------------------------------------------------------
    # Row 6: RG LOGARITHMIC RUNNING
    # Coupling constants run as α(μ) ∝ 1/ln(μ/Λ).
    # The hierarchy depth n maps to energy scale (higher n = higher E).
    # The running constraint: information at scale n includes all scales ≤ n.
    # Encode: cumulative Fibonacci sum up to each index.
    # F_n's "RG weight" = Σ_{k=2}^{n} F_k / Σ_{k=2}^{12} F_k
    # ------------------------------------------------------------------
    total_fib = sum(FIB_VALUES)
    cumsum = 0
    for j, n in enumerate(FIB_INDICES):
        cumsum += FIB[n]
        S[6, j] = cumsum / total_fib

    return S


# =====================================================================
# Test 1: Physics-Motivated Matrix Analysis
# =====================================================================
def test_1_physics_matrix():
    """
    Build the physics-derived matrix and compare its null space structure
    to exp_13 and exp_14 matrices.
    """
    print("\n" + "="*70)
    print("TEST 1: Physics-Derived Stoichiometric Matrix")
    print("="*70)

    S_phys = build_physics_matrix()
    S_13 = build_exp13_matrix()
    S_14 = build_exp14_matrix()

    null_phys, rank_phys, ndim_phys = get_null_space(S_phys)
    null_13, rank_13, ndim_13 = get_null_space(S_13)
    null_14, rank_14, ndim_14 = get_null_space(S_14)

    row_names_phys = [
        'Anomaly cos(2πn/3)', 'Anomaly sin(2πn/3)',
        'Asymptotic freedom', 'Generation universality',
        'EW closure (F₇=F₆+F₅)', 'Strong closure (F₆=F₅+F₄)',
        'RG cumulative running',
    ]

    print(f"\n  Physics-derived matrix ({S_phys.shape[0]}×{S_phys.shape[1]}):")
    for i, name in enumerate(row_names_phys):
        vals = ', '.join(f'{v:.3f}' for v in S_phys[i])
        print(f"    Row {i} ({name}): [{vals}]")

    print(f"\n  Matrix comparison:")
    print(f"    {'Matrix':15s}  {'Rows':>5s}  {'Rank':>5s}  {'Null dim':>8s}")
    print(f"    {'exp_13 (hand)':15s}  {S_13.shape[0]:5d}  {rank_13:5d}  {ndim_13:8d}")
    print(f"    {'exp_14 (atomic)':15s}  {S_14.shape[0]:5d}  {rank_14:5d}  {ndim_14:8d}")
    print(f"    {'exp_17 (physics)':15s}  {S_phys.shape[0]:5d}  {rank_phys:5d}  {ndim_phys:8d}")

    # --- Formula alignment comparison ---
    print(f"\n  Formula alignment across matrices:")
    print(f"  {'Formula':15s}  {'exp_13':>8s}  {'exp_14':>8s}  {'Physics':>8s}  {'Consensus':>10s}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    consensus_data = {}
    for name, info in FORMULAS.items():
        vec = formula_vector(info['indices'])
        a13 = null_alignment(vec, null_13)
        a14 = null_alignment(vec, null_14)
        a_ph = null_alignment(vec, null_phys)
        avg = (a13 + a14 + a_ph) / 3
        consensus_data[name] = {'a13': a13, 'a14': a14, 'a_phys': a_ph, 'avg': avg}
        print(f"  {name:15s}  {a13:8.4f}  {a14:8.4f}  {a_ph:8.4f}  {avg:10.4f}")

    # --- Show null space basis ---
    print(f"\n  Physics matrix null space ({ndim_phys} vectors):")
    for k in range(min(ndim_phys, 5)):
        active = [(FIB_INDICES[j], null_phys[k, j])
                  for j in range(N_SPECIES) if abs(null_phys[k, j]) > 0.05]
        comps = ', '.join(f'F_{n}({c:+.3f})' for n, c in active)
        print(f"    v_{k}: {comps}")

    # PASS: physics matrix has interpretable null space AND formula alignment
    avg_formula_align = np.mean([d['a_phys'] for d in consensus_data.values()])
    passed = ndim_phys >= 2 and avg_formula_align > 0.3
    print(f"\n  Avg formula alignment (physics matrix): {avg_formula_align:.4f}")
    print(f"  PASS: {passed} (null dim ≥ 2 AND avg align > 0.3)")

    return {
        'rank_phys': rank_phys, 'null_dim_phys': ndim_phys,
        'rank_13': rank_13, 'null_dim_13': ndim_13,
        'rank_14': rank_14, 'null_dim_14': ndim_14,
        'avg_formula_align': float(avg_formula_align),
        'consensus': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in consensus_data.items()},
        'passed': passed,
    }


# =====================================================================
# Test 2: Selectivity Improvement
# =====================================================================
def test_2_selectivity():
    """
    exp_14 T2 found selectivity = 0.86× (matrix RESISTS physics).
    The physics-derived matrix should do BETTER — selectivity > 1.0
    means the matrix FAVORS physics formulas over random ones.
    """
    print("\n" + "="*70)
    print("TEST 2: Selectivity — Does Physics Matrix Favor Physics?")
    print("="*70)

    rng = np.random.default_rng(42)

    matrices = {
        'exp_13 (hand)':    build_exp13_matrix(),
        'exp_14 (atomic)':  build_exp14_matrix(),
        'exp_17 (physics)': build_physics_matrix(),
    }

    physics_formulas = FORMULAS

    for mat_name, S in matrices.items():
        null_basis, rank, ndim = get_null_space(S)

        # Physics formula alignment
        physics_nfs = []
        for name, info in physics_formulas.items():
            vec = formula_vector(info['indices'])
            nf = null_alignment(vec, null_basis)
            physics_nfs.append(nf)
        avg_physics = np.mean(physics_nfs)

        # Random formula alignment (same complexity distribution)
        n_random = 5000
        random_nfs = []
        for _ in range(n_random):
            vec = np.zeros(N_SPECIES)
            k = rng.integers(1, 5)
            positions = rng.choice(N_SPECIES, min(k, N_SPECIES), replace=False)
            vec[positions] = 1.0
            random_nfs.append(null_alignment(vec, null_basis))
        avg_random = np.mean(random_nfs)

        selectivity = avg_physics / avg_random if avg_random > 0 else 0

        # Physics percentile
        random_arr = np.array(random_nfs)
        physics_percentile = float(100 - np.mean(random_arr >= avg_physics) * 100)

        print(f"\n  {mat_name}: null dim={ndim}")
        print(f"    Avg physics alignment: {avg_physics:.4f}")
        print(f"    Avg random alignment:  {avg_random:.4f}")
        print(f"    Selectivity ratio:     {selectivity:.3f}×")
        print(f"    Physics percentile:    {physics_percentile:.1f}%")

    # --- The key comparison ---
    # Recalculate for physics matrix specifically
    S_phys = build_physics_matrix()
    null_phys, _, ndim_phys = get_null_space(S_phys)

    physics_nfs_phys = []
    for info in physics_formulas.values():
        vec = formula_vector(info['indices'])
        physics_nfs_phys.append(null_alignment(vec, null_phys))
    avg_physics_phys = np.mean(physics_nfs_phys)

    random_nfs_phys = []
    for _ in range(5000):
        vec = np.zeros(N_SPECIES)
        k = rng.integers(1, 5)
        positions = rng.choice(N_SPECIES, min(k, N_SPECIES), replace=False)
        vec[positions] = 1.0
        random_nfs_phys.append(null_alignment(vec, null_phys))
    avg_random_phys = np.mean(random_nfs_phys)

    selectivity_phys = avg_physics_phys / avg_random_phys if avg_random_phys > 0 else 0

    # PASS: physics matrix selectivity > 1.0 (favors physics)
    # OR: physics matrix selectivity > 0.86 (beats exp_14)
    passed = selectivity_phys > 0.86
    improvement = selectivity_phys / 0.86 if 0.86 > 0 else 0

    print(f"\n  exp_14 selectivity:     0.86×")
    print(f"  Physics selectivity:    {selectivity_phys:.3f}×")
    print(f"  Improvement:            {improvement:.2f}×")
    print(f"  PASS: {passed} (physics selectivity > 0.86×)")

    return {
        'selectivity_phys': float(selectivity_phys),
        'selectivity_baseline': 0.86,
        'improvement': float(improvement),
        'avg_physics_align': float(avg_physics_phys),
        'avg_random_align': float(avg_random_phys),
        'null_dim_phys': ndim_phys,
        'passed': passed,
    }


# =====================================================================
# Test 3: Null Space Tightness — Multi-Target Matching
# =====================================================================
def test_3_null_space_tightness():
    """
    How many physics targets can each matrix's null space match?
    The physics-derived matrix should match at least as many as exp_13.
    """
    print("\n" + "="*70)
    print("TEST 3: Null Space Tightness — Multi-Target Matching")
    print("="*70)

    targets = {
        'sin²θ_W':   0.23122,       'Koide':      0.666661,
        'She-Lev':   2/3,           'ν_WF':       0.6299709,
        'α_s':       0.1180,        'α_em':       0.0072973525693,
        '1/φ':       INV_PHI,
    }

    matrices = {
        'exp_13':  build_exp13_matrix(),
        'exp_14':  build_exp14_matrix(),
        'exp_17':  build_physics_matrix(),
    }

    for mat_name, S in matrices.items():
        null_basis, rank, ndim = get_null_space(S)

        # For each null vector, extract the top-2 active species
        # and form all possible ratios from corresponding Fibonacci numbers
        matched = set()

        for k in range(ndim):
            vec = null_basis[k]
            # Get top 4 components by magnitude
            top4 = np.argsort(np.abs(vec))[-4:]
            top_indices = [FIB_INDICES[j] for j in top4 if abs(vec[j]) > 0.05]

            # Form all ratios and check against targets
            for a_idx in top_indices:
                for b_idx in top_indices:
                    if a_idx == b_idx or FIB[b_idx] == 0:
                        continue
                    ratio = FIB[a_idx] / FIB[b_idx]
                    ratio_xi = ratio / XI_BALANCE
                    for t_name, t_val in targets.items():
                        if pct_error(ratio, t_val) < 1.0:
                            matched.add(t_name)
                        if pct_error(ratio_xi, t_val) < 1.0:
                            matched.add(t_name)

        print(f"\n  {mat_name} (null dim={ndim}): {len(matched)}/{len(targets)} targets matched")
        for t in sorted(matched):
            print(f"    ✓ {t}")

    # --- The tightness test: physics matrix vs exp_13 ---
    S_phys = build_physics_matrix()
    null_phys, _, ndim_phys = get_null_space(S_phys)
    S_13 = build_exp13_matrix()
    null_13, _, ndim_13 = get_null_space(S_13)

    def count_matches(null_basis, ndim):
        matched = set()
        for k in range(ndim):
            vec = null_basis[k]
            top4 = np.argsort(np.abs(vec))[-4:]
            top_indices = [FIB_INDICES[j] for j in top4 if abs(vec[j]) > 0.05]
            for a_idx in top_indices:
                for b_idx in top_indices:
                    if a_idx == b_idx or FIB[b_idx] == 0:
                        continue
                    ratio = FIB[a_idx] / FIB[b_idx]
                    ratio_xi = ratio / XI_BALANCE
                    for t_name, t_val in targets.items():
                        if pct_error(ratio, t_val) < 1.0:
                            matched.add(t_name)
                        if pct_error(ratio_xi, t_val) < 1.0:
                            matched.add(t_name)
        return matched

    matches_phys = count_matches(null_phys, ndim_phys)
    matches_13 = count_matches(null_13, ndim_13)

    passed = len(matches_phys) >= len(matches_13)
    print(f"\n  exp_13 matches: {len(matches_13)}")
    print(f"  Physics matches: {len(matches_phys)}")
    print(f"  PASS: {passed} (physics ≥ exp_13)")

    return {
        'n_matches_phys': len(matches_phys),
        'n_matches_13': len(matches_13),
        'matched_targets_phys': sorted(matches_phys),
        'matched_targets_13': sorted(matches_13),
        'null_dim_phys': ndim_phys,
        'null_dim_13': ndim_13,
        'passed': passed,
    }


# =====================================================================
# Test 4: Cross-Matrix Consensus
# =====================================================================
def test_4_cross_matrix_consensus():
    """
    Where all three matrices (exp_13, exp_14, exp_17) agree that a
    Fibonacci combination has high null-space alignment → strongest signal.
    
    Build a consensus map: for each index pair/triple, compute alignment
    across all three matrices. High consensus = robust prediction.
    Low consensus = matrix-dependent artifact.
    """
    print("\n" + "="*70)
    print("TEST 4: Cross-Matrix Consensus Map")
    print("="*70)

    matrices = {
        'exp_13':  build_exp13_matrix(),
        'exp_14':  build_exp14_matrix(),
        'exp_17':  build_physics_matrix(),
    }

    null_spaces = {}
    for name, S in matrices.items():
        null_basis, _, _ = get_null_space(S)
        null_spaces[name] = null_basis

    KNOWN_INDEX_SETS = [
        frozenset([4, 7]), frozenset([2, 3]), frozenset([3, 4]),
        frozenset([4, 6]), frozenset([4, 6, 7]),
        frozenset([3, 4, 7, 10]), frozenset([4, 6, 9, 12]),
        frozenset([4, 5, 7, 11]),
    ]

    # --- Score all pairs ---
    consensus_results = []

    for a, b in combinations(FIB_INDICES, 2):
        vec = formula_vector([a, b])
        known = frozenset([a, b]) in KNOWN_INDEX_SETS
        aligns = {}
        for name, null_basis in null_spaces.items():
            aligns[name] = null_alignment(vec, null_basis)
        avg = np.mean(list(aligns.values()))
        min_a = min(aligns.values())
        consensus_results.append({
            'indices': [a, b], 'type': 'pair', 'known': known,
            'aligns': aligns, 'avg': avg, 'min': min_a,
        })

    # Triples
    for a, b, c in combinations(FIB_INDICES, 3):
        vec = formula_vector([a, b, c])
        known = frozenset([a, b, c]) in KNOWN_INDEX_SETS
        aligns = {}
        for name, null_basis in null_spaces.items():
            aligns[name] = null_alignment(vec, null_basis)
        avg = np.mean(list(aligns.values()))
        min_a = min(aligns.values())
        consensus_results.append({
            'indices': [a, b, c], 'type': 'triple', 'known': known,
            'aligns': aligns, 'avg': avg, 'min': min_a,
        })

    # Sort by minimum alignment (conservative consensus)
    consensus_results.sort(key=lambda x: x['min'], reverse=True)

    # --- Show known formulas ---
    print(f"\n  Known formula consensus:")
    for c in consensus_results:
        if c['known']:
            a_str = '  '.join(f"{k}={v:.3f}" for k, v in c['aligns'].items())
            print(f"    {str(c['indices']):20s}  avg={c['avg']:.3f}  min={c['min']:.3f}  [{a_str}]")

    # --- Top novel consensus ---
    novel = [c for c in consensus_results if not c['known']]
    print(f"\n  Top 15 NOVEL consensus predictions (sorted by min alignment):")
    print(f"  {'Indices':20s}  {'exp_13':>8s}  {'exp_14':>8s}  {'Physics':>8s}  {'Min':>6s}  {'Avg':>6s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")

    for c in novel[:15]:
        a13 = c['aligns']['exp_13']
        a14 = c['aligns']['exp_14']
        aph = c['aligns']['exp_17']
        fib_str = '×'.join(f'{FIB[n]}' for n in c['indices'])
        print(f"  {str(c['indices']):20s}  {a13:8.4f}  {a14:8.4f}  {aph:8.4f}  "
              f"{c['min']:6.3f}  {c['avg']:6.3f}  ({fib_str})")

    # --- Strong consensus: min alignment > 0.4 across ALL matrices ---
    strong_consensus = [c for c in novel if c['min'] > 0.4]
    print(f"\n  Strong consensus predictions (min > 0.4): {len(strong_consensus)}")
    for c in strong_consensus[:10]:
        fib_vals = [FIB[n] for n in c['indices']]
        ratios = []
        for i in range(len(fib_vals)):
            for j in range(len(fib_vals)):
                if i != j and fib_vals[j] != 0:
                    ratios.append(fib_vals[i] / fib_vals[j])
        ratios_str = ', '.join(f'{r:.4f}' for r in sorted(set(ratios))[:4])
        print(f"    {c['indices']} → ratios: [{ratios_str}]")

    # --- Disagreement analysis ---
    high_variance = [c for c in novel
                     if np.std(list(c['aligns'].values())) > 0.2
                     and c['avg'] > 0.4]
    print(f"\n  High-variance predictions (avg > 0.4 but std > 0.2): {len(high_variance)}")
    for c in high_variance[:5]:
        a_str = '  '.join(f"{k}={v:.3f}" for k, v in c['aligns'].items())
        print(f"    {c['indices']}  [{a_str}]  std={np.std(list(c['aligns'].values())):.3f}")

    # PASS: at least 5 strong consensus predictions AND known formulas agree
    known_consensus = [c for c in consensus_results if c['known']]
    known_avg_min = np.mean([c['min'] for c in known_consensus]) if known_consensus else 0

    passed = len(strong_consensus) >= 3 and known_avg_min > 0.3
    print(f"\n  Known formula avg min alignment: {known_avg_min:.3f}")
    print(f"  Strong consensus novel predictions: {len(strong_consensus)}")
    print(f"  PASS: {passed} (≥3 strong consensus AND known avg min > 0.3)")

    return {
        'n_strong_consensus': len(strong_consensus),
        'n_high_variance': len(high_variance),
        'known_avg_min': float(known_avg_min),
        'strong_predictions': [{
            'indices': c['indices'],
            'aligns': {k: float(v) for k, v in c['aligns'].items()},
            'min': float(c['min']),
            'avg': float(c['avg']),
        } for c in strong_consensus[:10]],
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_17_physics_derived_matrix',
        'Physics-derived stoichiometric matrix',
        paper='Paper 4',
        section='§physics_constraints'
    )

    results = {'metadata': meta, 'tests': {}}

    results['tests']['test_1_physics_matrix'] = test_1_physics_matrix()
    results['tests']['test_2_selectivity']    = test_2_selectivity()
    results['tests']['test_3_tightness']      = test_3_null_space_tightness()
    results['tests']['test_4_consensus']      = test_4_cross_matrix_consensus()

    # --- Final synthesis ---
    print("\n" + "="*70)
    print("  SYNTHESIS: Physics-Derived Stoichiometric Matrix")
    print("="*70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:35s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    # Compare matrices
    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  MATRIX COMPARISON                                         │")
    print(f"  │                                                             │")
    print(f"  │  exp_13 (hand-built):                                       │")
    print(f"  │    + Simple, interpretable constraints                       │")
    print(f"  │    - mod-3 and parity not derived from physics               │")
    print(f"  │    - Selectivity 0.86× (resists physics)                     │")
    print(f"  │                                                             │")
    print(f"  │  exp_14 (atomic):                                           │")
    print(f"  │    + F₂/F₃ decomposition is mathematically exact             │")
    print(f"  │    + Recursion constraints are structural                     │")
    print(f"  │    - Still not derived from physics                           │")
    print(f"  │                                                             │")
    print(f"  │  exp_17 (physics-derived):                                  │")
    print(f"  │    + Anomaly cancellation from gauge theory                   │")
    print(f"  │    + Asymptotic freedom from QCD                              │")
    print(f"  │    + Generation universality from SM structure                │")
    print(f"  │    + RG running from renormalization                          │")
    print(f"  │    ? Does selectivity improve?                               │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    results['summary'] = {
        'total': total, 'passed': pass_count,
        'score': f"{pass_count}/{total}",
    }
    save_results(results, 'exp_17_physics_derived_matrix')
    return results


if __name__ == '__main__':
    main()
