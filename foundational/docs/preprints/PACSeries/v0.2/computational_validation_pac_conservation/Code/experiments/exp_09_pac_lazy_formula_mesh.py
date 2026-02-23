"""
exp_21: PAC-Lazy Formula Mesh — Conservation + Gating + Profile Comparison

FROM THE GAIA POCs (poc_011, poc_016-018):
  PAC Lazy architecture gives us three principles exp_20 was missing:

  1. PAC CONSERVATION: f(parent) = Σf(children)
     Each formula distributes exactly 1.0 potential through its recursion tree.
     At each decomposition F_n → F_{n-1} + F_{n-2}, potential SPLITS φ-weighted:
       child_larger gets  φ/(φ+1) ≈ 0.618 of parent potential
       child_smaller gets 1/(φ+1) ≈ 0.382 of parent potential
     Deep nodes get geometrically less per decomposition step.
     But convergence from MANY formulas accumulates at shared nodes.
     The balance between φ-decay and multi-formula accumulation IS the signal.

  2. SEC GATING: C(S) = S * exp(-ξ * S)
     Only expand deeper when local potential diversity is high (many formulas
     "disagreeing" at a node). If all formulas agree → crystallize, stop expanding.
     High-pressure hubs with uniform agreement FREEZE. Contested hubs expand deeper.
     This makes depth ADAPTIVE — matching poc_018's lazy layers.

  3. PROFILE COMPARISON (not scalar pressure):
     Score novel formulas by comparing their PAC potential DISTRIBUTION against
     the known formula mesh. Cosine similarity between potential vectors.
     Not "how much total pressure" (which correlates with depth),
     but "does the SHAPE match" (which captures structure).

WHY exp_20 FAILED:
  Raw visit counting has no conservation. F_10 decomposes to ~55 visits total;
  F_3 decomposes to 2 visits. The pressure is just depth. No signal.
  PAC conservation means F_10's tree distributes 1.0 potential across its ~55 nodes,
  so each node gets geometrically less. But multiple formulas hitting the same
  node ACCUMULATE. The ratio of "how many formulas" to "how much each contributes"
  creates a non-trivial distribution. THAT has signal.

GAIA POC CONNECTIONS:
  - poc_011 PACLazyTransformer: nodes carry deltas, lazy depth via SEC
  - poc_016 extractor_v3_pac_lazy: φ-weighted splitting of potential
  - poc_018 PACTree: f(parent)=Σf(children), conservation propagation
  - poc_018 SECField: C(S)=S*exp(-ξ*S), crystallization threshold
  - poc_018 ComplexityLevel: adaptive depth per complexity

TESTS:
  Test 1 — PAC Distribution: Visualize how potential distributes through the
           recursion tree. Verify conservation. Show φ-decay vs accumulation.

  Test 2 — Profile-Based Discrimination: Novel formulas scored by cosine
           similarity of PAC profile to known mesh. Physics-matching should
           have higher similarity. THE CORE TEST.

  Test 3 — SEC-Gated Adaptive Depth: Only expand when potential diversity
           is high. Show that gating changes depth per formula. Gated depth
           as additional discriminant.

  Test 4 — PAC Profile vs Raw Pressure: Direct comparison show that PAC
           profile (conserved) discriminates better than raw pressure (exp_20).
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from dataclasses import dataclass, field as datafield
from typing import Dict, List, Tuple, Optional, FrozenSet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results


# =====================================================================
# Constants
# =====================================================================
FIB_INDICES = list(range(2, 13))   # indices 2..12
N_SPECIES = len(FIB_INDICES)        # 11
FIB_VALUES = [FIB[i] for i in FIB_INDICES]

# PAC conservation split ratio (from φ)
PHI_SHARE = PHI / (PHI + 1)       # ≈ 0.618  (larger child)
INV_PHI_SHARE = 1 / (PHI + 1)     # ≈ 0.382  (smaller child)

# SEC gating constants (from poc_018)
XI_CRITICAL = 1.0571
SEC_CRYSTALLIZATION_BASE = 0.10
# Depth-dependent crystallization (from poc_018 ComplexityLevel):
#   Shallow levels: low threshold → hard to crystallize → expand freely
#   Deep levels: high threshold → easy to crystallize → stop
# threshold(level) = base + (level / max_level)^gamma * (ceiling - base)
# gamma < 1 makes the ramp aggressive (hits ceiling earlier)
SEC_CRYSTALLIZATION_CEILING = 0.38   # at max depth, crystallize easily
SEC_RAMP_GAMMA = 0.5                 # sqrt ramp: aggressive at mid-depths
SEC_MIN_CONTRIBUTORS = 2             # need >= N formulas to even check

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

TRIVIAL_PROXIMITY = 0.02
TRIVIAL_VALUES = [1.0, PHI, INV_PHI, 2.0, 0.5, 3.0, 1.5]

NOVEL_TARGETS = {
    'V_us':           0.2243,   'V_cb':           0.0422,
    'V_ub':           0.00394,  'V_td':           0.00814,
    'V_ts':           0.0400,   'Jarlskog_J':     3.08e-5,
    'm_u/m_d':        0.474,    'm_s/m_d':        20.2,
    'm_c/m_s':        11.7,     'm_b/m_c':        3.41,
    'm_t/m_b':        41.3,     'm_mu/m_e':       206.768283,
    'm_tau/m_mu':     16.8170,  'sin2_theta12':   0.307,
    'sin2_theta23':   0.546,    'sin2_theta13':   0.0220,
    'Dm2_ratio':      0.0297,   'Lambda_QCD_mp':  0.217,
    'alpha_em_MZ':    1/127.951,'sin2_theta_eff': 0.23155,
    'alpha_s_Mtau':   0.332,    'Omega_b':        0.0493,
    'Omega_c':        0.265,    'Omega_Lambda':   0.685,
    'n_s':            0.965,    'sigma_8':        0.811,
    'zeta_3':         1.2020569,'von_Karman':     0.41,
}
NOVEL_TARGETS = {k: v for k, v in NOVEL_TARGETS.items()
                 if not any(abs(v - tv) / max(abs(tv), 1e-12) < TRIVIAL_PROXIMITY
                            for tv in TRIVIAL_VALUES)}


# =====================================================================
# PAC Formula Node — each node in the decomposition tree
# =====================================================================
@dataclass
class PACNode:
    """A node in the PAC decomposition tree."""
    index: int
    level: int
    potential: float
    children: List['PACNode'] = datafield(default_factory=list)
    source_formula: str = ''

    @property
    def is_base(self):
        return len(self.children) == 0


# =====================================================================
# PAC-conserved decomposition
# =====================================================================
def pac_decompose(n: int, potential: float, level: int = 0,
                  max_level: int = 20, min_potential: float = 1e-10) -> PACNode:
    """
    Decompose index n with PAC conservation.

    At each step F_n → F_{n-1} + F_{n-2}:
      child_{n-1} gets PHI_SHARE   (≈ 0.618) of parent potential
      child_{n-2} gets INV_PHI_SHARE (≈ 0.382) of parent potential

    Conservation: parent.potential = sum(child.potential) exactly.
    """
    node = PACNode(index=n, level=level, potential=potential)

    # Base cases: n <= 2, or potential too small, or depth exceeded
    if n <= 2 or level >= max_level or potential < min_potential:
        return node

    # Fibonacci decomposition: F_n = F_{n-1} + F_{n-2}
    # Larger operand (n-1) gets φ-share, smaller (n-2) gets 1/φ-share
    child_large = pac_decompose(n - 1, potential * PHI_SHARE,
                                level + 1, max_level, min_potential)
    child_small = pac_decompose(n - 2, potential * INV_PHI_SHARE,
                                level + 1, max_level, min_potential)
    node.children = [child_large, child_small]

    return node


def collect_potential(root: PACNode, leaves_only: bool = False) -> Dict[int, float]:
    """
    Collect potential at each index across the tree.

    leaves_only=False (default): flow-through profile — every node the
        potential passes through gets counted. This gives the convergence
        SHAPE used for cosine/KL comparison. Total exceeds 1.0 because
        internal nodes carry potential that also flows to children.

    leaves_only=True: strict PAC conservation — only leaf (terminal) nodes
        counted. Total equals root potential exactly. Use for conservation
        verification.
    """
    result = defaultdict(float)

    def _walk(node):
        if leaves_only:
            if node.is_base:
                result[node.index] += node.potential
            else:
                for child in node.children:
                    _walk(child)
        else:
            result[node.index] += node.potential
            for child in node.children:
                _walk(child)

    _walk(root)
    return dict(result)


def verify_conservation(root: PACNode) -> float:
    """Verify that leaf potential sums to root potential (PAC conservation)."""
    leaves = collect_potential(root, leaves_only=True)
    return sum(leaves.values())


def formula_pac_profile(indices: List[int], recurrence='fibonacci') -> Dict[int, float]:
    """
    Build PAC-conserved potential profile for a formula's index set.
    Each index in the formula gets potential = 1.0/len(indices).
    Total budget = 1.0 (conserved).
    """
    per_index_budget = 1.0 / len(indices)
    combined = defaultdict(float)

    for idx in indices:
        tree = pac_decompose(idx, per_index_budget)
        potentials = collect_potential(tree)
        for k, v in potentials.items():
            combined[k] += v

    return dict(combined)


# =====================================================================
# Convergence Mesh with PAC conservation
# =====================================================================
def build_pac_mesh(known_formulas: Dict[str, List[int]]) -> Dict[int, float]:
    """
    Build convergence mesh: overlay PAC profiles of all known formulas.
    Each formula contributes 1.0 total potential (conserved).
    """
    mesh = defaultdict(float)
    n_formulas = len(known_formulas)

    for name, indices in known_formulas.items():
        profile = formula_pac_profile(indices)
        for idx, pot in profile.items():
            mesh[idx] += pot

    return dict(mesh)


def profile_to_vector(profile: Dict[int, float], all_indices: List[int]) -> np.ndarray:
    """Convert a potential profile to a fixed-length vector over all_indices."""
    return np.array([profile.get(idx, 0.0) for idx in all_indices])


def profile_cosine_similarity(prof_a: Dict[int, float],
                               prof_b: Dict[int, float]) -> float:
    """Cosine similarity between two PAC profiles."""
    all_keys = sorted(set(prof_a.keys()) | set(prof_b.keys()))
    if not all_keys:
        return 0.0
    va = np.array([prof_a.get(k, 0.0) for k in all_keys])
    vb = np.array([prof_b.get(k, 0.0) for k in all_keys])
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def profile_kl_divergence(prof_novel: Dict[int, float],
                           prof_mesh: Dict[int, float]) -> float:
    """KL divergence from novel profile to mesh profile (lower = more similar)."""
    all_keys = sorted(set(prof_novel.keys()) | set(prof_mesh.keys()))
    if not all_keys:
        return float('inf')

    # Normalize to probability distributions
    eps = 1e-12
    va = np.array([prof_novel.get(k, 0.0) for k in all_keys]) + eps
    vb = np.array([prof_mesh.get(k, 0.0) for k in all_keys]) + eps
    va = va / va.sum()
    vb = vb / vb.sum()

    return float(np.sum(va * np.log(va / vb)))


# =====================================================================
# SEC Gating — adaptive depth
# =====================================================================
def sec_collapse_operator(entropy: float, xi: float = XI_CRITICAL) -> float:
    """SEC collapse: C(S) = S * exp(-xi * S). From poc_018."""
    return entropy * math.exp(-xi * entropy)


def sec_crystallization_threshold(level: int, max_level: int = 20) -> float:
    """
    Depth-dependent SEC crystallization threshold.

    Mirrors poc_018 ComplexityLevel with sqrt ramp (gamma=0.5):
      Level 0: threshold 0.10  → hard to crystallize, expand freely
      Level 5: threshold 0.24  → moderate gating
      Level 10: threshold 0.30 → significant gating
      Level 20: threshold 0.38 → easy to crystallize, stop early

    The sqrt ramp means mid-depth levels already face substantial gating,
    matching poc_018's observation that most complexity lives at levels 2-4.
    """
    t = min(level / max(max_level, 1), 1.0)
    ramped = t ** SEC_RAMP_GAMMA  # sqrt ramp: aggressive at mid-depths
    return SEC_CRYSTALLIZATION_BASE + ramped * (SEC_CRYSTALLIZATION_CEILING - SEC_CRYSTALLIZATION_BASE)


def pac_decompose_sec_gated(n: int, potential: float, formula_potentials: Dict[int, List[float]],
                             level: int = 0, max_level: int = 20,
                             min_potential: float = 1e-10) -> PACNode:
    """
    PAC decomposition with SEC gating.

    Depth-dependent crystallization (from poc_018 ComplexityLevel):
    - Shallow levels: low threshold → expand freely (like token/phrase level)
    - Deep levels: high threshold → crystallize unless high diversity
    - Fewer than SEC_MIN_CONTRIBUTORS formulas → crystallize (insufficient pressure)

    formula_potentials: {index: [pot_from_formula_1, pot_from_formula_2, ...]}
    """
    node = PACNode(index=n, level=level, potential=potential)

    if n <= 2 or level >= max_level or potential < min_potential:
        return node

    # SEC check: is this node "contested" enough to expand?
    incoming = formula_potentials.get(n, [])
    threshold = sec_crystallization_threshold(level, max_level)

    if len(incoming) < SEC_MIN_CONTRIBUTORS:
        # Too few formulas care about this node → crystallize
        return node

    if len(incoming) >= SEC_MIN_CONTRIBUTORS:
        arr = np.array(incoming)
        arr_norm = arr / (arr.sum() + 1e-12)
        entropy = -np.sum(arr_norm * np.log(arr_norm + 1e-12))
        max_entropy = np.log(len(incoming))
        normalized_entropy = entropy / (max_entropy + 1e-12) if max_entropy > 0 else 0

        collapsed = sec_collapse_operator(normalized_entropy)
        if collapsed < threshold:
            # Crystallized — formulas agree at this depth. Stop expanding.
            return node

    # Expand with PAC conservation
    child_large = pac_decompose_sec_gated(
        n - 1, potential * PHI_SHARE, formula_potentials,
        level + 1, max_level, min_potential)
    child_small = pac_decompose_sec_gated(
        n - 2, potential * INV_PHI_SHARE, formula_potentials,
        level + 1, max_level, min_potential)
    node.children = [child_large, child_small]

    return node


def tree_depth(root: PACNode) -> int:
    """Get maximum depth of a PAC tree."""
    if not root.children:
        return 0
    return 1 + max(tree_depth(c) for c in root.children)


# =====================================================================
# Index-set scanning (shared with exp_19/20)
# =====================================================================
def compute_index_set_matches(values, targets, threshold_pct=1.0):
    """For each unique index set, determine if any template matches a target."""
    n = len(values)
    results = {}

    for ai in range(n):
        for bi in range(n):
            if ai == bi or values[bi] == 0:
                continue
            s = frozenset([FIB_INDICES[ai], FIB_INDICES[bi]])
            if s not in results:
                results[s] = {'matched': False, 'best_error': float('inf'),
                              'best_target': None, 'size': 2}
            va, vb = values[ai], values[bi]
            for val in [va/vb, va/(vb*XI_BALANCE), va/(vb*PHI), va*XI_BALANCE/vb]:
                if not math.isfinite(val) or val <= 0:
                    continue
                for tn, tv in targets.items():
                    if tv == 0: continue
                    err = abs(val - tv) / abs(tv) * 100
                    if err < results[s]['best_error']:
                        results[s]['best_error'] = float(err)
                        results[s]['best_target'] = tn
                    if err < threshold_pct:
                        results[s]['matched'] = True

    for ai in range(n):
        for bi in range(n):
            for ci in range(n):
                if len({ai, bi, ci}) < 3 or values[ci] == 0:
                    continue
                s = frozenset([FIB_INDICES[ai], FIB_INDICES[bi], FIB_INDICES[ci]])
                if s not in results:
                    results[s] = {'matched': False, 'best_error': float('inf'),
                                  'best_target': None, 'size': 3}
                va, vb, vc = values[ai], values[bi], values[ci]
                templates = [va*vb/vc, va*vb/(vc*XI_BALANCE)]
                if vb*vc > 0:
                    templates += [va**2/(vb*vc), va/(vb*vc)]
                for val in templates:
                    if not math.isfinite(val) or val <= 0:
                        continue
                    for tn, tv in targets.items():
                        if tv == 0: continue
                        err = abs(val - tv) / abs(tv) * 100
                        if err < results[s]['best_error']:
                            results[s]['best_error'] = float(err)
                            results[s]['best_target'] = tn
                        if err < threshold_pct:
                            results[s]['matched'] = True

    return results


# =====================================================================
# Test 1: PAC Distribution Visualization
# =====================================================================
def test_1_pac_distribution():
    """
    Build PAC-conserved decomposition. Visualize potential distribution.
    Verify conservation holds. Show φ-decay vs convergence accumulation.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PAC-Conserved Potential Distribution")
    print("=" * 70)

    # Show individual formula distributions
    print(f"\n  Known formula PAC profiles (potential per formula = 1.0):")
    print(f"  {'Formula':15s}  {'Indices':12s}  {'Tree depth':>10s}  "
          f"{'Leaf pot':>10s}  {'Conserved?':>10s}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    all_profiles = {}
    for name, indices in KNOWN_FORMULAS.items():
        profile = formula_pac_profile(indices)
        all_profiles[name] = profile

        # Verify PAC conservation: leaf potential sums to input budget
        per_idx = 1.0 / len(indices)
        leaf_total = 0.0
        flow_total = sum(profile.values())
        depths = []
        for idx in indices:
            tree = pac_decompose(idx, per_idx)
            leaf_total += verify_conservation(tree)
            depths.append(tree_depth(tree))

        max_depth = max(depths)
        conservation_ok = abs(leaf_total - 1.0) < 1e-6

        print(f"  {name:15s}  {str(indices):12s}  {max_depth:10d}  "
              f"{leaf_total:10.6f}  {'YES' if conservation_ok else 'NO':>10s}  "
              f"(flow={flow_total:.3f})")

    # Build mesh
    mesh = build_pac_mesh(KNOWN_FORMULAS)
    all_idx = sorted(mesh.keys())

    print(f"\n  PAC Convergence Mesh (accumulated from {len(KNOWN_FORMULAS)} formulas):")
    print(f"  {'Index':>5s}  {'F_n':>5s}  {'PAC Pot':>10s}  {'%Total':>7s}  {'Bar':30s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*30}")

    total_mesh = sum(mesh.values())
    max_pot = max(mesh.values())

    for idx in range(1, 14):
        pot = mesh.get(idx, 0.0)
        pct = pot / total_mesh * 100 if total_mesh > 0 else 0
        bar_len = int(pot / max_pot * 30) if max_pot > 0 else 0
        bar = '#' * bar_len
        fval = str(FIB[idx]) if idx < len(FIB) else '?'
        hub_mark = ' << HUB' if pot > max_pot * 0.5 else ''
        print(f"  {idx:5d}  {fval:>5s}  {pot:10.4f}  {pct:6.1f}%  {bar}{hub_mark}")

    # Compare with exp_20's raw mesh (visit counting)
    print(f"\n  Comparison: PAC-conserved vs Raw (visit counting):")
    print(f"  {'Index':>5s}  {'PAC pot':>10s}  {'Raw visits':>10s}  {'PAC/Raw':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

    # Quick raw count for comparison
    raw_mesh = Counter()
    for name, indices in KNOWN_FORMULAS.items():
        for idx in indices:
            raw_tree = _raw_decompose(idx)
            raw_mesh += raw_tree

    for idx in range(1, 13):
        pac_p = mesh.get(idx, 0.0)
        raw_p = raw_mesh.get(idx, 0)
        ratio = pac_p / raw_p if raw_p > 0 else 0
        print(f"  {idx:5d}  {pac_p:10.4f}  {raw_p:10d}  {ratio:10.6f}")

    # Key metric: how much does PAC suppress depth bias?
    # Coefficient of variation (lower = less biased)
    pac_vals = [mesh.get(i, 0) for i in range(1, 13)]
    raw_vals = [raw_mesh.get(i, 0) for i in range(1, 13)]
    pac_cv = np.std(pac_vals) / np.mean(pac_vals) if np.mean(pac_vals) > 0 else 0
    raw_cv = np.std(raw_vals) / np.mean(raw_vals) if np.mean(raw_vals) > 0 else 0

    print(f"\n  Coefficient of variation (lower = more uniform):")
    print(f"  PAC mesh CV:  {pac_cv:.3f}")
    print(f"  Raw mesh CV:  {raw_cv:.3f}")
    print(f"  PAC reduces depth bias by: {(1 - pac_cv/raw_cv)*100:.1f}%" if raw_cv > 0 else "")

    # Verify leaf-level conservation across all formulas
    leaf_mesh = defaultdict(float)
    for name, indices in KNOWN_FORMULAS.items():
        per_idx = 1.0 / len(indices)
        for idx in indices:
            tree = pac_decompose(idx, per_idx)
            leaf_pots = collect_potential(tree, leaves_only=True)
            for k, v in leaf_pots.items():
                leaf_mesh[k] += v
    total_leaf = sum(leaf_mesh.values())
    total_flow = sum(mesh.values())
    conservation_global = abs(total_leaf - len(KNOWN_FORMULAS)) < 0.01
    passed = conservation_global and pac_cv < raw_cv

    print(f"\n  Leaf potential (strict PAC): {total_leaf:.4f} (expected: {len(KNOWN_FORMULAS)}.0)")
    print(f"  Flow-through potential:      {total_flow:.4f} (> budget, by design)")
    print(f"  Conservation (leaf == budget): {conservation_global}")
    print(f"  PASS: {passed} (leaf conservation holds AND PAC reduces depth bias)")

    return {
        'mesh': {str(k): v for k, v in mesh.items()},
        'total_leaf_potential': float(total_leaf),
        'total_flow_potential': float(total_flow),
        'conservation_ok': conservation_global,
        'pac_cv': float(pac_cv),
        'raw_cv': float(raw_cv),
        'bias_reduction_pct': float((1 - pac_cv/raw_cv)*100) if raw_cv > 0 else 0,
        'passed': passed,
    }


def _raw_decompose(n, memo=None):
    """Quick raw visit counting (as in exp_20)."""
    if memo is None:
        memo = {}
    if n in memo:
        return Counter(memo[n])
    result = Counter({n: 1})
    if n > 2:
        result += _raw_decompose(n - 1, memo)
        result += _raw_decompose(n - 2, memo)
    memo[n] = dict(result)
    return result


# =====================================================================
# Test 2: Profile-Based Discrimination (CORE TEST)
# =====================================================================
def test_2_profile_discrimination():
    """
    Score novel formulas by PAC profile similarity to known mesh.
    Physics-matching should have higher cosine similarity.
    """
    print("\n" + "=" * 70)
    print("TEST 2: PAC Profile-Based Discrimination (CORE TEST)")
    print("=" * 70)

    mesh_profile = build_pac_mesh(KNOWN_FORMULAS)
    idx_results = compute_index_set_matches(FIB_VALUES, NOVEL_TARGETS)

    n_matched = sum(1 for v in idx_results.values() if v['matched'])
    n_total = len(idx_results)
    print(f"\n  Index sets: {n_total} total, {n_matched} matched ({n_matched/n_total:.1%})")

    # Score each index set by cosine similarity of PAC profile to mesh
    cos_matched = []
    cos_unmatched = []
    kl_matched = []
    kl_unmatched = []

    scored_items = []

    for idx_set, info in idx_results.items():
        novel_profile = formula_pac_profile(sorted(idx_set))
        cos_sim = profile_cosine_similarity(novel_profile, mesh_profile)
        kl_div = profile_kl_divergence(novel_profile, mesh_profile)

        scored_items.append((idx_set, info, cos_sim, kl_div))

        if info['matched']:
            cos_matched.append(cos_sim)
            kl_matched.append(kl_div)
        else:
            cos_unmatched.append(cos_sim)
            kl_unmatched.append(kl_div)

    # Statistics — Cosine Similarity
    mean_cos_m = float(np.mean(cos_matched))
    mean_cos_u = float(np.mean(cos_unmatched))

    print(f"\n  Cosine Similarity (PAC profile to mesh):")
    print(f"  {'':30s}  {'Mean':>10s}  {'Std':>10s}  {'N':>6s}")
    print(f"  {'Physics matches (<1%)':30s}  {mean_cos_m:10.6f}  {np.std(cos_matched):10.6f}  {len(cos_matched):6d}")
    print(f"  {'Non-matches':30s}  {mean_cos_u:10.6f}  {np.std(cos_unmatched):10.6f}  {len(cos_unmatched):6d}")
    print(f"  Difference: {mean_cos_m - mean_cos_u:+.6f}")

    if len(cos_matched) >= 5 and len(cos_unmatched) >= 5:
        stat_cos, p_cos = stats.mannwhitneyu(
            cos_matched, cos_unmatched, alternative='greater')
        print(f"\n  Mann-Whitney U (matched > unmatched): U={stat_cos:.0f}, p={p_cos:.6f}")
    else:
        p_cos = 1.0

    # Statistics — KL Divergence
    mean_kl_m = float(np.mean(kl_matched))
    mean_kl_u = float(np.mean(kl_unmatched))

    print(f"\n  KL Divergence (novel -> mesh, lower = more similar):")
    print(f"  {'Physics matches (<1%)':30s}  {mean_kl_m:10.6f}")
    print(f"  {'Non-matches':30s}  {mean_kl_u:10.6f}")
    print(f"  Difference: {mean_kl_m - mean_kl_u:+.6f}")

    if len(kl_matched) >= 5 and len(kl_unmatched) >= 5:
        stat_kl, p_kl = stats.mannwhitneyu(
            kl_matched, kl_unmatched, alternative='less')
        print(f"  Mann-Whitney U (matched < unmatched): U={stat_kl:.0f}, p={p_kl:.6f}")
    else:
        p_kl = 1.0

    # Top matches by cosine similarity
    scored_matched = [(s, i, c, k) for s, i, c, k in scored_items if i['matched']]
    scored_matched.sort(key=lambda x: -x[2])

    print(f"\n  Top physics matches by PAC cosine similarity:")
    print(f"  {'Index set':25s}  {'Target':15s}  {'Err%':>7s}  {'CosSim':>8s}  {'KL':>8s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*7}  {'-'*8}  {'-'*8}")
    for idx_set, info, cos_s, kl_d in scored_matched[:15]:
        print(f"  {str(sorted(idx_set)):25s}  {info['best_target']:15s}  "
              f"{info['best_error']:7.3f}  {cos_s:8.4f}  {kl_d:8.4f}")

    # Enrichment at high cosine similarity
    pct_75 = np.percentile([c for _, _, c, _ in scored_items], 75)
    n_high_m = sum(1 for c in cos_matched if c >= pct_75)
    n_high_u = sum(1 for c in cos_unmatched if c >= pct_75)
    rate_m = n_high_m / len(cos_matched) if cos_matched else 0
    rate_u = n_high_u / len(cos_unmatched) if cos_unmatched else 0
    enrichment = rate_m / rate_u if rate_u > 0 else float('inf')

    print(f"\n  High-similarity enrichment (>= 75th percentile = {pct_75:.4f}):")
    print(f"    Matched rate:   {rate_m:.1%} ({n_high_m}/{len(cos_matched)})")
    print(f"    Unmatched rate: {rate_u:.1%} ({n_high_u}/{len(cos_unmatched)})")
    print(f"    Enrichment:     {enrichment:.2f}x")

    # Effect size
    all_cos = cos_matched + cos_unmatched
    if np.std(all_cos) > 0:
        cohens_d = (mean_cos_m - mean_cos_u) / np.std(all_cos)
    else:
        cohens_d = 0

    print(f"\n  Effect size (Cohen's d): {cohens_d:.4f}")

    passed = (p_cos < 0.05 and mean_cos_m > mean_cos_u) or \
             (p_kl < 0.05 and mean_kl_m < mean_kl_u)
    print(f"\n  PASS: {passed} (cosine p < 0.05 OR KL p < 0.05, in correct direction)")

    return {
        'mean_cos_matched': mean_cos_m,
        'mean_cos_unmatched': mean_cos_u,
        'cos_difference': float(mean_cos_m - mean_cos_u),
        'cos_p_value': float(p_cos),
        'mean_kl_matched': mean_kl_m,
        'mean_kl_unmatched': mean_kl_u,
        'kl_p_value': float(p_kl),
        'enrichment_75pct': float(enrichment),
        'cohens_d': float(cohens_d),
        'top_matches': [{'indices': sorted(s), 'target': i['best_target'],
                         'error': i['best_error'], 'cosine': c, 'kl': k}
                        for s, i, c, k in scored_matched[:10]],
        'passed': passed,
    }


# =====================================================================
# Test 3: SEC-Gated Adaptive Depth
# =====================================================================
def test_3_sec_gated_depth():
    """
    Apply SEC gating: only expand at contested nodes.
    Show that depth adapts per formula.
    Then test whether gated profiles discriminate better.
    """
    print("\n" + "=" * 70)
    print("TEST 3: SEC-Gated Adaptive Depth")
    print("=" * 70)

    # First build the formula_potentials map (which formulas contribute what at each index)
    formula_potentials = defaultdict(list)
    for name, indices in KNOWN_FORMULAS.items():
        profile = formula_pac_profile(indices)
        for idx, pot in profile.items():
            formula_potentials[idx].append(pot)

    # Show gated vs ungated depth for each formula
    print(f"\n  Formula depths — ungated vs SEC-gated:")
    print(f"  {'Formula':15s}  {'Indices':12s}  {'Ungated':>8s}  {'Gated':>8s}  {'Reduction':>10s}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}")

    ungated_depths = {}
    gated_depths = {}

    for name, indices in KNOWN_FORMULAS.items():
        per_idx = 1.0 / len(indices)
        max_ungated = 0
        max_gated = 0

        for idx in indices:
            tree_ug = pac_decompose(idx, per_idx)
            tree_g = pac_decompose_sec_gated(idx, per_idx, dict(formula_potentials))
            max_ungated = max(max_ungated, tree_depth(tree_ug))
            max_gated = max(max_gated, tree_depth(tree_g))

        ungated_depths[name] = max_ungated
        gated_depths[name] = max_gated
        reduction = (max_ungated - max_gated) / max_ungated * 100 if max_ungated > 0 else 0

        print(f"  {name:15s}  {str(indices):12s}  {max_ungated:8d}  {max_gated:8d}  {reduction:9.1f}%")

    # Build gated mesh
    gated_mesh = defaultdict(float)
    for name, indices in KNOWN_FORMULAS.items():
        per_idx = 1.0 / len(indices)
        for idx in indices:
            tree = pac_decompose_sec_gated(idx, per_idx, dict(formula_potentials))
            potentials = collect_potential(tree)
            for k, v in potentials.items():
                gated_mesh[k] += v
    gated_mesh = dict(gated_mesh)

    # Score novel formulas with gated profiles
    idx_results = compute_index_set_matches(FIB_VALUES, NOVEL_TARGETS)

    # Also score ungated for comparison
    ungated_mesh = build_pac_mesh(KNOWN_FORMULAS)

    cos_matched_gated = []
    cos_unmatched_gated = []
    cos_matched_ungated = []
    cos_unmatched_ungated = []

    for idx_set, info in idx_results.items():
        novel_profile = formula_pac_profile(sorted(idx_set))
        cos_g = profile_cosine_similarity(novel_profile, gated_mesh)
        cos_u = profile_cosine_similarity(novel_profile, ungated_mesh)

        if info['matched']:
            cos_matched_gated.append(cos_g)
            cos_matched_ungated.append(cos_u)
        else:
            cos_unmatched_gated.append(cos_g)
            cos_unmatched_ungated.append(cos_u)

    mean_m_g = float(np.mean(cos_matched_gated))
    mean_u_g = float(np.mean(cos_unmatched_gated))
    delta_gated = mean_m_g - mean_u_g

    mean_m_ug = float(np.mean(cos_matched_ungated))
    mean_u_ug = float(np.mean(cos_unmatched_ungated))
    delta_ungated = mean_m_ug - mean_u_ug

    if len(cos_matched_gated) >= 5 and len(cos_unmatched_gated) >= 5:
        _, p_gated = stats.mannwhitneyu(
            cos_matched_gated, cos_unmatched_gated, alternative='greater')
    else:
        p_gated = 1.0

    print(f"\n  SEC-gated vs ungated cosine similarity:")
    print(f"  {'':20s}  {'Gated':>10s}  {'Ungated':>10s}")
    print(f"  {'Matched mean':20s}  {mean_m_g:10.6f}  {mean_m_ug:10.6f}")
    print(f"  {'Unmatched mean':20s}  {mean_u_g:10.6f}  {mean_u_ug:10.6f}")
    print(f"  {'Delta (M-U)':20s}  {delta_gated:+10.6f}  {delta_ungated:+10.6f}")
    print(f"  Gated p-value: {p_gated:.6f}")
    delta_improvement = ((delta_gated - delta_ungated) / abs(delta_ungated) * 100
                         if abs(delta_ungated) > 1e-12 else 0)
    print(f"  Delta improvement from gating: {delta_improvement:+.1f}%")

    # Does gating improve depth gradient?
    # Higher-index formulas should have more reduction
    max_idx = {name: max(indices) for name, indices in KNOWN_FORMULAS.items()}
    reductions = {name: ungated_depths[name] - gated_depths[name] for name in KNOWN_FORMULAS}

    idx_vals = [max_idx[n] for n in KNOWN_FORMULAS]
    red_vals = [reductions[n] for n in KNOWN_FORMULAS]
    if np.std(idx_vals) > 0 and np.std(red_vals) > 0:
        depth_corr, depth_p = stats.spearmanr(idx_vals, red_vals)
    else:
        depth_corr, depth_p = 0, 1.0

    print(f"\n  Depth reduction vs formula max index:")
    print(f"  Spearman correlation: {depth_corr:.3f}, p={depth_p:.4f}")
    print(f"  (Positive = higher-index formulas get more depth reduction)")

    # Was SEC gating useful (adaptive)?
    gated_reduced = sum(1 for n in KNOWN_FORMULAS if gated_depths[n] < ungated_depths[n])
    mean_reduction = np.mean([ungated_depths[n] - gated_depths[n] for n in KNOWN_FORMULAS])

    print(f"\n  SEC gating summary:")
    print(f"  Formulas with reduced depth: {gated_reduced}/{len(KNOWN_FORMULAS)}")
    print(f"  Mean depth reduction: {mean_reduction:.1f} levels")

    # PASS criteria: gating is active AND improves or maintains discrimination
    # SEC gating should: (1) be observable (some depth reduction), AND
    # (2) not harm the signal (delta >= ungated OR p improves)
    gating_active = gated_reduced >= 1
    gating_helps = delta_gated >= delta_ungated
    passed = gating_active and (p_gated < 0.05 or (mean_m_g > mean_u_g and gating_helps))
    print(f"\n  PASS: {passed} (gating active [{gating_active}] AND "
          f"(p<0.05 OR correct direction with gating helps [{gating_helps}]))")

    return {
        'ungated_depths': ungated_depths,
        'gated_depths': gated_depths,
        'gated_reduced_count': gated_reduced,
        'mean_depth_reduction': float(mean_reduction),
        'gated_cos_matched': mean_m_g,
        'gated_cos_unmatched': mean_u_g,
        'ungated_cos_matched': mean_m_ug,
        'ungated_cos_unmatched': mean_u_ug,
        'delta_gated': float(delta_gated),
        'delta_ungated': float(delta_ungated),
        'delta_improvement_pct': float(delta_improvement),
        'gated_p_value': float(p_gated),
        'depth_corr': float(depth_corr),
        'depth_corr_p': float(depth_p),
        'passed': passed,
    }


# =====================================================================
# Test 4: PAC Profile vs Raw Pressure (exp_20 comparison)
# =====================================================================
def test_4_pac_vs_raw():
    """
    Direct comparison: does PAC-conserved profile similarity discriminate
    physics better than raw pressure (exp_20's approach)?
    Same data, same matches, different metric.
    """
    print("\n" + "=" * 70)
    print("TEST 4: PAC Profile vs Raw Pressure — Conservation Helps?")
    print("=" * 70)

    # PAC mesh
    pac_mesh = build_pac_mesh(KNOWN_FORMULAS)

    # Raw mesh (visit counting, as in exp_20)
    raw_mesh = Counter()
    for name, indices in KNOWN_FORMULAS.items():
        for idx in indices:
            raw_mesh += _raw_decompose(idx)

    idx_results = compute_index_set_matches(FIB_VALUES, NOVEL_TARGETS)

    # Score each index set both ways
    pac_cos_m, pac_cos_u = [], []
    raw_pres_m, raw_pres_u = [], []

    for idx_set, info in idx_results.items():
        # PAC: cosine similarity
        novel_pac = formula_pac_profile(sorted(idx_set))
        cos_sim = profile_cosine_similarity(novel_pac, pac_mesh)

        # Raw: dot product (as exp_20)
        novel_raw = Counter()
        for idx in sorted(idx_set):
            novel_raw += _raw_decompose(idx)
        raw_pressure = sum(novel_raw.get(k, 0) * raw_mesh.get(k, 0) for k in novel_raw)

        if info['matched']:
            pac_cos_m.append(cos_sim)
            raw_pres_m.append(raw_pressure)
        else:
            pac_cos_u.append(cos_sim)
            raw_pres_u.append(raw_pressure)

    # PAC statistics
    pac_delta = float(np.mean(pac_cos_m) - np.mean(pac_cos_u))
    _, pac_p = stats.mannwhitneyu(pac_cos_m, pac_cos_u, alternative='greater') \
        if len(pac_cos_m) >= 5 else (0, 1.0)

    # Raw statistics
    raw_delta = float(np.mean(raw_pres_m) - np.mean(raw_pres_u))
    _, raw_p = stats.mannwhitneyu(raw_pres_m, raw_pres_u, alternative='greater') \
        if len(raw_pres_m) >= 5 else (0, 1.0)

    # Effect sizes
    all_pac = pac_cos_m + pac_cos_u
    pac_d = pac_delta / np.std(all_pac) if np.std(all_pac) > 0 else 0
    all_raw = raw_pres_m + raw_pres_u
    raw_d = raw_delta / np.std(all_raw) if np.std(all_raw) > 0 else 0

    print(f"\n  {'Metric':20s}  {'mean_M':>10s}  {'mean_U':>10s}  {'Delta':>10s}  "
          f"{'p-val':>10s}  {'Cohen d':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'PAC cos sim':20s}  {np.mean(pac_cos_m):10.6f}  {np.mean(pac_cos_u):10.6f}  "
          f"{pac_delta:+10.6f}  {pac_p:10.6f}  {pac_d:+10.4f}")
    print(f"  {'Raw pressure':20s}  {np.mean(raw_pres_m):10.1f}  {np.mean(raw_pres_u):10.1f}  "
          f"{raw_delta:+10.1f}  {raw_p:10.6f}  {raw_d:+10.4f}")

    # Is PAC better?
    pac_better = pac_p < raw_p and pac_delta > 0
    pac_significant = pac_p < 0.05

    print(f"\n  PAC strictly better: {pac_better} (lower p AND positive delta)")
    print(f"  PAC significant: {pac_significant} (p < 0.05)")
    print(f"  Raw direction: {'correct' if raw_delta > 0 else 'WRONG (negative)'}")
    print(f"  PAC direction: {'correct' if pac_delta > 0 else 'WRONG (negative)'}")

    # Even if not significant, is direction changed?
    direction_improved = pac_delta > 0 and raw_delta <= 0
    if direction_improved:
        print(f"\n  ** DIRECTION CORRECTED: PAC flips raw's negative delta to positive! **")

    passed = pac_significant or (pac_better and pac_p < 0.10) or direction_improved
    print(f"\n  PASS: {passed} (significant, OR strictly better at p<0.10, "
          f"OR direction corrected)")

    return {
        'pac_delta': float(pac_delta),
        'pac_p': float(pac_p),
        'pac_cohens_d': float(pac_d),
        'raw_delta': float(raw_delta),
        'raw_p': float(raw_p),
        'raw_cohens_d': float(raw_d),
        'pac_better': pac_better,
        'pac_significant': pac_significant,
        'direction_improved': direction_improved,
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_21_pac_lazy_formula_mesh',
        'PAC-Lazy conservation + SEC gating + profile comparison on formula mesh',
        paper='Paper 4',
        section='$pac_lazy_mesh'
    )

    results = {'metadata': meta, 'tests': {}}

    results['tests']['test_1_pac_distribution'] = test_1_pac_distribution()
    results['tests']['test_2_profile_discrimination'] = test_2_profile_discrimination()
    results['tests']['test_3_sec_gated_depth'] = test_3_sec_gated_depth()
    results['tests']['test_4_pac_vs_raw'] = test_4_pac_vs_raw()

    # --- Synthesis ---
    print("\n" + "=" * 70)
    print("  SYNTHESIS: PAC-Lazy Formula Mesh")
    print("=" * 70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:40s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    # Key comparisons
    t2 = results['tests']['test_2_profile_discrimination']
    t4 = results['tests']['test_4_pac_vs_raw']

    print(f"""
  ---------------------------------------------------------------
  PAC LAZY ARCHITECTURE APPLIED TO FORMULA MESH

  From the GAIA POCs (poc_011, poc_016-018):
  1. PAC Conservation: f(parent) = sum(children)
     Each formula distributes exactly 1.0 potential
     through its recursion tree. phi-weighted splitting.

  2. SEC Gating: C(S) = S * exp(-xi * S)
     Expand only when potential diversity is high.
     Crystallize (stop) when formulas agree.

  3. Profile Comparison (not scalar pressure)
     Cosine similarity of PAC distribution captures
     SHAPE, not magnitude. Avoids depth bias.

  exp_20 FAILURE: raw pressure = visit count = depth bias.
  exp_21 CORRECTION: PAC conservation + profile comparison.

  Core test (T2):
    PAC cosine delta: {t2['cos_difference']:+.6f}
    p-value:          {t2['cos_p_value']:.6f}

  vs Raw (T4):
    PAC direction:    {'correct' if t4['pac_delta'] > 0 else 'wrong'}
    Raw direction:    {'correct' if t4['raw_delta'] > 0 else 'WRONG'}
    Direction fix:    {t4.get('direction_improved', False)}
  ---------------------------------------------------------------""")

    results['summary'] = {
        'total': total, 'passed': pass_count,
        'score': f"{pass_count}/{total}",
    }

    results['falsification'] = {
        'test_id': 'experimental (not in registry)',
        'hypothesis': (
            'PAC-conserved potential distribution profiles, scored by cosine '
            'similarity to the known formula mesh, discriminate physics-matching '
            'novel formulas from non-matching. SEC gating creates adaptive depth.'
        ),
        'falsified_if': (
            'PAC profile cosine similarity does not discriminate physics matches, '
            'AND PAC does not improve over raw pressure (exp_20).'
        ),
        'falsified': pass_count < 2,
        'assessment': f"{pass_count}/{total} tests pass.",
    }

    save_results(results, 'exp_21_pac_lazy_formula_mesh')
    return results


if __name__ == '__main__':
    main()
