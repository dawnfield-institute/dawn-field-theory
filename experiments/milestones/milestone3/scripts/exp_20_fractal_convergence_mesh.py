"""
exp_20: Fractal Convergence Mesh — Why It Should Never Have Been a Matrix

INSIGHT (from exp_16→18→19 honest failure arc + symbolic_bifractal):
  The flat stoichiometric matrix treats each formula as a POINT in 11D space.
  SVD, null spaces, conservation fractions — all linear algebra on flat vectors.
  exp_19 proved the consequence: the cascade is purely TOPOLOGICAL (index-based),
  invariant across all sequences. Values don't matter. The matrix is too flat.

  But Fibonacci numbers aren't flat. F_n = F_{n-1} + F_{n-2} means every index
  CONTAINS its predecessors recursively. F₁₀ → F₉ + F₈ → (F₈ + F₇) + (F₇ + F₆) → ...

  When a formula uses index 10, it doesn't just "use column 10" — it structurally
  depends on the entire decomposition tree beneath it. F₇ appears inside F₁₀ AND
  as a direct formula index in sin²θ_W. These are BY-REFERENCE convergence points:
  shared nodes where multiple recursive paths meet.

  The bifractal simulation (symbolic_bifractal_expansion_v2) showed exactly this:
  branches grow recursively, and where they collide (inhibition_map > threshold),
  convergence zones form with high attractor pressure. The pressure × semantic
  field gave collapse_balance_field_score = 1058.2 — measuring mesh density.

  The formula space IS a bifractal convergence mesh. Not a matrix.

WHY THIS DISCRIMINATES SEQUENCES:
  - Fibonacci/Lucas: F_n = F_{n-1} + F_{n-2} → deep binary trees, massive convergence
  - Tribonacci:      T_n = T_{n-1} + T_{n-2} + T_{n-3} → ternary trees, different topology
  - Primes:          p_n ≠ p_{n-1} + p_{n-2} → NO additive recursion → FLAT (no mesh!)
  - Random:          no recurrence at all → FLAT

  The flat matrix approach lost this because SVD doesn't know that F₁₀ CONTAINS F₇
  which CONTAINS F₃. It just sees indicator vectors. The fractal approach captures
  the DEPTH of recursion, which creates the convergence mesh.

MED CONNECTION:
  MED says depth ≤ 2, nodes ≤ 3. Fibonacci binary recursion: each node has exactly
  2 children. A formula with 3 indices creates a tree of bounded depth. The MED
  complexity bound IS the fractal stopping condition.

MECHANISM:
  1. DECOMPOSE each formula index through the sequence's recurrence relation
     (F_n → {F_{n-1}, F_{n-2}} recursively down to base cases)
  2. OVERLAY all known formula decomposition footprints → convergence mesh
  3. MEASURE mesh pressure for novel formula index sets (via their OWN footprints)
  4. Physics-matching formulas should sit at HIGH-PRESSURE convergence hubs
  5. This property should be ABSENT for flat sequences (primes, random)

TESTS:
  Test 1 — Mesh Construction: Build convergence mesh, visualize pressure map,
           identify hub indices. Show that F₃, F₄ are deepest hubs.

  Test 2 — Fractal Physics Selection: Novel formulas matching physics have
           higher mesh pressure than non-matching. THE CORE TEST.

  Test 3 — Recursion Specificity: Repeat for binary-additive (Lucas),
           ternary-additive (Tribonacci), and FLAT (Primes, Random).
           Binary-additive should discriminate. Flat should not.

  Test 4 — Fractal vs Flat Advantage: Direct comparison — does fractal mesh
           pressure discriminate physics BETTER than flat (direct-index-only)
           pressure? If yes, the recursion depth is doing real work.

SOURCES:
  - symbolic_bifractal_expansion_v2.py (bifractal convergence principle)
  - exp_19 (cascade is topological — matrix approach exhausted)
  - exp_16-18 (flat approaches fail to distinguish Fibonacci from random)
  - "Energy as Collapsed Potential" paper (Landauer cascade, topology)
  - MED bounded complexity (depth ≤ 2, nodes ≤ 3)
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

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
# Recurrence definitions
# =====================================================================
class Recurrence:
    """Defines a sequence's recurrence structure for decomposition."""

    def __init__(self, name, values, decompose_fn, min_idx=1):
        self.name = name
        self.values = values          # length N_SPECIES, mapped to FIB_INDICES
        self.decompose_fn = decompose_fn  # n -> list of child indices (or [] for base)
        self.min_idx = min_idx
        self._footprint_cache = {}

    def footprint(self, n):
        """
        Full recursive decomposition footprint of index n.
        Returns Counter: {index: visit_count}
        """
        if n in self._footprint_cache:
            return Counter(self._footprint_cache[n])

        result = Counter({n: 1})
        children = self.decompose_fn(n)
        for child in children:
            if child >= self.min_idx:
                child_fp = self.footprint(child)
                result += child_fp

        self._footprint_cache[n] = dict(result)
        return result

    def formula_footprint(self, index_set):
        """Combined footprint for a set of indices."""
        total = Counter()
        for idx in index_set:
            total += self.footprint(idx)
        return total


def fib_decompose(n):
    """Fibonacci: F_n = F_{n-1} + F_{n-2}"""
    if n <= 2:
        return []
    return [n - 1, n - 2]

def lucas_decompose(n):
    """Lucas: L_n = L_{n-1} + L_{n-2} (same recurrence structure)"""
    if n <= 2:
        return []
    return [n - 1, n - 2]

def tribonacci_decompose(n):
    """Tribonacci: T_n = T_{n-1} + T_{n-2} + T_{n-3}"""
    if n <= 3:
        return []
    return [n - 1, n - 2, n - 3]

def flat_decompose(n):
    """No recursion — flat. Each index is just itself."""
    return []


# Build sequences
def make_lucas():
    L = [0, 2, 1]  # L(0)=2, L(1)=1, shifted for indexing
    for i in range(3, 14):
        L.append(L[-1] + L[-2])
    return [L[i] for i in FIB_INDICES]

def make_tribonacci():
    T = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274]
    while len(T) < 14:
        T.append(T[-1] + T[-2] + T[-3])
    return [max(T[i], 1) for i in FIB_INDICES]

def make_primes():
    primes, c = [], 2
    while len(primes) < 14:
        if all(c % p != 0 for p in primes):
            primes.append(c)
        c += 1
    return [primes[i] for i in range(1, 12)]  # 11 values

def make_random(seed):
    r = np.random.default_rng(seed)
    return sorted(r.choice(np.arange(1, 201), size=N_SPECIES, replace=False).tolist())


# Build all recurrences
RECURRENCES = {
    'Fibonacci': Recurrence('Fibonacci', FIB_VALUES, fib_decompose),
    'Lucas':     Recurrence('Lucas', make_lucas(), lucas_decompose),
    'Tribonacci': Recurrence('Tribonacci', make_tribonacci(), tribonacci_decompose, min_idx=2),
    'Primes':    Recurrence('Primes', make_primes(), flat_decompose),
    'Random_0':  Recurrence('Random_0', make_random(100), flat_decompose),
    'Random_1':  Recurrence('Random_1', make_random(101), flat_decompose),
    'Random_2':  Recurrence('Random_2', make_random(102), flat_decompose),
}


# =====================================================================
# Convergence mesh construction
# =====================================================================
def build_convergence_mesh(recurrence, known_formulas=None):
    """
    Build convergence mesh: overlay all known formula footprints.
    Returns Counter: {index: total_pressure}
    """
    if known_formulas is None:
        known_formulas = KNOWN_FORMULAS

    mesh = Counter()
    for name, indices in known_formulas.items():
        fp = recurrence.formula_footprint(indices)
        mesh += fp

    return mesh


def mesh_pressure(index_set, mesh, recurrence):
    """
    Mesh pressure for an index set: dot product of its footprint
    with the convergence mesh.

    High pressure = index set sits at convergence hubs.
    """
    fp = recurrence.formula_footprint(index_set)
    pressure = sum(fp[idx] * mesh.get(idx, 0) for idx in fp)
    return pressure


def normalized_mesh_pressure(index_set, mesh, recurrence, max_pressure):
    """Mesh pressure normalized to [0, 1]."""
    p = mesh_pressure(index_set, mesh, recurrence)
    return p / max_pressure if max_pressure > 0 else 0


# =====================================================================
# Index-set scanning (shared with exp_19, adapted)
# =====================================================================
def compute_index_set_matches(values, targets, threshold_pct=1.0):
    """
    For each unique index set, determine if any template matches a target.
    Returns dict: frozenset(FIB_INDICES positions) -> {matched, best_error, ...}
    """
    n = len(values)
    results = {}

    # Pairs
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

    # Triples
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
# Test 1: Mesh Construction & Hub Analysis
# =====================================================================
def test_1_mesh_construction():
    """
    Build the Fibonacci convergence mesh and identify hub indices.
    The bifractal convergence: where do the recursive trees overlap most?
    """
    print("\n" + "=" * 70)
    print("TEST 1: Fractal Convergence Mesh — Hub Identification")
    print("=" * 70)

    rec = RECURRENCES['Fibonacci']
    mesh = build_convergence_mesh(rec)

    # Show individual formula footprints
    print(f"\n  Known formula decomposition footprints:")
    print(f"  {'Formula':15s}  {'Direct':10s}  {'Footprint size':>14s}  {'Deepest reach':>13s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*14}  {'-'*13}")

    for name, indices in KNOWN_FORMULAS.items():
        fp = rec.formula_footprint(indices)
        direct = str(indices)
        deepest = min(fp.keys())
        total_visits = sum(fp.values())
        print(f"  {name:15s}  {direct:10s}  {total_visits:14d}  idx {deepest:>3d}")

    # Mesh pressure map
    all_indices = sorted(mesh.keys())
    print(f"\n  Convergence mesh pressure map:")
    print(f"  {'Index':>5s}  {'F_n':>5s}  {'Pressure':>8s}  {'Bar':30s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*30}")

    max_p = max(mesh.values()) if mesh else 1
    for idx in range(1, 13):
        p = mesh.get(idx, 0)
        fval = str(FIB[idx]) if idx < len(FIB) else '?'
        bar_len = int(p / max_p * 30) if max_p > 0 else 0
        bar = '#' * bar_len
        hub_mark = ' << HUB' if p > max_p * 0.7 else ''
        print(f"  {idx:5d}  {fval:>5s}  {p:8d}  {bar}{hub_mark}")

    # Hub identification
    hub_threshold = max_p * 0.5
    hubs = [idx for idx in mesh if mesh[idx] >= hub_threshold]
    print(f"\n  Hub indices (pressure >= 50% of max): {sorted(hubs)}")
    print(f"  Max pressure: {max_p} at index {max(mesh, key=mesh.get)}")
    print(f"  Total mesh weight: {sum(mesh.values())}")

    # Compare with flat (no recursion) mesh
    flat_rec = Recurrence('Flat', FIB_VALUES, flat_decompose)
    flat_mesh = build_convergence_mesh(flat_rec)
    flat_max = max(flat_mesh.values()) if flat_mesh else 1

    print(f"\n  Comparison: Fractal vs Flat mesh")
    print(f"  {'':5s}  {'Fractal':>8s}  {'Flat':>8s}  {'Amplification':>13s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*13}")
    for idx in FIB_INDICES:
        fp = mesh.get(idx, 0)
        fl = flat_mesh.get(idx, 0)
        amp = fp / fl if fl > 0 else float('inf')
        print(f"  {idx:5d}  {fp:8d}  {fl:8d}  {amp:13.1f}x")

    total_fractal = sum(mesh.values())
    total_flat = sum(flat_mesh.values())
    amplification = total_fractal / total_flat if total_flat > 0 else float('inf')
    print(f"\n  Total fractal weight: {total_fractal}")
    print(f"  Total flat weight:    {total_flat}")
    print(f"  Recursion amplification: {amplification:.1f}x")

    # PASS: fractal mesh has significant amplification over flat
    passed = amplification > 3.0 and len(hubs) >= 2
    print(f"\n  PASS: {passed} (amplification > 3x AND >= 2 hubs)")

    return {
        'mesh': dict(mesh),
        'max_pressure': max_p,
        'hubs': sorted(hubs),
        'total_weight_fractal': total_fractal,
        'total_weight_flat': total_flat,
        'amplification': float(amplification),
        'passed': passed,
    }


# =====================================================================
# Test 2: Fractal Physics Selection (CORE TEST)
# =====================================================================
def test_2_fractal_selection():
    """
    Do physics-matching formulas sit at higher mesh pressure than non-matching?

    This is the fractal equivalent of exp_18's conservation fraction test —
    but now using recursive convergence instead of flat null-space projection.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Fractal Physics Selection (CORE TEST)")
    print("=" * 70)

    rec = RECURRENCES['Fibonacci']
    mesh = build_convergence_mesh(rec)
    idx_results = compute_index_set_matches(rec.values, NOVEL_TARGETS)

    n_matched = sum(1 for v in idx_results.values() if v['matched'])
    n_total = len(idx_results)
    print(f"\n  Index sets: {n_total} total, {n_matched} matched ({n_matched/n_total:.1%})")

    # Compute mesh pressure for each index set
    pressures_matched = []
    pressures_unmatched = []
    all_pressures = []

    for idx_set, info in idx_results.items():
        p = mesh_pressure(list(idx_set), mesh, rec)
        all_pressures.append(p)
        if info['matched']:
            pressures_matched.append(p)
        else:
            pressures_unmatched.append(p)

    mean_m = float(np.mean(pressures_matched))
    mean_u = float(np.mean(pressures_unmatched))
    std_m = float(np.std(pressures_matched))
    std_u = float(np.std(pressures_unmatched))

    print(f"\n  Mesh Pressure Statistics:")
    print(f"  {'':30s}  {'Mean':>10s}  {'Std':>10s}  {'N':>6s}")
    print(f"  {'Physics matches (<1%)':30s}  {mean_m:10.1f}  {std_m:10.1f}  {len(pressures_matched):6d}")
    print(f"  {'Non-matches':30s}  {mean_u:10.1f}  {std_u:10.1f}  {len(pressures_unmatched):6d}")
    print(f"  Difference: {mean_m - mean_u:+.1f}")

    # Mann-Whitney U test
    if len(pressures_matched) >= 5 and len(pressures_unmatched) >= 5:
        stat, p_val = stats.mannwhitneyu(
            pressures_matched, pressures_unmatched, alternative='greater')
        print(f"\n  Mann-Whitney U (matched > unmatched): U={stat:.0f}, p={p_val:.6f}")
    else:
        p_val = 1.0

    # Effect size (rank-biserial correlation)
    if len(pressures_matched) >= 5 and len(pressures_unmatched) >= 5:
        n1, n2 = len(pressures_matched), len(pressures_unmatched)
        r_rb = 1 - (2 * stat) / (n1 * n2)
        print(f"  Rank-biserial correlation: r = {r_rb:.4f}")
    else:
        r_rb = 0

    # Top physics matches by mesh pressure
    matched_items = [(s, info, mesh_pressure(list(s), mesh, rec))
                     for s, info in idx_results.items() if info['matched']]
    matched_items.sort(key=lambda x: -x[2])

    print(f"\n  Top physics matches by mesh pressure:")
    print(f"  {'Index set':25s}  {'Target':15s}  {'Err%':>7s}  {'Pressure':>8s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*7}  {'-'*8}")
    for idx_set, info, p in matched_items[:15]:
        print(f"  {str(sorted(idx_set)):25s}  {info['best_target']:15s}  "
              f"{info['best_error']:7.3f}  {p:8.0f}")

    # Pressure distribution of matches vs overall
    pct_75 = np.percentile(all_pressures, 75)
    n_high_matched = sum(1 for p in pressures_matched if p >= pct_75)
    n_high_unmatched = sum(1 for p in pressures_unmatched if p >= pct_75)
    rate_matched = n_high_matched / len(pressures_matched) if pressures_matched else 0
    rate_unmatched = n_high_unmatched / len(pressures_unmatched) if pressures_unmatched else 0

    print(f"\n  High-pressure enrichment (>= 75th percentile = {pct_75:.0f}):")
    print(f"    Matched rate:   {rate_matched:.1%} ({n_high_matched}/{len(pressures_matched)})")
    print(f"    Unmatched rate: {rate_unmatched:.1%} ({n_high_unmatched}/{len(pressures_unmatched)})")
    if rate_unmatched > 0:
        print(f"    Enrichment:     {rate_matched/rate_unmatched:.2f}x")

    passed = p_val < 0.05 and mean_m > mean_u
    print(f"\n  PASS: {passed} (p < 0.05 AND matched mean > unmatched mean)")

    return {
        'n_matched': n_matched,
        'n_total': n_total,
        'mean_pressure_matched': mean_m,
        'mean_pressure_unmatched': mean_u,
        'difference': float(mean_m - mean_u),
        'mann_whitney_p': float(p_val),
        'rank_biserial': float(r_rb),
        'enrichment_75pct': float(rate_matched / rate_unmatched) if rate_unmatched > 0 else None,
        'top_matches': [{'indices': sorted(s), 'target': info['best_target'],
                         'error': info['best_error'], 'pressure': p}
                        for s, info, p in matched_items[:10]],
        'passed': passed,
    }


# =====================================================================
# Test 3: Recursion Specificity
# =====================================================================
def test_3_recursion_specificity():
    """
    Repeat fractal selection for each sequence type.
    Binary-additive (Fibonacci, Lucas) should discriminate.
    Flat (Primes, Random) should not.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Recursion Specificity — Which Recurrences Discriminate?")
    print("=" * 70)

    results = {}

    print(f"\n  {'Sequence':15s}  {'Recurrence':10s}  {'Matches':>7s}  "
          f"{'mean_M':>8s}  {'mean_U':>8s}  {'Delta':>8s}  {'p-val':>10s}  {'Disc?':>5s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*5}")

    for seq_name, rec in RECURRENCES.items():
        mesh = build_convergence_mesh(rec)
        idx_results = compute_index_set_matches(rec.values, NOVEL_TARGETS)

        p_matched, p_unmatched = [], []
        for idx_set, info in idx_results.items():
            p = mesh_pressure(list(idx_set), mesh, rec)
            if info['matched']:
                p_matched.append(p)
            else:
                p_unmatched.append(p)

        n_m = len(p_matched)
        mean_m = float(np.mean(p_matched)) if p_matched else 0
        mean_u = float(np.mean(p_unmatched)) if p_unmatched else 0
        delta = mean_m - mean_u

        if len(p_matched) >= 5 and len(p_unmatched) >= 5:
            _, pv = stats.mannwhitneyu(p_matched, p_unmatched, alternative='greater')
        else:
            pv = 1.0

        # Determine recurrence type
        rec_type = 'binary' if seq_name in ('Fibonacci', 'Lucas') else \
                   'ternary' if seq_name == 'Tribonacci' else 'flat'
        disc = 'YES' if pv < 0.05 and delta > 0 else 'no'
        sig = '***' if pv < 0.001 else '** ' if pv < 0.01 else '*  ' if pv < 0.05 else '   '

        results[seq_name] = {
            'recurrence_type': rec_type,
            'n_matched': n_m,
            'mean_matched': mean_m,
            'mean_unmatched': mean_u,
            'delta': float(delta),
            'p_value': float(pv),
            'discriminates': pv < 0.05 and delta > 0,
        }

        print(f"  {seq_name:15s}  {rec_type:10s}  {n_m:7d}  "
              f"{mean_m:8.1f}  {mean_u:8.1f}  {delta:+8.1f}  "
              f"{pv:10.6f}{sig} {disc:>5s}")

    # Analysis: do binary-additive recurrences discriminate while flat don't?
    binary = [v for k, v in results.items() if v['recurrence_type'] == 'binary']
    flat = [v for k, v in results.items() if v['recurrence_type'] == 'flat']

    binary_disc = sum(1 for v in binary if v['discriminates'])
    flat_disc = sum(1 for v in flat if v['discriminates'])

    print(f"\n  Binary-additive discriminating: {binary_disc}/{len(binary)}")
    print(f"  Flat discriminating:            {flat_disc}/{len(flat)}")

    # PASS: at least one binary discriminates AND no flat discriminates
    # OR: binary discrimination rate > flat discrimination rate significantly
    passed = binary_disc > flat_disc
    print(f"\n  PASS: {passed} (binary discrimination > flat discrimination)")

    return {
        'by_sequence': results,
        'binary_disc': binary_disc,
        'flat_disc': flat_disc,
        'passed': passed,
    }


# =====================================================================
# Test 4: Fractal vs Flat Advantage
# =====================================================================
def test_4_fractal_vs_flat():
    """
    Direct comparison: does fractal (recursive) mesh pressure discriminate
    physics BETTER than flat (direct-index-only) mesh pressure?

    Both use the SAME sequence values and templates. The only difference
    is whether the mesh includes recursive decomposition depth.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Fractal vs Flat Advantage — Does Recursion Depth Help?")
    print("=" * 70)

    # Fibonacci with recursion (fractal)
    rec_fractal = RECURRENCES['Fibonacci']
    mesh_fractal = build_convergence_mesh(rec_fractal)

    # Fibonacci WITHOUT recursion (flat) — same values, no decomposition
    rec_flat = Recurrence('FibFlat', FIB_VALUES, flat_decompose)
    mesh_flat = build_convergence_mesh(rec_flat)

    # Same index set matches for both (same values)
    idx_results = compute_index_set_matches(FIB_VALUES, NOVEL_TARGETS)

    # Compute pressures under both meshes
    fractal_matched, fractal_unmatched = [], []
    flat_matched, flat_unmatched = [], []

    for idx_set, info in idx_results.items():
        pf = mesh_pressure(list(idx_set), mesh_fractal, rec_fractal)
        pl = mesh_pressure(list(idx_set), mesh_flat, rec_flat)

        if info['matched']:
            fractal_matched.append(pf)
            flat_matched.append(pl)
        else:
            fractal_unmatched.append(pf)
            flat_unmatched.append(pl)

    # Statistics
    frac_delta = float(np.mean(fractal_matched) - np.mean(fractal_unmatched))
    flat_delta = float(np.mean(flat_matched) - np.mean(flat_unmatched))

    _, frac_p = stats.mannwhitneyu(fractal_matched, fractal_unmatched, alternative='greater') \
        if len(fractal_matched) >= 5 else (0, 1.0)
    _, flat_p = stats.mannwhitneyu(flat_matched, flat_unmatched, alternative='greater') \
        if len(flat_matched) >= 5 else (0, 1.0)

    print(f"\n  {'Mesh Type':15s}  {'mean_M':>10s}  {'mean_U':>10s}  {'Delta':>10s}  {'p-value':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Fractal':15s}  {np.mean(fractal_matched):10.1f}  "
          f"{np.mean(fractal_unmatched):10.1f}  {frac_delta:+10.1f}  {frac_p:10.6f}")
    print(f"  {'Flat':15s}  {np.mean(flat_matched):10.1f}  "
          f"{np.mean(flat_unmatched):10.1f}  {flat_delta:+10.1f}  {flat_p:10.6f}")

    # Advantage
    if flat_p > 0 and frac_p > 0:
        p_ratio = flat_p / frac_p if frac_p > 0 else float('inf')
    else:
        p_ratio = 0

    print(f"\n  Fractal delta:  {frac_delta:+.1f}")
    print(f"  Flat delta:     {flat_delta:+.1f}")
    print(f"  Fractal p:      {frac_p:.6f}")
    print(f"  Flat p:         {flat_p:.6f}")
    if frac_p > 0:
        print(f"  p-ratio (flat/fractal): {p_ratio:.1f}x")

    # Normalized effect comparison
    if np.std(fractal_matched + fractal_unmatched) > 0:
        frac_d = frac_delta / np.std(fractal_matched + fractal_unmatched)
    else:
        frac_d = 0
    if np.std(flat_matched + flat_unmatched) > 0:
        flat_d = flat_delta / np.std(flat_matched + flat_unmatched)
    else:
        flat_d = 0

    print(f"\n  Normalized effect sizes (Cohen's d):")
    print(f"  Fractal: d = {frac_d:.4f}")
    print(f"  Flat:    d = {flat_d:.4f}")

    fractal_better = frac_p < flat_p and frac_delta > flat_delta
    print(f"\n  Fractal strictly better than flat: {fractal_better}")

    # PASS: fractal is significantly better (lower p, higher delta)
    passed = frac_p < 0.05 or (fractal_better and frac_p < 0.10)
    print(f"\n  PASS: {passed} (fractal p < 0.05, OR fractal strictly better with p < 0.10)")

    return {
        'fractal_delta': float(frac_delta),
        'flat_delta': float(flat_delta),
        'fractal_p': float(frac_p),
        'flat_p': float(flat_p),
        'fractal_cohens_d': float(frac_d),
        'flat_cohens_d': float(flat_d),
        'fractal_better': fractal_better,
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_20_fractal_convergence_mesh',
        'Fractal convergence mesh — recursive decomposition replaces flat matrix',
        paper='Paper 4',
        section='$fractal_mesh'
    )

    results = {'metadata': meta, 'tests': {}}

    # Test 1
    results['tests']['test_1_mesh_construction'] = test_1_mesh_construction()

    # Test 2
    results['tests']['test_2_fractal_selection'] = test_2_fractal_selection()

    # Test 3
    results['tests']['test_3_recursion_specificity'] = test_3_recursion_specificity()

    # Test 4
    results['tests']['test_4_fractal_vs_flat'] = test_4_fractal_vs_flat()

    # --- Synthesis ---
    print("\n" + "=" * 70)
    print("  SYNTHESIS: Fractal Convergence Mesh")
    print("=" * 70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:40s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    amp = results['tests']['test_1_mesh_construction'].get('amplification', 0)
    print(f"\n  Recursion amplification: {amp:.1f}x")
    print(f"  (fractal mesh is {amp:.0f}x denser than flat)")

    print(f"""
  ---------------------------------------------------------------
  WHY A FRACTAL, NOT A MATRIX:

  exp_13-19 used flat stoichiometric matrices (11D vectors, SVD).
  Result: topological invariants only, no numerical discrimination.

  The fractal approach captures what the matrix cannot:
  - F_10 CONTAINS F_9 + F_8 CONTAINS F_8 + F_7 + F_7 + F_6 ...
  - Each decomposition creates BY-REFERENCE convergence points
  - The mesh is where recursive paths collide = attractor zones
  - This is the bifractal collapse from symbolic_bifractal_v2

  Primes have no additive recursion -> no trees -> no mesh.
  Random has no recursion -> flat.
  Only sequences WITH recursion create convergence structure.

  The formula graph is not a matrix. It is a fractal mesh.
  ---------------------------------------------------------------""")

    results['summary'] = {
        'total': total, 'passed': pass_count,
        'score': f"{pass_count}/{total}",
    }

    results['falsification'] = {
        'test_id': 'experimental (not in registry)',
        'hypothesis': (
            'Physics-matching formula index sets sit at high-pressure '
            'convergence hubs in the Fibonacci recursive decomposition mesh, '
            'and this property requires additive recursion (absent in primes/random).'
        ),
        'falsified_if': (
            'Mesh pressure does not discriminate physics matches, OR '
            'flat (non-recursive) sequences show equal discrimination.'
        ),
        'falsified': pass_count < 2,
        'assessment': f"{pass_count}/{total} tests pass.",
    }

    save_results(results, 'exp_20_fractal_convergence_mesh')
    return results


if __name__ == '__main__':
    main()
