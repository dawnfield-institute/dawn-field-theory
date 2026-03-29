"""
exp_38_scoped_mediation.py -- Confluent Identity Phase 29

PURPOSE:
    Test the Scoped Mediation hypothesis: identity reweighting propagates
    level-by-level with no skip connections, matching ADE's arithmetic
    hierarchy where each operation is the recursive closure of the one
    immediately below.

    From the theoretical framework:
    - A grandparent cannot directly reweight a grandchild -- it must
      mediate through the parent (the permission boundary)
    - Each level transforms the signal before passing it on
    - The Fibonacci cascade is the natural propagation mode because
      F(n) = F(n-1) + F(n-2) -- each level integrates two below
    - The 1/phi^4 confounding (exp_37) is the skip penalty

MATHEMATICAL FRAMEWORK:
    For a region R at level k and a descendant D at level k-d:
    - Direct coupling: project D's state into R's eigenbasis
    - Mediated coupling: compose projections through intermediate levels
    - If no-skip holds: direct coupling(k, k-d) should be predictable
      from the product of d consecutive 1-hop couplings

    Attenuation model: coupling(d) = coupling(1) * alpha^(d-1)
    If Fibonacci: alpha should relate to 1/phi

VERIFICATION (4 tests, predict 2/4):
    1. Coupling attenuates monotonically with level distance     (PREDICT PASS)
    2. Attenuation per hop is consistent (CV < 0.3)              (PREDICT PASS)
    3. 2-hop coupling = product of 1-hop couplings (< 20% err)   (PREDICT FAIL)
       -- mediation transforms, not just attenuates
    4. Attenuation rate matches 1/phi within 20%                  (PREDICT FAIL)
       -- may be topology-dependent like coupling strength

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, K_MODES, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_parent_children_data,
)
from exp_08_gradient_coupling import compute_gradient_field
from exp_14_partial_correlation import partial_spearman


PHI = (1 + np.sqrt(5)) / 2


# ── Helpers ──────────────────────────────────────────────────────────

def get_region_indices(labels_by_level, level, rid):
    """Get flat indices for a region at a given level."""
    labels = labels_by_level[level]
    return np.where(labels.ravel() == rid)[0]


def compute_cross_level_coupling(state_flat, ancestor_indices, descendant_indices,
                                  adjacency, grad_flat):
    """
    Compute how much a descendant region contributes to an ancestor's
    spectral identity. This is the "direct" coupling across d hops.

    Returns: (projection_norm, cosine_alignment) or (nan, nan) if degenerate.
    """
    if len(ancestor_indices) < 10 or len(descendant_indices) < 4:
        return np.nan, np.nan

    # Ancestor's eigenbasis
    L_anc, _ = graph_laplacian_subgraph(adjacency, ancestor_indices)
    state_anc = state_flat[ancestor_indices]
    identity = compute_spectral_identity(L_anc, state_anc)
    if 'eigenvectors' not in identity:
        return np.nan, np.nan
    eigvecs = identity['eigenvectors']

    # Map descendant cells to ancestor's local coordinates
    anc_pos_map = {int(idx): pos for pos, idx in enumerate(ancestor_indices)}
    local_pos = []
    for idx in descendant_indices:
        if int(idx) in anc_pos_map:
            local_pos.append(anc_pos_map[int(idx)])
    local_pos = np.array(local_pos)

    if len(local_pos) < 2:
        return np.nan, np.nan

    # Project descendant's state and gradient into ancestor eigenbasis
    state_centered = state_anc - np.mean(state_anc)
    grad_anc = grad_flat[ancestor_indices]
    grad_centered = grad_anc - np.mean(grad_anc)

    desc_state = state_centered[local_pos]
    desc_grad = grad_centered[local_pos]
    desc_eigvec = eigvecs[local_pos, :]

    state_proj = desc_state @ desc_eigvec
    grad_proj = desc_grad @ desc_eigvec

    proj_norm = float(np.linalg.norm(state_proj))

    ns = np.linalg.norm(state_proj)
    ng = np.linalg.norm(grad_proj)
    if ns < 1e-15 or ng < 1e-15:
        return proj_norm, np.nan

    cos_align = float(np.dot(state_proj, grad_proj) / (ns * ng))
    return proj_norm, cos_align


def trace_ancestry(labels_by_level, hierarchy, level0_rid):
    """
    Trace a level-0 region up through all ancestor levels.
    Returns list of (level, region_id) from level 0 to top.
    """
    chain = [(0, level0_rid)]

    for level in range(1, len(labels_by_level)):
        # Find which region at this level contains the majority
        # of the current region's cells
        current_level, current_rid = chain[-1]
        current_indices = get_region_indices(labels_by_level, current_level, current_rid)
        if len(current_indices) == 0:
            break

        parent_labels = labels_by_level[level]
        parent_ids_at_cells = parent_labels.ravel()[current_indices]
        unique, counts = np.unique(parent_ids_at_cells, return_counts=True)
        if len(unique) == 0:
            break
        majority_parent = unique[np.argmax(counts)]
        chain.append((level, int(majority_parent)))

    return chain


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP 38: SCOPED MEDIATION")
    print("Phase 29 -- Confluent Identity")
    print("Does identity reweighting propagate level-by-level?")
    print("=" * 70)

    # Load baseline
    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    C_flat = C.ravel()
    state_flat = C_flat.copy()

    adjacency = build_lattice_adjacency(C)
    grad_C = compute_gradient_field(C)
    grad_flat = grad_C.ravel()

    n_levels = len(labels_by_level)
    print(f"\n  Hierarchy: {n_levels} levels")
    for i, labels in enumerate(labels_by_level):
        n_regions = len(np.unique(labels))
        print(f"    Level {i}: {n_regions} regions")

    # ── Step 1: Trace ancestry chains ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: TRACE ANCESTRY CHAINS")
    print("=" * 60)

    # Get all level-0 regions
    level0_ids = np.unique(labels_by_level[0])
    chains = []
    for rid in level0_ids:
        chain = trace_ancestry(labels_by_level, hierarchy, int(rid))
        if len(chain) >= 3:  # need at least 3 levels for 2-hop test
            chains.append(chain)

    print(f"  Level-0 regions: {len(level0_ids)}")
    print(f"  Chains with >= 3 levels: {len(chains)}")

    if len(chains) < 5:
        print("  INSUFFICIENT CHAINS -- cannot test mediation")
        return

    # Show a few example chains
    for i, chain in enumerate(chains[:5]):
        print(f"    Chain {i}: {' -> '.join(f'L{l}:R{r}' for l, r in chain)}")

    # ── Step 2: Measure coupling at each hop distance ─────────────
    print("\n" + "=" * 60)
    print("STEP 2: COUPLING vs HOP DISTANCE")
    print("=" * 60)

    # For each chain, compute coupling between level-0 region and each ancestor
    by_distance = {}  # distance -> list of (proj_norm, cos_align, size_ratio)

    for chain in chains:
        base_level, base_rid = chain[0]
        base_indices = get_region_indices(labels_by_level, base_level, base_rid)
        base_size = len(base_indices)

        if base_size < 4:
            continue

        for d in range(1, len(chain)):
            anc_level, anc_rid = chain[d]
            anc_indices = get_region_indices(labels_by_level, anc_level, anc_rid)
            anc_size = len(anc_indices)

            proj_norm, cos_align = compute_cross_level_coupling(
                state_flat, anc_indices, base_indices, adjacency, grad_flat
            )

            if not np.isnan(proj_norm):
                size_ratio = base_size / anc_size if anc_size > 0 else 0
                if d not in by_distance:
                    by_distance[d] = []
                by_distance[d].append({
                    'proj_norm': proj_norm,
                    'cos_align': cos_align,
                    'size_ratio': size_ratio,
                    'base_size': base_size,
                    'anc_size': anc_size,
                    'chain_idx': chains.index(chain),
                })

    print(f"\n  {'Dist':>4s} {'N':>4s} {'mean_proj':>10s} {'std_proj':>10s} "
          f"{'mean_cos':>10s} {'mean_szr':>10s}")
    print(f"  {'-'*52}")

    distance_stats = {}
    for d in sorted(by_distance.keys()):
        items = by_distance[d]
        projs = [x['proj_norm'] for x in items]
        coss = [x['cos_align'] for x in items if not np.isnan(x['cos_align'])]
        szrs = [x['size_ratio'] for x in items]

        mean_proj = np.mean(projs)
        std_proj = np.std(projs)
        mean_cos = np.mean(coss) if coss else np.nan
        mean_szr = np.mean(szrs)

        distance_stats[d] = {
            'n': len(items),
            'mean_proj': float(mean_proj),
            'std_proj': float(std_proj),
            'median_proj': float(np.median(projs)),
            'mean_cos': float(mean_cos) if not np.isnan(mean_cos) else None,
            'mean_size_ratio': float(mean_szr),
        }

        print(f"  {d:4d} {len(items):4d} {mean_proj:10.6f} {std_proj:10.6f} "
              f"{mean_cos:10.4f} {mean_szr:10.4f}")

    # ── Step 3: Attenuation analysis ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: ATTENUATION ANALYSIS")
    print("=" * 60)

    distances = sorted(distance_stats.keys())
    mean_projs = [distance_stats[d]['mean_proj'] for d in distances]

    # Monotonic decay?
    is_monotonic = all(mean_projs[i] >= mean_projs[i+1]
                       for i in range(len(mean_projs)-1))
    print(f"  Monotonic decay: {'YES' if is_monotonic else 'NO'}")
    print(f"  Projection norms by distance: {[f'{p:.6f}' for p in mean_projs]}")

    # Per-hop attenuation ratios
    hop_ratios = []
    for i in range(len(distances) - 1):
        if mean_projs[i] > 1e-15:
            ratio = mean_projs[i+1] / mean_projs[i]
            hop_ratios.append(ratio)
            print(f"  Hop {distances[i]}->{distances[i+1]}: "
                  f"ratio = {ratio:.4f} (1/phi = {1/PHI:.4f})")

    if hop_ratios:
        mean_ratio = np.mean(hop_ratios)
        cv_ratio = np.std(hop_ratios) / mean_ratio if mean_ratio > 0 else float('inf')
        print(f"\n  Mean attenuation per hop: {mean_ratio:.4f}")
        print(f"  CV of hop ratios: {cv_ratio:.4f}")
        print(f"  1/phi = {1/PHI:.4f}, delta = {abs(mean_ratio - 1/PHI):.4f} "
              f"({abs(mean_ratio - 1/PHI)/(1/PHI)*100:.1f}%)")
    else:
        mean_ratio = np.nan
        cv_ratio = np.nan

    # ── Step 4: Mediation test ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: MEDIATION TEST (2-hop vs product of 1-hops)")
    print("=" * 60)

    # For each chain with >= 3 levels, compare:
    #   direct(L0 -> L2) vs mediated = coupling(L0->L1) * coupling(L1->L2)
    mediation_errors = []

    for chain in chains:
        if len(chain) < 3:
            continue

        base_level, base_rid = chain[0]
        mid_level, mid_rid = chain[1]
        top_level, top_rid = chain[2]

        base_indices = get_region_indices(labels_by_level, base_level, base_rid)
        mid_indices = get_region_indices(labels_by_level, mid_level, mid_rid)
        top_indices = get_region_indices(labels_by_level, top_level, top_rid)

        # Direct: base -> top (2-hop)
        direct_norm, _ = compute_cross_level_coupling(
            state_flat, top_indices, base_indices, adjacency, grad_flat
        )

        # 1-hop: base -> mid
        hop1_norm, _ = compute_cross_level_coupling(
            state_flat, mid_indices, base_indices, adjacency, grad_flat
        )

        # 1-hop: mid -> top (mid projected into top's basis)
        # Use mid's cells that are in top
        hop2_norm, _ = compute_cross_level_coupling(
            state_flat, top_indices, mid_indices, adjacency, grad_flat
        )

        if (not np.isnan(direct_norm) and not np.isnan(hop1_norm)
                and not np.isnan(hop2_norm) and hop1_norm > 1e-15
                and hop2_norm > 1e-15):

            # Normalize: what fraction of the ancestor's total norm
            # does the descendant contribute?
            # Mediated prediction: product of normalized contributions
            # But norms don't multiply simply -- use log-space
            predicted_2hop = hop1_norm * (hop2_norm / np.linalg.norm(
                state_flat[top_indices] - np.mean(state_flat[top_indices])
            )) if np.linalg.norm(state_flat[top_indices] - np.mean(state_flat[top_indices])) > 1e-15 else np.nan

            if not np.isnan(predicted_2hop) and predicted_2hop > 1e-15:
                error = abs(direct_norm - predicted_2hop) / direct_norm
                mediation_errors.append({
                    'chain_base': (base_level, base_rid),
                    'direct_norm': float(direct_norm),
                    'hop1_norm': float(hop1_norm),
                    'hop2_norm': float(hop2_norm),
                    'predicted': float(predicted_2hop),
                    'relative_error': float(error),
                })

    if mediation_errors:
        errors = [m['relative_error'] for m in mediation_errors]
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        print(f"  N chains tested: {len(mediation_errors)}")
        print(f"  Mean relative error: {mean_err:.4f} ({mean_err*100:.1f}%)")
        print(f"  Median relative error: {median_err:.4f} ({median_err*100:.1f}%)")
        print(f"  Range: [{min(errors):.4f}, {max(errors):.4f}]")
    else:
        mean_err = np.nan
        print("  No valid mediation comparisons")

    # ── Step 5: Size-normalized coupling by distance ──────────────
    print("\n" + "=" * 60)
    print("STEP 5: SIZE-NORMALIZED COUPLING BY DISTANCE")
    print("=" * 60)

    # The projection norm scales with descendant size.
    # Normalize by size_ratio to get per-cell contribution.
    for d in sorted(by_distance.keys()):
        items = by_distance[d]
        normed = [x['proj_norm'] / x['size_ratio'] if x['size_ratio'] > 0
                  else np.nan for x in items]
        normed = [x for x in normed if not np.isnan(x)]
        if normed:
            print(f"  Distance {d}: norm/size_ratio = {np.mean(normed):.4f} "
                  f"+/- {np.std(normed):.4f} (n={len(normed)})")

    # ── Verification ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: Monotonic attenuation
    test1_pass = is_monotonic and len(distances) >= 3
    status1 = "VERIFIED" if test1_pass else "NOT VERIFIED"
    print(f"\n  Test 1: Coupling attenuates monotonically with distance")
    print(f"    Monotonic: {is_monotonic}, levels: {len(distances)}")
    print(f"    -> {status1}")

    # Test 2: Consistent attenuation (CV < 0.3)
    test2_pass = (not np.isnan(cv_ratio)) and cv_ratio < 0.3
    status2 = "VERIFIED" if test2_pass else "NOT VERIFIED"
    print(f"\n  Test 2: Attenuation per hop is consistent (CV < 0.3)")
    print(f"    CV = {cv_ratio:.4f}" if not np.isnan(cv_ratio) else "    CV = N/A")
    print(f"    -> {status2}")

    # Test 3: 2-hop = product of 1-hops (< 20% error)
    test3_pass = (not np.isnan(mean_err)) and mean_err < 0.20
    status3 = "VERIFIED" if test3_pass else "NOT VERIFIED"
    print(f"\n  Test 3: 2-hop coupling predictable from 1-hop product (< 20% err)")
    print(f"    Mean error = {mean_err:.4f}" if not np.isnan(mean_err)
          else "    Mean error = N/A")
    print(f"    -> {status3}")

    # Test 4: Attenuation rate ~ 1/phi
    delta_phi = abs(mean_ratio - 1/PHI) / (1/PHI) if not np.isnan(mean_ratio) else float('inf')
    test4_pass = delta_phi < 0.20
    status4 = "VERIFIED" if test4_pass else "NOT VERIFIED"
    print(f"\n  Test 4: Attenuation rate matches 1/phi within 20%")
    print(f"    Mean ratio = {mean_ratio:.4f}, 1/phi = {1/PHI:.4f}, "
          f"delta = {delta_phi*100:.1f}%" if not np.isnan(mean_ratio)
          else "    Mean ratio = N/A")
    print(f"    -> {status4}")

    n_verified = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    print(f"\n  TOTAL: {n_verified}/4 verified")

    # ── Save ──────────────────────────────────────────────────────
    results = {
        'experiment': 'exp_38_scoped_mediation',
        'phase': 29,
        'n_levels': n_levels,
        'n_chains': len(chains),
        'distance_stats': distance_stats,
        'hop_ratios': [float(r) for r in hop_ratios] if hop_ratios else [],
        'mean_attenuation': float(mean_ratio) if not np.isnan(mean_ratio) else None,
        'cv_attenuation': float(cv_ratio) if not np.isnan(cv_ratio) else None,
        'inv_phi': float(1/PHI),
        'mediation_errors': mediation_errors,
        'mean_mediation_error': float(mean_err) if not np.isnan(mean_err) else None,
        'verification': {
            'test1_monotonic': test1_pass,
            'test2_consistent': test2_pass,
            'test3_mediation': test3_pass,
            'test4_phi_rate': test4_pass,
            'verified_count': n_verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'exp_38_scoped_mediation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
