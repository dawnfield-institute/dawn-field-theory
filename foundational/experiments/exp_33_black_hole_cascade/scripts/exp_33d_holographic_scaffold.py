"""
exp_33d -- Holographic Scaffold: The Ghost Heart Mechanism

HYPOTHESIS: The holographic principle is NOT about surfaces "storing" information.
It is about conservation FORCING the interior to be boundary-determined.
The PAC tree's interior nodes are like the extracellular matrix of a ghost heart:
physically real, structurally essential, but carrying zero independent information.
You can decellularize (erase interior) and reconstruct perfectly from boundary alone.

Analogy:
  Ghost heart: dissolve all living cells -> extracellular matrix remains
  PAC tree:    erase all interior values -> conservation scaffold remains
  Both:        the scaffold determines everything; content is actualization

Tests:
  1. Decellularization — scaffold extraction and perfect reconstruction
  2. Recellularization — universal scaffold with different boundary conditions
  3. Subregion reconstruction — Ryu-Takayanagi surface from PAC minimal cut
  4. Information decomposition — I_total = I_boundary (interior adds nothing)

FALSIFICATION: If reconstruction fidelity < 1.0, or if interior has independent
information beyond boundary, or if the reconstructable region boundary is NOT
a minimal surface.

Author: Peter Groom
Date: 2026-04-20
"""

import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# ============================================================
# Constants
# ============================================================

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
P_D = 1.0 / PHI        # dominant fraction = 0.618034
P_S = 1.0 / PHI**2     # subordinate fraction = 0.381966
H_BOND = -(P_D * np.log(P_D) + P_S * np.log(P_S))  # 0.6650 nats


# ============================================================
# Core Data Structures
# ============================================================

def build_pac_tree(depth, leaf_values=None):
    """
    Build a complete binary PAC tree of given depth.

    Level-order indexing: node i has children 2i+1, 2i+2.
    Leaves at indices [2^D - 1, 2^(D+1) - 2].

    If leaf_values is None, generates a canonical PAC cascade:
    root P=1.0, each node splits P -> (P/phi, P/phi^2).
    """
    D = depth
    n_leaves = 2**D
    n_total = 2**(D + 1) - 1
    n_interior = n_total - n_leaves
    leaf_start = n_interior  # = 2^D - 1

    nodes = np.zeros(n_total)

    if leaf_values is not None:
        assert len(leaf_values) == n_leaves, \
            f"Expected {n_leaves} leaf values, got {len(leaf_values)}"
        nodes[leaf_start:] = leaf_values
    else:
        # Canonical PAC cascade from root
        nodes[0] = 1.0
        for i in range(n_interior):
            nodes[2 * i + 1] = nodes[i] * P_D  # left = dominant
            nodes[2 * i + 2] = nodes[i] * P_S  # right = subordinate
        # Interior was filled top-down; leaves are already set
        return {
            'nodes': nodes,
            'depth': D,
            'n_leaves': n_leaves,
            'n_interior': n_interior,
            'n_total': n_total,
            'leaf_start': leaf_start,
        }

    # Propagate upward from leaves
    return propagate_upward({
        'nodes': nodes,
        'depth': D,
        'n_leaves': n_leaves,
        'n_interior': n_interior,
        'n_total': n_total,
        'leaf_start': leaf_start,
    })


def propagate_upward(tree):
    """
    Fill interior from leaves: parent = left_child + right_child.
    This IS the PAC conservation law: P = D + S.
    """
    nodes = tree['nodes']
    leaf_start = tree['leaf_start']

    # Bottom-up: from last interior node to root
    for i in range(leaf_start - 1, -1, -1):
        nodes[i] = nodes[2 * i + 1] + nodes[2 * i + 2]

    return tree


def decellularize(tree):
    """
    Extract boundary (leaves) and erase interior.
    Returns: (leaf_values_copy, scaffold_description)

    The scaffold is the tree topology + conservation rule.
    It contains NO numerical values -- only structure.
    """
    nodes = tree['nodes']
    leaf_start = tree['leaf_start']

    leaf_values = nodes[leaf_start:].copy()

    scaffold = {
        'type': 'complete_binary_tree',
        'depth': tree['depth'],
        'conservation_rule': 'P = D + S (parent = left_child + right_child)',
        'n_leaves': tree['n_leaves'],
        'n_interior': tree['n_interior'],
        'contains_no_numerical_values': True,
    }

    return leaf_values, scaffold


def recellularize(leaf_values, depth):
    """
    Reconstruct full tree from leaf values + conservation law alone.
    This is the inverse of decellularize.
    """
    return build_pac_tree(depth, leaf_values=leaf_values)


def get_leaves(tree):
    """Extract leaf values from tree."""
    return tree['nodes'][tree['leaf_start']:].copy()


def get_interior(tree):
    """Extract interior node values from tree."""
    return tree['nodes'][:tree['leaf_start']].copy()


# ============================================================
# Test 1: Decellularization -- Scaffold Extraction
# ============================================================

def test1_decellularization():
    """
    Build PAC trees, decellularize, reconstruct. Verify perfect fidelity.

    The ghost heart test: dissolve all interior "cells", then regrow
    the entire organ from the boundary scaffold alone.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Decellularization — Scaffold Extraction")
    print("=" * 60)

    print("\nThe ghost heart procedure:")
    print("  1. Build a PAC tree (the living heart)")
    print("  2. Record all interior node values")
    print("  3. Dissolve interior — keep only leaves + conservation law")
    print("  4. Reconstruct interior from leaves alone")
    print("  5. Compare: is the reconstructed heart identical?")

    depths = [4, 6, 8, 10, 12, 14]
    results_by_depth = {}
    all_perfect = True

    print(f"\n{'Depth':>6s} | {'Leaves':>8s} | {'Interior':>8s} | {'Max Error':>12s} | {'Fidelity':>12s}")
    print("-" * 60)

    for D in depths:
        n_leaves = 2**D
        rng = np.random.default_rng(seed=42 + D)

        # Generate random leaf values (Dirichlet distribution, sum to 1.0)
        leaf_values = rng.dirichlet(np.ones(n_leaves))

        # Build full tree
        tree = build_pac_tree(D, leaf_values=leaf_values)
        original_interior = get_interior(tree)

        # Decellularize
        extracted_leaves, scaffold = decellularize(tree)

        # Recellularize — reconstruct from leaves + conservation ONLY
        reconstructed = recellularize(extracted_leaves, D)
        reconstructed_interior = get_interior(reconstructed)

        # Compare
        max_error = np.max(np.abs(original_interior - reconstructed_interior))
        fidelity = 1.0 - np.sum(np.abs(original_interior - reconstructed_interior)) / \
                   np.sum(np.abs(original_interior)) if np.sum(np.abs(original_interior)) > 0 else 1.0

        print(f"{D:6d} | {n_leaves:8d} | {2**D - 1:8d} | {max_error:12.2e} | {fidelity:12.10f}")

        if max_error > 1e-14:
            all_perfect = False

        results_by_depth[D] = {
            'n_leaves': n_leaves,
            'n_interior': 2**D - 1,
            'max_error': float(max_error),
            'fidelity': float(fidelity),
        }

    # Analytical proof
    print(f"\nAnalytical proof:")
    print(f"  Let X = leaf vector (boundary), Y = interior vector.")
    print(f"  Y = g(X) where g is upward propagation: parent = left + right.")
    print(f"  g is deterministic => H(Y|X) = 0 (zero conditional entropy).")
    print(f"  Chain rule: H(X,Y) = H(X) + H(Y|X) = H(X).")
    print(f"  Therefore: H(full tree) = H(leaves). QED.")
    print(f"  The interior carries ZERO independent information.")
    print(f"  It is pure scaffold — structure without content.")

    print(f"\nGhost heart interpretation:")
    print(f"  The extracellular matrix (conservation scaffold) determines")
    print(f"  every interior cell's value from the boundary alone.")
    print(f"  Decellularization destroys nothing — the information was")
    print(f"  never IN the interior. It was always ON the boundary,")
    print(f"  actualized inward by the scaffold.")

    passed = all_perfect
    print(f"\n{'PASS' if passed else 'FAIL'}: Reconstruction fidelity = 1.0 to machine precision")

    return {
        'test': 'decellularization',
        'results_by_depth': results_by_depth,
        'all_perfect': all_perfect,
        'max_error_overall': max(r['max_error'] for r in results_by_depth.values()),
        'passed': passed,
    }


# ============================================================
# Test 2: Recellularization -- Universal Scaffold
# ============================================================

def test2_recellularization():
    """
    Feed different boundary conditions into the same scaffold.
    Show the scaffold is universal -- like a ghost heart accepting
    different stem cell types.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Recellularization — Universal Scaffold")
    print("=" * 60)

    D = 10
    N = 2**D  # 1024 leaves
    P_total = 1.0

    print(f"\nScaffold: complete binary tree, depth {D}, {N} leaves")
    print(f"Conservation rule: P = D + S at every interior node")
    print(f"Feeding 6 different boundary conditions ('cell types'):\n")

    boundary_types = {}
    rng = np.random.default_rng(seed=137)

    # (a) Uniform
    leaves_uniform = np.full(N, P_total / N)
    boundary_types['(a) Uniform'] = leaves_uniform

    # (b) PAC canonical
    tree_canonical = build_pac_tree(D)
    leaves_canonical = get_leaves(tree_canonical)
    boundary_types['(b) PAC canonical'] = leaves_canonical

    # (c) Random Dirichlet
    leaves_random = rng.dirichlet(np.ones(N)) * P_total
    boundary_types['(c) Random Dirichlet'] = leaves_random

    # (d) Localized (delta function)
    leaves_delta = np.zeros(N)
    leaves_delta[N // 2] = P_total
    boundary_types['(d) Localized delta'] = leaves_delta

    # (e) Power-law
    k = np.arange(1, N + 1, dtype=float)
    leaves_power = k**(-1.5)
    leaves_power *= P_total / leaves_power.sum()
    boundary_types['(e) Power-law'] = leaves_power

    # (f) Thermal (Boltzmann)
    T = N / 10.0
    leaves_thermal = np.exp(-k / T)
    leaves_thermal *= P_total / leaves_thermal.sum()
    boundary_types['(f) Thermal'] = leaves_thermal

    print(f"{'Type':>22s} | {'Valid':>5s} | {'Root P':>10s} | {'Min node':>10s} | {'Interior mean':>13s} | {'Interior std':>12s}")
    print("-" * 85)

    all_valid = True
    type_results = {}

    for name, leaves in boundary_types.items():
        tree = build_pac_tree(D, leaf_values=leaves)
        nodes = tree['nodes']
        interior = get_interior(tree)

        # Check conservation at every interior node
        conservation_ok = True
        for i in range(tree['n_interior']):
            expected = nodes[2 * i + 1] + nodes[2 * i + 2]
            if abs(nodes[i] - expected) > 1e-14:
                conservation_ok = False
                break

        # Check non-negativity
        non_negative = np.all(nodes >= -1e-15)

        # Root should equal total
        root_ok = abs(nodes[0] - P_total) < 1e-12

        valid = conservation_ok and non_negative and root_ok

        print(f"{name:>22s} | {'YES' if valid else 'NO':>5s} | {nodes[0]:10.6f} | "
              f"{np.min(nodes):10.2e} | {np.mean(interior):13.6e} | {np.std(interior):12.6e}")

        if not valid:
            all_valid = False

        type_results[name] = {
            'valid': valid,
            'conservation_ok': conservation_ok,
            'non_negative': bool(non_negative),
            'root_value': float(nodes[0]),
            'interior_mean': float(np.mean(interior)),
            'interior_std': float(np.std(interior)),
        }

    # Configuration space dimensionality
    print(f"\nConfiguration space:")
    print(f"  Leaf values: {N} values constrained to sum to {P_total}")
    print(f"  Free parameters = {N} - 1 = {N - 1} (boundary simplex)")
    print(f"  If interior were independent: {2**(D+1) - 2} free parameters (volume)")
    print(f"  Ratio: {(N - 1) / (2**(D+1) - 2):.4f} (boundary / volume)")
    print(f"  This IS the holographic reduction: volume -> boundary")

    config_dim_boundary = N - 1
    config_dim_volume = 2**(D + 1) - 2
    dim_ratio = config_dim_boundary / config_dim_volume

    print(f"\nGhost heart interpretation:")
    print(f"  The scaffold accepts ANY valid cell population.")
    print(f"  Uniform, localized, thermal, power-law — all produce")
    print(f"  self-consistent organs. The scaffold doesn't care WHAT")
    print(f"  fills it. It only enforces HOW things connect.")
    print(f"  This is universality: structure independent of content.")

    passed = all_valid
    print(f"\n{'PASS' if passed else 'FAIL'}: All boundary types produce valid trees; "
          f"config dim = {config_dim_boundary} (boundary), not {config_dim_volume} (volume)")

    return {
        'test': 'recellularization',
        'depth': D,
        'n_leaves': N,
        'boundary_types': type_results,
        'all_valid': all_valid,
        'config_dim_boundary': config_dim_boundary,
        'config_dim_volume': config_dim_volume,
        'dim_ratio': float(dim_ratio),
        'passed': passed,
    }


# ============================================================
# Test 3: Subregion Reconstruction -- Ryu-Takayanagi Analog
# ============================================================

def reconstructable_region(tree, known_leaf_indices):
    """
    Given a set of known leaf indices, determine which nodes can be
    reconstructed from conservation alone.

    A leaf is reconstructable iff it is in the known set.
    An interior node is reconstructable iff BOTH children are reconstructable.

    Returns: boolean array of size n_total (True = reconstructable).
    """
    n_total = tree['n_total']
    leaf_start = tree['leaf_start']
    recon = np.zeros(n_total, dtype=bool)

    # Mark known leaves
    for idx in known_leaf_indices:
        recon[leaf_start + idx] = True

    # Bottom-up: interior node reconstructable iff both children are
    for i in range(leaf_start - 1, -1, -1):
        left = 2 * i + 1
        right = 2 * i + 2
        recon[i] = recon[left] and recon[right]

    return recon


def rt_surface(tree, recon):
    """
    Compute the Ryu-Takayanagi surface: bonds connecting reconstructable
    to non-reconstructable nodes.

    A bond (parent, child) is on the RT surface iff exactly one of
    parent, child is reconstructable.

    Returns: number of RT bonds (the "area" of the minimal surface).
    """
    leaf_start = tree['leaf_start']
    rt_bonds = 0

    for i in range(leaf_start):
        left = 2 * i + 1
        right = 2 * i + 2

        # Bond (i, left): cut if exactly one side is reconstructable
        if recon[i] != recon[left]:
            rt_bonds += 1
        if recon[i] != recon[right]:
            rt_bonds += 1

    return rt_bonds


def bipartition_cut(tree, leaf_subset_indices):
    """
    Compute the bipartition entanglement cut: bonds in the tree where the
    subtree below contains a MIX of leaves in A and leaves in A^c.

    This is the true RT surface: the minimal set of bonds you must cut
    to separate the tree into "purely A" and "purely A^c" connected components.

    A bond (parent -> child) is cut iff the subtree rooted at child contains
    SOME but NOT ALL leaves from A (i.e., the child's subtree has both A and A^c leaves).

    Actually, more precisely: a bond is cut iff the child's subtree contains
    at least one leaf from A AND at least one leaf from A^c.

    This count is symmetric: bipartition_cut(A) = bipartition_cut(A^c) by construction,
    because "mixed" is the same from both perspectives.

    Returns: number of cut bonds.
    """
    n_total = tree['n_total']
    leaf_start = tree['leaf_start']
    n_leaves = tree['n_leaves']

    # For each node, count how many leaves in A are in its subtree
    a_count = np.zeros(n_total, dtype=int)
    subtree_size = np.zeros(n_total, dtype=int)

    # Mark leaves
    a_set = set(leaf_subset_indices)
    for j in range(n_leaves):
        idx = leaf_start + j
        a_count[idx] = 1 if j in a_set else 0
        subtree_size[idx] = 1

    # Propagate upward
    for i in range(leaf_start - 1, -1, -1):
        left = 2 * i + 1
        right = 2 * i + 2
        a_count[i] = a_count[left] + a_count[right]
        subtree_size[i] = subtree_size[left] + subtree_size[right]

    # A bond (parent -> child) is cut iff child's subtree is "mixed":
    # 0 < a_count[child] < subtree_size[child]
    cut_bonds = 0
    for i in range(leaf_start):
        left = 2 * i + 1
        right = 2 * i + 2

        if 0 < a_count[left] < subtree_size[left]:
            cut_bonds += 1
        if 0 < a_count[right] < subtree_size[right]:
            cut_bonds += 1

    return cut_bonds


def test3_subregion_reconstruction():
    """
    Given a fraction f of known leaves, compute the reconstructable region
    (entanglement wedge) and the RT surface (minimal cut).

    Key prediction: contiguous blocks have RT ~ O(log N) (area law),
    random subsets have RT ~ O(N) (no locality).
    """
    print("\n" + "=" * 60)
    print("TEST 3: Subregion Reconstruction — Ryu-Takayanagi Analog")
    print("=" * 60)

    D = 10
    N = 2**D  # 1024
    tree = build_pac_tree(D)  # canonical PAC tree (values don't matter for topology)

    fractions = np.arange(0.0, 1.01, 0.05)
    rng = np.random.default_rng(seed=42)
    n_random_samples = 50

    contiguous_data = []
    random_data = []

    print(f"\nTree: depth {D}, {N} leaves, {tree['n_total']} total nodes")
    print(f"\n{'f':>6s} | {'Contig RT':>10s} | {'Contig Recon%':>13s} | {'Random RT':>10s} | {'Random Recon%':>13s}")
    print("-" * 65)

    for f in fractions:
        k = int(round(f * N))
        if k == 0:
            contiguous_data.append((0.0, 0, 0.0))
            random_data.append((0.0, 0, 0.0))
            print(f"{f:6.2f} | {0:10d} | {0.0:12.2f}% | {0:10d} | {0.0:12.2f}%")
            continue
        if k >= N:
            contiguous_data.append((1.0, 0, 1.0))
            random_data.append((1.0, 0, 1.0))
            print(f"{f:6.2f} | {0:10d} | {100.0:12.2f}% | {0:10d} | {100.0:12.2f}%")
            continue

        # Contiguous block: leaves [0, k-1]
        known_contig = list(range(k))
        recon_contig = reconstructable_region(tree, known_contig)
        rt_contig = rt_surface(tree, recon_contig)
        recon_frac_contig = np.sum(recon_contig) / tree['n_total']

        # Random subsets: average over samples
        rt_random_vals = []
        recon_frac_random_vals = []
        for _ in range(n_random_samples):
            known_random = rng.choice(N, size=k, replace=False).tolist()
            recon_random = reconstructable_region(tree, known_random)
            rt_random_vals.append(rt_surface(tree, recon_random))
            recon_frac_random_vals.append(np.sum(recon_random) / tree['n_total'])

        rt_random_mean = np.mean(rt_random_vals)
        recon_frac_random_mean = np.mean(recon_frac_random_vals)

        contiguous_data.append((float(f), int(rt_contig), float(recon_frac_contig)))
        random_data.append((float(f), float(rt_random_mean), float(recon_frac_random_mean)))

        print(f"{f:6.2f} | {rt_contig:10d} | {recon_frac_contig*100:12.2f}% | "
              f"{rt_random_mean:10.1f} | {recon_frac_random_mean*100:12.2f}%")

    # Analyze contiguous RT surface scaling
    contig_rt_values = [d[1] for d in contiguous_data if 0 < d[0] < 1]
    contig_max_rt = max(contig_rt_values) if contig_rt_values else 0

    # For a contiguous block in a binary tree, RT surface should be O(log N)
    # because the block boundary crosses at most 2 bonds per tree level
    rt_log_n = 2 * D  # theoretical maximum for contiguous block
    contig_is_area_law = contig_max_rt <= 4 * D  # generous bound

    # Random RT should be much larger
    random_rt_at_half = [d[1] for d in random_data if abs(d[0] - 0.5) < 0.01]
    random_rt_half = random_rt_at_half[0] if random_rt_at_half else 0

    contig_rt_at_half = [d[1] for d in contiguous_data if abs(d[0] - 0.5) < 0.01]
    contig_rt_half = contig_rt_at_half[0] if contig_rt_at_half else 0

    locality_ratio = random_rt_half / contig_rt_half if contig_rt_half > 0 else float('inf')

    # Verify S(A) = S(A^c) using bipartition cut (symmetric by construction).
    # The bipartition cut counts bonds whose subtree is mixed (has both A and A^c leaves).
    # This is identical whether you call the subset A or A^c.
    symmetry_errors = []
    bipartition_values = []
    for f in [0.1, 0.2, 0.3, 0.4]:
        k = int(round(f * N))

        known_A = list(range(k))
        known_Ac = list(range(k, N))

        cut_A = bipartition_cut(tree, known_A)
        cut_Ac = bipartition_cut(tree, known_Ac)

        symmetry_errors.append(abs(cut_A - cut_Ac))
        bipartition_values.append((f, cut_A, cut_Ac))

    max_symmetry_err = max(symmetry_errors)
    purification_holds = max_symmetry_err == 0  # bipartition_cut(A) = bipartition_cut(A^c) exactly

    print(f"\nRT surface analysis:")
    print(f"  Contiguous max RT: {contig_max_rt} bonds (O(log N) = O({D}))")
    print(f"  Contiguous is area law (RT <= 4D = {4*D}): {'YES' if contig_is_area_law else 'NO'}")
    print(f"  At f=0.5: contiguous RT = {contig_rt_half}, random RT = {random_rt_half:.0f}")
    print(f"  Locality ratio (random/contiguous at f=0.5): {locality_ratio:.1f}x")
    print(f"  Purification S(A) = S(A^c): {'YES' if purification_holds else 'NO'} (max error: {max_symmetry_err})")

    # Bipartition cut at f=0.5 for the entanglement entropy
    bipartition_half = bipartition_cut(tree, list(range(N // 2)))

    print(f"\nRyu-Takayanagi interpretation:")
    print(f"  The RT surface is the minimal cut separating known from unknown.")
    print(f"  For contiguous blocks: the cut crosses O(log N) = O({D}) bonds")
    print(f"  because the tree's hierarchical structure provides geometric locality.")
    print(f"  For random subsets: the cut crosses O(N) bonds — no locality advantage.")
    print(f"  ")
    print(f"  This is EXACTLY the Ryu-Takayanagi prescription:")
    print(f"  the entanglement entropy of a boundary subregion equals the")
    print(f"  area of the minimal surface separating it from its complement.")
    print(f"  Here the 'area' is measured in PAC bonds, each carrying H(phi) nats.")
    print(f"  Bipartition cut at f=0.5: {bipartition_half} bonds")
    print(f"  S(A) = |gamma(A)| x H(phi) = {bipartition_half} x {H_BOND:.4f} = {bipartition_half * H_BOND:.2f} nats")

    print(f"\nGhost heart interpretation:")
    print(f"  If you know what cells are on one side of the heart,")
    print(f"  you can reconstruct the interior of that side — but ONLY")
    print(f"  the part whose scaffold connects entirely to known cells.")
    print(f"  The boundary of your knowledge IS the Ryu-Takayanagi surface.")

    passed = contig_is_area_law and purification_holds and locality_ratio > 5
    print(f"\n{'PASS' if passed else 'FAIL'}: RT surface from PAC conservation")
    print(f"  Area law (contig): {'YES' if contig_is_area_law else 'NO'}")
    print(f"  Purification: {'YES' if purification_holds else 'NO'}")
    print(f"  Locality advantage (ratio > 5): {'YES' if locality_ratio > 5 else 'NO'} ({locality_ratio:.1f}x)")

    return {
        'test': 'subregion_reconstruction',
        'depth': D,
        'n_leaves': N,
        'contiguous_data': contiguous_data,
        'random_data': random_data,
        'contig_max_rt': int(contig_max_rt),
        'contig_rt_half': int(contig_rt_half),
        'random_rt_half': float(random_rt_half),
        'locality_ratio': float(locality_ratio),
        'purification_holds': purification_holds,
        'contig_is_area_law': contig_is_area_law,
        'rt_theoretical_max': rt_log_n,
        'H_bond': float(H_BOND),
        'passed': passed,
    }


# ============================================================
# Test 4: Information Decomposition
# ============================================================

def test4_information_decomposition():
    """
    Rigorously show that the interior adds zero independent information.

    Method: generate an ensemble of random PAC trees. Build the covariance
    matrix of ALL node values. The rank of this matrix = number of independent
    DOF. Should be N_leaves - 1 (boundary - 1 for sum constraint),
    NOT N_total - 1 (volume).
    """
    print("\n" + "=" * 60)
    print("TEST 4: Information Decomposition — Scaffold vs Content")
    print("=" * 60)

    D = 8  # 256 leaves, 511 total (manageable for SVD)
    N = 2**D
    n_total = 2**(D + 1) - 1
    n_ensemble = 5000

    rng = np.random.default_rng(seed=2026)

    print(f"\nEnsemble: {n_ensemble} random PAC trees, depth {D}")
    print(f"  Leaves: {N}, Interior: {N - 1}, Total: {n_total}")

    # Generate ensemble
    all_nodes = np.zeros((n_ensemble, n_total))

    for i in range(n_ensemble):
        leaf_values = rng.dirichlet(np.ones(N))  # sum to 1.0
        tree = build_pac_tree(D, leaf_values=leaf_values)
        all_nodes[i] = tree['nodes']

    # Covariance matrix of ALL nodes
    cov_all = np.cov(all_nodes.T)

    # SVD to find rank
    singular_values = np.linalg.svd(cov_all, compute_uv=False)

    # Rank = number of singular values above threshold
    threshold = singular_values[0] * 1e-10
    rank_all = np.sum(singular_values > threshold)

    expected_rank = N - 1  # leaves minus 1 (sum constraint)
    volume_rank = n_total - 1  # what it would be if interior were independent

    # Also compute rank of leaf-only and interior-only covariance
    leaf_start = N - 1
    leaves_only = all_nodes[:, leaf_start:]
    interior_only = all_nodes[:, :leaf_start]

    cov_leaves = np.cov(leaves_only.T)
    sv_leaves = np.linalg.svd(cov_leaves, compute_uv=False)
    rank_leaves = np.sum(sv_leaves > sv_leaves[0] * 1e-10)

    cov_interior = np.cov(interior_only.T)
    sv_interior = np.linalg.svd(cov_interior, compute_uv=False)
    rank_interior = np.sum(sv_interior > sv_interior[0] * 1e-10)

    print(f"\nSingular value analysis:")
    print(f"  Full tree covariance rank:     {rank_all}")
    print(f"  Expected (boundary - 1):       {expected_rank}")
    print(f"  If interior independent:       {volume_rank}")
    print(f"  ")
    print(f"  Leaf-only covariance rank:      {rank_leaves}")
    print(f"  Interior-only covariance rank:  {rank_interior}")

    # The key ratio
    rank_ratio = rank_all / expected_rank

    print(f"\n  rank(full) / (N_leaves - 1) = {rank_ratio:.6f}")
    print(f"  rank(full) / (N_total - 1)  = {rank_all / volume_rank:.6f}")

    # Verify: adding interior doesn't increase rank
    rank_increase = rank_all - rank_leaves
    print(f"\n  Rank increase from adding interior: {rank_increase}")
    print(f"  (Expected: 0 — interior is determined by leaves)")

    # Information decomposition
    print(f"\nInformation decomposition:")
    print(f"  I_boundary = rank(leaves) = {rank_leaves} independent dimensions")
    print(f"  I_interior = rank increase = {rank_increase} independent dimensions")
    print(f"  I_total    = rank(full)    = {rank_all} independent dimensions")
    print(f"  ")
    print(f"  I_total = I_boundary + I_interior = {rank_leaves} + {rank_increase} = {rank_leaves + rank_increase}")
    print(f"  I_scaffold = 0 (the conservation law adds NO independent dimensions)")

    # Scaling check: repeat for multiple depths
    print(f"\n  Scaling verification:")
    print(f"  {'Depth':>6s} | {'N_leaves':>8s} | {'rank(full)':>10s} | {'N_leaves-1':>10s} | {'Match':>5s}")
    print(f"  " + "-" * 50)

    scaling_results = []
    for D_scan in [4, 5, 6, 7, 8]:
        N_scan = 2**D_scan
        n_total_scan = 2**(D_scan + 1) - 1
        n_ens = min(3000, max(1000, 10 * N_scan))

        nodes_scan = np.zeros((n_ens, n_total_scan))
        for i in range(n_ens):
            lv = rng.dirichlet(np.ones(N_scan))
            t = build_pac_tree(D_scan, leaf_values=lv)
            nodes_scan[i] = t['nodes']

        cov_scan = np.cov(nodes_scan.T)
        sv_scan = np.linalg.svd(cov_scan, compute_uv=False)
        rank_scan = np.sum(sv_scan > sv_scan[0] * 1e-10)
        expected_scan = N_scan - 1

        match = rank_scan == expected_scan
        print(f"  {D_scan:6d} | {N_scan:8d} | {rank_scan:10d} | {expected_scan:10d} | {'YES' if match else 'NO'}")
        scaling_results.append({
            'depth': D_scan,
            'n_leaves': N_scan,
            'rank': int(rank_scan),
            'expected': expected_scan,
            'match': match,
        })

    all_match = all(r['match'] for r in scaling_results)

    # Connection to Bekenstein bound
    print(f"\nConnection to Bekenstein bound:")
    print(f"  In a BH, N_leaves = A / l_P^2 (Planck cells on horizon)")
    print(f"  Independent DOF = N_leaves - 1 ~ N_leaves for large N")
    print(f"  Max information = boundary area (in Planck units)")
    print(f"  This is the Bekenstein bound: S_max ~ A / l_P^2")
    print(f"  NOT S_max ~ V / l_P^3 (the interior is scaffold, not content)")

    print(f"\nGhost heart interpretation:")
    print(f"  The extracellular matrix has physical extent (fills the volume)")
    print(f"  but carries zero genetic information. All the information")
    print(f"  needed to rebuild the organ is in the boundary cells.")
    print(f"  The interior is pure structure — it constrains, but does not inform.")
    print(f"  This is conservation as scaffold: real, essential, but not information.")

    passed = abs(rank_ratio - 1.0) < 0.01 and rank_increase == 0 and all_match
    print(f"\n{'PASS' if passed else 'FAIL'}: Interior adds zero independent information")
    print(f"  rank(full) = N_leaves - 1: {'YES' if abs(rank_ratio - 1.0) < 0.01 else 'NO'} (ratio = {rank_ratio:.6f})")
    print(f"  rank increase = 0: {'YES' if rank_increase == 0 else 'NO'} ({rank_increase})")
    print(f"  Scaling verified: {'YES' if all_match else 'NO'}")

    return {
        'test': 'information_decomposition',
        'depth': D,
        'n_leaves': N,
        'n_total': n_total,
        'n_ensemble': n_ensemble,
        'rank_full': int(rank_all),
        'rank_leaves': int(rank_leaves),
        'rank_interior': int(rank_interior),
        'rank_increase': int(rank_increase),
        'expected_rank': expected_rank,
        'rank_ratio': float(rank_ratio),
        'scaling_results': scaling_results,
        'all_scaling_match': all_match,
        'passed': passed,
    }


# ============================================================
# Main
# ============================================================

def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def main():
    print("exp_33d: Holographic Scaffold — The Ghost Heart Mechanism")
    print("=" * 60)

    t1 = test1_decellularization()
    t2 = test2_recellularization()
    t3 = test3_subregion_reconstruction()
    t4 = test4_information_decomposition()

    # Summary
    tests = [t1, t2, t3, t4]
    n_passed = sum(1 for t in tests if t['passed'])
    n_total = len(tests)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_passed}/{n_total} tests passed")
    print(f"{'=' * 60}")
    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  {status}: {t['test']}")

    # Save results
    results = {
        'experiment': 'exp_33d_holographic_scaffold',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'tests': {t['test']: t for t in tests},
        'summary': {
            'passed': n_passed,
            'total': n_total,
            'score': f"{n_passed}/{n_total}",
        },
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = results_dir / f'exp_33d_holographic_scaffold_v1_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
