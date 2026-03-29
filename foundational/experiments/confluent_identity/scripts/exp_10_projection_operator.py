"""
exp_10_projection_operator.py -- Confluent Identity Phase 4

PURPOSE:
    Construct CI as a proper Hilbert space projection operator Pi_harm and
    verify its operator-algebraic properties. This replaces the informal
    CI(R) 4-tuple with a genuine linear operator.

DEFINITIONS:
    Pi_harm: L^2(V_R) -> ker(L_R)

    For connected region R with n cells:
        Pi = (1/n) * J_n    where J_n is the n x n all-ones matrix

    For disconnected region with components {V_1, ..., V_k}:
        Pi = block_diag(Pi_1, ..., Pi_k)

VERIFICATION (5 operator properties):
    1. Idempotency: Pi^2 = Pi
    2. Self-adjointness: Pi = Pi^T
    3. Orthogonality: <Pi(f), (I-Pi)(g)> = 0
    4. Composition: Pi_child . Pi_parent != Pi_parent (hierarchy matters)
    5. Trace: tr(Pi) = dim(ker(L)) = # connected components

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, get_parent_children_data,
)


@dataclass
class ProjectionOperator:
    """
    The Confluent Identity Operator as a proper Hilbert space projection.

    CI: L^2(V_R) -> ker(L_R)
    CI(f) = Pi_harm @ f

    This is the CORRECT formalization. The Fiedler value, spectral entropy,
    and coefficient vector are DERIVED quantities from (L_R, Pi), not part
    of the operator definition.
    """
    Pi: np.ndarray                 # Projection matrix (n x n)
    n_cells: int                   # Region size
    n_components: int              # Connected components (rank of Pi)
    component_sizes: List[int]     # Size of each component

    # Derived quantities (computed from L and Pi, not stored in operator)
    fiedler_value: float = 0.0
    spectral_entropy: float = 0.0

    def apply(self, f: np.ndarray) -> np.ndarray:
        """CI(f) = Pi @ f -- project onto harmonic space."""
        return self.Pi @ f

    def residual(self, f: np.ndarray) -> np.ndarray:
        """(I - Pi)(f) -- the gradient/exact component."""
        return f - self.Pi @ f


def build_projection_operator(adjacency, indices) -> ProjectionOperator:
    """
    Construct Pi_harm for a region defined by indices.

    Detects connected components and builds block-diagonal projection.
    """
    n = len(indices)

    # Get subgraph adjacency
    W_sub = adjacency[np.ix_(indices, indices)]

    # Detect connected components
    n_comp, comp_labels = connected_components(W_sub, directed=False)

    # Build projection matrix
    Pi = np.zeros((n, n))
    component_sizes = []

    for c in range(n_comp):
        mask = comp_labels == c
        nc = int(mask.sum())
        component_sizes.append(nc)

        if nc > 0:
            # Pi_c = (1/nc) * J_nc
            # In the full matrix: Pi[mask, mask] = 1/nc
            idx_c = np.where(mask)[0]
            for i in idx_c:
                for j in idx_c:
                    Pi[i, j] = 1.0 / nc

    # Derived quantities
    L, _ = graph_laplacian_subgraph(adjacency, indices)
    state = np.zeros(n)  # dummy for eigencompute
    identity = compute_spectral_identity(L, state)

    return ProjectionOperator(
        Pi=Pi,
        n_cells=n,
        n_components=n_comp,
        component_sizes=component_sizes,
        fiedler_value=identity['fiedler_value'],
        spectral_entropy=identity['spectral_entropy'],
    )


def verify_idempotency(Pi, tol=1e-12):
    """Pi^2 = Pi."""
    Pi2 = Pi @ Pi
    error = np.linalg.norm(Pi2 - Pi, 'fro')
    return error, error < tol


def verify_self_adjointness(Pi, tol=1e-12):
    """Pi = Pi^T."""
    error = np.linalg.norm(Pi - Pi.T, 'fro')
    return error, error < tol


def verify_orthogonality(Pi, n_tests=100, tol=1e-10, seed=42):
    """<Pi(f), (I-Pi)(g)> = 0 for random f, g."""
    rng = np.random.RandomState(seed)
    n = Pi.shape[0]
    I = np.eye(n)
    I_minus_Pi = I - Pi

    max_error = 0.0
    for _ in range(n_tests):
        f = rng.randn(n)
        g = rng.randn(n)
        ip = np.dot(Pi @ f, I_minus_Pi @ g)
        max_error = max(max_error, abs(ip))

    return max_error, max_error < tol


def verify_trace(Pi, expected_rank):
    """tr(Pi) = dim(ker(L)) = # connected components."""
    tr = np.trace(Pi)
    error = abs(tr - expected_rank)
    return float(tr), error, error < 0.5  # integer-valued, so 0.5 tolerance


def verify_composition(Pi_parent, Pi_child, parent_indices, child_indices,
                        tol=1e-10):
    """
    Pi_child . Pi_parent != Pi_parent in general.

    We embed Pi_child into parent's space via restriction/extension.
    """
    n_parent = len(parent_indices)
    parent_pos_map = {int(idx): pos for pos, idx in enumerate(parent_indices)}

    # Map child positions within parent
    child_in_parent = []
    for ci in child_indices:
        ci_int = int(ci)
        if ci_int in parent_pos_map:
            child_in_parent.append(parent_pos_map[ci_int])
    child_in_parent = np.array(child_in_parent)

    if len(child_in_parent) == 0:
        return 0.0, False  # Can't test

    # Restriction operator R: parent space -> child space
    # Extension operator E: child space -> parent space (zero-pad)
    n_child = Pi_child.shape[0]

    # Build the embedded child projection: E @ Pi_child @ R
    # This is an n_parent x n_parent matrix that projects onto child's
    # harmonic space, embedded in parent coordinates
    Pi_child_embedded = np.zeros((n_parent, n_parent))
    for i, pi in enumerate(child_in_parent):
        for j, pj in enumerate(child_in_parent):
            if i < n_child and j < n_child:
                Pi_child_embedded[pi, pj] = Pi_child[i, j]

    # Composition: Pi_child_embedded @ Pi_parent
    composed = Pi_child_embedded @ Pi_parent

    # Compare to Pi_parent
    diff = np.linalg.norm(composed - Pi_parent, 'fro')
    parent_norm = np.linalg.norm(Pi_parent, 'fro')
    relative_diff = diff / parent_norm if parent_norm > 1e-15 else 0.0

    # They should differ (hierarchy matters)
    differs = relative_diff > tol

    return float(relative_diff), differs


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 10")
    print("Projection Operator: CI as Pi_harm")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Build projection operators for all regions
    MAX_CELLS = 1500  # dense matrix limit
    operators = {}  # (level, rid) -> ProjectionOperator

    for level in range(len(labels_by_level)):
        labels = labels_by_level[level]
        region_ids = sorted(np.unique(labels).tolist())

        for rid in region_ids:
            indices = get_region_indices(labels_by_level, level, rid)
            if len(indices) < 3 or len(indices) > MAX_CELLS:
                continue

            op = build_projection_operator(adjacency, indices)
            operators[(level, rid)] = (op, indices)

    print(f"\nBuilt {len(operators)} projection operators")

    # ===== VERIFICATION 1: IDEMPOTENCY =====
    print(f"\n{'=' * 70}")
    print("Theorem 8: Idempotency (Pi^2 = Pi)")
    print(f"{'=' * 70}")

    idem_errors = []
    for (level, rid), (op, _) in operators.items():
        err, ok = verify_idempotency(op.Pi)
        idem_errors.append(err)

    max_idem = max(idem_errors) if idem_errors else float('inf')
    idem_pass = max_idem < 1e-12
    print(f"  Max ||Pi^2 - Pi||_F: {max_idem:.2e}")
    print(f"  {'[VERIFIED]' if idem_pass else '[FAILED]'} across {len(idem_errors)} regions")

    # ===== VERIFICATION 2: SELF-ADJOINTNESS =====
    print(f"\n{'=' * 70}")
    print("Theorem 9: Self-Adjointness (Pi = Pi^T)")
    print(f"{'=' * 70}")

    adj_errors = []
    for (level, rid), (op, _) in operators.items():
        err, ok = verify_self_adjointness(op.Pi)
        adj_errors.append(err)

    max_adj = max(adj_errors) if adj_errors else float('inf')
    adj_pass = max_adj < 1e-12
    print(f"  Max ||Pi - Pi^T||_F: {max_adj:.2e}")
    print(f"  {'[VERIFIED]' if adj_pass else '[FAILED]'} across {len(adj_errors)} regions")

    # ===== VERIFICATION 3: ORTHOGONALITY =====
    print(f"\n{'=' * 70}")
    print("Theorem 10: Orthogonality (<Pi(f), (I-Pi)(g)> = 0)")
    print(f"{'=' * 70}")

    orth_errors = []
    # Test on a sample of regions (orthogonality is expensive with 100 tests each)
    sample_keys = list(operators.keys())[:20]
    for key in sample_keys:
        op, _ = operators[key]
        err, ok = verify_orthogonality(op.Pi, n_tests=50)
        orth_errors.append(err)

    max_orth = max(orth_errors) if orth_errors else float('inf')
    orth_pass = max_orth < 1e-10
    print(f"  Max |<Pi(f), (I-Pi)(g)>|: {max_orth:.2e} (over {len(sample_keys)} regions)")
    print(f"  {'[VERIFIED]' if orth_pass else '[FAILED]'}")

    # ===== VERIFICATION 4: COMPOSITION (HIERARCHY MATTERS) =====
    print(f"\n{'=' * 70}")
    print("Theorem 11: Non-Trivial Composition (Pi_child . Pi_parent != Pi_parent)")
    print(f"{'=' * 70}")

    composition_tests = 0
    composition_differs = 0
    composition_diffs = []

    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        parent_key = (level, pid)
        if parent_key not in operators:
            continue

        op_parent, parent_indices = operators[parent_key]

        for child_level, child_id in children:
            child_key = (child_level, child_id)
            if child_key not in operators:
                continue

            op_child, child_indices = operators[child_key]

            rel_diff, differs = verify_composition(
                op_parent.Pi, op_child.Pi,
                parent_indices, child_indices
            )
            composition_tests += 1
            if differs:
                composition_differs += 1
            composition_diffs.append(rel_diff)

    comp_frac = composition_differs / composition_tests if composition_tests > 0 else 0
    comp_pass = comp_frac > 0.5
    print(f"  {composition_differs}/{composition_tests} pairs differ "
          f"({comp_frac:.1%})")
    if composition_diffs:
        print(f"  Mean relative diff: {np.mean(composition_diffs):.6f}")
    print(f"  {'[VERIFIED]' if comp_pass else '[FAILED]'}")

    # ===== VERIFICATION 5: TRACE =====
    print(f"\n{'=' * 70}")
    print("Theorem 12: Trace (tr(Pi) = dim(ker(L)))")
    print(f"{'=' * 70}")

    trace_errors = []
    for (level, rid), (op, _) in operators.items():
        tr, err, ok = verify_trace(op.Pi, op.n_components)
        trace_errors.append(err)

    max_trace_err = max(trace_errors) if trace_errors else float('inf')
    trace_pass = max_trace_err < 0.5
    n_connected = sum(1 for (_, _), (op, _) in operators.items() if op.n_components == 1)
    print(f"  Max |tr(Pi) - n_components|: {max_trace_err:.2e}")
    print(f"  {n_connected}/{len(operators)} regions are connected (tr=1)")
    print(f"  {'[VERIFIED]' if trace_pass else '[FAILED]'}")

    # ===== REDEFINE CI =====
    print(f"\n{'=' * 70}")
    print("Redefined CI Operator")
    print(f"{'=' * 70}")

    # Demonstrate on a sample region
    sample_key = list(operators.keys())[0]
    op, indices = operators[sample_key]
    f = state_flat[indices]

    harmonic = op.apply(f)        # CI(f)
    gradient = op.residual(f)     # (I - CI)(f)
    parseval_err = abs(np.dot(f, f) - np.dot(harmonic, harmonic) - np.dot(gradient, gradient))

    print(f"\n  Sample region {sample_key}: {op.n_cells} cells")
    print(f"    CI(f) = Pi @ f: mean={np.mean(harmonic):.6f}")
    print(f"    ||f||^2 = {np.dot(f, f):.6f}")
    print(f"    ||CI(f)||^2 + ||(I-CI)(f)||^2 = "
          f"{np.dot(harmonic, harmonic):.6f} + {np.dot(gradient, gradient):.6f}")
    print(f"    Parseval error: {parseval_err:.2e}")
    print(f"    Fiedler (derived): {op.fiedler_value:.6f}")
    print(f"    Components: {op.n_components}")

    # ===== SUMMARY =====
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")

    n_verified = sum([idem_pass, adj_pass, orth_pass, comp_pass, trace_pass])
    tests = [
        ("Idempotency (Pi^2 = Pi)", idem_pass, max_idem),
        ("Self-adjointness (Pi = Pi^T)", adj_pass, max_adj),
        ("Orthogonality (<Pi f, (I-Pi)g> = 0)", orth_pass, max_orth),
        ("Non-trivial composition", comp_pass, comp_frac),
        ("Trace (tr = dim ker)", trace_pass, max_trace_err),
    ]

    for name, passed, metric in tests:
        status = "[VERIFIED]" if passed else "[FAILED]"
        print(f"  {status} {name} (metric: {metric:.2e})")

    print(f"\n  OVERALL: {n_verified}/5 operator properties verified")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_10_projection_operator',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'CI as proper Hilbert space projection operator',
        'n_operators': len(operators),
        'verification': {
            'idempotency': {'max_error': float(max_idem), 'verified': bool(idem_pass)},
            'self_adjointness': {'max_error': float(max_adj), 'verified': bool(adj_pass)},
            'orthogonality': {'max_error': float(max_orth), 'n_tested': len(sample_keys),
                              'verified': bool(orth_pass)},
            'composition': {'n_tests': int(composition_tests), 'n_differs': int(composition_differs),
                            'fraction': float(comp_frac), 'verified': bool(comp_pass)},
            'trace': {'max_error': float(max_trace_err), 'n_connected': int(n_connected),
                      'verified': bool(trace_pass)},
            'n_verified': n_verified,
        },
        'sample_region': {
            'key': str(sample_key),
            'n_cells': int(op.n_cells),
            'parseval_error': float(parseval_err),
        },
    }

    output_file = RESULTS_DIR / f'exp_10_projection_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
