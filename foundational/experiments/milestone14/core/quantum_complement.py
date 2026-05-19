"""
quantum_complement.py -- Shared infrastructure for Milestone 14: Quantum Mechanics
as Complement-Indeterminacy.

Extends M13's identity_complement.py with orbit Hilbert space construction,
Born rule, measurement collapse, interference, uncertainty, and entanglement.

The central claim: quantum mechanics IS complement-indeterminacy on the orbit
quotient. States live on L^2(V/Aut(G)), the Born rule emerges from orbit measure,
interference requires SEC complexification, and uncertainty arises from non-commuting
automorphism group operations (D_4's S_3 is the key non-abelian case).

Provides:
- graph_automorphisms, conjugacy_classes: group theory (exp_01, exp_02, exp_07)
- orbit_hilbert_basis, orbit_projector: Hilbert space (exp_01, exp_03)
- permutation_rep_decompose: representation theory (exp_02)
- born_probability, measurement_collapse: measurement (exp_03, exp_04)
- two_path_amplitude: interference (exp_05, exp_06)
- robertson_uncertainty: uncertainty (exp_08)
- product_graph, partial_trace, von_neumann_entropy: entanglement (exp_09)
"""

import sys
import numpy as np
from pathlib import Path
from itertools import permutations

# Import M13 infrastructure (which chains M12 -> M11 -> M10 -> M9 -> M8)
M14_ROOT = Path(__file__).resolve().parent.parent
M13_ROOT = M14_ROOT.parent / "milestone13"
sys.path.insert(0, str(M13_ROOT / "core"))

from identity_complement import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI, LN2,
    # Physical constants
    HBAR, C_LIGHT, K_BOLTZMANN,
    # Fibonacci
    fib, fibonacci_depth_coupling, DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    # ADE infrastructure
    DynkinDiagram, all_ade_diagrams,
    # Complement operations
    complement_spectrum, complement_view, complement_distance,
    complement_transformation, complement_deformation_rate, max_deformation_rate,
    vertex_orbits,
    # Weyl / Lie algebra
    weyl_element_su2, weyl_conjugate, weyl_action_on_algebra,
    complement_derived_generators,
    SU2_GENERATORS, commutator,
    complexify_generators, sl2c_generators, verify_lie_algebra,
    killing_form, killing_form_for_algebra,
    # SEC complexification
    so31_from_sl2c, check_compactness,
    # Basin dynamics
    BasinAttractor,
    # PAC redistribution
    redistribute_on_graph, measure_entropy,
    # Utilities
    save_m13_results, _convert_numpy,
    # Graph utilities
    graph_distance, build_petersen_graph, build_complete_graph,
    generate_random_connected_graph,
)


# ============================================================
# Graph Automorphisms
# ============================================================

def graph_automorphisms(adjacency):
    """
    Compute Aut(G) as a list of permutation matrices.

    Brute-force: check all n! permutations, keeping those that preserve
    the adjacency matrix (P @ A @ P.T == A). Pruned by degree sequence.

    Feasible for ADE graphs up to rank 8 (max 8! = 40320).
    """
    n = adjacency.shape[0]
    A = adjacency.astype(float)

    # Compute degree sequence for pruning
    degrees = np.sum(A > 0, axis=1).astype(int)

    automorphisms = []
    for perm in permutations(range(n)):
        # Prune: permutation must preserve degree sequence
        perm_degrees = degrees[list(perm)]
        if not np.array_equal(perm_degrees, degrees):
            continue

        # Build permutation matrix
        P = np.zeros((n, n))
        for i, j in enumerate(perm):
            P[i, j] = 1.0

        # Check if it preserves adjacency
        if np.allclose(P @ A @ P.T, A):
            automorphisms.append(P)

    return automorphisms


def conjugacy_classes(group_elements):
    """
    Partition a list of group elements (matrices) into conjugacy classes.

    Two elements g, h are conjugate if there exists k such that g = k h k^{-1}.
    """
    n = len(group_elements)
    assigned = [False] * n
    classes = []

    for i in range(n):
        if assigned[i]:
            continue
        current_class = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            # Check if g_i and g_j are conjugate
            for k_elem in group_elements:
                k_inv = np.linalg.inv(k_elem)
                conjugated = k_elem @ group_elements[j] @ k_inv
                if np.allclose(conjugated, group_elements[i]):
                    current_class.append(j)
                    assigned[j] = True
                    break

        classes.append(current_class)

    return classes


# ============================================================
# Orbit Hilbert Space
# ============================================================

def orbit_hilbert_basis(adjacency):
    """
    Construct orthonormal basis for L^2(V/Aut(G)).

    Each orbit O_i gets a basis vector:
        |O_i> = (1/sqrt(|O_i|)) * sum_{v in O_i} |v>

    Returns:
        basis: n x m matrix (columns are basis vectors in C^n), m = number of orbits
        orbits: list of orbits (each a list of vertex indices)
    """
    orbits = vertex_orbits(adjacency)
    n = adjacency.shape[0]
    m = len(orbits)

    basis = np.zeros((n, m))
    for i, orbit in enumerate(orbits):
        for v in orbit:
            basis[v, i] = 1.0 / np.sqrt(len(orbit))

    return basis, orbits


def orbit_projector(adjacency, orbit_idx):
    """
    Projection matrix P_i = |O_i><O_i| onto the i-th orbit subspace.

    Returns n x n projection matrix.
    """
    basis, orbits = orbit_hilbert_basis(adjacency)
    if orbit_idx >= len(orbits):
        raise ValueError(f"orbit_idx {orbit_idx} >= number of orbits {len(orbits)}")
    v = basis[:, orbit_idx]
    return np.outer(v, v)


def all_orbit_projectors(adjacency):
    """Return list of all orbit projectors."""
    basis, orbits = orbit_hilbert_basis(adjacency)
    projectors = []
    for i in range(len(orbits)):
        v = basis[:, i]
        projectors.append(np.outer(v, v))
    return projectors, orbits


# ============================================================
# Permutation Representation Decomposition
# ============================================================

def permutation_rep_decompose(adjacency):
    """
    Decompose the permutation representation of Aut(G) on C^n into irreps.

    Uses character theory:
    - Character of a permutation = number of fixed points
    - Multiplicity of irrep rho = (1/|G|) * sum_{g} chi_rho(g)* * chi_perm(g)

    Returns dict with group info and decomposition.
    """
    auts = graph_automorphisms(adjacency)
    n_group = len(auts)
    n = adjacency.shape[0]

    # Character of permutation rep: number of fixed points
    perm_characters = []
    for P in auts:
        fixed = sum(1 for i in range(n) if P[i, i] > 0.5)
        perm_characters.append(fixed)

    # Get conjugacy classes
    classes = conjugacy_classes(auts)
    n_classes = len(classes)

    # For small groups, determine the group type and build character table
    group_info = {
        'order': n_group,
        'n_conjugacy_classes': n_classes,
        'perm_characters': perm_characters,
    }

    if n_group == 1:
        # Trivial group: one irrep (trivial), multiplicity = n
        group_info['type'] = 'trivial'
        group_info['irreps'] = [{'name': 'trivial', 'dim': 1, 'multiplicity': n}]

    elif n_group == 2:
        # Z_2: two irreps (trivial, sign), both 1D
        # Character table: trivial = [1, 1], sign = [1, -1]
        # Identify the non-identity element
        non_id_idx = 0 if not np.allclose(auts[0], np.eye(n)) else 1
        id_idx = 1 - non_id_idx

        chi_perm_id = perm_characters[id_idx]
        chi_perm_g = perm_characters[non_id_idx]

        # Multiplicities
        m_trivial = (chi_perm_id * 1 + chi_perm_g * 1) / 2
        m_sign = (chi_perm_id * 1 + chi_perm_g * (-1)) / 2

        group_info['type'] = 'Z_2'
        group_info['irreps'] = [
            {'name': 'trivial', 'dim': 1, 'multiplicity': int(round(m_trivial))},
            {'name': 'sign', 'dim': 1, 'multiplicity': int(round(m_sign))},
        ]

        # Also compute eigenspaces of the non-identity element
        P_g = auts[non_id_idx]
        eigs, vecs = np.linalg.eigh(P_g)
        # Eigenvalue +1 = symmetric (trivial), -1 = antisymmetric (sign)
        n_symmetric = int(np.sum(np.abs(eigs - 1.0) < 0.1))
        n_antisymmetric = int(np.sum(np.abs(eigs + 1.0) < 0.1))
        group_info['eigenspace_check'] = {
            'n_symmetric': n_symmetric,
            'n_antisymmetric': n_antisymmetric,
            'matches_character': (n_symmetric == int(round(m_trivial)) and
                                  n_antisymmetric == int(round(m_sign))),
        }

    elif n_group == 6 and n_classes == 3:
        # S_3: three irreps: trivial (1D), sign (1D), standard (2D)
        # Character table:
        #              e    (12)   (123)
        # trivial:     1     1      1
        # sign:        1    -1      1
        # standard:    2     0     -1

        # Identify conjugacy classes by size: {e}=1, transpositions=3, 3-cycles=2
        class_sizes = [len(c) for c in classes]

        # Sort classes by size to identify: size 1 = identity, size 3 = transpositions, size 2 = 3-cycles
        class_order = sorted(range(n_classes), key=lambda i: class_sizes[i])
        # size 1 (identity), size 2 (3-cycles), size 3 (transpositions)

        chi_table = {
            'trivial': [1, 1, 1],
            'sign': [1, -1, 1],
            'standard': [2, 0, -1],
        }

        # Average perm character over each conjugacy class
        class_perm_chars = []
        for cls in classes:
            avg_char = np.mean([perm_characters[i] for i in cls])
            class_perm_chars.append(avg_char)

        # Reorder to match character table convention: identity, transposition, 3-cycle
        # Identity class has size 1 and character = n
        id_class = [i for i, s in enumerate(class_sizes) if s == 1][0]
        trans_class = [i for i, s in enumerate(class_sizes) if s == 3][0] if 3 in class_sizes else None
        cycle_class = [i for i, s in enumerate(class_sizes) if s == 2][0] if 2 in class_sizes else None

        ordered_chars = [0, 0, 0]
        ordered_sizes = [0, 0, 0]
        if id_class is not None:
            ordered_chars[0] = class_perm_chars[id_class]
            ordered_sizes[0] = class_sizes[id_class]
        if trans_class is not None:
            ordered_chars[1] = class_perm_chars[trans_class]
            ordered_sizes[1] = class_sizes[trans_class]
        if cycle_class is not None:
            ordered_chars[2] = class_perm_chars[cycle_class]
            ordered_sizes[2] = class_sizes[cycle_class]

        # Compute multiplicities: m_rho = (1/|G|) * sum_C |C| * chi_rho(C)* * chi_perm(C)
        multiplicities = {}
        for irrep_name, chi_rho in chi_table.items():
            m = 0
            for j in range(3):
                m += ordered_sizes[j] * chi_rho[j] * ordered_chars[j]
            m /= n_group
            multiplicities[irrep_name] = int(round(m))

        group_info['type'] = 'S_3'
        group_info['irreps'] = [
            {'name': name, 'dim': chi_table[name][0], 'multiplicity': mult}
            for name, mult in multiplicities.items()
        ]
        group_info['class_sizes'] = class_sizes
        group_info['class_perm_characters'] = class_perm_chars

    else:
        # General case: just report basic info
        group_info['type'] = f'order_{n_group}'
        group_info['irreps'] = []

    # Verify: sum of (dim * multiplicity) should equal n
    total_dim = sum(ir['dim'] * ir['multiplicity'] for ir in group_info.get('irreps', []))
    group_info['dim_check'] = total_dim == n

    return group_info


# ============================================================
# Born Rule & Measurement
# ============================================================

def born_probability(state, orbit_idx, adjacency):
    """
    Born probability P(orbit_i) = |<psi|O_i>|^2.

    state: complex vector in C^n (not necessarily gauge-invariant)
    """
    basis, orbits = orbit_hilbert_basis(adjacency)
    if orbit_idx >= len(orbits):
        raise ValueError(f"orbit_idx {orbit_idx} >= {len(orbits)}")
    orbit_vec = basis[:, orbit_idx]
    return float(np.abs(np.vdot(orbit_vec, state)) ** 2)


def born_probabilities(state, adjacency):
    """All Born probabilities for a state."""
    basis, orbits = orbit_hilbert_basis(adjacency)
    probs = []
    for i in range(len(orbits)):
        p = float(np.abs(np.vdot(basis[:, i], state)) ** 2)
        probs.append(p)
    return probs


def measurement_collapse(state, orbit_idx, adjacency):
    """
    Post-measurement state: project onto orbit, renormalize.

    |psi'> = P_i |psi> / ||P_i |psi>||
    """
    P = orbit_projector(adjacency, orbit_idx)
    projected = P @ state
    norm = np.linalg.norm(projected)
    if norm < 1e-15:
        raise ValueError(f"State has zero overlap with orbit {orbit_idx}")
    return projected / norm


# ============================================================
# Interference
# ============================================================

def two_path_amplitude(amplitude1, amplitude2):
    """
    Compute interference between two complex amplitudes.

    Returns dict with quantum (|A1+A2|^2) vs classical (|A1|^2+|A2|^2) probabilities.
    """
    a1 = complex(amplitude1)
    a2 = complex(amplitude2)

    p_quantum = abs(a1 + a2) ** 2
    p_classical = abs(a1) ** 2 + abs(a2) ** 2
    interference = p_quantum - p_classical  # = 2 * Re(a1* * a2)

    return {
        'amplitude1': a1,
        'amplitude2': a2,
        'p_quantum': float(p_quantum),
        'p_classical': float(p_classical),
        'interference_term': float(interference),
        'has_interference': abs(interference) > 1e-10,
    }


def interference_visibility(amplitudes, phases):
    """
    Compute interference visibility for a set of amplitudes with relative phases.

    V = (P_max - P_min) / (P_max + P_min) over all phase values.
    """
    # Sweep through detection phases
    n_sweep = 100
    detection_phases = np.linspace(0, 2 * np.pi, n_sweep)
    probabilities = []

    for dp in detection_phases:
        total = sum(a * np.exp(1j * (p + dp)) for a, p in zip(amplitudes, phases))
        probabilities.append(abs(total) ** 2)

    p_max = max(probabilities)
    p_min = min(probabilities)
    denom = p_max + p_min

    if denom < 1e-15:
        return 0.0

    return float((p_max - p_min) / denom)


# ============================================================
# Uncertainty
# ============================================================

def robertson_uncertainty(state, op_A, op_B):
    """
    Robertson uncertainty relation: Delta_A * Delta_B >= |<[A,B]>| / 2.

    Returns dict with variances, product, bound, and whether relation holds.
    """
    state = np.asarray(state, dtype=complex)
    op_A = np.asarray(op_A, dtype=complex)
    op_B = np.asarray(op_B, dtype=complex)

    # Expectation values
    exp_A = float(np.real(np.vdot(state, op_A @ state)))
    exp_B = float(np.real(np.vdot(state, op_B @ state)))

    # Variances
    exp_A2 = float(np.real(np.vdot(state, op_A @ op_A @ state)))
    exp_B2 = float(np.real(np.vdot(state, op_B @ op_B @ state)))

    var_A = exp_A2 - exp_A ** 2
    var_B = exp_B2 - exp_B ** 2

    delta_A = np.sqrt(max(var_A, 0))
    delta_B = np.sqrt(max(var_B, 0))

    # Commutator expectation
    comm = op_A @ op_B - op_B @ op_A
    exp_comm = np.vdot(state, comm @ state)
    robertson_bound = abs(exp_comm) / 2.0

    product = delta_A * delta_B

    return {
        'delta_A': float(delta_A),
        'delta_B': float(delta_B),
        'product': float(product),
        'robertson_bound': float(robertson_bound),
        'satisfied': product >= robertson_bound - 1e-10,
        'bound_nontrivial': robertson_bound > 1e-10,
    }


def noncommutativity_measure(group_elements):
    """
    Measure of non-commutativity: NC = (1/|G|^2) * sum_{g,h} ||[P_g, P_h]||_F.
    """
    n_g = len(group_elements)
    total = 0.0
    for g in group_elements:
        for h in group_elements:
            comm = g @ h - h @ g
            total += np.linalg.norm(comm, 'fro')
    return float(total / (n_g ** 2))


# ============================================================
# Entanglement
# ============================================================

def product_graph(adj1, adj2):
    """
    Tensor product of two graphs G1 x G2.

    Vertex (i,j) connects to (k,l) iff i-k in G1 AND j-l in G2.
    This is the tensor/categorical product (NOT Cartesian product).
    """
    n1 = adj1.shape[0]
    n2 = adj2.shape[0]
    n = n1 * n2
    A = np.zeros((n, n))

    for i1 in range(n1):
        for j1 in range(n2):
            for i2 in range(n1):
                for j2 in range(n2):
                    if adj1[i1, i2] > 0 and adj2[j1, j2] > 0:
                        A[i1 * n2 + j1, i2 * n2 + j2] = 1.0

    return A


def cartesian_product_graph(adj1, adj2):
    """
    Cartesian product G1 [] G2.

    Vertex (i,j) connects to (k,l) iff (i=k and j-l in G2) OR (j=l and i-k in G1).
    This preserves connectivity better than tensor product.
    """
    n1 = adj1.shape[0]
    n2 = adj2.shape[0]
    n = n1 * n2
    A = np.zeros((n, n))

    for i1 in range(n1):
        for j1 in range(n2):
            for i2 in range(n1):
                for j2 in range(n2):
                    if i1 == i2 and adj2[j1, j2] > 0:
                        A[i1 * n2 + j1, i2 * n2 + j2] = 1.0
                    elif j1 == j2 and adj1[i1, i2] > 0:
                        A[i1 * n2 + j1, i2 * n2 + j2] = 1.0

    return A


def partial_trace(density_matrix, n1, n2, trace_out='second'):
    """
    Partial trace of a density matrix over one subsystem.

    density_matrix: (n1*n2) x (n1*n2) matrix
    trace_out: 'first' or 'second'
    """
    rho = density_matrix.reshape(n1, n2, n1, n2)
    if trace_out == 'second':
        # Trace over second subsystem: sum over j
        return np.trace(rho, axis1=1, axis2=3)
    else:
        # Trace over first subsystem: sum over i
        return np.trace(rho, axis1=0, axis2=2)


def von_neumann_entropy(density_matrix):
    """
    Von Neumann entropy: S = -Tr(rho * ln(rho)).

    Uses eigenvalues to compute: S = -sum_i lambda_i * ln(lambda_i).
    """
    eigs = np.linalg.eigvalsh(density_matrix)
    eigs = eigs[eigs > 1e-15]  # Remove zeros to avoid log(0)
    return float(-np.sum(eigs * np.log(eigs)))


def purity(density_matrix):
    """Purity Tr(rho^2). Equals 1 for pure states, < 1 for mixed."""
    return float(np.real(np.trace(density_matrix @ density_matrix)))


# ============================================================
# Utility: Results Saving (reuse M13 pattern)
# ============================================================

RESULTS_DIR = M14_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_m14_results(experiment_name, data):
    """Save experiment results as timestamped JSON to M14 results dir."""
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results saved: {filepath.name}")
    return filepath
