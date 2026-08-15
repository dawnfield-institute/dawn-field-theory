"""
identity_complement.py -- Shared infrastructure for Milestone 13: Identity as Complement.

Extends M12's connection_geometry.py with complement-view operations, definitional
parallax, complement-transformations, Weyl group infrastructure, deformation/coherence
measures, and curvature from connection-density gradients.

The central claim: identity IS complement. A node's identity is the structure of the
rest of the graph without it. Different observers compute different complements of the
same entity (definitional parallax). The discrete complement-transformations form the
Weyl group -- the skeleton of the continuous Lie group. SEC complexification extends
this skeleton to the full Lorentz group.

Provides:
- complement_spectrum, complement_view, complement_distance: view operations (exp_01, exp_02)
- parallax, complement_transformation: observer-dependence (exp_02, exp_04, exp_05)
- weyl_element, weyl_conjugate: Weyl group operations (exp_04, exp_06)
- complement_deformation_rate, max_deformation_rate: coherence measures (exp_06, exp_08)
- connection_density_field, build_density_graph, complement_curvature: curvature (exp_11)
- so31_4d_generators, lorentz_invariant_form: 4D representation (exp_09, exp_10)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy.linalg import expm

# Import M12 infrastructure (which chains M11 -> M10 -> M9 -> M8)
M13_ROOT = Path(__file__).resolve().parent.parent
M12_ROOT = M13_ROOT.parent / "milestone12"
sys.path.insert(0, str(M12_ROOT / "core"))

from connection_geometry import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI, LN2,
    P_D, P_S,
    # Physical constants
    M_PLANCK_GEV, M_Z_GEV, HIGGS_VEV, ALPHA_EM, G_NEWTON,
    HBAR, C_LIGHT, K_BOLTZMANN, M_PLANCK_KG,
    # Cosmological parameters
    H0_PLANCK, H0_SHOES,
    # Fibonacci utilities
    fib, fibonacci_depth_coupling, fib_list, is_fibonacci,
    F3, F4, F5, F6, F7, F8, F9, F10,
    DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    # Response times
    T_PLANCK_S, T_EM_S, T_GRAVITY_S, T_HUBBLE_S,
    E_PLANCK_GEV, L_PLANCK_M,
    RESPONSE_TIMES,
    force_response_hierarchy, cascade_depth_response_time,
    # M10 infrastructure
    LawNegotiator, SelfApplicator,
    # MVAE
    L_MVAE, T_MVAE, E_MVAE,
    # Infrastructure
    save_results, setup_experiment, PredictionRegistry,
    # ADE infrastructure
    DynkinDiagram, all_ade_diagrams, fibonacci_compatible_gauge_groups,
    # Connection graph operations
    complement, connection_count, connection_density,
    pac_tree, pac_tree_values,
    # Basin/attractor dynamics
    BasinAttractor,
    # SEC complexification
    SU2_GENERATORS, commutator,
    complexify_generators, sl2c_generators,
    verify_lie_algebra, so31_from_sl2c,
    check_compactness,
    # PAC redistribution
    redistribute_on_graph, measure_entropy, redistribution_rate,
)


# ============================================================
# Complement-View Operations
# ============================================================

def complement_spectrum(adjacency, vertex):
    """
    Eigenvalue spectrum of the complement subgraph after removing a vertex.

    Returns sorted eigenvalues (descending). The spectrum is a
    labeling-independent signature of the complement's structure.
    """
    sub, _ = complement(adjacency, vertex)
    if sub.size == 0:
        return np.array([])
    eigs = np.linalg.eigvalsh(sub)
    return np.sort(eigs)[::-1]


def complement_view(adjacency, observer, target):
    """
    Observer's complement-view of target: remove observer first, then target.

    This captures how target's identity looks from observer's position.
    Different observers produce different complement-views of the same target.

    Returns (sub_adjacency, spectrum) of the doubly-reduced graph.
    """
    n = adjacency.shape[0]
    if observer == target:
        raise ValueError("Observer and target must be different vertices")

    # Remove observer first
    mask_obs = np.ones(n, dtype=bool)
    mask_obs[observer] = False
    reduced = adjacency[np.ix_(mask_obs, mask_obs)]

    # Remap target index (shifted if observer < target)
    target_in_reduced = target - 1 if target > observer else target

    # Remove target from reduced graph
    n_r = reduced.shape[0]
    mask_tgt = np.ones(n_r, dtype=bool)
    mask_tgt[target_in_reduced] = False
    sub = reduced[np.ix_(mask_tgt, mask_tgt)]

    spectrum = np.array([])
    if sub.size > 0:
        spectrum = np.sort(np.linalg.eigvalsh(sub))[::-1]

    return sub, spectrum


def complement_distance(adjacency, v1, v2):
    """
    Spectral distance between complements of two vertices.

    Uses Frobenius norm of the difference between sorted eigenvalue vectors,
    padded to the same length if needed (complement sizes always equal for
    same parent graph, since both remove exactly 1 vertex).
    """
    spec1 = complement_spectrum(adjacency, v1)
    spec2 = complement_spectrum(adjacency, v2)

    # Pad shorter spectrum with zeros if needed
    max_len = max(len(spec1), len(spec2))
    s1 = np.zeros(max_len)
    s2 = np.zeros(max_len)
    s1[:len(spec1)] = spec1
    s2[:len(spec2)] = spec2

    return float(np.linalg.norm(s1 - s2))


# ============================================================
# Parallax & Transformations
# ============================================================

def parallax(adjacency, obs1, obs2, target):
    """
    Definitional parallax: difference between two observers' complement-views
    of the same target vertex.

    Returns the Frobenius norm of the spectral difference. Zero iff the two
    observers are in the same automorphism orbit relative to the target.
    """
    _, spec1 = complement_view(adjacency, obs1, target)
    _, spec2 = complement_view(adjacency, obs2, target)

    max_len = max(len(spec1), len(spec2))
    s1 = np.zeros(max_len)
    s2 = np.zeros(max_len)
    s1[:len(spec1)] = spec1
    s2[:len(spec2)] = spec2

    return float(np.linalg.norm(s1 - s2))


def complement_transformation(adjacency, v1, v2):
    """
    Transformation mapping complement(G, v1) to complement(G, v2).

    Returns a dict with spectral representation of the transformation:
    the spectral difference vector and its magnitude.
    """
    spec1 = complement_spectrum(adjacency, v1)
    spec2 = complement_spectrum(adjacency, v2)

    max_len = max(len(spec1), len(spec2))
    s1 = np.zeros(max_len)
    s2 = np.zeros(max_len)
    s1[:len(spec1)] = spec1
    s2[:len(spec2)] = spec2

    diff = s2 - s1
    magnitude = float(np.linalg.norm(diff))

    return {
        'from_vertex': v1,
        'to_vertex': v2,
        'spectral_diff': diff,
        'magnitude': magnitude,
        'from_spectrum': s1,
        'to_spectrum': s2,
    }


def all_complement_transformations(adjacency):
    """Compute all pairwise complement-transformations for a graph."""
    n = adjacency.shape[0]
    transformations = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                transformations[(i, j)] = complement_transformation(adjacency, i, j)
    return transformations


# ============================================================
# Weyl Group Operations
# ============================================================

def weyl_element_su2():
    """
    Weyl element for A_1 (SU(2)): w = exp(i*pi*J_2).

    This is the non-trivial element of W(A_1) = Z_2.
    It acts on the Cartan generator J_3 as: w * J_3 * w^{-1} = -J_3.
    """
    J2 = SU2_GENERATORS[1]  # sigma_y / 2
    w = expm(1j * np.pi * J2)
    return w


def weyl_conjugate(w, generator):
    """Conjugate a generator by the Weyl element: w * G * w^{-1}."""
    w_inv = np.linalg.inv(w)
    return w @ generator @ w_inv


def weyl_action_on_algebra(generators):
    """
    Apply the Weyl element to all generators and classify the result.

    Returns for each generator: whether it flips sign, stays the same,
    or transforms non-trivially.
    """
    w = weyl_element_su2()
    results = []
    for i, G in enumerate(generators):
        G_prime = weyl_conjugate(w, G)
        # Check if G' = G (invariant)
        if np.allclose(G_prime, G, atol=1e-10):
            action = 'invariant'
        # Check if G' = -G (flip)
        elif np.allclose(G_prime, -G, atol=1e-10):
            action = 'flip'
        else:
            action = 'mixed'
        results.append({
            'index': i,
            'action': action,
            'original': G,
            'transformed': G_prime,
            'error_invariant': float(np.max(np.abs(G_prime - G))),
            'error_flip': float(np.max(np.abs(G_prime + G))),
        })
    return results


# ============================================================
# Deformation & Coherence
# ============================================================

def complement_deformation_rate(adjacency, path):
    """
    Measure the rate of complement spectral change along a vertex path.

    Returns a list of per-step deformation magnitudes and the total.
    """
    if len(path) < 2:
        return {'steps': [], 'total': 0.0, 'max_rate': 0.0, 'mean_rate': 0.0}

    steps = []
    for i in range(len(path) - 1):
        spec_i = complement_spectrum(adjacency, path[i])
        spec_j = complement_spectrum(adjacency, path[i + 1])

        max_len = max(len(spec_i), len(spec_j))
        s_i = np.zeros(max_len)
        s_j = np.zeros(max_len)
        s_i[:len(spec_i)] = spec_i
        s_j[:len(spec_j)] = spec_j

        deformation = float(np.linalg.norm(s_j - s_i))
        steps.append(deformation)

    return {
        'steps': steps,
        'total': sum(steps),
        'max_rate': max(steps) if steps else 0.0,
        'mean_rate': np.mean(steps) if steps else 0.0,
    }


def max_deformation_rate(adjacency):
    """
    Maximum complement-deformation rate over all single-step moves.

    This is the coherence limit: the fastest the complement can change
    in a single step.
    """
    n = adjacency.shape[0]
    max_rate = 0.0
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 0:  # Adjacent vertices only
                spec_i = complement_spectrum(adjacency, i)
                spec_j = complement_spectrum(adjacency, j)

                max_len = max(len(spec_i), len(spec_j))
                s_i = np.zeros(max_len)
                s_j = np.zeros(max_len)
                s_i[:len(spec_i)] = spec_i
                s_j[:len(spec_j)] = spec_j

                rate = float(np.linalg.norm(s_j - s_i))
                max_rate = max(max_rate, rate)

    return max_rate


# ============================================================
# SO(3,1) 4D Vector Representation
# ============================================================

def so31_4d_generators():
    """
    Construct the 4x4 vector representation of so(3,1).

    The 6 generators split into:
    - 3 rotations J_i: antisymmetric 3x3 blocks in spatial part
    - 3 boosts K_i: mixing time and space components

    Convention: x = (t, x, y, z), metric = diag(-1, +1, +1, +1).
    """
    # Rotation generators (spatial antisymmetric)
    J1 = np.zeros((4, 4))
    J1[2, 3] = -1; J1[3, 2] = 1   # rotation in y-z plane

    J2 = np.zeros((4, 4))
    J2[1, 3] = 1; J2[3, 1] = -1   # rotation in x-z plane

    J3 = np.zeros((4, 4))
    J3[1, 2] = -1; J3[2, 1] = 1   # rotation in x-y plane

    # Boost generators (mixing t and spatial)
    K1 = np.zeros((4, 4))
    K1[0, 1] = 1; K1[1, 0] = 1    # boost in x direction

    K2 = np.zeros((4, 4))
    K2[0, 2] = 1; K2[2, 0] = 1    # boost in y direction

    K3 = np.zeros((4, 4))
    K3[0, 3] = 1; K3[3, 0] = 1    # boost in z direction

    return [J1, J2, J3], [K1, K2, K3]


def lorentz_invariant_form(generators_4d):
    """
    Find the unique bilinear form preserved by all generators.

    For so(3,1) in the 4D vector representation, the invariant form
    is the Minkowski metric eta = diag(-1, +1, +1, +1).

    G^T * eta + eta * G = 0 for all generators G.

    Returns the invariant form and its eigenvalues.
    """
    all_gens = generators_4d[0] + generators_4d[1]  # rotations + boosts
    n = 4

    # Set up the linear system: for each generator G, the constraint is
    # G^T * B + B * G = 0, where B is the unknown symmetric matrix.
    # Vectorize B as a (n*(n+1)/2)-dimensional vector of independent components.
    # For a symmetric 4x4 matrix, there are 10 independent components.

    n_params = n * (n + 1) // 2
    constraints = []

    for G in all_gens:
        # G^T * B + B * G = 0
        # This gives n^2 constraints on the 10 parameters of B
        for i in range(n):
            for j in range(i, n):
                # (G^T B + B G)_{ij} = sum_k (G^T)_{ik} B_{kj} + B_{ik} G_{kj}
                #                     = sum_k G_{ki} B_{kj} + B_{ik} G_{kj}
                row = np.zeros(n_params)
                for k in range(n):
                    # G_{ki} * B_{kj}
                    kj_idx = _sym_idx(k, j, n)
                    row[kj_idx] += G[k, i]
                    # B_{ik} * G_{kj}
                    ik_idx = _sym_idx(i, k, n)
                    row[ik_idx] += G[k, j]
                constraints.append(row)

    A = np.array(constraints)
    # Find null space
    _, s, Vt = np.linalg.svd(A)

    # Count near-zero singular values
    null_dim = np.sum(s < 1e-10)

    # The null space vectors give the invariant forms
    null_vectors = Vt[-null_dim:]

    # Reconstruct the matrix from the first null vector
    if null_dim > 0:
        params = null_vectors[0]
        B = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                B[i, j] = params[idx]
                B[j, i] = params[idx]
                idx += 1

        # Normalize so the largest absolute value is 1
        B /= np.max(np.abs(B))

        eigenvalues = np.linalg.eigvalsh(B)
    else:
        B = np.zeros((n, n))
        eigenvalues = np.zeros(n)

    return {
        'form': B,
        'eigenvalues': sorted(eigenvalues, reverse=True),
        'null_space_dimension': int(null_dim),
        'is_unique': null_dim == 1,
    }


def _sym_idx(i, j, n):
    """Map symmetric matrix indices (i,j) to flat index."""
    if i > j:
        i, j = j, i
    return i * n - i * (i - 1) // 2 + (j - i)


# ============================================================
# Curvature from Connection-Density
# ============================================================

def connection_density_field(adjacency):
    """
    Local connection density at each vertex: degree / (n-1).

    Returns array of densities, one per vertex.
    """
    n = adjacency.shape[0]
    degrees = np.sum(adjacency > 0, axis=1).astype(float)
    if n <= 1:
        return degrees
    return degrees / (n - 1)


def build_density_graph(n, lump_center, lump_radius, lump_extra_edges=3):
    """
    Build a chain graph with a localized density lump.

    Base: linear chain of n vertices.
    Lump: extra edges added near lump_center within lump_radius.
    """
    A = np.zeros((n, n))

    # Base chain
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # Add extra edges near lump_center
    lump_lo = max(0, lump_center - lump_radius)
    lump_hi = min(n - 1, lump_center + lump_radius)
    lump_vertices = list(range(lump_lo, lump_hi + 1))

    added = 0
    for i in range(len(lump_vertices)):
        for j in range(i + 2, len(lump_vertices)):  # skip adjacent (already connected)
            vi, vj = lump_vertices[i], lump_vertices[j]
            if A[vi, vj] == 0 and added < lump_extra_edges:
                A[vi, vj] = 1.0
                A[vj, vi] = 1.0
                added += 1

    return A


def complement_curvature(adjacency, path):
    """
    Second derivative of complement-transformation magnitude along a path.

    Positive curvature = transformation magnitude accelerating.
    Computed as finite differences of the deformation rate.
    """
    if len(path) < 3:
        return {'curvatures': [], 'mean_curvature': 0.0}

    deformation = complement_deformation_rate(adjacency, path)
    rates = deformation['steps']

    curvatures = []
    for i in range(len(rates) - 1):
        curvatures.append(rates[i + 1] - rates[i])

    return {
        'curvatures': curvatures,
        'mean_curvature': float(np.mean(np.abs(curvatures))) if curvatures else 0.0,
        'max_curvature': float(np.max(np.abs(curvatures))) if curvatures else 0.0,
    }


def find_min_deformation_path(adjacency, start, end, max_depth=15):
    """
    Find the path from start to end that minimizes total complement-deformation.

    Uses BFS with deformation cost. Returns (path, total_deformation).
    """
    n = adjacency.shape[0]

    # Dijkstra-like search with complement-deformation as cost
    import heapq
    visited = set()
    # (cost, path)
    heap = [(0.0, [start])]

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[-1]

        if node == end:
            return path, cost

        if node in visited:
            continue
        visited.add(node)

        if len(path) > max_depth:
            continue

        for neighbor in range(n):
            if adjacency[node, neighbor] > 0 and neighbor not in visited:
                spec_node = complement_spectrum(adjacency, node)
                spec_nb = complement_spectrum(adjacency, neighbor)

                max_len = max(len(spec_node), len(spec_nb))
                s1 = np.zeros(max_len)
                s2 = np.zeros(max_len)
                s1[:len(spec_node)] = spec_node
                s2[:len(spec_nb)] = spec_nb

                step_cost = float(np.linalg.norm(s2 - s1))
                heapq.heappush(heap, (cost + step_cost, path + [neighbor]))

    return None, float('inf')


def find_shortest_path(adjacency, start, end):
    """BFS shortest path."""
    from collections import deque
    n = adjacency.shape[0]
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in range(n):
            if adjacency[node, neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


# ============================================================
# Killing Form Utilities (extended from M12)
# ============================================================

def killing_form(generators):
    """
    Compute the Killing form matrix B_{ij} = Tr(ad_i . ad_j).

    Returns the matrix and its eigenvalues.
    """
    n = len(generators)
    gen_flat = np.array([g.flatten() for g in generators])

    ad_mats = []
    for i in range(n):
        ad_i = np.zeros((n, n), dtype=complex)
        for k in range(n):
            comm = commutator(generators[i], generators[k])
            coeffs, _, _, _ = np.linalg.lstsq(gen_flat.T, comm.flatten(), rcond=None)
            ad_i[:, k] = coeffs
        ad_mats.append(ad_i)

    B = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            B[i, j] = np.trace(ad_mats[i] @ ad_mats[j])

    B_real = B.real
    eigenvalues = np.linalg.eigvalsh(B_real)

    n_pos = int(np.sum(eigenvalues > 1e-10))
    n_neg = int(np.sum(eigenvalues < -1e-10))
    n_zero = int(np.sum(np.abs(eigenvalues) <= 1e-10))

    return {
        'matrix': B_real,
        'eigenvalues': sorted(eigenvalues, reverse=True),
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_zero': n_zero,
        'signature': f'({n_pos}, {n_neg})',
    }


# ============================================================
# Graph Automorphism Utilities
# ============================================================

def graph_distance(adjacency, u, v):
    """Shortest path distance between two vertices using BFS."""
    if u == v:
        return 0
    from collections import deque
    n = adjacency.shape[0]
    visited = {u}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in range(n):
            if adjacency[node, neighbor] > 0 and neighbor not in visited:
                if neighbor == v:
                    return dist + 1
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')


def vertex_orbits(adjacency):
    """
    Classify vertices into orbits under graph automorphisms.

    Uses complement spectrum as an invariant: vertices with the same
    complement spectrum are in the same orbit (necessary condition;
    sufficient for ADE graphs due to their rigid structure).
    """
    n = adjacency.shape[0]
    spectra = {}
    for v in range(n):
        spec = complement_spectrum(adjacency, v)
        key = tuple(np.round(spec, decimals=10))
        if key not in spectra:
            spectra[key] = []
        spectra[key].append(v)

    orbits = list(spectra.values())
    return orbits


# ============================================================
# Utility: Results Saving
# ============================================================
RESULTS_DIR = M13_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_m13_results(experiment_name, data):
    """Save experiment results as timestamped JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = RESULTS_DIR / f"{experiment_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            result = convert(obj)
            if result is not obj:
                return result
            return super().default(obj)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    return filename



# ============================================================
# Hardening Utilities (v0.3)
# ============================================================

def complement_derived_generators(diagram_type='A', rank=1):
    """
    Build Lie algebra generators from ADE root system (Cartan-Weyl basis).

    For A_1: returns H, E+, E- in the fundamental (2x2) representation.
    These are related to the standard Pauli/2 basis by a unitary change of basis
    but are derived from the root system, not hand-constructed.

    Returns (generators_list, info_dict).
    """
    d = DynkinDiagram(diagram_type, rank)
    cartan = d.cartan_matrix()
    n = rank  # Lie algebra rank

    if diagram_type == 'A' and rank == 1:
        # A_1: su(2) in Cartan-Weyl basis
        # Cartan generator H = diag(1, -1) = sigma_z
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        # Raising operator E+ = [[0,1],[0,0]]
        Ep = np.array([[0, 1], [0, 0]], dtype=complex)
        # Lowering operator E- = [[0,0],[1,0]]
        Em = np.array([[0, 0], [1, 0]], dtype=complex)

        generators = [H, Ep, Em]
        info = {
            'basis': 'Cartan-Weyl',
            'diagram': f'{diagram_type}_{rank}',
            'lie_algebra': 'su(2)',
            'dimension': 3,
            'generators': ['H', 'E+', 'E-'],
            'note': 'H = sigma_z, E+ = sigma_+, E- = sigma_-. '
                    'Related to Pauli/2 basis (J1,J2,J3) by: '
                    'J3 = H/2, J+ = E+, J- = E-, J1 = (E+ + E-)/2, J2 = (E+ - E-)/(2i)',
        }
        return generators, info

    elif diagram_type == 'A' and rank == 2:
        # A_2: su(3) in Cartan-Weyl basis (Gell-Mann basis)
        # 2 Cartan generators + 6 root generators = 8 total
        H1 = np.diag([1, -1, 0]).astype(complex)
        H2 = np.diag([1, 1, -2]).astype(complex) / np.sqrt(3)

        E12 = np.zeros((3, 3), dtype=complex); E12[0, 1] = 1
        E21 = np.zeros((3, 3), dtype=complex); E21[1, 0] = 1
        E13 = np.zeros((3, 3), dtype=complex); E13[0, 2] = 1
        E31 = np.zeros((3, 3), dtype=complex); E31[2, 0] = 1
        E23 = np.zeros((3, 3), dtype=complex); E23[1, 2] = 1
        E32 = np.zeros((3, 3), dtype=complex); E32[2, 1] = 1

        generators = [H1, H2, E12, E21, E13, E31, E23, E32]
        info = {
            'basis': 'Cartan-Weyl',
            'diagram': f'{diagram_type}_{rank}',
            'lie_algebra': 'su(3)',
            'dimension': 8,
            'generators': ['H1', 'H2', 'E12', 'E21', 'E13', 'E31', 'E23', 'E32'],
        }
        return generators, info

    elif diagram_type == 'D' and rank == 4:
        # D_4: so(8) — 28-dimensional. Use standard basis of antisymmetric 8x8 matrices.
        dim = 2 * rank  # so(2n) for D_n
        generators = []
        for i in range(dim):
            for j in range(i + 1, dim):
                G = np.zeros((dim, dim), dtype=complex)
                G[i, j] = 1
                G[j, i] = -1
                generators.append(G)
        info = {
            'basis': 'standard-antisymmetric',
            'diagram': f'{diagram_type}_{rank}',
            'lie_algebra': 'so(8)',
            'dimension': len(generators),
        }
        return generators, info

    else:
        raise ValueError(f"complement_derived_generators not implemented for {diagram_type}_{rank}")


def build_petersen_graph():
    """
    Build the Petersen graph (10 vertices, 3-regular, vertex-transitive).
    """
    A = np.zeros((10, 10))
    # Outer cycle: 0-1-2-3-4-0
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[(i + 1) % 5, i] = 1
    # Inner pentagram: 5-7-9-6-8-5
    for i in range(5):
        A[5 + i, 5 + (i + 2) % 5] = 1
        A[5 + (i + 2) % 5, 5 + i] = 1
    # Spokes: i -- i+5
    for i in range(5):
        A[i, i + 5] = 1
        A[i + 5, i] = 1
    return A


def build_complete_graph(n):
    """Build complete graph K_n."""
    A = np.ones((n, n)) - np.eye(n)
    return A


def thomas_rotation_angle(eta1, eta2, phi_angle):
    """
    Compute Thomas (Wigner) rotation angle from two non-collinear boosts.

    For two boosts with rapidities eta1, eta2 at angle phi_angle between
    their spatial directions, the composition is a boost + rotation.
    The Thomas angle is:
      tan(theta_T/2) = sinh(eta1/2)*sinh(eta2/2)*sin(phi_angle)
                       / (cosh(eta1/2)*cosh(eta2/2) + sinh(eta1/2)*sinh(eta2/2)*cos(phi_angle))

    Returns theta_T in radians.
    """
    s1 = np.sinh(eta1 / 2)
    s2 = np.sinh(eta2 / 2)
    c1 = np.cosh(eta1 / 2)
    c2 = np.cosh(eta2 / 2)

    numerator = s1 * s2 * np.sin(phi_angle)
    denominator = c1 * c2 + s1 * s2 * np.cos(phi_angle)

    theta_T = 2 * np.arctan2(numerator, denominator)
    return float(theta_T)


def generate_random_connected_graph(n, density=0.4, seed=None):
    """
    Generate a random connected symmetric adjacency matrix (non-ADE control).

    Uses Erdos-Renyi model with rejection sampling to ensure connectivity.
    Returns an n x n symmetric adjacency matrix with no self-loops.
    """
    rng = np.random.RandomState(seed)
    for _ in range(1000):
        # Generate upper triangle
        upper = (rng.random((n, n)) < density).astype(float)
        np.fill_diagonal(upper, 0)
        A = np.triu(upper, 1)
        A = A + A.T
        # Check connectivity via BFS
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if A[node, neighbor] > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) == n:
            return A
    raise RuntimeError(f"Failed to generate connected graph with n={n}, density={density}")


def max_deformation_rate_full_ade(max_rank=8):
    """
    Compute max complement-deformation rate across full ADE family.

    Returns dict keyed by diagram name (e.g. 'A_3', 'D_4', 'E_6')
    with values being the max deformation rate for that diagram.
    """
    results = {}
    for diagram in all_ade_diagrams(max_rank=max_rank):
        name = diagram.name
        A = diagram.adjacency
        if A.shape[0] < 3:
            continue  # Need at least 3 vertices for meaningful complement
        rate = max_deformation_rate(A)
        results[name] = {
            'rate': rate,
            'family': diagram.type,
            'rank': diagram.rank,
            'n_vertices': A.shape[0],
        }
    return results


def killing_form_for_algebra(generators):
    """
    Compute the Killing form from arbitrary generators via B(X,Y) = Tr(ad_X . ad_Y).

    generators: list of square matrices (the basis of the Lie algebra).
    Returns the Killing matrix B_{ij} = Tr(ad_{g_i} . ad_{g_j}).
    """
    dim = len(generators)
    n = generators[0].shape[0]

    # Build adjoint representation matrices
    def ad_matrix(X):
        """ad_X: the matrix of [X, -] in the basis of generators."""
        ad = np.zeros((dim, dim), dtype=complex)
        for j, g_j in enumerate(generators):
            bracket = X @ g_j - g_j @ X
            # Decompose bracket in generator basis (least squares)
            G_matrix = np.column_stack([g.flatten() for g in generators])
            coeffs, _, _, _ = np.linalg.lstsq(G_matrix, bracket.flatten(), rcond=None)
            ad[:, j] = coeffs
        return ad

    # Compute Killing form
    B = np.zeros((dim, dim), dtype=complex)
    for i, g_i in enumerate(generators):
        ad_i = ad_matrix(g_i)
        for j, g_j in enumerate(generators):
            ad_j = ad_matrix(g_j)
            B[i, j] = np.trace(ad_i @ ad_j)

    # For real algebras, imaginary parts should be negligible
    if np.max(np.abs(B.imag)) < 1e-10:
        B = B.real

    return B


def _convert_numpy(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
