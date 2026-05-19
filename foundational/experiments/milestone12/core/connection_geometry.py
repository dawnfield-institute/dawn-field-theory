"""
connection_geometry.py -- Shared infrastructure for Milestone 12: Connection as Primitive.

Extends M11's quantum_gravity.py with ADE root system infrastructure, connection
graph operations, basin/attractor dynamics, and SEC complexification utilities.

The central claim: connection = addition = ADE geometry. PAC recursion lives in
an ADE root lattice; the golden ratio is its spectral radius. Gauge groups are
the ADE types whose adjoint dimensions are Fibonacci numbers. Laws are standing
waves (basins) in connection space with measurable relaxation times. SEC
complexification of A_1 yields the Lorentz group.

Provides:
- DynkinDiagram: ADE Dynkin diagrams as adjacency matrices (exp_01, exp_02, exp_03)
- cartan_matrix, spectral_radius, adjoint_dimension: ADE algebraic properties
- complement, connection_density: Connection graph operations (exp_04, exp_05)
- BasinAttractor: Attractor dynamics in connection space (exp_06, exp_07, exp_08)
- complexify_generators, verify_lie_algebra: SEC complexification (exp_10, exp_11)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

# Import M11 infrastructure (which chains M10 -> M9 -> M8)
M12_ROOT = Path(__file__).resolve().parent.parent
M11_ROOT = M12_ROOT.parent / "milestone11"
sys.path.insert(0, str(M11_ROOT / "core"))

from quantum_gravity import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI, LN2,
    P_D, P_S,
    # Physical constants
    M_PLANCK_GEV, M_Z_GEV, HIGGS_VEV, ALPHA_EM, G_NEWTON,
    HBAR, C_LIGHT, K_BOLTZMANN, M_PLANCK_KG,
    # Cosmological parameters
    H0_PLANCK, H0_SHOES,
    # Fibonacci utilities
    fib, fibonacci_depth_coupling,
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
)


# ============================================================
# Fibonacci Lookup (extended)
# ============================================================
def fib_list(n):
    """Return first n Fibonacci numbers starting from F_1=1."""
    fibs = [1, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

# Precompute first 200 Fibonacci numbers for lookup
_FIB_CACHE = fib_list(200)

def is_fibonacci(n):
    """Check if n is a Fibonacci number."""
    return n in _FIB_CACHE


# ============================================================
# ADE Dynkin Diagram Infrastructure
# ============================================================
class DynkinDiagram:
    """
    ADE Dynkin diagram as adjacency matrix.

    Supports A_n, D_n (n>=4), E_6, E_7, E_8.
    The adjacency matrix encodes which nodes are connected.
    """

    def __init__(self, type_letter, rank):
        self.type = type_letter.upper()
        self.rank = rank
        self.name = f"{self.type}_{rank}"
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        n = self.rank
        A = np.zeros((n, n), dtype=float)

        if self.type == 'A':
            # A_n: linear chain 1-2-3-...-n
            for i in range(n - 1):
                A[i, i + 1] = 1.0
                A[i + 1, i] = 1.0

        elif self.type == 'D':
            if n < 4:
                raise ValueError("D_n requires n >= 4")
            # D_n: linear chain 1-2-..-(n-2), then (n-2) branches to (n-1) and n
            for i in range(n - 2):
                if i < n - 3:
                    A[i, i + 1] = 1.0
                    A[i + 1, i] = 1.0
            # Branch: node (n-3) connects to both (n-2) and (n-1)
            A[n - 3, n - 2] = 1.0
            A[n - 2, n - 3] = 1.0
            A[n - 3, n - 1] = 1.0
            A[n - 1, n - 3] = 1.0

        elif self.type == 'E':
            if n not in (6, 7, 8):
                raise ValueError("E_n requires n in {6, 7, 8}")
            # E_n: linear chain 1-2-3-4-..., with branch at node 3 (0-indexed: node 2)
            for i in range(n - 1):
                if i < n - 2:
                    A[i, i + 1] = 1.0
                    A[i + 1, i] = 1.0
            # Branch: node 2 connects to last node (the branch node)
            A[2, n - 1] = 1.0
            A[n - 1, 2] = 1.0

        else:
            raise ValueError(f"Unknown type: {self.type}. Use A, D, or E.")

        return A

    def cartan_matrix(self):
        """
        Cartan matrix C = 2I - A (for simply-laced ADE).

        The Cartan matrix encodes the inner products of simple roots.
        For ADE (simply-laced), off-diagonal entries are 0 or -1.
        """
        return 2 * np.eye(self.rank) - self.adjacency

    def spectral_radius(self):
        """Largest eigenvalue of the adjacency matrix."""
        eigenvalues = np.linalg.eigvalsh(self.adjacency)
        return float(np.max(np.abs(eigenvalues)))

    def eigenvalues(self):
        """All eigenvalues of the adjacency matrix, sorted."""
        eigs = np.linalg.eigvalsh(self.adjacency)
        return sorted(eigs, reverse=True)

    def adjoint_dimension(self):
        """
        Dimension of the adjoint representation of the corresponding Lie algebra.

        For A_n: n(n+2) = (n+1)^2 - 1 (SU(n+1) generators)
        For D_n: n(2n-1) (SO(2n) generators)
        For E_6: 78, E_7: 133, E_8: 248
        """
        n = self.rank
        if self.type == 'A':
            return n * (n + 2)  # = (n+1)^2 - 1
        elif self.type == 'D':
            return n * (2 * n - 1)
        elif self.type == 'E':
            dims = {6: 78, 7: 133, 8: 248}
            return dims[n]

    def lie_group_name(self):
        """Name of the corresponding compact Lie group."""
        n = self.rank
        if self.type == 'A':
            return f"SU({n + 1})"
        elif self.type == 'D':
            return f"SO({2 * n})"
        elif self.type == 'E':
            return f"E_{n}"

    def positive_root_count(self):
        """Number of positive roots = (adjoint_dim - rank) / 2 + rank = adjoint_dim/2 + rank/2.
        Actually for simply-laced: |Phi+| = (dim - rank) / 2."""
        return (self.adjoint_dimension() - self.rank) // 2

    def __repr__(self):
        return f"DynkinDiagram({self.name}, group={self.lie_group_name()}, adj_dim={self.adjoint_dimension()})"


def all_ade_diagrams(max_rank=20):
    """Generate all ADE Dynkin diagrams up to given rank."""
    diagrams = []
    for n in range(1, max_rank + 1):
        diagrams.append(DynkinDiagram('A', n))
    for n in range(4, max_rank + 1):
        diagrams.append(DynkinDiagram('D', n))
    for n in (6, 7, 8):
        diagrams.append(DynkinDiagram('E', n))
    return diagrams


def fibonacci_compatible_gauge_groups(max_rank=20):
    """
    Find all ADE types whose adjoint dimension is a Fibonacci number.

    This is the key theorem: only A_1 (SU(2), dim=3=F_4) and A_2 (SU(3), dim=8=F_6)
    have Fibonacci adjoint dimensions among ALL simple Lie algebras.
    """
    results = []
    for diagram in all_ade_diagrams(max_rank):
        dim = diagram.adjoint_dimension()
        if is_fibonacci(dim):
            results.append({
                'diagram': diagram.name,
                'group': diagram.lie_group_name(),
                'adjoint_dim': dim,
                'fibonacci_index': _FIB_CACHE.index(dim) + 1 if dim in _FIB_CACHE else None,
            })
    return results


# ============================================================
# Connection Graph Operations
# ============================================================
def complement(adjacency, vertex):
    """
    Compute the complement of a vertex in a connection graph.

    Remove vertex and all incident edges, return the induced subgraph
    on remaining vertices. This is the "identity as complement" operation
    from iddea.md Section 3.

    Returns: (sub_adjacency, removed_edges_count)
    """
    n = adjacency.shape[0]
    if vertex < 0 or vertex >= n:
        raise ValueError(f"Vertex {vertex} out of range [0, {n})")

    # Count removed edges (connections to this vertex)
    removed = int(np.sum(adjacency[vertex, :] > 0))

    # Build sub-adjacency by deleting row and column
    mask = np.ones(n, dtype=bool)
    mask[vertex] = False
    sub = adjacency[np.ix_(mask, mask)]

    return sub, removed


def connection_count(adjacency):
    """Total number of connections (edges) in an undirected graph."""
    return int(np.sum(adjacency) / 2)


def connection_density(adjacency, vertex):
    """Local connection density: degree of vertex / max possible degree."""
    n = adjacency.shape[0]
    degree = np.sum(adjacency[vertex, :] > 0)
    return float(degree / (n - 1)) if n > 1 else 0.0


def pac_tree(depth):
    """
    Build a PAC binary tree of given depth as adjacency matrix.

    Each node at level k connects to two children at level k+1.
    PAC conservation: value(parent) = value(child1) + value(child2).
    With phi-split: child1 gets fraction 1/phi, child2 gets 1/phi^2.
    """
    n_nodes = 2 ** (depth + 1) - 1
    A = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_nodes:
            A[i, left] = 1.0
            A[left, i] = 1.0
        if right < n_nodes:
            A[i, right] = 1.0
            A[right, i] = 1.0

    return A


def pac_tree_values(depth):
    """
    Compute PAC-conserved values on a binary tree with phi-split.

    Root = 1.0, each parent splits as: left = parent/phi, right = parent/phi^2.
    Conservation: parent = left + right = parent*(1/phi + 1/phi^2) = parent*1 (exact).
    """
    n_nodes = 2 ** (depth + 1) - 1
    values = np.zeros(n_nodes)
    values[0] = 1.0

    for i in range(n_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_nodes:
            values[left] = values[i] * INV_PHI
        if right < n_nodes:
            values[right] = values[i] * INV_PHI ** 2

    return values


# ============================================================
# Basin / Attractor Dynamics
# ============================================================
class BasinAttractor:
    """
    A law-as-attractor in connection space.

    An attractor is a configuration (set of connection weights) that
    PAC redistribution dynamics keep reinstating after perturbation.
    Characterized by: depth (perturbation energy to escape),
    width (range of perturbations absorbed), relaxation time (steps to return).
    """

    def __init__(self, name, equilibrium_value, cascade_depth,
                 coupling_strength=None):
        self.name = name
        self.equilibrium = equilibrium_value
        self.cascade_depth = cascade_depth
        # Physical coupling: phi^{-depth}. Pass coupling_strength explicitly
        # for simulation-tractable dynamics at extreme depths.
        self.coupling = coupling_strength or PHI ** (-cascade_depth)
        self.history = []

    def perturb(self, state, magnitude):
        """Apply perturbation to state and return perturbed state."""
        return state + magnitude * np.random.randn(*state.shape)

    def redistribute(self, state, dt=0.01):
        """
        One step of PAC redistribution dynamics.

        Drives state toward equilibrium via phi-weighted relaxation.
        Rate depends on coupling strength (deeper cascade = weaker coupling = slower).
        """
        deviation = state - self.equilibrium
        # Relaxation rate proportional to coupling strength
        rate = self.coupling * dt
        new_state = state - rate * deviation
        return new_state

    def measure_relaxation_time(self, perturbation_magnitude, dt=0.01,
                                tolerance=0.01, max_steps=100000):
        """
        Measure how many steps to return within tolerance of equilibrium.

        Returns (steps, converged, final_deviation).
        """
        if np.isscalar(self.equilibrium):
            state = np.array([self.equilibrium + perturbation_magnitude])
        else:
            state = self.equilibrium + perturbation_magnitude * np.random.randn(
                *self.equilibrium.shape)

        for step in range(max_steps):
            state = self.redistribute(state, dt)
            deviation = np.max(np.abs(state - self.equilibrium))
            if deviation < tolerance:
                return step + 1, True, float(deviation)

        return max_steps, False, float(np.max(np.abs(state - self.equilibrium)))

    def measure_basin_depth(self, n_trials=100, dt=0.01, tolerance=0.01,
                            max_steps=50000):
        """
        Estimate basin depth: largest perturbation that still relaxes back.

        Binary search over perturbation magnitude.
        """
        lo, hi = 0.0, 10.0
        for _ in range(20):  # Binary search iterations
            mid = (lo + hi) / 2
            escapes = 0
            for _ in range(n_trials):
                _, converged, _ = self.measure_relaxation_time(
                    mid, dt, tolerance, max_steps)
                if not converged:
                    escapes += 1
            if escapes > n_trials * 0.5:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def measure_variance_evolution(self, n_samples=100, n_steps=1000, dt=0.01,
                                   perturbation=0.1):
        """
        Track variance of attractor-resident states over time.

        If the basin is deepening (crystallizing), variance should narrow.
        If drifting, mean should shift. If fixed, both stable.
        """
        if np.isscalar(self.equilibrium):
            states = self.equilibrium + perturbation * np.random.randn(n_samples)
        else:
            states = np.tile(self.equilibrium, (n_samples, 1)) + \
                     perturbation * np.random.randn(n_samples, *self.equilibrium.shape)

        variances = []
        means = []

        for step in range(n_steps):
            if np.isscalar(self.equilibrium):
                for i in range(n_samples):
                    arr = np.array([states[i]])
                    arr = self.redistribute(arr, dt)
                    states[i] = arr[0]
                variances.append(float(np.var(states)))
                means.append(float(np.mean(states)))
            else:
                for i in range(n_samples):
                    states[i] = self.redistribute(states[i], dt)
                variances.append(float(np.mean(np.var(states, axis=0))))
                means.append(float(np.mean(states)))

        return {
            'variances': variances,
            'means': means,
            'initial_variance': variances[0],
            'final_variance': variances[-1],
            'variance_ratio': variances[-1] / variances[0] if variances[0] > 0 else 0,
            'mean_drift': abs(means[-1] - means[0]),
            'crystallizing': variances[-1] < 0.1 * variances[0],
            'drifting': abs(means[-1] - means[0]) > 0.1 * abs(means[0]) if means[0] != 0 else False,
        }


# ============================================================
# SEC Complexification
# ============================================================
# SU(2) generators (Pauli matrices / 2)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex) / 2
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex) / 2

SU2_GENERATORS = [SIGMA_X, SIGMA_Y, SIGMA_Z]


def commutator(A, B):
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def complexify_generators(real_generators):
    """
    Complexify a set of real Lie algebra generators.

    For su(2) -> sl(2, C):
    Real generators: J_1, J_2, J_3 (rotations)
    Imaginary generators: K_i = i * J_i (boosts)

    The complexified algebra has dimension 2 * dim(real algebra).
    Rotations generate compact SU(2); boosts generate non-compact part.
    """
    imaginary_generators = [1j * g for g in real_generators]
    return real_generators + imaginary_generators


def sl2c_generators():
    """
    Standard sl(2,C) generators: rotations J_i and boosts K_i.

    J_i = sigma_i / 2 (rotation generators, from SU(2))
    K_i = i * sigma_i / 2 (boost generators, from complexification)
    """
    rotations = SU2_GENERATORS  # J_1, J_2, J_3
    boosts = [1j * g for g in SU2_GENERATORS]  # K_1, K_2, K_3
    return rotations, boosts


def verify_lie_algebra(generators, expected_structure_constants=None):
    """
    Verify that generators close under commutation.

    Check all [G_i, G_j] and express as linear combination of generators.
    Returns structure constants f^k_{ij} where [G_i, G_j] = sum_k f^k_{ij} G_k.
    """
    n = len(generators)
    dim = generators[0].shape[0]

    # Build matrix of generators for decomposition
    gen_flat = np.array([g.flatten() for g in generators])  # n x dim^2

    structure_constants = np.zeros((n, n, n), dtype=complex)
    closure_errors = []

    for i in range(n):
        for j in range(i + 1, n):
            comm = commutator(generators[i], generators[j])
            comm_flat = comm.flatten()

            # Decompose [G_i, G_j] in terms of generators
            # Solve: sum_k f^k * G_k = comm
            # Using least squares
            coeffs, residual, _, _ = np.linalg.lstsq(gen_flat.T, comm_flat, rcond=None)
            structure_constants[i, j, :] = coeffs
            structure_constants[j, i, :] = -coeffs  # Antisymmetry

            # Check closure: residual should be ~0
            reconstructed = sum(coeffs[k] * generators[k] for k in range(n))
            error = np.max(np.abs(comm - reconstructed))
            closure_errors.append(float(error))

    return {
        'structure_constants': structure_constants,
        'max_closure_error': max(closure_errors) if closure_errors else 0.0,
        'mean_closure_error': np.mean(closure_errors) if closure_errors else 0.0,
        'closes': max(closure_errors) < 1e-10 if closure_errors else True,
    }


def so31_from_sl2c():
    """
    Construct SO(3,1) generators from SL(2,C) via the standard isomorphism.

    The Lorentz group SO(3,1) has 6 generators:
    - 3 rotations J_i
    - 3 boosts K_i

    Commutation relations:
    [J_i, J_j] = i * epsilon_ijk * J_k       (rotation algebra)
    [K_i, K_j] = -i * epsilon_ijk * J_k      (boosts don't close — give rotations!)
    [J_i, K_j] = i * epsilon_ijk * K_k       (rotations rotate boosts)

    The minus sign in [K,K] = -iJ is the signature of non-compactness (Minkowski).
    If it were +iJ, we'd have SO(4) (compact, Euclidean).
    """
    rotations, boosts = sl2c_generators()

    # Verify key commutation relations
    results = {}

    # [J_1, J_2] should be i * J_3
    jj_comm = commutator(rotations[0], rotations[1])
    expected_jj = 1j * rotations[2]
    results['JJ_relation'] = float(np.max(np.abs(jj_comm - expected_jj)))

    # [K_1, K_2] should be -i * J_3 (the Minkowski signature!)
    kk_comm = commutator(boosts[0], boosts[1])
    expected_kk = -1j * rotations[2]
    results['KK_relation'] = float(np.max(np.abs(kk_comm - expected_kk)))

    # [J_1, K_2] should be i * K_3
    jk_comm = commutator(rotations[0], boosts[1])
    expected_jk = 1j * boosts[2]
    results['JK_relation'] = float(np.max(np.abs(jk_comm - expected_jk)))

    results['all_exact'] = all(v < 1e-14 for v in results.values() if isinstance(v, float))

    return rotations, boosts, results


def check_compactness(generators):
    """
    Check whether a Lie algebra is compact or non-compact.

    Compact: Killing form is negative-definite (all generators anti-Hermitian)
    Non-compact: Killing form is indefinite (some generators are Hermitian)

    For SU(2): all generators are anti-Hermitian (i * sigma/2 is anti-Hermitian) -> compact
    For SL(2,C): boost generators are Hermitian -> non-compact
    """
    hermitian_count = 0
    antihermitian_count = 0

    for g in generators:
        # Check if g is anti-Hermitian: g^dag = -g
        if np.allclose(g.conj().T, -g, atol=1e-12):
            antihermitian_count += 1
        # Check if g is Hermitian: g^dag = g
        elif np.allclose(g.conj().T, g, atol=1e-12):
            hermitian_count += 1

    is_compact = (hermitian_count == 0)
    return {
        'hermitian_generators': hermitian_count,
        'antihermitian_generators': antihermitian_count,
        'is_compact': is_compact,
        'signature': 'compact' if is_compact else 'non-compact',
    }


# ============================================================
# PAC Redistribution on Connection Graphs
# ============================================================
def redistribute_on_graph(adjacency, values, dt=0.01, conservation='pac'):
    """
    One step of PAC-conserving redistribution on a connection graph.

    Uses graph Laplacian diffusion: dv/dt = -L*v where L = D - A.
    This is the correct discrete diffusion that preserves non-negativity
    and monotonically increases entropy for small enough dt.

    PAC conservation: total value is preserved exactly (Laplacian has
    zero row-sum, so total is invariant).
    """
    total_before = np.sum(values)

    # Laplacian diffusion: L*v = D*v - A*v
    degrees = np.sum(adjacency, axis=1)
    laplacian_v = degrees * values - adjacency @ values
    new_values = values - dt * laplacian_v

    # PAC enforcement: correct any numerical drift
    if conservation == 'pac':
        total_after = np.sum(new_values)
        if abs(total_after) > 1e-15 and abs(total_before) > 1e-15:
            new_values *= total_before / total_after

    return new_values


def measure_entropy(values):
    """Shannon entropy of value distribution (normalized to probabilities)."""
    total = np.sum(np.abs(values))
    if total == 0:
        return 0.0
    probs = np.abs(values) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def redistribution_rate(adjacency, values, dt=0.01, steps=100):
    """
    Measure the rate at which entropy changes under redistribution.

    This is the "entropy as redistribution rate" from iddea.md Section 2.2(a).
    """
    entropies = [measure_entropy(values)]
    current = values.copy()

    for _ in range(steps):
        current = redistribute_on_graph(adjacency, current, dt)
        entropies.append(measure_entropy(current))

    # Rate = slope of entropy over time
    times = np.arange(len(entropies)) * dt
    if len(times) > 1:
        slope = np.polyfit(times, entropies, 1)[0]
    else:
        slope = 0.0

    return {
        'initial_entropy': entropies[0],
        'final_entropy': entropies[-1],
        'entropy_rate': float(slope),
        'monotonic_increase': all(entropies[i] <= entropies[i + 1] + 1e-12
                                  for i in range(len(entropies) - 1)),
        'entropies': entropies,
    }


# ============================================================
# Utility: Results Saving
# ============================================================
RESULTS_DIR = M12_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_m12_results(experiment_name, data):
    """Save experiment results as timestamped JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = RESULTS_DIR / f"{experiment_name}_{timestamp}.json"

    # Convert numpy types for JSON serialization
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
