"""
radiation_physics.py -- Shared infrastructure for Milestone R: Radiation as
Ledger Severance.

Sidecar milestone applying M6-M14 machinery to radiation physics. Central
thesis: radiation is PAC ledger severance. When a system emits, the
conservation ledger splits into two independent branches. Wavelength encodes
Fibonacci depth of the severed connection, discrete vs continuous spectrum
signals actualization vs degradation, and line width measures disequilibrium.

Extends M14's quantum_complement.py with radiation-specific functions:
- ledger_severance: remove vertex from PAC graph, compute energy cost
- severance_energy: energy from scope boundary count at Fibonacci depth
- scope_boundary_count: inverse -- count boundaries from observed energy
- discrete/continuous_severance_spectrum: settled vs unsettled ledger
- equilibration_energy: gamma emission from daughter relaxation
- fibonacci_wavelength: depth + boundary count -> wavelength
- ejection_probability: survival probability 1/phi per boundary
- pressure_vs_temperature: equilibrium-shift vs brute-force comparison
- line_width_from_disequilibrium: spectral line width from PAC perturbation
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# Import chain: M14 -> M13 -> M12 -> M11 -> M10 -> M9 -> M8
# ============================================================
MR_ROOT = Path(__file__).resolve().parent.parent
M14_ROOT = MR_ROOT.parent / "milestone14"
M11_ROOT = MR_ROOT.parent / "milestone11"
M6_ROOT = MR_ROOT.parent / "milestone6"

sys.path.insert(0, str(M14_ROOT / "core"))
from quantum_complement import (
    # DFT constants
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE, XI_PAC, PI, LN2,
    # Physical constants
    HBAR, C_LIGHT, K_BOLTZMANN,
    # Fibonacci
    fib, fibonacci_depth_coupling, DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    # ADE infrastructure
    DynkinDiagram, all_ade_diagrams,
    # Complement / orbits
    complement_spectrum, vertex_orbits,
    graph_automorphisms, orbit_hilbert_basis, born_probability,
    # PAC redistribution
    redistribute_on_graph, measure_entropy, BasinAttractor,
    # Utilities
    save_m14_results, _convert_numpy,
)

# M11 extras (StochasticCascade, Planck scale)
sys.path.insert(0, str(M11_ROOT / "core"))
from quantum_gravity import (
    StochasticCascade,
    E_MVAE, E_PLANCK_GEV, L_PLANCK_M, M_PLANCK_KG,
    M_PLANCK_GEV,
)

# M6 extras (scope infrastructure)
sys.path.insert(0, str(M6_ROOT / "core"))
from scope import pac_budget, scope_attenuation


# ============================================================
# Physical Constants -- Nuclear / Atomic / X-ray Data
# ============================================================

# Alpha decay energies (MeV) -- from NNDC
ALPHA_U238 = 4.270
ALPHA_TH232 = 4.082
ALPHA_RA226 = 4.871
ALPHA_PO210 = 5.407

# U-238 decay chain: all 8 alpha decays (MeV)
U238_CHAIN_ALPHAS = [4.270, 4.784, 4.871, 5.590, 6.115, 5.407, 7.883, 8.954]
U238_CHAIN_LABELS = [
    "U238->Th234", "U234->Th230", "Th230->Ra226", "Ra226->Rn222",
    "Rn222->Po218", "Po210->Pb206", "Bi212->Tl208", "Po212->Pb208",
]

# Half-lives (seconds) for Geiger-Nuttall analysis -- NNDC/IAEA
# Matched to U238_CHAIN_ALPHAS ordering (first 6 are U-238 chain proper,
# last 2 are Th-232 chain -- documented honestly)
U238_CHAIN_HALFLIVES_S = [
    1.4099e17,   # U238  -> Th234  (4.468 Gyr)
    7.7476e12,   # U234  -> Th230  (245.5 kyr)
    2.3794e12,   # Th230 -> Ra226  (75.38 kyr)
    5.0491e10,   # Ra226 -> Rn222  (1600 yr)
    3.3004e5,    # Rn222 -> Po218  (3.82 d)
    1.1955e7,    # Po210 -> Pb206  (138.4 d)
    1.8274e2,    # Bi212 -> Tl208  (3.05 min, alpha branch ~36%)
    2.99e-7,     # Po212 -> Pb208  (299 ns)
]

# Clean dataset for Geiger-Nuttall: (label, alpha_MeV, halflife_seconds)
# Even-even alpha emitters with well-measured values
GN_ALPHA_DATA = [
    ('U238',  4.270, 1.4099e17),
    ('U234',  4.784, 7.7476e12),
    ('Th230', 4.687, 2.3794e12),
    ('Ra226', 4.871, 5.0491e10),
    ('Rn222', 5.590, 3.3004e5),
    ('Po210', 5.407, 1.1955e7),
    ('Po214', 7.883, 1.6430e-4),
    ('Po212', 8.954, 2.99e-7),
]

# Beta decay endpoints (MeV)
BETA_C14 = 0.156
BETA_TRITIUM = 0.0186
BETA_CO60 = 0.3173

# Gamma lines (MeV)
GAMMA_CO60 = [1.173, 1.332]
GAMMA_CS137 = 0.662
# U-238 chain gammas (keV) -- select prominent lines
U238_CHAIN_GAMMAS_KEV = [46.5, 63.3, 92.4, 186.2, 295.2, 351.9]

# X-ray characteristic lines (keV)
CU_K_ALPHA = 8.048
MO_K_ALPHA = 17.479

# Hydrogen (eV)
RYDBERG_EV = 13.605693009  # CODATA 2018
LYMAN_ALPHA_EV = 10.2
BALMER_ALPHA_EV = 1.89

# Conversion factors
MEV_TO_JOULE = 1.602176634e-13
EV_TO_JOULE = 1.602176634e-19
KEV_TO_MEV = 1e-3
GEV_TO_MEV = 1e3
PLANCK_ENERGY_MEV = E_PLANCK_GEV * GEV_TO_MEV  # ~1.22e22 MeV

# Particle masses (PDG 2022, not DFT-derived)
M_ELECTRON_MEV = 0.51099895
M_PROTON_MEV = 938.27208816

# DFT-derived electromagnetic coupling (M6 formula, 5.7 ppm)
ALPHA_EM_DFT = 2.0 / (3.0 * PHI * 55.0) * (1.0 - 55.0 / (4.0 * PI * 169.0))


def dft_energy_scale(depth, mediator_mass_mev):
    """Coupling-anchored energy scale: alpha(d)^2 * m_mediator."""
    alpha_d = fibonacci_depth_coupling(depth)
    return alpha_d ** 2 * mediator_mass_mev


def coupling_boundary_count(energy_mev, depth, mediator_mass_mev):
    """Boundary count using coupling-anchored energy scale (replaces Planck scale)."""
    e_scale = dft_energy_scale(depth, mediator_mass_mev)
    if e_scale == 0:
        return float('inf')
    return energy_mev / (XI_BALANCE * e_scale)


# Results directory
RESULTS_DIR = MR_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def ade_graphs(max_rank=8):
    """Iterate over ADE diagrams as (name, adjacency) tuples."""
    for d in all_ade_diagrams(max_rank=max_rank):
        yield d.name, d.adjacency


# ============================================================
# Core Functions: Ledger Severance
# ============================================================

def ledger_severance(adjacency, vertex):
    """
    Perform a ledger severance: remove vertex from graph.

    Returns dict with:
    - daughter_adj: adjacency matrix of remaining graph
    - severed_connections: number of edges cut
    - spectral_shift: change in spectral energy (sum of eigenvalues^2)
    - disconnected: whether the graph split into components
    """
    n = adjacency.shape[0]
    A = adjacency.astype(float)

    # Eigenvalues before
    eigs_before = np.linalg.eigvalsh(A)
    spectral_energy_before = np.sum(eigs_before ** 2)

    # Count severed connections
    severed = int(np.sum(A[vertex, :] > 0))

    # Remove vertex (delete row and column)
    mask = np.ones(n, dtype=bool)
    mask[vertex] = False
    daughter_adj = A[np.ix_(mask, mask)]

    # Eigenvalues after
    if daughter_adj.shape[0] > 0:
        eigs_after = np.linalg.eigvalsh(daughter_adj)
        spectral_energy_after = np.sum(eigs_after ** 2)
    else:
        eigs_after = np.array([])
        spectral_energy_after = 0.0

    spectral_shift = spectral_energy_before - spectral_energy_after

    # Check connectivity of daughter
    disconnected = False
    if daughter_adj.shape[0] > 1:
        # BFS from vertex 0
        visited = {0}
        queue = [0]
        while queue:
            v = queue.pop(0)
            for u in range(daughter_adj.shape[0]):
                if daughter_adj[v, u] > 0 and u not in visited:
                    visited.add(u)
                    queue.append(u)
        disconnected = len(visited) < daughter_adj.shape[0]

    return {
        'daughter_adj': daughter_adj,
        'severed_connections': severed,
        'spectral_shift': float(spectral_shift),
        'spectral_energy_before': float(spectral_energy_before),
        'spectral_energy_after': float(spectral_energy_after),
        'eigs_before': eigs_before,
        'eigs_after': eigs_after,
        'disconnected': disconnected,
    }


def severance_energy(depth, n_boundaries=1):
    """
    Energy released by severance at Fibonacci depth d, crossing n scope
    boundaries.

    E = n_boundaries * Xi * E_scale(depth)
    E_scale(depth) = E_Planck * phi^(-depth)

    Returns energy in MeV.
    """
    e_scale = PLANCK_ENERGY_MEV * PHI ** (-depth)
    return n_boundaries * XI_BALANCE * e_scale


def scope_boundary_count(energy_mev, depth):
    """
    Given observed radiation energy and Fibonacci depth, how many scope
    boundaries were traversed?

    n = E / (Xi * E_scale(depth))

    Should be integer for discrete spectra (actualization) and non-integer
    for continuous spectra (degradation).
    """
    e_scale = PLANCK_ENERGY_MEV * PHI ** (-depth)
    if e_scale == 0:
        return float('inf')
    return energy_mev / (XI_BALANCE * e_scale)


def discrete_severance_spectrum(adjacency, depth):
    """
    Compute all structurally distinct severance energies for a graph.

    One energy per automorphism orbit: vertices in the same orbit produce
    identical severance energies (by symmetry). Returns dict mapping
    orbit_index -> severance_spectral_shift.
    """
    orbits = vertex_orbits(adjacency)
    spectrum = {}

    for i, orbit in enumerate(orbits):
        # Use first vertex in orbit as representative
        result = ledger_severance(adjacency, orbit[0])
        spectrum[i] = {
            'orbit_size': len(orbit),
            'spectral_shift': result['spectral_shift'],
            'severed_connections': result['severed_connections'],
            'disconnected': result['disconnected'],
        }

    return spectrum


def continuous_severance_spectrum(adjacency, depth, n_samples=10000, seed=42):
    """
    Model unsettled-ledger severance using StochasticCascade noise.

    The PAC ledger hasn't equilibrated before severance, so the energy
    is drawn from a continuous distribution. Uses Landauer noise from M11.
    """
    orbits = vertex_orbits(adjacency)
    # Base severance energies per orbit
    base_energies = []
    for orbit in orbits:
        result = ledger_severance(adjacency, orbit[0])
        base_energies.append(result['spectral_shift'])

    # Add stochastic noise
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_samples):
        # Pick a random orbit weighted by size
        orbit_sizes = [len(o) for o in orbits]
        total = sum(orbit_sizes)
        probs = [s / total for s in orbit_sizes]
        orbit_idx = rng.choice(len(orbits), p=probs)
        base = base_energies[orbit_idx]
        # Landauer noise: amplitude ~ Xi * base_energy
        noise = rng.randn() * XI_BALANCE * abs(base) * 0.1
        samples.append(abs(base + noise))

    return np.array(samples)


def equilibration_energy(adjacency, vertex):
    """
    Energy released when daughter graph (post-severance) relaxes to its
    new PAC ground state. Models gamma emission.

    Computed as: spectral energy of excited daughter minus spectral energy
    after redistribution to equilibrium.
    """
    result = ledger_severance(adjacency, vertex)
    daughter = result['daughter_adj']
    n = daughter.shape[0]
    if n < 2:
        return 0.0

    # Excited state: uniform PAC values
    state_excited = np.ones(n) / n

    # Redistribute to equilibrium (100 steps)
    state_eq = state_excited.copy()
    for _ in range(100):
        new_state = np.zeros(n)
        for v in range(n):
            neighbors = np.where(daughter[v] > 0)[0]
            if len(neighbors) > 0:
                # Share with neighbors, keep proportion
                share = state_eq[v] * INV_PHI / len(neighbors)
                new_state[v] += state_eq[v] * (1 - INV_PHI)
                for u in neighbors:
                    new_state[u] += share
            else:
                new_state[v] += state_eq[v]
        state_eq = new_state

    # Energy = variance reduction (excited is uniform, equilibrium has structure)
    entropy_excited = -np.sum(state_excited * np.log(state_excited + 1e-30))
    entropy_eq = -np.sum(state_eq * np.log(state_eq + 1e-30))

    return float(entropy_eq - entropy_excited)


# ============================================================
# Wavelength and Boundary Counting
# ============================================================

def fibonacci_wavelength(depth, n_boundaries=1):
    """
    Map Fibonacci depth + boundary count to radiation wavelength.

    lambda = h*c / E, where E = n * Xi * E_Planck * phi^(-depth)

    Returns wavelength in meters.
    """
    energy_j = n_boundaries * XI_BALANCE * E_PLANCK_GEV * 1e9 * EV_TO_JOULE * PHI ** (-depth)
    if energy_j <= 0:
        return float('inf')
    h = 2 * PI * HBAR
    return h * C_LIGHT / energy_j


# ============================================================
# Ejection and Efficiency
# ============================================================

def ejection_probability(n_boundaries, perturbation_ratio=1.0):
    """
    Probability of ejection as function of scope boundaries to cross.

    Survival per boundary = 1/phi (from M6 scope attenuation).
    P_eject = 1 - phi^(-n) when perturbation_ratio >= 1.
    Scaled by perturbation_ratio for sub-threshold perturbations.
    """
    survival = PHI ** (-n_boundaries)
    p_base = 1.0 - survival
    return min(1.0, p_base * perturbation_ratio)


def pressure_vs_temperature(depth, delta_equilibrium, temperature_ev=None):
    """
    Compare equilibrium-shift mechanism vs thermal brute-force.

    Equilibrium shift: change the PAC split ratio by delta_equilibrium,
    shifting the potential landscape so severance becomes spontaneous.
    Cost = delta_eq * E_scale(depth).

    Thermal: supply enough random energy to overcome the binding.
    Cost = Xi * E_scale(depth) (full boundary crossing cost).

    Returns dict with both costs and efficiency ratio.
    """
    e_scale_mev = PLANCK_ENERGY_MEV * PHI ** (-depth)

    # Equilibrium shift cost: partial potential landscape change
    e_shift = abs(delta_equilibrium) * e_scale_mev

    # Thermal brute-force cost: full Xi per boundary
    e_thermal = XI_BALANCE * e_scale_mev

    # If temperature provided, compare to kT
    kt_mev = None
    if temperature_ev is not None:
        kt_mev = temperature_ev * 1e-3  # eV to MeV (approximate)

    ratio = e_thermal / e_shift if e_shift > 0 else float('inf')

    return {
        'e_shift_mev': float(e_shift),
        'e_thermal_mev': float(e_thermal),
        'efficiency_ratio': float(ratio),
        'depth': depth,
        'delta_equilibrium': float(delta_equilibrium),
        'kt_mev': float(kt_mev) if kt_mev is not None else None,
    }


def line_width_from_disequilibrium(adjacency, vertex, disequilibrium_frac, n_trials=1000, seed=42):
    """
    Natural line width as function of how far from PAC equilibrium.

    A perfectly settled ledger gives zero width (delta function).
    Increasing disequilibrium broadens the line. Computed as variance
    of severance spectral shift under perturbation.
    """
    rng = np.random.RandomState(seed)
    n = adjacency.shape[0]
    shifts = []

    for _ in range(n_trials):
        # Perturb adjacency weights
        perturbed = adjacency.astype(float).copy()
        noise = rng.randn(n, n) * disequilibrium_frac
        noise = (noise + noise.T) / 2  # Keep symmetric
        np.fill_diagonal(noise, 0)
        perturbed = perturbed + perturbed * noise
        perturbed = np.maximum(perturbed, 0)  # No negative weights

        result = ledger_severance(perturbed, vertex)
        shifts.append(result['spectral_shift'])

    shifts = np.array(shifts)
    return {
        'mean': float(np.mean(shifts)),
        'std': float(np.std(shifts)),
        'variance': float(np.var(shifts)),
        'disequilibrium_frac': float(disequilibrium_frac),
        'n_trials': n_trials,
    }


# ============================================================
# Balance-Seeking Decay (exp_11)
# ============================================================

def pac_deficit(adjacency, vertex, perturbation=1.0):
    """
    PAC deficit: L2 distance from perturbed state to equilibrium.

    Measures how far from balance the graph is when a vertex is
    perturbed. This is the DIMENSIONLESS measure of imbalance --
    the key reframe: decay energy = deficit, not absolute Planck energy.

    Uses redistribute_on_graph (M12) to find equilibrium.
    """
    n = adjacency.shape[0]
    if n < 2:
        return 0.0

    # Equilibrium: uniform distribution (PAC conserved)
    eq_state = np.ones(n) / n

    # Perturbed state: inject extra value at target vertex
    state = eq_state.copy()
    state[vertex] += perturbation / n
    total = np.sum(state)
    state = state * (1.0 / total)  # Renormalize (PAC)

    # Find true equilibrium by running redistribution to convergence
    relaxed = state.copy()
    for _ in range(500):
        relaxed = redistribute_on_graph(adjacency, relaxed, dt=0.01)

    deficit = np.linalg.norm(state - relaxed, 2)
    return float(deficit)


def stochastic_balance_walk(adjacency, initial_state, noise_amplitude=0.01,
                            threshold=0.01, max_steps=10000, seed=42):
    """
    Stochastic balance-seeking walk on a graph.

    At each step:
    1. Redistribute toward equilibrium (Laplacian diffusion, PAC-conserving)
    2. Add Landauer noise (zero-mean, PAC-preserving)

    Returns first-passage time to reach within threshold of equilibrium.
    Models decay as balance-seeking: the system wanders through the PAC
    landscape until it reaches the balanced state.
    """
    rng = np.random.RandomState(seed)
    n = adjacency.shape[0]
    total_pac = np.sum(initial_state)

    # Find equilibrium (noise-free convergence)
    eq_state = initial_state.copy()
    for _ in range(1000):
        eq_state = redistribute_on_graph(adjacency, eq_state, dt=0.01)

    # Stochastic walk
    state = initial_state.copy()
    deficits = [float(np.linalg.norm(state - eq_state, 2))]

    for step in range(max_steps):
        # Deterministic: redistribute toward equilibrium
        state = redistribute_on_graph(adjacency, state, dt=0.01)

        # Stochastic: Landauer noise (zero-mean for PAC conservation)
        noise = rng.randn(n) * noise_amplitude * LN2
        noise -= np.mean(noise)  # Zero-sum preserves PAC
        state = state + noise
        state = np.maximum(state, 1e-30)  # Non-negative
        state = state * (total_pac / np.sum(state))  # PAC conservation

        deficit = float(np.linalg.norm(state - eq_state, 2))
        deficits.append(deficit)

        if deficit < threshold:
            return {
                'first_passage_time': step + 1,
                'converged': True,
                'deficits': np.array(deficits),
                'final_deficit': deficit,
            }

    return {
        'first_passage_time': max_steps,
        'converged': False,
        'deficits': np.array(deficits),
        'final_deficit': deficits[-1],
    }


def stochastic_barrier_walk(adjacency, target_vertex, initial_state,
                            noise_amplitude=0.01, max_steps=10000, seed=42):
    """
    Balance-seeking walk with a TOPOLOGICAL BARRIER at the target vertex.

    Severance requires ALL edges of target_vertex to be simultaneously
    decoupled: max(|state[v] - state[u]| for u in neighbors(v)) < threshold.
    The threshold is noise_amplitude * LN2 (Landauer scale).

    This models the PAC analog of the Coulomb barrier: for a vertex with
    degree d, all d connections must be weak at the same instant. The
    probability of d independent fluctuations coinciding scales as p^d,
    creating exponential suppression in degree.

    Returns first-passage time to the barrier crossing event.
    """
    rng = np.random.RandomState(seed)
    n = adjacency.shape[0]
    total_pac = np.sum(initial_state)
    degree = int(np.sum(adjacency[target_vertex] > 0))
    neighbors = np.where(adjacency[target_vertex] > 0)[0]

    # Severance threshold: Landauer scale per edge
    edge_threshold = noise_amplitude * LN2

    # Find equilibrium
    eq_state = initial_state.copy()
    for _ in range(1000):
        eq_state = redistribute_on_graph(adjacency, eq_state, dt=0.01)

    state = initial_state.copy()

    for step in range(max_steps):
        state = redistribute_on_graph(adjacency, state, dt=0.01)

        noise = rng.randn(n) * noise_amplitude * LN2
        noise -= np.mean(noise)
        state = state + noise
        state = np.maximum(state, 1e-30)
        state = state * (total_pac / np.sum(state))

        # Check barrier condition: ALL edges of target must be decoupled
        if len(neighbors) > 0:
            edge_flows = np.array([abs(state[target_vertex] - state[u])
                                   for u in neighbors])
            max_flow = np.max(edge_flows)
            if max_flow < edge_threshold:
                return {
                    'first_passage_time': step + 1,
                    'converged': True,
                    'degree': degree,
                    'max_flow_at_crossing': float(max_flow),
                    'threshold': float(edge_threshold),
                }

    return {
        'first_passage_time': max_steps,
        'converged': False,
        'degree': degree,
        'max_flow_at_crossing': float(max_flow) if len(neighbors) > 0 else 0.0,
        'threshold': float(edge_threshold),
    }


# ============================================================
# Perspectival Barrier (exp_13)
# ============================================================

def perspective_divergence(adjacency, vertex, horizon=2):
    """
    Jensen-Shannon divergence between a vertex's local random-walk
    distribution and the global PAC equilibrium.

    LOCAL: Start delta at vertex, apply random walk transition matrix
    P = D^{-1}A for `horizon` steps. This IS the vertex's local
    perspective -- where it "thinks" weight should flow.

    GLOBAL: Uniform 1/n (PAC equilibrium for connected graphs).

    Returns JSD in nats, bounded [0, ln(2)].
    """
    n = adjacency.shape[0]
    if n < 2:
        return 0.0

    # Random walk transition matrix P[i,j] = A[i,j] / degree(i)
    degrees = np.sum(adjacency > 0, axis=1).astype(float)
    degrees = np.maximum(degrees, 1e-30)
    P = adjacency / degrees[:, np.newaxis]

    # Local: delta at vertex, walk horizon steps
    local = np.zeros(n)
    local[vertex] = 1.0
    for _ in range(horizon):
        local = P.T @ local

    # Ensure valid probability distribution
    local = np.maximum(local, 1e-30)
    local = local / np.sum(local)

    # Global: uniform
    global_eq = np.ones(n) / n

    # Jensen-Shannon Divergence (symmetric, bounded [0, ln2])
    m = 0.5 * (local + global_eq)
    kl_local_m = np.sum(local * np.log(local / m))
    kl_global_m = np.sum(global_eq * np.log(global_eq / m))
    jsd = 0.5 * kl_local_m + 0.5 * kl_global_m

    return float(jsd)


def perspectival_barrier_walk(adjacency, target_vertex, initial_state,
                              horizon=2, noise_amplitude=0.01,
                              jsd_threshold=0.01, max_steps=10000, seed=42):
    """
    Balance-seeking walk with a PERSPECTIVAL BARRIER at the target vertex.

    At each step:
    1. Redistribute toward equilibrium (Laplacian diffusion)
    2. Add Landauer noise (PAC-preserving)
    3. Check barrier: JSD of current state within target's k-hop
       neighborhood vs uniform over that neighborhood. Severance
       happens when local and global perspectives reconcile (JSD < threshold).

    The barrier is the information-theoretic gap between how the target
    vertex's neighborhood "sees itself" and what global equilibrium says.
    """
    rng = np.random.RandomState(seed)
    n = adjacency.shape[0]
    total_pac = np.sum(initial_state)
    degree = int(np.sum(adjacency[target_vertex] > 0))

    # Precompute k-hop neighborhood (fixed for the walk)
    visited = {target_vertex}
    frontier = {target_vertex}
    for _ in range(horizon):
        new_frontier = set()
        for v in frontier:
            for u in range(n):
                if adjacency[v, u] > 0 and u not in visited:
                    new_frontier.add(u)
                    visited.add(u)
        frontier = new_frontier
    nbhd = sorted(visited)
    nbhd_size = len(nbhd)

    def local_jsd(state):
        """JSD of state within neighborhood vs uniform within neighborhood."""
        if nbhd_size < 2:
            return 0.0
        vals = state[nbhd]
        vals = np.maximum(vals, 1e-30)
        vals = vals / np.sum(vals)
        eq = np.ones(nbhd_size) / nbhd_size
        m = 0.5 * (vals + eq)
        kl_l = np.sum(vals * np.log(vals / m))
        kl_g = np.sum(eq * np.log(eq / m))
        return float(0.5 * kl_l + 0.5 * kl_g)

    state = initial_state.copy()
    initial_jsd = local_jsd(state)

    for step in range(max_steps):
        # Deterministic: redistribute toward equilibrium
        state = redistribute_on_graph(adjacency, state, dt=0.01)

        # Stochastic: Landauer noise (zero-mean for PAC conservation)
        noise = rng.randn(n) * noise_amplitude * LN2
        noise -= np.mean(noise)
        state = state + noise
        state = np.maximum(state, 1e-30)
        state = state * (total_pac / np.sum(state))

        # Check perspectival barrier
        current_jsd = local_jsd(state)
        if current_jsd < jsd_threshold:
            return {
                'first_passage_time': step + 1,
                'converged': True,
                'jsd_at_crossing': current_jsd,
                'degree': degree,
                'perspective_divergence_initial': initial_jsd,
            }

    return {
        'first_passage_time': max_steps,
        'converged': False,
        'jsd_at_crossing': local_jsd(state),
        'degree': degree,
        'perspective_divergence_initial': initial_jsd,
    }


# ============================================================
# Stress Barrier (exp_15)
# ============================================================

def stress_barrier_walk(adjacency, target_vertex, initial_state,
                        stress_threshold=0.05, noise_amplitude=0.01,
                        max_steps=10000, seed=42):
    """
    Balance-seeking walk with a STRESS BARRIER at the target vertex.

    Severance requires ALL d edges of target_vertex to be simultaneously
    OVERSTRESSED: |state[v] - state[u]| > stress_threshold for all neighbors u.

    This is the INVERSE of stochastic_barrier_walk (which checks for
    relaxation below threshold). The physics: connections break when
    overstressed, not when relaxed. Higher noise (SEC flux) -> larger
    fluctuations -> easier to overstress all edges -> shorter FPT.

    The stress_threshold is FIXED (not noise-dependent), so noise and
    barrier are independent -- unlike the topological barrier where
    edge_threshold = noise * LN2.
    """
    rng = np.random.RandomState(seed)
    n = adjacency.shape[0]
    total_pac = np.sum(initial_state)
    degree = int(np.sum(adjacency[target_vertex] > 0))
    neighbors = np.where(adjacency[target_vertex] > 0)[0]

    state = initial_state.copy()

    for step in range(max_steps):
        # Deterministic: redistribute toward equilibrium
        state = redistribute_on_graph(adjacency, state, dt=0.01)

        # Stochastic: Landauer noise (zero-mean for PAC conservation)
        noise = rng.randn(n) * noise_amplitude * LN2
        noise -= np.mean(noise)
        state = state + noise
        state = np.maximum(state, 1e-30)
        state = state * (total_pac / np.sum(state))

        # Check STRESS barrier: ALL edges must be overstressed
        if len(neighbors) > 0:
            edge_flows = np.array([abs(state[target_vertex] - state[u])
                                   for u in neighbors])
            min_flow = np.min(edge_flows)
            if min_flow > stress_threshold:
                return {
                    'first_passage_time': step + 1,
                    'converged': True,
                    'degree': degree,
                    'min_flow_at_crossing': float(min_flow),
                    'threshold': float(stress_threshold),
                }

    return {
        'first_passage_time': max_steps,
        'converged': False,
        'degree': degree,
        'min_flow_at_crossing': float(np.min(
            [abs(state[target_vertex] - state[u]) for u in neighbors]
        )) if len(neighbors) > 0 else 0.0,
        'threshold': float(stress_threshold),
    }


# ============================================================
# PAC Tree Utilities
# ============================================================

def build_pac_tree(depth):
    """
    Build a balanced binary PAC tree as adjacency matrix.

    Depth d gives 2^(d+1) - 1 nodes. Root at index 0, children at
    2*i+1 and 2*i+2.
    """
    n = 2 ** (depth + 1) - 1
    A = np.zeros((n, n))
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            A[i, left] = 1.0
            A[left, i] = 1.0
        if right < n:
            A[i, right] = 1.0
            A[right, i] = 1.0
    return A


def pac_tree_values(depth):
    """
    Compute PAC values on a binary tree: root = 1, each child gets
    parent * INV_PHI.
    """
    n = 2 ** (depth + 1) - 1
    values = np.zeros(n)
    values[0] = 1.0
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            values[left] = values[i] * INV_PHI
        if right < n:
            values[right] = values[i] * INV_PHI
    return values


# ============================================================
# Result Saving
# ============================================================

def save_mr_results(data, experiment_name):
    """Save experiment results as timestamped JSON to milestone-r results dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    converted = _convert_numpy(data)
    with open(filepath, 'w') as f:
        json.dump(converted, f, indent=2, default=str)
    print(f"  Results saved: {filepath.name}")
    return filepath
