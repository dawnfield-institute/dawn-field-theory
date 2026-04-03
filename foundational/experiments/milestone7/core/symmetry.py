"""
symmetry.py -- Shared infrastructure for Milestone 7: The Symmetry Primitive.

Provides:
- Constants: PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE
- Self-referential map families and non-self-referential baselines
- Symmetry metrics (global, local, spectral gap)
- Lattice/graph builders
- Result saving utilities
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import json
from datetime import datetime
from pathlib import Path


# ============================================================
# Constants
# ============================================================
PHI = (1 + np.sqrt(5)) / 2          # 1.6180339887...
INV_PHI = 1 / PHI                    # 0.6180339887...
LN_PHI = np.log(PHI)                 # 0.4812118250...
GAMMA_EM = 0.5772156649015329        # Euler-Mascheroni
XI_BALANCE = GAMMA_EM + LN_PHI       # 1.0584274899...
PI = np.pi


# ============================================================
# Self-referential map families
# ============================================================

def get_self_referential_maps():
    """
    Return a list of (name, function) pairs where each function f(x)
    is self-referential: the map's structure encodes x relating to itself.

    Classification criterion (syntactic, defined BEFORE running):
    A map f(x) is self-referential if it can be written as x = g(x)
    for some g, meaning x appears on both sides of a fixed-point equation
    where the RHS explicitly depends on x in a non-trivial way.

    These are defined by their FIXED-POINT EQUATION, not by f(x) directly.
    The iteration is x_{n+1} = f(x_n).
    """
    maps = [
        # Classic self-referential: x = 1 + 1/x -> phi
        ("1+1/x", lambda x: 1 + 1/x if abs(x) > 1e-15 else 1e15),
        # x = sqrt(1+x) -> phi
        ("sqrt(1+x)", lambda x: np.sqrt(max(1 + x, 0))),
        # x = (x+1)/x = 1 + 1/x (same fixed point, different approach)
        ("(x^2+1)/(x+1)", lambda x: (x**2 + 1) / (x + 1) if abs(x + 1) > 1e-15 else 1e15),
        # x = x/(x-1) + 1/(x+1)
        ("x/(x-1)+1/(x+1)", lambda x: x / (x - 1) + 1 / (x + 1) if abs(x - 1) > 1e-15 and abs(x + 1) > 1e-15 else x),
        # x = (x^2 + 2) / (2x + 1)
        ("(x^2+2)/(2x+1)", lambda x: (x**2 + 2) / (2 * x + 1) if abs(2 * x + 1) > 1e-15 else x),
        # x = sqrt(x + sqrt(x))
        ("sqrt(x+sqrt(x))", lambda x: np.sqrt(max(x + np.sqrt(max(x, 0)), 0))),
        # x = (1 + sqrt(1 + 4x)) / 2
        ("(1+sqrt(1+4x))/2", lambda x: (1 + np.sqrt(max(1 + 4 * x, 0))) / 2),
        # x = cbrt(1 + x^2)
        ("cbrt(1+x^2)", lambda x: np.cbrt(1 + x**2)),
        # x = 1 + 1/(1+x)
        ("1+1/(1+x)", lambda x: 1 + 1 / (1 + x) if abs(1 + x) > 1e-15 else 1e15),
        # x = exp(1/x) / e  (transcendental self-ref)
        ("exp(1/x)/e", lambda x: np.exp(1 / x) / np.e if abs(x) > 1e-15 else 1e15),
        # x = (x^3 + 2) / (x^2 + x + 1)
        ("(x^3+2)/(x^2+x+1)", lambda x: (x**3 + 2) / (x**2 + x + 1) if abs(x**2 + x + 1) > 1e-15 else x),
        # x = log(1 + e^x) / x  (softplus self-ref)
        ("log(1+exp(x))/x", lambda x: np.log1p(np.exp(min(x, 50))) / x if abs(x) > 1e-15 else 1.0),
        # x = (x + 2/x) / 2  (Babylonian for sqrt(2))
        ("(x+2/x)/2", lambda x: (x + 2 / x) / 2 if abs(x) > 1e-15 else 1e15),
        # x = (x + 3/x) / 2  (Babylonian for sqrt(3))
        ("(x+3/x)/2", lambda x: (x + 3 / x) / 2 if abs(x) > 1e-15 else 1e15),
        # x = cos(1/x) + sin(x)/x  (oscillatory self-ref)
        ("cos(1/x)+sin(x)/x", lambda x: np.cos(1 / x) + np.sin(x) / x if abs(x) > 1e-15 else 1.0),
        # x = (2x + 1) / (x + 2)
        ("(2x+1)/(x+2)", lambda x: (2 * x + 1) / (x + 2) if abs(x + 2) > 1e-15 else x),
        # x = x^(1/(1+x))
        ("x^(1/(1+x))", lambda x: x ** (1 / (1 + x)) if x > 0 and abs(1 + x) > 1e-15 else max(x, 1e-15)),
        # x = (x^2 + x + 1) / (x^2 + 1)
        ("(x^2+x+1)/(x^2+1)", lambda x: (x**2 + x + 1) / (x**2 + 1)),
        # x = tanh(x) + 1  (sigmoid self-ref)
        ("tanh(x)+1", lambda x: np.tanh(x) + 1),
        # x = 1/(1 + 1/(1+x))  continued fraction
        ("1/(1+1/(1+x))", lambda x: 1 / (1 + 1 / (1 + x)) if abs(1 + x) > 1e-15 else 0.5),
        # Higher-order self-ref: x = (x^4 + 1)/(x^3 + x)
        ("(x^4+1)/(x^3+x)", lambda x: (x**4 + 1) / (x**3 + x) if abs(x**3 + x) > 1e-15 else x),
        # x = 2 - 1/(x+1)
        ("2-1/(x+1)", lambda x: 2 - 1 / (x + 1) if abs(x + 1) > 1e-15 else 2.0),
        # x = (3x + 1) / (x + 3)
        ("(3x+1)/(x+3)", lambda x: (3 * x + 1) / (x + 3) if abs(x + 3) > 1e-15 else x),
        # x = sqrt(2 + sqrt(x))
        ("sqrt(2+sqrt(x))", lambda x: np.sqrt(2 + np.sqrt(max(x, 0)))),
        # x = 1 + x/(x^2 + 1)
        ("1+x/(x^2+1)", lambda x: 1 + x / (x**2 + 1)),
    ]
    return maps


def get_non_self_referential_maps():
    """
    Return a list of (name, function) pairs where each function f(x)
    is NOT self-referential: standard polynomial/transcendental maps
    that don't encode x = g(x) identity structure.

    These have fixed points but the map doesn't define x in terms of itself.
    They are contractive maps toward specific values.
    """
    maps = [
        ("2x+3 mod 10", lambda x: (2 * x + 3) % 10),
        ("sin(x)+1", lambda x: np.sin(x) + 1),
        ("cos(x)", lambda x: np.cos(x)),
        ("0.5*x+1", lambda x: 0.5 * x + 1),  # fixed point at 2
        ("0.3*x+2.1", lambda x: 0.3 * x + 2.1),  # fixed point at 3
        ("exp(-x)", lambda x: np.exp(-min(x, 50))),
        ("exp(-x^2)+0.5", lambda x: np.exp(-x**2) + 0.5),
        ("atan(x)+1", lambda x: np.arctan(x) + 1),
        ("tanh(2x)", lambda x: np.tanh(2 * x)),
        ("0.7*sin(x)+0.8", lambda x: 0.7 * np.sin(x) + 0.8),
        ("x/3+1.5", lambda x: x / 3 + 1.5),  # fixed point at 2.25
        ("0.4*x+0.9", lambda x: 0.4 * x + 0.9),  # fixed point at 1.5
        ("cos(x)+0.5", lambda x: np.cos(x) + 0.5),
        ("0.6*x+sqrt(2)/2", lambda x: 0.6 * x + np.sqrt(2) / 2),
        ("sin(2x)/2+1", lambda x: np.sin(2 * x) / 2 + 1),
        ("atan(x/2)+0.7", lambda x: np.arctan(x / 2) + 0.7),
        ("0.8*cos(x)+0.3", lambda x: 0.8 * np.cos(x) + 0.3),
        ("exp(-abs(x))+0.2", lambda x: np.exp(-abs(x)) + 0.2),
        ("0.25*x+1.875", lambda x: 0.25 * x + 1.875),  # fixed point at 2.5
        ("sin(x)*cos(x)+1", lambda x: np.sin(x) * np.cos(x) + 1),
        ("0.5*atan(x)+1.2", lambda x: 0.5 * np.arctan(x) + 1.2),
        ("tanh(x)+0.5", lambda x: np.tanh(x) + 0.5),
        ("0.9*cos(x/2)", lambda x: 0.9 * np.cos(x / 2)),
        ("1/(1+exp(-x))", lambda x: 1 / (1 + np.exp(-min(x, 50)))),  # sigmoid -> 0 or 1
        ("0.35*x+1.3", lambda x: 0.35 * x + 1.3),  # fixed point at 2.0
    ]
    return maps


PHI_FAMILY = [PHI, INV_PHI, PHI**2, PHI**3, 1/PHI**2, np.sqrt(PHI), PHI**4, 1/PHI**3]


def is_phi_related(value, tol=0.01):
    """Check if a value is within tol (relative) of any phi family member."""
    for pf in PHI_FAMILY:
        if abs(pf) > 1e-15 and abs(value - pf) / abs(pf) < tol:
            return True
    return False


def iterate_map(f, x0, n_iter=500, tol=1e-10):
    """
    Iterate x_{n+1} = f(x_n) from x0.
    Returns (converged, fixed_point, trajectory).
    """
    x = x0
    trajectory = [x]
    for i in range(n_iter):
        try:
            x_new = f(x)
            if not np.isfinite(x_new) or abs(x_new) > 1e15:
                return False, np.nan, trajectory
            trajectory.append(x_new)
            if abs(x_new - x) < tol:
                return True, x_new, trajectory
            x = x_new
        except (ZeroDivisionError, ValueError, OverflowError):
            return False, np.nan, trajectory
    # Check if last few values are oscillating stably
    if len(trajectory) > 10:
        tail = trajectory[-10:]
        if np.std(tail) < 0.01:
            return True, np.mean(tail), trajectory
    return False, trajectory[-1] if trajectory else np.nan, trajectory


# ============================================================
# Symmetry metrics
# ============================================================

def global_symmetry_spectral(L, state):
    """
    Global symmetry via spectral gap of state-weighted Laplacian.
    S_global = lambda_1 / lambda_max (algebraic connectivity / max eigenvalue).
    High S_global = well-connected, balanced state.
    """
    n = L.shape[0]
    if n < 3:
        return 1.0

    # Weight Laplacian by state
    if sparse.issparse(L):
        L_dense = L.toarray()
    else:
        L_dense = np.array(L, dtype=float)

    eigs = np.linalg.eigvalsh(L_dense)
    eigs = np.sort(eigs)

    # lambda_1 = smallest nonzero eigenvalue (Fiedler)
    lambda_1 = eigs[1] if len(eigs) > 1 and eigs[1] > 1e-10 else 0.0
    lambda_max = eigs[-1] if eigs[-1] > 1e-10 else 1.0

    return lambda_1 / lambda_max


def local_asymmetry(state, adj_matrix):
    """
    Local asymmetry: mean |node_value - mean(neighbor_values)| / std(all_values).
    High = locally asymmetric.
    """
    n = len(state)
    std_all = np.std(state)
    if std_all < 1e-15:
        return 0.0

    if sparse.issparse(adj_matrix):
        A = adj_matrix.toarray()
    else:
        A = np.array(adj_matrix, dtype=float)

    diffs = []
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) > 0:
            mean_neighbors = np.mean(state[neighbors])
            diffs.append(abs(state[i] - mean_neighbors))
        else:
            diffs.append(0.0)

    return np.mean(diffs) / std_all


def mutual_information_halves(state, adj_matrix):
    """
    Mutual information between two halves of the system.
    Uses spectral bisection for the partition.
    """
    n = len(state)
    if n < 4:
        return 0.0

    if sparse.issparse(adj_matrix):
        L = sparse.csgraph.laplacian(adj_matrix)
        L_dense = L.toarray()
    else:
        D = np.diag(np.sum(adj_matrix, axis=1))
        L_dense = D - adj_matrix

    eigs, vecs = np.linalg.eigh(L_dense)
    fiedler = vecs[:, 1]
    half1 = np.where(fiedler >= 0)[0]
    half2 = np.where(fiedler < 0)[0]

    if len(half1) < 2 or len(half2) < 2:
        return 0.0

    # Discretize states into bins
    n_bins = max(5, int(np.sqrt(n)))
    bins = np.linspace(state.min() - 1e-10, state.max() + 1e-10, n_bins + 1)
    h1 = np.histogram(state[half1], bins=bins)[0].astype(float) + 1e-10
    h2 = np.histogram(state[half2], bins=bins)[0].astype(float) + 1e-10
    h1 /= h1.sum()
    h2 /= h2.sum()

    # Joint histogram
    h12 = np.histogram2d(state[half1][:min(len(half1), len(half2))],
                         state[half2][:min(len(half1), len(half2))],
                         bins=[bins, bins])[0].astype(float) + 1e-10
    h12 /= h12.sum()

    mi = np.sum(h12 * np.log(h12 / (h1[:, None] * h2[None, :])))
    return max(mi, 0.0)


# ============================================================
# Lattice/graph builders
# ============================================================

def build_ring(n):
    """1D ring with periodic BC. Returns adjacency matrix (sparse)."""
    row = list(range(n)) + list(range(n))
    col = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    data = [1.0] * (2 * n)
    return sparse.csr_matrix((data, (row, col)), shape=(n, n))


def build_torus(nx, ny):
    """2D torus with periodic BC. Returns adjacency matrix (sparse)."""
    n = nx * ny
    row, col, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            # Right neighbor
            r = i * ny + (j + 1) % ny
            row.extend([idx, r]); col.extend([r, idx]); data.extend([1.0, 1.0])
            # Down neighbor
            d = ((i + 1) % nx) * ny + j
            row.extend([idx, d]); col.extend([d, idx]); data.extend([1.0, 1.0])
    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    A.data[:] = 1.0  # Remove duplicates
    return A


def build_cubic(n_side):
    """3D cubic lattice with periodic BC. Returns adjacency matrix (sparse)."""
    n = n_side ** 3
    row, col, data = [], [], []
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                idx = x * n_side**2 + y * n_side + z
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx_ = (x + dx) % n_side
                    ny_ = (y + dy) % n_side
                    nz_ = (z + dz) % n_side
                    nidx = nx_ * n_side**2 + ny_ * n_side + nz_
                    row.append(idx); col.append(nidx); data.append(1.0)
    return sparse.csr_matrix((data, (row, col)), shape=(n, n))


def build_random_regular(n, k=4, seed=42):
    """Random k-regular graph. Returns adjacency matrix (sparse)."""
    rng = np.random.RandomState(seed)
    # Simple approach: start with ring, add random edges
    A = build_ring(n).toarray()
    edges_needed = (k - 2) * n // 2
    added = 0
    attempts = 0
    while added < edges_needed and attempts < edges_needed * 100:
        i, j = rng.randint(0, n, size=2)
        if i != j and A[i, j] == 0 and np.sum(A[i]) < k and np.sum(A[j]) < k:
            A[i, j] = A[j, i] = 1.0
            added += 1
        attempts += 1
    return sparse.csr_matrix(A)


def graph_laplacian(A):
    """Compute graph Laplacian L = D - A from adjacency matrix."""
    if sparse.issparse(A):
        d = np.array(A.sum(axis=1)).flatten()
        D = sparse.diags(d)
        return D - A
    else:
        d = np.sum(A, axis=1)
        return np.diag(d) - A


# ============================================================
# Result saving
# ============================================================

def save_results(results_dict, experiment_name, results_dir):
    """Save results to JSON with timestamp."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = results_dir / f"{experiment_name}_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    return outpath
