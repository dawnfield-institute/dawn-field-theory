"""
M15 core -- class/representative machinery.

Builds on milestone13's identity_complement (complement spectra, orbits,
deformation). Adds:
  - ADE + affine-A + unicyclic graph builders
  - the complement-eigenvector CONNECTION (Procrustes transport over shared
    support) and its cycle holonomy -- the genuinely non-exact object.
    (Scalar/vector spectral differences are potentials -> exact -> zero
    holonomy identically; M15 founding journal, exp_01 registration.)

Gauge note: each vertex's frame is the eigenvector matrix of its complement
subgraph, computed once (deterministic eigh). Holonomy is defined up to
conjugation by the start vertex's frame; its eigenvalue ANGLES and the
Frobenius deficit ||H - I|| are conjugation-invariant up to the registered
tolerance, and labeling-invariance is tested explicitly, not assumed.
"""

import sys
import numpy as np
from pathlib import Path

_M13_CORE = Path(__file__).resolve().parent.parent.parent / "milestone13" / "core"
sys.path.insert(0, str(_M13_CORE))
from identity_complement import (   # noqa: E402
    PHI, INV_PHI, LN_PHI,
    complement_spectrum, vertex_orbits,
    complement_deformation_rate, max_deformation_rate,
    find_shortest_path, _convert_numpy,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_m15_results(experiment_name, data):
    import json
    from datetime import datetime
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{experiment_name}_{ts}.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")
    return out


# ============================================================
# Graph builders
# ============================================================

def build_path(n):
    """A_n Dynkin diagram: path on n vertices."""
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def build_d(n):
    """D_n: path on n-1 vertices with an extra leaf on vertex 1."""
    a = np.zeros((n, n))
    for i in range(n - 2):
        a[i, i + 1] = a[i + 1, i] = 1.0
    a[1, n - 1] = a[n - 1, 1] = 1.0
    return a


def build_cycle(m):
    """Affine A_{m-1} extended Dynkin diagram: cycle on m vertices."""
    a = np.zeros((m, m))
    for i in range(m):
        a[i, (i + 1) % m] = a[(i + 1) % m, i] = 1.0
    return a


def build_tadpole(cycle_len, tail_len):
    """Cycle with a path tail attached (unicyclic, non-transitive)."""
    m = cycle_len + tail_len
    a = np.zeros((m, m))
    for i in range(cycle_len):
        a[i, (i + 1) % cycle_len] = a[(i + 1) % cycle_len, i] = 1.0
    prev = 0
    for t in range(tail_len):
        j = cycle_len + t
        a[prev, j] = a[j, prev] = 1.0
        prev = j
    return a


def random_unicyclic(n, rng):
    """Random connected unicyclic graph on n vertices (tree + one extra edge)."""
    a = np.zeros((n, n))
    nodes = list(rng.permutation(n))
    for i in range(1, n):                      # random tree (random attachment)
        j = nodes[rng.randint(0, i)]
        a[nodes[i], j] = a[j, nodes[i]] = 1.0
    while True:                                # add one non-edge -> single cycle
        u, v = rng.randint(0, n), rng.randint(0, n)
        if u != v and a[u, v] == 0:
            a[u, v] = a[v, u] = 1.0
            return a


def cycle_basis_single(adjacency):
    """For a unicyclic graph, return the unique cycle as a vertex list."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(int)
    a = adjacency.copy()
    # iteratively strip leaves
    changed = True
    alive = set(range(n))
    while changed:
        changed = False
        for v in list(alive):
            if a[v].sum() == 1:
                u = int(np.argmax(a[v]))
                a[v, u] = a[u, v] = 0
                alive.discard(v)
                changed = True
    cyc_nodes = sorted(alive)
    # order the cycle by walking
    start = cyc_nodes[0]
    cycle = [start]
    prev, cur = None, start
    while True:
        nbrs = [j for j in np.nonzero(a[cur])[0] if j != prev]
        nxt = int(nbrs[0])
        if nxt == start:
            break
        cycle.append(nxt)
        prev, cur = cur, nxt
    return cycle


# ============================================================
# The complement-eigenvector connection
# ============================================================

def complement_frame(adjacency, vertex):
    """Eigen-decomposition of the complement subgraph G \\ vertex.

    Returns (eigvals ascending, eigvecs columns, kept_vertices list)."""
    n = adjacency.shape[0]
    keep = [i for i in range(n) if i != vertex]
    sub = adjacency[np.ix_(keep, keep)]
    vals, vecs = np.linalg.eigh(sub)
    return vals, vecs, keep


def edge_transport(adjacency, u, v, k, frames=None):
    """Orthogonal transport (Procrustes) from u's complement frame to v's,
    over the shared support V \\ {u, v}, using the top-k eigenvectors
    (largest eigenvalues). Returns (T [k x k orthogonal], min_eigengap)."""
    if frames is None:
        frames = {}
    for w in (u, v):
        if w not in frames:
            frames[w] = complement_frame(adjacency, w)
    vals_u, vecs_u, keep_u = frames[u]
    vals_v, vecs_v, keep_v = frames[v]
    common = [w for w in keep_u if w != v]      # = V \ {u, v}
    rows_u = [keep_u.index(w) for w in common]
    rows_v = [keep_v.index(w) for w in common]
    Vu = vecs_u[rows_u, :][:, -k:]              # top-k by eigenvalue
    Vv = vecs_v[rows_v, :][:, -k:]
    gap_u = float(vals_u[-k] - vals_u[-k - 1]) if len(vals_u) > k else np.inf
    gap_v = float(vals_v[-k] - vals_v[-k - 1]) if len(vals_v) > k else np.inf
    M = Vv.T @ Vu
    U, _, Wt = np.linalg.svd(M)
    T = U @ Wt                                  # orthogonal k x k
    return T, min(gap_u, gap_v)


def cycle_holonomy(adjacency, cycle, k):
    """Holonomy of the connection around an ordered vertex cycle.

    Returns dict: deficit ||H - I||_F, sorted |rotation angles| (conjugation
    invariants), min eigengap encountered (degeneracy guard)."""
    frames = {}
    H = np.eye(k)
    min_gap = np.inf
    m = len(cycle)
    for i in range(m):
        u, v = cycle[i], cycle[(i + 1) % m]
        T, gap = edge_transport(adjacency, u, v, k, frames)
        min_gap = min(min_gap, gap)
        H = T @ H
    eig = np.linalg.eigvals(H)
    angles = np.sort(np.abs(np.angle(eig)))
    deficit = float(np.linalg.norm(H - np.eye(k)))
    return {'deficit': deficit,
            'angles': [float(a) for a in angles],
            'det': float(np.linalg.det(H)),
            'min_eigengap': float(min_gap)}


def relabeled(adjacency, perm):
    """Apply vertex permutation: perm[i] = new label of old vertex i."""
    n = adjacency.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        P[perm[i], i] = 1.0
    return P @ adjacency @ P.T


# ============================================================
# Scalar potential (the exact part -- for exp_01 T1)
# ============================================================

def spectral_potential(adjacency, vertex):
    """g(v) = sum of complement spectrum -- a vertex potential. Signed edge
    differences of g are exact by construction (telescoping)."""
    return float(np.sum(complement_spectrum(adjacency, vertex)))
