"""
ade_cascade core -- Dynkin graph-distance coupling matrices.

The legacy cascade kernel exp(-|i-j| * cd) is the graph-distance kernel
exp(-d_G(i,j) * cd) on the A-family (path) Dynkin diagram, since the path
graph's shortest-path metric is |i-j|. This module generalizes the kernel
to any diagram: A/D/E via milestone12's DynkinDiagram, affine-A (cycle)
via milestone15's build_cycle.

Registration: journals/2026-07-17_ade-cascade-round1-preregistration.md.
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque

# core/coupling.py sits at experiments/studies/ade_cascade/core/, so the experiments root is
# FOUR levels up, and milestones live under experiments/milestones/. The old expression went
# three levels (landing on studies/) and looked for studies/milestone12 -- correct only before
# the August 2026 layer reorganization (MIGRATION.md). This module is imported by every script
# here, so the stale path made the whole study unrunnable.
_EXPERIMENTS = Path(__file__).resolve().parents[3]
_MILESTONES = _EXPERIMENTS / "milestones"
sys.path.insert(0, str(_MILESTONES / "milestone12" / "core"))
sys.path.insert(0, str(_MILESTONES / "milestone15" / "core"))
sys.path.insert(0, str(_MILESTONES / "milestone4" / "core"))

from connection_geometry import DynkinDiagram    # noqa: E402
from representative import build_cycle           # noqa: E402


def graph_distance_matrix(adjacency):
    """All-pairs shortest-path distances via BFS (unweighted graph)."""
    n = adjacency.shape[0]
    D = np.full((n, n), np.inf)
    for src in range(n):
        D[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in np.nonzero(adjacency[u])[0]:
                if D[src, v] == np.inf:
                    D[src, v] = D[src, u] + 1
                    q.append(v)
    if not np.isfinite(D).all():
        raise ValueError("adjacency is not connected")
    return D


def distance_kernel(adjacency, cd):
    """C[i,j] = exp(-d_G(i,j) * cd) -- the registered coupling map."""
    return np.exp(-graph_distance_matrix(adjacency) * cd)


def dynkin_coupling(family, rank, cd):
    """Coupling matrix for A_rank / D_rank / E_rank (canonical node order)."""
    return distance_kernel(DynkinDiagram(family, rank).adjacency, cd)


def affine_a_coupling(rank, cd):
    """Affine A~_rank = cycle on rank+1 nodes (milestone15 builder)."""
    return distance_kernel(build_cycle(rank + 1), cd)


def legacy_kernel(n_modes, cd):
    """The pre-refactor kernel, for the exp_00 identity gate."""
    C = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            C[i, j] = np.exp(-abs(i - j) * cd)
    return C
