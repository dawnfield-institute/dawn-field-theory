"""
exp_03 -- The Representative Gauge  [DEFERRED -- see pre-registration journal]

Milestone 15 (The Representative Problem)

STATUS: DEFERRED at pre-registration. The smoke test showed this minimal harness
does not reproduce M14 exp_06's null (evolved V(0) = 0.32; the M14 null is about
STATIC cross-terms with disjoint orbit supports). The registered design requires a
faithful reimplementation of M14's measurement construction. Kept as documentation
of the wrinkle; do not run for scored results.

PRE-REGISTERED: journals/2026-06-11_m15-exp01-03-preregistration.md (same commit).
Re-poses M14 exp_06 (vertex-space interference = machine zero, 1/4): a position
representative of the (passing) orbit-interference class requires Aut-breaking
frame data. Visibility should scale with the breaking and vanish in the
symmetric limit -- re-deriving the M14 failure as a limit statement.

Setup: D_4 (hub 0, leaves 1,2,3). Orbit superposition with SEC phase theta;
Hamiltonian = graph Laplacian of the (perturbed) adjacency; perturbation eps
added to ONE leaf edge weight (3 orbit-equivalent choices). Vertex visibility
V(eps) = max-min over theta of a leaf-vertex probability after evolution.

Tests:
  T1: V(0) <= 1e-10 AND the two smallest nonzero eps give the two smallest
      nonzero V (limit -> 0)
  T2: V(eps) monotone increasing (Spearman rho >= 0.9)
  T3: small-eps scaling exponent p equal across the 3 orbit-equivalent edges
      (CV(p) < 0.1); p reported [D]

Outputs: results/exp_03_representative_gauge_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from representative import save_m15_results, _convert_numpy

EPS_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
THETAS = np.linspace(0, 2 * np.pi, 25)
T_EVOLVE = 1.0


def d4_adjacency(eps=0.0, leaf=1):
    a = np.zeros((4, 4))
    for l in (1, 2, 3):
        a[0, l] = a[l, 0] = 1.0
    a[0, leaf] += eps
    a[leaf, 0] += eps
    return a


def orbit_superposition(theta):
    """|psi> = (|O_hub> + e^{i theta} |O_leaves>)/sqrt(2) in vertex space."""
    hub = np.array([1.0, 0, 0, 0], dtype=complex)
    leaves = np.array([0, 1.0, 1.0, 1.0], dtype=complex) / np.sqrt(3)
    return (hub + np.exp(1j * theta) * leaves) / np.sqrt(2)


def vertex_visibility(eps, leaf=1, probe=1):
    """Interference visibility at a leaf vertex across SEC phase theta."""
    a = d4_adjacency(eps, leaf)
    deg = np.diag(a.sum(axis=1))
    L = deg - a
    evals, evecs = np.linalg.eigh(L)
    probs = []
    for th in THETAS:
        psi0 = orbit_superposition(th)
        U = evecs @ np.diag(np.exp(-1j * evals * T_EVOLVE)) @ evecs.conj().T
        psi = U @ psi0
        probs.append(float(np.abs(psi[probe])**2))
    probs = np.array(probs)
    return float(probs.max() - probs.min())


def run():
    print("\n  Visibility vs symmetry-breaking eps (leaf edge 0-1, probe vertex 1):")
    V = {}
    for eps in EPS_GRID:
        V[eps] = vertex_visibility(eps, leaf=1, probe=1)
        print(f"    eps={eps:<8g} V={V[eps]:.3e}")

    # T1: symmetric limit
    v0_ok = V[0.0] <= 1e-10
    nz = [(e, V[e]) for e in EPS_GRID if e > 0]
    order_by_eps = [v for _, v in sorted(nz)]
    smallest_two_ok = (np.argsort(order_by_eps)[:2] == np.array([0, 1])).all()
    t1 = bool(v0_ok and smallest_two_ok)
    print(f"\n  T1: V(0)={V[0.0]:.2e} (<=1e-10: {v0_ok}), "
          f"limit->0 ordering: {smallest_two_ok} -> {'PASS' if t1 else 'FAIL'}")

    # T2: monotone
    rho, _ = spearmanr([e for e, _ in sorted(nz)], order_by_eps)
    t2 = bool(rho >= 0.9)
    print(f"  T2: Spearman(V, eps) = {rho:.3f} -> {'PASS' if t2 else 'FAIL'}")

    # T3: exponent equality across orbit-equivalent edges
    print("\n  T3: small-eps exponent across the 3 leaf edges:")
    exps = []
    for leaf in (1, 2, 3):
        es = [1e-4, 3e-4, 1e-3, 3e-3]
        vs = [vertex_visibility(e, leaf=leaf, probe=leaf) for e in es]
        if min(vs) <= 0:
            exps.append(np.nan)
            continue
        p = np.polyfit(np.log(es), np.log(vs), 1)[0]
        exps.append(float(p))
        print(f"    leaf {leaf}: p = {p:.4f}")
    exps_arr = np.array([p for p in exps if np.isfinite(p)])
    cv = float(np.std(exps_arr) / np.mean(exps_arr)) if len(exps_arr) == 3 and np.mean(exps_arr) != 0 else np.inf
    t3 = bool(cv < 0.1)
    print(f"    exponent CV = {cv:.4f} -> {'PASS' if t3 else 'FAIL'}")

    score = sum([t1, t2, t3])
    killed = (V[0.0] > 1e-6) or (len(exps_arr) == 3 and cv > 0.5)
    verdict = 'SUPPORTED' if score == 3 else ('KILLED' if killed else 'PARTIAL')
    print(f"\n  Overall: {score}/3  VERDICT: {verdict}")
    return {
        'experiment': 'exp_03_representative_gauge', 'milestone': 'M15',
        'visibilities': {str(e): v for e, v in V.items()},
        'T1': {'V0': V[0.0], 'PASS': t1},
        'T2': {'spearman': float(rho), 'PASS': t2},
        'T3': {'exponents': exps, 'cv': cv, 'PASS': t3},
        'score': f"{score}/3", 'verdict': verdict,
    }


def selftest():
    print("SELFTEST: harness only (symmetric-case sanity, not scored)")
    v = vertex_visibility(0.0)
    print(f"  V(eps=0) = {v:.2e} (informational)")
    a = d4_adjacency(0.1, 2)
    assert a[0, 2] == 1.1 and a[0, 1] == 1.0
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_03: The Representative Gauge")
    print("Milestone 15 -- pre-registered")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_m15_results('exp_03_representative_gauge', _convert_numpy(data))
