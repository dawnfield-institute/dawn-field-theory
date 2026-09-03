#!/usr/bin/env python3
"""exp_03: Negative controls — the instrument must be able to say NO.
Registered 2026-08-31 (cf886c00)."""
import sys, json, numpy as np
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from folding import (e8_roots, a5_roots, A5_SIMPLE, E8_SIMPLE,
                     coxeter_element, eigenplane_basis, shell_split, PHI)

res = {"experiment_id": "exp_03_negative_controls", "registration": "cf886c00", "tests": {}}
rng = np.random.default_rng(20260831)

def phi_split(roots, basis, want=None, tol=1e-6):
    """True iff projection gives exactly two EQUAL shells at ratio phi."""
    if len(basis) == 0: return False
    sh, P = shell_split(roots, np.atleast_2d(basis), dec=6)
    rr = sorted(sh)
    if len(rr) != 2 or sh[rr[0]] != sh[rr[1]]: return False
    if want and sh[rr[0]] != want: return False
    raw = np.linalg.norm(P, axis=1)
    r0 = raw[np.isclose(raw, rr[0], atol=1e-4)].mean()
    r1 = raw[np.isclose(raw, rr[1], atol=1e-4)].mean()
    return abs(r1/r0 - PHI) < tol

def all_subspaces(simples, h, plane_ms, neg_mult):
    """Yield bases for every union of eigenplanes (and -1 directions)."""
    W = coxeter_element(simples)
    planes = [eigenplane_basis(W, [m], h) for m in plane_ms]
    negs = []
    if neg_mult:
        vals, vecs = np.linalg.eigh((W + W.T)/2)
        idx = [k for k in range(len(vals)) if abs(vals[k] + 1) < 1e-8]
        negs = [vecs[:, k] for k in idx]
    items = planes + [np.atleast_2d(v) for v in negs]
    for r in range(1, len(items)+1):
        for combo in combinations(items, r):
            yield np.vstack(combo)

# T1: A5 (h=6, planes m=1,2; one -1 direction) — no phi-scaled equal split of its 30 roots
hits = sum(phi_split(a5_roots(), B) for B in all_subspaces(A5_SIMPLE, 6, [1, 2], True))
res["tests"]["T1"] = {"system": "A5", "phi_splits_found": int(hits), "pass": hits == 0}

# T2: D5 and E6 fail the construction
D5_SIMPLE = np.array([[1,-1,0,0,0],[0,1,-1,0,0],[0,0,1,-1,0],[0,0,0,1,-1],[0,0,0,1,1]], float)
d5 = np.array([v for i, j in combinations(range(5), 2)
               for v in ([1,0],[0,1]) for _ in [0]
               ], float) if False else None
d5 = []
for i, j in combinations(range(5), 2):
    for si in (1,-1):
        for sj in (1,-1):
            v = np.zeros(5); v[i] = si; v[j] = sj; d5.append(v)
d5 = np.array(d5)
hits_d5 = sum(phi_split(d5, B) for B in all_subspaces(D5_SIMPLE, 8, [1, 3], True))
# E6 as the sub-root-system of E8 in the span of six of its simple roots (indices 0-5:
# the branch node and five chain nodes give the E6 diagram in our E8_SIMPLE ordering —
# verified below by root count 72)
S6 = E8_SIMPLE[[0,1,2,3,4,5]]
Q, _ = np.linalg.qr(S6.T)
R8 = e8_roots()
in_span = np.linalg.norm(R8 - (R8 @ Q) @ Q.T, axis=1) < 1e-9
E6R = (R8[in_span]) @ Q                       # 6D coordinates
W6 = None
S6c = S6 @ Q                                   # simples in the 6D frame
hits_e6 = sum(phi_split(E6R, B) for B in all_subspaces(S6c, 12, [1, 4, 5], False))
res["tests"]["T2"] = {"D5_phi_splits": int(hits_d5), "E6_root_count": int(in_span.sum()),
                      "E6_phi_splits": int(hits_e6),
                      "pass": hits_d5 == 0 and in_span.sum() == 72 and hits_e6 == 0}

# T3: 100 random orthogonal 4D projections of E8 — accidental phi-splits
acc = 0
for _ in range(100):
    M = rng.normal(size=(8, 8)); Qr, _ = np.linalg.qr(M)
    if phi_split(R8, Qr[:4], want=120): acc += 1
res["tests"]["T3"] = {"random_projections": 100, "accidental_phi_splits": acc, "pass": acc == 0}

# T4: isometrically scrambled roots against the FIXED H4 projector — split destroyed
W8 = coxeter_element(E8_SIMPLE)
B4 = eigenplane_basis(W8, [1, 11], 30)
destroyed = 0
for _ in range(20):
    M = rng.normal(size=(8, 8)); Qr, _ = np.linalg.qr(M)
    if not phi_split(R8 @ Qr.T, B4, want=120): destroyed += 1
res["tests"]["T4"] = {"scrambles": 20, "splits_destroyed": destroyed, "pass": destroyed == 20}

res["score"] = sum(res["tests"][k]["pass"] for k in res["tests"])
print(json.dumps(res["tests"], indent=1, default=str)); print("SCORE", res["score"], "/4")
(Path(__file__).parent.parent / "results" / "exp_03_controls_20260831.json").write_text(
    json.dumps(res, indent=1, default=str))
