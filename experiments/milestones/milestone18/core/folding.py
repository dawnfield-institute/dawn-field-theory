"""Milestone 18 core: root systems, Coxeter elements, and golden eigenplane projections.

The construction verified here is classical (Coxeter; Moody-Patera; Conway-Sloane):
project a root system onto the invariant subspace of its Coxeter element spanned by
the eigenplanes of the H-partner's exponents. For E8 (h=30), the H4 exponents are
{1,11,19,29}; the projection of the 240 roots must yield two concentric shells of
120, radius ratio phi, each an H4 (600-cell) root system.
"""
import numpy as np
from itertools import combinations

PHI = (1 + 5**0.5) / 2

def e8_roots():
    R = []
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8); v[i] = si; v[j] = sj; R.append(v)
    for signs in range(256):
        s = np.array([1 if (signs >> k) & 1 else -1 for k in range(8)], float)
        if (s < 0).sum() % 2 == 0:
            R.append(s / 2)
    return np.array(R)

def d6_roots():
    R = []
    for i, j in combinations(range(6), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(6); v[i] = si; v[j] = sj; R.append(v)
    return np.array(R)

def a5_roots():
    R = []
    for i in range(6):
        for j in range(6):
            if i != j:
                v = np.zeros(6); v[i] = 1; v[j] = -1; R.append(v)
    return np.array(R)

E8_SIMPLE = np.array([
    [ .5, -.5, -.5, -.5, -.5, -.5, -.5,  .5],
    [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0., -1.,  1.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0., -1.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.]])

D6_SIMPLE = np.array([[1,-1,0,0,0,0],[0,1,-1,0,0,0],[0,0,1,-1,0,0],
                      [0,0,0,1,-1,0],[0,0,0,0,1,-1],[0,0,0,0,1,1]], float)

A5_SIMPLE = np.array([[1,-1,0,0,0,0],[0,1,-1,0,0,0],[0,0,1,-1,0,0],
                      [0,0,0,1,-1,0],[0,0,0,0,1,-1]], float)

def reflection(alpha):
    a = alpha / np.linalg.norm(alpha)
    return np.eye(len(alpha)) - 2 * np.outer(a, a)

def coxeter_element(simples):
    W = np.eye(simples.shape[1])
    for a in simples:
        W = reflection(a) @ W
    return W

def eigenplane_basis(W, angles_frac, h, tol=1e-8):
    """Real orthonormal basis of the invariant planes at rotation angles 2*pi*m/h.

    Uses the SYMMETRIC operator (W + W^T)/2, whose eigenspace at cos(2*pi*m/h) is
    exactly the invariant plane — numpy.eigh gives it to ~1e-15, where eig on the
    nonsymmetric W loses ~5 digits (exp_02 first run failed its 1e-12 threshold at
    1.9e-10 for exactly this reason)."""
    S = (W + W.T) / 2
    vals, vecs = np.linalg.eigh(S)
    cols = []
    for m in angles_frac:
        c = np.cos(2 * np.pi * m / h)
        idx = [k for k in range(len(vals)) if abs(vals[k] - c) < tol]
        for k in idx:
            u = vecs[:, k].copy()
            for prev in cols:
                u = u - (u @ prev) * prev
            n = np.linalg.norm(u)
            if n > tol:
                cols.append(u / n)
    return np.array(cols)          # rows orthonormal

def shell_split(roots, basis, dec=9):
    """Project roots onto span(basis); return dict radius -> count (radii rounded)."""
    P = roots @ basis.T
    r = np.round(np.linalg.norm(P, axis=1), dec)
    out = {}
    for x in r:
        out[x] = out.get(x, 0) + 1
    return out, P

def is_h4_shell(P, tol=1e-7):
    """120 unit-normalized 4D vectors: check the 600-cell inner-product spectrum
    {0, ±1/2, ±phi/2, ±1/(2phi), ±1} and reflection closure."""
    V = P / np.linalg.norm(P, axis=1)[:, None]
    G = np.abs(V @ V.T)
    allowed = np.array([0.0, 0.5, PHI/2, 1/(2*PHI), 1.0])
    ok_spectrum = np.all(np.min(np.abs(G[:, :, None] - allowed[None, None, :]), axis=2) < tol)
    # reflection closure on a sample
    rng = np.random.default_rng(1)
    idx = rng.choice(len(V), size=min(20, len(V)), replace=False)
    closed = True
    for i in idx:
        R = np.eye(V.shape[1]) - 2 * np.outer(V[i], V[i])
        img = V @ R.T
        d = np.min(np.linalg.norm(img[:, None, :] - V[None, :, :], axis=2), axis=1)
        if d.max() > 1e-6:
            closed = False; break
    return bool(ok_spectrum), bool(closed)
