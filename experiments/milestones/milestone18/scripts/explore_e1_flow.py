"""E-A/E-D (EXPLORING): eigenvalue flow lambda_k(s) — crossings, near-crossings,
and what actually happens AT the self-dual point."""
import numpy as np

def M_of(n, edges, s):
    A = np.zeros((n, n))
    for i, j in edges: A[i, j] = A[j, i] = 1
    D = np.diag(A.sum(1))
    return (D - A) + s * (2*np.eye(n) - D)

TREES = {
  "D6-tree":  (6, [(0,1),(1,2),(2,3),(3,4),(3,5)]),
  "E8-tree":  (8, [(i,i+1) for i in range(6)]+[(2,7)]),
  "path4":    (4, [(0,1),(1,2),(2,3)]),
  "path5":    (5, [(0,1),(1,2),(2,3),(3,4)]),
  "D8-tree":  (8, [(i,i+1) for i in range(6)]+[(5,7)]),
  "cat8":     (8, [(0,1),(1,2),(2,3),(1,4),(2,5),(0,6),(3,7)]),
  "rand8b":   (8, [(0,1),(0,2),(2,3),(2,4),(4,5),(4,6),(6,7)]),
}
S = np.linspace(0, 2, 2001)
print(f"{'tree':<9} {'min adjacent gap over s':>24} {'at s':>6} {'at lam':>7}   crossings (gap<1e-9)")
for name, (n, e) in TREES.items():
    evs = np.array([np.linalg.eigvalsh(M_of(n, e, s)) for s in S])
    gaps = np.diff(evs, axis=1)                     # adjacent-level gaps at each s
    mg = gaps.min(); idx = np.unravel_index(gaps.argmin(), gaps.shape)
    s_at = S[idx[0]]; lam_at = evs[idx[0], idx[1]]
    crossings = [(round(S[i],3), round(evs[i,k],3)) for i in range(len(S)) for k in range(n-1)
                 if gaps[i,k] < 1e-9]
    # dedupe
    seen=set(); cr=[]
    for c in crossings:
        k=(round(c[0],2), round(c[1],2))
        if k not in seen: seen.add(k); cr.append(c)
    print(f"{name:<9} {mg:>24.2e} {s_at:>6.3f} {lam_at:>7.3f}   {cr if cr else '—'}")

print("\nD6 anatomy: the tine-antisymmetric mode is EXACTLY lambda = 1+s (Aut-odd line):")
n, e = TREES["D6-tree"]
for s in (0.6, 1.0, 1.4):
    ev = np.linalg.eigvalsh(M_of(n, e, s))
    print(f"  s={s}: eigenvalues {np.round(ev,4)}   contains 1+s={1+s}: {np.any(np.abs(ev-(1+s))<1e-9)}")
swap = np.eye(6)[[0,1,2,3,5,4]]                     # Aut: swap the two tines
s0 = 1.0
w, V = np.linalg.eigh(M_of(n, e, s0))
deg = [k for k in range(6) if abs(w[k]-2) < 1e-9]
print(f"  at s=1: eigenvalue 2 multiplicity {len(deg)}")
for k in deg:
    v = V[:, k]; parity = float(v @ swap @ v)
    print(f"    mode {k}: <v|swap|v> = {parity:+.3f}")
print("\nE8: closest adjacent-level approach NEAR s=1 specifically:")
n, e = TREES["E8-tree"]
evs = np.array([np.linalg.eigvalsh(M_of(n, e, s)) for s in S])
near1 = (S > 0.9) & (S < 1.1)
g = np.diff(evs, axis=1)[near1]
print(f"  min gap in s in (0.9,1.1): {g.min():.4f}  (vs global min above)")
