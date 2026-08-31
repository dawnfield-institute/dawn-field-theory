#!/usr/bin/env python3
"""exp_02: The projection carries phi. Registered 2026-08-31 (cf886c00)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from folding import (e8_roots, d6_roots, E8_SIMPLE, D6_SIMPLE,
                     coxeter_element, eigenplane_basis, shell_split, is_h4_shell, PHI)

res = {"experiment_id": "exp_02_projection_carries_phi", "registration": "cf886c00", "tests": {}}
R8 = e8_roots()
W8 = coxeter_element(E8_SIMPLE)

# H4 subspace: eigenplanes of the Coxeter element at exponents {1, 11} of h=30
B = eigenplane_basis(W8, [1, 11], 30)
shells, P = shell_split(R8, B)
radii = sorted(shells)
res["tests"]["T1"] = {"n_shells": len(radii), "counts": [shells[r] for r in radii],
                      "pass": len(radii) == 2 and all(shells[r] == 120 for r in radii)}

# ratio from RAW norms (shell_split's rounded radii are bin labels, not measurements —
# first run computed the ratio from 9-decimal bins and failed its own 1e-12 threshold)
raw = np.linalg.norm(P, axis=1)
if len(radii) == 2:
    r_in  = raw[np.isclose(raw, radii[0], atol=1e-6)].mean()
    r_out = raw[np.isclose(raw, radii[1], atol=1e-6)].mean()
    ratio = r_out / r_in
else:
    ratio = None
res["tests"]["T2"] = {"ratio": ratio, "phi": PHI,
                      "dev": None if ratio is None else abs(ratio - PHI),
                      "pass": ratio is not None and abs(ratio - PHI) < 1e-12}

# T3: the ORTHOGONAL PROJECTOR onto the H4 subspace (basis-independent) has entries
# in Q(sqrt5): each entry x satisfies x = p + q*sqrt5 with small rational p, q.
proj = B.T @ B
def golden_rational(x, maxden=64, tol=1e-9):
    """Is x = (a + b*sqrt5)/c for small integers? Proper 2D search per denominator."""
    s5 = 5**0.5
    for c in range(1, maxden+1):
        y = x * c
        for b in range(-3*c, 3*c+1):
            a = round(y - b*s5)
            if abs(a + b*s5 - y) < tol * c:
                return (a, b, c)
    return None
flat = np.unique(np.round(proj, 12))
recog = {float(x): golden_rational(float(x)) for x in flat}
res["tests"]["T3"] = {"n_distinct_entries": len(flat),
                      "entries": {f"{k:.12f}": (f"({v[0]}+{v[1]}*sqrt5)/{v[2]}" if v else None)
                                  for k, v in recog.items()},
                      "pass": all(v is not None for v in recog.values())}
# and each shell is a genuine H4 root system
inner, outer = (P[np.isclose(np.linalg.norm(P,axis=1), radii[0])],
                P[np.isclose(np.linalg.norm(P,axis=1), radii[1])])
s_in, c_in  = is_h4_shell(inner);  s_out, c_out = is_h4_shell(outer)
res["tests"]["T3"]["shells_are_H4"] = [s_in and c_in, s_out and c_out]
res["tests"]["T3"]["pass"] = bool(res["tests"]["T3"]["pass"] and s_in and c_in and s_out and c_out)

# T4: D6 -> H3 + phi*H3. h=10, H3 exponents {1,5,9}: plane(m=1) + one direction in the
# 2D (-1)-eigenspace (m=5 twice). The correct direction is the H3-isotypic one; locate it
# by scanning the circle and then VERIFY exactness (30/30, ratio phi).
R6 = d6_roots(); W6 = coxeter_element(D6_SIMPLE)
B1 = eigenplane_basis(W6, [1], 10)                       # 2 rows
vals, vecs = np.linalg.eig(W6)
neg = [k for k in range(6) if abs(vals[k] + 1) < 1e-8]
u1 = np.real(vecs[:, neg[0]]); u2 = np.real(vecs[:, neg[1]])
for c in B1:
    u1 -= (u1 @ c) * c; u2 -= (u2 @ c) * c
u1 /= np.linalg.norm(u1); u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
best = None
for th in np.linspace(0, np.pi, 3601):
    v = np.cos(th)*u1 + np.sin(th)*u2
    Bd = np.vstack([B1, v])
    sh, _ = shell_split(R6, Bd, dec=6)
    if len(sh) == 2:
        rr = sorted(sh)
        if all(sh[r] == 30 for r in rr):
            best = (th, rr[1]/rr[0]); break
res["tests"]["T4"] = {"found_direction": best is not None,
                      "ratio": best[1] if best else None,
                      "pass": best is not None and abs(best[1] - PHI) < 1e-4}
res["score"] = sum(res["tests"][k]["pass"] for k in res["tests"])
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "counts" or True}
                  for k, v in res["tests"].items()}, indent=1, default=str))
print("SCORE", res["score"], "/4")
(Path(__file__).parent.parent / "results" / "exp_02_projection_20260831.json").write_text(
    json.dumps(res, indent=1, default=str))
