"""
exp_05 -- The general-k holonomy limit: K1 (odd-harmonic) vs K2 (Fibonacci).

Registered: journals/2026-07-17_m15-exp05-preregistration.md (commit c5e05712).

Observable (locked): per-edge polar rotation factor R (= the Procrustes
orthogonal factor already returned by core.representative.edge_transport);
theta_T = sum of positive eigen-angles of R counting each conjugate pair
once and excluding real eigenvalues (+/-1, which carry the Z2 data, not
rotation content -- at k=2 this reduces exactly to exp_04's
tan theta_T = (M21-M12)/(M11+M22), the standard 2x2 polar identity).
L_k = lim m * theta_T(m), Richardson-extrapolated over m <= 400.

Anchor gate: k=2 must give 8/3 within 1% before k=3,4 are read.
K1: L_k = 2 * sum_{q odd <= 2k-1} 1/q  ->  k=3: 46/15, k=4: 352/105.
K2: L_k = F_{2k+2}/F_{2k}              ->  k=3: 21/8,  k=4: 55/21.
Rule: within 1% of exactly one candidate for BOTH k -> that candidate;
disagreement or neither -> both die, OPEN, values reported [D].

K3 (exploratory): det(edge transport) pattern + det(H) around full cycles.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))

from representative import (   # noqa: E402
    build_cycle, complement_frame, edge_transport, cycle_holonomy,
    save_m15_results)

M_GRID = list(range(100, 401, 50))       # extrapolation grid (large-m regime)
M_ANCHOR_CHECK = list(range(24, 61, 6))  # small-m sanity trace
GAP_MIN = 1e-9

K1 = {2: 8/3, 3: 46/15, 4: 352/105}
K2 = {2: 8/3, 3: 21/8, 4: 55/21}


def theta_T(adjacency, k):
    """Sum of positive rotation angles of the per-edge polar rotation factor.

    The raw overlap is computed under the ANALYTIC SIGN CONVENTION
    (diag(M) > 0, matching the closed form's diag = cos(pi*j/m) > 0):
    eigh's per-column sign ambiguity is fixed by flipping rows of M with
    negative diagonal. Without this, arbitrary eigenvector signs can turn
    M into a reflection and the polar factor loses the rotation content
    (anchor-gate failure on the first run, 2026-07-17, disclosed in the
    outcomes journal). At k=2 the resulting angle equals
    atan2(M21-M12, M11+M22), exp_04's formula, on the sign-fixed M.
    """
    vals_u, vecs_u, keep_u = complement_frame(adjacency, 0)
    vals_v, vecs_v, keep_v = complement_frame(adjacency, 1)
    if len(vals_u) > k and (vals_u[-k] - vals_u[-k - 1]) < GAP_MIN:
        return None, float(vals_u[-k] - vals_u[-k - 1])
    common = [w for w in keep_u if w != 1]
    rows_u = [keep_u.index(w) for w in common]
    rows_v = [keep_v.index(w) for w in common]
    Vu = vecs_u[rows_u, :][:, -k:]
    Vv = vecs_v[rows_v, :][:, -k:]
    M = Vv.T @ Vu
    for j in range(k):                    # analytic sign convention
        if M[j, j] < 0:
            M[j, :] = -M[j, :]
    U, _, Wt = np.linalg.svd(M)
    R = U @ Wt                            # polar rotation factor
    eig = np.linalg.eigvals(R)
    total = 0.0
    for lam in eig:
        ang = np.angle(lam)
        if abs(lam.imag) > 1e-12 and ang > 0:   # one per conjugate pair
            total += ang
    gap = float(vals_u[-k] - vals_u[-k - 1]) if len(vals_u) > k else np.inf
    return float(total), gap


def richardson(ms, Ls, order=2):
    """Fit L(m) = L_inf + a/m + b/m^2 by least squares; return L_inf."""
    A = np.vstack([np.ones_like(ms, dtype=float)] +
                  [1.0 / np.array(ms, float)**p for p in range(1, order + 1)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(Ls), rcond=None)
    return float(coef[0])


def limit_for_k(k):
    ms, Ls, skips = [], [], []
    for m in M_GRID:
        adj = build_cycle(m)
        th, gap = theta_T(adj, k)
        if th is None:
            skips.append({"m": m, "gap": gap})
            continue
        ms.append(m)
        Ls.append(m * th)
    L_inf = richardson(ms, Ls)
    return {"k": k, "m_grid": ms, "m_theta": [float(x) for x in Ls],
            "richardson_limit": L_inf, "skips": skips}


def classify(L, k):
    d1 = abs(L - K1[k]) / K1[k]
    d2 = abs(L - K2[k]) / K2[k]
    hit1, hit2 = d1 < 0.01, d2 < 0.01
    return {"K1_pred": K1[k], "K2_pred": K2[k],
            "K1_reldev": d1, "K2_reldev": d2,
            "match": "K1" if (hit1 and not hit2) else
                     "K2" if (hit2 and not hit1) else
                     "AMBIGUOUS" if (hit1 and hit2) else "NEITHER"}


def k3_z2_scan():
    """det(edge transport) sequence + det(H) for k=2,3,4 on small cycles."""
    out = {}
    for k in (2, 3, 4):
        rows = []
        for m in range(max(6, k + 3), 31, 2):
            adj = build_cycle(m)
            frames = {}
            dets = []
            ok = True
            for i in range(m):
                T, gap = edge_transport(adj, i, (i + 1) % m, k, frames)
                if gap < GAP_MIN:
                    ok = False
                    break
                dets.append(int(np.sign(np.linalg.det(T))))
            if not ok:
                continue
            hol = cycle_holonomy(adj, list(range(m)), k)
            rows.append({"m": m, "n_reflections": int(dets.count(-1)),
                         "reflections_even": dets.count(-1) % 2 == 0,
                         "det_H": round(hol["det"], 6),
                         "holonomy_angles": hol["angles"]})
        out[f"k={k}"] = rows
    return out


def main():
    out = {"experiment": "exp_05_general_k_limit",
           "registration_commit": "c5e05712"}

    # Anchor gate: k=2 -> 8/3
    lim2 = limit_for_k(2)
    anchor_dev = abs(lim2["richardson_limit"] - 8/3) / (8/3)
    out["anchor"] = {**lim2, "target": 8/3, "reldev": anchor_dev,
                     "pass": bool(anchor_dev < 0.01)}
    print(f"anchor k=2: limit={lim2['richardson_limit']:.6f} "
          f"target={8/3:.6f} reldev={anchor_dev:.2e} "
          f"{'PASS' if anchor_dev < 0.01 else 'FAIL'}", flush=True)

    if out["anchor"]["pass"]:
        verdicts = {}
        for k in (3, 4):
            lim = limit_for_k(k)
            lim["classification"] = classify(lim["richardson_limit"], k)
            out[f"k{k}"] = lim
            verdicts[k] = lim["classification"]["match"]
            print(f"k={k}: limit={lim['richardson_limit']:.6f} "
                  f"K1={K1[k]:.4f} K2={K2[k]:.4f} -> {verdicts[k]}",
                  flush=True)
        vs = set(verdicts.values())
        out["verdict"] = (f"{vs.pop()} CONFIRMED" if len(vs) == 1 and
                          vs != {"NEITHER"} and "AMBIGUOUS" not in verdicts.values()
                          else "OPEN (both candidates dead or disagreement)")
    else:
        out["verdict"] = "VOID (anchor gate failed -- instrument bug)"

    print("K3 Z2 scan ...", flush=True)
    out["k3_z2_scan"] = k3_z2_scan()

    save_m15_results("exp_05_general_k_limit", out)
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
