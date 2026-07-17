"""
exp_06 -- k=5,6 confirmation of the momentum-generator derivation.

Registered: journals/2026-07-17_m15-exp06-preregistration.md (committed
before this script ran). Model: G[j',j] = 4jj'/(j'^2 - j^2) for j'-j odd
(box momentum matrix); L_k = sum of positive angle pairs of G_k.

Anchor gates: k=2 -> 8/3 within 1%; k=3,4 reproduce exp_05 limits within
0.05%. Decision: CONFIRM if both k=5,6 within 0.1% of predictions; KILL if
either >1%; else INCONCLUSIVE (one denser-grid rerun permitted).
Secondary: entrywise |m*skew(M)| vs |G| at m=2000 (2%, diag +-1 gauge);
Z2 scan extension.
"""

import sys
import json
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))

from representative import build_cycle, complement_frame, save_m15_results  # noqa: E402

M_GRID = list(range(100, 401, 50))
GAP_MIN = 1e-9
REGISTERED = {5: 17.010952, 6: 25.778092}
EXP05 = {2: 8/3, 3: 5.491004, 4: 11.185742}


def G_matrix(k):
    G = np.zeros((k, k))
    for j2 in range(1, k + 1):
        for j1 in range(1, k + 1):
            if (j2 - j1) % 2 == 1:
                G[j2 - 1, j1 - 1] = 4 * j1 * j2 / (j2**2 - j1**2)
    return G


def L_predicted(k):
    ev = np.linalg.eigvals(G_matrix(k))
    return float(sum(abs(e.imag) for e in ev) / 2)


def raw_overlap(m, k):
    adj = build_cycle(m)
    vu, Vu_, ku = complement_frame(adj, 0)
    vv, Vv_, kv = complement_frame(adj, 1)
    if len(vu) > k and (vu[-k] - vu[-k - 1]) < GAP_MIN:
        return None
    common = [w for w in ku if w != 1]
    Vu = Vu_[[ku.index(w) for w in common], :][:, -k:]
    Vv = Vv_[[kv.index(w) for w in common], :][:, -k:]
    M = Vv.T @ Vu
    for j in range(k):
        if M[j, j] < 0:
            M[j, :] = -M[j, :]
    return M


def theta_T(m, k):
    M = raw_overlap(m, k)
    if M is None:
        return None
    U, _, Wt = np.linalg.svd(M)
    R = U @ Wt
    eig = np.linalg.eigvals(R)
    return float(sum(np.angle(lam) for lam in eig
                     if abs(lam.imag) > 1e-12 and np.angle(lam) > 0))


def richardson(ms, Ls, order=2):
    A = np.vstack([np.ones_like(ms, dtype=float)] +
                  [1.0 / np.array(ms, float)**p for p in range(1, order + 1)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(Ls), rcond=None)
    return float(coef[0])


def measure_limit(k, grid=M_GRID):
    ms, Ls = [], []
    for m in grid:
        th = theta_T(m, k)
        if th is not None:
            ms.append(m)
            Ls.append(m * th)
    return richardson(ms, Ls), ms, [float(x) for x in Ls]


def entrywise_check(k, m=2000, tol=0.02):
    M = raw_overlap(m, k)
    A = m * (M - M.T) / 2
    Gp = G_matrix(k)
    # numerical frame is eigh-ascending; top-k reversed -> physical j = k - idx
    Gp_perm = np.zeros((k, k))
    for i2 in range(k):
        for i1 in range(k):
            Gp_perm[i2, i1] = Gp[k - 1 - i2, k - 1 - i1]
    err = float(np.max(np.abs(np.abs(A) - np.abs(Gp_perm))) /
                np.max(np.abs(Gp_perm)))
    return {"m": m, "max_rel_entry_dev_absvals": err, "pass": bool(err < tol)}


def main():
    out = {"experiment": "exp_06_k56_confirmation"}

    # prediction self-check vs registered values
    preds = {k: L_predicted(k) for k in (5, 6)}
    out["prediction_selfcheck"] = {
        str(k): {"computed": preds[k], "registered": REGISTERED[k],
                 "ok": bool(abs(preds[k] - REGISTERED[k]) < 1e-5)}
        for k in (5, 6)}
    if not all(v["ok"] for v in out["prediction_selfcheck"].values()):
        out["verdict"] = "VOID (prediction self-check failed)"
        save_m15_results("exp_06_k56_confirmation", out)
        print(json.dumps(out, indent=2))
        return

    # anchor gates
    anchors = {}
    for k in (2, 3, 4):
        L, ms, Ls = measure_limit(k)
        anchors[str(k)] = {"limit": L, "target": EXP05[k],
                           "reldev": abs(L - EXP05[k]) / EXP05[k]}
    tol = {2: 0.01, 3: 5e-4, 4: 5e-4}
    anchor_ok = all(anchors[str(k)]["reldev"] < tol[k] for k in (2, 3, 4))
    out["anchors"] = anchors
    out["anchor_pass"] = bool(anchor_ok)
    print("anchors:", {k: round(v["reldev"], 9) for k, v in anchors.items()},
          "PASS" if anchor_ok else "FAIL", flush=True)
    if not anchor_ok:
        out["verdict"] = "VOID (anchor gate failed)"
        save_m15_results("exp_06_k56_confirmation", out)
        return

    # registered measurement
    verdicts = {}
    for k in (5, 6):
        L, ms, Ls = measure_limit(k)
        rd = abs(L - REGISTERED[k]) / REGISTERED[k]
        verdicts[k] = ("CONFIRM" if rd < 1e-3 else
                       "KILL" if rd > 1e-2 else "INCONCLUSIVE")
        out[f"k{k}"] = {"measured_limit": L, "registered_pred": REGISTERED[k],
                        "reldev": rd, "m_grid": ms, "m_theta": Ls,
                        "verdict": verdicts[k],
                        "entrywise": entrywise_check(k)}
        print(f"k={k}: measured={L:.6f} pred={REGISTERED[k]:.6f} "
              f"reldev={rd:.2e} -> {verdicts[k]}", flush=True)

    vs = set(verdicts.values())
    out["verdict"] = ("CONFIRM (momentum-generator derivation)" if vs == {"CONFIRM"}
                      else "KILL" if "KILL" in vs else "INCONCLUSIVE")
    save_m15_results("exp_06_k56_confirmation", out)
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
