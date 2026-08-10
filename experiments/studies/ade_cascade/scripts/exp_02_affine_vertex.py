"""
exp_02 -- R2: the affine-vertex reading of the k-1 offset.

Per rank r in {6,7,8}:
  shift_affine(r) = |mean_exp(A~_r) - mean_exp(A_r)|   (add the affine node -> cycle)
  shift_path(r)   = |mean_exp(A_{r+1}) - mean_exp(A_r)| (add an ordinary node)
  rho(r) = shift_affine / shift_path

R2 rule (locked): CONFIRM if median rho < 0.25; KILL if median rho > 0.75;
else INCONCLUSIVE.

A-arms recomputed with the identical registered seeds (deterministic, equal
to exp_01's values by construction).

Registration: journals/2026-07-17_ade-cascade-round1-preregistration.md
(commit c5e05712).
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))

from coupling import dynkin_coupling, affine_a_coupling   # noqa: E402
from runner import run_arm                                # noqa: E402

RESULTS = _HERE.parent / "results"
RANKS = (6, 7, 8)


def main():
    arms = {}
    for r in (6, 7, 8, 9):
        name = f"A_{r}"
        print(f"running {name} ...", flush=True)
        arms[name] = run_arm(name, dynkin_coupling("A", r, 0.1))
    for r in RANKS:
        name = f"Atilde_{r}"
        print(f"running {name} (cycle on {r+1}) ...", flush=True)
        arms[name] = run_arm(name, affine_a_coupling(r, 0.1))

    rhos, detail = [], {}
    for r in RANKS:
        sa = abs(arms[f"Atilde_{r}"]["mean"] - arms[f"A_{r}"]["mean"])
        sp = abs(arms[f"A_{r+1}"]["mean"] - arms[f"A_{r}"]["mean"])
        rho = sa / sp if sp > 0 else float('inf')
        rhos.append(rho)
        detail[r] = {"shift_affine": sa, "shift_path": sp, "rho": rho,
                     "exp_A": arms[f"A_{r}"]["mean"],
                     "exp_Atilde": arms[f"Atilde_{r}"]["mean"],
                     "exp_A_next": arms[f"A_{r+1}"]["mean"]}
    med = float(np.median(rhos))
    verdict = ("CONFIRM" if med < 0.25 else
               "KILL" if med > 0.75 else "INCONCLUSIVE")

    out = {
        "experiment": "exp_02_affine_vertex",
        "registration_commit": "c5e05712",
        "R2": {"per_rank": detail, "median_rho": med, "verdict": verdict},
        "t1_flags": {k: v["t1_contaminated"] for k, v in arms.items()},
        "arms": arms,
    }
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"exp_02_affine_vertex_{ts}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    slim = {k: v for k, v in out.items() if k != "arms"}
    print(json.dumps(slim, indent=2, default=str))
    print("saved ->", path)


if __name__ == "__main__":
    main()
