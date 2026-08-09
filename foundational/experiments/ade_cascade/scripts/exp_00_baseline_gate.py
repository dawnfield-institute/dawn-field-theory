"""
exp_00 -- Refactor-safety gate (must PASS before any registered run).

G1: A-family distance kernel == legacy kernel, exactly (matrix identity).
G2: energy_cascade(default) == energy_cascade(coupling_matrix=A-kernel)
    bit-identical per-scale outputs at equal seed.
G3: exp_14 canonical baseline reproduces: n_modes=8, cd=0.1, ns=0.3,
    20 seeds -> mean exponent within 3 sigma of the recorded -1.6083.

Registration: journals/2026-07-17_ade-cascade-round1-preregistration.md
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))
sys.path.insert(0, str(_HERE.parent.parent / "milestone4" / "core"))

from coupling import dynkin_coupling, legacy_kernel   # noqa: E402
from utils import energy_cascade, measure_exponent    # noqa: E402

RESULTS = _HERE.parent / "results"
CANON = dict(n_scales=25, n_samples=15000, coupling_decay=0.1,
             nonlinear_strength=0.3)


def main():
    out = {"experiment": "exp_00_baseline_gate", "gates": {}}

    # G1: kernel identity
    a8 = dynkin_coupling('A', 8, 0.1)
    leg = legacy_kernel(8, 0.1)
    g1 = float(np.max(np.abs(a8 - leg)))
    out["gates"]["G1_kernel_max_abs_diff"] = g1
    out["gates"]["G1_pass"] = bool(g1 == 0.0)

    # G2: bit-identical cascade at equal seed
    np.random.seed(42)
    r_default = energy_cascade(1.0, n_modes=8, **CANON)
    np.random.seed(42)
    r_injected = energy_cascade(1.0, n_modes=8, coupling_matrix=a8, **CANON)
    diffs = [abs(a['P_input'] - b['P_input']) + abs(a['org_fraction'] - b['org_fraction'])
             for a, b in zip(r_default, r_injected)]
    g2 = float(max(diffs))
    out["gates"]["G2_max_scale_diff"] = g2
    out["gates"]["G2_pass"] = bool(g2 == 0.0)

    # G3: exp_14 canonical baseline reproduction (20 seeds, exp_15 seed rule)
    exps = []
    for i in range(20):
        np.random.seed(42 + i * 1000)
        res = energy_cascade(1.0, n_modes=8, **CANON)
        slope, r2, org, se = measure_exponent(res)
        if slope is not None:
            exps.append(slope)
    mean, std = float(np.mean(exps)), float(np.std(exps, ddof=1))
    ref = -1.6083   # exp_14 part_a_baseline recorded mean (20 seeds, std 0.006)
    out["gates"]["G3_mean_exponent"] = mean
    out["gates"]["G3_std"] = std
    out["gates"]["G3_reference"] = ref
    out["gates"]["G3_pass"] = bool(abs(mean - ref) < 0.01)

    out["all_pass"] = all(out["gates"][k] for k in
                          ("G1_pass", "G2_pass", "G3_pass"))
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"exp_00_baseline_gate_{ts}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("GATE:", "PASS" if out["all_pass"] else "FAIL", "->", path)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
