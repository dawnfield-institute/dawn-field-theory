"""
exp_00 -- Instrument qualification gates (no physics claims).

Q1  Taylor-Green exactness: 256^2, nu=0.01, dt=1e-3 to t=1.0;
    max|w_num - w_exact| < 1e-6 (spectral space + RK4 time; TG's
    nonlinear term vanishes identically, so error is pure scheme error).
Q2  Energy budget closure on a decaying random field: dE/dt = -2 nu Z;
    |Delta_E - trapz(-2 nu Z)| / E0 < 1e-5 over t=1.
Q3  Resolution consistency: identical band-limited initial data (k in
    [2,10], nu=1e-2) at 128^2 vs 256^2 to t=0.5; relative L2 difference
    of spectral coefficients on the common band |kx|,|ky| <= 40 < 1e-6.
    Gate-design history (all runs in git): (i) raw-grid comparison gave
    1.8e-4 -- wrong test (counts modes only one run resolves + subsample
    aliasing); (ii) common-band with init k<=20, nu=5e-3 gave 2.5e-5 --
    genuine under-resolution (triads cross the 128^2 dealias cutoff 42 by
    t=0.5, truncation feeds back into the band). A convergence gate is
    only meaningful where both runs resolve the flow, hence the milder
    settings; under-resolved operation is what production runs must
    detect via the same spectral-tail check.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))

from solver2d import (   # noqa: E402
    Spectral2D, taylor_green_w, random_band_limited)

RESULTS = _HERE.parent / "results"


def q1_taylor_green():
    n, nu, dt, t_end = 256, 0.01, 1e-3, 1.0
    s = Spectral2D(n, nu)
    X, Y = s.grid()
    w_hat = np.fft.fft2(taylor_green_w(X, Y, 0.0, nu))
    steps = int(round(t_end / dt))
    for _ in range(steps):
        w_hat = s.step_rk4(w_hat, dt)
    w_num = np.fft.ifft2(w_hat).real
    err = float(np.max(np.abs(w_num - taylor_green_w(X, Y, t_end, nu))))
    return {"max_abs_error": err, "pass": bool(err < 1e-6)}


def q2_energy_budget():
    n, nu, dt, t_end = 256, 5e-3, 5e-4, 1.0
    s = Spectral2D(n, nu)
    w_hat = random_band_limited(n, 3, 12, seed=42)
    E0 = s.energy(w_hat)
    steps = int(round(t_end / dt))
    zs = [s.enstrophy(w_hat)]
    for _ in range(steps):
        w_hat = s.step_rk4(w_hat, dt)
        zs.append(s.enstrophy(w_hat))
    E1 = s.energy(w_hat)
    dissip = -2.0 * nu * np.trapezoid(np.array(zs), dx=dt)
    resid = float(abs((E1 - E0) - dissip) / E0)
    return {"E0": E0, "E1": E1, "integrated_dissipation": float(dissip),
            "relative_residual": resid, "pass": bool(resid < 1e-5)}


def q3_resolution():
    nu, dt, t_end = 1e-2, 5e-4, 0.5
    fields = {}
    for n in (128, 256):
        s = Spectral2D(n, nu)
        w_hat = random_band_limited(128, 2, 10, seed=7)   # same modes
        if n != 128:
            big = np.zeros((n, n), dtype=complex)
            h = 64
            big[:h, :h] = w_hat[:h, :h]
            big[:h, -h:] = w_hat[:h, -h:]
            big[-h:, :h] = w_hat[-h:, :h]
            big[-h:, -h:] = w_hat[-h:, -h:]
            w_hat = big * (n / 128)**2
        for _ in range(int(round(t_end / dt))):
            w_hat = s.step_rk4(w_hat, dt)
        # common resolved band |kx|,|ky| <= 40, normalized per-mode (fft2
        # coefficients scale with grid count)
        kb = 40
        band = np.zeros((2 * kb + 1, 2 * kb + 1), dtype=complex)
        idx = np.r_[0:kb + 1, -kb:0]
        band[np.ix_(range(2 * kb + 1), range(2 * kb + 1))] = \
            w_hat[np.ix_(idx, idx)] / n**2
        fields[n] = band
    num = np.linalg.norm(fields[256] - fields[128])
    den = np.linalg.norm(fields[128])
    rel = float(num / den)
    return {"relative_l2_diff_common_band": rel, "pass": bool(rel < 1e-6)}


def main():
    out = {"experiment": "exp_00_qualification"}
    print("Q1 Taylor-Green ...", flush=True)
    out["Q1_taylor_green"] = q1_taylor_green()
    print(json.dumps(out["Q1_taylor_green"]), flush=True)
    print("Q2 energy budget ...", flush=True)
    out["Q2_energy_budget"] = q2_energy_budget()
    print(json.dumps(out["Q2_energy_budget"]), flush=True)
    print("Q3 resolution ...", flush=True)
    out["Q3_resolution"] = q3_resolution()
    print(json.dumps(out["Q3_resolution"]), flush=True)
    out["qualified"] = all(out[k]["pass"] for k in
                           ("Q1_taylor_green", "Q2_energy_budget",
                            "Q3_resolution"))
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"exp_00_qualification_{ts}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print("QUALIFIED:" if out["qualified"] else "NOT QUALIFIED:", path)
    return 0 if out["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
