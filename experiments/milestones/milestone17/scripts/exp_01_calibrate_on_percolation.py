#!/usr/bin/env python3
"""M17 exp_01 — calibrate the criticality instruments on a system with exact known answers.

No DFT system is measured until this passes. The preceding round produced seven instrument
faults, every one caught only by a reference whose answer was already known, so the first
experiment of this milestone measures nothing new on purpose.

2D site percolation on a square lattice is the target because it is one of the few critical
systems with EXACT results:

    p_c   = 0.5927460     located here by the spanning-probability crossing
    gamma/nu = 43/24 = 1.7917   from how the susceptibility peak grows with L
    tau   = 187/91 = 2.0549     cluster size distribution at p_c

Three tests, each of which can fail:

  T1  The spanning-probability curves for different L must CROSS, and the crossing must land
      on p_c. This is finite-size scaling doing the one job it exists for: locating a critical
      point without being told where it is.
  T2  chi_max must scale as L^(gamma/nu). Recovers a critical EXPONENT, which is universal.
  T3  n_s at p_c must be a power law with slope -tau, and must NOT be one away from p_c.
      The second half matters as much as the first -- an instrument that finds power laws
      everywhere has found nothing.

    python exp_01_calibrate_on_percolation.py [--sizes 32 64 128] [--samples 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))

import numpy as np  # noqa: E402

from criticality import (GAMMA_OVER_NU_2D, P_C_2D, TAU_2D,  # noqa: E402
                         finite_size_crossing, fit_power_law, order_parameter,
                         pooled_cluster_distribution, site_lattice, spans, susceptibility)


def sweep(L, ps, samples, rng):
    span, chi, order = [], [], []
    for p in ps:
        sp, ch, od = [], [], []
        for _ in range(samples):
            occ = site_lattice(L, p, rng)
            sp.append(1.0 if spans(occ) else 0.0)
            ch.append(susceptibility(occ))
            od.append(order_parameter(occ))
        span.append(np.mean(sp)); chi.append(np.mean(ch)); order.append(np.mean(od))
    return np.array(span), np.array(chi), np.array(order)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[32, 64, 128])
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--points", type=int, default=21)
    args = ap.parse_args()

    rng = np.random.default_rng(17)
    ps = np.linspace(0.50, 0.70, args.points)

    print(f"  2D site percolation, L = {args.sizes}, {args.samples} samples/point, "
          f"{args.points} points over p in [0.50, 0.70]")
    print(f"  exact: p_c = {P_C_2D:.7f}   gamma/nu = {GAMMA_OVER_NU_2D:.4f}   "
          f"tau = {TAU_2D:.4f}\n")

    span_curves, chi_curves, ord_curves = {}, {}, {}
    for L in args.sizes:
        s, c, o = sweep(L, ps, args.samples, rng)
        span_curves[L], chi_curves[L], ord_curves[L] = s, c, o
        k = int(np.argmax(c))
        print(f"    L={L:<4} chi_max {c[k]:8.2f} at p={ps[k]:.4f}   "
              f"P_span(p_c)={np.interp(P_C_2D, ps, s):.3f}")

    # --- T1: the crossing locates p_c ---
    p_cross, spread, ratio = finite_size_crossing(ps, span_curves)
    t1 = abs(p_cross - P_C_2D) < 0.02 and ratio < 0.5
    print(f"\n  T1 crossing        p_c = {p_cross:.4f}  (exact {P_C_2D:.4f}, "
          f"err {abs(p_cross-P_C_2D):.4f})   spread ratio {ratio:.3f}   "
          f"{'PASS' if t1 else 'FAIL'}")

    # --- T2: susceptibility peak scaling gives gamma/nu ---
    Ls = np.array(args.sizes, float)
    chi_max = np.array([chi_curves[L].max() for L in args.sizes])
    gn, r2, _ = fit_power_law(Ls, chi_max, min_points=len(Ls))
    t2 = abs(gn - GAMMA_OVER_NU_2D) < 0.35 and r2 > 0.95
    print(f"  T2 chi_max ~ L^     {gn:.4f}  (exact {GAMMA_OVER_NU_2D:.4f}, "
          f"err {abs(gn-GAMMA_OVER_NU_2D):.4f})   R2 {r2:.4f}   "
          f"{'PASS' if t2 else 'FAIL'}")

    # --- T3: n_s is power-law AT p_c and not away from it ---
    # Pooled across realisations -- one distribution with the full sample count behind each
    # bin, rather than a slope fitted per lattice and averaged.
    L = max(args.sizes)
    nrep = max(40, args.samples)
    s1, n1 = pooled_cluster_distribution(site_lattice(L, P_C_2D, rng) for _ in range(nrep))
    s2, n2 = pooled_cluster_distribution(site_lattice(L, 0.45, rng) for _ in range(nrep))
    tau_fit, r2_at, _ = fit_power_law(s1, n1)
    _, r2_away, _ = fit_power_law(s2, n2)
    tau_at = -tau_fit
    t3 = abs(tau_at - TAU_2D) < 0.4 and r2_at > 0.9 and r2_away < r2_at
    print(f"  T3 n_s ~ s^-tau     tau = {tau_at:.4f}  (exact {TAU_2D:.4f}, "
          f"err {abs(tau_at-TAU_2D):.4f})   R2 at p_c {r2_at:.3f} vs "
          f"{r2_away:.3f} at p=0.45   {'PASS' if t3 else 'FAIL'}")

    ok = t1 and t2 and t3
    print(f"\n  CALIBRATION: {'PASS' if ok else 'FAIL'} — "
          f"{'instruments may be used on DFT systems' if ok else 'do NOT use these yet'}")

    out = HERE.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"exp_01_calibration_{stamp}.json").write_text(json.dumps({
        "sizes": args.sizes, "samples": args.samples, "p": ps.tolist(),
        "exact": {"p_c": P_C_2D, "gamma_over_nu": GAMMA_OVER_NU_2D, "tau": TAU_2D},
        "measured": {"p_c": p_cross, "spread_ratio": ratio, "gamma_over_nu": gn,
                     "gamma_over_nu_r2": r2, "tau": float(tau_at),
                     "r2_at_pc": float(r2_at), "r2_away": float(r2_away)},
        "tests": {"T1_crossing": bool(t1), "T2_exponent": bool(t2), "T3_distribution": bool(t3)},
        "spanning": {str(L): span_curves[L].tolist() for L in args.sizes},
        "susceptibility": {str(L): chi_curves[L].tolist() for L in args.sizes},
        "order_parameter": {str(L): ord_curves[L].tolist() for L in args.sizes},
    }, indent=2), encoding="utf-8")
    print(f"  wrote results/exp_01_calibration_{stamp}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
