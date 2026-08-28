#!/usr/bin/env python3
"""M17 exp_02 — correlation length as a SCALED critical quantity.

Pre-registered: journals/2026-08-27_exp02_prereg.md. T1 and T2 are disclosed there as
postdictive (scouted before registration); only T3 and T4 are predictive.

The question is not "is the engine critical". It is whether the estimator that produced the
"xi pinned at the white-noise floor of 0.63 cells" reading -- the fourth route into M17's
retracted wall, and the one the 2026-08-17 retraction never examined -- can distinguish a
critical system from a sub-critical one at all.

Two estimators, same lattices, same run:

  A  structure.correlation_length   1/e decay of the DENSITY-FIELD autocorrelation.
                                    Documented white-noise floor 1 - 1/e = 0.6321.
  B  connectivity_length            second moment of the PAIR-CONNECTEDNESS function over
                                    finite clusters. The length that diverges at p_c.

2D site percolation is the target because the answers are exact:

    p_c = 0.5927460      nu = 4/3      gamma/nu = 43/24      tau = 187/91

Four tests:

  T1  Discrimination power D = |value(p_c) - median(off-crit)| / std(off-crit).
      Requires D_B >= 5, D_A < 1, D_B >= 5*D_A.   [postdictive]
  T2  B's peak lands within 0.02 of exact p_c for every L >= 64; A's does not.  [postdictive]
  T3  Data collapse of xi/L against (p - p_c)*L^(1/nu) recovers nu in [1.10, 1.60], biased
      BELOW 4/3 by 5-20% -- the same direction as exp_01's gamma/nu and tau.   [predictive]
  T4  Off-critical control, same instrument same run: B's L-scaling at FIXED p = 0.45 and
      p = 0.75 must give |alpha| < 0.25, against alpha ~ 1 at p_c, ratio >= 3.  [predictive]

    python exp_02_correlation_length.py [--sizes 32 64 128 256] [--samples 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))

import numpy as np  # noqa: E402

from criticality import (NU_2D, P_C_2D, collapse_residual,  # noqa: E402
                         connectivity_length, scaling_exponent, site_lattice)

# --- estimator A lives in reality-engine -------------------------------------------------
# Deliberate: `structure.correlation_length` is the instrument under test, so a vendored copy
# would test a copy rather than the thing that produced the 0.63 reading. One concept, one
# definition (STANDARDS §4). The milestone README flags cross-repo import vs a shared module
# as an open structural question for Block B; this experiment needs the ORIGINAL and says so
# loudly rather than degrading quietly if it is missing.
WHITE_NOISE_FLOOR = 1.0 - 1.0 / np.e          # 0.6321205588285577, per structure.selftest


def _load_structure():
    override = os.environ.get("REALITY_ENGINE_V4")
    candidates = [Path(override)] if override else []
    root = HERE.resolve()
    for _ in range(8):                          # walk up to the workspace root
        root = root.parent
        candidates.append(root / "reality-engine" / "proof_of_concepts" / "v4")
    for c in candidates:
        if (c / "structure.py").is_file():
            sys.path.insert(0, str(c))
            import structure                    # noqa: PLC0415
            return structure, str(c)
    raise SystemExit(
        "FATAL: could not locate reality-engine/proof_of_concepts/v4/structure.py.\n"
        "  Estimator A IS the instrument under test -- exp_02 cannot run without it and will\n"
        "  not substitute a reimplementation. Set REALITY_ENGINE_V4 to that directory."
    )


def build_grid() -> np.ndarray:
    """Fine through the critical region, coarse in the wings. Registered before running."""
    return np.unique(np.round(np.concatenate([
        np.arange(0.40, 0.50, 0.01),
        np.arange(0.50, 0.7001, 0.005),
        np.arange(0.71, 0.8001, 0.01),
    ]), 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    structure, re_path = _load_structure()
    ps = build_grid()
    sizes = sorted(args.sizes)

    print("M17 exp_02 — correlation length as a scaled critical quantity")
    print(f"  estimator A from {re_path}")
    print(f"  sizes {sizes}   samples {args.samples}   {len(ps)} values of p")
    print(f"  exact p_c = {P_C_2D:.7f}   exact nu = {NU_2D:.5f}")
    print(f"  estimator A white-noise floor = {WHITE_NOISE_FLOOR:.6f}\n")

    A = {L: np.full(len(ps), np.nan) for L in sizes}
    B = {L: np.full(len(ps), np.nan) for L in sizes}
    occupancy = {L: np.full(len(ps), np.nan) for L in sizes}
    excluded = {L: np.zeros(len(ps), dtype=int) for L in sizes}

    rng = np.random.default_rng(args.seed)
    for L in sizes:
        for i, p in enumerate(ps):
            a, b, occ_f, exc = [], [], [], 0
            for _ in range(args.samples):
                occ = site_lattice(L, float(p), rng, dims=2)
                # periodic=False is explicit: the estimator's default is the Mobius manifold's
                # per-axis geometry and is wrong on an open lattice.
                a.append(structure.correlation_length(occ.astype(float), axis=0,
                                                      periodic=False))
                xi, _, dropped = connectivity_length(occ, exclude_spanning=True)
                b.append(xi)
                exc += dropped
                occ_f.append(float(occ.mean()))
            A[L][i] = np.nanmean(a)
            B[L][i] = np.nanmean(b)
            occupancy[L][i] = float(np.mean(occ_f))
            excluded[L][i] = exc
        print(f"  L={L:4d} done   A range [{np.nanmin(A[L]):.3f}, {np.nanmax(A[L]):.3f}]"
              f"   B range [{np.nanmin(B[L]):.2f}, {np.nanmax(B[L]):.2f}]")

    i_pc = int(np.argmin(np.abs(ps - P_C_2D)))
    off = np.abs(ps - P_C_2D) > 0.08

    # --- secondary failure condition 2: the exclusion must actually have fired -------------
    hi = ps > 0.62
    exclusion_fired = all(int(excluded[L][hi].sum()) > 0 for L in sizes)
    print(f"\n  spanning clusters excluded above p=0.62: "
          f"{ {L: int(excluded[L][hi].sum()) for L in sizes} }   "
          f"{'OK' if exclusion_fired else 'VOID — exclusion never fired'}")

    # --- T1: discrimination power ---------------------------------------------------------
    def discrimination(curve):
        v_pc = curve[i_pc]
        med = float(np.nanmedian(curve[off]))
        sd = float(np.nanstd(curve[off]))
        return abs(v_pc - med) / sd if sd > 0 else float("nan")

    D_A = float(np.nanmean([discrimination(A[L]) for L in sizes]))
    D_B = float(np.nanmean([discrimination(B[L]) for L in sizes]))
    t1 = (D_B >= 5.0) and (D_A < 1.0) and (D_B >= 5.0 * D_A)
    print(f"\n  T1 discrimination   D_A = {D_A:.3f} (need < 1)   D_B = {D_B:.3f} (need >= 5)"
          f"   {'PASS' if t1 else 'FAIL'}   [postdictive]")
    for L in sizes:
        print(f"       L={L:4d}  A at p_c {A[L][i_pc]:.4f}  vs floor {WHITE_NOISE_FLOOR:.4f}"
              f"   |   B at p_c {B[L][i_pc]:6.2f}")

    # --- T2: location ---------------------------------------------------------------------
    peak_B = {L: float(ps[int(np.nanargmax(B[L]))]) for L in sizes}
    peak_A = {L: float(ps[int(np.nanargmax(A[L]))]) for L in sizes}
    t2_B = all(abs(peak_B[L] - P_C_2D) <= 0.02 for L in sizes if L >= 64)
    t2_A = any(abs(peak_A[L] - P_C_2D) <= 0.02 for L in sizes if L >= 64)
    t2 = t2_B and not t2_A
    print(f"\n  T2 location   B peaks {peak_B}   A peaks {peak_A}"
          f"\n       B within 0.02 of p_c for L>=64: {t2_B}   A locates p_c: {t2_A}"
          f"   {'PASS' if t2 else 'FAIL'}   [postdictive]")

    # --- T3: nu by data collapse of xi/L --------------------------------------------------
    near = np.abs(ps - P_C_2D) <= 0.10
    curves = {L: (B[L] / L)[near] for L in sizes}
    p_near = ps[near]
    nus = np.arange(0.60, 3.001, 0.01)
    res = np.array([collapse_residual(p_near, curves, P_C_2D, float(n)) for n in nus])
    if np.isfinite(res).any():
        nu_fit = float(nus[int(np.nanargmin(res))])
        nu_res = float(np.nanmin(res))
    else:
        nu_fit, nu_res = float("nan"), float("nan")
    # A minimiser that lands on the edge of its own scan has not found a minimum -- it has run
    # out of room. Reporting that as "nu = 3.00" is exactly the fault class this milestone
    # exists to catch: a number returned for an input that determines nothing.
    at_boundary = bool(np.isfinite(nu_fit) and
                       (nu_fit <= nus[0] + 1e-9 or nu_fit >= nus[-1] - 1e-9))
    dev = (nu_fit - NU_2D) / NU_2D
    t3 = (not at_boundary) and (1.10 <= nu_fit <= 1.60) and (-0.20 <= dev <= -0.05)
    print(f"\n  T3 collapse   nu = {nu_fit:.4f}  (exact {NU_2D:.4f}, "
          f"deviation {dev*100:+.1f}%)   residual {nu_res:.4f}"
          + ("   UNRESOLVED — minimiser hit the scan boundary" if at_boundary else "")
          + f"\n       need nu in [1.10, 1.60] AND deviation in [-20%, -5%] "
          f"(same direction as exp_01)   {'PASS' if t3 else 'FAIL'}   [predictive]")

    # --- T4: off-critical control, same instrument ----------------------------------------
    def alpha_at(p_target):
        j = int(np.argmin(np.abs(ps - p_target)))
        a, r2 = scaling_exponent(sizes, [B[L][j] for L in sizes])
        return a, r2, float(ps[j])

    a_pc, r2_pc, _ = alpha_at(P_C_2D)
    a_sub, r2_sub, p_sub = alpha_at(0.45)
    a_sup, r2_sup, p_sup = alpha_at(0.75)
    worst = max(abs(a_sub), abs(a_sup))
    ratio = abs(a_pc) / worst if worst > 0 else float("inf")
    t4 = (abs(a_sub) < 0.25) and (abs(a_sup) < 0.25) and (ratio >= 3.0)
    print(f"\n  T4 control    alpha at p_c    = {a_pc:+.4f}  (R2 {r2_pc:.3f})"
          f"\n                alpha at p={p_sub:.3f} = {a_sub:+.4f}  (R2 {r2_sub:.3f})  "
          f"need |a| < 0.25"
          f"\n                alpha at p={p_sup:.3f} = {a_sup:+.4f}  (R2 {r2_sup:.3f})  "
          f"need |a| < 0.25"
          f"\n                ratio {ratio:.2f} (need >= 3)   "
          f"{'PASS' if t4 else 'FAIL'}   [predictive]")

    ok = t1 and t2 and t3 and t4 and exclusion_fired
    predictive_ok = t3 and t4 and exclusion_fired
    print(f"\n  exp_02: {'PASS' if ok else 'FAIL'}   "
          f"(predictive tests T3+T4: {'PASS' if predictive_ok else 'FAIL'})")
    print(f"  Block A: {'B is licensed for Block B' if predictive_ok else 'B is NOT licensed'}")

    out = HERE.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_02_correlation_length_{stamp}.json"
    path.write_text(json.dumps({
        "prereg": "journals/2026-08-27_exp02_prereg.md",
        "sizes": sizes, "samples": args.samples, "seed": args.seed,
        "p": ps.tolist(),
        "reality_engine_path": re_path,
        "exact": {"p_c": P_C_2D, "nu": NU_2D},
        "white_noise_floor": float(WHITE_NOISE_FLOOR),
        "estimator_A_density_autocorr": {str(L): A[L].tolist() for L in sizes},
        "estimator_B_connectivity": {str(L): B[L].tolist() for L in sizes},
        "occupancy": {str(L): occupancy[L].tolist() for L in sizes},
        "spanning_excluded_count": {str(L): excluded[L].tolist() for L in sizes},
        "measured": {
            "D_A": D_A, "D_B": D_B,
            "A_at_pc": {str(L): float(A[L][i_pc]) for L in sizes},
            "B_at_pc": {str(L): float(B[L][i_pc]) for L in sizes},
            "peak_A": peak_A, "peak_B": peak_B,
            "nu": nu_fit, "nu_residual": nu_res, "nu_deviation": dev,
            "nu_at_scan_boundary": at_boundary,
            "alpha_pc": a_pc, "alpha_sub": a_sub, "alpha_sup": a_sup,
            "alpha_ratio": ratio,
        },
        "tests": {
            "T1_discrimination": bool(t1), "T2_location": bool(t2),
            "T3_collapse_nu": bool(t3), "T4_offcritical_control": bool(t4),
            "exclusion_fired": bool(exclusion_fired),
            "predictive_only": bool(predictive_ok),
        },
    }, indent=2), encoding="utf-8")
    print(f"  wrote results/{path.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
