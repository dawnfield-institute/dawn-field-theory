"""exp_25 -- Is the SECOND MOMENT structured? Residual-variance test across channels.

EXPLORATORY -- no pre-registration, no thresholds, no scoring (STANDARDS 2.8).

MOTIVATION. Almost every signal that has survived anywhere in Midnight lives in a WIDTH,
SPREAD or SHAPE, and almost none in a mean: exp_05 EW spread, exp_03 line widths (220%),
exp_06 FWHM discrepancy, exp_07 CIV Doppler b spread, exp_08D doublet shape. If an unmodelled
primitive shows up as excess variance, a corpus whose surviving signals all live in the
variance is the expected signature -- and those results are the PRIMARY channel rather than
weaker versions of a mean-shift.

THE TEST. exp_07 showed z-detrending kills the mean-level EW signal. So: after removing a
smooth trend in z, does the residual VARIANCE still track cascade disequilibrium?

  model contains : a quadratic trend in z. Nothing else. No cascade term is fitted.
  statistic      : Spearman rho between per-z-bin residual sd and per-bin mean disequilibrium
  control        : shuffle z, preserving the one-point distribution exactly and destroying
                   all z-structure; 200 draws give the null for rho.

SCOPE NOTE, recorded because it was a live error. The first pass ran on EW spread -- the one
channel exp_07 had ALREADY killed -- and returned a clean null. Correct scope is the channels
that SURVIVED detrending. Same class of mistake as probing a global observable with a local
initial condition; see the 2026-08-28 journal.

RESULT (2026-08-28).
  FWHM discrepancy    rho=+0.585  z=+3.71  p<=0.005   skew +1.43  exkurt  +1.74
  FWHM ratio          rho=+0.483  z=+2.93  p<=0.005   skew +6.61  exkurt +82.38  (outlier-driven)
  log FWHM 2796       rho=+0.237  z=+1.61  p=0.139
  log FWHM product    rho=+0.034  z=+0.19  p=0.871
  log EW 2796         rho=-0.050  z=-0.43  p=0.816    (dead, exactly as exp_07 found)

The signal is in the DOUBLET DISCREPANCY -- the difference between two lines of the same
doublet, in the same absorber, at the same redshift. Not in either width alone, not in the
product. That is a relational within-scope quantity (exp_23's frame-clean form), which is what
the invariant-registration rule says should survive.

Both moments agree: exp_06 found the MEAN discrepancy anticorrelates with disequilibrium
(lines lock together at transitions); this finds the VARIANCE rises away from them.

LIMITS. p <= 0.005 is the permutation floor (200 draws), not a measured value. The FWHM ratio
is discounted -- skew 6.6 and excess kurtosis 82 mean outliers drive it, while the discrepancy
is well behaved. And this was NOT blind: channels were chosen because exp_07 reports they
survived, so it is a targeted test of an existing claim rather than a discovery. The nulls are
what make it credible -- a fishing expedition does not produce this pattern of specific nulls
beside a specific survivor.
"""
from __future__ import annotations

import importlib.util as iu
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew, spearmanr

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import save_midnight_results, _convert_numpy  # noqa: E402

_spec = iu.spec_from_file_location("exp08", MIDNIGHT_ROOT / "scripts" / "exp_08_cascade_panel.py")
e8 = iu.module_from_spec(_spec); _spec.loader.exec_module(e8)

NBINS, MIN_PER_BIN, NPERM = 40, 50, 200


def load():
    m = e8.load_mgii()
    z = np.asarray(m["z"])
    zg = np.linspace(z.min(), z.max(), 3000)
    N = np.interp(z, zg, np.array([e8.n_at_z(v) for v in zg]))
    return m, z, N, np.abs(N - np.round(N))


def variance_vs_diseq(z, y, diseq, seed=0):
    """rho between per-bin residual sd and per-bin disequilibrium, with a shuffled-z null."""
    edges = np.unique(np.quantile(z, np.linspace(0, 1, NBINS + 1)))
    b = np.clip(np.digitize(z, edges[1:-1]), 0, len(edges) - 2)

    def stat(res):
        v, d = [], []
        for bb in np.unique(b):
            mm = b == bb
            if mm.sum() < MIN_PER_BIN:
                continue
            v.append(res[mm].std()); d.append(diseq[mm].mean())
        return spearmanr(d, v)[0], len(v)

    resid = y - np.polyval(np.polyfit(z, y, 2), z)
    rho, nb = stat(resid)
    rng = np.random.default_rng(seed)
    null = np.empty(NPERM)
    for i in range(NPERM):
        yy = y[rng.permutation(len(z))]
        null[i] = stat(yy - np.polyval(np.polyfit(z, yy, 2), z))[0]
    return dict(rho=float(rho), n_bins=int(nb),
                z_score=float((rho - null.mean()) / null.std()),
                p=float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + NPERM)),
                skew=float(skew(resid)), excess_kurtosis=float(kurtosis(resid)),
                null_mean=float(null.mean()), null_sd=float(null.std()))


def main():
    m, z, N, diseq = load()
    fw1, fw2, ew1 = np.asarray(m["fw1"]), np.asarray(m["fw2"]), np.asarray(m["ew1"])
    channels = {
        "fwhm_discrepancy": np.abs(fw1 - fw2) / ((fw1 + fw2) / 2),
        "fwhm_ratio": fw1 / fw2,
        "log_fwhm_2796": np.log(fw1),
        "log_fwhm_product": np.log(fw1 * fw2),
        "log_ew_2796_known_dead": np.log(ew1),
    }
    print(f"absorbers {len(z)}   z {z.min():.3f}-{z.max():.3f}   N {N.min():.2f}-{N.max():.2f}")
    print(f"{'channel':<26}{'rho':>9}{'z':>8}{'p':>9}{'skew':>9}{'exkurt':>10}")
    out = {}
    for name, y in channels.items():
        r = variance_vs_diseq(z, y, diseq)
        out[name] = r
        print(f"{name:<26}{r['rho']:+9.4f}{r['z_score']:+8.2f}{r['p']:9.4f}"
              f"{r['skew']:+9.3f}{r['excess_kurtosis']:+10.3f}")
    print("\n  signal is in the DOUBLET DISCREPANCY, not either width alone, not the product.")
    print("  EW spread is dead, exactly as exp_07 reported.")
    save_midnight_results("exp_25_second_moment_structure", _convert_numpy({
        "experiment": "exp_25_second_moment_structure",
        "initiative": "midnight", "mode": "exploratory_no_scoring",
        "n_absorbers": int(len(z)), "n_permutations": NPERM,
        "model_contains": "quadratic trend in z only; no cascade term fitted",
        "control": "shuffled z, one-point distribution preserved",
        "channels": out,
        "verdict": ("second moment of the doublet discrepancy carries cascade structure after "
                    "mean detrending; EW spread does not; targeted not blind; p floor 0.005"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
