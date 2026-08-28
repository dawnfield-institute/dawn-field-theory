"""exp_24 -- Adversarial replication of exp_08 Panel C (z-immune straddling-pair signal).

Midnight's forward program names "adversarial replication of z-immune survivors"; Panel C was
recorded as single-pass and unreplicated. EXPLORATORY -- no pre-registration, no thresholds,
no scoring (STANDARDS 2.8, exploring-vs-predicting).

Three attacks, in order of severity:

  1. SEPARATION CONFOUND. Panel C claims z-immunity because both absorbers share a sightline.
     But "straddles an integer N" is largely a proxy for "far apart in N", and under any smooth
     trend |dEW| would grow with |dN| regardless of transitions. Sharing a sightline does not
     help -- the confound is WITHIN the pair.
  2. STRATIFIED PERMUTATION. Shuffle the straddle label only WITHIN |dN| bins, so separation is
     matched by construction. Exact and non-parametric. Plus leave-one-bin-out.
  3. PLACEBO THRESHOLDS. Offset the transition set across [0,1) and re-run. If straddling an
     arbitrary threshold does the same thing, the effect is real but NOT about cascade
     transitions.

RESULT (2026-08-28). Panel C reproduces exactly. The confound EXISTS in the covariate --
straddling pairs are 2.6x more separated -- but has no path to the outcome,
corr(|dN|,|dEW|) = +0.0075. The effect SURVIVES |dN|-matching at z ~ +4, and leave-one-bin-out
keeps p <= 0.004, so no single bin carries it. BUT integer N is not special: offsets of
+0.20..+0.26 reproduce it with the identical 19-bin structure, and across a 50-point sweep
integer N sits at empirical p ~ 0.10.

  => The OBSERVATION stands and is not a separation artifact.
     The INTERPRETATION -- that it evidences cascade transitions at integer N -- does not.

Caveat: usable bin counts vary across the sweep (19 near offset 0 and 0.18-0.26, but 10-12
around 0.32-0.50), so it is not strictly like-for-like everywhere. The load-bearing comparison
-- integer N against offsets sharing all 19 bins -- is clean.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
sys.path.insert(0, str(MIDNIGHT_ROOT / "scripts"))
from phase_rate import save_midnight_results, _convert_numpy  # noqa: E402

import importlib.util as _iu  # noqa: E402
_spec = _iu.spec_from_file_location("exp08", MIDNIGHT_ROOT / "scripts" / "exp_08_cascade_panel.py")
e8 = _iu.module_from_spec(_spec); _spec.loader.exec_module(e8)

TRANSITIONS = [2, 3, 4, 5, 6, 7]
NBINS, MIN_PER_CELL = 21, 30


def build_pairs():
    """All within-sightline absorber pairs, with cascade positions and |dEW|."""
    m = e8.load_mgii()
    z, ew = np.asarray(m["z"]), np.asarray(m["ew1"])
    key = (np.asarray(m["plate"]).astype(np.int64) * 10**9
           + np.asarray(m["mjd"]).astype(np.int64) * 10**4
           + np.asarray(m["fiber"]).astype(np.int64))
    # n_at_z integrates per call; interpolate THE SAME function on a fine grid
    zg = np.linspace(float(z.min()), float(z.max()), 3000)
    N = np.interp(z, zg, np.array([e8.n_at_z(v) for v in zg]))
    groups = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    lo, hi, dew = [], [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for i, j in combinations(idx, 2):
            a, b = (N[i], N[j]) if N[i] < N[j] else (N[j], N[i])
            lo.append(a); hi.append(b); dew.append(abs(ew[i] - ew[j]))
    return np.array(lo), np.array(hi), np.array(dew), len(z)


def straddle_mask(lo, hi, offset=0.0):
    s = np.zeros(len(lo), bool)
    for t in (x + offset for x in TRANSITIONS):
        s |= (lo < t) & (t < hi)
    return s


def weighted_gap(dew, strad, bins, usable):
    keep = np.isin(bins, usable)
    tot = 0.0
    for bb in usable:
        mm = bins == bb
        tot += (np.median(dew[mm & strad]) - np.median(dew[mm & ~strad])) * mm.sum()
    return tot / keep.sum()


def usable_bins(strad, bins):
    return [bb for bb in np.unique(bins)
            if ((bins == bb) & strad).sum() >= MIN_PER_CELL
            and ((bins == bb) & ~strad).sum() >= MIN_PER_CELL]


def permute(dew, strad, bins, usable, nperm, seed):
    obs = weighted_gap(dew, strad, bins, usable)
    rng = np.random.default_rng(seed)
    null = np.empty(nperm)
    for t in range(nperm):
        lab = strad.copy()
        for bb in usable:
            mm = bins == bb
            lab[mm] = rng.permutation(strad[mm])
        null[t] = weighted_gap(dew, lab, bins, usable)
    return obs, (1 + np.sum(null >= obs)) / (1 + nperm), (obs - null.mean()) / null.std()


def main(nperm=1500, seed=7):
    lo, hi, dew, n_abs = build_pairs()
    dN = hi - lo
    strad = straddle_mask(lo, hi, 0.0)
    edges = np.unique(np.quantile(dN, np.linspace(0, 1, NBINS)))
    bins = np.clip(np.digitize(dN, edges[1:-1]), 0, len(edges) - 2)
    usable = usable_bins(strad, bins)

    print(f"absorbers {n_abs}   pairs {len(dN)}   straddling {int(strad.sum())}")
    print("\n[1] reproduce Panel C")
    mw = mannwhitneyu(dew[strad], dew[~strad], alternative="greater").pvalue
    print(f"    medians  straddle {np.median(dew[strad]):.4f}  non {np.median(dew[~strad]):.4f}"
          f"   MW p={mw:.3e}")
    print("\n[2] the separation confound")
    r = float(np.corrcoef(dN, dew)[0, 1])
    print(f"    median |dN|  straddle {np.median(dN[strad]):.4f}  non {np.median(dN[~strad]):.4f}"
          f"   ({np.median(dN[strad])/np.median(dN[~strad]):.1f}x)")
    print(f"    corr(|dN|,|dEW|) = {r:+.4f}   <- no path to the outcome")

    print("\n[3] |dN|-matched stratified permutation")
    obs, p, zs = permute(dew, strad, bins, usable, nperm, seed)
    print(f"    bins {len(usable)}   gap {obs:+.5f}   p={p:.4f}   z={zs:+.2f}")
    loo = {}
    for bb in usable:
        u2 = [x for x in usable if x != bb]
        loo[int(bb)] = permute(dew, strad, bins, u2, 600, seed + 1)[1]
    print(f"    leave-one-bin-out: max p = {max(loo.values()):.4f}")

    print("\n[4] placebo threshold offsets")
    sweep = {}
    for off in np.arange(0.0, 1.0, 0.02):
        s = straddle_mask(lo, hi, float(off))
        u = usable_bins(s, bins)
        if len(u) >= 8:
            sweep[round(float(off), 2)] = (weighted_gap(dew, s, bins, u), len(u))
    real = sweep[0.0][0]
    gaps = np.array([v[0] for v in sweep.values()])
    beat = int((gaps >= real).sum())
    print(f"    REAL gap {real:+.5f}   sweep mean {gaps.mean():+.5f}  sd {gaps.std():.5f}")
    print(f"    offsets >= REAL: {beat}/{len(gaps)}  -> empirical p = {beat/len(gaps):.3f}"
          f"   z = {(real-gaps.mean())/gaps.std():+.2f}")
    same19 = {k: v for k, v in sweep.items() if v[1] == sweep[0.0][1]}
    print(f"    like-for-like (same {sweep[0.0][1]} bins): "
          + ", ".join(f"{k}:{v[0]:+.4f}" for k, v in sorted(same19.items())[:8]))

    print("\n  OBSERVATION survives |dN|-matching. INTERPRETATION (integer N) does not.")
    save_midnight_results("exp_24_panelC_adversarial_replication", _convert_numpy({
        "experiment": "exp_24_panelC_adversarial_replication",
        "initiative": "midnight", "mode": "exploratory_no_scoring",
        "n_absorbers": n_abs, "n_pairs": int(len(dN)), "n_straddling": int(strad.sum()),
        "panelC_reproduction": {"median_straddle": float(np.median(dew[strad])),
                                "median_non": float(np.median(dew[~strad])),
                                "mannwhitney_p": float(mw)},
        "separation_confound": {"median_dN_straddle": float(np.median(dN[strad])),
                                "median_dN_non": float(np.median(dN[~strad])),
                                "corr_dN_dEW": r},
        "matched_permutation": {"n_bins": len(usable), "gap": float(obs),
                                "p": float(p), "z": float(zs),
                                "leave_one_out_max_p": float(max(loo.values()))},
        "offset_sweep": {"real_gap": float(real), "n_offsets": len(gaps),
                         "n_at_or_above_real": beat,
                         "empirical_p": float(beat / len(gaps)),
                         "gaps": {str(k): float(v[0]) for k, v in sweep.items()},
                         "n_bins": {str(k): int(v[1]) for k, v in sweep.items()}},
        "verdict": "observation survives dN-matching; integer-N interpretation not supported",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
