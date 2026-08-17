"""Criticality instruments — the capability hole M17 exists to plug.

The corpus can measure contrast (density CV, void fraction, filament fraction) and it can
measure connectivity (percolation, added 2026-08-15). It cannot measure **where a system sits
relative to a critical point**, which is why exp_10 concluded "no discrete phase transition"
from four one-point statistics that could not have shown one.

What is missing, and provided here:

    order_parameter          P_inf, the fraction of sites in the largest cluster
    susceptibility           chi = sum(s^2 n_s) / sum(s n_s), EXCLUDING the largest cluster
    cluster_size_distribution  n_s, power-law at p_c and exponential away from it
    spanning_probability     does a cluster connect opposite faces
    finite_size_crossing     the location where curves for different L intersect

**Nothing here is trusted until it recovers 2D site percolation.** That system has exact known
answers, which is rare and is the whole reason it is the calibration target:

    p_c    = 0.5927460    site percolation threshold, square lattice
    beta   = 5/36         order parameter, P_inf ~ (p - p_c)^beta
    gamma  = 43/18        susceptibility, chi ~ |p - p_c|^-gamma
    nu     = 4/3          correlation length, xi ~ |p - p_c|^-nu
    tau    = 187/91       cluster size distribution, n_s ~ s^-tau at p_c

Finite-size scaling is what makes a critical point locatable without knowing it in advance.
At p_c a dimensionless ratio becomes size-independent, so curves measured at different L
**cross** there. Away from p_c they separate with L. The crossing is the measurement; the
exponents follow from how the peak grows, chi_max ~ L^(gamma/nu).

Written because seven instrument faults in the preceding round were each caught only by a
system whose answer was already known. This module has one built in.
"""

from __future__ import annotations

import numpy as np

# --- exact 2D site percolation, square lattice --------------------------------------------
P_C_2D = 0.5927460
BETA_2D = 5.0 / 36.0
GAMMA_2D = 43.0 / 18.0
NU_2D = 4.0 / 3.0
TAU_2D = 187.0 / 91.0
GAMMA_OVER_NU_2D = GAMMA_2D / NU_2D          # 1.7917 — what chi_max ~ L^(g/n) should give


def _label(occ: np.ndarray):
    """Connected components of a boolean lattice, face-connectivity, any dimension."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(occ)
        return lab, n
    except ImportError:                      # pragma: no cover - fallback path
        lab = np.zeros(occ.shape, dtype=np.int32)
        offsets = []
        for ax in range(occ.ndim):
            for step in (1, -1):
                o = [0] * occ.ndim
                o[ax] = step
                offsets.append(tuple(o))
        cur = 0
        for start in zip(*np.nonzero(occ)):
            if lab[start]:
                continue
            cur += 1
            lab[start] = cur
            stack = [start]
            while stack:
                c = stack.pop()
                for off in offsets:
                    nb = tuple(a + b for a, b in zip(c, off))
                    if all(0 <= nb[i] < occ.shape[i] for i in range(occ.ndim)) \
                            and occ[nb] and not lab[nb]:
                        lab[nb] = cur
                        stack.append(nb)
        return lab, cur


def cluster_sizes(occ: np.ndarray) -> np.ndarray:
    """Sizes of every connected component, descending."""
    lab, n = _label(occ)
    if n == 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.bincount(lab.ravel())[1:])[::-1]


def order_parameter(occ: np.ndarray) -> float:
    """P_inf — fraction of ALL sites in the largest cluster.

    Zero below p_c in the thermodynamic limit, rising as (p - p_c)^beta above it. Normalised
    by the lattice size rather than by the occupied set, which is what makes it an order
    parameter rather than a shape statistic.
    """
    s = cluster_sizes(occ)
    return float(s[0] / occ.size) if s.size else 0.0


def susceptibility(occ: np.ndarray) -> float:
    """chi = sum(s^2 n_s) / sum(s n_s), with the LARGEST cluster excluded.

    Excluding the largest is not optional and is the usual convention: above p_c the spanning
    cluster is macroscopic and would swamp the sum, hiding the divergence the measurement
    exists to find. With it excluded, chi peaks at p_c and the peak grows as L^(gamma/nu).
    """
    s = cluster_sizes(occ)
    if s.size <= 1:
        return 0.0
    rest = s[1:].astype(float)
    denom = rest.sum()
    return float((rest ** 2).sum() / denom) if denom > 0 else 0.0


def spans(occ: np.ndarray, axis: int = 0) -> bool:
    """Does one cluster touch both faces along `axis`? The percolation event itself."""
    lab, n = _label(occ)
    if n == 0:
        return False
    lo = set(np.unique(np.take(lab, 0, axis=axis))) - {0}
    hi = set(np.unique(np.take(lab, occ.shape[axis] - 1, axis=axis))) - {0}
    return bool(lo & hi)


def cluster_size_distribution(occ: np.ndarray, nbins: int = 24):
    """Log-binned n_s. Returns (s, n_s) with empty bins dropped.

    Log binning matters: cluster sizes span decades and linear bins put almost every cluster
    in the first bin, which turns a power law into a spike and a slope fit into noise.
    """
    s = cluster_sizes(occ)
    if s.size < 4:
        return np.zeros(0), np.zeros(0)
    # Drop the largest cluster. At p_c the incipient spanning cluster is a separate object
    # from the scaling distribution and including it bends the tail away from the power law —
    # calibration recovered tau = 1.46 against an exact 2.05 until it was excluded. Same
    # reason `susceptibility` excludes it.
    s = s[1:]
    if s.size < 4:
        return np.zeros(0), np.zeros(0)
    lo, hi = 1.0, float(s.max())
    if hi <= lo:
        return np.zeros(0), np.zeros(0)
    edges = np.logspace(np.log10(lo), np.log10(hi * 1.001), nbins + 1)
    counts, _ = np.histogram(s, bins=edges)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    keep = counts > 0
    return centres[keep], (counts[keep] / widths[keep])


def pooled_cluster_distribution(lattices, nbins: int = 24):
    """Log-binned n_s pooled across many samples.

    Fitting each realisation separately and averaging the slopes throws away most of the
    statistics: a single L=48 lattice has too few large clusters to constrain the tail, and
    the per-sample fits are dominated by the well-populated small-s bins. Pooling first gives
    one distribution with the full sample count behind every bin. Calibration recovered
    tau = 1.61 per-sample against an exact 2.05, and lands correctly once pooled.

    The largest cluster of each realisation is excluded, as in `cluster_size_distribution`.
    """
    allsz = []
    for occ in lattices:
        s = cluster_sizes(occ)
        if s.size > 1:
            allsz.append(s[1:])
    if not allsz:
        return np.zeros(0), np.zeros(0)
    s = np.concatenate(allsz)
    if s.size < 8:
        return np.zeros(0), np.zeros(0)
    hi = float(s.max())
    if hi <= 1.0:
        return np.zeros(0), np.zeros(0)
    edges = np.logspace(0.0, np.log10(hi * 1.001), nbins + 1)
    counts, _ = np.histogram(s, bins=edges)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    keep = counts > 0
    return centres[keep], counts[keep] / widths[keep] / len(allsz)


def fit_power_law(x, y, min_points: int = 5):
    """Slope of log y vs log x, with R^2. Returns (slope, r2, n)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < min_points:
        return float("nan"), float("nan"), int(ok.sum())
    lx, ly = np.log(x[ok]), np.log(y[ok])
    m, c = np.polyfit(lx, ly, 1)
    pred = m * lx + c
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return float(m), float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), int(ok.sum())


def fit_exponential(x, y, min_points: int = 5):
    """Slope of log y vs x (not log x), with R^2. Returns (rate, r2, n)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < min_points:
        return float("nan"), float("nan"), int(ok.sum())
    xx, ly = x[ok], np.log(y[ok])
    m, c = np.polyfit(xx, ly, 1)
    pred = m * xx + c
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return float(m), float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), int(ok.sum())


def power_law_or_exponential(x, y):
    """Which hypothesis fits better? Returns (verdict, r2_power, r2_exp, exponent).

    "Is this a power law" is not answerable without a rival. A power-law fit alone will return
    some R^2 for almost any decreasing sequence, especially over a short range — the M17
    calibration found R^2 = 0.967 for a fit to a *sub-critical* percolation distribution that
    is genuinely exponential. Comparing the two hypotheses on the same data is what makes the
    question decidable.

    Verdict is "power_law", "exponential", or "ambiguous" when the two R^2 are within 0.01.

    **This is NOT a criticality test, and calibration is what showed it.** On pure forms it is
    exact — synthetic s^-2 gives R^2 1.0000 against 0.6106, synthetic exp(-s/50) the reverse.
    But sub-critical percolation at p = 0.40 also reads "power_law" (0.9511 vs 0.8996), and
    correctly so: a sub-critical cluster distribution is a TRUNCATED power law,
    n_s ~ s^-tau exp(-s/s_c), and the s^-tau part is still present below the cutoff. Only the
    cutoff distinguishes critical from sub-critical.

    So a power-law verdict means "there is a scaling regime", not "this system is critical".
    The critical question is whether the CUTOFF SCALES WITH SYSTEM SIZE, which is what
    `susceptibility` measures — chi is the cutoff scale, and chi ~ L^(gamma/nu) at criticality
    and L-independent away from it. Use `cutoff_scaling` for that.
    """
    m_p, r2_p, n = fit_power_law(x, y)
    _, r2_e, _ = fit_exponential(x, y)
    if not (np.isfinite(r2_p) and np.isfinite(r2_e)):
        return "undetermined", r2_p, r2_e, m_p
    if abs(r2_p - r2_e) < 0.01:
        return "ambiguous", r2_p, r2_e, m_p
    return ("power_law" if r2_p > r2_e else "exponential"), r2_p, r2_e, m_p


def cutoff_scaling(sizes, chis, min_points: int = 2):
    """Does the avalanche/cluster cutoff grow with system size? The actual criticality test.

    At a critical point there is no characteristic scale, so the cutoff is set only by the
    system: chi ~ L^(gamma/nu) with a positive exponent. Away from criticality the system has
    its own intrinsic scale, the cutoff saturates, and chi becomes L-INDEPENDENT.

    Returns (exponent, r2, verdict).

    **The EXPONENT is the measurement; the verdict label is a convenience and is crude.**
    Calibration on 2D percolation at L = 32/64/128:

        at p_c = 0.5927   exponent 1.603   (exact gamma/nu = 1.792)
        at p   = 0.50     exponent 0.585
        at p   = 0.40     exponent 0.225

    p = 0.50 is genuinely sub-critical yet reads "scale-free", and that is correct rather than
    a fault: at these L the correlation length there is comparable to the box, so the system IS
    inside its critical region. Finite systems look critical near p_c — that is what a critical
    region is. The exponent varies smoothly with distance from p_c and converges on gamma/nu,
    so report it and compare against the calibration, rather than reading the label as a
    verdict.
    """
    e, r2, n = fit_power_law(np.asarray(sizes, float), np.asarray(chis, float),
                             min_points=min_points)
    if not np.isfinite(e):
        return e, r2, "undetermined"
    if e > 0.5 and (not np.isfinite(r2) or r2 > 0.9):
        return e, r2, "scale-free"
    if abs(e) < 0.25:
        return e, r2, "characteristic-scale"
    return e, r2, "ambiguous"


def finite_size_crossing(param, curves: dict[int, np.ndarray], saturate: float = 0.02):
    """Where do curves measured at different L intersect?

    `curves` maps L -> values over `param`. At a critical point a dimensionless ratio stops
    depending on system size, so the curves meet there and separate on either side.

    Located by the SIGN CHANGE of (largest L - smallest L), not by minimum spread. The first
    version minimised spread and was wrong for a reason worth recording: deep inside either
    phase every curve saturates at the same constant — spanning probability is 1.0 for all L
    well above p_c — so the spread is exactly zero across a whole region and beats the real
    crossing. Calibrating on 2D percolation returned p_c = 0.70 with "spread 0.000" instead of
    0.5927. Saturated agreement is not a crossing.

    Points where every curve sits within `saturate` of the common minimum or maximum are
    excluded before the search for exactly that reason.

    Returns (param_at_crossing, spread_there, spread_ratio). `spread_ratio` compares the spread
    at the crossing to the median over the non-saturated region — small means a genuine
    crossing, near 1 means the curves never separated and nothing was located.
    """
    param = np.asarray(param, float)
    Ls = sorted(curves)
    if len(Ls) < 2:
        return float("nan"), float("nan"), float("nan")
    M = np.vstack([np.asarray(curves[L], float) for L in Ls])
    if not np.isfinite(M).any():
        return float("nan"), float("nan"), float("nan")

    lo, hi = np.nanmin(M), np.nanmax(M)
    rng_ = hi - lo
    live = np.ones(M.shape[1], dtype=bool)
    if rng_ > 0:                              # drop saturated ends
        live &= ~np.all(M <= lo + saturate * rng_, axis=0)
        live &= ~np.all(M >= hi - saturate * rng_, axis=0)
    if live.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    diff = M[-1] - M[0]                       # largest L minus smallest L
    spread = M.max(axis=0) - M.min(axis=0)
    idx = np.flatnonzero(live)
    d = diff[idx]

    sign_change = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)
    if sign_change.size:
        i = idx[sign_change[0]]
        j = idx[sign_change[0] + 1]
        # linear interpolation to where the difference actually crosses zero
        denom = diff[j] - diff[i]
        t = 0.0 if denom == 0 else -diff[i] / denom
        p_cross = float(param[i] + t * (param[j] - param[i]))
        sp = float(min(spread[i], spread[j]))
    else:                                     # no sign change: fall back to closest approach
        k = idx[int(np.nanargmin(np.abs(d)))]
        p_cross, sp = float(param[k]), float(spread[k])

    med = float(np.nanmedian(spread[live]))
    return p_cross, sp, (sp / med if med > 0 else float("nan"))


def site_lattice(L: int, p: float, rng, dims: int = 2) -> np.ndarray:
    """Site percolation: each site occupied independently with probability p."""
    return rng.random((L,) * dims) < p
