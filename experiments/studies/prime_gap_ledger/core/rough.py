"""prime_gap_ledger core — y-rough numbers in a window and on the loop of units mod a primorial.

The objects (registration: journals/2026-09-07_exp01_registration.md):
  LOCAL  — the y-rough integers (coprime to every prime ≤ y) in the window [N, N + L). For N = 10^m,
           L = 10^m and y ≥ sqrt(2·10^m) these are exactly the primes of the decade.
  GLOBAL — the loop of units mod P(y) = ∏_{p ≤ y} p: bounded, boundaryless, in CRT coordinates. It is
           sampled CRT-UNIFORMLY: draw N mod p independently and uniformly for every p ≤ y and sieve a
           segment of length L with those residues. A segment at an explicit huge integer N is NOT
           uniform on the loop — it sits on the number line at depth u = log N / log y — so residues
           are drawn, never derived from an offset. For k ≤ 9 primes the whole loop is enumerated.
One sieve routine serves both: segmented_rough(residues, primes, L). Depth is u = log x / log y.
Everything is numpy and exact integers; nothing here knows about φ, Ξ or Fibonacci.
"""
import math
from fractions import Fraction
import numpy as np

EULER_GAMMA = 0.5772156649015329
LOOP_ENUM_KMAX = 9                     # P_9 = 223,092,870 (0.22 GB alive array); k = 10 is 6.5 GB — excluded
G_MAX = 200                            # fixed gap cap with an overflow bin (a data-dependent cap is a knob)
SCALED_EDGES = np.round(np.arange(0.0, 6.0 + 1e-9, 0.1), 3)   # gap / mean-gap bins, fixed; overflow beyond 6
U_GRID = (6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.25, 2.0)
Q_LIST = (3, 4, 10)

__all__ = ["EULER_GAMMA", "LOOP_ENUM_KMAX", "G_MAX", "SCALED_EDGES", "U_GRID", "Q_LIST", "odd_sieve",
           "primes_upto", "primorial", "phi_of_primorial", "segmented_rough", "window_residues",
           "loop_residues", "loop_enumerate", "gaps_of", "gap_hist", "scaled_hist", "tv", "mertens_product",
           "buchstab_omega", "omega_at", "transition_matrix", "diagonal_deficit", "diagonal_deficit_exact",
           "residues_mod", "n_mod_q_from_draws", "depth_y", "lpf_table", "termination_depths", "null_depths",
           "pair_count_formula", "loop_sample"]


# ---- primes and the sieve ------------------------------------------------------------------------------------
def odd_sieve(X):
    """Primes ≤ X as int64 (odd-only sieve; index i ↔ 2i + 1). 2·10^8 in ~5 s, 0.1 GB."""
    X = int(X)
    n = (X - 1) // 2 + 1
    s = np.ones(n, dtype=bool)
    s[0] = False
    for p in range(3, math.isqrt(X) + 1, 2):
        if s[p >> 1]:
            s[(p * p) >> 1::p] = False
    return np.concatenate(([2], 2 * np.flatnonzero(s) + 1)).astype(np.int64)


def primes_upto(primes, y):
    return primes[primes <= y]


def primorial(ps):
    P = 1
    for p in ps:
        P *= int(p)
    return P


def phi_of_primorial(ps):
    f = 1
    for p in ps:
        f *= int(p) - 1
    return f


def segmented_rough(residues, primes, L):
    """Offsets k ∈ [0, L) with N + k coprime to every prime in `primes`, given residues[i] = N mod primes[i].
    Σ L/p ≈ 3L slice writes. Used identically for the window (residues of N) and the loop (drawn residues)."""
    alive = np.ones(int(L), dtype=bool)
    for p, r in zip(primes, residues):
        p = int(p)
        alive[(-int(r)) % p::p] = False
    return np.flatnonzero(alive)


def window_residues(N, primes):
    return [int(N) % int(p) for p in primes]


def loop_residues(primes, rng):
    """CRT-uniform point on the loop: an independent uniform residue per prime."""
    return [int(rng.integers(0, int(p))) for p in primes]


def loop_enumerate(primes_k):
    """The whole loop of units mod P_k as offsets in [0, P_k) (k ≤ LOOP_ENUM_KMAX)."""
    assert len(primes_k) <= LOOP_ENUM_KMAX, "loop enumeration is capped at k = 9"
    P = primorial(primes_k)
    return segmented_rough([0] * len(primes_k), primes_k, P), P


def loop_sample(primes_y, L, W, rng):
    """W CRT-uniform windows of length L on the loop mod P(y). Returns a list of (offsets, draws) where draws
    maps each prime to its drawn residue (needed to know N mod q for q | P(y))."""
    out = []
    for _ in range(W):
        res = loop_residues(primes_y, rng)
        out.append((segmented_rough(res, primes_y, L), dict(zip((int(p) for p in primes_y), res))))
    return out


# ---- gaps and histograms ------------------------------------------------------------------------------------------
def gaps_of(offsets, period=None):
    """Consecutive differences; on the loop (period given) the wrap-around gap closes the cycle."""
    g = np.diff(offsets)
    if period is not None and len(offsets) >= 1:
        g = np.append(g, int(period) - int(offsets[-1]) + int(offsets[0]))   # one unit: its gap is the whole period
    return g


def gap_hist(g, G=G_MAX):
    """Probability vector over gaps 1..G plus one overflow bin (position G ↔ gaps > G)."""
    gc = np.minimum(g, G + 1)
    h = np.bincount(gc, minlength=G + 2)[1:].astype(float)
    return h / h.sum()


def scaled_hist(g, edges=SCALED_EDGES):
    """The SHAPE: gaps in units of the sample's own mean gap, fixed edges, overflow beyond the last edge."""
    s = g / float(np.mean(g))
    full = np.append(edges, edges[-1] + 0.1)
    h, _ = np.histogram(np.minimum(s, edges[-1] + 0.05), bins=full)
    return h / h.sum()


def tv(h1, h2):
    return 0.5 * float(np.abs(np.asarray(h1) - np.asarray(h2)).sum())


# ---- the density words: Mertens and Buchstab -------------------------------------------------------------------------
def mertens_product(primes):
    """∏_{p ≤ y} (1 − 1/p): the loop's density, exactly (as a float of an exact rational)."""
    num, den = 1, 1
    for p in primes:
        p = int(p)
        num *= p - 1
        den *= p
    return num / den


def buchstab_omega(u_max=8.0, h=1e-3):
    """ω(u) on a grid: ω(u) = 1/u on [1, 2]; (u ω(u))' = ω(u − 1) for u > 2 (trapezoid). ω(2) = ½, ω(∞) → e^{−γ}."""
    grid = np.arange(1.0, u_max + h / 2, h)
    w = np.empty_like(grid)
    n1 = int(round(1.0 / h))
    for i, u in enumerate(grid):
        if u <= 2.0 + 1e-12:
            w[i] = 1.0 / u
        else:
            w[i] = (grid[i - 1] * w[i - 1] + 0.5 * h * (w[i - 1 - n1] + w[i - n1])) / u
    return grid, w


def omega_at(u, table=None):
    grid, w = table if table is not None else buchstab_omega()
    return float(np.interp(u, grid, w))


# ---- residues and transitions --------------------------------------------------------------------------------------
def residues_mod(offsets, n_mod_q, q):
    return ((offsets % q) + n_mod_q) % q


def n_mod_q_from_draws(q, draws, rng):
    """N mod q for a CRT-uniform loop point given the drawn residues N mod p. q = 3: the draw at 3. q = 10: CRT of the
    draws at 2 and 5. q = 4: the draw at 2 lifts to two classes mod 4 — pick one uniformly (the loop mod 2P(y))."""
    if q == 3:
        return draws[3]
    if q == 10:
        r2, r5 = draws[2], draws[5]
        return next(v for v in range(10) if v % 2 == r2 and v % 5 == r5)
    if q == 4:
        return draws[2] + 2 * int(rng.integers(0, 2))
    raise ValueError(q)


def transition_matrix(res, q):
    """Counts T[a, b] of consecutive residues a → b."""
    idx = res[:-1] * q + res[1:]
    return np.bincount(idx, minlength=q * q).reshape(q, q)


def diagonal_deficit(T):
    """Relative deficit of the diagonal mass against the product-of-marginals null: 1 − Σ_a T[a,a] / Σ_a row_a col_a / n."""
    T = np.asarray(T, dtype=float)
    n = T.sum()
    if n == 0:
        return float("nan")
    row, col = T.sum(1), T.sum(0)
    expected = float((row * col).sum() / n)
    if expected == 0.0:                       # too few transitions for the null to have any diagonal mass: undefined
        return float("nan")
    return 1.0 - float(np.trace(T)) / expected


def diagonal_deficit_exact(T):
    """The same, as an exact rational (for enumerated loops); None where undefined."""
    T = [[int(v) for v in r] for r in np.asarray(T)]
    q = len(T)
    n = sum(map(sum, T))
    row = [sum(r) for r in T]
    col = [sum(T[a][b] for a in range(q)) for b in range(q)]
    expected_num = sum(row[a] * col[a] for a in range(q))
    if n == 0 or expected_num == 0:
        return None
    return 1 - Fraction(sum(T[a][a] for a in range(q))) / Fraction(expected_num, n)


def pair_count_formula(g, primes_k):
    """Exact count of units r mod P_k with r + g also a unit: ∏_{p | g} (p − 1) · ∏_{p ∤ g} (p − 2) (the CRT product
    behind the Hardy–Littlewood weight)."""
    c = 1
    for p in primes_k:
        p = int(p)
        c *= (p - 1) if g % p == 0 else (p - 2)
    return c


# ---- depth and the cascade ---------------------------------------------------------------------------------------------
def depth_y(x, u):
    """y for depth u: x^{1/u}; at u = 2 exactly, y = ⌈sqrt(2x)⌉ so that the window's y-rough numbers are the primes."""
    if abs(u - 2.0) < 1e-12:
        return math.isqrt(2 * int(x)) + 1
    return int(round(float(x) ** (1.0 / u)))


def lpf_table(N, L, primes_sqrt):
    """Least prime factor of N + k, k ∈ [0, L), among the given primes (decreasing order: the smallest prime writes last).
    0 where no listed prime divides — the primes of the window when primes_sqrt reaches sqrt(N + L)."""
    t = np.zeros(int(L), dtype=np.uint16 if int(np.max(primes_sqrt)) < 65536 else np.uint32)
    for p in np.asarray(primes_sqrt)[::-1]:
        p = int(p)
        t[(-int(N)) % p::p] = p
    return t


def termination_depths(prime_offsets, table):
    """max lpf over each interior (p_i, p_{i+1}); prime positions hold 0 so the max is over the composites."""
    return np.maximum.reduceat(table, prime_offsets)[:-1]


def null_depths(interior_counts, table, rng, chunk=20_000_000):
    """Matched null: for each gap with c interior integers, the max lpf of c independent composites of the same window.
    Composites are drawn as uniform offsets with the primes rejected (no index array of the composites is materialised,
    so the window can be 10^9 long)."""
    Lt = len(table)
    out = np.empty(len(interior_counts), dtype=np.uint32)
    i = 0
    n = len(interior_counts)
    while i < n:
        j = i
        tot = 0
        while j < n and tot + int(interior_counts[j]) <= chunk:
            tot += int(interior_counts[j])
            j += 1
        if j == i:
            j = i + 1
            tot = int(interior_counts[i])
        cs = interior_counts[i:j]
        parts, have = [], 0
        while have < tot:
            cand = table[rng.integers(0, Lt, size=int(1.1 * (tot - have)) + 64)]
            cand = cand[cand > 0]
            parts.append(cand)
            have += len(cand)
        vals = np.concatenate(parts)[:tot]
        starts = np.concatenate(([0], np.cumsum(cs)[:-1])).astype(np.int64)
        out[i:j] = np.maximum.reduceat(vals, starts)
        i = j
    return out
