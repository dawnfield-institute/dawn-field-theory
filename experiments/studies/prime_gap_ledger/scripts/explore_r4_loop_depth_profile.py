#!/usr/bin/env python3
"""explore_r4 (EXPLORING, unregistered — disclosed as pre-seal for round 4): the loop's OWN smooth-depth profile.

Round 3 derived the depth SHIFT exactly (log y_eff / log y = 1/Buchstab ratio, four decimals, fifteen cells) but its
other constant is measured, not derived: β_q, from δ_q(y) ~ (log y)^{−β_q}, fitted per modulus off two depths each
(0.85, 0.74, 0.83, 0.79, 0.78 for q = 3, 9, 4, 8, 5) and recorded, never explained. This asks whether δ_q(y) has a
closed form, on the cheapest object in the study: for k ≤ 9 the loop of units mod P_k is fully ENUMERABLE and δ_q is
an exact rational. No primes, no window sieving, no 10^10 read.

Two things worth seeing before any seal:
  (1) the mechanism. δ_q is carried by how often q divides a gap; measure P(q | g) directly beside δ_q and check the
      registration's identity δ_q = 1 − φ(q)·P(q | g), which holds only under uniform marginals — on a finite loop the
      marginals are not exactly uniform, so the gap between the two IS the marginal inhomogeneity, reported per cell.
  (2) the limit. As y grows the gaps grow, so P(q | g) → 1/q and δ_q → 1 − φ(q)/q, a NONZERO constant. A power law in
      log y cannot hold globally against a nonzero limit: β_q must decay. Round 1's recorded exact q = 3 deficits
      (5/12, 223/552, 2860783/8291520 = .4167, .4040, .3450) are already walking down toward 1 − φ(3)/3 = 1/3.

Semantics match round 1 exactly so the record reproduces: the enumerated loop rooted at 0, residues = residues_mod(
off, 0, q), a LINEAR transition matrix (as exp_01 line 100), exact rationals only where q | P (the primorial is
squarefree, which is why only q = 3 has exact values on record — no prime power ever divides it). Gaps are cyclic
(gaps_of(off, P), as exp_01 line 49); the one wrap transition the linear matrix omits is reported as wrap_note.

Writes results/explore_r4_loop_depth_<ts>.json. Nothing here is scored, nothing here is a prediction.
"""
import sys, json, time, math, argparse
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("enumerated", "sampled"), default="enumerated")
ap.add_argument("--kmax", type=int, default=R.LOOP_ENUM_KMAX, help="largest enumerated loop (k <= 9)")
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--qmax", type=int, default=120, help="largest modulus considered")
ap.add_argument("--windows", type=int, default=40, help="sampled mode: CRT-uniform windows per depth")
ap.add_argument("--L", type=int, default=2_000_000, help="sampled mode: window length")
ap.add_argument("--seed", type=int, default=20260908)
args = ap.parse_args()

primes = R.odd_sieve(200 if args.mode == "enumerated" else 200_000)


def phi(q):
    """Euler totient from the instrument's factorisation."""
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def admissible(P, y, qmax):
    """Moduli q with 3 <= q <= qmax that DIVIDE the primorial — the only ones with an exact rational deficit.
    The primorial is squarefree, so this is exactly the squarefree q built from primes <= y."""
    return [q for q in range(3, qmax + 1) if P % q == 0]


out = {"mode": "exploring", "registered": False, "generated": ts, "sweep": args.mode,
       "note": "unregistered exploration; disclosed in the round-4 seal's postdiction section",
       "depths": {}}
prev = {}


def sampled_sweep():
    """The enumerable loops stop at y = 23 (log log y = 1.14) — far too shallow to read a power law in log y;
    round 3 measured its beta_q at y = 44722 and 141422. Here the same delta_q on CRT-UNIFORM sampled loops
    across four decades of depth, with the window-scatter SE, so beta_q = -d log delta / d log log y can be
    read where round 3 read it and tested for constancy rather than assumed."""
    rng = np.random.default_rng(args.seed)
    QS = (3, 4, 5, 7, 8, 9)                     # round 3's scored set; prime powers by uniform lift (its G1)
    YS = (30, 60, 120, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 141422)
    for y in YS:
        ps = R.primes_upto(primes, y); t0 = time.time()
        sample = R.loop_sample(ps, args.L, args.windows, rng)
        per_q = {q: [] for q in QS}
        for o, dr in sample:
            for q in QS:
                r = R.residues_mod(o, R.n_mod_q_from_draws(q, dr, rng), q)
                per_q[q].append(R.diagonal_deficit(R.transition_matrix(r, q)))
        cells = {}
        print(f"\n=== y = {y} (k = {len(ps)}) log log y = {math.log(math.log(y)):.4f}  "
              f"W = {args.windows} x {args.L}  [{time.time() - t0:.0f}s]")
        print("     q    delta_q      SE        beta_local   saturated?")
        for q in QS:
            v = np.array([x for x in per_q[q] if x == x])
            d = float(v.mean()); se = float(v.std(ddof=1) / math.sqrt(len(v)))
            b = float("nan")
            if q in prev and prev[q]["delta"] > 0 and d > 0:
                dll = math.log(math.log(y)) - math.log(math.log(prev[q]["y"]))
                b = -(math.log(d) - math.log(prev[q]["delta"])) / dll
            sat = d > 0.9
            cells[str(q)] = dict(q=q, delta=d, se=se, n=int(len(v)), beta_local=b, saturated=bool(sat))
            print(f"   {q:3d}   {d:.6f}  {se:.6f}   {'' if math.isnan(b) else f'{b:+.4f}'}"
                  f"        {'SATURATED (not scored)' if sat else ''}")
            prev[q] = dict(y=y, delta=d)
        out["depths"][str(y)] = dict(y=y, k=len(ps), log_log_y=math.log(math.log(y)),
                                     windows=args.windows, L=args.L, seconds=round(time.time() - t0, 1), q=cells)
        (RES / f"explore_r4_sampled_depth_{ts}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/explore_r4_sampled_depth_{ts}.json")


if args.mode == "sampled":
    sampled_sweep(); sys.exit(0)

for k in range(args.kmin, args.kmax + 1):
    ps = primes[:k]; y = int(ps[-1]); t0 = time.time()
    off, P = R.loop_enumerate(ps)
    g = R.gaps_of(off, P)                                  # cyclic, as exp_01
    n_units = int(len(off)); mean_gap = float(g.mean())
    density = n_units / P
    mert = R.mertens_product(ps)
    qs = admissible(P, y, args.qmax)
    print(f"\n=== k = {k}  y = {y}  P = {P}  units = {n_units}  mean gap = {mean_gap:.4f} "
          f"(1/Mertens = {1 / mert:.4f})  moduli: {qs}")
    print("     q   phi(q)   delta_q (exact)            delta_q      1-phi/q     P(q|g)    1-phi*P    beta_local")
    cells = {}
    for q in qs:
        res = R.residues_mod(off, 0, q)
        T = R.transition_matrix(res, q); del res
        ex = R.diagonal_deficit_exact(T)
        d = float(ex) if ex is not None else float("nan")
        pq = float((g % q == 0).mean())                    # the mechanism, measured directly
        f = phi(q)
        ident = 1.0 - f * pq                               # exact only under uniform marginals
        limit = 1.0 - f / q
        b = float("nan")
        if q in prev and prev[q]["delta"] > 0 and d > 0:
            dll = math.log(math.log(y)) - math.log(math.log(prev[q]["y"]))
            if abs(dll) > 1e-12:
                b = -(math.log(d) - math.log(prev[q]["delta"])) / dll
        cells[str(q)] = dict(q=q, phi=f, exact=str(ex), delta=d, limit=limit, p_q_divides_gap=pq,
                             identity_uniform=ident, marginal_gap=d - ident, beta_local=b)
        print(f"   {q:3d}   {f:5d}   {str(ex)[:24]:<24} {d:.6f}   {limit:.6f}   {pq:.6f}  {ident:.6f}   "
              f"{'' if math.isnan(b) else f'{b:+.4f}'}")
        prev[q] = dict(y=y, delta=d)
    out["depths"][str(k)] = dict(k=k, y=y, P=str(P), units=n_units, density=density, mertens=mert,
                                 mean_gap=mean_gap, inv_mertens=1 / mert,
                                 wrap_note="gaps cyclic (incl. wrap); transition matrix linear, as round 1",
                                 seconds=round(time.time() - t0, 1), q=cells)
    del off, g
    (RES / f"explore_r4_loop_depth_{ts}.json").write_text(json.dumps(out, indent=1))

print(f"\nwrote results/explore_r4_loop_depth_{ts}.json")
