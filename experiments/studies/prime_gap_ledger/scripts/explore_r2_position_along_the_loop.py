#!/usr/bin/env python3
"""explore_r2 (EXPLORING, unregistered — disclosed as pre-seal for round 2): the loop read at different POSITIONS.
Peter's reading of exp_01's residual (2026-09-07): the primes are not a random arc of the loop, they are the arc at the
ORIGIN — the stretch below y^2 where every residue is the integer's own name and all moduli agree at once (the fully
actualized identity); far from the origin the residues decorrelate and the uniform loop is right. So the delta is not a
third object: it is the inhomogeneity of ONE object along its own length. Test: fix the depth y, slide a window of
length L along the loop from the origin outward (N = 10^m; only N mod p is needed) and measure the consecutive-survivor
residue bias δ3 against the position's own depth u_N = log N / log y. Prediction (Peter): the bias climbs from the
primes' value at u_N ≈ 2 to the uniform loop's value and stays there — a function of u_N, collapsing across y.
Writes results/explore_r2_position_<ts>.json. Nothing here is scored."""
import sys, json, time, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")
primes = R.odd_sieve(200000); rng = np.random.default_rng(1); out = {"mode": "exploring", "depths": {}}
def bias_at(N, ps, L, q=3):
    off = R.segmented_rough(R.window_residues(N, ps), ps, L)
    return R.diagonal_deficit(R.transition_matrix(R.residues_mod(off, N % q, q), q)), int(len(off))
def loop_bias(ps, L, W, q=3):
    Ts = 0
    for _ in range(W):
        res = R.loop_residues(ps, rng); off = R.segmented_rough(res, ps, L); d = dict(zip((int(p) for p in ps), res))
        Ts = Ts + R.transition_matrix(R.residues_mod(off, R.n_mod_q_from_draws(q, d, rng), q), q)
    return R.diagonal_deficit(Ts)
for y, L in ((142, 10**6), (448, 10**6), (1415, 2 * 10**6), (4473, 4 * 10**6)):
    ps = R.primes_upto(primes, y); loop = loop_bias(ps, L, 12); rows = []
    print(f"\n=== depth y = {y} (k = {len(ps)}); uniform loop δ3 = {loop:.4f}; window L = {L}")
    print("   position N      u_N=logN/logy   δ3(window)   loop−δ3   survivors")
    for e in (4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50):
        N = 10 ** e
        if N < (y * y) // 4: continue
        d, n = bias_at(N, ps, L); uN = math.log(N) / math.log(y); prime_arc = abs(N - y * y / 2) / N < 0.6
        rows.append(dict(exp=e, u_N=uN, bias=d, loop_minus=loop - d, survivors=n, primes_arc=prime_arc))
        print(f"   1e{e:<3}         {uN:6.2f}         {d:.4f}       {loop - d:+.4f}   {n}{' <- the primes' if prime_arc else ''}")
    out["depths"][str(y)] = dict(k=int(len(ps)), L=L, loop_bias=loop, rows=rows)
(RES / f"explore_r2_position_{ts}.json").write_text(json.dumps(out, indent=1)); print(f"\nwrote results/explore_r2_position_{ts}.json")
