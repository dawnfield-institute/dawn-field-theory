#!/usr/bin/env python3
"""explore_r4 qsweep (EXPLORING, unregistered): delta_q over a wide modulus range at three depths.

Written to break the phi/delta-level confound of the first r4 sweep. Because phi(q) is NON-monotone in q, a
fixed phi class spans a wide range of phi/q -- and delta follows phi/q -- so this sweep supplies both controlled
comparisons: beta at fixed phi with delta varying, and beta at matched delta with the modulus varying.

q = 2*odd is EXCLUDED: delta_{2q} = delta_q exactly for odd q (proved and gated in exp_04_gates.py, G2/G3),
so those are not independent cells and including them would count one measurement twice.

Writes results/explore_r4_qsweep_<ts>.json. Nothing here is scored.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")
SEED, W, L = 20260908, 32, 2_000_000
import os
DEPTHS = tuple(int(v) for v in os.environ.get('R4_DEPTHS', '25000,50000,141422').split(','))
def phi(q):
    n = 1
    for p, a in R.factor_int(q).items(): n *= (p - 1) * p ** (a - 1)
    return n
# every q in 3..60 EXCEPT 2*odd (delta_2q == delta_q -- our theorem; they are not independent cells)
QS = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
primes = R.odd_sieve(200_000)
out = {"mode": "exploring", "registered": False, "sweep": "qsweep_matched_delta",
       "note": "find moduli at MATCHED delta with DIFFERENT phi -- the pair that separates "
               "'beta = f(phi)' from 'beta = f(delta-level)'. Unregistered.",
       "generated": ts, "seed": SEED, "excluded": "q = 2*odd (identity delta_2q = delta_q)", "depths": {}}
for y in DEPTHS:
    rng = np.random.default_rng(SEED); ps = R.primes_upto(primes, y); per = {q: [] for q in QS}
    t0 = time.time()
    for o, dr in R.loop_sample(ps, L, W, rng):
        for q in QS:
            r = R.residues_mod(o, R.n_mod_q_from_draws(q, dr, rng), q)
            per[q].append(R.diagonal_deficit(R.transition_matrix(r, q)))
    cells = {}
    for q in QS:
        v = np.array([x for x in per[q] if x == x])
        if len(v) < 8: raise RuntimeError(f"q={q} y={y}: only {len(v)} usable windows")
        cells[str(q)] = dict(q=q, phi=phi(q), delta=float(v.mean()),
                             se=float(v.std(ddof=1)/math.sqrt(len(v))), n=int(len(v)))
    out["depths"][str(y)] = dict(y=y, log_log_y=math.log(math.log(y)), q=cells,
                                 seconds=round(time.time()-t0,1))
    print(f"y={y} done [{time.time()-t0:.0f}s]", flush=True)
    (RES / f"explore_r4_qsweep_{ts}.json").write_text(json.dumps(out, indent=1))
print(f"wrote results/explore_r4_qsweep_{ts}.json")
