#!/usr/bin/env python3
"""explore_r4 cyclic convention (EXPLORING, unregistered): the enumerated loop's transition matrix has ends it
should not have.

exp_01 formed the loop's exact deficits with cyclic GAPS (`gaps_of(off, P)`, which carries the wrap) and a
LINEAR transition matrix (`res[:-1] -> res[1:]`, which does not). The loop is defined in this study as bounded
and BOUNDARYLESS; a gap is the loop cut open. The linear matrix applies the local object's convention to the
global one, and it shows: every recorded denominator is divisible by phi(P)/2 - 1 — 3, 23, 239, 2879, 46079
for phi(P) = 8, 48, 480, 5760, 92160 (prime for k<=6, and 46079 = 11*59*71 at k=7) — the
boundary term, not arithmetic.

Closing the loop removes it: 2860783/8291520 becomes 497/1440. Every cyclic denominator here is 5-smooth.

NOTHING SCORED MOVES. Every scored cell in rounds 1-3 used SAMPLED loops, which are true arcs with two ends,
where the linear matrix is correct. The artifact is confined to the enumerated exact rationals (recorded, not
claimed) and is O(1/phi(P)): 17% at k=3, 0.03% by k=6. exp_04's gate G1 deliberately reproduces the LINEAR
values — a gate matches the record as it stands.

Writes results/explore_r4_cyclic_convention_<ts>.json.
"""
import sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

try:
    from sympy import factorint
except ImportError as e:                      # loud, never a silent degrade (STANDARDS: silent-failure trap)
    raise SystemExit(f"sympy is required for the factorisation column: {e}")

QS = (3, 5, 7)
primes = R.odd_sieve(200)
out = {"mode": "exploring", "registered": False, "generated": ts,
       "claim": "the enumerated loop's transition matrix should be cyclic; the linear one injects phi(P)/2 - 1",
       "scope": "nothing scored moves — scored cells used sampled loops, which are true arcs",
       "depths": {}}


def cyclic_matrix(res, q):
    """Close the loop: the wrap transition last -> first, which a boundaryless object has and a linear read drops."""
    return R.transition_matrix(np.concatenate([res, res[:1]]), q)


print(f"{'k':>2}{'y':>4}{'phi(P)':>9}{'q':>4}   {'linear (round 1)':<26}{'cyclic':<20}{'phi/2-1':>10}  divides?")
for k in range(3, 8):
    ps = primes[:k]; off, P = R.loop_enumerate(ps); phiP = R.phi_of_primorial(ps)
    cells = {}
    for q in QS:
        if P % q:
            continue
        res = R.residues_mod(off, 0, q)
        lin = R.diagonal_deficit_exact(R.transition_matrix(res, q))
        cyc = R.diagonal_deficit_exact(cyclic_matrix(res, q))
        del res
        big_l = max(factorint(lin.denominator)) if lin.denominator > 1 else 1
        cyc_fac = factorint(cyc.denominator) if cyc.denominator > 1 else {}
        # the boundary term need not be PRIME -- at k=7 it is 46079 = 11*59*71, and all three divide.
        # Test divisibility, not primality (an earlier version tested the largest prime and reported a false 'no').
        bt = phiP // 2 - 1
        divides = bool(bt > 1 and lin.denominator % bt == 0)
        cells[str(q)] = dict(q=q, linear=str(lin), cyclic=str(cyc),
                             linear_denom_factors={str(a): b for a, b in factorint(lin.denominator).items()},
                             cyclic_denom_factors={str(a): b for a, b in cyc_fac.items()},
                             linear_largest_prime=int(big_l), phi_half_minus_1=bt,
                             boundary_term_divides_linear_denom=divides,
                             boundary_term_factors={str(a): b for a, b in factorint(bt).items()} if bt > 1 else {},
                             cyclic_is_5_smooth=bool(all(int(p) <= 5 for p in cyc_fac)),
                             abs_diff=float(abs(lin - cyc)))
        print(f"{k:>2}{int(ps[-1]):>4}{phiP:>9}{q:>4}   {str(lin):<26}{str(cyc):<20}"
              f"{bt:>10}  {'DIVIDES' if divides else 'no'}")
    out["depths"][str(k)] = dict(k=k, y=int(ps[-1]), P=str(P), phi_P=phiP, q=cells)
    del off

(RES / f"explore_r4_cyclic_convention_{ts}.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote results/explore_r4_cyclic_convention_{ts}.json")
