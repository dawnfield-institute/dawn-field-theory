"""Exact gap distribution of the loop of units mod P(y), by inclusion-exclusion.

Derived, not modelled. For a unit u, "the gap is exactly g" means u and u+g are units and none of the
interior odd positions u+2j (j = 1 .. g/2-1) is. Inclusion-exclusion over those interior positions:

    N(gap = g) = sum over S subset of {2,4,...,g-2} of (-1)^|S| T(S union {0, g})
    T(A)       = #{u mod P : u+a a unit for all a in A} = prod_{p <= y} (p - |A mod p|)

T(A) is a k-tuple count and is EXACT -- a product over primes, no approximation. Two facts make it cheap:
gaps concentrate at small g, so the interior is small; and for p > max(A) the offsets never collide, so
T(A)/phi(P) factors into a short product over p <= g times prod (p - |A|)/(p - 1) over the rest.

A term is exactly zero when A covers every residue class mod some p -- no u can avoid all of them.

This module knows nothing about phi_golden, Xi or Fibonacci. Every "phi" here is Euler's totient.
"""
import math
import functools
from itertools import combinations

__all__ = ["gap_dist_exact", "p_divides_gap", "delta_from_gaps"]


def gap_dist_exact(primes_upto_y, G):
    """{g: P(gap = g)} for even g <= G on the loop of units mod P(y). Exact up to float rounding."""
    ps = [int(p) for p in primes_upto_y]
    small = [p for p in ps if p <= G]
    big = [p for p in ps if p > G]

    @functools.lru_cache(maxsize=None)
    def big_log(c):
        # p > G >= |A| for every A considered, so p - c > 0 always
        return sum(math.log((p - c) / (p - 1)) for p in big)

    out = {}
    for g in range(2, G + 1, 2):
        interior = list(range(2, g, 2))
        tot = 0.0
        for size in range(len(interior) + 1):
            sgn = -1.0 if size % 2 else 1.0
            for S in combinations(interior, size):
                A = (0,) + S + (g,)
                lg = big_log(len(A))
                zero = False
                for p in small:
                    c = len({a % p for a in A})
                    if c >= p:                      # A covers every class mod p -> T(A) = 0
                        zero = True
                        break
                    lg += math.log((p - c) / (p - 1))
                if not zero:
                    tot += sgn * math.exp(lg)
        out[g] = tot
    return out


def p_divides_gap(exact, q, tail=None):
    """P(q | g) = exact part below G, plus an optional MEASURED tail above it.

    `tail` maps g -> P(gap = g) for g > G, from sampling. Truncation then becomes a small measured
    quantity rather than a bias -- for q with no multiple below G the exact part alone would be 0.
    Returns (total, exact_part, tail_part)."""
    ex = sum(v for g, v in exact.items() if g % q == 0)
    tl = 0.0 if tail is None else sum(v for g, v in tail.items() if g % q == 0)
    return ex + tl, ex, tl


def delta_from_gaps(pq, phi_q):
    """delta_q = 1 - phi(q) P(q | g). Exact under uniform marginals on the phi(q) unit classes,
    which is the cyclic convention (see the 2026-09-08 cyclic-convention note)."""
    return 1.0 - phi_q * pq
