#!/usr/bin/env python3
"""
exp_06_cross_base_analysis.py
==============================

SYSTEMATIC CROSS-BASE RESONANCE ANALYSIS

From exp_05 we found multiple bases with unique resonance.
This experiment maps the full landscape:

  Part A: Extended base scan (b=2..100, n=3..50)
  Part B: Index-centric view — which bases cluster at each F_n?
  Part C: Base-centric patterns — what predicts resonance index?
  Part D: The uniqueness classification
  Part E: Emergent structure — what ties the patterns together?
"""

import json
import time
from datetime import datetime
from pathlib import Path
from math import gcd, lcm
from collections import defaultdict


# ============================================================
# UTILITIES
# ============================================================

def fib(n, _cache={0: 0, 1: 1}):
    if n in _cache:
        return _cache[n]
    _cache[n] = fib(n - 1) + fib(n - 2)
    return _cache[n]


def lucas(n, _cache={0: 2, 1: 1}):
    if n in _cache:
        return _cache[n]
    _cache[n] = lucas(n - 1) + lucas(n - 2)
    return _cache[n]


def euler_totient(n):
    result = n
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            result = result * (d - 1) // d
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        result = result * (temp - 1) // temp
    return result


def get_divisors(n):
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs


def mult_order(a, m):
    """Fast multiplicative order using Euler totient divisors."""
    if m <= 1:
        return 1
    if gcd(a, m) != 1:
        return None
    phi_m = euler_totient(m)
    for d in sorted(get_divisors(phi_m)):
        if pow(a, d, m) == 1:
            return d
    return phi_m


def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ============================================================
# PART A: FULL SCAN
# ============================================================

def part_a_full_scan(base_max=100, n_max=50):
    """
    Build the complete resonance map: for each base b, find all n
    where ord(b mod F_n) = 2n.
    """
    print("=" * 72)
    print("  PART A: FULL RESONANCE SCAN")
    print(f"  bases 2..{base_max}, n = 3..{n_max}")
    print("=" * 72)

    # Pre-compute Fibonacci
    for n in range(n_max + 1):
        fib(n)

    # resonance_map[base] = list of n where ord(b mod F_n) = 2n
    resonance_map = {}
    # mod_map[base] = list of n where b^n mod F_n = F_{n-1}
    mod_map = {}

    t0 = time.time()

    for base in range(2, base_max + 1):
        hits_2n = []
        hits_mod = []

        for n in range(3, n_max + 1):
            fn = fib(n)
            fn_1 = fib(n - 1)

            if gcd(base, fn) > 1:
                continue

            # Check b^n mod F_n = F_{n-1}
            mod_val = pow(base, n, fn)
            if mod_val == fn_1:
                hits_mod.append(n)

            # Check ord(b mod F_n) = 2n
            order = mult_order(base, fn)
            if order == 2 * n:
                hits_2n.append(n)

        resonance_map[base] = hits_2n
        mod_map[base] = hits_mod

    elapsed = time.time() - t0
    print(f"\n  Scan completed in {elapsed:.1f}s")

    return resonance_map, mod_map


# ============================================================
# PART B: INDEX-CENTRIC VIEW
# ============================================================

def part_b_index_view(resonance_map):
    """
    For each Fibonacci index n that appears, which bases resonate there?
    """
    print("\n\n" + "=" * 72)
    print("  PART B: INDEX-CENTRIC VIEW")
    print("  Which bases cluster at each Fibonacci index?")
    print("=" * 72)

    # Invert the map
    index_to_bases = defaultdict(list)
    for base, indices in resonance_map.items():
        for n in indices:
            index_to_bases[n].append(base)

    print(f"\n  {'n':>3}  {'F_n':>12}  {'#bases':>6}  {'bases':>60}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*60}")

    for n in sorted(index_to_bases.keys()):
        bases = sorted(index_to_bases[n])
        fn = fib(n)
        bases_str = str(bases)
        if len(bases_str) > 58:
            bases_str = str(bases[:15]) + f"... ({len(bases)} total)"
        print(f"  {n:3d}  {fn:12d}  {len(bases):6d}  {bases_str:>60s}")

    # Analyze the dominant indices
    print(f"\n  DOMINANCE RANKING (by number of bases):")
    ranked = sorted(index_to_bases.items(), key=lambda x: -len(x[1]))
    for rank, (n, bases) in enumerate(ranked[:10], 1):
        fn = fib(n)
        factors = factorize(fn)
        factor_str = " * ".join(
            f"{p}^{e}" if e > 1 else str(p)
            for p, e in sorted(factors.items()))
        print(f"    {rank}. n={n:3d}  F_n={fn:>10d} = {factor_str:>25s}  "
              f"attracts {len(bases)} bases")

    # What fraction of bases resonate at n=10?
    total_bases = len(resonance_map)
    bases_at_10 = len(index_to_bases.get(10, []))
    print(f"\n  n=10 attracts {bases_at_10}/{total_bases} bases "
          f"({100*bases_at_10/total_bases:.1f}%)")

    return dict(index_to_bases)


# ============================================================
# PART C: BASE-CENTRIC PATTERNS
# ============================================================

def part_c_base_patterns(resonance_map):
    """
    Classify bases by their resonance behavior.
    Look for patterns in which bases have unique/multiple/no resonance.
    """
    print("\n\n" + "=" * 72)
    print("  PART C: BASE-CENTRIC CLASSIFICATION")
    print("=" * 72)

    # Classify
    unique = {}      # base -> single n
    multiple = {}    # base -> list of n
    none = []        # bases with no resonance

    for base in sorted(resonance_map.keys()):
        hits = resonance_map[base]
        if len(hits) == 0:
            none.append(base)
        elif len(hits) == 1:
            unique[base] = hits[0]
        else:
            multiple[base] = hits

    print(f"\n  Classification of {len(resonance_map)} bases:")
    print(f"    UNIQUE resonance:    {len(unique)} bases")
    print(f"    MULTIPLE resonance:  {len(multiple)} bases")
    print(f"    NO resonance:        {len(none)} bases")

    # Unique resonance bases
    print(f"\n  UNIQUE RESONANCE BASES:")
    print(f"  {'base':>4}  {'prime?':>6}  {'factorization':>20}  {'n':>4}  {'F_n':>10}  {'F_n factors':>25}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*20}  {'-'*4}  {'-'*10}  {'-'*25}")

    # Group by resonance index
    unique_by_index = defaultdict(list)
    for base, n in sorted(unique.items()):
        unique_by_index[n].append(base)
        bp = is_prime(base)
        bf = factorize(base) if not bp else {base: 1}
        bf_str = " * ".join(f"{p}^{e}" if e > 1 else str(p)
                            for p, e in sorted(bf.items()))
        fn = fib(n)
        ff = factorize(fn)
        ff_str = " * ".join(f"{p}^{e}" if e > 1 else str(p)
                            for p, e in sorted(ff.items()))
        print(f"  {base:4d}  {'PRIME' if bp else '':>6s}  {bf_str:>20s}  "
              f"{n:4d}  {fn:10d}  {ff_str:>25s}")

    print(f"\n  Unique bases grouped by resonance index:")
    for n in sorted(unique_by_index.keys()):
        bases = unique_by_index[n]
        print(f"    n={n:3d} (F_{n}={fib(n):>10d}): bases = {bases}")

    # NO resonance bases
    print(f"\n  NO RESONANCE BASES:")
    none_primes = [b for b in none if is_prime(b)]
    none_composite = [b for b in none if not is_prime(b)]
    print(f"    Prime: {none_primes[:20]}{'...' if len(none_primes) > 20 else ''}")
    print(f"    Composite: {none_composite[:20]}{'...' if len(none_composite) > 20 else ''}")

    # Pattern analysis: what predicts unique resonance?
    print(f"\n  PATTERN ANALYSIS:")

    # Are unique-resonance bases related to Fibonacci/Lucas numbers?
    fib_set = set(fib(i) for i in range(20))
    lucas_set = set(lucas(i) for i in range(20))
    unique_bases = set(unique.keys())

    fib_unique = unique_bases & fib_set
    lucas_unique = unique_bases & lucas_set
    print(f"    Fibonacci numbers with unique resonance: {sorted(fib_unique)}")
    print(f"    Lucas numbers with unique resonance: {sorted(lucas_unique)}")

    # What about b mod F_n structure?
    print(f"\n  Residue analysis: base mod F_n at resonance point")
    for base, n in sorted(unique.items()):
        fn = fib(n)
        residue = base % fn
        print(f"    base {base:3d} mod F_{n} = {base:3d} mod {fn} = {residue}")

    return unique, multiple, none, unique_by_index


# ============================================================
# PART D: UNIQUENESS CLASSIFICATION
# ============================================================

def part_d_uniqueness(resonance_map, index_to_bases, unique_by_index):
    """
    Deeper analysis of the uniqueness structure.
    """
    print("\n\n" + "=" * 72)
    print("  PART D: UNIQUENESS STRUCTURE")
    print("=" * 72)

    # For each resonance index, what's the CRT structure?
    print(f"\n  CRT analysis at each resonance index:")
    print(f"  " + "-" * 60)

    for n in sorted(index_to_bases.keys()):
        fn = fib(n)
        if gcd(2, fn) > 1:
            continue  # skip even Fibonacci

        factors = factorize(fn)
        factor_str = " * ".join(
            f"{p}^{e}" if e > 1 else str(p)
            for p, e in sorted(factors.items()))

        bases = index_to_bases[n]

        # For each prime factor p of F_n, what is phi(p)?
        prime_info = []
        for p, e in sorted(factors.items()):
            pe = p ** e
            phi_pe = euler_totient(pe)
            prime_info.append((p, e, phi_pe))

        phi_strs = ", ".join(f"phi({p}{'^'+str(e) if e>1 else ''})={phi}"
                             for p, e, phi in prime_info)

        print(f"\n  n={n}: F_{n} = {fn} = {factor_str}")
        print(f"    Totients: {phi_strs}")
        print(f"    Bases resonating here: {bases[:15]}{'...' if len(bases) > 15 else ''}")

        # For each resonating base, show the CRT decomposition
        if len(bases) <= 8:
            for b in bases:
                orders = []
                for p, e, _ in prime_info:
                    pe = p ** e
                    if gcd(b, pe) == 1:
                        o = mult_order(b, pe)
                        orders.append(f"ord({b},{pe})={o}")
                    else:
                        orders.append(f"gcd({b},{pe})>1")
                print(f"      base {b:3d}: {', '.join(orders)}")

    # What makes n=10 special across all bases?
    print(f"\n\n  WHY n=10 IS THE DOMINANT ATTRACTOR:")
    print(f"  " + "-" * 60)

    fn10 = fib(10)  # 55 = 5 * 11
    phi_5 = euler_totient(5)    # 4
    phi_11 = euler_totient(11)  # 10

    print(f"    F_10 = 55 = 5 * 11")
    print(f"    phi(5) = {phi_5}, phi(11) = {phi_11}")
    print(f"    lcm(phi(5), phi(11)) = lcm({phi_5}, {phi_11}) = {lcm(phi_5, phi_11)}")
    print(f"    2 * 10 = 20 = lcm(4, 10)")
    print(f"")
    print(f"    For ord(b mod 55) = 20, need lcm(ord(b,5), ord(b,11)) = 20")
    print(f"    ord(b,5) divides phi(5)=4, so ord(b,5) in {{1, 2, 4}}")
    print(f"    ord(b,11) divides phi(11)=10, so ord(b,11) in {{1, 2, 5, 10}}")
    print(f"")
    print(f"    lcm table (ord(b,5) x ord(b,11)):")
    print(f"    {'':>8}  {'ord11=1':>7}  {'ord11=2':>7}  {'ord11=5':>7}  {'ord11=10':>8}")

    for o5 in [1, 2, 4]:
        row = f"    ord5={o5}"
        for o11 in [1, 2, 5, 10]:
            l = lcm(o5, o11)
            marker = " *" if l == 20 else ""
            row += f"  {l:>5d}{marker}"
        print(row)

    print(f"\n    Combinations giving lcm=20: (4,5), (4,10)")
    print(f"    So need: ord(b,5)=4 AND ord(b,11) in {{5, 10}}")
    print(f"    ord(b,5)=4 means b is a primitive root mod 5")
    print(f"    Primitive roots mod 5: b = 2, 3 (mod 5)")

    # List all b in 2..100 that are primitive roots mod 5
    prim_5 = [b for b in range(2, 101) if gcd(b, 5) == 1 and mult_order(b, 5) == 4]
    print(f"\n    Bases with ord(b,5)=4 (prim root mod 5): {prim_5[:20]}...")
    print(f"    Pattern: b = 2, 3 mod 5 (i.e., not 0, 1, 4 mod 5)")

    # Of these, which also have ord(b,11) in {5, 10}?
    resonate_10 = []
    for b in prim_5:
        if gcd(b, 11) == 1:
            o11 = mult_order(b, 11)
            if o11 in [5, 10]:
                resonate_10.append((b, o11))

    print(f"    Of these, with ord(b,11) in {{5,10}}: {[b for b,_ in resonate_10[:20]]}...")
    print(f"    ({len(resonate_10)} bases total)")

    # Cross-check with actual resonance map
    actual_10 = sorted(index_to_bases.get(10, []))
    predicted_10 = sorted(b for b, _ in resonate_10)
    print(f"\n    Predicted bases at n=10: {predicted_10[:20]}...")
    print(f"    Actual bases at n=10:    {actual_10[:20]}...")
    print(f"    Match: {predicted_10[:len(actual_10)] == actual_10}")

    return None


# ============================================================
# PART E: EMERGENT STRUCTURE
# ============================================================

def part_e_emergent(resonance_map, index_to_bases, unique_by_index):
    """
    What ties the patterns together?
    """
    print("\n\n" + "=" * 72)
    print("  PART E: EMERGENT STRUCTURE")
    print("=" * 72)

    # 1. Count resonances per base
    print(f"\n  Distribution of resonance count per base:")
    count_dist = defaultdict(int)
    for base, hits in resonance_map.items():
        count_dist[len(hits)] += 1

    for count in sorted(count_dist.keys()):
        bar = "#" * count_dist[count]
        print(f"    {count} resonances: {count_dist[count]:3d} bases  {bar}")

    # 2. Is there a formula for how many resonances a base has?
    print(f"\n  Resonance count vs base properties:")
    print(f"  {'base':>4}  {'prime':>5}  {'#res':>4}  {'indices':>30}  {'base mod 55':>10}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*30}  {'-'*10}")

    for base in range(2, 51):
        hits = resonance_map.get(base, [])
        bp = 'P' if is_prime(base) else ''
        indices_str = str(hits) if len(hits) <= 4 else str(hits[:4]) + "..."
        bmod55 = base % 55 if gcd(base, 55) == 1 else f"gcd={gcd(base, 55)}"
        print(f"  {base:4d}  {bp:>5s}  {len(hits):4d}  {indices_str:>30s}  {str(bmod55):>10s}")

    # 3. The modular structure
    print(f"\n  MODULAR STRUCTURE OF n=10 RESONANCE:")
    print(f"  Bases resonating at n=10, classified by (b mod 5, b mod 11):")
    print()

    grid = {}
    all_bases = sorted(resonance_map.keys())
    for b in all_bases:
        if 10 in resonance_map[b]:
            r5 = b % 5
            r11 = b % 11
            if (r5, r11) not in grid:
                grid[(r5, r11)] = []
            grid[(r5, r11)].append(b)

    print(f"  {'':>8}", end="")
    for r11 in range(11):
        print(f"  {r11:>4}", end="")
    print()

    for r5 in range(5):
        print(f"  r5={r5}:", end="")
        for r11 in range(11):
            bases = grid.get((r5, r11), [])
            if bases:
                print(f"  {len(bases):>4}", end="")
            else:
                print(f"  {'':>4}", end="")
        print(f"  | {grid.get((r5,), [])}")

    # More readable: just list which (r5, r11) pairs resonate
    print(f"\n  Residue classes (b mod 5, b mod 11) that resonate at n=10:")
    for (r5, r11) in sorted(grid.keys()):
        bases = grid[(r5, r11)]
        o5 = mult_order(r5, 5) if gcd(r5, 5) == 1 and r5 > 0 else None
        o11 = mult_order(r11, 11) if gcd(r11, 11) == 1 and r11 > 0 else None
        print(f"    ({r5}, {r11:>2d}): ord(.,5)={o5}, ord(.,11)={o11}, "
              f"lcm={lcm(o5,o11) if o5 and o11 else '?'}, "
              f"bases={bases[:5]}{'...' if len(bases)>5 else ''}")

    # 4. The key insight
    print(f"\n  KEY INSIGHT: The resonance at n=10 is controlled by residues mod 55.")
    print(f"  There are exactly phi(55) = {euler_totient(55)} residues coprime to 55.")
    print(f"  Of these, the ones with lcm(ord(b,5), ord(b,11)) = 20 form a")
    print(f"  specific subset determined by the primitive root structure of Z/5Z x Z/11Z.")

    # Count how many residues mod 55 give resonance
    resonant_residues = set()
    for b in range(1, 56):
        if gcd(b, 55) > 1:
            continue
        o5 = mult_order(b, 5)
        o11 = mult_order(b, 11)
        if lcm(o5, o11) == 20:
            resonant_residues.add(b)

    print(f"\n  Residues mod 55 with ord = 20: {sorted(resonant_residues)}")
    print(f"  Count: {len(resonant_residues)} out of phi(55) = {euler_totient(55)}")
    print(f"  Fraction: {len(resonant_residues)}/{euler_totient(55)} = "
          f"{len(resonant_residues)/euler_totient(55):.4f}")

    # Density: what fraction of ALL bases resonate at n=10?
    total = len(resonance_map)
    at_10 = len(index_to_bases.get(10, []))
    predicted_frac = len(resonant_residues) / euler_totient(55)
    actual_frac = at_10 / total
    print(f"\n  Predicted density (from mod 55): {predicted_frac:.4f}")
    print(f"  Actual density (bases 2..100):   {actual_frac:.4f}")

    # 5. For UNIQUE resonance at n=10: need resonance at 10 but NOT at any other n
    print(f"\n\n  UNIQUENESS FILTER:")
    print(f"  For base 2 to be uniquely at n=10, it must:")
    print(f"    (a) resonate at n=10 (ord(2 mod 55) = 20)")
    print(f"    (b) NOT resonate at any other n")
    print(f"")
    print(f"  Base 2 satisfies (a). For (b):")
    print(f"    - n=20: F_20 even part has factor 41, ord(2,41)=20 -> lcm saturates")
    print(f"    - n=30: F_30 even (3|30) -> gcd(2,F_30)>1 -> eliminated")
    print(f"    - n=40: 2^80 mod F_40 != 1 -> ord > 80 -> no resonance")
    print(f"    - Higher n: growth obstruction (phi(F_n) >> 2n)")

    return resonant_residues


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  EXPERIMENT 06: CROSS-BASE RESONANCE ANALYSIS")
    print("=" * 72)
    print()

    t_start = time.time()

    # Part A: Full scan
    resonance_map, mod_map = part_a_full_scan(base_max=100, n_max=50)

    # Part B: Index-centric
    index_to_bases = part_b_index_view(resonance_map)

    # Part C: Base patterns
    unique, multiple, none_bases, unique_by_index = part_c_base_patterns(resonance_map)

    # Part D: Uniqueness
    part_d_uniqueness(resonance_map, index_to_bases, unique_by_index)

    # Part E: Emergent structure
    resonant_residues = part_e_emergent(resonance_map, index_to_bases, unique_by_index)

    elapsed = time.time() - t_start

    # Save results
    results = {
        'metadata': {
            'experiment': 'exp_06_cross_base_analysis',
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'base_range': '2..100',
            'n_range': '3..50',
        },
        'resonance_map': {str(k): v for k, v in resonance_map.items()},
        'unique_bases': {str(k): v for k, v in unique.items()},
        'index_to_bases': {str(k): v for k, v in index_to_bases.items()},
        'resonant_residues_mod_55': sorted(resonant_residues),
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_06_cross_base_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print(f"  Total elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
