#!/usr/bin/env python3
"""exp_04 gates — run and PASS before the round-4 seal (STANDARDS §2.7).

Round 4 registers that the loop's depth exponent beta_q, from delta_q(y) ~ (log y)^(-beta_q), depends on q only
through phi(q). These gates establish the instrument, the one exact theorem the design leans on, and — G7 — whether
the held-out classes are SCOREABLE AT ALL. A class whose moduli never leave saturation cannot be tested, and finding
that out after the seal would be finding it out too late.

G1  reproduces round 1's recorded exact loop deficits at q = 3 (5/12, 223/552, 2860783/8291520 at k = 3, 4, 6)
G2  the identity delta_{2q} = delta_q for ODD q, exactly, on every enumerated loop (proved; verified here)
G3  no two moduli inside a held-out class are related by that identity (else the class double-counts one measurement)
G4  loop mean gap = 1/mertens_product on every enumerated loop
G5  the sampler is unbiased: sampled delta_q agrees with the enumerated exact value within the window scatter
G6  exact reproducibility — the same seed reproduces delta_q bit for bit
G7  FEASIBILITY: every held-out modulus clears saturation (delta < 0.5) at the depths the round will read
G8  phi is computed correctly, and each held-out class spreads q widely at fixed phi (this is what decorrelates
    "beta tracks phi" from "beta tracks q" — the confound Phase A could not break)

Writes results/exp_04_gates_<ts>.json. Any gate FAIL is fatal and loud: the seal must not be written.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEED = 20260908
CLASSES = {8: (15, 16, 20, 24), 12: (13, 21, 28, 36), 16: (17, 32, 40, 48)}
ANCHOR = (3, 4, 5, 7, 8, 9)                      # Phase A's set, re-read here as the control arm
DEPTHS = (25000, 50000, 141422)                  # where the scored slopes will be taken
W_GATE, L_GATE = 24, 2_000_000

primes = R.odd_sieve(200_000)
gates, ok = {}, True


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def record(name, passed, claim, value):
    global ok
    gates[name] = dict(claim=claim, value=value, status="PASS" if passed else "FAIL")
    ok = ok and bool(passed)
    print(f"  {name}  {'PASS' if passed else '*** FAIL ***'}  {claim}\n        {value}")


def sampled_delta(y, qs, W=W_GATE, L=L_GATE, seed=SEED):
    """delta_q per modulus on CRT-uniform loops, with the window-scatter SE."""
    rng = np.random.default_rng(seed)
    ps = R.primes_upto(primes, y)
    per = {q: [] for q in qs}
    for o, dr in R.loop_sample(ps, L, W, rng):
        for q in qs:
            r = R.residues_mod(o, R.n_mod_q_from_draws(q, dr, rng), q)
            per[q].append(R.diagonal_deficit(R.transition_matrix(r, q)))
    out = {}
    for q in qs:
        v = np.array([x for x in per[q] if x == x])
        if len(v) < 8:                                            # loud, never a silent degrade
            raise RuntimeError(f"G-instrument: q={q} at y={y} produced only {len(v)} usable windows")
        out[q] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))))
    return out


print(f"exp_04 gates  seed={SEED}\n")

# ---- G1: the record reproduces ------------------------------------------------------------------------------------
want = {3: "5/12", 4: "223/552", 6: "2860783/8291520"}
got, g1 = {}, True
for k, w in want.items():
    off, P = R.loop_enumerate(primes[:k])
    e = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, 3), 3))
    got[f"k={k}"] = str(e); g1 = g1 and (str(e) == w)
    del off
record("G1", g1, "round 1's exact q=3 loop deficits reproduce", got)

# ---- G2: the identity, exactly, on enumerated loops ----------------------------------------------------------------
pairs, g2 = [], True
for k in range(3, 8):
    off, P = R.loop_enumerate(primes[:k])
    for q in [q for q in range(3, 60, 2) if P % q == 0]:
        a = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, q), q))
        b = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, 2 * q), 2 * q))
        g2 = g2 and (a == b); pairs.append(f"k={k},q={q}")
    del off
record("G2", g2, f"delta_2q == delta_q exactly for odd q ({len(pairs)} cases)", f"{len(pairs)}/{len(pairs)} identical")

# ---- G3: no class double-counts via that identity -------------------------------------------------------------------
dupes = []
for ph, qs in CLASSES.items():
    for a in qs:
        for b in qs:
            if a != b and b == 2 * a and a % 2 == 1:
                dupes.append((ph, a, b))
record("G3", not dupes, "no held-out class contains both q and 2q for odd q", dupes or "none — every class independent")

# ---- G4: mean gap is 1/Mertens --------------------------------------------------------------------------------------
mg, g4 = {}, True
for k in range(2, 9):
    ps = primes[:k]; off, P = R.loop_enumerate(ps)
    obs = float(R.gaps_of(off, P).mean()); exp = 1.0 / R.mertens_product(ps)
    mg[f"k={k}"] = dict(observed=obs, expected=exp); g4 = g4 and abs(obs - exp) < 1e-9
    del off
record("G4", g4, "loop mean gap == 1/mertens_product on every enumerated loop", f"max |diff| < 1e-9 over k=2..8")

# ---- G5: the sampler is unbiased against the exact value --------------------------------------------------------------
off, P = R.loop_enumerate(primes[:8]); y8 = int(primes[7])
exact8 = {q: float(R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, q), q)))
          for q in (3, 5, 7)}
del off
samp8 = sampled_delta(y8, (3, 5, 7))
g5, det = True, {}
for q in (3, 5, 7):
    m, se = samp8[q]; z = abs(m - exact8[q]) / se if se > 0 else 99
    det[q] = dict(exact=exact8[q], sampled=m, se=se, sigma=round(z, 2)); g5 = g5 and z < 3.0
record("G5", g5, f"sampled delta agrees with the enumerated exact value at y={y8} (<3 sigma)", det)

# ---- G6: exact reproducibility ----------------------------------------------------------------------------------------
a1 = sampled_delta(2500, (3, 5), W=8, seed=SEED)
a2 = sampled_delta(2500, (3, 5), W=8, seed=SEED)
record("G6", a1 == a2, "the same seed reproduces delta_q bit for bit", {str(k): v[0] for k, v in a1.items()})

# ---- G8 (computed before G7, which needs it): phi and the q-spread ------------------------------------------------------
spread, g8 = {}, True
for ph, qs in CLASSES.items():
    phis = {q: phi(q) for q in qs}
    good = all(v == ph for v in phis.values())
    spread[ph] = dict(q=list(qs), phi=phis, q_min=min(qs), q_max=max(qs), q_ratio=round(max(qs) / min(qs), 2))
    g8 = g8 and good and (max(qs) / min(qs) >= 1.5)
record("G8", g8, "each class has constant phi and a wide q-spread (decorrelates phi from q)", spread)

# ---- G7: feasibility — do the held-out moduli clear saturation where we will read? ---------------------------------------
allq = tuple(sorted({q for qs in CLASSES.values() for q in qs} | set(ANCHOR)))
feas, g7 = {}, True
for y in DEPTHS:
    d = sampled_delta(y, allq)
    feas[y] = {q: round(d[q][0], 4) for q in allq}
for ph, qs in CLASSES.items():
    for q in qs:
        worst = max(feas[y][q] for y in DEPTHS)      # shallowest depth is the hardest
        if worst >= 0.5:
            g7 = False
record("G7", g7, "every held-out modulus clears saturation (delta < 0.5) at all three scored depths", feas)

payload = dict(script="exp_04_gates.py", generated=ts, seed=SEED, classes={str(k): list(v) for k, v in CLASSES.items()},
               anchor=list(ANCHOR), depths=list(DEPTHS), windows=W_GATE, L=L_GATE,
               all_pass=ok, gates=gates)
(RES / f"exp_04_gates_{ts}.json").write_text(json.dumps(payload, indent=1))
print(f"\nwrote results/exp_04_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
