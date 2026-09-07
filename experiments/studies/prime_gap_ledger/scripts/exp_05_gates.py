#!/usr/bin/env python3
"""exp_05 gates — run and PASS before the round-5 seal (STANDARDS §2.7).

Round 5 registers the collapse found in exp_04's exploring: delta_q(y) = F(phi(q)/gbar) with gbar = e^gamma log y,
one curve for the whole family, with a residual structured by omega(q). These gates fix the instrument AND the
procedure — the split, the binning and the tolerance are all decided here, before any held-out cell is measured.

G1  reproduces round 1's recorded exact q=3 loop deficits (5/12, 223/552, 2860783/8291520)
G2  delta_2q = delta_q exactly for odd q -- the theorem that justifies excluding 2*odd from the modulus pool
G3  the sampler is unbiased: sampled delta agrees with the enumerated exact value within window scatter
G4  exact reproducibility at the declared seed
G5  the train/test split is MECHANICAL (every third modulus in sorted order), disjoint, and recorded here
G6  no extrapolation: every held-out phi/gbar lies inside the training phi/gbar range, at every depth
G7  gbar is what we say it is -- loop mean gap == 1/mertens_product on the enumerated loops
G8  the fit is deterministic: fixed bin count, fixed edges rule, linear interpolation, stated here

Writes results/exp_05_gates_<ts>.json. Any FAIL is fatal and loud: the seal must not be written.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

EULER = R.EULER_GAMMA
SEED = 20260909                                    # fresh; exploring used 20260908
DEPTHS = (90, 360, 1440, 5760, 23040, 92160)       # fresh: none of these was read in exp_04's exploring
W, L = 40, 2_000_000
NBINS = 18                                          # fixed here, before any held-out cell exists
# the modulus pool: 3..60 excluding q = 2*odd (delta_2q = delta_q -- G2; not independent cells)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
HELDOUT = tuple(q for i, q in enumerate(POOL) if i % 3 == 2)     # mechanical, not chosen
TRAIN = tuple(q for q in POOL if q not in HELDOUT)

primes = R.odd_sieve(200_000)
gates, ok = {}, True


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def gbar(y):
    return math.exp(EULER) * math.log(y)


def record(name, passed, claim, value):
    global ok
    gates[name] = dict(claim=claim, value=value, status="PASS" if passed else "FAIL")
    ok = ok and bool(passed)
    print(f"  {name}  {'PASS' if passed else '*** FAIL ***'}  {claim}\n        {value}")


def sampled_delta(y, qs, W=W, L=L, seed=SEED):
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
        if len(v) < 8:
            raise RuntimeError(f"instrument: q={q} y={y} gave only {len(v)} usable windows")
        out[q] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))))
    return out


print(f"exp_05 gates  seed={SEED}  depths={DEPTHS}\n")

# G1 ---------------------------------------------------------------------------------------------------------
want, got, g1 = {3: "5/12", 4: "223/552", 6: "2860783/8291520"}, {}, True
for k, w in want.items():
    off, P = R.loop_enumerate(primes[:k])
    e = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, 3), 3))
    got[f"k={k}"] = str(e); g1 = g1 and str(e) == w
    del off
record("G1", g1, "round 1's exact q=3 loop deficits reproduce", got)

# G2 ---------------------------------------------------------------------------------------------------------
n2, g2 = 0, True
for k in range(3, 8):
    off, P = R.loop_enumerate(primes[:k])
    for q in [q for q in range(3, 60, 2) if P % q == 0]:
        a = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, q), q))
        b = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, 2 * q), 2 * q))
        g2 = g2 and a == b; n2 += 1
    del off
record("G2", g2, f"delta_2q == delta_q exactly for odd q ({n2} cases) -- justifies the pool's exclusion", f"{n2}/{n2}")

# G3 ---------------------------------------------------------------------------------------------------------
off, P = R.loop_enumerate(primes[:8]); y8 = int(primes[7])
exact8 = {q: float(R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, q), q))) for q in (3, 5, 7)}
del off
s8, det, g3 = sampled_delta(y8, (3, 5, 7)), {}, True
for q in (3, 5, 7):
    m, se = s8[q]; z = abs(m - exact8[q]) / se if se > 0 else 99
    det[q] = dict(exact=exact8[q], sampled=m, sigma=round(z, 2)); g3 = g3 and z < 3.0
record("G3", g3, f"sampler unbiased against the enumerated exact value at y={y8}", det)

# G4 ---------------------------------------------------------------------------------------------------------
a1 = sampled_delta(1440, (3, 5), W=8)
a2 = sampled_delta(1440, (3, 5), W=8)
record("G4", a1 == a2, "the declared seed reproduces delta bit for bit", {str(k): v[0] for k, v in a1.items()})

# G5 ---------------------------------------------------------------------------------------------------------
disjoint = not (set(TRAIN) & set(HELDOUT)) and set(TRAIN) | set(HELDOUT) == set(POOL)
record("G5", disjoint and len(HELDOUT) >= 12,
       "train/test split is mechanical (every 3rd modulus), disjoint, and covers the pool",
       dict(pool=len(POOL), train=len(TRAIN), heldout=len(HELDOUT), heldout_moduli=list(HELDOUT)))

# G6 ---------------------------------------------------------------------------------------------------------
ranges, g6 = {}, True
for y in DEPTHS:
    gb = gbar(y)
    tr = [phi(q) / gb for q in TRAIN]; ho = [phi(q) / gb for q in HELDOUT]
    inside = min(ho) >= min(tr) and max(ho) <= max(tr)
    ranges[y] = dict(train=[round(min(tr), 4), round(max(tr), 4)],
                     heldout=[round(min(ho), 4), round(max(ho), 4)], inside=bool(inside))
    g6 = g6 and inside
record("G6", g6, "no extrapolation: held-out phi/gbar lies inside the training range at every depth", ranges)

# G7 ---------------------------------------------------------------------------------------------------------
mg, g7 = {}, True
for k in range(2, 9):
    ps = primes[:k]; off, P = R.loop_enumerate(ps)
    obs = float(R.gaps_of(off, P).mean()); exp = 1.0 / R.mertens_product(ps)
    g7 = g7 and abs(obs - exp) < 1e-9; mg[f"k={k}"] = round(obs, 6)
    del off
record("G7", g7, "mean gap == 1/mertens_product (gbar's definition is the measured one)", mg)

# G8 ---------------------------------------------------------------------------------------------------------
record("G8", True,
       "the fit is fully specified before the run and has no free choices left",
       dict(bins=NBINS, edges="linspace(min, max) of the TRAINING x only", statistic="mean of delta per bin",
            interpolation="linear between bin centres, clipped to the end bins",
            x_primary="phi(q)/gbar", x_control="q/gbar", gbar="exp(EULER_GAMMA)*log(y)",
            tolerance="R1 passes if heldout_rms <= 1.5 * train_rms; KILL if > 2.0 *",
            note="no held-out cell is measured until after the seal commit"))

payload = dict(script="exp_05_gates.py", generated=ts, seed=SEED, depths=list(DEPTHS), windows=W, L=L,
               nbins=NBINS, pool=list(POOL), train=list(TRAIN), heldout=list(HELDOUT),
               all_pass=ok, gates=gates)
(RES / f"exp_05_gates_{ts}.json").write_text(json.dumps(payload, indent=1))
print(f"\nwrote results/exp_05_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
