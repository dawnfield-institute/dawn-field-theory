#!/usr/bin/env python3
"""exp_09 gates — run and PASS before the round-9 seal (STANDARDS §2.7).

Round 9 asks a question every previous round was confounded on: is F's residual REAL STRUCTURE, or was it
our sampling noise? Every residual so far (rms ~0.030 under F_c, the omega(q) correlation at t ~ 8) was
measured against SAMPLED delta. `core/exact_gaps.py` now derives delta_q exactly by inclusion-exclusion over
the interior positions, so the two can finally be separated.

G1  the derived gap distribution reproduces the ENUMERATED loop exactly (k <= 9)
G2  derived delta_q matches enumerated delta_q, and the discrepancy is accounted for by the truncated tail
G3  truncation is controlled: captured mass per depth, and the hybrid tail correction is small
G4  F_c and a load from the sealed exp_07 gate file (a = 1.2998, never refitted)
G5  delta_2q = delta_q for odd q -- pool exclusion
G6  the (q, y) grid's x-range is recorded, INCLUDING how many cells sit below the x ~ 0.085 extrapolation
    wall documented in the 2026-09-08 small-x note
G7  POWER, simulating the ACTUAL procedure on the ACTUAL grid -- three rounds of botched power estimates
    (exp_06 per-cell vs aggregate, exp_08's gates twice) say do not simulate an idealisation
G8  all thresholds fixed here, before any residual is formed

Writes results/exp_09_gates_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core")); import rough as R; import exact_gaps as X
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEED = 20260913
DEPTHS = (23, 47, 97, 199, 401)
G = 40                                   # 2^19 subsets at the top g; seconds
QS = tuple(q for q in range(3, 31) if not (q % 2 == 0 and (q // 2) % 2 == 1))
W_TAIL, L_TAIL = 24, 2_000_000           # sampling for the measured tail above G
GBAR = math.exp(R.EULER_GAMMA)

primes = R.odd_sieve(200_000)
gates, ok = {}, True
rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))


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


print(f"exp_09 gates  seed={SEED}  depths={DEPTHS}  G={G}\n")

# G1 -----------------------------------------------------------------------------------------------------------
off, P = R.loop_enumerate(primes[:9]); gt = R.gaps_of(off, P); del off
emp = np.bincount(gt, minlength=G + 2)[:G + 1] / len(gt)
ex23 = X.gap_dist_exact(R.primes_upto(primes, 23), G)
worst = max(abs(ex23[g] - emp[g]) for g in ex23)
record("G1", worst < 1e-9, "derived gap distribution reproduces the enumerated loop (k=9) exactly",
       dict(max_abs_diff=worst, captured_mass=sum(ex23.values())))

# G2/G3 ---------------------------------------------------------------------------------------------------------
tails, exacts, mass = {}, {}, {}
rng = np.random.default_rng(SEED)
for y in DEPTHS:
    ps = R.primes_upto(primes, y)
    t0 = time.time(); exacts[y] = X.gap_dist_exact(ps, G); mass[y] = sum(exacts[y].values())
    # measured tail above G
    hist = np.zeros(4001)
    for o, _ in R.loop_sample(ps, L_TAIL, W_TAIL, rng):
        g = R.gaps_of(o)
        hist += np.bincount(g, minlength=4001)[:4001]
    hist /= hist.sum()
    tails[y] = {int(g): float(hist[g]) for g in range(G + 2, 4001, 2) if hist[g] > 0}
    print(f"   y={y:<5} gbar={1/R.mertens_product(ps):6.3f}  exact mass {mass[y]:.6f}  "
          f"tail mass {sum(tails[y].values()):.6f}  [{time.time()-t0:.0f}s]", flush=True)

d_der, d_enum, g2 = {}, {}, True
for q in QS:
    pq, ex, tl = X.p_divides_gap(exacts[23], q, tails[23])
    d_der[q] = X.delta_from_gaps(pq, phi(q))
    d_enum[q] = 1 - phi(q) * float(np.mean(gt % q == 0))
    if abs(d_der[q] - d_enum[q]) > 5e-3:
        g2 = False
record("G2", g2, "derived delta_q matches the enumerated loop at y=23 within 5e-3 (hybrid: exact + measured tail)",
       {q: dict(derived=round(d_der[q], 6), enumerated=round(d_enum[q], 6),
                diff=round(d_der[q] - d_enum[q], 8)) for q in (3, 5, 7, 9, 11, 13)})

g3 = all(mass[y] > 0.97 for y in DEPTHS)
record("G3", g3, "truncation controlled: exact mass > 0.97 at every depth, remainder measured",
       {y: dict(exact_mass=round(mass[y], 6), tail_mass=round(sum(tails[y].values()), 6),
                gbar=round(1 / R.mertens_product(R.primes_upto(primes, y)), 3)) for y in DEPTHS})

# G4 -------------------------------------------------------------------------------------------------------------
g7f = json.loads(sorted(RES.glob("exp_07_gates_*.json"))[-1].read_text())
A = g7f["a"]; LOOP_HO = g7f["loop_heldout_rms"]
record("G4", abs(A - 1.2998) < 1e-6, "F_c and a load from the sealed exp_07 gate file, never refitted",
       dict(a=A, loop_heldout_rms=LOOP_HO))

# G5 -------------------------------------------------------------------------------------------------------------
n5, g5 = 0, True
for k in range(3, 8):
    o, PP = R.loop_enumerate(primes[:k])
    for q in [q for q in range(3, 60, 2) if PP % q == 0]:
        a1 = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(o, 0, q), q))
        b1 = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(o, 0, 2 * q), 2 * q))
        g5 = g5 and a1 == b1; n5 += 1
    del o
record("G5", g5, f"delta_2q == delta_q exactly for odd q ({n5} cases)", f"{n5}/{n5}")

# G6 -------------------------------------------------------------------------------------------------------------
xs = []
for y in DEPTHS:
    gb = 1 / R.mertens_product(R.primes_upto(primes, y))
    xs += [phi(q) / gb for q in QS]
below = sum(1 for x in xs if x < 0.085)
record("G6", len(xs) >= 60, "x-range recorded, including cells below the x~0.085 extrapolation wall",
       dict(n_cells=len(xs), x_min=round(min(xs), 4), x_max=round(max(xs), 4),
            n_below_extrapolation_wall=below,
            note="see journals/2026-09-08_note_F_below_x_0085_is_extrapolation.md"))

# G7 — POWER, on the ACTUAL procedure ------------------------------------------------------------------------------
# R2 asks whether the omega(q) correlation survives on exact values. Simulate THAT test: n cells, the actual
# omega vector, an effect of the size exp_05/06 reported, and the actual t-statistic.
om = np.array([len(R.factor_int(q)) for q in QS] * len(DEPTHS), dtype=float)
n = len(om); rng2 = np.random.default_rng(SEED); ts_ = []
target_r = 0.60                                    # the correlation exp_05 (+0.686) and exp_06 (+0.597) found
for _ in range(4000):
    z = rng2.normal(size=n)
    v = target_r * (om - om.mean()) / om.std() + math.sqrt(max(0, 1 - target_r ** 2)) * z
    c = float(np.corrcoef(om, v)[0, 1])
    ts_.append(c * math.sqrt((n - 2) / max(1e-12, 1 - c * c)))
ts_ = np.array(ts_); frac = float((ts_ >= 3).mean())
record("G7", frac >= 0.95,
       "POWER on the ACTUAL grid: if the omega effect is real at its reported size, R2 resolves at >=3 sigma",
       dict(n_cells=n, assumed_correlation=target_r, median_t=float(np.median(ts_)),
            fraction_reaching_3sigma=frac,
            note="simulates the actual t-test on the actual omega vector, not an i.i.d. abstraction"))

# G8 -------------------------------------------------------------------------------------------------------------
record("G8", True, "all thresholds fixed here, before any residual is formed",
       dict(a=A, form="tanh(a*phi(q)/gbar)", gbar="1/mertens_product (exact, not the asymptotic e^gamma log y)",
            R1="the residual is REAL STRUCTURE, not sampling noise: CONFIRM if rms(exact delta - F_c) "
               ">= 0.60 * the loop's held-out rms 0.03043 = 0.01826; KILL if <= 0.30 * = 0.00913, which "
               "would mean the residual was mostly our sampling and F is near-exact",
            R2="the omega(q) correlation on EXACT values, >=3 sigma, sign registered positive",
            R3="truncation control: the residual changes by < 20% between G=36 and G=40",
            depths=list(DEPTHS), moduli=list(QS), G=G, seed=SEED))

payload = dict(script="exp_09_gates.py", generated=ts, seed=SEED, depths=list(DEPTHS), G=G,
               moduli=list(QS), a=A, loop_heldout_rms=LOOP_HO,
               exact_mass={str(k): v for k, v in mass.items()}, all_pass=ok, gates=gates)
(RES / f"exp_09_gates_{ts}.json").write_text(json.dumps(payload, indent=1, default=str))
print(f"\nwrote results/exp_09_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
