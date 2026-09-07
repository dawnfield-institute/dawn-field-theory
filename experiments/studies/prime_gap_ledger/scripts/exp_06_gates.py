#!/usr/bin/env python3
"""exp_06 gates — run and PASS before the round-6 seal (STANDARDS §2.7).

Round 6 asks whether the PRIMES inherit the loop's collapse. F was fitted in exp_05 on loop cells only and is
sealed; round 3 established that the primes are the loop read at the effective depth y_eff solved from the
arc's measured density. So

    delta_q(primes)  =  F( phi(q) / gbar(y_eff) )

has ZERO free parameters. This is the study's first a priori prediction rather than a postdiction.

DISCIPLINE: these gates read the decades for their COUNTS, DENSITIES and TRANSITION COUNTS ONLY. No delta_q at
any decade is formed here — exp_03's gates set exactly this precedent ("read the 10^10 decade for its count and
transition count only; no delta at 10^10 was formed"). Density is a gate quantity; the deficit is not.

G1  F loads from exp_05's sealed result and is the same object (bin centres/means, monotone, non-degenerate)
G2  the read at u=2 IS the primes: count in [10^m, 2*10^m) sieved to sqrt(2*10^m) == sympy's primepi difference
G3  the chunk carry is exact: transitions == n - 1 at every decade
G4  delta_2q = delta_q for odd q -- justifies excluding q = 2*odd from the pool (exp_04's theorem)
G5  the y_eff solver is consistent: mertens_product(y_eff) reproduces the measured density within its bracket
G6  DOMAIN: every scored phi(q)/gbar(y_eff) lies inside F's fitted range -- outside it F would extrapolate
G7  gbar is the measured quantity at y_eff: |mean gap - e^gamma log y_eff| within 2% per decade
G8  the tolerances, pool, split and comparison are fully specified here, before any delta_q is formed

Writes results/exp_06_gates_<ts>.json. Any FAIL is fatal: the seal must not be written.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

try:
    from sympy import primepi
except ImportError as e:
    raise SystemExit(f"sympy needed for G2's independent prime count: {e}")

SEED = 20260910
DECADES = (7, 8, 9)
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
EXP05 = sorted(RES.glob("exp_05_collapse_*.json"))[-1]
HELDOUT_RMS_05 = None                        # read from the sealed exp_05 result below

primes = R.odd_sieve(200_000)
gates, ok = {}, True


def phi(q):
    n = 1
    for p, a in R.factor_int(q).items():
        n *= (p - 1) * p ** (a - 1)
    return n


def gbar(y):
    return math.exp(R.EULER_GAMMA) * math.log(y)


def record(name, passed, claim, value):
    global ok
    gates[name] = dict(claim=claim, value=value, status="PASS" if passed else "FAIL")
    ok = ok and bool(passed)
    print(f"  {name}  {'PASS' if passed else '*** FAIL ***'}  {claim}\n        {value}")


print(f"exp_06 gates  seed={SEED}  decades={DECADES}\n")

# G1 -----------------------------------------------------------------------------------------------------------
d05 = json.loads(EXP05.read_text())
cx = np.array(d05["F_phi_bins"]["centres"]); cy = np.array(d05["F_phi_bins"]["means"])
HELDOUT_RMS_05 = d05["verdicts"]["R1"]["heldout_rms"]
mono = bool(np.all(np.diff(cy) >= -1e-9)) or bool(np.all(np.diff(cy) <= 1e-9))
g1 = len(cx) >= 10 and (cy.max() - cy.min()) > 0.5 and mono
record("G1", g1, f"F loads from the sealed exp_05 result ({EXP05.name}) and is non-degenerate",
       dict(bins=len(cx), x_range=[round(float(cx.min()), 4), round(float(cx.max()), 4)],
            F_range=round(float(cy.max() - cy.min()), 4), monotone=mono, exp05_heldout_rms=HELDOUT_RMS_05))

# G2/G3/G5/G7 — counts, densities and transition counts ONLY. No delta_q is formed. ------------------------------
dec, g2, g3, g5, g7 = {}, True, True, True, True
for m in DECADES:
    N0 = 10 ** m; y = math.isqrt(2 * N0) + 1; ps = R.primes_upto(primes, y)
    chunk = max(N0 // 10, 1)
    rd = R.chunked_read(N0, N0, chunk, ps, (3,))            # Q=(3,) only to exercise the carry; no delta taken
    n = rd["n"]; dens = rd["density"]
    want = int(primepi(2 * N0) - primepi(N0))
    ye = R.y_eff_from_density(dens, primes); yq = ye["y_eff"]
    mert_eff = R.mertens_product(R.primes_upto(primes, yq))
    gb_eff = gbar(yq); mean_gap = 1.0 / dens
    dec[m] = dict(N=N0, y=y, n=n, primepi_expected=want, count_matches=bool(n == want),
                  transitions=int(rd["transitions"][3]), carry_exact=bool(rd["transitions"][3] == n - 1),
                  density=dens, y_eff=yq, mertens_at_y_eff=mert_eff,
                  density_vs_mertens_rel=abs(dens - mert_eff) / dens,
                  mean_gap=mean_gap, e_gamma_log_y_eff=gb_eff,
                  gbar_rel_err=abs(mean_gap - gb_eff) / gb_eff,
                  bracket=dict(p_lo=ye["p_lo"], p_hi=ye["p_hi"], mismatch_log=ye["mismatch_log"]))
    g2 = g2 and dec[m]["count_matches"]
    g3 = g3 and dec[m]["carry_exact"]
    g5 = g5 and dec[m]["density_vs_mertens_rel"] < 1e-3
    g7 = g7 and dec[m]["gbar_rel_err"] < 0.02
    print(f"   m={m}: n={n} (primepi {want}) y={y} y_eff={yq} density={dens:.8f} "
          f"gbar_err={100*dec[m]['gbar_rel_err']:.2f}%", flush=True)

record("G2", g2, "the read at u=2 IS the primes (count == primepi difference)",
       {m: dict(n=v["n"], expected=v["primepi_expected"]) for m, v in dec.items()})
record("G3", g3, "chunk carry exact: transitions == n - 1 at every decade",
       {m: v["transitions"] for m, v in dec.items()})
record("G5", g5, "y_eff solver consistent: mertens_product(y_eff) reproduces the measured density < 1e-3",
       {m: dict(y_eff=v["y_eff"], rel=round(v["density_vs_mertens_rel"], 8)) for m, v in dec.items()})
record("G7", g7, "gbar(y_eff) is the measured mean gap within 2% at every decade",
       {m: dict(mean_gap=round(v["mean_gap"], 4), gbar=round(v["e_gamma_log_y_eff"], 4),
                rel=round(v["gbar_rel_err"], 5)) for m, v in dec.items()})

# G4 -----------------------------------------------------------------------------------------------------------
n4, g4 = 0, True
for k in range(3, 8):
    off, P = R.loop_enumerate(primes[:k])
    for q in [q for q in range(3, 60, 2) if P % q == 0]:
        a = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, q), q))
        b = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(off, 0, 2 * q), 2 * q))
        g4 = g4 and a == b; n4 += 1
    del off
record("G4", g4, f"delta_2q == delta_q exactly for odd q ({n4} cases)", f"{n4}/{n4}")

# G6 -----------------------------------------------------------------------------------------------------------
dom, g6 = {}, True
for m in DECADES:
    gb = gbar(dec[m]["y_eff"])
    xs = {q: phi(q) / gb for q in POOL}
    inside = [q for q, x in xs.items() if cx.min() <= x <= cx.max()]
    outside = [q for q in POOL if q not in inside]
    dom[m] = dict(x_range=[round(min(xs.values()), 4), round(max(xs.values()), 4)],
                  F_domain=[round(float(cx.min()), 4), round(float(cx.max()), 4)],
                  n_inside=len(inside), outside=outside)
    g6 = g6 and len(inside) >= 20
record("G6", g6, "F's domain covers >=20 moduli per decade (those outside are recorded, never scored)", dom)

# G8 -----------------------------------------------------------------------------------------------------------
record("G8", True, "the comparison is fully specified here, before any delta_q at any decade is formed",
       dict(prediction="delta_q(primes) = F(phi(q)/gbar(y_eff)), F from the sealed exp_05 bins, y_eff from density",
            free_parameters=0,
            R1="CONFIRM if rms(measured - predicted) <= 2.0 * exp_05 heldout rms; KILL if > 3.0 *",
            exp05_heldout_rms=HELDOUT_RMS_05,
            R2="positive control: gbar(y) in place of gbar(y_eff) must do WORSE, >=3 sigma paired bootstrap",
            R3="the omega(q) residual persists at the primes with the SAME registered sign: positive, >=3 sigma",
            pool="q in [3,60] excluding 2*odd; moduli outside F's domain or saturated are recorded, never scored",
            seed=SEED, decades=list(DECADES)))

payload = dict(script="exp_06_gates.py", generated=ts, seed=SEED, decades=list(DECADES), pool=list(POOL),
               exp05_source=EXP05.name, exp05_heldout_rms=HELDOUT_RMS_05,
               decade_gate_data=dec, domain=dom, all_pass=ok, gates=gates)
(RES / f"exp_06_gates_{ts}.json").write_text(json.dumps(payload, indent=1, default=str))
print(f"\nwrote results/exp_06_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
