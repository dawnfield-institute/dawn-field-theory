#!/usr/bin/env python3
"""exp_07 gates — run and PASS before the round-7 seal (STANDARDS §2.7).

Round 7 replaces the 18-bin F with a ONE-PARAMETER closed form, tanh(a * phi(q)/gbar), fits a on the LOOP's
training moduli only, and tests it against the primes at a FRESH decade, m = 10, never read in this study.

DISCIPLINE: as exp_06, these gates read m = 10 for its COUNT, DENSITY, y_eff and TRANSITION COUNT ONLY.
No delta_q at m = 10 is formed here.

G1  a is fitted on loop TRAINING moduli only, and generalises to the loop's held-out moduli
G2  the closed form beats the 18-bin F on the LOOP's held-out cells (established before the primes are touched)
G3  the m = 10 read IS the primes: count == sympy primepi(2N) - primepi(N)
G4  the chunk carry is exact at m = 10: transitions == n - 1
G5  the y_eff solver is consistent at m = 10: mertens_product(y_eff) reproduces the density
G6  gbar(y_eff) is the measured mean gap within 2% at m = 10
G7  POWER, registered properly this time (exp_06's lesson): the depth scan must be able to resolve. The rms
    difference between the scan minimum and the endpoints must exceed the SAMPLING SPREAD of that difference
    -- computed by simulation at the expected residual scale over the actual cell count, NOT by comparing
    per-cell signal to per-cell residual, which is the error corrected in exp_06's outcomes.
G8  every threshold, the lambda forecast and its tolerance are fixed here, before any m = 10 delta exists

Writes results/exp_07_gates_<ts>.json. Any FAIL is fatal.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

try:
    from sympy import primepi
except ImportError as e:
    raise SystemExit(f"sympy needed for G3's independent prime count: {e}")

SEED = 20260911
M = 10
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
LAMBDA_FORECAST, LAMBDA_TOL = 0.652, 0.15      # from the linear trend on m = 7,8,9 (0.799, 0.750, 0.701)
GBAR = math.exp(R.EULER_GAMMA)

primes = R.odd_sieve(200_000)          # the sieving list: y at m=10 is 141422, comfortably inside
# y_eff at m=10 lands near 5e5 -- BEYOND the sieving list. The solver needs its own, longer grid, or it
# runs off the end (it raises rather than returning a wrong answer, which is the right behaviour).
primes_eff = R.odd_sieve(2_000_000)
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


print(f"exp_07 gates  seed={SEED}  decade m={M}\n")

# G1/G2 — fit a on the loop's TRAINING moduli, check it on the loop's HELD-OUT moduli -------------------------
d5 = json.loads(sorted(RES.glob("exp_05_collapse_*.json"))[-1].read_text())
tr = [c for c in d5["cells"] if not c["heldout"]]; ho = [c for c in d5["cells"] if c["heldout"]]
xt = np.array([c["phi"] / c["gbar"] for c in tr]); dt = np.array([c["delta"] for c in tr])
xh = np.array([c["phi"] / c["gbar"] for c in ho]); dh = np.array([c["delta"] for c in ho])
grid = np.linspace(0.5, 3.0, 25001)
A = float(grid[int(np.argmin([rms(dt - np.tanh(g * xt)) for g in grid]))])
tr_rms, ho_rms = rms(dt - np.tanh(A * xt)), rms(dh - np.tanh(A * xh))
record("G1", ho_rms <= 1.5 * tr_rms,
       "a is fitted on loop TRAINING moduli only and generalises to the loop's held-out moduli",
       dict(a=A, train_rms=tr_rms, heldout_rms=ho_rms, ratio=ho_rms / tr_rms))

CX = np.array(d5["F_phi_bins"]["centres"]); CY = np.array(d5["F_phi_bins"]["means"])
bin_ho = rms(dh - np.interp(xh, CX, CY))
record("G2", ho_rms < bin_ho,
       "the closed form beats the 18-bin F on the loop's held-out cells, before any prime is touched",
       dict(tanh_heldout=ho_rms, binned_heldout=bin_ho, improvement=bin_ho - ho_rms))

# G3/G4/G5/G6 — m = 10, counts and densities ONLY. No delta formed. --------------------------------------------
N0 = 10 ** M; y = math.isqrt(2 * N0) + 1; ps = R.primes_upto(primes, y)
t0 = time.time()
rd = R.chunked_read(N0, N0, 10 ** 8, ps, (3,))
dens = rd["density"]; n = rd["n"]
print(f"   m=10 read: n={n} in {time.time()-t0:.0f}s", flush=True)
t1 = time.time(); want = int(primepi(2 * N0) - primepi(N0)); print(f"   primepi: {want} ({time.time()-t1:.0f}s)", flush=True)
ye = R.y_eff_from_density(dens, primes_eff); y_eff = ye["y_eff"]
mert_eff = R.mertens_product(R.primes_upto(primes_eff, y_eff))
gb_eff = GBAR * math.log(y_eff); mean_gap = 1.0 / dens

record("G3", n == want, "the m=10 read IS the primes (count == primepi difference)",
       dict(n=n, primepi=want))
record("G4", rd["transitions"][3] == n - 1, "chunk carry exact at m=10: transitions == n - 1",
       dict(transitions=int(rd["transitions"][3]), n_minus_1=n - 1))
record("G5", abs(dens - mert_eff) / dens < 1e-3, "y_eff solver consistent at m=10",
       dict(y=y, y_eff=y_eff, density=dens, mertens_at_y_eff=mert_eff,
            rel=abs(dens - mert_eff) / dens, bracket=dict(p_lo=ye["p_lo"], p_hi=ye["p_hi"])))
record("G6", abs(mean_gap - gb_eff) / gb_eff < 0.02, "gbar(y_eff) is the measured mean gap within 2% at m=10",
       dict(mean_gap=mean_gap, gbar_y_eff=gb_eff, rel=abs(mean_gap - gb_eff) / gb_eff))

# G7 — POWER, done the right way: simulate the sampling spread over the actual cell count ----------------------
xs = np.array([phi(q) for q in POOL])
g_y, g_e = GBAR * math.log(y), gb_eff
pred_y = np.tanh(A * xs / g_y); pred_e = np.tanh(A * xs / g_e)
sep = float(np.mean(np.abs(pred_e - pred_y)))
sigma_exp = ho_rms                                   # expected per-cell residual, from the loop's held-out rms
rng = np.random.default_rng(SEED); sims = []
for _ in range(4000):
    noise = rng.normal(0, sigma_exp, len(POOL))
    sims.append(rms(noise + (pred_e - pred_y)) - rms(noise))
sims = np.array(sims)
power_sigma = float(sims.mean() / sims.std(ddof=1))
record("G7", power_sigma >= 3.0,
       "POWER: the depth comparison resolves at >=3 sigma by simulation over the actual cell count",
       dict(mean_separation_in_prediction=sep, expected_cell_residual=sigma_exp, n_cells=len(POOL),
            expected_rms_gap=float(sims.mean()), sampling_spread=float(sims.std(ddof=1)),
            power_sigma=power_sigma,
            note="exp_06's error was comparing per-cell signal to per-cell residual; this simulates the "
                 "sampling spread of the rms difference over the actual number of cells"))

# G8 -----------------------------------------------------------------------------------------------------------
record("G8", True, "all thresholds fixed here, before any m=10 delta_q exists",
       dict(a=A, form="tanh(a * phi(q)/gbar)", free_parameters_at_the_primes=0,
            R1="CONFIRM if rms(primes at m=10) <= 1.5 * the loop's held-out rms; KILL if > 2.5 *",
            R2="the closed form must beat the 18-bin F on the m=10 primes, >=3 sigma paired bootstrap",
            R3=f"lambda(m=10) within {LAMBDA_TOL} of {LAMBDA_FORECAST} (linear trend on m=7,8,9: 0.799, 0.750, 0.701); "
               f"KILL if outside [0.35, 0.95]",
            lambda_forecast=LAMBDA_FORECAST, lambda_tol=LAMBDA_TOL, seed=SEED))

payload = dict(script="exp_07_gates.py", generated=ts, seed=SEED, m=M, a=A, pool=list(POOL),
               loop_train_rms=tr_rms, loop_heldout_rms=ho_rms, binned_heldout_rms=bin_ho,
               y=y, y_eff=y_eff, density=dens, n=n, all_pass=ok, gates=gates)
(RES / f"exp_07_gates_{ts}.json").write_text(json.dumps(payload, indent=1, default=str))
print(f"\nwrote results/exp_07_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
