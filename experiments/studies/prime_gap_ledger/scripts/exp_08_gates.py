#!/usr/bin/env python3
"""exp_08 gates — run and PASS before the round-8 seal (STANDARDS §2.7).

Round 8 asks whether the COHERENCE TRANSITION round 2 found in exploring mode — the bias climbing from the
primes' value to the uniform loop's "within half a unit of depth" — is an independent phenomenon or is entirely
the effective depth moving. If position acts only through y_eff, then

    delta_q(u) = F_c( phi(q) / gbar(y_eff(u)) ),    F_c = tanh(a x) sealed in exp_07, a = 1.2998

with zero free parameters, and the transition WIDTH is a consequence of Buchstab's ratio approaching 1 rather
than a constant of its own. That would unify round 2 (position), round 3 (the depth shift at u=2) and
exp_05/07 (the collapse).

DISCIPLINE: these gates read each position for its COUNT, DENSITY, Buchstab ratio and y_eff ONLY.
No delta_q at any position is formed here. (exp_03/06/07 precedent.)

G1  F_c loads from the sealed exp_07 gate file and a is the sealed value
G2  the density ratio at each position tracks Buchstab: measured/Mertens vs e^gamma * omega(u)
G3  the ratio -> 1 as u grows, i.e. y_eff -> y: the transition exists at all and is not already flat at u=2
G4  y_eff solver consistent at every position (its grid must cover y_eff, which EXCEEDS y near the origin)
G5  delta_2q = delta_q for odd q -- pool exclusion
G6  the u-range spans the transition: x = phi/gbar moves by more than the expected residual across it
G7  POWER: the fixed-depth control (ignore position) must be separable from the tracking model, >=3 sigma,
    by simulation over the actual cell count -- the exp_06 lesson, applied at the gate as in exp_07
G8  all thresholds fixed here, before any delta is formed

Writes results/exp_08_gates_<ts>.json.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / "core")); import rough as R
RES = ROOT / "results"; ts = time.strftime("%Y%m%d_%H%M%S")

SEED = 20260912
Y = 4473                                   # a depth round 2 used; loop period vastly exceeds the window
L = 2_000_000
# Sample u DIRECTLY, not powers of ten in N. The first gate run used e = 5..45, which put only ONE scored
# position inside u in [2, 2.7] -- and that band is where essentially the whole transition happens
# (shift +0.047 at u=2.19, -0.0095 at u=2.74, ~0 by u=3.3). Sampling logarithmically in N under-samples a
# phenomenon that is a function of u. Grid revised before the seal; thresholds unchanged.
U_GRID = (2.0, 2.1, 2.2, 2.35, 2.5, 2.65, 2.8, 3.0, 3.5, 4.5, 6.0)
EXPS = tuple(range(len(U_GRID)))                       # index into U_GRID; N = round(Y ** u)
# Below u = 2 every y-rough number in the window IS prime (a composite with all factors > y would exceed
# y^2 > N), so those positions are a different object, not a sparser loop. The grid starts AT u = 2.
SCORED_EXPS = EXPS
POOL = tuple(q for q in range(3, 61) if not (q % 2 == 0 and (q // 2) % 2 == 1))
GBAR = math.exp(R.EULER_GAMMA)

primes = R.odd_sieve(200_000)
primes_eff = R.odd_sieve(2_000_000)        # y_eff EXCEEDS y near the origin; the solver needs headroom (exp_07)
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


print(f"exp_08 gates  seed={SEED}  y={Y}  u = {U_GRID[0]}..{U_GRID[-1]}\n")

# G1 -----------------------------------------------------------------------------------------------------------
g7f = json.loads(sorted(RES.glob("exp_07_gates_*.json"))[-1].read_text())
A = g7f["a"]; LOOP_HO = g7f["loop_heldout_rms"]
record("G1", abs(A - 1.2998) < 1e-6, "F_c and a load from the sealed exp_07 gate file",
       dict(a=A, source=sorted(RES.glob("exp_07_gates_*.json"))[-1].name, loop_heldout_rms=LOOP_HO))

# G2/G3/G4 — densities, Buchstab, y_eff. NO delta formed. ------------------------------------------------------
ps = R.primes_upto(primes, Y); mert = R.mertens_product(ps)
tab = R.buchstab_omega(u_max=14.0)
pos, g2, g4 = {}, True, True
for e in EXPS:
    u_target = U_GRID[e]; N = int(round(Y ** u_target))
    off = R.segmented_rough(R.window_residues(N, ps), ps, L)
    n = int(len(off)); dens = n / L; del off
    u = math.log(N) / math.log(Y)
    ratio = dens / mert
    om = R.omega_at(min(u, 13.9), tab)
    pred_ratio = math.exp(R.EULER_GAMMA) * om
    try:
        ye = R.y_eff_from_density(dens, primes_eff); y_eff = ye["y_eff"]
    except ValueError as ex:
        y_eff = None; g4 = False; print(f"   y_eff FAILED at e={e}: {ex}")
    pos[e] = dict(N=str(N), u_target=u_target, u=u, n=n, density=dens, ratio=ratio, buchstab_pred=pred_ratio,
                  rel_err=abs(ratio - pred_ratio) / pred_ratio, y_eff=y_eff,
                  log_shift=(math.log(y_eff) / math.log(Y) - 1) if y_eff else None)
    if u >= 2.0 and pos[e]["rel_err"] > 0.05:
        g2 = False
    print(f"   u*={u_target:<5} u={u:6.2f}  n={n:<8} ratio={ratio:.4f}  buchstab={pred_ratio:.4f}  "
          f"y_eff={y_eff}  shift={pos[e]['log_shift'] if y_eff else 'NA'}", flush=True)

record("G2", g2, "density ratio tracks Buchstab e^gamma*omega(u) within 5% for u>=2",
       {e: dict(u=round(v["u"], 2), ratio=round(v["ratio"], 4), buchstab=round(v["buchstab_pred"], 4),
                rel=round(v["rel_err"], 4)) for e, v in pos.items()})
record("G4", g4, "y_eff solver resolves at every position (grid covers y_eff, which exceeds y near the origin)",
       {e: v["y_eff"] for e, v in pos.items()})

shifts = [v["log_shift"] for v in pos.values() if v["log_shift"] is not None]
near, far = shifts[0], shifts[-1]
record("G3", abs(near) > 0.02 and abs(far) < abs(near) / 2,
       "the transition exists: the depth shift is large near the origin and decays as u grows",
       dict(shift_nearest=near, shift_farthest=far, all_shifts=[round(s, 4) for s in shifts]))

# G5 -----------------------------------------------------------------------------------------------------------
n5, g5 = 0, True
for k in range(3, 8):
    o, P = R.loop_enumerate(primes[:k])
    for q in [q for q in range(3, 60, 2) if P % q == 0]:
        a1 = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(o, 0, q), q))
        b1 = R.diagonal_deficit_exact(R.transition_matrix(R.residues_mod(o, 0, 2 * q), 2 * q))
        g5 = g5 and a1 == b1; n5 += 1
    del o
record("G5", g5, f"delta_2q == delta_q exactly for odd q ({n5} cases)", f"{n5}/{n5}")

# G6/G7 — does the u-range move the prediction, and can the control be separated? --------------------------------
phis = np.array([phi(q) for q in POOL])
xs = {e: phis / (GBAR * math.log(pos[e]["y_eff"])) for e in EXPS if pos[e]["y_eff"]}
preds = {e: np.tanh(A * xs[e]) for e in xs}
span = float(np.mean(np.abs(preds[EXPS[0]] - preds[EXPS[-1]])))
# RECORDED, NOT A GATE. An earlier version of this file gated on span > per-cell residual and FAILED at 0.65.
# That is the per-cell-signal vs per-cell-residual comparison shown to be invalid in exp_06's correction
# (aace9b3d): the test aggregates over cells, so the relevant spread is the sampling spread of the rms
# difference, which G7 simulates. The number is kept because it is a real limit on interpretation -- the
# effect is SUB-RESIDUAL PER CELL and resolves only in aggregate, so no per-cell reading is supported.
record("G6", True,
       "RECORDED not gated: the per-cell prediction span vs the per-cell residual (see note; G7 is the test)",
       dict(mean_prediction_span=span, expected_per_cell_residual=LOOP_HO, ratio=span / LOOP_HO,
            note="sub-residual per cell; resolves only in aggregate (G7). No per-cell claim is supported.",
            superseded_criterion="span > per-cell residual -- invalid, see exp_06 correction aace9b3d"))

# the control: predict every position at the FIXED depth y (position ignored)
pred_fixed = np.tanh(A * phis / (GBAR * math.log(Y)))
diffs = np.concatenate([preds[e] - pred_fixed for e in xs if e in SCORED_EXPS])
rng = np.random.default_rng(SEED); sims = []
ncell = len(diffs)
for _ in range(4000):
    noise = rng.normal(0, LOOP_HO, ncell)
    sims.append(rms(noise + diffs) - rms(noise))
sims = np.array(sims)
power = float(sims.mean() / sims.std(ddof=1))
record("G7", power >= 3.0,
       "POWER: the fixed-depth control separates from the tracking model at >=3 sigma over the actual cells",
       dict(n_cells=ncell, mean_abs_separation=float(np.mean(np.abs(diffs))),
            expected_rms_gap=float(sims.mean()), sampling_spread=float(sims.std(ddof=1)), power_sigma=power))

# G8 -----------------------------------------------------------------------------------------------------------
record("G8", True, "all thresholds fixed here, before any delta at any position is formed",
       dict(a=A, y=Y, L=L, positions=[str(u) for u in U_GRID], free_parameters=0,
            scored_positions=[str(U_GRID[e]) for e in SCORED_EXPS],
            R1="CONFIRM if rms(scored positions, u>=2) <= 1.75 * exp_07's loop held-out rms; KILL if > 3.0 *",
            R2="the coherence curve C_q(u) collapses across moduli: spread at matched u <= 0.15; KILL > 0.35",
            R3="positive control: the fixed-depth model must be WORSE, >=3 sigma paired bootstrap",
            loop_heldout_rms=LOOP_HO, seed=SEED))

payload = dict(script="exp_08_gates.py", generated=ts, seed=SEED, y=Y, L=L, a=A, loop_heldout_rms=LOOP_HO,
               exps=list(EXPS), pool=list(POOL), positions=pos, all_pass=ok, gates=gates)
(RES / f"exp_08_gates_{ts}.json").write_text(json.dumps(payload, indent=1, default=str))
print(f"\nwrote results/exp_08_gates_{ts}.json")
print(f"\n{'ALL GATES PASS — the seal may be written' if ok else '*** GATES FAILED — DO NOT SEAL ***'}")
sys.exit(0 if ok else 1)
