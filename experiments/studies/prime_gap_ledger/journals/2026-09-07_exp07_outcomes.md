# exp_07 outcomes — the closed form holds; the depth is still not measurable: 1/3

**Date:** 2026-09-07 (night) · **Layer:** arithmetic · **Seal:** `b43dca3f`
(`journals/2026-09-07_exp07_registration.md` + `results/exp_07_gates_20260907_195941.json`, G1–G8 all PASS).
**Run:** `scripts/exp_07_closed_form.py` → `results/exp_07_closed_form_20260907_202055.json`.
Scored to the sealed text; no threshold relaxed. **44 of 44 cells scored**, none saturated.

## Verdicts

| | relation | verdict |
|---|---|---|
| R1 | the closed form predicts the 10¹⁰ primes, zero further freedom | **CONFIRM** — rms **0.04018** against the sealed 0.04565 (KILL 0.07608) |
| R2 | it beats the 18-bin F on those cells | **INCONCLUSIVE** — 0.04018 against 0.06358, a large margin, but only **2.14σ**, under the sealed 3σ bar |
| R3 | λ(m=10) within 0.15 of the forecast 0.652 | **INCONCLUSIVE** — λ measured **0.6446**, but the §4 guard failed: the minimum is not resolved below y_eff |

**Score 1/3.** Study total **11/18**.

## R1: the closed form carries to a fresh decade

`tanh(1.2998 · φ(q)/ḡ(y_eff))`, with `a` fitted on the loop's training moduli and frozen at the seal, predicts
the consecutive-prime residue bias across all 44 moduli of the 10¹⁰ decade — 427,154,205 primes — to rms
0.0402, with **nothing fitted at the primes**. The prediction is within 32 % of the loop's own held-out rms.

A second thing fell out. exp_06 recorded a systematic offset of **+0.0115** using the binned F. Under the
closed form the same offset is **+0.0054** — less than half. So a substantial part of what exp_06 reported as
an unexplained physical offset was **my estimator**, not the object. The remainder is still non-zero and still
unexplained.

Recorded, never scored (§7.5): the best-fit `a` at the primes is **1.3465** against the loop's **1.2998**,
3.6 % higher. The primes want a slightly steeper curve than the loop does.

## R2: the right direction, but it did not clear its own bar — and my §0(a) is only half right

rms 0.04018 (closed form) against 0.06358 (bins) is a 37 % improvement, but the paired bootstrap put it at
**2.14σ**, short of the sealed 3σ. INCONCLUSIVE, as sealed.

More importantly, the recorded-not-scored breakdown undercuts my own claim:

| | n | closed form | bins |
|---|---|---|---|
| all scored cells | 44 | 0.0402 | 0.0636 |
| **within the bins' own fitted domain** | 39 | 0.0421 | **0.0457** |

Nearly all of the apparent margin comes from the **5 cells outside the bins' domain**, where `np.interp` clips
to the end bin and the binned predictor fails badly — a penalty for extrapolation, not for shape. Inside the
domain the bins were only modestly worse.

**So §0(a) of the registration overstated it.** The 18-bin F was worse, but not badly worse where it was
entitled to be used; its real defect was having a domain at all, which forced exp_06 to drop 5 moduli per
decade. That is a genuine advantage of the closed form, and a smaller one than I claimed.

## R3: the forecast landed, and it does not count

λ was forecast at **0.652** from the linear trend at m = 7, 8, 9. It measured **0.6446** — a miss of 0.0074
against a tolerance of 0.15.

**It is INCONCLUSIVE, and it should be.** The §4 guard required the scan minimum to sit below the rms at
*both* endpoints by more than the bootstrap spread:

```
rms at the minimum (y* = 323,424)   0.038653
rms at y                            0.043801    below by 0.005148   > spread   OK
rms at y_eff                        0.040183    below by 0.001530   < spread   FAILS
bootstrap spread                    0.003932
```

The curve is flat between the minimum and y_eff. λ is not resolved to better than roughly ±0.3, so agreeing
with a forecast to 0.0074 carries no information — **the apparent bullseye is a coincidence of a broad
minimum, not a confirmed prediction.** Registering that guard is the only reason this is not being written up
as a triumph.

Note what did change, though: under the closed form y_eff is now within 0.0015 rms of the optimum, where
under the binned F it was excluded at −4.8σ (exp_06 correction). Round 3's depth shift looks materially better
under a better estimator. That is recorded as an observation, not a verdict — R3 is the relation that was
supposed to establish it, and it did not.

## Registered threats, as they fell

- **§7.1 one decade** — R1 and R3 rest on m = 10 alone, as registered.
- **§7.2 tanh is empirical** — no derivation claimed; `x/(a+x)` remains decisively worse.
- **§7.3 λ's trend is three points** — and the fourth point could not adjudicate it. The threat was correctly
  registered and the round could not retire it.
- **§7.4 depth/offset degeneracy** — λ measured at zero offset by construction.
- **§7.5 a frozen from the loop** — the primes' preferred a is 3.6 % higher; recorded, not scored.

## What is not claimed

That λ follows a line — R3 did not establish it and could not. That the depth shift is confirmed — better
under the closed form, still unresolved. Any derivation of tanh. Anything above 10¹⁰. That the closed form is
decisively better than the bins *within their domain* — it is not, at the sealed bar.

## Forward note

Two things, and the first is now the study's clearest open problem:

1. **λ cannot be measured this way, at any decade.** The rms-vs-depth curve is intrinsically flat near its
   minimum because F is smooth and the residual is ~0.04. More decades will not fix it; a *sharper observable*
   would. The natural candidate is to stop using rms over all moduli and instead use the moduli where
   dF/d(log y) is largest — a weighted statistic designed for depth sensitivity rather than an unweighted fit.
2. **The residual is now the object.** With F_c the residual is 0.040 at the primes and 0.030 on the loop,
   carrying a +0.0054 offset and the ω(q) structure (exp_05/06, t ≈ 8). That is what a derivation of F would
   have to explain, and it is no longer dominated by estimator error.
