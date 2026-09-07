# exp_06 outcomes — do the primes inherit the collapse? 2/3

**Date:** 2026-09-07 (night) · **Layer:** arithmetic · **Seal:** `863d8081`
(`journals/2026-09-07_exp06_registration.md` + `results/exp_06_gates_20260907_190634.json`, G1–G8 all PASS).
**Run:** `scripts/exp_06_primes_inherit.py` → `results/exp_06_primes_inherit_20260907_191016.json`.
Scored to the sealed text; no threshold relaxed. **117 of 132 cells scored** (q = 3, 4, 5, 8, 12 fall below
F's fitted domain and are recorded, never scored; no modulus was saturated).

## Verdicts

| | relation | verdict |
|---|---|---|
| R1 | the primes inherit F, zero free parameters | **CONFIRM** — rms 0.04116 against the sealed tolerance 0.06919 (KILL 0.10378); F's span over the scored range 0.663 clears the not-flat guard |
| R2 | the depth shift does the work (positive control) | **INCONCLUSIVE** — rms 0.04116 with ḡ(y_eff) against 0.03981 with ḡ(y), a difference of **−0.43σ**. Not resolved either way; the point estimate marginally favours the *unshifted* depth |
| R3 | the ω(q) structure persists at the primes, sign registered as positive | **CONFIRM** — corr **+0.597**, t = **+7.98**, against corr with q of **−0.030**; residual rms 0.0412 far exceeds the median cell SE 0.00048 |

**Score 2/3.** Study total **10/15**.

## What R1 does and does not establish

A curve fitted **entirely on the loop of units mod a primorial** — an auxiliary object, with no prime involved
at any stage — predicts the consecutive-prime residue bias at 39 moduli across three decades to rms 0.041,
with **nothing tuned**. F was not refitted, rescaled or shifted; y_eff was solved from the measured density.
This is the study's first *a priori* prediction and it confirmed.

**But R2's failure narrows what R1 means, and the narrowing is real.** Because ḡ(y) predicts as well as
ḡ(y_eff) — indeed marginally better, though at −0.43σ that is noise — R1 establishes that **the primes lie on
F at approximately the right place**, not that round 3's depth shift is what puts them there. The sharper
claim the round was built to test is not established.

**A systematic offset is present and is not noise.** The mean signed residual is **+0.0115**, 28 % of the rms:
the primes sit slightly *above* F across the scored cells. Unexplained here. It is not the ω(q) effect, which
is a correlation rather than an offset, and it is 24× the median cell SE.

## CORRECTION filed 2026-09-07 (same night) — the instrument note below is WRONG

**The section that follows was mistaken and is retained only as the record of the error.** It claimed R2 was
"dead at the seal" with signal 0.83× noise. That comparison put the *per-cell* prediction shift (0.0255)
against the *per-cell* residual (0.0308). The test does not work per cell — it compares two rms values over
117 cells, so the relevant spread is the sampling spread of that difference, **0.0023**, not 0.0308. I
compared the wrong two numbers.

**R2 had roughly 4σ of power, and it returned a real result.** Simulating at the observed residual scale:

| hypothesis | predicted | observed | |
|---|---|---|---|
| y_eff is the true depth | rms(y) ≈ 0.0509 | 0.0398 | **−4.8σ** |
| y is the true depth | rms(y_eff) − rms(y) ≈ +0.0100 | +0.0013 | **−3.7σ** |

**Both named depths are excluded.** The primes read at neither y nor y_eff; the minimising depth sits between
them — a scan over the scored cells puts it at roughly 40 % of the way from y to y_eff in log at all three
decades (6,652 / 20,584 / 71,448 against y = 4,473 / 14,143 / 44,722 and y_eff = 10,429 / 38,321 / 139,891).
That is a substantive constraint on round 3's reading, not a null.

Two caveats on that constraint, both real. First, depth and a constant offset are **degenerate**: allowing an
offset moves the best depth below y and swings the offset to −0.022, while the rms improves by under 0.001. So
"the depth is between" holds only at zero offset. Second, this whole comparison runs through F, and F is the
18-bin estimator, which is itself poor (see the same-night finding that a one-parameter `tanh(1.30 x)` beats it
on both training and held-out cells). A better F could move the location.

**R2's verdict is unchanged: INCONCLUSIVE**, as sealed — the rule required rms(y_eff) < rms(y) at ≥ 3σ, and it
was not. Only the diagnosis changes. The registered §4 guard on the shift's existence was, as it happens,
adequate; my post-hoc power argument against it was the error.

## Instrument note — SUPERSEDED, see the correction above

The registration's §4 required |log y_eff / log y − 1| > 0.02 so that the control could distinguish the two
depths. It passed comfortably: 0.1007, 0.1043, 0.1065 at m = 7, 8, 9. **The guard was still wrong.** What
matters is not whether the shift exists but whether it moves the *prediction* by more than the residual:

```
x shifts by       10.4 %      (log y_eff / log y ~ 1.10)
F(x) moves by      0.0255     the control's signal
the residual is    0.0308     the noise it must beat
                   ratio 0.83x
```

At 0.83× the signal could not have reached 3σ under any outcome. **R2 was dead at the seal and its guard did
not see it.** The correct guard is a *power* condition — the mean |F(x_eff) − F(x_y)| must exceed the expected
residual rms by some registered factor — not an existence condition on the shift. Any future round comparing
two nearby arguments of a smooth curve should register the power condition instead; it is the same class of
mistake as testing a modulus that saturation has already made unscoreable (exp_04, G7).

This does not rescue R2. It is INCONCLUSIVE as sealed, and it is recorded as INCONCLUSIVE.

## What R3 settled, against my expectation

I expected R3 to fail, on the grounds that ω(q) and q might not separate at the primes even though they did on
the loop, and §7.4 pre-committed to reporting consistent-with-both if so. They separated as cleanly as before:

```
              loop (exp_05)    primes (exp_06)
corr omega       +0.686           +0.597
corr q           -0.027           -0.030
```

The residual carries the count of distinct prime factors and essentially nothing of the modulus itself, at
both objects, with the sign registered in advance. That is the singular-series shape transferring from the
loop to the primes.

## Registered threats, as they fell

- **§7.1 (bin-edge error).** Recorded: correlation between |residual| and distance to the nearest bin centre.
- **§7.2 (one arc, not a sample).** SEs are the de-trended chunk scatter, 10 chunks per decade, as round 3.
- **§7.3 (short lever).** Three decades. Nothing is claimed above 10⁹.
- **§7.4 (ω vs q).** Did not materialise; they separated.
- **§7.5 (F's own floor).** exp_05's held-out rms 0.0346 is a floor under R1's 0.0412 — the prediction is
  within about 19 % of the best F can do on its own object, which is why the tolerance was a multiple of it.

## What is not claimed

That the depth shift carries the primes onto F — **R2 did not establish it and this round cannot**. That the
inheritance holds above 10⁹. Any closed form for F. Any statement about the +0.0115 offset's origin. No
physics; every "φ" is Euler's totient.

Round 3's ρ_q = 0 is untouched: it measures a ratio against the loop at y and never involves F.

## Forward note

Layer: arithmetic. Two things this opens:

1. **Locate the depth, do not re-run the two-way comparison.** The correction above shows R2 had power and
   excluded both named depths; the open question is therefore *where* the primes read, not whether y beats
   y_eff. That needs the depth/offset degeneracy broken and a better F — a one-parameter `tanh(a·x)` beats the
   18-bin estimator on both training and held-out loop cells, so the residual this comparison runs through is
   inflated by my own choice of estimator.
2. **The +0.0115 offset.** A constant offset between the primes and the loop's own curve is exactly the shape
   of a residual the study has chased before (round 1's ε ≈ 0.0095 at the primes). Whether these are the same
   object is not established and would need its own registration — but it is the obvious next question.
