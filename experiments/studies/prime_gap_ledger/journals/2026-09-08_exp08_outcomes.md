# exp_08 outcomes — position is depth, but the control could not prove it: 1/3

**Date:** 2026-09-08 · **Layer:** arithmetic · **Seal:** `1fd25acf`
(`journals/2026-09-08_exp08_registration.md` + `results/exp_08_gates_20260908_113619.json`, 8/8 PASS).
**Run:** `scripts/exp_08_position_is_depth.py` → `results/exp_08_position_is_depth_20260908_114208.json`.
Scored to the sealed text; no threshold relaxed. **484 of 484 cells scored**, none saturated.

## Verdicts

| | relation | verdict |
|---|---|---|
| R1 | position is depth: δ_q(u) = F_c(φ(q)/ḡ(y_eff(u))), zero free parameters | **CONFIRM** — rms **0.03517** against the sealed 0.05325 (KILL 0.09129) |
| R2 | the transition is universal across moduli | **INCONCLUSIVE** — mean sd of C_q(u) across 35 eligible moduli **0.1624**, against a 0.15 bar (KILL 0.35) |
| R3 | positive control: tracking beats a fixed depth | **INCONCLUSIVE** — 0.03517 vs 0.03637, only **1.75σ** |

**Score 1/3.** Study total **12/21**.

## R1, and why it is stronger than the number alone

rms 0.03517 sits close to exp_07's loop held-out rms of 0.03043 — the model does nearly as well predicting
δ across eleven positions spanning N = 10⁷ to 8·10²¹ as F_c does on the loop it was fitted to. With **nothing
fitted**: F_c and a = 1.2998 are sealed from exp_07, y_eff(u) is solved from each position's measured density.

The residual has no structure anywhere:

| u | shift | mean residual | rms residual |
|---|---|---|---|
| 2.0 | +0.1189 | +0.00055 | 0.0368 |
| 2.1 | +0.0764 | +0.00037 | 0.0347 |
| 2.2 | +0.0430 | +0.00000 | 0.0341 |
| 2.35 | +0.0151 | +0.00001 | 0.0359 |
| 2.5 | +0.0003 | −0.00009 | 0.0351 |
| 2.8 | −0.0104 | −0.00060 | 0.0354 |
| 6.0 | −0.0009 | −0.00054 | 0.0347 |

Mean signed residual is ≈ 0 at **every** position, including u = 2.0 where the effective depth is shifted by
+0.119 in log. The model tracks the moving depth without bias. Overall mean signed residual −0.00027 —
compare the **+0.0054** offset at the primes in exp_07. There is no offset here.

**q = 3's non-freshness (§0) does not matter:** rms 0.03517 with it, 0.03521 without.

## R2: a narrow miss on universality

The normalised coherence curves C_q(u) = [δ_q(u) − δ_q(6.0)] / [δ_q(2.0) − δ_q(6.0)] collapse across 35
eligible moduli with mean sd **0.1624**, against a CONFIRM bar of 0.15 and a KILL bar of 0.35. So the curves
substantially do collapse — the transition is largely modulus-independent — but not tightly enough for the
bar I set. INCONCLUSIVE as sealed, and not a kill.

## R3: the control failed, and the reason is my power estimate — for the third time

Tracking beat the fixed depth (0.03517 vs 0.03637) but at 1.75σ, under the sealed 3σ. G7 had established ≥3σ
power *before* the run, at 4.07σ. It was wrong for two reasons, both design-side:

**Dilution.** Only **5 of 11** positions have |log y_eff/log y − 1| > 0.01. The mean separation between the
tracking and fixed predictions is 0.0257 at u = 2.0 but ≤ 0.0023 at six positions, which sit at y_eff ≈ y and
contribute noise to the paired bootstrap with no signal. Averaged over the grid, the effective contrast is
roughly a third of what the signal-carrying positions alone would give.

**~~Correlated resampling.~~ — RETRACTED, see the correction below.**

Observed advantage 0.00120, against G7's predicted 0.00207 — 58 %.

**This is the third power-estimation failure in three rounds, each by a different mechanism:** per-cell versus
aggregate (exp_06, corrected in `aace9b3d`); an under-sampled position grid (exp_08's own first gate run);
and now dilution plus correlated resampling. The common fault is computing power against an idealisation of
the design rather than against the design as built. A future power gate should simulate **the actual
resampling procedure on the actual grid**, not an i.i.d. abstraction of it.


## CORRECTION filed 2026-09-08 (same day) — the "correlated resampling" cause is FALSE, and the real one is a result

**Retracted.** I claimed above that the bootstrap understated its spread because the 44 cells at a position
share a window. **Measured, it does not:** the intra-class correlation of the residual within a position is
**ICC = −0.023**, effective sample size **484.0 of 484 cells**. Cells within a position are independent and the
cell-level bootstrap was correct. I asserted a plausible mechanism without measuring it — the same failure as
exp_06's "underpowered" claim, which is now twice.

**The measured causes are two, and the second is a finding.**

**(a) Dilution**, as stated — 5 of 11 positions carry a resolvable shift.

**(b) The model over-shifts.** Regressing each position's mean residual on its log depth shift:

```
corr(log_shift, mean residual) = +0.897    t = +6.09 on 9 df    SIGNIFICANT
slope = +0.0098 per unit log-shift
best-fit tracking fraction alpha = 0.812   (0 = fixed depth, 1 = full y_eff tracking)
    rms at alpha = 1 (as registered)  0.03517
    rms at alpha = 0.812              0.03510
    rms at alpha = 0 (fixed)          0.03637
```

The density-solved y_eff moves the prediction roughly 20 % too far. That is why R3 was weak: the **rms is
nearly blind** to α (0.03517 vs 0.03510) while the **residual-vs-shift correlation is overwhelming** (t = 6.09).
R3 was measuring the insensitive statistic.

**This converges with exp_07 from an independent direction.** exp_07 measured λ = 0.6446 at the primes of the
10¹⁰ decade (unresolved — its guard failed). exp_08 gives α = 0.812 from position-tracking on a y = 4473 loop.
Different objects, different statistics, both short of 1.0: **the true effective depth sits below the
density-solved y_eff.**

**Status of this finding: EXPLORING, post-hoc.** It was found by analysing exp_08's residuals after the run.
Nothing here is scored, and R3's verdict stands as INCONCLUSIVE. It identifies the depth-sensitive observable
exp_07's forward note asked for — the regression of per-position mean residual on the shift, not an rms over
moduli — and that is what a next round should register.

## What this round does and does not establish

**Does:** the position dependence of the residue bias is predicted, with zero free parameters, by evaluating
the sealed collapse at the effective depth solved from the local density — with no bias at any position across
fifteen orders of magnitude in N. Round 2's qualitative "coherence at the origin, decoherence away from it"
now has a mechanism: it is ω(u) relaxing to e^{−γ}, and the transition width is a consequence of Buchstab
rather than a constant of its own. The gates measured that ratio tracking Buchstab to ≤ 0.4 %, crossing zero
at u ≈ 2.5 and ringing slightly negative before damping.

**Does not:** establish that tracking the depth is *better* than ignoring it — R3 could not resolve it, so R1's
interpretation rests on the fit being good rather than on the control separating. Nor establish universality
across moduli at the sealed bar. Nothing here is claimed per cell or per modulus (§7.1): the effect is
sub-residual per cell and resolves only in aggregate.

## Registered threats, as they fell

- **§7.1 sub-residual per cell** — held, and honoured: no per-cell claim is made.
- **§7.2 q = 3 not fresh** — recorded both ways; immaterial.
- **§7.3 one depth** — y = 4473 only.
- **§7.4 F_c empirical** — its ~0.030 floor sits under this round's 0.0352.
- **§7.5 Buchstab as gate not input** — y_eff came from measured density throughout; G2's agreement is a
  check on the object, and it is the round's cleanest single number.

## Forward note

1. **Redo R3 with a grid that is not diluted.** Score only positions where the shift is resolvable — u ∈
   [2.0, 2.4] — and register the power against the *actual* bootstrap, resampling positions rather than cells.
   The signal is there; the design spent it.
2. **R2 at 0.1624 is close.** A sharper C would come from more windows per position, since its denominator is
   a difference of two measured endpoints and carries both their errors.
3. **The derivation of F remains the open problem**, now with a cleaner target: F_c describes the loop, the
   primes, and every position between them, with one fitted constant a = 1.2998 and a residual of 0.030–0.040
   that carries the ω(q) structure and, at the primes only, a +0.0054 offset that does not appear here.
