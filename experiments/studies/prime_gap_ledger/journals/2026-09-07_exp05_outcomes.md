# exp_05 outcomes — the collapse δ_q(y) = F(φ(q)/ḡ): 3/3

**Date:** 2026-09-07 (night) · **Layer:** arithmetic · **Seal:** `5d6acfc6`
(`journals/2026-09-07_exp05_registration.md` + `results/exp_05_gates_20260907_184901.json`, G1–G8 all PASS).
**Run:** `scripts/exp_05_collapse.py` → `results/exp_05_collapse_20260907_185245.json`.
Scored to the sealed text; no threshold was relaxed.

## Verdicts

| | relation | verdict |
|---|---|---|
| R1 | the collapse generalises — F fitted on 30 training moduli predicts the 14 held-out | **CONFIRM** — held-out rms 0.03459 against training 0.05636, ratio **0.614** (CONFIRM ≤ 1.5, KILL > 2.0); F's range 0.702 > 10 × training rms, so the not-flat guard holds |
| R2 | the variable is φ/ḡ, not q/ḡ | **CONFIRM** — held-out rms 0.03459 (φ/ḡ) against 0.09864 (q/ḡ), a factor 2.85, resolved at **9.68σ** by the sealed paired bootstrap; the two orderings are Spearman 0.852, below the 0.99 guard, so the control is informative |
| R3 | the residual is structured by ω(q), sign registered in advance as positive | **CONFIRM** — corr +0.686, **t = +7.78**; residual rms 0.0346 far exceeds the median cell SE 0.0004, so it is not measurement error |

**Score 3/3.** Study total **8/12**.

## The sentence

The loop's departure from residue-independence, at every modulus and every depth, is **one curve read at
φ(q)/ḡ** — the size of the unit group against the mean gap. Depth and modulus do not enter separately; they
enter only through that ratio. What the curve leaves over is not noise and not the modulus: it is **ω(q), how
many distinct primes build q**, at t = +7.78 — with the correlation against q itself sitting at −0.027.

## What R3 settled that the seal was prepared to lose

Registered threat §7.5 anticipated that ω(q) and q might not separate over this pool, and committed to
reporting R3 as consistent-with-both if so. It did not happen:

```
corr(residual, omega(q)) = +0.6864   t = +7.78
corr(residual, q)        = -0.0271
```

After the φ/ḡ curve is divided out, the residual retains essentially no dependence on q. In exp_04's exploring
grid the same residual correlated with q at +0.198 (t = 3.25) and with ω at +0.359; the held-out fit sharpened
both, and in opposite directions. The leftover is the count of distinct prime factors, which is the
singular-series shape — whether q divides a gap carries an enhancement ∏_{p|q}(p−1)/(p−2) over the primes
dividing q, so how composite the modulus is should matter beyond how large its unit group is.

## Registered threats that fired, and were handled as sealed

- **§7.2 (ḡ is asymptotic).** At y = 90 the measured mean gap was 8.2256 against e^γ log y = 8.0145 — **2.63 %**,
  over the 2 % bar. That depth's 44 cells are **recorded and not scored**, exactly as registered, before any
  verdict was formed. Scored basis: 150 training + 70 held-out cells over five depths. Every other depth was
  inside 0.88 %, and 0.01 % at y = 92,160.
- **§7.3 (binning is a choice).** Recorded, never scored: held-out rms is 0.04456 at 12 bins and 0.03643 at 24,
  against 0.03459 at the sealed 18. The verdict does not turn on the bin count.

## The honest caveat on R1

Held-out rms came in *below* training rms (ratio 0.614), which is not the usual direction. The reason is
structural and was built in by G6: the held-out φ/ḡ range is required to lie **inside** the training range so
that nothing extrapolates. Held-out cells therefore avoid the ends of the fit, where a binned curve is
coarsest, while training cells include them. R1's threshold is generous in that light. It does not affect R2
or R3, which are comparisons evaluated on the same held-out cells under both variables.

## What is not claimed

This is the **loop's own** profile. Nothing here says the primes inherit the collapse — the primes are the loop
read at its origin (round 2), and whether F survives that read is untouched and would need its own round.
No closed form for F is claimed: F is a binned curve, and 18 bins is a convention, not a discovery. No
statement about β_q beyond exp_04's negative — β remains the local slope of F, which is why it was never a
constant of the modulus. Nothing above y = 92,160. No physics; no golden ratio, Ξ or Fibonacci enters — every
"φ" in this round is Euler's totient.

The result remains a **postdiction confirmed out of sample** (seal §0): the collapse was found in exp_04's
data, then tested on fresh depths, a fresh seed, and moduli the fit never saw, under a mechanical split. That
is stronger than a fit and weaker than an a priori prediction, and it should be cited as such.

## Forward note

Layer: arithmetic. Three things this opens, none registered here:

1. **Does F have a closed form?** It is currently 18 bins. The asymptotics are pinned at both ends — F → 0 as
   φ/ḡ → 0 (independence) and F → 1 as φ/ḡ → ∞ (saturation) — so a two-parameter shape is a candidate. Fitting
   one to this data would be postdiction again; it needs its own held-out design.
2. **The ω(q) correction, made exact.** The singular-series product ∏_{p|q}(p−1)/(p−2) is computable, not just
   correlatable. A round could register the predicted residual cell by cell rather than a correlation.
3. **Do the primes inherit it?** Round 3 established that the primes are the loop read at a shifted depth. If
   that holds, F should describe the primes' own δ_q at y_eff with no new freedom. That is the round that would
   connect this back to the study's spine.

Round 4's forward note stands unchanged: round 3's five β_q are local slopes of this curve, not modulus
constants, and its scored ρ_q = 0 is untouched.
