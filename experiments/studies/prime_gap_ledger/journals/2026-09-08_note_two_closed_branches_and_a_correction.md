# Note — two closed branches, and a correction to the recursion note

**Date:** 2026-09-08 · **Status:** EXPLORING, unregistered, unscored.
**Supersedes** part of `2026-09-08_note_the_sieve_recursion_and_where_it_fails.md` — see §3.

## 1. δ_q is not a functional of the gap process, at any order of memory

The recursion note proposed iterating the derived resolvent and testing it. Done by simulation: generate
synthetic gap sequences matching the k = 9 loop's gap statistics, thin them repeatedly (delete each unit at
rate 1/p, merge adjacent gaps), and compare δ₃ against measurement. The thinning was validated first — mean
gap 6.1129 → 6.3314 against the exact expected 6.3312 — and the generators reproduce their targets
(Markov corr −0.1594 against the real −0.1591; i.i.d. −0.0002).

| y | measured | actual sequence | Markov (1-step memory) | i.i.d. (renewal) |
|---|---|---|---|---|
| 97 | 0.22278 | 0.22416 (+0.0014) | −0.15120 (−0.374) | −0.04384 (−0.267) |
| 997 | 0.16035 | 0.16229 (+0.0019) | −0.20086 (−0.361) | −0.02879 (−0.189) |

Thinning the **real** sequence reproduces reality to +0.002. Both synthetic models give the **wrong sign**,
and the Markov model — which was 5.5× better than i.i.d. at a *single* step — is **worse** than i.i.d. under
iteration. That inversion is the tell. The cause is a constraint, not a modelling error:

**Residue histogram of the unit values** (real units are coprime to q, so class 0 must be empty):

| | r0 | r1 | r2 | |
|---|---|---|---|---|
| actual loop gaps, q = 3 | **0.00 %** | 50.00 % | 50.00 % | |
| i.i.d. from f | 33.33 % | 33.33 % | 33.34 % | |
| Markov from (f, K) | 33.35 % | 33.31 % | 33.34 % | |

At q = 5 the same: real 0.00 % on class 0 and 25 % on each of the other four; both synthetic models ~20 % on
all five. **A gap-level stochastic model generates positions that cannot be units.** Matching f and K better
improves the gap statistics while leaving the residue support just as wrong — which is exactly why more memory
made it worse.

**The statement:** δ_q depends on where the units sit modulo q; gaps only give differences. Two sequences with
identical gap statistics can have different residue support, and δ_q separates them. So δ_q is **not a
functional of the gap process** — the whole model class is wrong, not under-parameterised.

This is the study's local/global split appearing as the reason the derivation cannot close from below:
coprimality is a *global* condition on positions, the gap process is a purely *local* description.

## 2. q is not a product of independent cascades — and the defect lives on φ(q)/ḡ

Peter's framing: treat q as a confluence rather than one thing. By CRT, "same residue mod q" is the AND of
"same residue mod p^a" over the prime powers, so independence would give
**1 − δ_q = ∏ (1 − δ_{p^a})**. Tested on 20 composite moduli at y = 1000, 25 000, 141 422:

Median |ratio − 1| = **81.7 %, 60.4 %, 52.4 %**. Multiplicativity fails decisively — **q is indeed not one
thing.** The only exact cases are q = 6, 10, 14, and those are this study's own δ_{2q} = δ_q theorem, not
independence.

The mechanism is that the components share one gap: "same mod 3 and same mod 5" is the single condition
15 | g, which is crushed when typical gaps are below 15.

**But the confluence opens no new axis.** The defect's collapse, over 68 composite cells:

| variable | R² of one curve in log(ratio) |
|---|---|
| **φ(q)/ḡ** | **0.8444** |
| q/ḡ | 0.5373 |
| q unscaled | 0.4383 |
| ω(q) | 0.0085 |

It collapses on **φ(q)/ḡ** — the same variable as the main effect. **This cost me a hypothesis:** the coupling
mechanism above predicts the scale should be q (the condition is q | g), and the data says φ. No account of
why is offered.

It also does **not** explain the ω(q) residual carried since exp_05 — R² = 0.0085 against ω says they are
unrelated structures.

## 3. Correction to the recursion note

That note concluded: *"the fault is not the coarseness of the residue state space — it is the renewal
assumption itself."* **That was too optimistic.** Renewal failing is a symptom. §1 shows the gap process
cannot represent δ_q at any order, so a second-order (or n-th order) resolvent was never going to close it.
The single-step 18 × improvement it reported is real but measures agreement of *gap distributions*, where the
coprimality constraint does not bite; under iteration the constraint dominates and the ranking inverts.

The note's derived merge operator, the exact first-moment result, and the two-poset Möbius reading all stand.
Only its forward path — "a second-order resolvent would fix it" — is retracted.

## What this leaves

- **F(φ(q)/ḡ)** stands: loop, primes, and every position between, zero free parameters.
- **φ(q)/ḡ is more universal than exp_05 established** — it governs the deficit, and the failure of the
  deficit's own prime-power factorisation.
- Three exploratory branches are closed: iterated resolvent, informational observable, independent-cascade
  confluence. None touches a sealed verdict.
- The ω(q) residual is still unexplained: not the singular series, not the confluence coupling.
