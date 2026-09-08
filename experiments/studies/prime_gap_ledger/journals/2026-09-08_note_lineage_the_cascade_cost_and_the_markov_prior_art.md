# Note — lineage: why it is a cascade, the Markov prior art, and a thermodynamic reading that half works

**Date:** 2026-09-08 · **Status:** EXPLORING, unregistered, unscored. Filed as lineage and as a correction to
this study's own "cited, not built on" list.

## 1. Prior art this study should have cited: `prime_harmonic_manifold`

`archive/era2-prefield/prime_harmonic_manifold` (Era-2, archived, validated 2025-12-12) established:

> *"Prime gap pairs (g_n, g_{n+1}) form a Markov chain with leading eigenvalue that decays at rate −1/π² per
> log-decade. This structure cannot be reproduced by any random model, including shuffled gaps, proving the
> ordering itself carries information."*

```
lambda_1 ~ 1.12 - (1/pi^2) log10(N)     50k: 0.705   200k: 0.631   1M: 0.572   2M: 0.550
real vs Cramer   z = 30.4        real vs shuffled   z = 5.9
optimal chord length: 2 gaps  —  100% significant at n = 2, 0% at n = 4
```

**This is prior art on exactly the obstruction of `2026-09-08_note_the_sieve_recursion_and_where_it_fails`.**
Three correspondences:

1. "Cannot be reproduced by any random model, including shuffled gaps" **is** the failure of our renewal
   resolvent. They established it on the primes in Era 2; we rediscovered it on the loop.
2. "100 % significant at n = 2, 0 % at n = 4" says the memory is **exactly one step**. That independently
   corroborates our second-order result — one gap of memory took the single-step error from 3.3× to 18×
   better than doing nothing, and n = 4 adding nothing predicts that third order will not help.
3. Their eigenvalue decays logarithmically in scale; our correlation decays like 1/ḡ ~ 1/log y. Different
   objects and different statistics, so **not the same number** — but the same structure.

They also ran the same discipline arc: refuted λ₁ = 1/φ as a crossing-point artifact, validated 1/π². That is
the sibling case to `sec_prime_manifold`'s 1/φ frame artifact already cited in the README.

**Action:** the README's lineage section lists `sec_prime_manifold`, `asymmetric_conservation`, Milestone 3's
prime cascade reachability and `prime_growth_dynamics_v2`. `prime_harmonic_manifold` is the most relevant of
all of them and was missing. Added.

## 2. Why it is a cascade — and why the "why" was hard to see

The corpus's cascade is *"cascade structure with ln(φ) cost per level"* — uniform information cost, giving the
1/ln(φ) clock slope and 1/φ attenuation. The sieve cascade has the same shape with a different cost, and today
the level operation was derived rather than asserted: level k → k+1 deletes the units divisible by p_{k+1} and
**merges adjacent gaps**, multiplying the mean gap by exactly p/(p−1). So

```
cost per level          = -log(1 - 1/p)
total over p <= y       = -log prod(1 - 1/p) = log(e^gamma log y) = gamma + log log y
```

**The levels are not uniform.** Each prime costs a different amount, so in the k coordinate the cascade looks
irregular and there is no constant per-level cost to find — which is why the "why" resisted.

It uniformizes in u = log x / log y, where levels become unit steps. That is exactly why Buchstab's ω obeys a
**delay-1** equation, (uω)′ = ω(u−1): the cascade level *is* the unit step in u. And it is why every result in
this study is a function of u — round 2's position, round 3's shift, exp_08's transition. We have been working
in the uniformized cascade coordinate throughout without naming it.

## 3. The thermodynamic reading: exact for the cost, false for the observable

**Exact.** −log(1 − 1/p) is the surprisal of surviving level p; γ + log log y is the information that a number
is y-rough. This is the same quantity the corpus calls "the Landauer erasure cost per cascade level", with a
different per-level value. The operation is coarse-graining: deleting a unit merges two gaps, erasing the
distinction between two intervals. So δ_q decaying to zero **is** relaxation under repeated information
erasure — which is a mechanism for F's decay, not a metaphor.

**False.** The natural next step — that the observable should therefore be informational — does not survive.
Over 60 cells (10 moduli × 6 depths), R² of a single curve:

| observable | in φ(q)/ḡ | in q/ḡ |
|---|---|---|
| **δ_q (the deficit)** | **0.9373** | 0.8554 |
| mutual information I (nats) | 0.7489 | 0.7648 |
| I / log φ(q) | 0.6715 | 0.7218 |

Mutual information collapses **worse**, and its preference flips to q/ḡ where δ's is decisively φ/ḡ. The
likely reason is that δ is a **matched filter** — it measures one structure, the same-residue diagonal, while
MI aggregates every departure from independence and dilutes that signal. So δ is not an arbitrary statistic;
it is the right one, and that is now measured rather than assumed.

The thermodynamics explains **why F decays**. It does not improve **how F is measured**.

## What is not claimed

That −1/π² and our 1/ḡ decay are the same constant — different objects, different statistics. That the DFT
cascade and the sieve cascade are the same cascade — same shape, different per-level cost, uniformized only
by reparameterization. No physics enters this study; every "φ" here is Euler's totient except where
`prime_harmonic_manifold`'s refuted 1/φ claim is quoted as history.
