# Phase 7c registration (DRAFT — unsealed, uncomputed): the branch fingerprint at n = 20

**Status:** draft. Sealed when committed with SEALED in the commit message and this heading.
Scored to the sealed text (§2.6). Lessons 1–8 applied; the pre-seal null computation (lesson 8)
is marked **[FILL FROM r19 BEFORE SEALING]** and the seal will carry the numbers.

## Why 7c

Phase 7b found the Π-asymmetry decaying from the defect on 18/18 informative folds and peaking
*adjacent* to it on 21/21 — but at n ≤ 16 the permutation null could not separate "organized by
the branch" from "organized by tree geometry", and the sealed d ≤ 1 peak statistic had a
saturated null. n = 20 supplies 47 evaluable strict folds with 9 Π-pairs each and larger
diameters, and the sharper statistic (d = 0) can be registered with its null computed first.

## Instruments (by construction)

- Objects: the 47 evaluable partnered strict folds at n = 20 (exp_15 T3, rigidity 47/47), with
  defect e₀, matching Π and signs S recovered from R (`explore_r19_n20_defects.py`).
- Observable: bond occupation u_e(β) = K_ab/tr K, K = exp(−βC), mpmath 40 digits; β = 1 scored,
  β ∈ {1/2, 2} recorded. Π-pairs and asymmetry a(e) as in Phase 7b; distance d(pair, ref) as in
  Phase 7b.
- **Pre-seal null numbers (lesson 8), computed from the trees alone before any observable:**
  - per-fold P₀ = (# pairs at distance 0 from e₀)/(# pairs) — **[FILL: list and mean]**
  - Poisson-binomial 99% quantile of Σ over folds of Bernoulli(P₀) — **[FILL: q₀]**
  - permutation-null resolution: (# edges) = 19 reference edges per fold, so the minimal per-fold
    p is 1/19 ≈ 0.053; informative threshold p ≤ 0.10 is reachable.

## Tests

- **T1 (PREDICTION — the peak is AT the branch).** The number of folds whose maximum-asymmetry
  pair is at distance 0 from the defect exceeds q₀ (the 99% Poisson-binomial quantile computed
  above). Fails otherwise.
- **T2 (PREDICTION — the profile is organized by the branch).** The number of folds with
  per-fold permutation p ≤ 0.10 (Spearman ρ(a, d) against distances from every other reference
  edge) exceeds the 99% binomial quantile for Bin(47, 0.10) — **[FILL: quantile]**. Fails
  otherwise.
- **T3 (recorded).** Sheet sign (copy − conjugate occupation) per fold; β-robustness of T1/T2.

Kill scope: T1/T2 failures kill "the heat-kernel sheet asymmetry is organized around the branch"
at n = 20, β = 1; the construction theorem and the matching laws are untouched. A pass is the
first dynamical fingerprint of the reinjection port under a null that respects both geometry and
the construction's symmetry.
