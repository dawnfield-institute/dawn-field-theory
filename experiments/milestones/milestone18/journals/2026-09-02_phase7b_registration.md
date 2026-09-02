# Phase 7b registration (SEALED): the Π-asymmetry profile

**Status: SEALED 2026-09-02** (the commit carrying this heading; nothing below ran before it).
Scored to this text (§2.6). Lessons 1–7 applied; pre-seal review verified the binomial quantile
(Bin(18, 0.10): 99% quantile 5) and the power count (18/21) from the objects alone.

## The claim under test

exp_16 asked whether the defect edge stands out among degree-matched edges and found the design
underpowered. This phase uses the construction's own symmetry as the null: the two sheets of the
branched cover are isomorphic except at the branch (r17, Theorem 1), so a σ-blind observable
should treat an edge and its Π-image nearly alike — **except near the defect**, where the sheets
differ. Peter's reinjection hypothesis then predicts a *profile*: the Π-asymmetry of bond
occupation is largest at the branch and decays with distance from it.

## Instruments (by construction)

- Objects: the 21 strict folds at n ≤ 16; defect e₀ and matching Π recovered from R (as in
  exp_16). Later tiers (n = 20) may be added only under a fresh seal.
- Observable: bond occupation u_e(β) = K_ab/tr K, K = exp(−βC), mpmath 40 digits; β = 1 scored,
  β ∈ {1/2, 2} recorded.
- **Π-pairs:** every non-defect edge e has Π(e) an edge (r15 Addendum 5); the (n−2)/2 unordered
  pairs {e, Π(e)} are the sample. **Asymmetry** a(e) = |u_e − u_{Π(e)}| / (u_e + u_{Π(e)}).
  **Distance** d(e) = graph distance from the pair to the defect: min over endpoints of e and
  Π(e) of the vertex distance to the nearer endpoint of e₀.
- Per fold: Spearman ρ between a and d over the pairs (ties handled by average ranks).
- **Null (permutation, per fold):** recompute ρ with the reference edge replaced by each other
  edge of the tree in turn (distances measured from that edge; the Π-pairs and asymmetries
  unchanged). The fold's p-value is the fraction of reference edges giving ρ ≤ ρ_obs (the
  defect included), so the null asks: *is the defect a special place for the asymmetry profile,
  or would any edge do?*
- **Power (lesson 7, from the objects alone):** pairs per fold = (n−2)/2 — 1 at n = 4, 3 at
  n = 8, 5 at n = 12, 7 at n = 16. A fold is informative if it has ≥ 5 pairs: **18 of 21**
  (n ≥ 12). The n = 4 and n = 8 folds are declared uninformative before the run.

## Tests

- **T1 (PREDICTION, can fail — the profile).** Across the 18 informative folds, the number with
  per-fold permutation p ≤ 0.10 exceeds the 99% binomial quantile for P = 0.10 (that quantile is
  5 for 18 trials; so PASS requires ≥ 6). Fails otherwise.
- **T2 (PREDICTION, can fail — the peak).** Across the 18 informative folds, the number whose
  maximum-asymmetry pair lies at distance ≤ 1 from the defect exceeds the 99% quantile of the
  Poisson-binomial null with per-fold probability = (pairs at distance ≤ 1)/(all pairs). Fails
  otherwise.
- **T3 (recorded, not scored).** The sign of the asymmetry at the branch (which sheet carries
  more occupation), and β-robustness of T1/T2 counts.

Instrument gates before the run: exp_16's orbit-invariance and edge-sum gates; on E₈, Π-pairs
are exactly the 3 non-defect pairs with the defect (2,3) recovered.

Kill scope: a T1/T2 failure kills "the heat-kernel asymmetry between sheets is organized around
the branch" at n ≤ 16, β = 1; the construction theorem and matching laws are untouched. A pass
is a *dynamical* fingerprint of the port — the first, if it comes.

---

**Layer (forward note, 2026-09-02, per the re-separation):** Measures a physical reach (the Π-asymmetry profile) — feeds `theory/`.
