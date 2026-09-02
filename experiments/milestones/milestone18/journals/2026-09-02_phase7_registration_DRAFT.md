# Phase 7 registration (DRAFT — unsealed, uncomputed): does dynamics see the reinjection port?

**Status:** draft. Sealed when committed with SEALED in the commit message and this heading;
nothing below runs before that. Scored to the sealed text (§2.6). Lessons 1–6 applied. Frame
declared per STANDARDS §2.9.

## The claim under test

Peter's hypothesis (2026-09-01): the dynamics is entropy reinjection — Landauer-type
compensation — as redistribution of potential in the tree. The construction theorem
(`2026-09-02_r17_construction_theorem.md`) supplies the port: on every strict fold the reflection
√5·R = S + 2Π is an **integer, σ-fixed** matrix, the fold is a branched double cover, and the
single **defect edge** (the branch) is where the golden content, the trace, and the leakage are
generated. Because S + 2Π is σ-fixed, the σ-blindness theorem (exp_05 T1) does not forbid
rational dynamics from seeing it — this is the first dynamical question about the fold that is
not pre-killed. exp_04 asked whether dynamics sees the *mirror* (a σ-object) and failed; this
asks whether it sees the *branch*.

## Instruments (by construction)

- **Objects.** All 21 strict Galois folds at n ≤ 16 (r15/r17b; construction parents), each with
  its defect edge e₀ computed from R (not from the construction — the instrument must find it).
  **Controls:** (C1) the sector-strict tree at n = 16 (det −775), (C2) the five quotient folds at
  n ≤ 12, (C3) for every fold, a degree-sequence-matched random tree (fixed seed, 5 per fold).
- **Dynamics.** The heat kernel of the Cartan matrix, K(β) = exp(−βC), the s = 1 member of the
  Mirror family. β = 1 is the scored frame; β ∈ {1/2, 2} recorded for robustness.
- **Per-edge observable (bond occupation).** For an edge e = {a, b}: u_e(β) = K_ab(β)/tr K(β).
  Rational functional of C (allowed), exact via sympy `exp` of the diagonalized C or high-precision
  numerics with a declared tolerance 1e−12 for rank comparisons.
- **Frame.** Ranks, never absolute values. An edge's *matched class* is the set of edges of the
  same tree with the same unordered endpoint-degree pair. In a tree with automorphisms edges in
  one orbit have identical u_e (instrument gate: verified, else fault); an orbit counts once.
- **Distinguished** := u_{e₀} is strictly extremal (maximum or minimum) within its matched class,
  the class having at least 3 orbits (classes of size < 3 are declared uninformative and excluded
  with the exclusion count reported).

## Tests

- **T1 (PREDICTION, can fail).** Across the informative folds, the number whose defect edge is
  distinguished exceeds the 99% binomial quantile of the uniform-rank null (per fold,
  P = 2/m for a class of m orbits). Fails otherwise. This is the reinjection-port claim.
- **T2 (PREDICTION, can fail).** On controls C1–C3 the count of trees with *any* distinguished
  edge among their informative classes does not exceed the 99% quantile of the same null. Fails
  if controls light up — the effect must be specific to the branch.
- **T3 (recorded, not scored).** The two cut edges over the special bond: their ranks within
  their classes, and whether the three lifts of the special bond jointly occupy extremal ranks.
- **T4 (recorded, not scored).** β-robustness: T1's count at β = 1/2 and 2.

Instrument gates before the run: orbit-invariance of u_e on E₈ and D̃₄; tr K reproduces
Σ e^{−βλ} to tolerance on A₄.

Kill scope: a T1 failure kills "SEC-type dynamics couples to the reinjection port" for the
heat-kernel frame at β = 1; it does not touch the construction theorem, the matching laws, or the
milestone kill sentence. A T2 failure retires the specificity claim and turns T1's signal, if
any, into a degree artifact until a sharper null is registered.

## Compute

21 + ~110 trees at n ≤ 16, one matrix exponential each: seconds.
