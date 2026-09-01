# REGISTERED: Phase 3 — the one-5 prediction and the fold-invariant laws

**Status: SEALED by this commit (Peter's go, 2026-09-01 night). Nothing at n = 14 has been
computed. Every fail-clause was checked against the record's theorems before sealing.**

## Pre-seal self-consistency pass (the exp_05 lesson)
Each fail-clause below was checked against the theorems already in the record: strict
pairing ⇒ even n (so odd-n strict hits are impossible and are not predicted); the
signature-doubling claim follows from the folding identity and is (S), not scored.

## exp_11 — The one-5 prediction at 14 vertices (k = 7, odd) — 4 tests

Frame: all 3,159 non-isomorphic trees on 14 vertices, enumerated (no sampling); expectation:
the Coxeter-diagram polynomials. Same scope. Deliberately NOT computed before sealing.

- T1: every tree-shaped Coxeter diagram of 3s on 7 nodes with exactly one 5-bond (all 11
  tree shapes × edge positions, deduplicated by polynomial) has at least one 14-vertex tree
  parent with charpoly = q·σ(q). Fails if any diagram is an orphan.
- T2: no 7-node linear diagram with two 5s has a 14-vertex tree parent. Fails if any does.
- T3: every parent found is **core-grade** (odd k ⇒ λ = 2 is a rational root of q ⇒ (t−2)²
  divides the parent's charpoly). Fails if any parent is strict.
- T4: there is **no strict √5-golden tree on 14 vertices at all** — the odd-k theorem plus the
  "strict ⇒ fold" conjecture together. Fails if the exhaustive census finds one.

Pre-seal self-consistency: T3 and T4 are consistent with the bipartite-duality argument
(odd-dimensional self-dual spectrum has the fixed point 2) and with the n ≤ 12 record (all
strict trees are even-k folds). T1/T2 have no theorem forcing them; they are the bet.

## exp_12 — Resonance, signature, and fold-invariant laws (6 tests; T4–T6 are predictions)
- T1: field-resonance law on the ADE trees with 13 ≤ n ≤ 14 (A₁₃, A₁₄, D₁₃, D₁₄) and on the
  affine trees D̃ₙ, Ẽ₆, Ẽ₇, Ẽ₈: pairs over ℚ(√d) iff √d ∈ ℚ(ζ_{2h}) (affine: h of the
  corresponding finite diagram is NOT the right invariant — the affine spectrum contains 0;
  predicted: affine trees are never strict; grade to be recorded, not scored).
- T2: det(parent) = N(q(0)) for every fold at n ≤ 14.
- T3: copy/conjugate signatures (k,0)/(k,0) for definite diagrams and (k,0)/(k−1,1) for
  hyperbolic ones, for every fold at n ≤ 14.
- T4 (PREDICTION, can fail) — the fold invariant. For every strict Galois fold at n ≤ 14
  (7/7 at n ≤ 12; exp_11 T4 predicts none at 14, so any strict fold found there is a fresh
  test), with B = 2I − D and R = P − σ(P):
      tr(RD) = 2/√5        (equivalently tr(PB) = 1 − 1/√5, given tr(PA) = 0 and tr(B) = 2)
      ‖(I−P)BP‖²_F = 2/5
  Fails if any strict fold violates either. Derived so far: tr(PC) = n and tr(P) = n/2 for
  any strict fold, hence tr(PA) = 0 exactly; the values 2/√5 and 2/5 are observed, unproved.
- T5 (PREDICTION, can fail) — the same invariant on the core-grade Galois folds found by
  exp_11 T1 (P = Bezout projector for the diagram polynomial with rational core removed,
  core resolved on the golden conic; the core traces are gauge-independent when the core
  sits on leaf-difference vectors, and the gauge dependence is declared otherwise).
- T6 (PREDICTION, can fail) — the diagnostic. For every QUOTIENT fold at n = 14 (golden
  trees that are not Galois folds, folding by their automorphism-orbit quotient — 5/5 at
  n ≤ 12 with pure (t−2)^m cores): tr(RD) = 0 and tr(PB) = 1. Fails if any quotient fold
  shows the Galois values or any Galois fold shows the quotient values. Registered claim:
  **tr(RD) distinguishes the two fold mechanisms** — 2/√5 for Coxeter-diagram folds, 0 for
  symmetry quotients.

Kill relevance: none for the milestone kill sentence (Block C territory). A T1/T2 failure
retires the one-5 conjecture at the size it fails; the six verified fold families and the
resonance law stand (kill scope).
