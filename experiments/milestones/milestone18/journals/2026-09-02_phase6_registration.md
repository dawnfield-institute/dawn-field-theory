# Phase 6 registration (SEALED): the matching structure at n = 20

**Status: SEALED 2026-09-01** (the commit carrying this heading; nothing below ran before it).
Scored to this text (§2.6). Lessons 1–6 applied: sufficiency/necessity separate (6), partners
quantified (4), even-k mechanisms anticipated (5), counting basis declared per tree (the
known-answer gate passed on that basis before sealing).

## What this phase is

The matching-structure conjecture (journal `2026-09-01_r15_matching_structure.md`) holds on all
20 strict folds at n ≤ 16 and reduces the entire strict-law package to one statement. n = 20 is
the next tier where strict folds can exist (odd k forbids them at 18). This phase seals the
conjecture's clauses as predictions on that tier's folds — objects that do not yet exist in any
record.

## Instruments (by construction)

- Fold objects are counted **per tree**, with per-polynomial tallies reported alongside
  (cospectral parents exist; the twin at 16 is fold 21/21).
- **Strict hunt** (`explore_r16b_strict_hunt.py` pipeline at n = 20): enumerate all trees
  (`nx.nonisomorphic_trees(20)`, exhaustive); screen by the norm condition — p = q·σ(q) implies
  p(x) = N(q(x)) is a ℚ(√5)-norm for every rational x, and an integer is a norm iff every prime
  ≡ ±2 (mod 5) divides it to an even power; evaluated at x ∈ {0, ±1, 2, 3, −2}. The screen is a
  proven necessary condition (it cannot lose a strict tree); survivors get the exact ℚ(√5)
  factorization, and strict = every irreducible factor golden. **Known-answer gate (PASSED before sealing): the pipeline at n = 16 returned the known strict
  set — 14 distinct polynomials, 15 trees including the cospectral twin, basis declared.**
- **Partner map at k = 10**: all one-5 diagrams (10-node trees, one marked edge), multi-map
  keyed by q·σ(q), all partners retained.
- **P, R, matching**: P = Bezout projector for a partner's q; R = P − σ(P); the matching test
  reads √5·R directly: diagonal entries ±1, exactly one off-diagonal entry per row equal to ±2,
  all else 0; Π from the off-diagonal support; S = diag(√5·R).
- Ledger identities gate every fold (P² = P, P + σP = I, rank 10); failure = instrument fault.

## Tests (T2–T6 are predictions on objects not yet seen)

- **T1 (one-5 conjecture at k = 10).** Every 10-node one-5 tree diagram has a 20-vertex tree
  parent. Fails if any diagram is orphaned. (Zero orphans at k ≤ 8.)
- **T2 (strict trees are folds or sector-strict).** Every strict tree at n = 20 is a Galois fold
  of a one-5 diagram OR is sector-strict (its automorphism sectors each σ-pair, per exp_13's T6
  instrument). Fails on any strict tree that is neither. (Lesson 5: the sector mechanism is now
  part of the sealed classification.)
- **T3 (the matching form).** On every strict Galois fold at 20: √5·R = S + 2Π with S = diag(±1),
  Π a perfect matching, and SΠ = −ΠS. Fails on any fold without this exact form. (20/20 at
  n ≤ 16.)
- **T4 (quotient = diagram; multiplicities).** On every such fold: the Π-quotient of the parent
  is graph-isomorphic to SOME one-5 partner, with every diagram edge covered by exactly 2 parent
  edges except a single edge covered by 3, and under some isomorphism the multiplicity-3 edge is
  that partner's 5-bond. Fails on any violation.
- **T5 (the single defect).** On every such fold: Π maps parent edges to parent edges except
  exactly one; the defect edge projects onto the multiplicity-3 edge; **and the defect edge is
  copy-internal** (both endpoints sign +1). Fails on any violation. (21/21 at n ≤ 16.)
- **T6 (consequences).** On every such fold: tr(RD) = 2/√5, ‖(I−P)BP‖² = 2/5, |R_vv| = 1/√5 at
  every vertex. (The trace clause is now a PROVED consequence of T3 + T5 — proposition in the r15 journal — so
  it doubles as a consistency check; the leakage and vertex clauses are scored on their own
  merits, the vertex clause being a consequence of T3 alone.)

If no strict tree exists at n = 20, T2–T6 are vacuous and recorded as such (not passed); T1 is
still scored. Kill scope: none of these touches the milestone kill sentence; failures retire the
corresponding clause at n = 20 and leave the n ≤ 16 record standing.

## Compute estimate

Charpoly + norm screen: ~823k trees × ~4 ms ≈ 1 h. Exact factorization on survivors: unknown
until the screen's selectivity at 20 is seen; the n = 16 validation measures it. Partner map at
k = 10: ~10³ polynomials. Bezout + matching checks per strict fold at 20×20: minutes each.
