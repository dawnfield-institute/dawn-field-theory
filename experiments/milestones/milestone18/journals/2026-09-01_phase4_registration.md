# Phase 4 registration (SEALED): n = 16, the first fresh strict folds

**Status: SEALED 2026-09-01** (the commit carrying this heading; nothing above ran before it).
Scored to this text (§2.6; no relaxation after the run).
Written under the exp_12 lessons: every operator is defined by construction here, and the cases
each recipe cannot reach are enumerated before sealing.

## Why n = 16

k = 8 is even. The odd-k theorem (bipartite self-duality forces λ = 2 as a rational root for odd
k) made every n = 14 fold core-grade; at k = 8 nothing forces a rational root, so strict golden
trees are expected again, as at n = 4, 8, 12. They would be the first strict folds not used in
building the T4 invariant or the two-component observations — the first genuinely fresh test bed
for both.

## Instruments (defined by construction)

- **Tree enumeration:** `nx.nonisomorphic_trees(16)` (19,320 trees), exhaustive, no sampling.
- **Golden grade** of a tree: factor the characteristic polynomial p of its Cartan matrix
  C = 2I − A over ℚ(√5). *Strict* — p = q·σ(q) with no rational factor. *Core* — the same after
  removing rational factors of even multiplicity. *Partial* — some but not all non-rational
  factors pair. *None* — otherwise. Known-answer self-test before the run: E₈ strict, D₆ core,
  A₅ none, and the n ≤ 13 census grades reproduced on a sample.
- **One-5 diagram** at k = 8: a tree on 8 nodes with one marked edge; Gram matrix M = 2I with −1
  on unmarked edges and −φ on the marked edge; q = charpoly(M). *Parent test:* q·σ(q) equals the
  characteristic polynomial of some 16-vertex tree (matched on the expanded polynomial).
- **Galois fold:** a golden tree whose p is q·σ(q) for a one-5 diagram q. **P** is the Bezout
  projector for q (extended Euclid in ℚ(√5)[t]: u·q + v·σq = 1, P = v(C)·σq(C)); on a strict
  fold this is the complete construction. **R = P − σ(P)**; **D** the degree matrix; **B = 2I − D**.
  Ledger identities checked before any evaluation: P² = P, P + σP = I, rank P = 8; a fold failing
  them is an instrument fault, reported, not scored.
- **Core-grade folds:** registered domain as in exp_12 T5 — rational cores of multiplicity
  exactly 2, resolved on the golden conic with a rational point in the search |x|, y ≤ 12, and
  invariants agreeing at two conic points. Outside the recipe, declared and not scored:
  multiplicity > 2 cores, quadratic rational cores, conics with no rational point in the search.
- **Sign split** of a fold: copy side = vertices with R_vv > 0, conjugate side = R_vv < 0
  (vertices with R_vv = 0, if any, reported and the fold excluded from T4/T5b with the exclusion
  declared).
- **Sector decomposition** of a non-Galois golden tree: orbits by the AHU rooted canonical form;
  S = span of orbit indicators, S⊥ its complement; the characteristic polynomial of C on a
  subspace with basis matrix V is det(VᵀCV − t·VᵀV)/det(VᵀV).

## Tests

- **T1 (the one-5 conjecture at k = 8).** Every tree-shaped 8-node Coxeter diagram with exactly
  one 5-bond has a 16-vertex tree parent. Fails if any diagram has none. (26/26 at k ≤ 6, all
  parented at k = 7.)
- **T2 (strict trees are folds).** Every strict golden tree at n = 16 is a Galois fold of a one-5
  diagram. Fails if a strict tree has no partner. Recorded, not scored: whether strict trees
  exist at 16 at all (their absence would be a fact about even k, not a failure of this law).
- **T3 (PREDICTION — the strict invariant on fresh folds).** On every strict Galois fold at 16:
  tr(RD) = 2/√5 and ‖(I−P)BP‖²_F = 2/5. Fails if either is violated on any strict fold.
- **T4 (PREDICTION — vertex law and two-component structure, strict folds).** On every strict
  Galois fold at 16: (a) |R_vv| = 1/√5 at every vertex; (b) the copy side induces a connected
  subtree; (c) the conjugate side induces exactly two components whose sizes equal (as a
  multiset) the sizes of the two halves of the diagram with its 5-bond removed. Fails if any of
  (a)–(c) fails on any strict fold. (Evidence: 7/7 at n ≤ 12.)
- **T5 (PREDICTION — trace law on core-grade folds).** On every registered-domain core-grade
  fold at 16: tr(RD) = 2/√5. The leakage norm is recorded, not scored (exp_12 separated the two
  laws). Domain size reported; an empty domain makes T5 vacuous, not passed. (Evidence: 12/12
  at n ≤ 14.)
- **T5b (PREDICTION — component relations on core-grade folds).** On every registered-domain
  core-grade fold at 16, under the sign split: components(conjugate) = components(copy) + 1 and
  cut = 2·components(copy). Fails on any violation. Recorded, not scored: the halves law of
  T4(c) on these folds (11/12 at n ≤ 14; the known exception det −464 satisfies T5b).
- **T6 (PREDICTION — sector classification).** For every golden tree at 16 that is not a Galois
  fold: the golden factors of C on S are exactly the quotient's golden factors, and the golden
  part on S⊥ is either empty (pure quotient fold) or pairs as q·σ(q). Fails if any golden factor
  on S⊥ does not pair, **and also fails if any non-Galois golden tree has the trivial orbit
  partition (orbit count = 16)** — such a tree folds by neither mechanism in any sector and is
  UNEXPLAINED, not vacuously a quotient fold. (Evidence at 14: 11 pure quotient, 2 mixed,
  0 unpaired, 0 unexplained.)

Consistency checks (theorems, recorded not scored): det(parent) = N(q(0)); signature(copy) =
signature(diagram); hyperbolic diagram's conjugate definite iff det(parent) < 0.

Kill scope: none of T1–T6 touches the milestone kill sentence. A T1/T2 failure retires the one-5
conjecture at k = 8, leaving k ≤ 7 standing. A T3/T4 failure retires that law at n = 16, leaving
the n ≤ 12 folds as instances. T5/T5b/T6 failures retire the corresponding extension only.

## Compute estimate

Grading 19,320 trees at 16×16: ~30 min (exp_11's 3,159 at n = 14 took 176 s). One-5 diagrams at
k = 8: 23 trees × 7 edges = 161 polynomials. Bezout projectors at 16×16: minutes per strict fold.
