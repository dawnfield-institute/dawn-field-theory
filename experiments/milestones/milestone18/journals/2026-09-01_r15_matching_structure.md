# explore_r15 — the matching structure of the reflection (exploration, not registered)

Scripts `explore_r15_matching_structure.py` (n ≤ 12 and n = 16) and the placement check below;
results `explore_r15_matching_n12.json`, `explore_r15_matching_n16.json`. Objects: **all 20
strict Galois folds known** (7 at n ≤ 12; the 13 fresh folds at n = 16).

## The conjecture the data forces (20/20 on every clause)

On every strict Galois fold with reflection R = P − σ(P):

1. **√5·R = S + 2Π**, where S = diag(±1) and Π is a symmetric permutation matrix — a perfect
   matching on the parent's vertices. (Each row: diagonal ±1, exactly one off-diagonal entry ±2…
   observed +2 throughout, everything else 0.)
2. **S·Π = −Π·S** — matched vertices carry opposite signs. This alone gives (S + 2Π)² = 5I,
   i.e. R² = I, and |R_vv| = 1/√5 at every vertex: the vertex law is a corollary of the form.
3. **The quotient of the parent tree by Π is graph-isomorphic to a one-5 partner diagram**, with
   every diagram edge covered by exactly **2** parent edges except a single edge covered by
   **3** — and that multiplicity-3 edge is the diagram's 5-bond. Edge count check:
   2(k−2) + 3 = 2k − 1 = n − 1 — every parent edge projects; Π never matches adjacent vertices.
4. **The multiplicity-3 edge is the *realized* 5-bond.** On the cospectral-placement fold
   (n = 16, det −239 — the same tree admits two inequivalent 5-bond placements with identical
   q·σ(q)), the mult-3 edge maps onto the (0,5) placement with halves [3, 5] and not onto the
   (1, 4) placement with halves [1, 7] — precisely the placement exp_13's vertex structure
   selected. The quotient reads the selection off directly.

## What this reduces

If the conjecture holds, the entire strict-fold law package becomes corollaries: the vertex law
(from 1–2), R² = I (from 2), the Binet vertex weights, and — by sign bookkeeping over the lifted
edges, using that each diagram edge lifts to two parent edges and the 5-bond to three — the
trace law tr(RD) = 2/√5, the two-component structure, and the halves law. The single proof
target is now: *why does the Bezout reflection of a strict fold have the matching form, and why
does the golden bond lift with multiplicity three?* The extra lifted edge over the 5-bond is
also the cleanest mechanism yet for placement selection (exp_13 T4's discovery) and gives the
2:1 fold map an explicit combinatorial realization: Π's pairs are the fibers.

Reading, labelled: the "3 = 2 + 1" over the golden bond looks like the double cover plus one
extra passage — where the weight φ = 2cos(π/5) is generated. Not formalized.

## Status

Exploration-grade, 20/20, no counterexample; not registered. Natural Phase 6 material: seal the
matching form and the mult-3/realized-bond clause as predictions at the next strict tier
(n = 20), and attempt the proof in parallel — the form is rigid enough that a proof may fall to
direct computation with the Bezout construction.

## Addendum — the single defect (20/20)

Two further clauses, checked on all 20 folds (pure combinatorics on the saved matchings):

5. **Π maps parent edges to parent edges except exactly one.** The defect edge — a copy-side
   edge whose Π-image is a non-edge — is unique on every fold.
6. **The defect edge projects onto the multiplicity-3 edge**, i.e. it sits over the realized
   5-bond. So Π is an involution of the parent that is an isomorphism everywhere except one
   edge, and that single failure is where the golden bond lives: the mult-3 edge's three lifts
   are the defect edge plus the Π-paired couple, and its "missing fourth lift" is exactly the
   non-edge Π points at.

The conjecture in full: on a strict fold, √5·R = S + 2Π with SΠ = −ΠS; Π is a near-automorphism
with a single defect; the quotient is the one-5 partner; multiplicity 2 everywhere except 3 over
the realized bond, whose position the defect marks. One object — the reflection — carries the
fibers, the diagram, the bond, and the placement.

## Addendum 2 — the trace law reduces to the matching clauses (proposition), and a 21st fold

**Defect side (explore_r15c, 21/21):** the defect edge is copy-internal — both endpoints carry
sign +1 — on every strict fold known.

**Proposition (trace-law reduction).** Assume clauses 1–2 and 5 (√5·R = S + 2Π, SΠ = −ΠS, Π maps
edges to edges except the single defect edge, which is copy-internal). Then tr(R·D) = 2/√5.
*Proof.* tr(R·D) = (1/√5)·Σ_v d_v s_v = (1/√5)·Σ_{edges {a,b}} (s_a + s_b). Π is an involution;
pair each non-defect edge {a,b} with its image {Π(a), Π(b)} (also an edge, by clause 5). By
SΠ = −ΠS, s_{Π(a)} = −s_a and s_{Π(b)} = −s_b, so the pair's contributions cancel; a non-defect
edge fixed by Π would force s_a = −s_b and contributes 0 on its own. The defect edge is the only
edge left unpaired, and it is copy-internal: contribution s_a + s_b = 2. Hence Σ d_v s_v = 2. ∎

So T6's trace clause is a theorem conditional on the matching clauses; the vertex law already
follows from clauses 1–2. Still open as reductions: the leakage value 2/5 and the two-component/
halves structure (the latter is plausibly clause-5 bookkeeping as well; not yet written).

**The 21st fold — cospectral parents (explore_r16b + r15d).** The strict-hunt validation at
n = 16 found 15 strict *trees* on 14 strict *polynomials*: two non-isomorphic 16-vertex trees
share one strict characteristic polynomial, so the same one-5 diagram (same q·σq, two cospectral
placements) has **two non-isomorphic parents**. exp_13 only ever evaluated one tree per
polynomial; the twin passes the full battery — form, anticommuting signs, single copy-internal
defect, quotient isomorphic to the diagram, multiplicities [2×6, 3], defect over the mult-3
edge. Matching-structure record: **21/21**. New phenomenon for the classification: the strict
fold map is not injective on parents; a diagram can be "heard" by more than one tree.

**Instrument note.** The 15-vs-14 discrepancy was a counting-basis mismatch (exp_13 counted
distinct polynomials, r16b counted trees). The pipeline's validation line now states its basis.
Lesson: a known-answer gate must declare the basis it counts in.

## Addendum 3 — how far the structure law reduces (written while exp_15 runs)

From the matching clauses alone:
- SΠ = −ΠS makes the two lifts of any multiplicity-2 diagram edge a Π-pair with **jointly flipped
  signs**, so each such pair is either {copy-internal, conjugate-internal} or {cut, cut}; a
  mixed-sign (cut) edge maps to a cut edge.
- The trace proposition gives Σd_v s_v = 2, hence degree-sum(copy) − degree-sum(conj) = 2, hence
  e_copy − e_conj = 1 and (r7 lemma) components(conj) = components(copy) + 1,
  cut = 2·components(copy). Since both sides are nonempty in a tree, cut ≥ 2 — so **at least one
  Π-pair of lifts is {cut, cut}** by parity.
- **If exactly one pair is {cut, cut}** — necessarily the mult-3 edge's non-defect pair, since
  the defect is copy-internal — then the copy side receives one internal lift of every mult-2
  edge plus the defect: k − 1 edges on k vertices, a spanning tree, hence **connected**; the
  conjugate side receives k − 2 internal lifts on k vertices, hence **exactly two components**;
  and the cut is the two lifts over the realized 5-bond, giving the halves law by projection.

So the full package (vertex, trace, two-component, halves) now rests on the matching clauses plus
one residual statement: *no multiplicity-2 edge lifts as a cut pair* (equivalently, cut = 2, or
copy connected — all equivalent given the above). 21/21 observed; not yet derived. The leakage
value 2/5 remains a separate open reduction.
