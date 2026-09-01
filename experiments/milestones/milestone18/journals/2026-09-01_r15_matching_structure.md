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
