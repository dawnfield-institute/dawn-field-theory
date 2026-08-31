# 2026-08-31 (night): The σ-Ledger — three exact results

**Mode: EXPLORING, declared.** Nothing here is scored. Everything here is exact (sympy, no
floats) and reproducible from `scratchpad` scripts R1–R3 (to be promoted into `scripts/`
under the amended registrations before any scoring). These results reshape Blocks B and C.

## Result 1 — Galois conjugation IS complementation (the Complement Identity)

Let P be the orthogonal projector onto the H₄ subspace of E₈ (Block A's verified
instrument; entries all golden rationals). Apply the field conjugation σ(√5 → −√5)
entrywise. Verified exactly:

    σ(P) = I − P          P·σ(P) = 0          tr(P) = 4
    R := P − σ(P) is PURE golden (rational part zero), and R² = I

Same identity holds for A₄ → H₂ in its 4D root space. **Vieta's relations lift to
operators**: numbers φ+ψ = 1, φψ = −1; projectors P + σP = I, P·σP = 0, (P−σP)² = I.

Readings, each exact where stated:
- **M13**: identity-IS-complement is realized as the field automorphism — the conjugate of
  the projection *is* its complement. Andy's ψ ("the algebraic shadow") is complementation.
- **PAC**: P + σP = I is the ledger (P + A = C with C the identity); R is the Δ channel —
  purely golden, and an involution.
- **M15**: choosing a factor/copy is gauge; σ is the observer swap.

## Result 2 — The Ledger Theorem: the landscape is complete

For X the Cartan matrix and q the H-partner's characteristic polynomial:

    charpoly(A₄) = q·σ(q), q = charpoly(H₂)   ✓ exact
    charpoly(D₆) = q·σ(q), q = charpoly(H₃)   ✓ exact
    charpoly(E₈) = q·σ(q), q = charpoly(H₄)   ✓ exact

and the negative side is CLEAN: A₅, D₄, D₅, E₆, E₇ Cartans have **zero** golden content
over ℚ(√5). Golden ledger factorization occurs on exactly the folding sources {A₄, D₆, E₈}
and nowhere else in ADE. The whole (rational charpoly) is pinned; the two conjugate halves
carry equal-and-opposite golden content. Class-level = the rational product; representative
= which factor you stand in.

## Result 3 — The Knife-Edge: golden structure is a boundary-condition resonance

M(s) = (1−s)·Laplacian + s·Cartan is a Robin family (leaf diagonal 1+s, branch 3−s).
Factor shapes over ℚ(√5) at s ∈ {0, ¼, ½, ¾, 1} (golden marked *):

    path4 (A₄): [1,1,2] | [2,2] | [1,1,2] | [2,2] | *[1,1,1,1]
    path5 (A₅): *[1,1,1,1,1] | [2,3] | [2,3] | [2,3] | [1,1,1,2]
    D6-tree:    [1,1,1,3] | [1,5] | [1,5] | [1,5] | *[1,1,2,2]
    E8-tree:    [1,1,1,2,3] | [8] | [8] | [4,4] | *[4,4]

Golden structure exists only at isolated boundary conditions: s=1 (Dirichlet/Cartan) for
the folding diagrams; s=0 (Neumann/Laplacian) for the 5-path — which is why stray golden
sightings occur in A-family Laplacians while E₈'s golden structure is invisible in the
corpus's Laplacian channel. **Field membership is a boundary-condition resonance; the
φ-casualty list is (at least partly) an instrument-channel fact.** Follow-up in the
registered version: full n-sweep (P₁₀'s field contains √5 — partial content expected),
finer s-grid, and the E8 s=¾ curiosity ([4,4] over ℚ without √5).

## What must now happen (registration drafts, this directory, DRAFT-marked)

- Block C re-pose: the σ-Ledger claims C1–C3, including the LOCAL-shadow question — R in
  the simple-root basis vs the M13.5 orbit kernel (Aut-equivariance of R; do its ±1
  eigenspaces refine the orbit quotient?). Structural claims labeled (S).
- Block B re-pose: the dynamically-visible-field question — stochastic stress/FPT dynamics
  (M-R exp_15/16 machinery, refactor-gated) in the s=1 channel, ledger-antisymmetric
  observables on {A₄,D₆,E₈} vs controls. The only part that can fail against noise — and
  therefore the only part that counts as evidence.

Nothing runs for score until Peter commits the amended registrations.

---

# Late-night addendum: R4–R6

## Result 4 — The Complement Identity is a LAW, and Andy's Q-matrix is its smallest instance

Scope-check on Result 1 (persistence rule: check the premise). The identity σ(P) = I − P is
not special to foldings: for ANY rational matrix whose spectrum splits into ℚ(√5)-conjugate
halves, P = f(M) with f ∈ ℚ(√5)[t] (Lagrange), σ commutes with polynomials in rational M,
so σ(P) is the conjugate-half projector — the complement. Two-line general proof.

- **Minimal instance: the Fibonacci Q-matrix [[1,1],[1,0]]** (spectrum {φ, ψ}) — verified
  exactly; its φ-projector P = [[1/2+√5/10, √5/5],[√5/5, 1/2−√5/10]] satisfies σ(P) = I−P.
  Andy's §2.9 already contained the identity.
- **Refinement (honest):** the identity is clean iff the spectrum is FULLY golden-paired
  (A₄, E₈, Q). D₆'s spectrum has a rational core (eigenvalue 2, multiplicity 2, one per H₃
  copy) where σ gives no guidance — exactly the isotypic ambiguity exp_02 T4 hit
  numerically. The ledger there has a self-conjugate middle.
- So the foldings' special content is Result 2 (WHICH systems are golden-paired), and the
  identity is the general mechanism those systems inherit.

## Result 5 — The ledger swap is invisible from inside the lattice

R = P − σP is an isometry of R⁸ (norms preserved) mapping the E₈ root system entirely off
itself: **0/240 images are roots**. R ∉ W(E₈); it carries E₈ onto a golden twin. The swap
acts on the FIELD, not the lattice — consistent with M15: the copy you stand in is frame
data, unreachable by lattice-internal (class-level) operations.

## Result 6 — Bipartite Duality, with a CONFIRMED PREDICTION

All trees are bipartite ⇒ parity conjugation gives **spec M(2−s) = 4 − spec M(s)** (verified
exactly at s = 1/3 vs 5/3 on path5 and the E8-tree). Consequences:

- Golden boundary points come in dual pairs about the self-dual point **s = 1 — the Cartan**.
- Dense exact map (97 rationals): path4 golden at {−1, 1}; path5 at {0, 2}; E8-tree at {1}
  only. The duality then PREDICTED path4 golden at s = 3 (dual of −1, outside the sampled
  grid) — **verified: golden, with control point s = 5/2 correctly barren.** A prediction
  made before measurement, from a symmetry, confirmed after. First of the milestone.
- Character: the branched (folding-bearing) tree is golden ONLY at the self-dual point.
  Paths carry golden dual pairs: path5 at s ∈ {0,2} and path4's Cartan share the SAME
  golden quadruple {2−φ, 3−φ, 1+φ, 2+φ}; path4's outer points (s = −1, 3) carry the
  Q-matrix pair {φ, ψ} itself plus a ℚ(√13) pair — free ends put Binet's spectrum on the
  4-path.

Registered-version consequences for the Block B/C DRAFT: exp_07 registers the duality as
the organizing invariant (golden set closed under s ↦ 2−s; Cartan self-dual; branched
folding trees golden only there), with the full n-sweep enumerated before running.
