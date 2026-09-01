# 2026-09-01: Provenance (prior art) and proof notes for Panels G/L

## Prior art — what is classical, what is anticipated, what appears to be ours

Searched 2026-09-01 (web; not exhaustive). Classical and well documented:
- E₈ → H₄, D₆ → H₃, A₄ → H₂ as Galois projections / Coxeter-Dynkin foldings: Moody & Patera
  (quasicrystals and icosians, 1993); Koca et al., *Noncrystallographic Coxeter group H₄ in
  E₈* (E₈ roots = H₄ ⊕ σH₄ as icosians); Dechant, *The E₈ geometry from a Clifford
  perspective* (arXiv:1603.04805); Moxness, *Mapping the fourfold H₄ 600-cells emerging from
  E₈* (the two shells each carry chiral L/R copies — consistent with Block A's 120 + 120).
- Dechant, Bœhm & Twarock, *Affine extensions of non-crystallographic Coxeter groups induced
  by projection* (arXiv:1110.5228): affine H₂/H₃/H₄ from projections of affine A₄/D₆/E₈;
  Cartan matrices over ℚ[τ]; and the remark that admitting other number fields in the Cartan
  matrix "opens up possibilities in hyperbolic geometry." This is the closest anticipation of
  the hyperbolic extension found.

Not found in the searches (so, until a specialist says otherwise, ours in this form):
- the exhaustive classification of trees whose Cartan polynomial is q·σ(q) (n ≤ 13);
- the hyperbolic parents: cat8 → [3,5,3] and the one-5 rule (26/26 at k ≤ 6);
- the Galois-fold / quotient-fold dichotomy with no residue;
- the fold invariant tr(RD) = 2/√5 and its role as a mechanism diagnostic;
- the field-resonance law as a stated, tested rule (its proof is standard cyclotomic theory).

Register: the assembly and the specific theorems are the contribution; the foldings are not.

## Proof notes — the fold invariant is combinatorial

On every strict Galois fold at n ≤ 12 (7/7), with P the node-space Bezout projector and
R = P − σ(P):
1. **|R_vv| = 1/√5 at every vertex.** Equivalently P_vv ∈ {(5±√5)/10} — half the squared
   shell radii of Block A. The node-space projector encodes the two-shell structure vertex by
   vertex.
2. **The copy-heavy vertices (+) form a connected subtree on n/2 vertices; the conjugate-
   heavy set (−) has exactly two components; the cut is exactly two edges.**
3. Hence Σ_v ε_v·deg(v) = [2(n/2−1) + 2] − [2(n/2−2) + 2] = 2, and tr(RD) = 2/√5. Together
   with tr(PA) = 0 (from tr(PC) = n, tr(P) = n/2) this is the whole invariant.

Open for a proof: (1), and the two-edge cut in (2). Conjecture: the cut edges are the two
parent edges lying over the diagram's 5-bond. Quotient folds have R_vv = 0 on the symmetric
modes, hence tr(RD) = 0 — the diagnostic.

## exp_12 T1 (registration 5bc8faff) — PASS
Finite: A₁₃ (h=14) √7 only; A₁₄ (h=15) √5 only; D₁₃ (h=24) √2, √3; D₁₄ (h=26) √13 (core) —
4/4 as predicted by √d ∈ ℚ(ζ_{2h}). Affine D̃₅–D̃₈, Ẽ₆–Ẽ₈: never strict (partial at most), as
predicted. Remaining exp_12 tests await exp_11's fold list.

## Addendum (after exp_12) — the vertex law |R_vv| = 1/√5 as a rational statement

On a strict fold, P = a(C) with a ∈ ℚ(√5)[t] (the Bezout combination), so R = P − σ(P) =
(a − σa)(C) = √5·b(C) with **b ∈ ℚ[t]**. R² = I (Panel L2) reads 5·b(t)² ≡ 1 (mod q·σq). The
vertex law says the **rational** symmetric matrix b(C) has diagonal ±1/5 at every vertex.
Equivalent forms:

- P_vv = (1 + R_vv)/2 ∈ {(5+√5)/10, (5−√5)/10} = {φ/√5, 1/(φ√5)} — the Binet weights. Every
  vertex vector splits between the H-copy and its conjugate in Binet proportions; the same pair
  is the diagonal of the Fibonacci Q-matrix's φ-eigenline projector (the minimal instance of the
  Complement Identity): φ²/(φ²+1) = φ/√5.
- Row norms: Σ_w b(C)_vw² = 1/5, so diagonal ±1/5 forces Σ_{w≠v} b(C)_vw² = 4/25 at every vertex.
- With tr(b(C)) = 0 and tr(b(C)·D) = 2/5 (T4), the sign pattern gives the two-component lemma of
  explore_r7.

**Scope (explore_r7b):** the vertex law is a *strict-fold* statement. On the 12 registered-domain
core folds the diagonal of R takes four values — ±1/√5 on most vertices and a second pair
(±3√5/10, ±4√5/15 or ±√5/10, fold-dependent) on others — while tr(RD) = 2/√5 still holds 12/12.
The sign-split structure largely survives: 11/12 have a connected copy side, a 2-edge cut, and a
two-component conjugate side whose sizes equal the diagram's halves at the 5-bond; the exception
(n = 14, det −464) has copy side in 3 components, cut 6, conjugate side in 4 components — and
still satisfies components(conj) = components(copy) + 1 and cut = 2·components(copy), the two
relations the lemma derives without using constancy of |R_vv|. So the component relations are the
robust part (19/19 strict + core-domain); the vertex law and the halves law are where a proof
should start, on strict folds.

What a proof needs: why the diagonal of b(C) is constant in absolute value on a strict fold. The
known facts are spectral (b(C) is determined by q); the vertex law lives in the vertex basis,
where diagonal entries of polynomials in C count weighted closed walks. A walk-counting argument
through the tree's bipartite structure (C = 2I − A) is the natural route; not yet done. Status:
open lemma — 7/7 strict folds.
