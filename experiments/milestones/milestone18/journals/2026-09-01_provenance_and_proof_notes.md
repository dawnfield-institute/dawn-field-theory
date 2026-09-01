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
