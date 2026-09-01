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

**explore_r10:** b(t) computed explicitly on all seven strict folds (`explore_r10_b_polynomial.json`).
Finding: **5·b(t) is an integer polynomial on every fold** (leading coefficient 2 or 4), so the
vertex law is an integrality statement — diag(b(C)) = ±1/5 with 5b ∈ ℤ[t] — and
(σq − q)/√5 is a monic integer polynomial on every fold (constant 1 for A₄; (t−3)(t−1) for E₈;
(t−2)² for cat8; degree-4 products at n = 12). No closed form for b yet; the mod-5 reductions of
5b show no common factor pattern. The 5-adic structure (everything happens over the ramified
prime) is the visible thread.

## Two lemmas proved (2026-09-01, evening) — the off-core sector

**Lemma 1 (off-ledger identity).** Let T be a tree with Cartan matrix C whose characteristic
polynomial is p = r·g, where r ∈ ℚ[t] is the product of the rational irreducible factors and
g = q·σ(q) with q ∈ ℚ(√5)[t], gcd(q, σq) = 1, and gcd(r, g) = 1. Let Qc be the orthogonal
projector onto ker r(C) and P_off the Bezout projector for q with the core removed
(P_off = (v·σq)(C)·(I − Qc), where u·q + v·σq = 1). Then

    P_off + σ(P_off) = I − Qc,   P_off² = P_off,   P_off·σ(P_off) = 0.

*Proof.* (a) Qc is σ-fixed: the minimal polynomial m of C is squarefree; by CRT in ℚ(√5)[t]/(m)
there is h with h ≡ 1 mod each rational factor of m and h ≡ 0 mod each golden factor, and
Qc = h(C). σ permutes the golden factors of m and fixes the rational ones, so σ(h) satisfies the
same interpolation conditions; by uniqueness mod m, σ(h) = h, hence h ∈ ℚ[t] and σ(Qc) = Qc.
(b) σ(P_off) = (σv·q)(C)·(I − Qc) by (a). The sum is [(v·σq + σv·q)(C)]·(I − Qc). Modulo q:
v·σq ≡ 1 (Bezout) and σv·q ≡ 0, so the bracket ≡ 1; modulo σq: apply σ to the Bezout identity
to get σv·q ≡ 1, and v·σq ≡ 0, so again ≡ 1. Since gcd(q, σq) = 1, CRT gives
v·σq + σv·q ≡ 1 (mod g). Hence the bracket acts as the identity on every eigenspace of C for a
root of g, and (I − Qc) kills the eigenspaces for roots of r; as spec(C) = roots(r) ⊔ roots(g)
and C is diagonalizable, the sum is I − Qc. (c) Idempotence and orthogonality: on a root of q
the factor v·σq evaluates to 1 and σv·q to 0; on a root of σq the reverse; on roots of r both
are annihilated by (I − Qc). ∎

This upgrades r13's 61/61 observation (i) to a theorem; nothing about the tree beyond the
factorization shape is used, so it holds at every n.

**Lemma 2 (B-invariant cores contribute no mixed leakage).** With the notation above and
B = 2I − D, suppose B·Qc = Qc·B. Then for every core line P_v ≤ Qc (P_v·Qc = Qc·P_v = P_v):

    (Qc − P_v)·B·P_off = 0   and   σ(P_off)·B·P_v = 0.

*Proof.* Qc·B·P_off = B·Qc·P_off = 0 since Qc·P_off = 0. Hence
(Qc − P_v)·B·P_off = (Qc − P_v)·Qc·B·P_off = 0. For the second: B·P_v = B·Qc·P_v = Qc·B·P_v,
and σ(P_off)·Qc = 0. ∎

Consequently, for a B-invariant core the total leakage ‖(I−P)BP‖² reduces to the off→off block
plus the core→core block, independent of the core gauge — which is why the r11 block anatomy
found the mixed blocks equal to zero exactly on the B-invariant folds. (The empirical fact that
the core→core block also vanished on all 12 domain folds, including the two non-B-invariant
ones, is *not* covered by this lemma and remains an observation.)

**What remains open, precisely (updated same day — see the r15 journal, Addenda 2–5).** The
strict-fold laws are no longer independently open: conditional on the matching form
(√5·R = S + 2Π with SΠ = −ΠS, observed 21/21), the commutator [R, C] = 0 — automatic, R being a
polynomial in C — derives the non-adjacent matching, the single copy-internal defect, the
multiplicity pattern with one special edge, cut = 2, the two-component and halves structure, the
trace law, the leakage law, and the vertex law (r15 journal, assembled theorem in Addendum 5).
**The open conjecture is the matching form itself**, plus T4's spectral half (that the Π-quotient
is the one-5 partner with the special edge at its realized 5-bond). For core folds, Lemmas 1–2
above transfer the frame to the off-core sector; the modulated vertex law and the third clean
kernel class (exp_14) remain open there. The integrality route (5·b ∈ ℤ[t], explore_r10) is the
standing suggestion for the matching form: it says the integer matrix 5·b(C) squares to 5I with
row-norm² = 5, whose integer solutions are a short list of patterns — why the (1, 4) pattern is
the one realized is the question.
