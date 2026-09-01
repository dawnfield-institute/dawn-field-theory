# explore_r17 — the construction theorem: parents are branched double covers

Script `explore_r17_construction_theorem.py`; log `explore_r17_log.txt`. Written while exp_15's
battery runs; nothing here depends on its outcome.

## The construction

Given a one-5 diagram (D, e*) — a tree D on k nodes with marked edge e* = {A, B} — build
**parent(D, e\*)** on 2k vertices (v, 0), (v, 1):
- every ordinary edge {x, y}: two lifts (x,0)–(y,0) and (x,1)–(y,1) — the cover is trivial there;
- the 5-bond {A, B}: three lifts — (A,0)–(B,0) (the **defect**), (A,0)–(B,1), (A,1)–(B,0).

Edge count 2(k−2) + 3 = 2k − 1: a tree. For H₂ it produces the A₄ path; for H₄ it produces E₈,
leaf position and all.

## Theorem 1 (the charpoly identity — proof by golden decoupling)

charpoly(parent(D, e*)) = q·σ(q), with q the Gram polynomial of D (bond weight −φ at e*).

*Proof.* For γ ∈ {1/φ, −φ} let W_γ = span{ (v,0) + γ·(v,1) : v ∈ D }. Ordinary edges act
identically on both sheets, so they preserve each W_γ with weight −1. At the 5-bond the total
edge operator sends (B,0) + γ(B,1) ↦ (1+γ)(A,0) + (A,1), which lies in W_γ with weight
μ = 1/γ precisely when 1 + γ = 1/γ, i.e. **γ² + γ − 1 = 0** — roots γ = 1/φ (giving bond weight
μ = φ) and γ = −φ (giving μ = −1/φ). The same holds in the other direction by symmetry. The two
sectors are orthogonal since 1 + (1/φ)(−φ) = 0, and each has dimension k, so
C_parent ≅ Gram_{−φ}(D) ⊕ Gram_{+1/φ}(D) = Gram(D) ⊕ σ(Gram(D)). ∎

Exhaustive check: 117/117 placements at k ≤ 7.

**Corollary (T1, existence — the one-5 conjecture is PROVED).** Every one-5 diagram has a tree
parent at every k: the construction is it. Four phases of zero-orphan searches (k ≤ 8) are now
instances of a theorem.

## Theorem 2 (the matching form holds on construction parents)

If gcd(q, σq) = 1 (strict case), the Bezout projector for q is the orthogonal projector onto
W_{1/φ}, which is **block-diagonal over sheet pairs** (its basis vectors ((v,0) + (v,1)/φ)/‖·‖
are orthonormal and each involves one pair). Per 2×2 block, with weights (1, 1/φ)/norm against
(1, −φ)/norm:

    R_{(v,0),(v,0)} = (φ² − 1)/(φ² + 1) = 1/√5,   R_{(v,1),(v,1)} = −1/√5,
    R_{(v,0),(v,1)} = 2φ/(φ² + 1) = 2/√5,          all cross-pair entries 0.

So √5·R = S + 2Π **exactly**, with S = ±1 by sheet and Π the sheet swap — verified entrywise on
strict construction samples (4/4). All laws follow (r15 Addenda 2–5): vertex, trace, leakage,
structure, halves — every sealed invariant of Phases 3–6 is a **theorem on construction
parents**. Π is the deck transformation; the defect is the branch; the multiplicity-3 edge is the
branch locus; placement selection is trivial on constructions (the bond is where you built it).

## What remains — one rigidity conjecture

- (proved) construction ⇒ matching form ⇒ all laws (Theorems 1–2 + r15 Addenda).
- (proved, r15 Addendum 5) matching form ⇒ the fold has exactly the construction's lift pattern.
- **Open (rigidity):** every strict Galois fold — every tree whose characteristic polynomial is
  q·σ(q) for a one-5 diagram — has the matching form; equivalently, is isomorphic to
  parent(D, e*) for some placement. This is exactly what exp_15's sealed T3 is testing on the 66
  strict trees at n = 20. Cospectral twins are explained: distinct placements with equal q·σ(q)
  give distinct constructions with equal spectra.

Also settled by the construction: the sector-strict trees (exp_13's det −775) are NOT
construction parents of one-5 diagrams — the classification's two strict species are
"construction parents" and "sector-strict", and exp_15's T2 tests exactly that dichotomy.

## Retro-check (explore_r17b): rigidity holds on everything known

Every one of the 21 known strict Galois folds at n ≤ 16 is **graph-isomorphic to a construction
parent** (21/21). And the n = 16 cospectral pair resolves completely: its two members are the
constructions of the SAME diagram polynomial at its TWO 5-bond placements — one matches (0, 5)
with halves [3, 5], the other (1, 4) with halves [1, 7]. Cospectral parents are not a wrinkle;
they are the construction enumerating placements. At n ≤ 16 the strict classification is
airtight: **strict trees = construction parents ∪ sector-strict trees**, with zero residue.
Rigidity at n = 20 is exp_15's sealed T3, in flight.
