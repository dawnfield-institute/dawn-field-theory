# M15 exp_05 Pre-Registration: The General-k Holonomy Limit (K1 vs K2)

**Date:** 2026-07-17
**Status:** REGISTERED BEFORE DATA — no k > 2 holonomy computation has been run as of
this commit. The only holonomy data in existence are the k = 2 results (exp_01,
exp_04), which serve as the anchor gate below.
**Target script:** `scripts/exp_05_general_k_limit.py`

## Data state at registration

`milestone15/results/` contains only exp_01/exp_02/exp_04 outputs. No k = 3 or k = 4
frame transport, holonomy, or limit has been computed, numerically or analytically,
beyond the conjecture sentence recorded in
`journals/2026-06-12_holonomy_closed_form.md` §6–7.

## The two candidates (both derived here, before any computation)

The k = 2 closed form gave lim θ(m) = 8/3 = 2·(1 + 1/3), from the odd-harmonic
cotangent poles q = 1, 3 available to the two-mode frame. Two structural readings were
left open, and they **diverge sharply at k = 3**, making this a clean discriminator:

- **K1 (odd-harmonic reading, the derivation's own structure):**
  lim θ_k = 2·Σ_{q odd, q ≤ 2k−1} 1/q.
  Predictions: k=3 → 2·(1 + 1/3 + 1/5) = **46/15 ≈ 3.0667**;
  k=4 → 2·(1 + 1/3 + 1/5 + 1/7) = **352/105 ≈ 3.3524**.
- **K2 (Fibonacci-ratio reading, the [D]-grade coincidence 8/3 = F₆/F₄):**
  lim θ_k = F_{2k+2}/F_{2k}.
  Predictions: k=3 → F₈/F₆ = 21/8 = **2.625**; k=4 → F₁₀/F₈ = 55/21 ≈ **2.6190**.

K1 predicts the limit *grows* with k; K2 predicts it *shrinks* toward φ² ≈ 2.618.
The candidates differ by > 14% at k = 3 — any sane observable separates them.

## Registered observable (locked)

For the k-frame at each vertex of C_m: frame = top-k adjacency eigenvectors of the
deleted-vertex path P_{m−1} (extending exp_04's k = 2 construction verbatim). Per-edge
transport = polar rotation part R of the k×k overlap matrix on shared support
(Procrustes, as in `core/representative.py::edge_transport`). Per-edge angle
θ_T = **sum of the positive rotation angles** of R (for k = 2 this reduces to the
single rotation angle — see anchor gate). The registered quantity is the large-m limit
of m·θ_T(m), computed for m up to 400 with Richardson extrapolation in 1/m, and —
derivation-first, per the exp_04 lesson that raw extrapolation overshot 8/3 into a
false e — an analytic large-m limit of the skew part of the overlap matrix is
attempted alongside; where the analytic limit exists it overrides extrapolation.

## Anchor gate

The k = 2 pipeline of this script must reproduce lim m·θ_T = 8/3 (within 0.1% via the
analytic route, 1% via extrapolation) before k = 3, 4 results are read. Failure of the
anchor voids the run (instrument bug, not physics).

## Decision rule (locked)

Let L_k be the measured limit for k = 3 and k = 4.
- If both L_3 and L_4 fall within 1% of K1's predictions and outside 1% of K2's → **K1
  CONFIRMED**, K2 (the Fibonacci reading of 8/3) is DEAD as a coincidence, formally.
- Symmetrically for K2 (this would kill the odd-harmonic reading and make the
  Fibonacci structure real — the more surprising outcome).
- If k = 3 and k = 4 disagree on which candidate they match, or neither matches within
  1% → **both candidates die**; the general-k limit is recorded OPEN with the measured
  values reported [D], and no third formula is fitted post hoc in this round.

## Registered secondary (exploratory, no kill attached)

**K3 (ℤ₂ parity classification):** record det(edge transport) sequences around C_m for
k = 2, 3, 4 and whether reflections telescope to an even count (det H = +1) at every
m — first data for the Phase-2 twist-classification question (§4 of the closed-form
journal). Classification only; no pass/fail.

## Threats to validity

- Eigenvalue near-degeneracy of P_{m−1} at larger k can make the frame ill-defined at
  small m; registered handling: skip any (m, k) where the k-th spectral gap
  < 1e-9 and report skips. Limits are about large m, where gaps are clean.
- Angle-sum observable choice: locked above. If the analytic derivation reveals the
  natural additive object is different (e.g. only a subset of angle pairs
  contributes), the discrepancy is REPORTED against this registration, not silently
  substituted.

## Outcome commitment

Whichever way it lands — K1, K2, or double-death — the result updates
`journals/` (outcomes file citing this commit), the milestone15 README Phase-2
section, and the `milestone15-representative-problem` FDO. The 8/3 = F₆/F₄ note in
the closed-form journal is upgraded to a claim or formally retired accordingly.
