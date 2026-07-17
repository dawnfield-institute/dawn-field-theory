# The General-k Generator Is the Box Momentum Operator (Derivation; Confirmation Pre-Registered as exp_06)

**Date:** 2026-07-17
**Status:** DERIVATION, developed *after* exp_05's measured L₃, L₄ were known —
disclosed plainly: its evidential weight therefore rests entirely on the
pre-registered k = 5, 6 confirmation (exp_06 registration, committed with this
journal, before any k = 5, 6 computation).

## 1. The derivation

Extend exp_04's overlap asymptotics to the k-frame. Entries of the raw overlap
M (analytic sine convention, diag > 0):

  M_{j'j} = (2/m)[cos(πj/m)·S1(j',j) + sin(πj/m)·S2(j',j)]

For j' ≠ j the S1 term is O(1/m²) (full-range orthogonality; only the removed
endpoint survives) and drops. The sine sums F̃(q) = Σ_{s=1}^{m−2} sin(πqs/m)
obey: q odd → cot(qπ/2m) − O(1/m) → 2m/(qπ); **q even → O(1/m)** (the full
root-of-unity circle cancels; only the endpoint term survives). Hence a
**parity selection rule**: the generator couples only opposite-parity modes.
With s = j+j', d = j'−j (d odd):

  m·M_{j'j} → 2j·(1/s + 1/d),   m·M_{jj'} → 2j'·(1/s − 1/d)

  **G_{j'j} ≡ lim m·skew(M)_{j'j} = (s² − d²)/(sd) · … = 4jj'/(j'² − j²)**  (d odd; else 0)

This is *exactly* the particle-in-a-box momentum matrix element on [0, 1]:
⟨j'|d/dx|j⟩ = 4jj'/(j'² − j²) for j + j' odd, 0 otherwise — selection rule
included. Conceptually forced: transporting the complement frame along the
cycle shifts the deleted vertex by one site; the box translates; the connection
generator is the generator of translations, i.e. momentum, projected to the
top-k frame. The limit is then

  **L_k = Σ (positive angle pairs of G) = Σ singular-value pairs of the k×k
  skew matrix G.**

## 2. Checks (instrument-grade, not the confirmation)

- k = 2: G has the single entry 8/3 → L₂ = 8/3 — reproduces the proven exp_04
  closed form exactly.
- k = 3: L₃ = 8√106/15 = 5.4910027… vs exp_05 measured 5.491004 (Δrel 2.3×10⁻⁷).
- k = 4: L₄ = √(Σ(sv²) + 2·Pf(G)) with Pf = ac + eb = 4096/175 → 11.1857376 vs
  measured 11.185742 (Δrel 3.9×10⁻⁷).
- Entrywise finite-m check (m = 200, 800, 2000, k = 4): |m·skew(M)| matches
  |G| entry-for-entry (48/7, 24/5, 8/3, 16/15) **up to diagonal ±1 gauge**
  (the numerical sign-fixing picks a different frame gauge; sign patterns are
  conjugate by diag(±1), so all invariants — angles, svals, L_k — coincide).

## 3. Why K1 and K2 had to die

For k ≥ 3, L_k is a sum of singular values of a *rational* skew matrix —
an **algebraic, generically irrational number** (L₃ = 8√106/15). Both dead
candidates presumed the limit stays inside ℚ-patterned families (harmonic
sums, Fibonacci ratios). Only k = 2 is rational, because a 2×2 skew matrix has
a single singular value equal to |its entry|. The "nice formula" hunt was
structurally doomed; the object was a spectrum, not a series.

## 4. Structural connections (recorded, not scored)

- **ℤ₂/parity**: the generator's selection rule (opposite parity only) is the
  frame-level face of the parity structure; exp_05 K3's universal
  even-reflection telescoping (det H = +1 at every sampled m, k) is its global
  ledger. **SEC-local / PAC-global reading**: locally, edge transports violate
  orientation freely (det −1 reflections — locality unbounded by the
  conservation law); globally the loop always reconciles to det H = +1 — the
  ℤ₂ ledger balances. Locality is a component; globality is the whole.
- **Ξ connection** (from `arithmetic/PACEngine/modules/pac_sec_unification.py:30-44`):
  Ξ = 1 + π/55 is there derived as a ratio of momentum-operator spectra —
  periodic circle (eigenvalues n²) vs anti-periodic Möbius ((n+½)²). The
  balance constant and this connection generator are **the same operator under
  two boundary twists**; the anti-periodic twist is a ℤ₂ holonomy. This makes
  the M15 Phase-2 twist classification and the Ξ story one subject. Registered
  as a direction, not a claim.

## 5. What would falsify the derivation

exp_06 (pre-registered, committed with this journal): the k = 5, 6 limits must
match the singular-value predictions to 0.1%. Failure kills the momentum
identification regardless of how pretty §1 looks.
