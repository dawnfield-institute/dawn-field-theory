# exp_04 — the loop's own depth profile (EXPLORING; no seal, no score)

**Date:** 2026-09-07 (evening) · **Layer:** arithmetic (a study; no physics; no φ, Ξ or Fibonacci enters).
**Mode: EXPLORING throughout. Nothing in this journal is registered and nothing is scored.** The round was
designed to end in a seal; the gates refused it (§5), and the reason is a property of the observable rather
than of the design. Scripts: `scripts/explore_r4_loop_depth_profile.py` (enumerated and sampled sweeps),
`scripts/exp_04_gates.py` (gates, run and recorded — G7 FAIL).

## §0 Why the round was opened

Round 3 derived the depth **shift** exactly (log y_eff / log y = 1/Buchstab ratio, four decimals, fifteen
cells) but its other constant is measured: β_q, from δ_q(y) ~ (log y)^{−β_q}, fitted per modulus off two
depths each — 0.85, 0.74, 0.83, 0.79, 0.78 for q = 3, 9, 4, 8, 5 — and recorded, never derived. Round 3's own
registration says it: "its constants (β_q, y_eff) come from round 2's objects". The question was whether
δ_q(y) has a closed form, which would remove the model's last fitted input.

Round 3's forward note named a second candidate — the depth-shift law at moduli with a prime factor above y.
**That candidate is vacuous and was not run.** δ_q is carried by how often q divides a gap; a modulus whose
prime factor is r > y divides a gap only if that gap is ≥ r > y; the loop's mean gap at depth y is
1/∏(1 − 1/p) ≈ e^γ log y, and e^γ log y < y for every y ≥ 3, with the margin widening. So such moduli are
saturated by construction at every depth (δ → 1, above round 3 §1's 0.9 cutoff) and the mechanism is never
exercised. It is the failure class of `asymmetric_conservation`'s Δ ≡ 0 — a relation that passes without
running — and it is recorded here so no one opens it again.

## §1 Instrument check against the record

The enumerated loop reproduces round 1's recorded exact q = 3 deficits: **5/12** (k = 3), **223/552** (k = 4),
**2860783/8291520** (k = 6). Semantics match exp_01 exactly — loop rooted at 0, `residues_mod(off, 0, q)`,
a linear transition matrix, exact rationals only where q | P. Gate G1.

## §2 An exact theorem: δ_{2q} = δ_q for odd q

Every unit is odd, so a unit's residue mod 2 is always 1; by CRT its residue mod 2q is determined by its
residue mod q. The transition matrix mod 2q is therefore the matrix mod q relabelled and padded with zero
rows and columns, and the diagonal deficit is invariant under that. Hence **δ_{2q} = δ_q exactly** for odd q,
and φ(2q) = φ(q) besides.

Verified 42/42 on every enumerated loop k = 3..7 for every odd q | P (gate G2).

This generalises round 3's forward correction. Round 3 filed "δ₁₀ ≡ δ₅ — every gap is even, so the transition
mod 10 is the transition mod 5" as a one-off; it is the q = 5 instance of a law. **Consequence for any future
round: q ∈ {6, 10, 14, 22, 26, 33·2, …} are never independent cells.** Scoring both q and 2q for odd q counts
one measurement twice. Gate G3 enforces this on any proposed modulus set.

## §3 A correction: δ_q → 0, not 1 − φ(q)/q

The round's own plan asserted that δ_q approaches a nonzero limit 1 − φ(q)/q, and argued from that the power
law could not hold globally. **That was wrong.** δ_3 passes straight through 1/3:

| k | y | δ₃ (exact) | |
|---|---|---|---|
| 6 | 13 | 2860783/8291520 = 0.34502516 | |
| 7 | 17 | 0.32460205 | below the claimed limit |
| 9 | 23 | 0.29450197 | and still falling |

The deficit is measured against a product-of-marginals null whose marginals sit on the φ(q) **unit** classes,
not all q classes. Independence therefore gives P(q | g) = 1/φ(q) and δ_q = 1 − φ(q)·(1/φ(q)) = **0**. The
measured P(q | g) climbs toward 1/φ(q) = 0.5, 0.25, 0.167 for q = 3, 5, 7 — not toward 1/q. δ_q is distance
from independence, with zero as the floor, and a decaying power law is entirely coherent.

## §4 The power law holds, over four decades

Sampled CRT-uniform loops, W = 40 windows of 2·10⁶, y = 60 … 141,422 (12 depths):

| q | 3 | 4 | 5 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| R² of log δ vs log log y | 0.99993 | 0.99855 | 0.99411 | 0.99877 | 0.99940 | 0.99989 |

Round 3 fitted β off two depths each; this is the same functional form on twelve, and it holds. No break, no
loss of monotonicity, no transition anywhere in the range — worth stating because a phase change between
order and disorder would have shown as exactly that, and there is none between y = 60 and y = 141,422.

## §5 What the round set out to register, and why it cannot be

Sorting the exponents by φ(q) rather than by q collapses them, and the pairs are structurally unalike:

| φ | q | β | |
|---|---|---|---|
| 2 | 3, 4 | 0.8114 ± 0.0117, 0.8231 ± 0.0159 | odd prime vs 2² — agree at 0.59σ |
| 4 | 5, 8 | 0.7971 ± 0.0241, 0.7881 ± 0.0080 | ℤ/4 cyclic vs Klein four — agree at 0.35σ |
| 6 | 7, 9 | 0.7041 ± 0.0204, 0.7389 ± 0.0114 | prime vs prime square — agree at 1.49σ |

One common β: χ² /dof = 8.55, rejected. β = f(φ(q)): χ² /dof = 0.90, consistent. The φ = 4 pair is the
striking one — same group order, different group structure, same exponent.

**It does not survive its control. The collapse is the δ-level in disguise.**

The decisive test holds φ **exactly** fixed and varies δ, which is possible because φ(q) is non-monotone in q:
within one φ class the ratio φ/q ranges widely (q = 17 has φ/q = 0.94, q = 60 has 0.27, both φ = 16), and δ
follows φ/q. If β were φ-indexed it would be flat across each class. It is flat in none of them:

| φ | moduli (by δ) | β | δ span | flat-β χ²/dof | corr(δ, β) |
|---|---|---|---|---|---|
| 8 | 16, 24, 20, 15 | 0.639 … 0.759 | 0.456–0.554 | 22.5 | +0.62 |
| 12 | 28, 13, 36, 21 | 0.531 … 0.611 | 0.616–0.725 | 10.9 | +0.29 |
| 16 | 32, 17, 40, 48, 60 | 0.510 … 0.389 | 0.710–0.854 | 63.6 | −0.74 |
| 20 | 44, 25, 33 | 0.389 … 0.310 | 0.810–0.892 | 51.7 | −0.98 |
| 24 | 52, 56, 35, 39, 45 | 0.332 … 0.142 | 0.857–0.962 | 332.6 | −0.99 |

At φ = 24, β runs 0.332 → 0.142 across five moduli of identical totient. **β is not a function of φ(q) alone.**

The Phase A collapse arose because in that six-modulus anchor set φ and δ were collinear at 0.995 — the
correlation was read off the wrong variable.

**But δ is not the whole story either.** The mirror test — matched δ, different modulus — separates too:

| | q | φ | δ | β |
|---|---|---|---|---|
| | 11 | 10 | 0.5413 | 0.5311 ± 0.0117 |
| | 15 | 8 | 0.5540 | 0.7591 ± 0.0107 |

δ matched to 0.013, β apart by 0.228 — about 14σ, where the δ mismatch accounts for ~0.006 on the pooled
β(δ) slope. Pooled over all 44 moduli β falls monotonically with δ (corr −0.84 below δ = 0.65, −0.97 above,
both significant; no sign change anywhere). Regression: δ alone adj-R² 0.881, δ + log φ 0.906.

**Honest state: β is dominated by the δ-level, with residual modulus structure that δ does not absorb, and
the two are collinear at 0.977 so neither is cleanly isolable in this object.** What is settled is the
negative: the Phase A reading — β indexed by φ(q) — is false, because β varies strongly at fixed φ.

**What this means for the profile.** δ_q(y) is not a power law with a modulus-dependent exponent. The local
log-log slope is governed by how far the cell sits from saturation, which is to say the curve bends as it
approaches its bound at 1. Round 3's five "β_q" are local slopes read at whatever δ each modulus happened to
occupy, not five constants of the moduli.

**Why no seal, additionally:**

- corr(φ, δ at fixed depth) = **+0.9954** on the anchor set; +0.9773 over 44 moduli.
- Regression over 44 moduli: log φ alone adj-R² = 0.9068; δ alone 0.8805; **both together 0.9061 — no gain
  from adding δ to φ**, the signature of two collinear predictors rather than one being correct.
- δ-level demonstrably biases β: restricting to δ < 0.5 moved q = 5 from 0.666 to 0.734 and q = 7 from 0.635
  to 0.698.
- **G7 FAIL** (`results/exp_04_gates_20260907_182007.json`): the held-out classes never clear saturation at
  reachable depth. At y = 141,422 only q = 16 (0.456) and q = 24 (0.483) fall below 0.5; q = 13 is 0.624,
  q = 17 is 0.715, q = 48 is 0.779. Bringing q = 21 to a matched δ = 0.40 needs y ~ 10^11.4.
- A matched-δ sweep over q = 3..60 (2·odd excluded by §2) finds pairs only at δ > 0.85 — inside saturation,
  where β is compressed to 0.03–0.39 and cannot be read.

The obstruction is structural, not computational: δ at fixed depth rises monotonically with q, and φ rises
with q, so matching δ forces matching q and hence matching φ — everywhere except the saturated tail where β
is unmeasurable. **No experiment in this object separates "β is φ-indexed" from "β is a shadow of the
δ-level."** Deeper sieving does not help; it needs every prime ≤ y, and y ~ 10⁹ is ~5·10⁷ primes per window.

## §6 What is not claimed

No closed form for β_q. No statement that β is or is not φ-indexed — only that this object cannot decide it.
Nothing about the primes: all of this is the loop's own profile, and whether the primes inherit any of it is
untouched. Nothing above y = 141,422. No physics; nothing here touches φ, Ξ or any milestone.

## §7 Forward note

Round 3's scored result is unaffected — ρ_q = 0 used the correct β per cell whatever β means. But **its
reading of β_q as modulus-dependent carries this confound**, and a future round should not treat the five
recorded exponents as five independent constants without addressing it.

Worth a registration if an object ever offers the leverage: a family where the deficit level and the unit-group
order can be varied independently. Not this one. What would be reachable here is the shape of the profile at
fixed q under a different sieve support — changing which primes are sieved rather than how many.

**Score: unchanged at 5/9.** Nothing here was registered, so nothing here scores.
