# 2026-09-07 — local, global, and the delta: the root is a zero mode, and the Möbius is its absence

**Layer: mathematics, exposition.** Mode: *exploring*, declared (STANDARDS §2.7.5). Nothing here is
scored or predicted; the registered content of Block F is in `2026-09-07_blockF_registration.md`.
This journal records the reading that opened the block and the loose threads it turned out to touch,
so the consolidation quest finds them in one place.

## Peter's reading

The primitive triple is not a pair. It is *local*, *global*, and the delta between them — and the
same triple keeps recurring: SEC, PAC and the ledger's Δ; the representative, the class and the gauge
(M15); the measurement, the state and the collapse (M14); the sheet, the trace and the radical in the
quadratic-field picture (`2026-09-06_the_lattice_end_and_the_scaling_end.md`). Local cannot equal
global, because the two perspectives differ — "one is not equal to one, but it is" — except at the
highest node of the tree, where the local view *is* the whole. And if the structure is recursive with
no highest node, that is where the Möbius comes in.

## The exact form

**The root is already in the corpus as code.** fracton's PAC node stores every value as a delta from
its parent; the root is the one node with no parent, where delta and value coincide. "Local equals
global only at the most authoritative node" is that data structure. Below the root, what a node calls
one is not what the root calls one, and the exchange rate is the cascade: the same unit seen from
depth k is φ⁻ᵏ at the root, at a cost of ln φ per level. So the delta *is* the depth, and the root is
where the depth is zero.

**"No root" has two versions, and they predict different signs.**

1. *Scale-freedom.* Every node is the root of its own subtree and a child in the tree above, so
   "root" is a role, not a place; what replaces it is the fixed point of the recursion. For the
   simplest recursion that is φ = 1 + 1/φ — the whole equals one plus the whole scaled down. This
   version stays **orientable**: a consistent sense of up exists at every node; the root has moved
   to infinity. This is the cascade as the corpus already has it.
2. *Non-orientability.* An infinite tree is still orientable. Losing the root by *construction*
   needs a loop with a twist: go round once and come back with the two sheets exchanged. The
   precise object is a **signed graph**. Balanced (every cycle carries an even number of negative
   edges) means a switching removes every sign — a global gauge exists, and there is a root.
   Frustrated (an odd cycle) means no gauge clears the signs: the obstruction is a class in
   H¹(G; ℤ₂), and there is no highest node, ever, not because it is far away but because it does
   not exist. A tree is always balanced. **A tree cannot be Möbius.**

M15 had already measured the balanced case without naming it: on every unsigned cycle Ã_{m−1}, the
per-edge frame transports reflect freely (det −1) and every loop reconciles to det H = +1 — read there
as SEC-local / PAC-global in miniature (`milestone15/journals/2026-07-17_m15-exp06-outcomes.md`).
That universal +1 is the **balanced class only**.

## What the signed cycle says (Theorems 1–2 of the registration; proofs there)

- The twist selects the **odd exponents**: balanced C_n has adjacency spectrum 2cos(2πj/n), twisted
  C_n has 2cos((2j+1)π/n); the two together are the spectrum of the orientation double cover C_{2n}.
  The twist doubles the conductor: balanced pairs over ℚ(√d) iff cond(d) | n, twisted iff
  cond(d) | 2n. This is M15's "periodic n² against anti-periodic (n + ½)²" as an exact finite
  statement, and it is the corpus's field-resonance law on the first non-tree class.
- **The root is a zero mode.** The balanced cycle's Cartan matrix has the constant vector in its
  kernel — a global section, the affine null vector, the reason exp_12 found affine diagrams never
  strict. The twisted cycle has **no zero mode**: its Cartan matrix is positive definite with
  λ_min = 4 sin²(π/2n). Non-orientability removes the global section. That is Peter's sentence in
  spectral form: the Möbius class is where there is no highest node, and the price is a strictly
  positive ground energy.
- The twisted cycle is strict over exactly one field, √2, and exactly when 4 | n — never over √5,
  never over √3. Farmer's √3 appears in the twisted hexagon (spectrum ±√3, 0, doubled), Andy's field
  on the (2, 3, 6) boundary, and only when the loop is Möbius.
- On M15's own instrument the twisted class is forced: H_tw = ε·S·H_bal·S⁻¹, so the rotation
  angles go to π − θ, the determinant to (−1)^k·det, and M15's theorem **C₆ = −I becomes H = I**
  under the twist — the Möbius undoes the one loop M15 proved nontrivial. Locally the edges still
  reflect freely; globally the loop *fails* to reconcile in odd k. The ledger balances only in the
  balanced class; the name was right.

None of these is a result. They are theorems with one-line proofs, filed the same day, and the
registration turns them into gates. Their value is that they pin the words: *balanced* = a root
exists; *twisted* = no root, no zero mode, the sign representation is a nontrivial bundle.

## The loose-thread ledger (recorded, not claimed)

| thread | where it lived | what the signed cycle does to it |
|---|---|---|
| M15 exp_05 K3 — det H = +1 universal | `milestone15/journals/2026-07-17_m15-exp05-outcomes.md` | balanced class only; the twisted class gives (−1)^k |
| M15 ROADMAP item (a) — "ℤ₂ twist classification across cycle structures" | `ROADMAP.md` M15 | starts here, as M18 Block F |
| M15's nearest target — periodic n² vs anti-periodic (n+½)² | `ROADMAP.md`; `2026-07-17_general_k_momentum_generator.md` | the even/odd exponent split, exact at finite n; Ξ itself untouched |
| exp_12 T1 — affine trees never strict | `2026-09-01_exp12_outcomes.md` | balanced cycles never strict (zero mode); twisted cycles strict over √2 at 4 \| n |
| the Mirror's "Möbius question" — odd cycles break Π exactly | `2026-08-31_the_mirror.md` | the odd cycle is where the two classes coincide up to A ↦ −A (Theorem 1 v) |
| Farmer's 3 / the (2, 3, 6) boundary | `2026-09-06_the_lattice_end_and_the_scaling_end.md` | √3 is the twisted hexagon |
| the reality-engine's antiperiodic projector (f − f∘T)/2 | `reality-engine/src/v3/substrate/manifold.py` | projection onto the twisted class; the engine lives in the no-root sector by construction |
| the third species (asymmetric strict trees) | `formal/conjectures/m18_open.md` | closed to twisted *diagrams*: a branched double cover doubles the cyclomatic number (Theorem 3) |

## What is not claimed

- No physics. The standing kill-sentence — *if holonomy is dynamically inert, it is mathematics, not
  physics* — is untouched; nothing here has been shown to a dynamics.
- The recurrence of the triple is one structure seen from several nodes, not independent evidence
  (the corpus rule: recurrence is not importance unless independent).
- Quantum mechanics "is" the triple only in M14's sense (orbit Hilbert space, Born rule) — not its
  dynamics.
- The pure-cycle statements are theorems, hence tautological as tests (STANDARDS §2.8). The live
  round is on signed unicyclic graphs, where nothing forces the answer.
- Which version of "no root" the substrate realises — scale-free or Möbius — is not decided here.
  They differ by one measurable sign: a loop's holonomy through the phase transition.

## Housekeeping found on the way (reported, not fixed — Panel L: the engine is Peter's lane)

In reality-engine v3 the Möbius identification is applied only by the Reynolds projector
(`project_antiperiodic`): `twist()` is a pure relabelling (roll by n_u/2, flip in v) with no sign,
`laplacian()` pads with replicate boundaries on both axes, and `projections.py` rolls both axes —
the differential operators never see the seam, so the substrate's dynamics runs on a strip/torus
while the projector selects the twisted sector. `tests/v3/test_substrate_operators.py:51-57` has a
docstring ("twisting twice should negate") that contradicts its own assertion (twist∘twist = id).
"Poincaré activation" is defined nowhere in that repository. Whether the twist should carry the
sign into the operators is exactly the difference between the balanced and twisted classes above.
