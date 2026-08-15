# Milestone 16: Relational Locality — How Neighbours Come to Cohere

**Status**: active (re-founded 2026-08-14)
**Founded**: 2026-08-13
**Origin**: An empirical dead end in `reality-engine`, reached from the other side.

> Re-founded one day after founding. The empirical fact survived re-measurement; the proposed
> mechanism did not survive corpus reading. See `journals/2026-08-14_refounding.md` — the
> founding journal stands as written, and is lineage.

---

## The empirical fact that opened this

The reality engine evolves three fields (E, I, M) on a Möbius manifold under sixteen
operators. It **does** build a large-scale component — and that component has **no web
geometry whatsoever**.

Measured on the manifold's own 128 × 32 grid, 3000 ticks, five seeds per configuration,
against a white-noise sampling distribution measured on the same grid (n = 40):

| configuration | ξ_u | coherent power fraction | excess |
|---|---|---|---|
| white noise (control) | 0.6243 ± 0.0100 | 0.0722 ± 0.0059 | — |
| RBF, qp = 0.02 (default) | 0.6453 ± 0.0091 | 0.0991 ± 0.0031 | +4.6σ |
| PAC balance, qp = 0.00 | 0.7778 ± 0.0147 | **0.1457 ± 0.0105** | **+12.5σ** |

Twice the white-noise share of low-k power, still growing at t = 3000. That is a real signal
and quantum pressure controls it.

But against the corpus's own web criterion (exp_09: filaments > 0.05, voids > 0.3, CV > 1.0):

| | void fraction | density CV | is_web |
|---|---|---|---|
| **exp_09 particle web** | **0.50** | **~2.0** | **yes** |
| white noise \|N(0,1)\| | 0.053 | 0.741 | no |
| PAC, qp = 0.02 | 0.241 | 0.661 | no |
| PAC, qp = 0.00 | 0.081 | 0.487 | no |

**The engine's density contrast is below white noise's.** It fails `is_web` in every
configuration. Rendering the isolated low-k component confirms the shape of the failure: a
field of random blobs, morphologically indistinguishable from low-passed white noise. No
filaments, no voids, no junctions.

And the two properties move *apart*: qp = 0.02 gives the highest void fraction, qp = 0.00 the
highest large-scale power. Nothing in the engine produces both at once.

> **One apparent exception, and it is a metric fault.** At qp = 0.30 the engine scores
> `is_web = True` — void 0.605, CV 1.273, passing all three of exp_09's conditions. Rendered,
> it is a **checkerboard**: C(1) = −0.358, anti-correlated neighbours at single-cell scale.
> exp_09's criterion is purely statistical, with no scale or connectivity requirement, and a
> lattice checkerboard satisfies it trivially — a failure mode that cannot arise in a particle
> substrate. `web_metrics` now requires ξ above the white-noise floor, with the checkerboard
> as a permanent selftest case.

**The engine clumps. It does not web.** That is the fact this milestone exists to explain, and
it is a sharper one than "no structure" — which is what the founding measurement claimed,
using an estimator that could not have detected the signal above. See
`journals/2026-08-14_refounding.md`.

## The claim

> **Spatial structure is the boundary set of a tiling of locally-conserving patches. A system
> with a single global ledger cannot produce it at any coupling strength; the correlation
> length is set by the patch scale, and the coordination cost of the tiling is Ξ.**

This is not a new proposal. It is what the corpus already says, assembled:

| source | statement |
|---|---|
| **exp_36** local-global tiling (8/8, zero free parameters) | Part D, verbatim: *"Cosmic web = visible tiling pattern: voids = patch interiors, filaments = boundaries, nodes = multi-boundary junctions."* Local PAC is exact **within** a patch; the observed residual is the SEC cost of coordinating patches, and that cost is **Ξ**. |
| **`asymmetric_conservation`** (5/5) | `P + A + Δ = C`. Δ is the **unreconciled boundary buffer**, cleared at reconciliation boundaries, over a parent/child hierarchy. `async_pac.py` instantiates `ReconciliationBoundary(delta_threshold=XI)` — Ξ is the reconciliation threshold in running code. |
| **exp_09 – exp_12** cosmic web | A web already emerged: 5000 particles, finite-range gravity `exp(−r/r₀)/r`, SEC entropy pressure. Void 50%, filament 12%, clustering 0.54, P(k) slope −1.73 — 85% match to the observed matter spectrum, with **no** 1/r². |
| **exp_10** SEC sweep | **No discrete transition** at Ξ. Ξ is the *optimal operating point* for structural complexity; SEC is continuous control. |

**The engine has exactly one patch.** `src/v3/operators/normalization.py` takes one global sum
over the whole lattice and adds a single uniform scalar to every cell. No region conserves
locally, no boundary exists, no region carries its own Δ, and no reconciliation event ever
fires.

This predicts precisely the failure that was measured. **A web is a boundary set; clumping is
what a field does without boundaries.** A one-patch universe can accumulate large-scale power —
and does, +12.5σ of it — but it has no interior/exterior distinction to evacuate voids or
concentrate filaments. Hence excess low-k power with contrast *below* white noise, which is
otherwise a strange combination.

### Relational identity is the reading, not the mechanism

The framework holds identity to be relational — M13's **identity IS complement** (4/4, across
A₅/D₄/E₆), `confluent-identity`'s "identity of any PAC node is the weighted confluence of its
children", M14's states on the **orbit quotient**. That survives here, and it is what the
tiling picture *means*: a patch's identity is its boundary with the rest of the manifold, which
is identity-as-complement at region scale.

What changed is the level at which it is testable. "Give cells relational identity" names no
mechanism and no length scale. "Give regions their own ledger, let Δ accumulate at their
boundaries, reconcile at Ξ" names both, and the machinery already exists and is validated.

## What this milestone must establish

**Q1 — Does patch-local conservation produce web geometry?** Not correlation — that already
exists. Partition the manifold into patches that conserve locally, accumulate cross-boundary
flux into Δ, reconcile at Ξ, and measure whether **voids and contrast** appear: does the field
cross exp_09's threshold of void > 0.3 and CV > 1.0, from a control at 0.24 and 0.66?

**Q2 — Is there a threshold?** exp_10 supplies a prior, and it is a *negative* one: expect a
continuous optimum near Ξ and **no** discrete transition. A transition would contradict exp_10
in a different substrate, which is worth knowing either way. M10's laws-as-equilibria
(exp_05–07) is where a genuine threshold would live.

**Q3 — What sets the correlation length?** The candidate is now named: the patch scale. If ξ
tracks the patch scale, that is the answer. If ξ is constant in cells while the patch scale
varies, the patch is not the mechanism.

**Q4 — Does coherence survive the continuum?** M13.5 proved PSD degeneracy fundamental and M15
reclassified the class/representative split. A tiling is a class-level object; whether it
survives being given metric representatives is the same boundary.

## What would falsify the claim

> **If patch-local conservation with boundary-Δ reconciliation yields ξ/L_patch
> indistinguishable from the single-ledger control across the registered patch-scale range,
> the tiling picture is not what produces spatial coherence in this substrate, and exp_36
> Part D does not carry from the cosmological argument to the engine.**

Recorded as a result either way. Two secondary failures also count: coherence that appears
only where the global ledger fails to close, and coherence that appears only in runs that
diverge. Patch-local conservation must **sum** to the global invariant — if it does not, PAC is
broken and the result is void regardless of what ξ does.

## Registered invariant, not coordinates

The registered quantity is **ξ / L_patch** — the ratio of correlation length to patch scale.
Dimensionless, independent of grid size and of absolute coupling. Not ξ in cells, which moves
with resolution.

## Method commitments

- **Pre-registration before any run** (STANDARDS §2.7): hypothesis, thresholds, and
  falsification condition committed first; outcomes committed separately.
- **Render before trusting a statistic.** A box-counting estimator returning D = 2.000 for a
  straight filament survived a full day of statistics computed on a field nobody had looked at.
- **Every metric ships a `selftest()` with an analytically derived expectation.** Not a tuned
  one. White noise gives ξ = 1 − 1/e exactly; a Gaussian of width σ gives ξ = 2σ; a cosine of
  wavelength λ gives ξ = arccos(1/e)·λ/2π. A metric that matches one value by luck cannot
  match the scaling.
- **Report against a control measured on the same grid**, never against an asserted floor.
- **Never report one isotropic number for an anisotropic manifold.** 128 × 32, one axis
  periodic and one bounded — a circular FFT along the bounded axis underestimates ξ by ~2×.

## Relationship to prior milestones

This does not re-derive exp_36, M13 or M14. It asks the **dynamical** question they leave open:
given that conservation is local and structure lives at the boundaries between locally-conserving
regions, does an extended system actually build that structure — and does it build the one the
particle model (exp_09–12) already produced?

M15's class/representative split is the likely boundary condition (Q4). M10's laws-as-equilibria
is the likely home for any threshold found (Q2).

## Why there is no hierarchy: the relation graph is nearly complete

M13 makes this a theorem rather than an intuition. Identity is the complement — a vertex's
identity is the structure of the graph without it. Remove any vertex from **K_n** and you get
**K_{n−1}**, identical for every vertex: same spectrum, same orbit, Aut(K_n) = S_n. **A complete
graph has exactly one identity.**

The engine's coupling is close to complete: gravity solves Poisson spectrally (every cell ←
every cell, one tick), normalization takes one global sum and returns an identical scalar to
every cell, and TimeEmergence, Temperature and Adaptive are global scalar reductions. The
laplacian is the only sparse structured term, and it damps as k² — weakest exactly where
structure would live.

So the engine has **two scales and no more**: the cell and the box. That is the same fact as
"excess low-k power with no voids and sub-noise contrast", seen from the other side.

Every relation is recomputed, globally and instantaneously, every tick. **No relation is ever
privileged by history, so nothing ever becomes anyone's parent** — and a hierarchy is precisely
a persistent asymmetric relation.

### The corpus's answer is an accumulating local refractory

`archive/era1-symbolic/legacy/brain.py` — QPL, Quantum Pressure Logic. A suppression field grows
multiplicatively where collapse fires (`QPL *= 1.05`, capped), and that accumulated record
subtracts from the drive at the same site (`val_info -= QPL * damping`). Cells that have been
active become harder to activate; activity moves; regions take turns. `brain.md`: *"QPL regions
specialize — structural learning emerges naturally."* Alongside it, `time[x,y,z] += 1` — a
**per-cell event counter**, depth laid down by history rather than imposed by geometry.

`exp_05_pac_tree_construction` supplies the other half: *"the hierarchy emerges from convergence
structure"* — strong mutual convergence makes siblings, weak convergence makes levels. Levels
are **inferred**, not partitioned.

`SpinStatisticsOperator` already samples neighbours at 1, 2, 4, 8 with φ-decay weights — the
*geometry* of a φ-spaced hierarchy is implemented and running. It recomputes from scratch every
tick, so nothing accumulates. **Geometry without history**, and history is what makes a relation
asymmetric.

## Experiments

| Exp | Name | Status |
|-----|------|--------|
| 01 | Accumulating local refractory (QPL) vs the single global ledger | pre-registered v2, not yet run |

v1 of the pre-registration (a fixed single-level φ-spaced tiling) is superseded and was never
run: one imposed partition is still one level, and at that level global and local remain the
same thing.

## Core machinery (reuse, do not reimplement)

- `experiments/studies/asymmetric_conservation/core/async_pac.py` — `AsyncPACNode` with
  `P/A/delta`, `receive_event`, `reconcile`, and `ReconciliationBoundary(delta_threshold=XI)`
- `experiments/studies/minimum_actualization_resolution/scripts/exp_36_local_global_tiling.py` —
  the tiling argument and its Ξ coordination cost
- `experiments/studies/gravity_from_maxwell_pac/scripts/exp_09_pac_web_emergence.py` — the
  web that already emerged, and the structural thresholds this milestone must beat
- `experiments/milestones/milestone13/core/identity_complement.py` — complement spectra
- `reality-engine/proof_of_concepts/v4/structure.py` — calibrated structure metrics
