# Roadmap

What is being worked, what is queued, and what would falsify it.

Three horizons. **Now** is in flight. **Next** is defined and unstarted — each item has a
stated question and a way to be wrong. **Horizon** is real direction that cannot be closed
from here, either because it waits on an instrument or because nobody has done the work of
turning it into a registrable claim yet.

Per-experiment status is generated in
[`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md). This file is the *direction*.

---

## Now

### M15 Phase 2 — is holonomy dynamical?

Phase 1 closed 2026-06-12. The affine holonomy is proven in closed form: θ(m) = m·θ_T(m),
**C₆ = −I as a theorem** (θ(6) = π exactly), cos θ(C₄) = −7/9 from first principles, and
the large-rank limit exactly 8/3 — which killed the `e` candidate by derivation before
measurement could flatter it.

Then 2026-07-17 solved the general-k limit: **the connection generator is the
particle-in-a-box momentum operator**, G[j′,j] = 4jj′/(j′²−j²) for j′−j odd. Registered
predictions for L₅ and L₆ confirmed at 6.7×10⁻⁷ and 1.1×10⁻⁶. Both naive candidates —
odd-harmonic and Fibonacci — died first, by registered measurement.

Phase 2 asks the only question that decides whether this is physics:

- **(a) ℤ₂ twist classification** across cycle structures, now that the parity mechanism
  is identified.
- **(b) The field-equation hunt** — does cascade ledger density *source* holonomy, and is
  the coupling φ-structured? Candidate substrate: the ADE-coupled cascade.

> **Standing kill-sentence.** *If holonomy is dynamically inert, it is mathematics, not
> physics, and M15 caps at a reclassification.* Phase 2 exists to answer exactly this.

**Nearest registrable target:** Ξ = 1 + π/55 derived as a ratio of momentum-operator
spectra — periodic circle (n²) against anti-periodic Möbius ((n+½)²). If that holds, the
balance constant and the M15 connection generator are the same operator under two boundary
twists, and the twist classification and the Ξ story collapse into one result.

### M18 — is φ structural, or projected?

Founded 2026-08-31 on the thesis that φ enters the corpus as a projection (the foldings
A₄ → H₂, D₆ → H₃, E₈ → H₄), not a magnitude. In three days the mathematics closed to a single
conjecture: parents of golden Coxeter diagrams are **branched double covers** whose sheets
decouple at γ² + γ − 1 = 0 (proven), every strict-fold law is a theorem on those covers
(proven), and *rigidity* — that every strict fold is such a cover — held 47/47 at n = 20
without proof. Seven theorems are indexed in [`formal/theorems/`](formal/theorems/README.md);
the open half is in [`formal/conjectures/m18_open.md`](formal/conjectures/m18_open.md).

The layers are now kept apart: **mathematics** (Phases 3–6, closed to rigidity), **physics**
(Blocks B and D, Phases 7–7c — the fold's branch is dynamically visible, exp_18 47/47), and
**instruments** (the exact census and certificate machinery in the milestone's `core/`).

> **Standing kill-sentence.** *If E₈-derived spectra do not split into two φ-scaled families
> (Block B), and the indefinite golden form adds no separating power at the orbit boundary
> (Block C), then φ is not structural in this corpus and this milestone dies.* Clause C met,
> clause B not: wounded, not dead. Block D can neither kill nor rescue it.

**Block D ran 2026-09-03** (sealed `0130ebe0`; `experiments/milestones/milestone18/journals/2026-09-03_exp09_outcomes.md`):
the fold *certificate* (tr(R·D) = 2/√5 on the Bezout reflection), evaluated on the corpus's own
generative operators — complete PAC trees to n = 511, 166 growth trees, M15's unicyclic controls —
found **0 carriers of 173**. The H₂- and H₃-type sectors are present in the PAC family; their fold
representative is not (M15's class-pass / representative-fail shape). The physics reach at measured
sizes is retired; the mathematics stands; P1's registered form is forward-corrected there.

**Nearest registrable target:** Phase 8's parity law at n = 24 (census running) and the exact
denominator of the reflection polynomial (three cases carry the odd-prime half).

### Milestone R — the trigger is a core detector; the energy scale is still in the wrong slot

65/120 across 29 experiments. **exp_28 (2026-09-05, 0/4, sealed `bf833113`)** ran the
derived severance trigger (exp_15's all-edges overstress) for the first time as a
*dynamical* step, on reality-engine's v4 particle substrate: `min_j |S_i − S_j| > τ` fires
only at extrema of the field, and on a structured field the extrema are the collapse cores
— severance removed the bound, connected part and left the rest hotter and less connected
than a random subset at the same count (T2 −2.6 σ at the onset τ; a null at n = 4000 where
half is removed; never above random). The exp_15/16 graph results
stand (their field was noise); what closes is the entropy-gradient barrier as an amount-free
*dynamical* sink. Registrable next: a barrier on *under*-stress, and the sign question it
raises beside exp_14.

**exp_29 (2026-09-06, 3/4, sealed `43e4ebc9`)** found where exp_28's zero came from — the substrate
had no PAC ledger, and the SEC pressure's entropy ratchet created eleven binding energies from
nothing — and added the ledger: a per-particle potential budget paying for entropy growth at
the pair-energy price, the total `KE + U + E_SEC + ΣP` conserved. The substrate is then bound
(KE/|U| ≈ 0.5) and a web survives at both sizes, 4.8–5.7 σ over the unbounded engine — but the
registered κ = 0.5 arm fails the 2 σ bar against gravity alone (0.2 σ on the proxy, 1.1 σ at
n = 4000), and the kill fired as registered: the mapping is retired as the object that holds
structure beyond gravity at κ = 0.5 with the proxy deciding; the ledger stays as an instrument.
Unscored, post hoc: at n = 4000, κ = 1 holds more web than gravity alone in every seed (2.8 σ)
and κ = 2 less; on the proxy, whose pressure range exceeds half its box, the engine never adds.
**Registrable next: R1b** — κ = 1 at n = 4000 on fresh seeds, the box-to-range ratio declared,
the proxy retired as a decider for pressure-range questions; then R2, severance on a substrate
that is now actually bound; the pressure's *form* stays a candidate behind R1b.

exp_24 replaced `E_Planck · φ^(−d)` with **`α(d)² · m_mediator`** (EM scale at 11.4 ppm of
the Rydberg). The 2026-08-27 propagation showed it fixes **two** of its six named failures,
not eight, and that the (depth, mediator) pair is exactly degenerate — a fitted depth
measures the mediator choice. Open: the energy scale needs a representation that carries
φ⁻¹ once and a Fibonacci index multiset, which `α(d)² · m` does not; several remaining
passes are tautological at the Planck scale and may fail once the scale is right.

Also standing from Block C: Geiger–Nuttall shown to be a **universality theorem** for any
d-simultaneous-threshold barrier (exp_16, 4/4), universal exponent k = 1.16 ± 0.02 across
A, D and E families.

### Repository — closing Phase 0

The layer reorganization is done. Remaining:

- **Publication repackaging** — 11 packages marked `needs_repackage` against 7 current,
  2 incomplete, 1 ready. The largest single block of stale state in the repo.
- **Link rot** — 201 unresolved relative links, mostly per-link archaeology inside archived
  documents whose targets moved or were never committed. Recorded rather than hidden, and
  now measured by `tools/check_links.py` rather than hand-counted; CI pins the count as a
  ceiling so the number can fall but not rise. Low priority to reduce, high priority not to
  add to.
- **`UNIFIED_EVIDENCE.md`** — 20 copies in five drifted versions across published packages.
  Frozen artifacts, so the fix is a forward note, not an edit.

---

## Next

Defined, not started. Each carries the question that would close it.

### The orbit-flow direction — announced, renumbered past, never done

Milestone 14's Forward Path announced:

> **Dynamics as Orbit Flow**: Schrodinger equation from SEC-driven orbit flow, Hamiltonian
> from graph Laplacian restricted to orbit space, time evolution as
> automorphism-equivariant unitary propagation.

M15 then became The Representative Problem instead, and M15's README records the slot as
"fulfilled by P13–P16 under Milestone 14." **The two documents disagree**, and P13–P16 are
propositions with a conjectured propagator — not a derived Schrödinger equation.

Treated here as **open**. It is the one announced direction in the milestone stack that
was superseded by renumbering rather than by a result. Either it gets derived, or the
claim that M14 already covered it gets withdrawn into
[`theory/corrections.md`](theory/corrections.md). Both outcomes are acceptable; the
current state — announced, assumed done, not done — is not.

### M10 — the finite-size correction

φ^(−1/N) converges to φ, but N = 8 is still 3.3% off and the correction is underived.
Closes when the correction is derived rather than fitted, or when the convergence is shown
to be asymptotic-only and the N = 8 gap is accepted as structural.

### M9 — the 8.9% slope gap

Three data points, may be noise. Closes by adding points, not by argument. Until then it
stays on this list rather than being quietly absorbed into the M9 score.

### M5 — CP violation at 3%

The weakest number in the Standard Model block, and the only one whose error is large
enough that a competing derivation could plausibly beat it.

---

## Horizon

Real direction that cannot be closed from inside the repository.

### Observational contact

Roughly 60% of M11's tests are structural — internal consistency, not empirical
validation. The hard tests wait on instruments:

| Prediction | Waits on |
|---|---|
| Z′ at 395 GeV (M1) | collider reach |
| Cosmological-constant and structure-growth predictions (M8, M9) | Euclid |
| Gravitational-wave signatures (M11) | LISA |
| High-energy astrophysical tests (M11) | CTA |
| DESI w(z) tension — w_a = −0.15 predicted vs −0.75 observed | further DESI data releases |

The DESI line is the one that could go badly. It is carried as an open tension rather than
explained away, and if the observed value holds it is a falsification, not a discrepancy.

**Midnight** (22/32) is the sidecar where observational contact is worked. It is also
where the invariant-registration rule was adopted — *registered relations survive,
registered coordinates die* — which now governs the whole corpus.

### Deferred from M11

Spacetime topology change, multi-loop graviton corrections, graviton scattering
amplitudes, and the LQG/spin-foam connection. Each was scoped and set down deliberately;
none has a registrable formulation yet. Listing them here is the honest alternative to
letting them disappear from the record.

---

## Known open ends

Carried honestly. Each is a soft spot in otherwise settled work.

| Where | What's open |
|---|---|
| M5 | CP violation at 3% error |
| M6 | exp_03 T2 at R² = 0.67 against a 0.75 threshold — genuine scatter in geometric decay |
| M9 | 8.9% slope gap; DESI w(z) tension |
| M10 | φ^(−1/N) at N = 8 still 3.3% off — finite-size correction underived |
| M13.5 | Coherence limit is **not** universal (exp_15, 0/4) — geometric, not Fibonacci-arithmetic |
| M13.5 | PSD degeneracy proven **fundamental** (exp_16, 0/4) — no isomorphism-invariant metric can fix it |
| M18 | Rigidity unproved (47/47 at n = 20); a third strict species (asymmetric) unidentified; Block D reach retired at measured sizes (0/173, 2026-09-03); the clean regime's third kernel class; the branch profile's null; the exact denominator of the reflection polynomial (polynomial integrality killed 2026-09-02 at n = 20 — three parents with den(5·b) = 3; the bound by the diagram resultant is proved, equality open) — [`formal/conjectures/m18_open.md`](formal/conjectures/m18_open.md) |

The M13.5 entries are not defects to repair. M15 reclassifies them: class-level content
passes, representative-level demands fail, and that split *is* the DFT-Hodge conjecture.

---

## How direction is set here

Claims are pre-registered before they are tested ([`STANDARDS.md`](STANDARDS.md) §2.7).
Thresholds are fixed before the run and never relaxed afterward. A test that passes for a
reason unrelated to what it guards is tautological and gets replaced, which is why
hardening cycles sometimes *lower* a score — M11 went 52 → 49 → 52, and that was the
process working.

Claims register **invariants, never absolute coordinates**.

Failures are kept. [`theory/corrections.md`](theory/corrections.md) is the standing record
of what was claimed too strongly and later withdrawn.

Superseded roadmaps are in [`roadmaps/`](roadmaps/) — kept because the gap
between what was predicted and what happened is itself evidence. The April 2026 roadmap
forecast "M12 = Topology Change"; M12 became Connection as Primitive. The orbit-flow item
above is on this roadmap precisely so it does not become the next such gap.
