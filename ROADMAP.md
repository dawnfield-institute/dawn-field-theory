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

### Milestone R — propagate the energy-scale fix

60/112 across 27 experiments. The decisive result is exp_24: replacing `E_Planck · φ^(−d)`
with **`α(d)² · m_mediator`** puts the EM scale at 11.4 ppm of the Rydberg and the nuclear
scale within 1.75×. That one change resolves **eight** earlier failures sharing a single
root cause.

Open: propagate it back through exp_03–09, currently scored against the old scale and
recorded as failures for a reason now known to be wrong. This is the highest-value
unfinished work in the corpus — it converts recorded failures into results without
weakening a threshold, because the threshold never changed; the scale was wrong.

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
