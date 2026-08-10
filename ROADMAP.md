# Roadmap

What is being worked, what is open, and what would falsify it.

Status of every experiment is generated in
[`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md). This file is the *direction* —
the questions currently open and what answering them would take.

---

## On deck

### Milestone 15 Phase 2 — is holonomy dynamical?

Phase 1 closed 2026-06-12. The affine holonomy is proven in closed form: θ(m) = m·θ_T(m),
**C₆ = −I as a theorem** (θ(6) = π exactly), cos θ(C₄) = −7/9 from first principles, and
the large-rank limit exactly 8/3 — which killed the `e` candidate by derivation before
measurement could flatter it.

Then 2026-07-17 solved the general-k limit: **the connection generator is the
particle-in-a-box momentum operator**, G[j′,j] = 4jj′/(j′²−j²) for j′−j odd. Registered
predictions for L₅ and L₆ were confirmed at 6.7×10⁻⁷ and 1.1×10⁻⁶. Both naive candidates —
odd-harmonic and Fibonacci — died first, by registered measurement.

Phase 2 asks the only question that matters for whether this is physics:

- **(a) ℤ₂ twist classification** across cycle structures, now that the parity mechanism
  is identified.
- **(b) The field-equation hunt** — does cascade ledger density *source* holonomy, and is
  the coupling φ-structured? Candidate substrate: the ADE-coupled cascade.

> **Standing kill-sentence.** *If holonomy is dynamically inert, it is mathematics, not
> physics, and M15 caps at a reclassification.* Phase 2 exists to answer exactly this.

**Nearest registrable target:** Ξ = 1 + π/55 derived as a ratio of momentum-operator
spectra — periodic circle (n²) against anti-periodic Möbius ((n+½)²). If that holds, the
balance constant and the M15 connection generator are the same operator under two boundary
twists, and the twist classification and the Ξ story become one result.

### Milestone R — propagate the energy-scale fix

60/112 across 27 experiments. The decisive result is exp_24: replacing
`E_Planck · φ^(−d)` with **`α(d)² · m_mediator`** puts the EM scale at 11.4 ppm of the
Rydberg and the nuclear scale within 1.75×. That single change resolves **eight** earlier
failures which all shared one root cause.

Open: propagate it back through exp_03–09, which were scored against the old scale and
are currently recorded as failures for a reason now known to be wrong.

Also standing from Block C: the Geiger–Nuttall law shown to be a **universality theorem**
for any d-simultaneous-threshold barrier (exp_16, 4/4), with universal exponent
k = 1.16 ± 0.02 across A, D and E families.

### Midnight — observational contact

22/32. The invariant-registration rule was adopted here and now governs the whole corpus:
*registered relations survive, registered coordinates die.*

---

## Known open ends

Carried honestly rather than quietly. Each is a soft spot in otherwise settled work.

| Where | What's open |
|---|---|
| M5 | CP violation at 3% error — the weakest number in the Standard Model block |
| M6 | exp_03 T2 at R² = 0.67 against a 0.75 threshold — genuine scatter in geometric decay |
| M9 | 8.9% slope gap (3 data points, may be noise); DESI w(z) tension, w_a = −0.15 vs −0.75 |
| M13.5 | Coherence limit is **not** universal (exp_15, 0/4) — geometric, not Fibonacci-arithmetic |
| M13.5 | PSD degeneracy proven **fundamental** (exp_16, 0/4) — no isomorphism-invariant metric can fix it |
| M10 | φ^(−1/N) converges to φ, but N = 8 is still 3.3% off — finite-size correction underived |

The M13.5 entries are not defects to repair. M15 reclassifies them: class-level content
passes, representative-level demands fail, and that split *is* the DFT-Hodge conjecture.

---

## Repository work

Phase 0 — reorganizing the repository by layer — is largely done. What remains:

- **`formal/theorems/`** — index the proven results (C₆ = −I, the holonomy closed form,
  PAC as exact eigenvector fixity at 2.4e-15, the origin of Ξ) into the journals that
  established them.
- **`THEORY_MAP.md` as the spine** — every claim resolving across all four layers:
  claimed → derived → tested → published.
- **`INVENTORY.md`** — generated lifecycle view: on deck / settled / legacy.
- **Lexicon lineage** — each term carrying the era it was coined in and what it replaced,
  so the pre-PAC vocabulary reads as etymology rather than clutter.
- **Lore `source_paths`** — migrate for the August 2026 relocations.
- **Link rot** — 185 dead links, only a handful mechanically fixable; per-link archaeology
  in mostly archived documents. Low priority, recorded rather than hidden.

---

## How direction is set here

Claims are pre-registered before they are tested ([`STANDARDS.md`](STANDARDS.md) §2.7).
Thresholds are fixed before the run and never relaxed afterward. A test that passes for a
reason unrelated to what it guards is tautological and gets replaced, which is why
hardening cycles sometimes *lower* a score — M11 went 52 → 49 → 52, and that was the
process working.

Failures are kept. `theory/corrections.md` is the standing record of what was claimed too
strongly and later withdrawn.

Superseded roadmaps are in [`roadmaps/`](roadmaps/) — kept because the gap between what
was predicted and what happened is itself evidence. The April 2026 roadmap forecast
"M12 = Topology Change"; M12 became Connection as Primitive.
