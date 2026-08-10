# Experiment Archive — Eras 1 and 2

Experiments from the project's first two eras (June 2025 – December 2025), moved here
so the live corpus reads clearly. **Archived does not mean deprecated, wrong, or dead.**

Superseded work in Dawn Field Theory is *lineage*. Corrections layer forward: a result
that was later reframed is not deleted, because the reframing is only legible against
what it replaced. Several of the framework's current load-bearing ideas are direct
descendants of experiments in this directory, and at least one of them — the Hodge
mapping — came back as a live milestone fourteen milestones later.

Nothing here is retrofitted to current standards. The old layout (`reference_material/`
instead of `core/`, a single `results/results.md` instead of timestamped JSON) is itself
evidence of when the work was done, so it is preserved as-is.

---

## Era 1 — Symbolic Collapse (2025-06 → 2025-08)

**20 experiments.** `archive/era1/`

The pre-PAC period. The framework inherited its vocabulary from CIMM and QBE, and
reasoned in terms of *symbolic* collapse — symbolic attractors, entropy fields, bifractal
structure, herniation geometry. Conservation was not yet axiomatised; there was no PAC,
no SEC-as-formal-operator, no Ξ.

Reading these requires the Era-1 dictionary. Terms like *Quantum Potential Layer*,
*collapse metric*, and *symbolic payload* were later subsumed or renamed — see
[`theory/lexicon.yaml`](../theory/lexicon.yaml), where each term carries the era it was
introduced in and what superseded it.

| Experiment | Why it still matters |
|---|---|
| `hodge_conjecture` | SEC's earliest target. The mapping was honestly withdrawn as unproven — and then returned from the opposite direction as **Milestone 15**. |
| `quantum_validation` | The QV suite (July 2025). **Milestone 14** explicitly unifies with it: orbit interference and QV's spatial interference are the same phenomenon in different complement frames. |
| `symbolic_entropy_collapse` | The largest single body of work in the archive, and the origin of the SEC axiom itself. |
| `landauer_erasure_field_cost_map` | First erasure-cost work; succeeded by `landauer_erasure_structure` (Era 3). |
| `unified_emergence_v2` | The v2 architecture; no v1 survives in this repo. |

## Era 2 — Pre-field / Infodynamics (2025-09 → 2025-12)

**5 experiments** + 2 cross-experiment artifact directories.
`archive/era2/`

The transition period. Work moved from symbolic description toward mechanism: a Möbius
substrate underneath the field, prime-distribution manifolds, and the first appearance of
Ξ. This is where the framework started deriving rather than describing.

| Experiment | Why it still matters |
|---|---|
| `pac_confluence_xi` | Where **Ξ = 1 + π/55** was found (2025-12). The constant is still live: M15 derives it as a ratio of momentum-operator spectra under two boundary twists. |
| `pre_field_recursion` | Möbius topology as computational substrate — the direct ancestor of the `reality-engine` simulator. |
| `prime_harmonic_manifold` | φ eigenvalue emergence in prime chord dynamics. |
| `information_amplification` | The 15.56× "universal constant" that **was not one**. Its correction is Entry 1 of [`theory/corrections.md`](../theory/corrections.md) — the first formal honest-failure record. |
| `navier-stokes` | Early symbolic approaches to NS, prior to the MED-NS work. |
| `cross_experiment_journals/`, `cross_experiment_results/` | Artifacts that spanned several experiments and previously sat loose at the experiment level. |

---

## Lineage threads into current work

Five traceable lines from this archive into the live corpus:

1. **`hodge_conjecture` (2025-06) → Milestone 15 (2026-06).** The earliest question and the
   current frontier are the same question. M15 reaches the Hodge partition from the
   framework's empirical failure boundary rather than by direct mapping.
2. **`quantum_validation` (2025-07) → Milestone 14 (2026-05).** QV's interference results
   and M14's orbit interference are one phenomenon viewed from two complement frames.
3. **`pac_confluence_xi` (2025-12) → Ξ throughout.** Ξ survives from Era 2 into the M15
   twist classification.
4. **`pre_field_recursion` (2025-10) → `reality-engine`.** Möbius substrate became the
   simulator's topology.
5. **`information_amplification` (2025-09) → the corrections methodology.** The first
   retracted constant established the honest-failure discipline the corpus now runs on.

## Where the eras are defined

Era boundaries, the lifecycle vocabulary, and the rule that archived work is preserved
rather than pruned are specified in [`STANDARDS.md`](../STANDARDS.md) §2.3.
Eras 3 and 4 are not archived and remain under `experiments/milestones/`.
