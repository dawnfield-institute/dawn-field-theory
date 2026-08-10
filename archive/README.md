# Archive — Lineage

Work from the project's earlier eras. **Archived does not mean deprecated, wrong, or dead.**

Superseded work in Dawn Field Theory is *lineage*. Corrections layer forward: a result that
was later reframed is only legible against what it replaced. Several of the framework's
current load-bearing ideas descend directly from this directory, and one of them — the
Hodge mapping — came back as a live milestone fourteen milestones later.

Nothing here is retrofitted to current standards. The old layouts (`reference_material/`
instead of `core/`, a single `results/results.md` instead of timestamped JSON, CIP bracket
filenames) are themselves evidence of when the work was done.

Era definitions are in [`STANDARDS.md`](../STANDARDS.md) §2.3. Every relocation is mapped
in [`MIGRATION.md`](../MIGRATION.md).

---

## Era 1 — Symbolic Collapse (2025-06 → 2025-08)

**`era1-symbolic/`** — 20 experiments, plus the essays, empirical-alignment studies,
bridges and Word originals of that period.

The pre-PAC era. Vocabulary inherited from CIMM and QBE: symbolic attractors, bifractal
collapse, entropy fields, herniation geometry. Conservation was not yet axiomatised — no
PAC, no SEC as a formal operator, no Ξ. Reading these needs the Era-1 dictionary; see
[`theory/lexicon.yaml`](../theory/lexicon.yaml), where every term carries the era it was
coined in and what replaced it.

| Experiment | Why it still matters |
|---|---|
| `hodge_conjecture` | SEC's earliest target. The mapping was honestly withdrawn as unproven — then returned from the opposite direction as **Milestone 15**. |
| `quantum_validation` | The QV suite (July 2025). **Milestone 14** explicitly unifies with it: orbit interference and QV's spatial interference are one phenomenon in different complement frames. |
| `symbolic_entropy_collapse` | The largest single body of work here, and the origin of the SEC axiom. |
| `landauer_erasure_field_cost_map` | First erasure-cost work; succeeded by `landauer_erasure_structure`. |

## Era 2 — Pre-field / Infodynamics (2025-09 → 2025-12)

**`era2-prefield/`** — 5 experiments, the Era-2 essays, `PACEngine/`, and two
cross-experiment artifact directories.

The turn from description to mechanism: a Möbius substrate under the field, prime
manifolds, and the first appearance of Ξ.

| Experiment | Why it still matters |
|---|---|
| `pac_confluence_xi` | Where **Ξ = 1 + π/55** was found. Ξ is still live — M15 derives it as a ratio of momentum-operator spectra under two boundary twists. |
| `pre_field_recursion` | Möbius topology as computational substrate — direct ancestor of the `reality-engine` simulator. |
| `information_amplification` | The 15.56× "universal constant" that **was not one**. Its correction is the first entry in [`theory/corrections.md`](../theory/corrections.md) and established the honest-failure discipline the corpus runs on. |
| `PACEngine/` | 62 files with zero imports repo-wide — a citation target and results archive, not a library. Its `results/*.json` are cited by published papers. |

## Non-experiment material

| Directory | Was | Why it's here |
|---|---|---|
| `legacy-docs/` | `foundational/legacy_docs_archive/` | Pre-Dawn-Field CIMM/QBE whitepapers. Filed under `foundational/`, which implied current theory when it predates the theory. **12 of its `.docx` have no markdown twin — that directory is the only copy of their content.** |
| `blueprints/` | `blueprints/` | Speculative applications — nuclear containment, balance-based energy generation, AI detection. No content commit since 2025-06-30. |
| `spike-darkmatter-sec/` | `spikes/darkmatter_SEC_WIP/` | Cosmic-web and galaxy temporal-gradient simulations. Self-labelled WIP, cold since 2025-09. |
| `devkit-designs/` | `devkit/` | 2025-era architecture design documents — aletheia, brainstem, prometheus. Despite the name it was never repository tooling; that lives in `tools/`. |
| `todo-2025/` | `todo/` | Planning material across theory, sdk, infra and biological tracks. Superseded by `roadmaps/` and the milestone structure. |
| `citation-pipeline/` | `citations/pending`, `processed` + 2 workflows | A PR-citation pipeline built and tested in August 2025 that never processed a live citation in the eleven months following. |
| `REPOSITORY_RESTRUCTURE_PLAN.md` | repo root | The September 2025 multi-repo split plan, self-declared superseded — preserved because the split it describes is why `fracton`, `reality-engine` and the rest are separate repos. |

---

## Lineage threads into current work

1. **`era1-symbolic/hodge_conjecture` (2025-06) → Milestone 15 (2026-06).** The earliest
   question and the current frontier are the same question, reached from opposite sides.
2. **`era1-symbolic/quantum_validation` (2025-07) → Milestone 14 (2026-05).**
3. **`era2-prefield/pac_confluence_xi` (2025-12) → Ξ throughout**, into the M15 twist
   classification.
4. **`era2-prefield/pre_field_recursion` (2025-10) → `reality-engine`.**
5. **`era2-prefield/information_amplification` (2025-09) → the corrections methodology.**
   The first retracted constant established the discipline.

## Not archived, deliberately

`papers/series/PACSeries/v0.1` declares itself superseded by v0.2 and would otherwise
belong here, but it carries its own Zenodo DOI (10.5281/zenodo.17295103). A published
package is a frozen artifact — the snapshot is what makes the DOI reproducible — so it
stays where a reader following the DOI expects to find it.
