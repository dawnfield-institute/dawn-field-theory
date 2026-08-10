# Repository Reorganized by Layer

**Date**: 2026-08-10 12:00
**Commits**: fff231bf, 637c9cac, 10130a16, 655a4805
**Type**: refactor

## Summary

The theory had four layers and nothing connected them. Claims lived in root `.md` files and
`foundational/docs/`, the reasoning in `arithmetic/macro_emergence_dynamics/proofs/`, the
measurements in `foundational/experiments/`, the published argument in
`foundational/docs/preprints/`. Four trees, four naming schemes, no cross-layer path — ask
why sin²θ_W = 3/13 and there was no route from claim to proof to test to paper.

The tree is now the argument: `theory/` → `formal/` → `experiments/` → `papers/`, with
`archive/` as terminal lineage by era. `foundational/` is dissolved.

## Changes

### Added
- `MIGRATION.md` — lookup table for every relocation
- `ROADMAP.md` — current direction: M15 Phase 2 and its kill-sentence, Milestone R's
  energy-scale propagation, six known open ends
- `INVENTORY.md` + `tools/generate_inventory.py` — generated corpus view by lifecycle
- `formal/theorems/` — the proven results, graded derivation / structural / negative,
  each indexing the journal that derived it
- `legal/`, `papers/registry/`

### Changed
- `THEORY_MAP.md` evolved into the spine: every claim resolved across all four layers,
  including the open, falsified and conjectural ones. Milestone stack extended to M15 plus
  sidecars; three drifted scores corrected against their own READMEs.
- `theory/lexicon.yaml`: 33 → 50 terms. Every term carries its era and, where evidenced,
  what replaced it. Status is now core / superseded / historical.
- `timeline.md` rewritten as the actual arc, Era 0 through M15.
- `STANDARDS.md` §3 and `CLAUDE.md` describe the new tree; CLAUDE.md's Current State now
  points at generated sources instead of restating scores.

### Removed
- `citations/` PR-citation pipeline and its two workflows — archived. Last real run
  2025-08-25; never processed a live citation in the eleven months since.
- `theory/lexicon.md` — the YAML was a strict superset and the two had drifted.

## Details

**Nothing was lost.** 9147 renames. Each of the 15 paths git reported as deleted was
verified present at its mapped destination — rename detection could not pair them because
`UNIFIED_EVIDENCE.md` is byte-identical across ten papers and `trace.yaml` across five.

**The formal layer was inverted and is now honest.** All eight documents in what was called
`proofs/` were written on 2025-08-20 and describe themselves internally as "Conjecture" or
"Computational Investigation". They are filed under `formal/conjectures/` with their text
unchanged. The results that *are* proven live in milestone journals, and `formal/theorems/`
indexes them there.

**`arithmetic/` split three ways** along what its contents actually were: formal material to
`formal/`, `euclidean_distance_validation` to `experiments/studies/` (it has `core/`, 25
numbered scripts, `tests/`, `journals/`, `results/`), and `PACEngine/` to `archive/` — 62
files with zero imports repo-wide.

**Lore migrated**: 119 FDOs, ~1130 `source_paths`. Final state 189 nodes / 1120 paths with
**2 unresolved**, both pointing at files that never existed in git history. Updates used
`lore_update(fields=…)` and never sent a body — `milestone6-planning-seed`'s body is 8191
characters, one byte under the 8KB truncation boundary, so writing one back would have
silently truncated the record.

**Link health**: 221 dead before, 185 after. Two of my own repair passes had to be undone —
resolving by basename similarity sent 31 experiment references into paper `Data/`
directories and pointed `milestone7` at `milestone9`, which is worse than dead because it
is silently wrong. Re-resolved using link text as ground truth.

**Known remaining**: 185 dead links, only a handful mechanically fixable; per-link
archaeology in mostly archived documents.

## Related
- `MIGRATION.md`, `ROADMAP.md`, `STANDARDS.md`
- `.changelog/20260810_004500_repo_reorg_standardization.md` (the preceding pass)
