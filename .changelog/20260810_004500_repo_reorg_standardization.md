# Repository Reorganization and Restandardization

**Date**: 2026-08-10 00:45
**Commits**: 85095aa2, 45ac7696, cdd6360e, bb079933
**Type**: refactor

## Summary

The repo had outgrown its documentation. Seven documents claimed five different
experiment counts against a real 73; `map.yaml` was eleven months stale; `THEORY_MAP.md`
stopped at M11; and `CLAUDE.md` pointed at a canonical `STANDARDS.md` that did not exist,
in whose absence three incompatible `meta.yaml` specs and two journal specs had grown up
with nothing validating any of them. Four vintages of work sat in one flat namespace, so
2025-era symbolic-collapse experiments were indistinguishable from the live M15 frontier.

This pass establishes the standard, relocates the 2025 eras to an archive that reads as
lineage, replaces hand-maintained indexes with generated ones, and adds a CI gate so the
drift cannot silently return.

## Changes

### Added
- `STANDARDS.md` — canonical spec: structure, `meta.yaml` schema v2.1 as two explicit
  zones (generated vs authored), journal format, scoring, pre-registration protocol,
  eras, vault sync. Lives in this repo because `core_workspace` is not a git repository.
- `tools/validate_experiment_structure.py`, `tools/generate_experiment_index.py`
- `foundational/experiments/EXPERIMENTS.md` — generated index, the authoritative list
- `foundational/experiments/archive/README.md` — era guide with five lineage threads
- `archive/README.md`, `.gitattributes`, `.github/workflows/repo-standards.yml`
- `foundational/experiments/milestone10/README.md` — absent despite a 64/71 score

### Changed
- 25 experiments relocated to `archive/era1` and `archive/era2`; `devkit/` and `todo/`
  to a repo-root `archive/`
- `era` and `status` stamped into 73 `meta.yaml`; 12 status spellings normalised to 4
- 430 stale references rewritten across 79 files (`THEORY_MAP.md`: 42 dead links → 0)
- Counts reconciled across seven documents to 73 directories / 752 numbered experiments
- 36 Lore FDOs migrated: 191 `source_paths` corrected

### Fixed
- `generate_path.py` and `update_meta_yamls.py` walked the filesystem with no
  `.gitignore` awareness. Regenerating swept the private `internal/` tree into artifacts
  committed to a **public** repo. Both now enumerate from `git ls-files`.
- `milestone1` exp_27 and exp_31 built import paths as Python expressions, resolved to
  vanished directories, caught `ImportError`, warned, and continued at exit 0 with
  transcribed values. Fixed and confirmed by execution.
- Longest tracked path was 226 chars and would exceed Windows `MAX_PATH` from a
  conventional clone root; era directories named `era1`/`era2` bring it to 201.
- Text corruption in `MISSION.md` (subtitle spliced into a section) and
  `roadmaps/core_project_roadmap.md` (duplicated header block)

### Removed
- `mcp/` — hardcoded the pre-split repo path and loaded a CIP file deleted in Feb 2026;
  every response returned "(CIP instruction unavailable)". Nothing depended on it.
- `models/` — one `meta.yaml` for a `GAIA/` directory that does not exist
- `.github/workflows/update-meta-yamls.yml` — opened a PR on every push, producing 13
  duplicate open PRs against one branch

## Details

**Archived is not deprecated.** Superseded work here is lineage: corrections layer
forward, and a reframing is only legible against what it replaced. `hodge_conjecture`
(2025-06) returned as Milestone 15; the July 2025 quantum-validation suite is what M14
unifies with. Nothing is retrofitted — the `reference_material/` layout is itself
evidence of when the work was done.

**The gate is the point.** Structure, index freshness, and generated-artifact currency
now fail CI rather than rot quietly. Golem was considered and rejected for this repo: the
doctrine in `bert/.bert/pipeline.yaml` is that Golem must never watch a public repo,
since a fork PR would run a stranger's code on the executor. The validation logic lives
in `tools/`, so the runner is portable if that ever changes.

**Two rules in `STANDARDS.md` were wrong on first writing** and were corrected rather
than left to be found: scoring is `N/M` with `M` fixed before the run, not always `N/4`
(the corpus has 19/19, 7/7, 0/3); and `tags` is recommended, not required, since 53 of 73
lack it — a required field that only warns is not a standard.

**Known remaining**: repo-wide link rot is pre-existing and larger than this change
(224 dead links before, 237 after — this pass fixed 90 and introduced 103, concentrated
in archived historical docs). It deserves its own pass via `tools/link_checker.py`.

## Related

- `STANDARDS.md`
- `foundational/experiments/archive/README.md`
- `.changelog/20260612_184500_m15_holonomy_closed_form.md` (previous entry)
