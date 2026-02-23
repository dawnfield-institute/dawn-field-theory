# Zenodo Version Update Packages: 6 Papers

**Date**: 2026-02-23 14:47
**Type**: release

## Summary
Created v1.1/v2.1 Zenodo packages for 6 papers with substantial changes since their last upload. Updated source meta.yaml versions, ZENODO_REGISTRY.yaml, and UPLOAD_CHECKLIST.md.

## Changes

### Added
- New Zenodo packages (in `foundational/docs/preprints/packages/`):
  - `symbolic_entropy_collapse_v1.1_20260223_144738.zip` (121 KB)
  - `qbe_pac_unification_v1.1_20260223_144738.zip` (439 KB)
  - `dawn_field_theory_infodynamics_v2.1_20260223_144738.zip` (67 KB)
  - `ml_validation_pythia_gpt2_v1.1_20260223_144738.zip` (1026 KB)
  - `cellular_automata_xi_clustering_v1.1_20260223_144738.zip` (796 KB)
  - `pac_necessity_proof_v1.1_20260223_144738.zip` (251 KB)
- `packages/create_v11_packages.ps1` - reusable packaging script

### Changed
- Source meta.yaml versions updated for all 6 papers
- `qbe_pac_unification/paper.md` version bumped v1.0 -> v1.1
- `ZENODO_REGISTRY.yaml` rewritten to reflect current state (was stale since 2025-12-28)
- `UPLOAD_CHECKLIST.md` replaced with current version update instructions

## Details
Compared all 11 papers with existing Zenodo packages against their current source. Found all had been modified by commits `2f68765` (preprint tightening) and `72e7e3f` (versioning + appendices). Selected the 6 with substantial changes for repackaging. The remaining 5 had minor edits (4-37 lines) and were deferred.

Papers with symbolic_entropy_collapse having the largest change (-319 lines from a major rewrite/tightening). Three papers received new cross-reference sections from the PACSeries work.

## Related
- Previous: PACSeries v0.2 upload (DOI: 10.5281/zenodo.18743674)
- Packages ready for upload via Zenodo "New version" workflow
