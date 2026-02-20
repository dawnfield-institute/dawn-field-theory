# Versioning Scheme Implementation

**Date**: 2026-02-20 18:50
**Type**: engineering

## Summary
Implemented formal semantic versioning across all 16 papers and CITATION.cff. Cleaned up test author entries from CITATION.cff.

## Changes

### Changed
- 6 new PACSeries papers (Papers 1–6): Version set to 2.1
  - structure_cost_of_erasure: 1.0 → 2.1
  - balance_constant_decomposition: (none) → 2.1 (field added)
  - feigenbaum_fibonacci_arithmetic: (none) → 2.1 (field added)
  - standard_model_fibonacci_arithmetic: 0.1 → 2.1
  - classical_physics_information_geometry: 0.1 → 2.1
  - computational_validation_pac_conservation: 0.1 → 2.1
- 5 legacy PACSeries papers: Version set to 1.1
  - xi_bounded_invariant: 1.0 → 1.1
  - sec_med_framework: 1.0 → 1.1
  - mobius_confluence: 1.0 → 1.1
  - relativistic_mas: v1.0 → v1.1
  - gaia_computational_validation: 1.0 → 1.1
- 2 standalone preprints: Version field corrected to v1.1
  - pac_necessity_proof: v1.0 → v1.1 (was contradicting status line)
  - ml_validation_pythia_gpt2: v1.0 → v1.1 (was contradicting status line)
- 2 standalone preprints already at v1.1: no changes needed
  - cellular_automata_xi_clustering: already v1.1
  - she_leveque_fibonacci_turbulence: already v1.1
- CITATION.cff: version 1.0 → 2.1, date-released 2025-09-01 → 2026-02-20, year 2025 → 2026

### Fixed
- Removed test authors (Test User, Test Contributor with fake ORCIDs) from CITATION.cff
- Fixed version/status contradictions in pac_necessity_proof and ml_validation_pythia_gpt2

## Details
Versioning scheme: papers get semantic versioning (major.minor), theory gets git tags.
- New PACSeries = v2.1 (v2.0 era papers, .1 for edit passes)
- Legacy PACSeries = v1.1 (v1.0 papers with February 2026 appendices)
- Standalone preprints = v1.1 (original papers with February 2026 updates)
- CITATION.cff tracks theory-level version (2.1)
- Git tags: theory-v2.1 for this release
