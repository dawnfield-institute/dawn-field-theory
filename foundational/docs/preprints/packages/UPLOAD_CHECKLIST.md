# Zenodo Upload Checklist - 2026-03-22

## Status

All 12 standalone preprints have been updated with M4/M5 milestone results (March 2026).
The v1.1/v2.1 packages from Feb 23 are now SUPERSEDED — new packages needed with M4/M5 content.

### Upload Process (Version Updates)
1. Go to the existing Zenodo record URL
2. Click "New version"
3. Remove the old zip, upload the new one
4. Update the version number in the form
5. Publish

---

## Superseded Packages (Feb 23 — do NOT upload these)

These 6 packages were built Feb 23 but are now outdated. Build fresh packages instead.

| Paper | Old Package | Superseded By |
|-------|------------|---------------|
| Symbolic Entropy Collapse | v1.1 | v1.3 source (M4/M5 headers added) |
| QBE-PAC Unification | v1.1 | v1.2 source (M5 headers added) |
| Infodynamics | v2.1 | v1.2 source (M4/M5 headers added) |
| ML Validation Pythia/GPT2 | v1.1 | v1.2 source (M5 headers added) |
| Cellular Automata Xi | v1.1 | v1.2 source (M4 headers added) |
| PAC Necessity Proof | v1.1 | v1.2 source (M4 headers added) |

---

## Papers Needing New Packages (12 total)

Use `preprint_package(slug, version)` MCP tool or build manually.

### Published on Zenodo (need version update upload)

| Paper | Current Zenodo | New Source Version |
|-------|---------------|-------------------|
| symbolic_entropy_collapse | [v1.0](https://zenodo.org/records/17024434) | v1.3 |
| dawn_field_theory_synthesis | [v2.0](https://zenodo.org/records/18087136) | v1.2 |
| dawn_field_theory_infodynamics | [v2.0](https://zenodo.org/records/18087191) | v1.2 |
| macro_emergence_dynamics_navier_stokes | [v2.0](https://zenodo.org/records/18087212) | v1.2 |
| cellular_automata_xi_clustering | [v1.0](https://zenodo.org/records/18086711) | v1.2 |
| golden_ratio_prime_distribution | [v1.0](https://zenodo.org/records/18086778) | v1.2 |
| ml_validation_pythia_gpt2 | [v1.0](https://zenodo.org/records/18086821) | v1.2 |
| pac_necessity_proof | [v1.0](https://zenodo.org/records/18086893) | v1.2 |
| qbe_pac_unification | [v1.0](https://zenodo.org/records/18086941) | v1.2 |
| potential_actualization_conservation_comprehensive | [v1.0](https://zenodo.org/records/18087020) | v1.4 |

### Not Yet on Zenodo

| Paper | Status |
|-------|--------|
| worldseed_evolutionary_architecture | Ready for first upload |
| pac_cosmology_jwst_validation | M4/M5 update added, still needs figures |
| she_leveque_fibonacci_turbulence | M4/M5 update added, still needs figures |
| sec_threshold_detection | Incomplete draft |
| bidirectional_sec_pac_fluid | Incomplete draft |

---

## What Changed (March 2026)

All updated papers received a `> **March 2026 Update.**` blockquote with references to:
- **Milestone 4** (15 experiments): Xi 800x stability, turbulence mode count universality, Lorentz uniqueness, Gaussian envelope, cascade amplification
- **Milestone 5** (13 experiments): Higgs mass 83ppm, PMNS < 0.3deg, sin²θ_W = tan(θ_C) = 3/13, strong force implicit, de-actualization

---

## After Upload

For each paper uploaded:
1. Update `citations/doi_registry.yaml` with the new DOI and version
2. Update `ZENODO_REGISTRY.yaml` status to `current`
3. Git tag the release
