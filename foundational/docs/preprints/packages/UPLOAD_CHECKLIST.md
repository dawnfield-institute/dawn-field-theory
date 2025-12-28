# Zenodo Upload Checklist - 2025-12-28 (Updated)

## Quick Reference

**Zenodo Upload URL:** https://zenodo.org/deposit/new

**Note:** PDF versions of all papers will be added in v1.1. Current v1.0 packages contain markdown papers with full Code/Data/Figures.

**Upload Process:**
1. Go to https://zenodo.org/deposit/new
2. Click "Upload files" and select the .zip package
3. Metadata auto-populates from .zenodo.json
4. Review/edit metadata
5. Choose license: AGPL-3.0
6. Publish (or Save Draft first)

---

## NEW Papers (First Upload) - READY NOW

### 🔥 Priority 1 - Tier 1 Validation Papers

| Paper | Package | Status |
|-------|---------|--------|
| **Cellular Automata Xi Clustering** | `cellular_automata_xi_clustering_v1.0_20251228_095048.zip` | ✅ Ready |
| **Golden Ratio Prime Distribution** | `golden_ratio_prime_distribution_v1.0_20251228_095054.zip` | ✅ Ready |
| **ML Validation Pythia/GPT2** | `ml_validation_pythia_gpt2_v1.0_20251228_095102.zip` | ✅ Ready |

**Notes:**
- CA paper: 7 experiments, 8 figures, p < 10^-7 validation
- Golden Ratio: 32 experiments, 1 figure, analytical proof
- ML Validation: 3 experiments, 9 figures, multi-model

---

## Papers Needing Data Relocation (Before Upload)

| Paper | Issue | Fix |
|-------|-------|-----|
| pac_necessity_proof | JSON in Code/results/ | Copy to Data/results/ |
| qbe_pac_unification | JSON in Code/results/ | Copy to Data/results/ |
| gaia_field_native_intelligence_comprehensive | JSON in Code/results/ | Copy to Data/results/ |
| potential_actualization_conservation_comprehensive | Data in Code/experiments/results/ | Copy to Data/results/ |

---

## UPDATED Papers (Version 2.0) - Already on Zenodo

| Paper | Previous DOI | Status |
|-------|--------------|--------|
| symbolic_entropy_collapse | 10.5281/zenodo.17024434 | ⚠️ Needs figures |
| dawn_field_theory_synthesis | 10.5281/zenodo.17024367 | ⚠️ Needs data move |
| dawn_field_theory_infodynamics | 10.5281/zenodo.17041188 | ⚠️ Needs data move |
| symbolic_cognition_collapse_interpretability | 10.5281/zenodo.17024098 | ⚠️ Needs v2.0 |
| macro_emergence_dynamics_navier_stokes | 10.5281/zenodo.17041215 | ⚠️ Needs data move |

**For updates:** Use "New version" button on existing record.

---

## Papers Current (No Update Needed)

| Paper | DOI | Status |
|-------|-----|--------|
| cognition_index_protocol | 10.5281/zenodo.17024220 | ✅ Current |
| resonant_symbolic_convergence_human_agent | 10.5281/zenodo.17023921 | ✅ Current |
| recursive_mathematical_plasticity_entropy_architecture | 10.5281/zenodo.17041249 | ✅ Current |
| PACSeries (5 papers) | 10.5281/zenodo.17295103 | ✅ Current |

---

## Recommended Upload Order

1. **cellular_automata_xi_clustering** - 🔥 Hot off the press, Tier 1!
2. **golden_ratio_prime_distribution** - Major Tier 1 proof
3. **ml_validation_pythia_gpt2** - Multi-model ML validation
4. Then fix data locations for remaining papers
5. Then v2.0 updates for published papers

---

## After Upload

Update `ZENODO_REGISTRY.yaml` with:
- New DOI numbers
- Upload dates
- Change status from `pending_upload` → `published`
