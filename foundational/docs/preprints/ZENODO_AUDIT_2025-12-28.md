# Zenodo Preprints Audit Report

**Date:** December 28, 2025 (Updated)  
**Audited by:** GitHub Copilot  
**Location:** `dawn-field-theory/foundational/docs/preprints/`

---

## Executive Summary

| Metric | Count | Status |
|--------|-------|--------|
| **Total Preprints (Main)** | 15 | |
| **PACSeries Papers** | 6 | Already on Zenodo |
| **✅ READY for Upload** | 3 | cellular_automata, golden_ratio, ml_validation |
| **⚠️ Needs Data Move** | 4 | JSON in Code/results → Data/results |
| **🔧 Needs Update (v2.0)** | 5 | Already published, need revision |
| **✅ Current (no update)** | 3 | Protocol/framework papers |

**Status:** 3 new papers ready for immediate upload. 4 papers need JSON files copied from `Code/results/` or `Code/experiments/results/` to `Data/results/`. 5 already-published papers flagged for v2.0 revision.

---

## 1. Already Published on Zenodo

### PACSeries (Bundled Upload)
| DOI | Version | Status |
|-----|---------|--------|
| [10.5281/zenodo.17295103](https://doi.org/10.5281/zenodo.17295103) | 1.0 | ✅ Current |

Contains 5 papers:
1. xi_bounded_invariant_universal_balance_operator
2. mobius_confluence_operator_temporal_emergence
3. gaia_computational_validation_dawn_field_theory
4. relativistic_mas_universal_frequency
5. sec_med_framework_information_amplification

### Individual Papers - Published

| Paper | DOI | Status | Action Needed |
|-------|-----|--------|---------------|
| symbolic_entropy_collapse | [10.5281/zenodo.17024434](https://doi.org/10.5281/zenodo.17024434) | ⚠️ needs_update | Upload v2.0 |
| dawn_field_theory_synthesis | [10.5281/zenodo.17024367](https://doi.org/10.5281/zenodo.17024367) | ⚠️ needs_update | Upload v2.0 |
| dawn_field_theory_infodynamics | [10.5281/zenodo.17041188](https://doi.org/10.5281/zenodo.17041188) | ⚠️ needs_update | Upload v2.0 |
| cognition_index_protocol | [10.5281/zenodo.17024220](https://doi.org/10.5281/zenodo.17024220) | ✅ current | None (protocol paper) |
| symbolic_cognition_collapse_interpretability | [10.5281/zenodo.17024098](https://doi.org/10.5281/zenodo.17024098) | ⚠️ needs_update | Upload v2.0 |
| resonant_symbolic_convergence_human_agent | [10.5281/zenodo.17023921](https://doi.org/10.5281/zenodo.17023921) | ✅ current | None (framework paper) |
| recursive_mathematical_plasticity_entropy_architecture | [10.5281/zenodo.17041249](https://doi.org/10.5281/zenodo.17041249) | ✅ current | None (theory paper) |
| macro_emergence_dynamics_navier_stokes | [10.5281/zenodo.17041215](https://doi.org/10.5281/zenodo.17041215) | ⚠️ needs_update | Upload v2.0 |

---

## 2. Detailed Audit by Paper

### ✅ READY FOR UPLOAD

#### 1. cellular_automata_xi_clustering
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | schema 2.0, v1.0 |
| paper.md | ✅ | Complete |
| Code/ | ✅ | trace.yaml, reproduce.py, core/, experiments/ |
| Data/results/ | ✅ | 10 JSON files |
| Figures/ | ✅ | 8 files (PDF+PNG) |
| Package | ✅ | `cellular_automata_xi_clustering_v1.0_20251228_095048.zip` |

**Notes:** Tier 1 validation paper with p < 10^-7. **UPLOAD FIRST.**

---

#### 2. golden_ratio_prime_distribution
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | schema 2.0, v1.0 |
| paper.md | ✅ | Complete |
| Code/ | ✅ | trace.yaml, reproduce.py, core/, experiments/ |
| Data/results/ | ✅ | 27 JSON files |
| Figures/ | ✅ | 1 file |
| Package | ✅ | `golden_ratio_prime_distribution_v1.0_20251228_095054.zip` |

**Notes:** Tier 1 validation with 32 experiments. Analytical + numerical proof.

---

#### 3. ml_validation_pythia_gpt2
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | schema 2.0, v1.0 |
| paper.md | ✅ | Complete |
| Code/ | ✅ | trace.yaml, reproduce.py, core/, experiments/, **results/** |
| Data/results/ | ✅ | 2 JSON files |
| Figures/ | ✅ | 9 files |
| Package | ✅ | `ml_validation_pythia_gpt2_v1.0_20251228_095102.zip` |

**Notes:** Tier 1 multi-model ML validation. Previously marked "needs_code" but now complete.

---

### ⚠️ NEEDS MINOR FIXES (JSON Location)

These papers have results in `Code/results/` or `Code/experiments/results/` that need to be copied to `Data/results/`:

#### 4. pac_necessity_proof
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | Complete |
| paper.md | ✅ | Complete |
| Code/ | ✅ | 45 validated experiments + `Code/results/` with 1 JSON |
| Data/results/ | ❌ | **Empty - needs files from Code/results/** |
| Figures/ | ❌ | Empty |

**Fix:** Copy `Code/results/exp_26_pac_violation_20251222_094620.json` → `Data/results/`

---

#### 5. qbe_pac_unification
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | Complete |
| paper.md | ✅ | Complete |
| Code/ | ✅ | 45 validated experiments + `Code/results/` with 1 JSON |
| Data/results/ | ❌ | **Empty - needs files from Code/results/** |
| Figures/ | ✅ | 1 file (exp_32_qbe_frequency_derivation.png) |

**Fix:** Copy `Code/results/exp_32_qbe_pac_unification_20251222_094825.json` → `Data/results/`

---

#### 6. gaia_field_native_intelligence_comprehensive
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | Complete |
| paper.md | ✅ | Complete |
| Code/ | ✅ | Has `Code/results/` with 2 JSON files |
| Data/results/ | ❌ | **Empty - needs files from Code/results/** |
| Figures/ | ✅ | 22 files |

**Fix:** Copy `Code/results/*.json` → `Data/results/`

---

#### 7. potential_actualization_conservation_comprehensive
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | Complete |
| paper.md | ✅ | Complete |
| Code/ | ✅ | Has `Code/experiments/results/` with JSON |
| Data/results/ | ❌ | **Empty - needs files from Code/experiments/results/** |
| Figures/ | ✅ | 28 files |

**Fix:** Copy `Code/experiments/results/20251222_095505/*` → `Data/results/`

---

### 🔧 ALREADY PUBLISHED - NEEDS V2.0 UPDATE

These papers are on Zenodo but have new code/experiments to add:

#### 8. symbolic_entropy_collapse (DOI: 10.5281/zenodo.17024434)
| Component | Status | Details |
|-----------|--------|---------|
| meta.yaml | ✅ | Complete |
| paper.md | ✅ | Complete |
| Code/ | ✅ | 5 experiments with core/ |
| Data/results/ | ✅ | **27 JSON files** ✅ |
| Figures/ | ❌ | **Empty - needs generation** |

**Status:** Data present, needs figures generated, then upload v2.0.

---

#### 9. dawn_field_theory_synthesis (DOI: 10.5281/zenodo.17024367)
| Component | Status | Details |
|-----------|--------|---------|
| Data/results/ | ❌ | Empty (but `Code/experiments/results/` has data) |
| Figures/ | ❌ | Empty |

**Fix:** Move data from `Code/experiments/results/20251223_134822/` → `Data/results/`

---

#### 10. dawn_field_theory_infodynamics (DOI: 10.5281/zenodo.17041188)
| Component | Status | Details |
|-----------|--------|---------|
| Code/ | ✅ | 5 experiments + `Code/experiments/output/` has data |
| Data/results/ | ❌ | Empty |
| Figures/ | ❌ | Empty |

**Fix:** Move data from `Code/experiments/output/` → `Data/results/`

---

#### 11. macro_emergence_dynamics_navier_stokes (DOI: 10.5281/zenodo.17041215)
| Component | Status | Details |
|-----------|--------|---------|
| Code/ | ✅ | Has `Code/experiments/results/run_20251223_135158/` with JSON + graphs |
| Data/results/ | ❌ | Empty |
| Figures/ | ❌ | Empty |

**Fix:** Move JSON from `Code/experiments/results/` → `Data/results/`, graphs → `Figures/`

---

#### 12. symbolic_cognition_collapse_interpretability (DOI: 10.5281/zenodo.17024098)
| Component | Status | Details |
|-----------|--------|---------|
| Code/ | ✅ | 4 experiments |
| Data/results/ | ❌ | Empty |
| Figures/ | ❌ | Empty |

**Status:** Has experiments but no results yet. Run experiments or mark as theoretical.

---

### ✅ CURRENT - NO UPDATE NEEDED

These papers are theoretical/framework papers without empirical data requirements:

#### 13. cognition_index_protocol (DOI: 10.5281/zenodo.17024220)
**Status:** ✅ CURRENT - Protocol/methodology paper, no code needed.

---

#### 14. recursive_mathematical_plasticity_entropy_architecture (DOI: 10.5281/zenodo.17041249)
**Status:** ✅ CURRENT - Theoretical architecture paper, no code needed.

---

#### 15. resonant_symbolic_convergence_human_agent (DOI: 10.5281/zenodo.17023921)
**Status:** ✅ CURRENT - Framework paper, no code needed.

---

## 3. PACSeries Bundle (Already on Zenodo)

**DOI:** [10.5281/zenodo.17295103](https://doi.org/10.5281/zenodo.17295103)

| Paper | Data | Figures | Status |
|-------|------|---------|--------|
| xi_bounded_invariant_universal_balance_operator | ❌ Empty | ❌ Empty | Theoretical |
| mobius_confluence_operator_temporal_emergence | ❌ Empty | ❌ Empty | Theoretical |
| gaia_computational_validation_dawn_field_theory | ✅ 12+ JSON | ❌ Empty | Has data |
| relativistic_mas_universal_frequency | ✅ 18 JSON | ❌ Empty | Has data |
| sec_med_framework_information_amplification | ✅ 27 JSON | ❌ Empty | Has data |

**Status:** Published. Some have data, some are theoretical. Consider v2.0 if figures needed.

---

## 4. Recommended Upload Order

### Phase 1: NEW Papers (Immediate - Upload Today)
These are ready NOW:

| # | Paper | Package Date | Priority |
|---|-------|--------------|----------|
| 1 | **cellular_automata_xi_clustering** | 2025-12-28 | 🔥 Tier 1 (p < 10^-7) |
| 2 | **golden_ratio_prime_distribution** | 2025-12-28 | 🔥 Tier 1 (32 exps) |
| 3 | **ml_validation_pythia_gpt2** | 2025-12-28 | 🔥 Tier 1 (ML valid) |

### Phase 2: Quick Fixes Then Upload
Copy JSON files, regenerate packages, upload:

| # | Paper | Fix Needed |
|---|-------|------------|
| 4 | pac_necessity_proof | Copy 1 JSON |
| 5 | qbe_pac_unification | Copy 1 JSON |
| 6 | gaia_field_native_intelligence_comprehensive | Copy 2 JSON |
| 7 | potential_actualization_conservation_comprehensive | Copy folder |

### Phase 3: V2.0 Updates (Already Published)
Move data, generate figures, upload as new versions:

| # | Paper | Current DOI |
|---|-------|-------------|
| 8 | symbolic_entropy_collapse | 10.5281/zenodo.17024434 |
| 9 | dawn_field_theory_synthesis | 10.5281/zenodo.17024367 |
| 10 | dawn_field_theory_infodynamics | 10.5281/zenodo.17041188 |
| 11 | macro_emergence_dynamics_navier_stokes | 10.5281/zenodo.17041215 |
| 12 | symbolic_cognition_collapse_interpretability | 10.5281/zenodo.17024098 |

---

## 5. Action Items

### ✅ Ready to Upload NOW
```bash
# Navigate to packages
cd dawn-field-theory/foundational/docs/preprints/packages

# Upload these 3 packages to Zenodo
cellular_automata_xi_clustering_v1.0_20251228_095048.zip
golden_ratio_prime_distribution_v1.0_20251228_095054.zip
ml_validation_pythia_gpt2_v1.0_20251228_095102.zip
```

### 🔧 Data Relocation Commands
```powershell
# pac_necessity_proof
Copy-Item "pac_necessity_proof/Code/results/*.json" "pac_necessity_proof/Data/results/"

# qbe_pac_unification  
Copy-Item "qbe_pac_unification/Code/results/*.json" "qbe_pac_unification/Data/results/"

# gaia_field_native_intelligence_comprehensive
Copy-Item "gaia_field_native_intelligence_comprehensive/Code/results/*.json" "gaia_field_native_intelligence_comprehensive/Data/results/"

# potential_actualization_conservation_comprehensive
Copy-Item "potential_actualization_conservation_comprehensive/Code/experiments/results/20251222_095505/*" "potential_actualization_conservation_comprehensive/Data/results/" -Recurse
```

### 📝 Registry Updates After Upload
Update `ZENODO_REGISTRY.yaml`:
- Add new DOIs for the 3 new papers
- Move entries from `pending_upload` to `published`
- Change status from `needs_update` to `current` after v2.0 uploads

---

## 6. Verification Summary

| Check | Status |
|-------|--------|
| All 15 preprints have meta.yaml | ✅ |
| All have paper.md, README.md, CITATION.md | ✅ |
| All have LICENSE (AGPL-3.0) | ✅ |
| All have Code/trace.yaml, requirements.txt, reproduce.py | ✅ |
| All have Code/core/, Code/experiments/ | ✅ |
| All have Data/, Figures/ directories | ✅ |
| 3 papers fully ready for upload | ✅ |
| 4 papers need JSON relocation | ⚠️ |
| 5 papers need v2.0 update | ⚠️ |
| 3 papers are current (no update) | ✅ |

**Overall Structure Compliance:** 100% with SCHEMA.md v2.0
