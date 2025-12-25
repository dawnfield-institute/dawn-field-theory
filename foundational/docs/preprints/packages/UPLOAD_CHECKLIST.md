# Zenodo Upload Checklist - 2025-12-20

## Quick Reference

**Zenodo Upload URL:** https://zenodo.org/deposit/new

**Upload Process:**
1. Go to https://zenodo.org/deposit/new
2. Click "Upload files" and select the .zip package
3. Metadata auto-populates from .zenodo.json
4. Review/edit metadata
5. Choose license: AGPL-3.0
6. Publish (or Save Draft first)

---

## NEW Papers (First Upload)

### 🔥 Priority 1 - Validation Papers

| Paper | Package | Size | Status |
|-------|---------|------|--------|
| **Cellular Automata Xi Clustering** | `cellular_automata_xi_clustering_v1.0_20251220_120517.zip` | 122 KB | ⬜ Ready |
| **Golden Ratio Prime Distribution** | `golden_ratio_prime_distribution_v1.0_20251220_120518.zip` | 243 KB | ⬜ Ready |

**Notes:**
- These are Tier 1 validation papers with full code and data
- CA paper: 7 experiments, p < 10^-7 validation
- Golden Ratio: 32 experiments, analytical proof

---

## UPDATED Papers (Version 2.0)

| Paper | Package | Size | Previous DOI | Status |
|-------|---------|------|--------------|--------|
| Symbolic Entropy Collapse | `symbolic_entropy_collapse_v1.0_20251220_120524.zip` | 256 KB | 10.5281/zenodo.17024434 | ⬜ Ready |
| Dawn Field Theory Synthesis | `dawn_field_theory_synthesis_v1.0_20251220_120525.zip` | 32 KB | 10.5281/zenodo.17024367 | ⬜ Ready |
| Dawn Field Theory Infodynamics | `dawn_field_theory_infodynamics_v1.0_20251220_120525.zip` | 39 KB | 10.5281/zenodo.17041188 | ⬜ Ready |
| Symbolic Cognition Interpretability | `symbolic_cognition_collapse_interpretability_v1.0_20251220_120526.zip` | 33 KB | 10.5281/zenodo.17024098 | ⬜ Ready |
| Recursive Mathematical Plasticity | `recursive_mathematical_plasticity_entropy_architecture_v1.0_20251220_120526.zip` | 30 KB | 10.5281/zenodo.17041249 | ⬜ Ready |
| Macro Emergence Dynamics | `macro_emergence_dynamics_navier_stokes_v1.0_20251220_120526.zip` | 34 KB | 10.5281/zenodo.17041215 | ⬜ Ready |

**For updates:** Use "New version" button on existing record, or use Zenodo API.

---

## NOT Ready (Need Work)

| Paper | Issue |
|-------|-------|
| ml_validation_pythia_gpt2 | needs_code - No experiments yet |
| pac_necessity_proof | needs_code - No experiments yet |
| qbe_pac_unification | needs_code - No experiments yet |
| potential_actualization_conservation_comprehensive | needs_code - No experiments yet |
| gaia_field_native_intelligence_comprehensive | needs_code - No experiments yet |

---

## Package Locations

All packages at:
```
dawn-field-theory/foundational/docs/preprints/packages/
├── cellular_automata_xi_clustering/
│   └── cellular_automata_xi_clustering_v1.0_20251220_120517.zip
├── golden_ratio_prime_distribution/
│   └── golden_ratio_prime_distribution_v1.0_20251220_120518.zip
├── symbolic_entropy_collapse/
│   └── symbolic_entropy_collapse_v1.0_20251220_120524.zip
├── dawn_field_theory_synthesis/
│   └── dawn_field_theory_synthesis_v1.0_20251220_120525.zip
├── dawn_field_theory_infodynamics/
│   └── dawn_field_theory_infodynamics_v1.0_20251220_120525.zip
├── symbolic_cognition_collapse_interpretability/
│   └── symbolic_cognition_collapse_interpretability_v1.0_20251220_120526.zip
├── recursive_mathematical_plasticity_entropy_architecture/
│   └── recursive_mathematical_plasticity_entropy_architecture_v1.0_20251220_120526.zip
└── macro_emergence_dynamics_navier_stokes/
    └── macro_emergence_dynamics_navier_stokes_v1.0_20251220_120526.zip
```

---

## After Upload

Update `ZENODO_REGISTRY.yaml` with:
- New DOI numbers
- Upload dates
- Change status from `needs_update` → `current`
- Move from `pending_upload` → `published`

## Recommended Upload Order

1. **cellular_automata_xi_clustering** - Hot off the press, Tier 1 validation!
2. **golden_ratio_prime_distribution** - Major Tier 1 proof with 32 experiments
3. **symbolic_entropy_collapse** - Core theory update (has 77 files!)
4. Then remaining 5 updates in any order
