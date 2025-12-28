# Zenodo Upload Plan - December 28, 2025

## Quick Links
- **Zenodo Upload**: https://zenodo.org/uploads/new
- **Packages Folder**: `foundational/docs/preprints/packages/`

---

## Part 1: NEW UPLOADS (7 papers)

Upload these in order. Each gets a NEW Zenodo record.

### Tier 1 - Validation Papers (Upload First)

| # | Paper | Package File |
|---|-------|--------------|
| 1 | **Cellular Automata Xi Clustering** | `cellular_automata_xi_clustering/cellular_automata_xi_clustering_v1.0_20251228_103333.zip` |
| 2 | **Golden Ratio Prime Distribution** | `golden_ratio_prime_distribution/golden_ratio_prime_distribution_v1.0_20251228_103344.zip` |
| 3 | **ML Validation Pythia GPT2** | `ml_validation_pythia_gpt2/ml_validation_pythia_gpt2_v1.0_20251228_103345.zip` |

### Tier 2 - Supporting Papers

| # | Paper | Package File |
|---|-------|--------------|
| 4 | **PAC Necessity Proof** | `pac_necessity_proof/pac_necessity_proof_v1.0_20251228_122846.zip` |
| 5 | **QBE-PAC Unification** | `qbe_pac_unification/qbe_pac_unification_v1.0_20251228_122847.zip` |
| 6 | **GAIA Field Native Intelligence** | `gaia_field_native_intelligence_comprehensive/gaia_field_native_intelligence_comprehensive_v1.0_20251228_122848.zip` |
| 7 | **PAC Comprehensive** | `potential_actualization_conservation_comprehensive/potential_actualization_conservation_comprehensive_v1.0_20251228_122849.zip` |

---

## Part 2: VERSION UPDATES (3 papers)

These are v2.0 uploads to EXISTING Zenodo records. Use "New version" button on each record.

| Paper | Existing DOI | Package File |
|-------|--------------|--------------|
| **Dawn Field Theory Synthesis** | 10.5281/zenodo.17024367 | `dawn_field_theory_synthesis/dawn_field_theory_synthesis_v1.0_20251228_122856.zip` |
| **Dawn Field Theory Infodynamics** | 10.5281/zenodo.17041188 | `dawn_field_theory_infodynamics/dawn_field_theory_infodynamics_v1.0_20251228_122857.zip` |
| **MED Navier-Stokes** | 10.5281/zenodo.17041215 | `macro_emergence_dynamics_navier_stokes/macro_emergence_dynamics_navier_stokes_v1.0_20251228_122857.zip` |

---

## Upload Workflow

### For NEW Uploads:
1. Go to https://zenodo.org/uploads/new
2. Upload the `.zip` file (drag & drop)
3. **Unzip the package first** - upload both:
   - The paper `.md` file
   - The `Code/`, `Data/`, `Figures/` folders (or zip them separately)
4. Fill in metadata from `.zenodo.json` inside the package
5. Set **Resource type**: Preprint
6. Set **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
7. **Publish**
8. Copy the DOI

### For VERSION Updates:
1. Go to existing record (use DOI link)
2. Click "New version"
3. Delete old files, upload new package contents
4. Update version to "2.0"
5. Add note: "v2.0: Added reproducibility package with Code, Data, and Figures"
6. **Publish**

---

## Metadata Quick Reference

All packages contain `.zenodo.json` with pre-filled metadata. Key fields:

```json
{
  "title": "[Paper Title]",
  "upload_type": "publication",
  "publication_type": "preprint",
  "creators": [{"name": "Field, Peter", "affiliation": "Dawn Field Institute"}],
  "license": "AGPL-3.0",
  "keywords": ["Dawn Field Theory", "Infodynamics", ...]
}
```

---

## Post-Upload Checklist

After each upload, record the DOI:

### New Uploads
- [ ] cellular_automata_xi_clustering → DOI: _______________
- [ ] golden_ratio_prime_distribution → DOI: _______________
- [ ] ml_validation_pythia_gpt2 → DOI: _______________
- [ ] pac_necessity_proof → DOI: _______________
- [ ] qbe_pac_unification → DOI: _______________
- [ ] gaia_field_native_intelligence_comprehensive → DOI: _______________
- [ ] potential_actualization_conservation_comprehensive → DOI: _______________

### Version Updates (v2.0)
- [ ] dawn_field_theory_synthesis (v2.0) → Confirmed
- [ ] dawn_field_theory_infodynamics (v2.0) → Confirmed  
- [ ] macro_emergence_dynamics_navier_stokes (v2.0) → Confirmed

---

## After All Uploads Complete

1. Update `foundational/docs/preprints/ZENODO_REGISTRY.yaml` with new DOIs
2. Commit changes: `git add -A && git commit -m "Add Zenodo DOIs for all papers"`
3. Push: `git push`

---

## Notes

- **PDF versions**: Planned for v1.1 (not blocking this upload)
- **Total packages**: 10 (7 new + 3 updates)
- **Estimated time**: ~30-45 minutes for all uploads
