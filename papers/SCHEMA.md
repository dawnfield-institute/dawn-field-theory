# Dawn Field Theory Preprints Schema v2.0

## Overview

This folder contains all Dawn Field Theory preprints in a self-contained, Zenodo-ready format. Each paper lives in its own folder with all code, data, and metadata needed for reproduction.

## Structure

```
preprints/
├── meta.yaml                           # Master index of all papers
├── SCHEMA.md                           # This file
├── migrate_to_new_structure.py         # Migration script (can be removed)
│
├── {paper_slug}/                       # One folder per paper
│   ├── meta.yaml                       # Paper metadata
│   ├── paper.md                        # The paper (Markdown)
│   ├── paper.tex                       # LaTeX version (when generated)
│   ├── paper.pdf                       # PDF version (when generated)
│   ├── README.md                       # Quick start guide
│   ├── CITATION.md                     # How to cite
│   ├── LICENSE                         # MIT (code) + CC-BY-4.0 (paper)
│   │
│   ├── Code/
│   │   ├── trace.yaml                  # Links to original source files
│   │   ├── requirements.txt            # Python dependencies
│   │   ├── reproduce.py                # Single entry point
│   │   ├── core/                       # Reusable modules
│   │   └── experiments/                # Numbered experiment scripts
│   │
│   ├── Data/
│   │   └── results/                    # Generated JSON results
│   │
│   └── Figures/                        # Visualizations
│
└── drafts/                             # DEPRECATED - to be removed
```

## Paper Naming Convention

Paper slugs are derived from the original filename:
```
[pac][D][v1.0][C4][I5][E]_cellular_automata_xi_clustering_preprint.md
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          This becomes the folder name
```

## Metadata Fields

### meta.yaml (paper level)

```yaml
schema_version: "2.0"
slug: "cellular_automata_xi_clustering"
title: "Cellular Automata Xi Clustering"
category: "pac"           # pac, sec, id, m, ai, cip
status: "Draft"           # Draft or Final
version: "1.0"
complexity: 4             # 1-5
impact: 5                 # 1-5
evidence_type: "E"        # E=Empirical, R=Review, A=Analytical, O=Other
created: "2025-12-20T..."

original_filename: "[pac][D][v1.0]..._preprint.md"
source_location: "drafts"

files:
  paper: "paper.md"
  latex: "paper.tex"
  pdf: "paper.pdf"

code:
  has_experiments: true
  trace_file: "Code/trace.yaml"
```

### trace.yaml (code traceability)

```yaml
schema_version: "1.0"
created: "2025-12-20T..."
description: "Traces code files back to their original repository locations"
source_repository: "dawn-field-theory"
files:
  - local: "core/ca_simulator.py"
    source: "dawn-field-theory/experiments/milestones/.../core/ca_simulator.py"
    repo: "dawn-field-theory"
```

## Categories

| Tag | Category | Description |
|-----|----------|-------------|
| `pac` | PAC Theory | Potential-Actualization-Conservation framework |
| `sec` | SEC | Symbolic Entropy Collapse |
| `id` | Infodynamics | Dawn Field Theory core |
| `m` | Mathematics | Mathematical physics applications |
| `ai` | AI/ML | Artificial intelligence applications |
| `cip` | CIP | Cognition Index Protocol |

## Priority Tiers

### Tier 1: High Validation (Release First)
Papers with strongest empirical evidence:
- `cellular_automata_xi_clustering` - p < 10⁻⁷
- `ml_validation_pythia_gpt2` - Multi-model validation
- `golden_ratio_prime_distribution` - Analytical + numerical

### Tier 2: Framework Papers
Core theoretical foundations:
- `potential_actualization_conservation_comprehensive`
- `xi_bounded_invariant_universal_balance_operator`
- `dawn_field_theory_infodynamics`

### Tier 3: Application Papers
Domain-specific applications:
- `gaia_field_native_intelligence_comprehensive`
- `symbolic_cognition_collapse_interpretability`
- `macro_emergence_dynamics_navier_stokes`

## Zenodo Upload

Each paper folder is self-contained and ready for Zenodo:

```bash
# Zip a paper for upload
cd preprints/cellular_automata_xi_clustering
zip -r ../cellular_automata_xi_clustering_v1.0.zip .

# Upload to Zenodo
# - Use meta.yaml for metadata
# - Add to dawn-field-institute community
# - Update CITATION.md with assigned DOI
```

## Quick Reference

```bash
# Run experiments for a paper
cd preprints/cellular_automata_xi_clustering
python Code/reproduce.py

# List available experiments
python Code/reproduce.py --list

# Run specific experiment
python Code/reproduce.py 7
```

## Version History

- **v2.0** (2025-12-20): Restructured to one folder per paper with trace.yaml
- **v1.0** (2025-10-06): Original PACSeries package structure

---

*Last updated: December 20, 2025*
