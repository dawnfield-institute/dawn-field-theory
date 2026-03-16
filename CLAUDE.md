# Dawn Field Theory

## What This Is

The core physics repository for Dawn Field Institute. Contains the theoretical framework, 170+ experiments across 14 domains, and published papers for Dawn Field Theory (DFT) — a framework that derives physical constants and dynamics from two information-theoretic axioms: PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse).

This is the **primary source of truth** for the physics. The Kronos vault's 56 physics FDOs reference experiments and documents in this repo.

## Architecture

```
dawn-field-theory/
├── foundational/
│   ├── experiments/          # 51 experiment directories (THE MAIN CONTENT)
│   │   ├── milestone1/       # Standard Model parameter derivations
│   │   ├── milestone2/       # Mass derivations, Navier-Stokes, Koide
│   │   ├── milestone3/       # Quantum validation, Landauer erasure
│   │   ├── pac_confluence_xi/ # PAC-Ξ convergence proofs
│   │   ├── sec_prime_manifold/ # SEC in number theory
│   │   └── ... (51 total)
│   ├── arithmetic/           # PACEngine — core mathematical tools
│   │   ├── PACEngine/        # Conservation math, geometric SEC
│   │   ├── EuclideanDistanceValidation/
│   │   └── HodgeMapping/
│   ├── docs/                 # Bridges, preprints, empirical alignment
│   │   └── preprints/        # Packaged papers with Code/ directories
│   └── lexicon.yaml          # Formal term definitions
├── spikes/                   # Exploratory work (not yet promoted to experiments)
│   ├── darkmatter_SEC_WIP/
│   └── infodynamic_gravity/
├── blueprints/               # Speculative applications
├── citations/                # DOI and citation management
├── models/                   # (minimal)
├── tools/                    # Utility scripts for repo maintenance
├── resources/                # External resources
├── roadmaps/                 # Planning documents
└── [ROOT .md files]          # Theory overview docs
```

## Key Root Documents

| File | Purpose |
|------|---------|
| `dawn-field-theory.md` | Full theory overview (start here for physics) |
| `infodynamics.md` | Infodynamics foundation |
| `origin_of_infodynamics.md` | Origin story and motivation |
| `for_ai_labs.md` | AI-targeted overview |
| `EPISTEMIC_CORRECTIONS_REGISTRY.md` | Honest record of corrections |
| `CITATION.cff` | Citation metadata (requires DOI verification to modify) |
| `map.yaml` | Generated CIP navigation map (~104KB, DO NOT edit manually) |

## Conventions

### Experiment Structure (REQUIRED)
Every experiment in `foundational/experiments/` must have:
- `meta.yaml` — schema v2.0 metadata
- `README.md` — hypothesis, status, key results, FDO links
- `scripts/` — numbered scripts (`exp_NN_name.py`)
- `results/` — output data (if scripts produce any)
- `journals/` — daily research logs (recommended for active work)
- `SYNTHESIS.md` — cross-connections (recommended)

See `STANDARDS.md` at workspace root for full spec.

### Script Naming
- `exp_01_baseline.py`, `exp_02_scaling.py`, etc.
- Results: `results/exp_NN_name_YYYYMMDD_HHMMSS.json`

### Spikes vs Experiments
- `spikes/` — exploratory, no structure requirements, may be promoted to experiments
- `foundational/experiments/` — structured, documented, must meet standards

### Status Values for Experiments
- `active` — currently being worked on
- `completed` — validated, results documented
- `archived` — historical, kept for reference
- `falsified` — hypothesis disproven (these are valuable)

## Related Repos

| Repo | Relationship |
|------|-------------|
| `kronos-vault` | 56 physics FDOs reference experiments here via `source_paths` |
| `fracton` | PAC math library consumed by experiments |
| `reality-engine` | Simulator that implements DFT dynamics |
| `dawn-models` | GAIA ML models that validate DFT predictions |
| `GRIM` | AI companion with skills for experiment management |

## Current State

- **64 experiments** in `foundational/experiments/` (51 prior + 13 in M5)
- **Milestones 1-4** complete (SM parameters, mass derivations, quantum validation, relativity/gravity)
- **Milestone 5** complete — SM completion & simulator validation (13 experiments)
  - Higgs mass 83 ppm (lambda = phi/4pi), PMNS < 0.3 deg, sin^2(theta_W) = tan(theta_C) = 3/13
  - De-actualization completes PAC cycle, 24% scorecard improvement
- **PACSeries** published on Zenodo (DOI: 10.5281/zenodo.15783623)
- **Active organization effort**: bringing all experiments to full standard, adding FDO source links

## Do Not

- Edit `map.yaml` manually (it's generated, ~104KB)
- Modify `CITATION.cff` without DOI verification
- Create experiments outside `foundational/experiments/`
- Create new root-level .md files (use `.changelog/` entries instead)
- Remove or rename experiment directories without updating Kronos FDO `source_paths`
