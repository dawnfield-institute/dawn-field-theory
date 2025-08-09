# Evidence Map: Claims -> Artifacts (Release 1.0)

Purpose: provide fast, auditable links from high-level claims to concrete code, figures, logs, and protocols in this repo.

- Training-free, online adaptation (TinyCIMM + SCBF)
  - Code (TinyCIMM): models/TinyCIMM/TinyCIMM-Euler/experiments/; models/TinyCIMM/TinyCIMM-Planck/experiments/
  - Code (SCBF harness + metrics): models/scbf/ (runner, loggers, visualization)
  - Key metrics: models/scbf/metrics/ (entropy_collapse.py, activation_ancestry.py, bifractal_lineage.py, phase_alignment.py, semantic_attractors.py)
  - Notes: wording in docs/code updated to "online adaptation / inference-time updates"; no offline training.

- Explainability and transparency (SCBF)
  - Visualization: models/scbf/visualization/
  - Runner and experiment scaffold: models/scbf/scbf_runner.py, models/scbf/tinycimm_scbf_experiment.py

- Quantum/thermo correspondences (experiments)
  - Entry points: internal/QuantumTesting/; foundational/experiments/
  - Theory context: infodynamics.md; origin_of_infodynamics.md

- Machine-native navigation and provenance
  - Protocol: cognition_index_protocol/README.md; cognition_index_protocol/architecture/
  - Repository map: map.yaml; directory-level meta.yaml files

Contribution guidance for evidence:
- Add sub-bullets under the relevant claim with paths to new artifacts (code, logs, figures); include date and brief 1-line summary.
- Keep historical logs untouched; add a short NOTE if terminology evolved.

This map is a living index and will be expanded during the 1.0 polish window. 
