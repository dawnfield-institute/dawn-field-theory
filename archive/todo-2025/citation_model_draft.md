Title: Design Document: Pull Request Citation Schema for Dawn Field Repository

Author: Peter Groom
Date: 2025-08-01


---

Overview

This document outlines a proposed system for modular citation within the Dawn Field Theory repository. While the core repository remains attributable to Peter Groom, individual pull requests can carry their own scholarly attribution through metadata and DOI registration. This design promotes transparency, reproducibility, and proper credit distribution without compromising the central authorship of the theory.


---

Motivation

Preserve intellectual authorship for foundational work (Peter Groom).

Enable fair attribution to contributors for significant pull requests.

Align with scientific citation norms and Zenodo integration.

Create a structured, verifiable history of contributions.

Support future CIP and SCBF audit pipelines.



---

System Architecture

1. Root-Level Repository Citation

The repository should retain a top-level BibTeX or .citation.cff file citing Peter Groom as the primary author.

@misc{dawnfield_theory,
  author       = {Peter Groom},
  title        = {Dawn Field Theory Repository},
  year         = {2025},
  url          = {https://github.com/dawnfield-institute/dawn-field-theory},
  note         = {For component-specific citations, see pull request metadata}
}

2. Per-Pull Request Citation Metadata

Each meaningful PR may optionally include a .citation.yaml file structured as follows:

title: "Entropy Collapse Simulator V2"
authors:
  - name: Jane Doe
    orcid: 0000-0002-1825-0097
doi: "10.5281/zenodo.1234567"
repository: https://github.com/dawnfield-institute/dawn-field-theory/pull/42
date: "2025-08-01"
related_to: entropy/simulator-v2

> Location: Placed in the root of the pull request, committed as part of the PR.



3. Zenodo Integration (Optional)

Contributors may upload their PR to Zenodo and attach a DOI.

A GitHub Action or manual link may be used to insert the DOI into the .citation.yaml.


4. Central Citation Index

A new folder: citations/

citations/
├── PR42_entropy-simulator-v2.yaml
├── PR51_bifractal-metrics.yaml
└── contributors-index.json

Additionally, a JSON index file may be created for machine parsing:

[
  {
    "pr": 42,
    "title": "Entropy Collapse Simulator V2",
    "doi": "10.5281/zenodo.1234567",
    "author": "Jane Doe"
  }
]


---

Benefits

Maintains clean authorship boundaries.

Enables modular citation for contributors.

Integrates with Git history and Zenodo.

Readable by both humans and automated audit systems.

Enhances reproducibility and version traceability.

Citations persist even if files or code are later removed — Git history serves as a long-term citation ledger.



---

Next Steps

[ ] Finalize .citation.yaml schema.

[ ] Add GitHub Action (optional) for Zenodo DOI integration.

[ ] Write parser to generate/update citations/index.json from PRs.

[ ] Add documentation for contributors on how to create PR citations.

[ ] Integrate with CIP or SCBF infrastructure (future).



---

Appendix: Challenges and Solutions

1. Granularity & Thresholds

Not all PRs are worthy of citation (e.g., typos, refactors).

Define a threshold for citable contributions.

Use GitHub labels or a citable: true/false field in .citation.yaml to flag relevance.


2. Citation Conflicts and Merge Complexity

Conflicts may occur during merges when two PRs alter the citation index.

Solution: Auto-generate citations/index.json using a script or GitHub Action post-merge.


3. Collaborative PR Attribution

Multiple contributors per PR should be supported.

.citation.yaml can support authors: [] array with optional ORCID, affiliation, and role.


4. Scale & Administrative Load

Maintain a pr-citation-template.yaml for contributors to copy.

Consider assigning a "citation steward" for reviewing new PRs.


5. DOI Versioning & Synchronization

If Zenodo is used, DOI should be minted at merge time.

Optionally trigger Zenodo sync via GitHub Action when merging main.


6. Historical Citation Auditing

Citations persist in Git history even if code is removed.

CIP tooling can support cip citations list to reconstruct citation history.


7. Quality Control and Schema Validation

Pre-commit or CI-based validation can ensure .citation.yaml integrity.

Use schema validation and linting to prevent malformed files.



---

Scalability Contingency: Centralized Citation Repository

As the Dawn Field ecosystem scales beyond a single repository, the per-repository citation model may become administratively complex. A potential solution is migrating to a centralized citation repository that manages attribution across all Dawn Field Institute repositories.

Architecture Overview:

A dedicated repository (e.g., dawn-field-citations) would contain:

```
dawn-field-citations/
├── repositories/
│   ├── dawn-field-theory/
│   │   ├── PR-42-entropy-simulator.yaml
│   │   └── PR-51-bifractal-metrics.yaml
│   ├── quantum-substrate/
│   └── consciousness-framework/
├── contributors/
│   ├── jane-doe.yaml
│   └── peter-groom.yaml
├── indices/
│   ├── by-repository.json
│   ├── by-contributor.json
│   └── by-topic.json
└── tools/
    ├── citation-validator.py
    └── cross-reference-builder.py
```

Schema Example:

```yaml
repository_url: "https://github.com/dawnfield-institute/dawn-field-theory"
repository_name: "dawn-field-theory"
citation_type: "pull_request"
reference_id: "PR-42"
title: "Entropy Collapse Simulator V2"
authors:
  - name: "Jane Doe"
    orcid: "0000-0002-1825-0097"
    role: "lead_developer"
doi: "10.5281/zenodo.1234567"
date: "2025-08-01"
related_paths:
  - "entropy/simulator-v2/"
dependencies:
  - repository: "quantum-substrate"
    citation_id: "PR-15"
```

Benefits:

- Cross-repository attribution for contributors working across multiple projects
- Unified dependency mapping and research graph construction
- Centralized DOI management and Zenodo integration
- Single citation schema and review process
- Enhanced administrative efficiency at ecosystem scale
- Citation-as-a-service model keeping individual repositories clean and focused

This approach would be triggered when citation management overhead exceeds the benefits of per-repository systems, likely around 5-10 active repositories with regular contributions.


---

Long-Term Vision

This citation schema may form the basis for a more general-purpose audit layer in CIP — enabling symbolic provenance, DOI-backed module lineage, and structured epistemic recursion over all computational work.


---

End of Document

