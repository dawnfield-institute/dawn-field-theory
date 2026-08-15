# Papers — What Was Published

Published and preprint artifacts. **Everything under this directory that carries a DOI is
frozen.**

---

## Published packages are not maintained

A DOI points at a snapshot. If the snapshot changes, the DOI stops being reproducible —
so package internals are never edited, never reformatted, and never retrofitted to current
repository standards, even when the current repository standard would say otherwise.

Three consequences visible in this tree:

- `series/PACSeries/v0.1` declares itself superseded by v0.2 and would otherwise belong in
  [`archive/`](../archive/). It stays, because a reader following DOI
  10.5281/zenodo.17295103 has to find it here.
- `meta.yaml` files inside packages were kept when 429 CIP-era copies were removed
  elsewhere in the repo. They are part of published snapshots.
- [`UNIFIED_EVIDENCE.md`](UNIFIED_EVIDENCE.md) exists in 20 copies across five drifted
  versions. Reconciling them would mean editing published packages, so the divergence is
  recorded rather than fixed.

When published content is wrong, the correction goes in
[`theory/corrections.md`](../theory/corrections.md) and the next version — never into the
frozen package.

## Layout

| | |
|---|---|
| [`series/PACSeries/`](series/PACSeries/) | the preprint series — v0.1, v0.2, v0.3 |
| [`standalone/`](standalone/) | 16 individual papers |
| [`legacy/`](legacy/) | 4 published papers no longer updated |
| [`registry/`](registry/) | **the source of truth** — DOIs, publication state, hardware provenance |
| [`pdfs/`](pdfs/) · [`tex_sources/`](tex_sources/) · [`resources/`](resources/) | build outputs and shared assets |

## Start with the registry

[`registry/`](registry/) is authoritative for anything about publication status; the
per-package documents are not, and disagree with it in places.

| File | Holds |
|---|---|
| `ZENODO_REGISTRY.yaml` | per-package publication state |
| `doi_registry.yaml` | DOI assignments |
| `publications_registry.yaml` | publication records and PDF paths |
| `hardware_timeline.yaml` | what hardware produced which result |
| `external_citations/` | citations of this work |

Current package state: **7 current · 11 needs_repackage · 2 incomplete · 1 ready**. The 11
are the largest block of stale state in the repository and are tracked in
[`ROADMAP.md`](../ROADMAP.md).

## Reading the series

[`series/PACSeries/HOW_TO_READ.md`](series/PACSeries/HOW_TO_READ.md) is the entry point.
`derivation_classification.md` separates derived results from measured ones from
speculation — the distinction the series is organised around.

## Citing

There are **two separate Zenodo record families**, and conflating them is what made the
repository appear to contradict itself. Verified against the Zenodo API on 2026-08-10:

| | Concept DOI | Latest version |
|---|---|---|
| **PACSeries** — the paper series | `10.5281/zenodo.17295102` | v0.3 = `…21228036` (2026-07-06) |
| **Repository archive** — this repo as software | `10.5281/zenodo.15595182` | v1.0_prerelease = `…15783623` (2025-07-01) |

- **To cite the papers**, use the PACSeries concept DOI. That is what
  [`README.md`](../README.md) advertises, and a concept DOI always resolves to the newest
  version.
- **To cite the repository**, use [`CITATION.cff`](../CITATION.cff), which points at the
  repository-archive concept DOI.

The repository-archive family has had **only one version, from July 2025**. Its
`CITATION.cff` entry still reads `version: 2.1, date-released: 2026-02-20`, which describes
a state Zenodo has never been given — cutting a fresh GitHub release would close that gap.
