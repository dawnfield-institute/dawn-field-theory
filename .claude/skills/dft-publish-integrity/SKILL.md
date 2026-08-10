---
name: dft-publish-integrity
description: Rules for anything under papers/ in dawn-field-theory — published packages, DOIs, and citation metadata. Use before editing, moving, reformatting or repackaging any paper, running a repo-wide sweep that could touch papers/, or answering a question about which DOI to cite.
---

# Published packages are frozen

**A DOI points at a snapshot. If the snapshot changes, the DOI stops being reproducible.**

Anything under `papers/` carrying a DOI is a frozen artifact: never edited, never
reformatted, never retrofitted to current repository standards — even when the current
standard says otherwise.

This has been violated twice and both times had to be reverted:

- A repo-wide reference sweep rewrote links inside **113 files** in DOI-bearing packages.
- A `meta.yaml` cleanup removed **62 files** from inside published packages, because the
  rule it applied ("meta.yaml at experiment roots only") does not scope to papers at all.

Before any bulk operation, exclude `papers/`. After any bulk operation, verify:

```bash
# papers/ must be byte-identical to its last published state
git diff --stat main..HEAD -- papers/
```

## Three consequences that look like inconsistencies

They all follow from the freeze rule. Do not "fix" them:

- `series/PACSeries/v0.1` declares itself superseded by v0.2 and would otherwise belong in
  `archive/` — it stays, because a reader following its DOI must find it here.
- Package `meta.yaml` files survived the repo-wide removal of 429 others.
- `UNIFIED_EVIDENCE.md` exists in 20 copies across 5 drifted versions. Reconciling them
  would mean editing published packages. The divergence is recorded in `papers/README.md`
  instead.

## Corrections go forward, never backward

When published content is wrong, the correction goes into `theory/corrections.md` and the
next version. Never into the frozen package. Superseded work is **lineage**: a result that
was later reframed is only legible against what it replaced.

## Two Zenodo record families — do not conflate them

Verified against the Zenodo API 2026-08-10:

| | Concept DOI | Latest version |
|---|---|---|
| **PACSeries** — the papers | `10.5281/zenodo.17295102` | v0.3 = `…21228036` (2026-07-06) |
| **Repository archive** — this repo as software | `10.5281/zenodo.15595182` | v1.0_prerelease = `…15783623` (2025-07-01) |

- Citing the **papers** → PACSeries concept DOI. This is what `README.md` advertises.
- Citing the **repository** → `CITATION.cff`, which carries the repo-archive concept DOI.

**Always cite a concept DOI, never a version DOI.** A concept DOI resolves forward to the
newest version; `CITATION.cff` previously carried `15783623`, a *version* DOI, which pinned
every citation of the repository to a July 2025 snapshot.

`CITATION.cff` is not modified without verifying against Zenodo first.

## `registry/` is the source of truth

For publication status, DOIs and provenance, read `papers/registry/` — not the per-package
documents, which disagree with it in places.

`status: needs_repackage` means the **source** moved ahead of the **uploaded artifact**
(v1.2 source vs v1.1/v2.1 packaged). Repackaging cuts a new DOI version, and production
publishing is TTY-CLI only per ADR-0029 — it is not an agent action.
