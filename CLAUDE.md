# Dawn Field Theory

## What This Is

The core physics repository for Dawn Field Institute. Contains the theoretical framework, 75 experiments (724 numbered scripts) across 15 milestones, and published papers for Dawn Field Theory (DFT) — a framework that derives physical constants and dynamics from two information-theoretic axioms: PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse).

This is the **primary source of truth** for the physics. Lore's physics FDOs reference experiments and documents here via `source_paths`. **kronos is frozen — never write through it.**

## Architecture

```
dawn-field-theory/
├── THEORY_MAP.md          THE SPINE — every claim across all four layers
├── ROADMAP.md             what is on deck, what is open, what would falsify it
├── MIGRATION.md           where everything went in the Aug 2026 reorganization
├── STANDARDS.md           canonical spec (structure, meta.yaml, journals, scoring)
│
├── theory/                WHAT IS CLAIMED
│   ├── dawn-field-theory.md, infodynamics.md, for_ai_labs.md
│   ├── lexicon.yaml       50 terms, each with its era and what replaced it
│   ├── corrections.md     the epistemic corrections registry
│   └── essays/            expositional pieces still current (Era 3–4)
│
├── formal/                WHY IT HOLDS
│   ├── theorems/          proven — indexes the journal that derived each result
│   ├── derivations/       axiom → constant chains
│   └── conjectures/       attempted, unproven (the Era-1 MED material)
│
├── experiments/           WHAT WAS MEASURED
│   ├── EXPERIMENTS.md     GENERATED index — the authoritative list
│   ├── milestones/        milestone1 … milestone15 — the derivation chain
│   ├── sidecars/          milestone-r, midnight
│   ├── studies/           33 standalone investigations
│   └── spikes/            exploratory, exempt from the experiment standard
│
├── papers/                WHAT WAS PUBLISHED
│   ├── series/PACSeries/  v0.1 · v0.2 · v0.3
│   ├── standalone/        16 papers
│   ├── legacy/            4 published, no longer updated
│   └── registry/          DOIs, publications, hardware provenance
│
├── archive/               LINEAGE — terminal, by era
│   ├── era1-symbolic/     2025-06 → 08, DFT's first form
│   ├── era2-prefield/     2025-09 → 12, the transition
│   ├── legacy-docs/       pre-DFT CIMM/QBE whitepapers
│   └── README.md          the era guide
│
├── legal/ · roadmaps/ · tools/ · internal/ (gitignored)
```

## Key Root Documents

| File | Purpose |
|------|---------|
| `dawn-field-theory.md` | Full theory overview (start here for physics) |
| `infodynamics.md` | Infodynamics foundation |
| `origin_of_infodynamics.md` | Origin story and motivation |
| `for_ai_labs.md` | AI-targeted overview |
| `theory/corrections.md` | Honest record of corrections |
| `CITATION.cff` | Citation metadata (requires DOI verification to modify) |
| `experiments/EXPERIMENTS.md` | **Generated experiment index** — the authoritative list |
| `STANDARDS.md` | Canonical spec: structure, meta.yaml, journals, scoring, pre-registration |
| `map.yaml` | Generated repo tree (do not edit manually) |

## Conventions

### Experiment Structure (REQUIRED)
Every experiment under `experiments/{milestones,sidecars,studies}/` must have:
- `meta.yaml` — schema v2.1, experiment root ONLY (not in subdirectories)
- `README.md` — thesis, status, score, key results, FDO link
- `scripts/` — numbered scripts (`exp_NN_name.py`)
- `results/` — `exp_NN_name_YYYYMMDD_HHMMSS.json`
- `journals/` — dated research logs
- `SYNTHESIS.md` — cross-connections (recommended)

`experiments/spikes/` is exempt by definition.

See [`STANDARDS.md`](STANDARDS.md) in this repository for the full spec.

### Script Naming
- `exp_01_baseline.py`, `exp_02_scaling.py`, etc.
- Results: `results/exp_NN_name_YYYYMMDD_HHMMSS.json`

### Spikes vs Experiments
- `spikes/` — exploratory, no structure requirements, may be promoted to experiments
- `experiments/milestones/` — structured, documented, must meet standards

### Status Values for Experiments
- `active` — currently being worked on
- `completed` — validated, results documented
- `archived` — historical, kept for reference
- `falsified` — hypothesis disproven (these are valuable)

## Related Repos

| Repo | Relationship |
|------|-------------|
| `lore` (CT106) | The knowledge graph. Physics FDOs reference experiments here via `source_paths`. **kronos is frozen — never write through it.** |
| `fracton` | PAC math library consumed by experiments |
| `reality-engine` | Simulator that implements DFT dynamics |
| `dawn-models` | GAIA ML models that validate DFT predictions |
| `bert` | The platform that superseded GRIM |

## Current State

**Do not restate scores here.** They drift — that is how five documents came to claim five
different experiment counts. Authoritative sources:

| For | Read |
|---|---|
| Per-experiment status and score | `experiments/EXPERIMENTS.md` (generated) |
| Claims across all four layers | `THEORY_MAP.md` |
| What is proven vs measured vs conjectured | `formal/theorems/README.md` |
| What is on deck and what would falsify it | `ROADMAP.md` |
| How the framework got here | `timeline.md` |
| Where anything moved in Aug 2026 | `MIGRATION.md` |

Shape as of 2026-08: **75 experiments** — 50 live (15 milestones, 2 sidecars, 33 studies)
and 25 archived across Eras 1–2. Milestones 1–14 complete; **M15 Phase 1 closed**, Phase 2
open on the field-equation hunt with a standing kill-sentence. Sidecars: **Milestone R**
(radiation as ledger severance, 60/112) and **Midnight** (observational contact, 22/32).

### Working here

- `meta.yaml` lives at the **experiment root only**. The 429 subdirectory copies were CIP
  artifacts and were removed.
- Run `python tools/validate_experiment_structure.py` before committing. CI enforces it,
  plus index freshness and generated-artifact currency.
- Generated files — `map.yaml`, `experiments/EXPERIMENTS.md`, the `files`/
  `child_directories` keys of any `meta.yaml` — are never hand-edited.
- Archived work is **not retrofitted**. Its old layout and filenames are evidence of when
  it was done.
- Pre-register before running: hypothesis, quantified thresholds, falsification condition
  (`STANDARDS.md` §2.7). Register invariants, never absolute coordinates.

## Do Not

- Edit `map.yaml` manually (it's generated, ~104KB)
- Modify `CITATION.cff` without DOI verification
- Create experiments outside `experiments/{milestones,sidecars,studies}/`
- Create new root-level .md files (use `.changelog/` entries instead)
- Remove or rename experiment directories without updating the corresponding Lore FDO `source_paths` (and typed-node `slots.sources`) in the same change
