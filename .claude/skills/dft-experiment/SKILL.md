---
name: dft-experiment
description: Create or modify an experiment in dawn-field-theory so it passes the structure validator and records results honestly. Use when adding an experiment, adding scripts or results to one, writing a journal entry, or pre-registering a claim before running it.
---

# Working on a dawn-field-theory experiment

`STANDARDS.md` is canonical; this is the operational subset plus the traps that have
actually cost time here.

## Where experiments live

`experiments/{milestones,sidecars,studies}/` — the validator covers these three.
`experiments/spikes/` is **exempt by definition** (STANDARDS §3): exploratory, no
structure required, may be promoted later. Do not "fix" a spike into an experiment
unless you are deliberately promoting it.

Never create an experiment outside those four directories.

## Required structure

```
experiment_name/
├── meta.yaml        REQUIRED — experiment root ONLY
├── README.md        REQUIRED — thesis, status, score, key results, FDO link
├── SYNTHESIS.md     recommended — cross-connections
├── core/            reusable modules imported by scripts
├── scripts/         exp_NN_name.py
├── results/         exp_NN_name_YYYYMMDD_HHMMSS.json
└── journals/        YYYY-MM-DD_slug.md
```

**`meta.yaml` goes at the experiment root and nowhere else.** Not in `core/`, `scripts/`,
`results/` or `journals/`. 429 subdirectory copies were a CIP artifact and were removed;
recreating them re-breaks `git log` as a staleness signal.

## meta.yaml has two zones

```yaml
schema_version: "2.1"
directory_name: milestone15
description: "What this directory contains."
title: "Milestone 15: The Representative Problem"   # experiment roots
status: active                                      # active|completed|archived|falsified
era: era4-milestone-stack-2026q2
```

`files:` and `child_directories:` are the **generated zone** — owned by
`tools/update_meta_yamls.py`. Never hand-edit them; the updater round-trips the document
and rewrites only that zone, so your authored fields survive.

## Results are append-only

`results/exp_NN_name_YYYYMMDD_HHMMSS.json`. The timestamp makes every run addressable and
**nothing overwrites a prior run**. If you find yourself replacing a result file, stop —
that is how a disagreeing earlier run disappears.

## Scores are `N/M`, and hardening may lower them

Put the score in the README as `Score: 34/40`. A test that passes for a reason unrelated
to what it guards is tautological and gets **replaced, not counted** — which is why
hardening cycles sometimes reduce a score. M11 went 52 → 49 → 52 and the dip was the
process working. Never tune a test to pass.

A failed experiment is a result. Record it; `theory/corrections.md` is a first-class
artifact.

## Pre-registration (STANDARDS §2.7)

Before running: commit hypothesis, quantified thresholds, and the falsification condition.
Commit outcomes separately, citing the registration hash. **Thresholds are never relaxed
after seeing results.**

**Register invariants, never absolute coordinates.** Registered relations survive;
registered coordinates die. This rule came from Midnight and now governs the whole corpus
— a prediction of "the peak sits at x = 4.2" is worthless where "the ratio of the two
peaks is φ" is testable.

## Silent-failure trap

Two milestone1 scripts once built import paths as expressions, resolved them to
directories that no longer existed, caught the `ImportError`, printed a warning, and
**exited 0 with transcribed values** — a green run reporting numbers nothing had computed.
If a script has a fallback path, make the fallback fail loudly. An undetected failure is
itself a bug.

## Before committing

```bash
python tools/validate_experiment_structure.py    # must print "No errors."
```

Then follow the `dft-repo-gates` skill for the regeneration order — the generators read
the git index, so running them before `git add` produces files that fail CI.
