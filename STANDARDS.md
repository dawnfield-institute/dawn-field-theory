# Dawn Field Institute — Workspace Standards

**Status:** canonical · **Version:** 1.0 · **Established:** 2026-08-09

This is the single normative specification for repository structure, experiments,
metadata, journals, changelogs, and knowledge-graph sync across `core_workspace`.

It lives in `dawn-field-theory/` because it must be version-controlled and its links
must resolve on GitHub; `core_workspace/` is not a git repository. Other repos reference
it by URL.

## Scope and precedence

This document is **canonical**. Where it conflicts with anything else, it wins:

1. `STANDARDS.md` (this file)
2. Per-repo `CLAUDE.md` — repo-specific context and current state
3. `.claude/instructions/*.instructions.md` — task-specific operating procedures

It was written to close a real gap: `CLAUDE.md` and `.claude/instructions/main.instructions.md`
both referenced `STANDARDS.md` as the canonical spec, but no such file existed. In its
absence three mutually incompatible `meta.yaml` specs and two overlapping journal specs
grew up, and nothing validated any of them.

### Why a workspace-wide document lives in this repository

`core_workspace/` is **not a git repo**, so a copy at the workspace root would be
untracked, unbacked-up, and free to drift from any tracked copy. The canonical copy
therefore lives here, at `dawn-field-theory/STANDARDS.md`, where it is version-controlled
and public. Everything that references it — the workspace `CLAUDE.md`,
`.claude/instructions/main.instructions.md`, and per-repo `CLAUDE.md` files — points at
that path. **There is exactly one copy; do not create a second.**

---

## 1. Workspace layout

Root is `core_workspace/` — **not itself a git repo**. Each subdirectory is an independent
repo with its own remote. The manifest is `repos.yaml`.

Every repo must have: `README.md`, `meta.yaml`, `CLAUDE.md`, `.changelog/`.
Research repos additionally have `.spec/` where code is spec-driven.

---

## 2. Experiment standard

Applies to `experiments/{milestones,sidecars,studies}/`, and to any repo carrying
experiments (`dawn-models`, `reality-engine`).

### 2.1 Required structure

```
experiment_name/
├── meta.yaml              # REQUIRED — §5
├── README.md              # REQUIRED — thesis, status, score, key results, FDO link
├── SYNTHESIS.md           # recommended — cross-connections to other experiments
├── core/                  # reusable module code imported by scripts
├── scripts/               # exp_NN_name.py — §2.4
├── results/               # exp_NN_name_YYYYMMDD_HHMMSS.json — §2.4
└── journals/              # YYYY-MM-DD_slug.md — §6
```

`meta.yaml` lives at the **experiment root only** — not in `core/`, `scripts/`, `results/`
or `journals/`. Per-directory metadata was a Cognition Index Protocol artifact; CIP was
removed in February 2026, and the 429 subdirectory files that survived it only restated
the directory listing that `map.yaml` already generates. They cost 7% of the repo's file
count, covered 55% of directories so could never serve as a complete index, and produced
the `Auto-update meta.yaml` commits that made `git log` useless as a staleness signal.

`core/` supersedes the older `reference_material/` convention. Era 1–2 experiments retain
`reference_material/` and are **not** retrofitted — the layout is itself era evidence.

### 2.2 Status

| Value | Meaning |
|---|---|
| `active` | currently being worked |
| `completed` | validated, results documented, not being extended |
| `archived` | historical; superseded but preserved as lineage |
| `falsified` | hypothesis disproven — **valuable, never deleted** |

A falsified experiment is a result, not a failure. It stays in the corpus with its
falsification documented.

### 2.3 Era

Every experiment carries an `era`. Eras are conceptual, not merely chronological — they
track what the framework's vocabulary and methods actually were at the time.

| Era | Window | Character |
|---|---|---|
| `era1-symbolic-collapse-2025h1` | 2025-06 → 2025-08 | CIMM/QBE heritage; symbolic attractors, bifractal collapse, entropy fields |
| `era2-prefield-infodynamics-2025h2` | 2025-09 → 2025-12 | Möbius substrate, prime manifolds, Ξ discovery |
| `era3-pac-formalization-2026q1` | 2026-01 → 2026-04 | PAC/SEC axioms, M1–M5, constants derivation |
| `era4-milestone-stack-2026q2` | 2026-04 → present | M6–M15 + R; ADE, complement, holonomy |

Era 1–2 experiments live under `archive/era1-symbolic/` and `archive/era2-prefield/`.
Era 3–4 experiments live under `experiments/{milestones,sidecars,studies}/`.

**Assigning an era.** Use the date the work *matured* — its last substantive commit —
not its first. Experiments are frequently opened in one era and concluded in the next
(`navier-stokes` was opened 2025-08 and finished under PAC in 2025-12; it is Era 2).
Ignore repo-wide hygiene commits when reading dates: a bulk pass on 2026-03-23 created
`journals/` and `results/` directories across ~28 dormant 2025-era experiments and will
otherwise date all of them to 2026.

Era and status are **independent**. An Era-3 experiment can be `archived` in place
without relocating; only Eras 1–2 are relocated as whole eras.

Archiving is **relocation, not deprecation**. Superseded work is lineage: corrections
layer forward, and the archive is where the framework's own history is readable. Nothing
is deleted for being old.

### 2.4 Naming

```
scripts/exp_NN_short_name.py
results/exp_NN_short_name_YYYYMMDD_HHMMSS.json
```

`NN` is zero-padded and monotonic within an experiment. Scripts write results as JSON;
never overwrite a prior result file — the timestamp makes each run addressable.

**Lettered sub-experiments.** A numbered experiment may branch into `exp_NNx_name.py`
(`exp_30a_conformal_generation.py` … `exp_30q_primes_l2_closure.py`). Use this when the
sub-experiments share one hypothesis and are read as a series; use a new number when the
hypothesis differs. A trailing letter also marks a refinement of an earlier run
(`exp_01b_refined.py`).

**Helpers.** Shared code imported by several scripts belongs in `core/`. Small
script-local helpers may sit in `scripts/` as `_name.py`, `constants.py`, or
`run_all_experiments.py`. Anything else in `scripts/` should be a numbered experiment.

### 2.5 Milestone structure

Milestones are experiments with a thesis that spans several sub-experiments. Directory
name is `milestoneN` (or a letter for sidecars, e.g. `milestone-r`). In addition to §2.1:

- `README.md` carries the thesis, the **scorecard**, key results, honest failures, the
  predictions registry, dependencies, and a forward path.
- Sub-experiments are grouped into lettered **blocks** (A, B, C…) by theme.

### 2.6 Scoring convention

Each experiment defines a fixed set of tests with quantified pass criteria and scores
`N/M`, where `M` is fixed **before** the run. Four tests (`T1`–`T4`) is the common case
and the default; a derivation-verification or synthesis experiment may use a different
count (M10 exp_17 scores `7/7`, M15 exp_02 `0/3`). A milestone's score is the sum over
its experiments, reported as `total/possible` with a percentage — which is why milestone
totals are not generally multiples of four.

Rules:
- Pass criteria are fixed **before** the run (§2.7). Thresholds are never relaxed after
  seeing results.
- A test that passes for a reason unrelated to what it guards is **tautological** and must
  be replaced, not counted. Hardening cycles that *lower* a score are correct behaviour.
- Failures are reported with what they reveal. A milestone README without an honest-failures
  section is incomplete.

### 2.7 Pre-registration protocol

Standard practice since 2026-06-11. For any experiment intended as evidence:

1. **Register before running.** Commit a journal entry stating hypothesis, the four tests
   with quantified thresholds, and what outcome would falsify the claim.
2. **Register invariants, never absolute coordinates.** Claims are relational — registered
   relations survive, registered coordinates die. This is the *invariant-registration rule*.
3. **Run, then commit outcomes separately**, referencing the pre-registration commit hash.
4. **Disclose postdiction.** A match found after the fact is labelled postdictive; only
   predictions registered before measurement count as confirmations.
5. **Kill-sentences stand.** If a milestone declares a condition under which it fails, that
   condition is honoured when met.

### 2.8 Recursive, tautological, circular — three things that look alike

This framework's generative primitive is **self-application** (M10: self-applied symmetry is
the unique primitive that neither regresses nor produces noise). So DFT results sit
permanently next to a line that most frameworks never approach, and the words get mixed up in
both directions: real recursive results get dismissed as circular, and genuinely circular ones
get defended with *"it's recursive, that's the point."* **"It is recursive" is never by itself
a defence.**

| | what it is | the test |
|---|---|---|
| **Recursive** | Self-reference that is **productive** — it generates distinctions. Has a base case and terminates or converges. | **Vary the input.** Different inputs must give different outputs. |
| **Tautological** | Self-reference or construction that generates **no** distinction. True for every input, so no outcome could have differed. | **Could any input have changed the verdict?** If no → tautological. |
| **Circular** | The conclusion was **available to the procedure that produced it**. Unlike a tautology this is not visible on the face of it — it hides in the fitting. | **What could the search see?** If the "emergent" quantity was reachable by the tuning, it did not emerge. |

**The discriminator is discrimination.** A recursion earns its keep by separating cases; a
tautology cannot separate anything; a circularity separates cases using information it should
not have had.

Each has a real instance in this corpus:

- **Recursive (valid).** `Ψ(k) = Ψ(k+1) + Ψ(k+2)`, and M13's identity-IS-complement — a
  vertex's identity is the graph without it. Self-referential, but *different vertices have
  different complements*, so it discriminates. That is what makes it a definition rather than
  a restatement.
- **Tautological (caught, and scores lowered).** Milestone R's `scope_boundary_count`:
  `E_Planck · φ^(−d)` exceeds every nuclear and atomic energy by 15–24 orders, so the boundary
  count rounds to zero for *every* physical input. exp_06 T1/T2/T4 and exp_08 T1 "passed"
  because nothing could have failed. Recorded in that milestone's README as passes that are
  not evidence.
- **Circular (withdrawn).** The claim that Ξ ≈ 1.057 "emerged in Navier–Stokes before it was
  derived". It was grid-searched over a tunable, with `1 + π/55` fitted afterwards
  (`phi_artifact_test`). The constant did not emerge from the dynamics; it was reachable by
  the search.

### 2.8.1 A good attack makes a claim *more* falsifiable

The point of attacking your own result is not only to catch errors. The better outcome is that
a soft claim gets replaced by a hard one — and the tell is that the survivor has **more** ways
to be wrong than the original did.

Worked example. "Gauge couplings carry exactly one power of φ" classified nine constants and
correctly predicted a held-out case. It was then killed by one line:

    F3/(F4*phi*F10)  ==  (-1 + sqrt(5))/165        same number, zero phi's written

A φ-count is a property of the chosen notation, so the claim could never die — any
counterexample is answered by rewriting. What replaced it was **field membership**: mixing
angles and mass ratios are exactly rational, couplings are not. That is a fact about the
numbers, invariant under rewriting, and it *can* die — one irrational mass ratio or one
exactly-rational coupling ends it.

This is the invariant-registration rule (§2.7.2) applied to claims rather than to
registrations. **A φ-count is a coordinate; field membership is a relation.** In practice, the
claims that survive scrutiny here are the ones stateable without reference to a representation,
and the ones that die are the ones that quietly depend on one.

When a result survives an attack, ask what it can now be killed by. If the answer is "nothing",
the attack did not finish.

Practical consequences:

- A test whose result is invariant under its own inputs is replaced, not counted (§2.6).
- Before claiming a quantity **emerged**, state what the fitting procedure had access to.
  "Emergent" and "reachable by the search" are mutually exclusive.
- A recursion reported as a result must show its **base case** and its termination or fixed
  point. Self-reference without a base case is not a recursion, it is an unfinished sentence.
- Near-tautology is a matter of degree and must be reported: a test with a factor-1000
  acceptance window discriminates weakly even when it is not strictly vacuous.

---

## 3. Where work lives

The tree follows the **argument**, not the artifact type. Four layers plus terminal
lineage:

| Layer | Directory | Holds |
|---|---|---|
| What is claimed | `theory/` | framework, constants, lexicon, corrections, current essays |
| Why it holds | `formal/` | `theorems/`, `derivations/`, `conjectures/` |
| What was measured | `experiments/` | see below |
| What was published | `papers/` | `series/`, `standalone/`, `legacy/`, `registry/` |
| Lineage | `archive/` | terminal, organised by era |

Inside `experiments/`:

| Location | Purpose | Structure required |
|---|---|---|
| `experiments/milestones/` | the derivation chain, M1–M15 | §2 in full |
| `experiments/sidecars/` | real programs off the main chain | §2 in full |
| `experiments/studies/` | standalone thematic investigations | §2 in full |
| `experiments/spikes/` | exploratory | **none** — exempt by definition |
| `archive/<era>/` | Era 1–2 experiments | as-was; frozen, never retrofitted |

Promotion path is `experiments/spikes/` → `experiments/studies/`, or a new milestone when
the work carries its own thesis. Never create an experiment outside `experiments/`.

`formal/theorems/` **indexes** rather than restates: an entry gives the statement, its
grade, and pointers to the journal that derived it and the experiment that verified it.
A second copy of a proof is a second thing that can drift.

---

## 4. Repository hygiene

- **No summary `.md` files at repo roots.** Use `.changelog/` (§7).
- **Do not hand-edit generated files** — `map.yaml`, and the `files`/`child_directories`
  keys of any `meta.yaml`.
- **Never commit** `.env`, credentials, or API keys.
- **Superseded code is deleted**, not maintained — one concept, one definition. This does
  **not** apply to research artifacts: superseded experiments, papers, and results are
  lineage and are archived, never deleted.
- Renaming or moving an experiment directory **requires** updating the corresponding FDO
  `source_paths` in the same change (§8).

---

## 5. `meta.yaml` — schema v2.1

Replaces the former `schema.yaml` registry, which was removed: it had not been updated
in 14 months, described a `meta_yaml` shape no file in the repo actually used, and was
never read by any validator. Files declaring
`schema_version: "2.0"` remain valid; v2.1 is a superset that names the two zones and adds
the research fields already in use.

### 5.1 Two zones

Scope: the repo root and each experiment root. Nowhere else.

**Generated zone** — owned by `tools/update_meta_yamls.py` and CI. Never hand-edit.

| Field | Type |
|---|---|
| `files` | list of strings |
| `child_directories` | list of strings |

**Authored zone** — owned by humans and agents. CI preserves it (the updater round-trips
the whole document and rewrites only the generated zone).

### 5.2 Required fields

```yaml
schema_version: "2.1"
directory_name: milestone15        # `name` accepted as an alias
description: "What this directory contains."
```

### 5.3 Additionally required for experiment roots

```yaml
title: "Milestone 15: The Representative Problem (DFT-Hodge Boundary)"
status: active                     # §2.2
era: era4-milestone-stack-2026q2   # §2.3
```

Every field listed as required here is enforced as a CI **error** by
`tools/validate_experiment_structure.py`. Nothing is described as required unless it is
actually enforced — a "required" field that only warns is not a standard, it is a wish.

### 5.4 Recommended for experiment roots

```yaml
tags: [milestone15, hodge, holonomy]
created: "2026-06-11"
confidence: 0.35                   # 0.0–1.0
score: "60/112"                    # §2.6
fdo: milestone15-representative-problem   # Lore node id — §8
related_experiments: [milestone13, milestone14]
core_dependencies: [milestone13/core/identity_complement.py]
superseded_by: sec_prime_manifold  # archived experiments only
```

### 5.5 Accepted legacy fields

`semantic_scope`, `proficiency_level`, `estimated_context_weight` are accepted where they
already exist and are not required for new directories.

---

## 6. Journal schema

Supersedes `experiments/milestones/JOURNAL_SCHEMA.md` v1.1 and the journal section of
`experiment-schema.instructions.md`, which differed on heading format.

**Filename:** `journals/YYYY-MM-DD_descriptive_slug.md` — lowercase, underscores, 3–5 words.
One file per day of active work; a major discovery may warrant its own file.

```markdown
# YYYY-MM-DD: Descriptive Title

**Date**: YYYY-MM-DD
**Session**: brief focus area

## Summary
2–5 sentences on outcomes.

## Timeline
### HH:MM — Activity Type
What was done and what was found.
**Status**: ✅ Confirmed | ❌ Failed | 🔄 In Progress | 💡 Insight

## Key Findings
Bulleted or tabular.

## Next Steps
- [ ] Task

## Files Modified
- path/to/file
```

**Activity types:** Setup · Experiment · Analysis · Discovery · Bug Fix · Documentation · Planning

**Rules:** write as you go; record failures (usually the more valuable data); link scripts
and results by path; be explicit about uncertainty; cross-reference related experiments.

---

## 7. Changelog standard

```
.changelog/YYYYMMDD_HHMMSS_brief_slug.md
```

```markdown
# Brief Title

**Date**: YYYY-MM-DD HH:MM
**Commit**: hash (if applicable)
**Type**: engineering | research | documentation | refactor | bugfix | release

## Summary
## Changes
### Added / ### Changed / ### Fixed / ### Removed
## Details
## Related
```

One entry per meaningful unit of work. Record *why*, not only *what*. Do not create entries
for typos or for exploration that led nowhere — unless the learning is worth keeping.

---

## 8. Knowledge graph (Lore)

The graph is the institutional memory layer. Physics FDOs carry trust grades:
`source` · `curated` · `reference` · `legacy`.

**Search before writing.** `lore_search(query, grade?)` before starting on any topic.

**Vault sync is MANDATORY.** After any change to code, experiments, or structure:

1. Identify affected FDOs (by tag, `source_paths`, or the `fdo:` key in `meta.yaml`).
2. Update `source_paths` when files move; update counts, scores, and status when
   experiments change; add log entries to project FDOs.
3. Land the graph update with the change, not later.

**Operational rules:**
- MCP calls to lore are **sequential, never parallel**.
- `lore_update(body=…)` **REPLACES** the body — there is no append mode, and `lore_get`
  truncates at 8KB. Fetch the full body over the API before writing one back, or you will
  silently commit a truncated record.
- `kronos_*` tools are **frozen** — read-only. Never write through them.

**Bidirectional linkage.** An experiment points to its node via `meta.yaml: fdo:`; the node
points back via `source_paths`. Both sides are checked (§9).

---

## 9. Validation

| Check | Tool |
|---|---|
| Experiment structure + meta.yaml schema | `tools/validate_experiment_structure.py` |
| Generated index freshness | `tools/generate_experiment_index.py` |
| Corpus inventory freshness | `tools/generate_inventory.py` |
| Repo tree map | `tools/generate_path.py` → `map.yaml` |
| meta.yaml generated zone | `tools/update_meta_yamls.py` |
| Internal link integrity | `tools/check_links.py` — ceilinged in CI (`--max`), so rot may fall but not rise |
| Citations | `tools/validate_citations.py` |
| FDO `source_paths` resolve | see the `dft-lore-sync` skill (§10) |

`tools/link_checker.py` is superseded by `check_links.py` and is dead — it hardcodes an
absolute path to a directory that does not exist. It survives only because nothing has
deleted it yet.

Structure, index, map, and link checks run in CI on push. A standards violation is a CI
failure, not a style note — that is the mechanism preventing the drift this document was
written to end.

---

## 10. Agent skills

The conventions in this document are only worth writing if something hands them to whoever
is doing the work. Skills are that delivery mechanism.

They live in `dawn-field-theory/.claude/skills/<name>/SKILL.md` and are **tracked**, so
they travel with the repository. `.gitignore` excludes `.claude/` (local settings) with an
explicit `!.claude/skills/` exception; git will not descend into an excluded directory, so
that exception has to un-exclude `.claude` itself before it can re-include `skills`.

| Skill | Covers |
|---|---|
| `dft-experiment` | experiment structure, meta.yaml zones, scoring, pre-registration |
| `dft-repo-gates` | the CI gates and the regeneration order |
| `dft-lore-sync` | vault sync, and the body-replacement / 8KB-truncation traps |
| `dft-publish-integrity` | frozen packages, DOIs, which record family to cite |

### The rule for writing one

**A skill states the rule, the failure it prevents, and the command that verifies it.**
Prose that restates this document is not a skill — an agent already has this document. What
it does not have is the knowledge that `INVENTORY.md` will silently undercount if the
generators run before `git add`, or that `lore_get` truncates at 8KB and `milestone6-planning-seed`
sits one byte under that boundary. Earn the content from a failure that actually happened.

This replaces the previous mechanism. The 35 slash commands under
`core_workspace/.claude/commands/` all loaded their protocol through
`mcp__kronos__kronos_skill_load`, and kronos is retired — every one was a dead pointer.
