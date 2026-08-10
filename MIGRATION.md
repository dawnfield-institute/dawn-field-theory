# Migration Map — August 2026

Where everything went when the repository was reorganized by layer.

If you have an old link, a bookmark, a citation, or a memory of where something lived,
this is the lookup table. **Nothing was deleted.** 9147 files were relocated; every one
has a home.

## Why the tree changed

The theory had four layers and nothing connected them: what is claimed, why it holds,
what was measured, what was published. They lived in four separate trees with four naming
schemes and no path between them, so there was no route from a claim to its proof to its
test to its paper. The tree is now the argument.

| Layer | Directory |
|---|---|
| What is claimed | [`theory/`](theory/) |
| Why it holds | [`formal/`](formal/) |
| What was measured | [`experiments/`](experiments/) |
| What was published | [`papers/`](papers/) |
| Lineage | [`archive/`](archive/) |

`foundational/` no longer exists. In a physics repository everything is foundational; the
wrapper carried no information and cost a level of depth on every path.

## Lookup

### Experiments

| Was | Now |
|---|---|
| `foundational/experiments/milestone1` … `milestone15` | `experiments/milestones/milestone1` … `milestone15` |
| `foundational/experiments/milestone-r`, `midnight` | `experiments/sidecars/` |
| `foundational/experiments/<anything else>` | `experiments/studies/` |
| `spikes/` | `experiments/spikes/` |
| `foundational/experiments/archive/era1/…` | `archive/era1-symbolic/…` |
| `foundational/experiments/archive/era2/…` | `archive/era2-prefield/…` |

### Documents

| Was | Now |
|---|---|
| `dawn-field-theory.md`, `infodynamics.md`, `origin_of_infodynamics.md`, `for_ai_labs.md` | `theory/` |
| `EPISTEMIC_CORRECTIONS_REGISTRY.md` | `theory/corrections.md` |
| `foundational/lexicon.yaml` | `theory/lexicon.yaml` |
| `foundational/docs/` essays, Era 3–4 | `theory/essays/` |
| `foundational/docs/` essays, Era 1 | `archive/era1-symbolic/essays/` |
| `foundational/docs/` essays, Era 2 | `archive/era2-prefield/essays/` |
| `foundational/docs/empirical_alignment/`, `bridges/`, `docx/` | `archive/era1-symbolic/` |

Documents also **lost their CIP bracket prefixes** — `[m][F][v1.0][C4][I5][E]_name.md`
became `name.md`. The full 51-row old→new table is in
[`theory/essays/README.md`](theory/essays/README.md), along with why the ratings were not
carried forward.

### The formal layer

| Was | Now |
|---|---|
| `foundational/arithmetic/macro_emergence_dynamics/proofs/` | `formal/conjectures/med-proofs/` |
| `foundational/arithmetic/macro_emergence_dynamics/formal_papers/`, `notes/` | `formal/conjectures/` |
| `foundational/arithmetic/constants_derivation_lineage.md` and the other root `.md` | `formal/derivations/` |
| `foundational/arithmetic/euclidean_distance_validation/` | `experiments/studies/euclidean_distance_validation/` |
| `foundational/arithmetic/macro_emergence_dynamics/` (the code) | `experiments/studies/macro_emergence_dynamics/` |
| `foundational/arithmetic/PACEngine/` | `archive/era2-prefield/PACEngine/` |

`arithmetic/` split three ways because it was three things wearing one name: a formal
layer, an experiment, and an engine nothing imported. The `proofs/` directory became
`conjectures/` because every document in it describes itself internally as a *"Conjecture"*
or *"Computational Investigation"* — see
[`formal/conjectures/README.md`](formal/conjectures/README.md).

### Papers

| Was | Now |
|---|---|
| `foundational/docs/preprints/PACSeries/` | `papers/series/PACSeries/` |
| `foundational/docs/preprints/legacy/` | `papers/legacy/` |
| `foundational/docs/preprints/<paper>/` | `papers/standalone/<paper>/` |
| `foundational/docs/preprints/{pdfs,tex_sources,resources}/` | `papers/` |
| `citations/doi_registry.yaml`, `resources/publications_registry.yaml`, `ZENODO_REGISTRY.yaml` | `papers/registry/` |

**Published package internals are untouched.** Anything carrying a DOI is a frozen
artifact — the snapshot is what makes the DOI reproducible. Zenodo records point at Zenodo,
not at repository paths, so no published citation broke.

### Other

| Was | Now |
|---|---|
| `devkit/` | `archive/devkit-designs/` |
| `todo/` | `archive/todo-2025/` |
| `blueprints/` | `archive/blueprints/` |
| `spikes/darkmatter_SEC_WIP/` | `archive/spike-darkmatter-sec/` |
| `foundational/legacy_docs_archive/` | `archive/legacy-docs/` |
| `citations/pending`, `processed`, and their 2 workflows | `archive/citation-pipeline/` |
| `LICENSE_APPENDIX.md`, `REGISTRY_TERMS.md` | `legal/` — the latter renamed `TRADE_SECRET_REGISTRY_TEMPLATE.md`, since it is a licensing contract and the old name was a trap for anyone grepping for terminology |
| `REPOSITORY_RESTRUCTURE_PLAN.md` | `archive/` |

## Removed, not moved

Four things were genuinely deleted, all superseded infrastructure with no content:

| | |
|---|---|
| `mcp/` | Hardcoded the pre-split repo path and loaded a CIP file deleted in Feb 2026; every response returned "(CIP instruction unavailable)". Nothing depended on it. |
| `models/` | One `meta.yaml` declaring a `GAIA/` directory that does not exist. |
| `schema.yaml` | Schema registry, unchanged in 14 months, describing a `meta.yaml` shape no file used and no validator read. Replaced by [`STANDARDS.md`](STANDARDS.md) §5. |
| 429 subdirectory `meta.yaml` | CIP-era artifacts holding only `files` and `child_directories` — a restatement of what `map.yaml` generates. Experiment-root `meta.yaml` remain and carry the authored metadata. |

All four remain in git history.

## Verifying this yourself

```bash
git log --follow -- <new/path>        # full history through the move
git diff --summary main...HEAD        # renames, not deletions
```
