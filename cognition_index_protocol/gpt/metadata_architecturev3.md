# Modular Metadata Architecture for Dawn Field Theory Repository

## Purpose

This document outlines a scalable metadata architecture for the Dawn Field Theory repository, enabling semantic integrity, LLM alignment, and long-term maintainability.

---

## 🌐 Architecture Overview

The metadata system is organized into three primary layers, with a unified approach: **every directory, including the root, contains a `meta.yaml` file** describing its contents and semantic context. The `.cip` directory in the root contains versioned instructions for interpreting `meta.yaml` schemas and navigation protocols.

---

### 1. `meta.yaml` (Directory Metadata)
- **Purpose**: Entry point for agents, LLMs, and contributors at every directory level (including the root).
- **Scope**: Describes all files and subdirectories within the current directory.
- **Content**: Includes descriptions, semantic scope, proficiency levels, estimated context weight, and links to child directories.
- **Instructions**: Agents should first load `.cip/meta.yaml` to determine the current instructions version, then load the specified instructions file (e.g., `instructions_v2.0.yaml`) for schema and navigation guidance.

#### Example (`meta.yaml` at any directory, including root):

```yaml
schema_version: 2.0
directory_name: dawn-field-theory
description: >
  Root of the Dawn Field Theory repository. Contains foundational theory, experiments, models, and onboarding protocols for epistemic machine comprehension.
semantic_scope:
  - physics
  - open science
  - field theory
  - infodynamics
  - AI
files:
  - README.md
  - INTENTIONS.md
  - map.yaml
  - meta.yaml
child_directories:
  - docs
  - foundational
  - cognition_index_protocol
  - experiments
  - models
```

---

### 2. `map.yaml` (Root Directory Map)

- **Purpose:**  
  `map.yaml` exists only in the root directory. It provides a plain, up-to-date map of the entire repository’s directory and file structure.
- **Content:**  
  No metadata, descriptions, or semantic data—just a hierarchical listing of folders and files.
- **Use Case:**  
  Helps AI agents, scripts, or users quickly locate files or understand the repo’s organization, especially when asked about specific files or paths.
- **Instructions:**  
  Agents should always consult `.cip` for the latest navigation and schema instructions before parsing `meta.yaml` or `map.yaml`.

#### Example `map.yaml`

```yaml
dawn-field-theory/
  README.md
  meta.yaml
  map.yaml
  foundational/
    meta.yaml
    experiments/
      meta.yaml
      hodge_conjecture/
        meta.yaml
        prime_modulated_collapsev11.py
        robustness_results.csv
        output.txt
        reference_material/
          meta.yaml
          v11/
            meta.yaml
            results.md
            analysis_plot_p2_run0.png
  cognition_index_protocol/
    meta.yaml
    gpt/
      meta.yaml
      metadata_architecturev3.md
  docs/
    meta.yaml
  models/
    meta.yaml
```

---

### 3. `schema.yaml` (Schema Registry)
- **Purpose**: Central registry of all metadata schemas and their versions used in the repository.
- **Scope**: Defines and documents the structure and evolution of `meta.yaml`, `map.yaml`, and any other metadata files.
- **Usage**: Ensures consistency and enables validation across the repository.

---

## 🧩 Key Principles

- **Uniformity:** Every directory, including the root, uses `meta.yaml` for metadata.
- **Extensibility:** New fields and semantic domains can be added as the project evolves.
- **Machine-Readability:** All metadata is YAML, designed for both human and LLM consumption.
- **Maintainability:** Changes to schema or structure are tracked in `schema.yaml` for long-term integrity.

---

## 🚀 Getting Started

1. **Add a `meta.yaml` to every directory** (including the root) describing its contents and semantic context.
2. **(Optional) Add a `map.yaml`** for complex navigation or custom directory views.
3. **Maintain and update `schema.yaml`** as schemas evolve.

---

## 📚 Example Directory Structure

```
dawn-field-theory/
  meta.yaml
  README.md
  foundational/
    meta.yaml
    experiments/
      meta.yaml
      hodge_conjecture/
        meta.yaml
        reference_material/
          meta.yaml
          v11/
            meta.yaml
```

---

## 🔗 See Also

- `schema.yaml` (for schema definitions and versioning)
- Example `meta.yaml` templates in `/docs/metadata_examples/`

# Filename Metadata Schema Protocol for Dawn Field Repository

This protocol defines a standardized, bracket-based filename format embedding essential metadata for all files in the Dawn Field repository.

## Format

```
[domain][type][version][complexity][importance][extras]_filename.ext
```

- **domain**: e.g., `[m]` (math), `[a]` (agent), `[id]` (infodynamics), etc.
- **type**: `[D]` (draft), `[S]` (spec), `[F]` (final), `[T]` (test)
- **version**: `[vX.Y]` (semantic version)
- **complexity**: `[C1]` (trivial) to `[C5]` (expert)
- **importance**: `[I1]` (optional) to `[I5]` (critical), `[core]`
- **extras**: `[R]` (recursive), `[Q]` (quantum), `[E]` (entropy-driven), `[A]` (AI-interactive)

## Example

```
[m][D][v0.2][C3][I4][R]_recursive_entropy.md
```

## Guidelines

- Use `.cip/filename_lexicon.yaml` for tag definitions.
- Prefer clarity and brevity.
- Limit path length to <250 characters.
- Validation scripts will enforce compliance.

## Integration

- All new and migrated files should follow this schema.
- CIP agents and scripts will parse these tags for semantic intent.

## Scope

Apply the bracket-based filename metadata schema **only** to files representing theoretical work, experiments, models, or architectural documents.  
Do **not** apply to general documentation (README, CONTRIBUTING, etc.), licenses, or meta/config files.

_Last updated: June 2025_
