
# Modular Metadata Architecture for Dawn Field Theory Repository

## Purpose

This document outlines a scalable metadata architecture for the Dawn Field Theory repository, enabling semantic integrity, LLM alignment, and long-term maintainability.

---

## 🌐 Architecture Overview

The metadata system is organized into three primary layers:

### 1. `repo.yaml`
- **Purpose**: Entry point for agents, GPTs, and contributors.
- **Scope**: Describes all root-level files and directories.
- **Content**: Includes descriptions, semantic scope, proficiency levels, and estimated context weight for each item.

### 2. `map.yaml`
- **Purpose**: Pure directory tree mapping (static).
- **Scope**: Reflects the current file/folder structure.
- **Use Case**: Fast reference and structural validation.
- **Auto-updated**: Can be regenerated on push via script.

### 3. `meta.yaml` (One per directory)
- **Purpose**: Contextual metadata for each directory.
- **Scope**:
  - `directory_name`: Name of the directory
  - `description`: Purpose and contents
  - `semantic_scope`: Tags like [entropy, collapse, symbolic]
  - `files`: List of files
  - `child_directories`: List of subfolders
  - Optional: `breadcrumbs`, `priority_tier`, `parent_directory`
- **Schema Validation**: Tied to `schema_version` (see `/cognition_index_protocol/schema/`)

---

## ✅ Validation Workflow

1. **File Generation**
   - `map.yaml` is auto-generated on push or via `generate_map.py`.

2. **Validation**
   - `validate_meta.py` checks:
     - All folders in `map.yaml` have a `meta.yaml`
     - All `meta.yaml`s match real directory contents
     - Schema conformity

3. **CI Integration**
   - Optional GitHub Action can reject push if:
     - `meta.yaml` is missing or invalid
     - `schema_version` is inconsistent

---

## 📁 Recommended Directory Structure

```
repo.yaml
map.yaml
/docs/meta.yaml
/experiments/meta.yaml
/experiments/pi_harmonics/meta.yaml
/models/CIMM/meta.yaml
...
/cognition_index_protocol/schema/meta_schema_v1.yaml
/cognition_index_protocol/schema/meta_schema_v2.yaml
```

---

## 🧪 Schema Example (`meta_schema_v2.yaml`)

```yaml
schema_version: 2.0
required_fields:
  - directory_name
  - description
  - semantic_scope
  - files
  - child_directories
optional_fields:
  - parent_directory
  - breadcrumbs
  - priority_tier
field_constraints:
  files: list
  semantic_scope: list
  priority_tier: int
```

---

## 🛡️ Benefits

- ✅ Reduces token bloat in LLM queries
- ✅ Allows dynamic, contextual ingest
- ✅ Supports modular editing and scalability
- ✅ Enables recursive, self-validating navigation
- ✅ Positions the repo for future automation

---

## 📌 Contributor Note

All contributors must:
- Update the appropriate `meta.yaml` when modifying directory content
- Use `validate_meta.py` before committing
- Adhere to current `schema_version` as declared in each file

---

**Version**: 1.0  
**Maintainer**: [You]  
**Last Updated**: 2025-06-18
