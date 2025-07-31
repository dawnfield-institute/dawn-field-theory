# CIP Multi‑Repository & MCP Integration Design (v0.1)

> **Goal:** Define how CIP scales across many repositories and how an **MCP server with CIP tools** enables post‑symbolic, auditably‑navigable knowledge ingestion today (creation later). Repos are split by purpose: **Field Theory**, **DevKit (infodynamics)**, **SDK (infodynamics)**, **CIP Core**, plus **infrastructure** repos. CIP remains its *own* repo and protocol layer.

---

## 1) Scope & Principles

* **Separation of concerns**

  * **Field Theory repo**: theory, models, experiments (domain content).
  * **DevKit (infodynamics)**: developer utilities, templates, generators; *not* CIP‑specific.
  * **SDK (infodynamics)**: programmatic access to domain models/APIs; *not* CIP‑specific.
  * **CIP Core repo**: spec, schemas, validators, reference clients, scorer.
  * **Infrastructure repos**: registries, discovery, deployment, CI, observability.
* **MCP as the copilot bus**: MCP server exposes tools to read/inspect repos with CIP semantics.
* **Post‑symbolic dataset replacement**: Repos become the live, licensed, traceable “dataset.” No offline pretraining requirement to use; agents ingest *on demand*.
* **Auditability end‑to‑end**: Using SCBF, we log from navigation → file access → reasoning steps → (later) model activation fingerprints.
* **Phase focus now**: **Ingestion** only; **Creation** staged for later.

---

## 2) High‑Level Architecture

```
[Clients: Claude Desktop, future ChatGPT Desktop, VS Code, etc.]
          │
          ▼
[MCP Server: "repo-mcp" + CIP Tool Suite]
  ├─ Resources: repo://{repo-id}/{path}
  ├─ Tools: cip.get_meta, cip.get_map, cip.list_meta, cip.search, ...
  ├─ Audit: audit.start/end, audit.log_event
  └─ Policy: allowlist repos, branches, capabilities
          │
          ▼
[Git Layer]
  ├─ Field Theory repo (domain)
  ├─ DevKit (infodynamics)
  ├─ SDK (infodynamics)
  ├─ CIP Core repo (spec, schemas, scorer)
  └─ Infra repos (registry, discovery, CI)
```

**Multi‑repo traversal:** CIP metadata provides cross‑repo links; MCP resolves them and maintains an audit trail.

---

## 3) Repository Roles & Interfaces

### 3.1 Field Theory Repository (domain)

* Domain documents, experiments, models, results.
* Uses **CIP metadata** files for navigability (`.cip/meta.yaml`, versioned `instructions_*.yaml`, per‑dir `meta.yaml`, optional `map.yaml`).

### 3.2 DevKit (infodynamics)

* Code templates, generators, CLI tools for building domain artifacts.
* Not CIP‑specific; can optionally consume CIP metadata for scaffolding.

### 3.3 SDK (infodynamics)

* Programmatic access to domain components (APIs/engines/libraries).
* Not CIP‑specific; can expose helpers to read domain repos.

### 3.4 CIP Core Repository

* **Spec & Schemas**: `.cip` formats, `meta.yaml`, `map.yaml`, validation rules.
* **CIP Scorer**: compares answers to validation ground truth.
* **Reference clients**: example agents, validators.

### 3.5 Infrastructure Repositories

* **Registry/Discovery** of repos.
* **Deployment**: MCP server packaging, Docker, CI workflows.
* **Observability**: audit sinks, SCBF dashboards.

---

## 4) Multi‑Repository Navigation Model

### 4.1 Repo Identity & URIs

* Use normalized IDs and URIs:

  * `repo://{repo-id}/{path}` (fast local access)
  * `git://{host}/{org}/{repo}@{ref}//{path}` (fully qualified, immutable)

**Example**

```txt
repo://field-theory/docs/infodynamics/overview.md
repo://cip-core/.cip/meta.yaml
repo://devkit-generators/README.md
```

### 4.2 Cross‑Repo Links (in `meta.yaml`)

Add a `links:` block referencing concepts or files in other repos:

```yaml
links:
  concepts:
    - id: infodynamics.core
      title: Core Infodynamics Concepts
      href: repo://field-theory/docs/infodynamics/overview.md
  see_also:
    - repo: sdk-infodynamics
      path: api/reference/entropy.md
    - repo: devkit-infodynamics
      path: templates/experiment/scaffold.md
```

### 4.3 Cross‑Repo Map (Registry Repo)

A central **discovery index** to resolve `repo-id → git remote` and trust/policy:

```yaml
repos:
  - id: field-theory
    remote: https://github.com/dawnfield-institute/dawn-field-theory
    default_branch: main
    schema_version: 2.0
    trust: allow
    license: CC-BY-4.0
  - id: cip-core
    remote: https://github.com/dawnfield-institute/cip-core
    default_branch: main
    schema_version: 2.0
    trust: allow
```

MCP loads this registry at startup to know which repos it can traverse.

---

## 5) CIP Tooling in the MCP Server (Ingestion Phase)

> Implement these as MCP **tools** and **resources**. Start read‑only.

### 5.1 Resources

* **`repo://{repo-id}/{path}`** → read file via Git working copy or cached clone.

### 5.2 Tools (initial set)

```json
[
  {
    "name": "cip.get_meta",
    "input": {"repo": "string", "path": "string"},
    "output": {"text": "string"},
    "desc": "Read a meta.yaml at a path; validate against CIP schema."}
  ,
  {
    "name": "cip.get_map",
    "input": {"repo": "string"},
    "output": {"text": "string"},
    "desc": "Read root map.yaml if present; authoritative structure overview."}
  ,
  {
    "name": "cip.list_meta",
    "input": {"repo": "string", "path": "string", "recursive": "boolean"},
    "output": {"items": [{"repo": "string", "path": "string"}]},
    "desc": "List meta.yaml files under a path (optionally recursive)."}
  ,
  {
    "name": "cip.search",
    "input": {"repo": "string", "query": "string", "path": "string"},
    "output": {"matches": [{"path": "string", "line": "number", "text": "string"}]},
    "desc": "Literal search over repo contents (text files)."}
  ,
  {
    "name": "cip.resolve_links",
    "input": {"repo": "string", "meta_path": "string"},
    "output": {"targets": [{"repo": "string", "path": "string"}]},
    "desc": "Read meta.yaml and expand cross‑repo links into concrete targets."}
  ,
  {
    "name": "cip.batch_fetch",
    "input": {"targets": [{"repo": "string", "path": "string"}]},
    "output": {"files": [{"uri": "string", "text": "string"}]},
    "desc": "Fetch many files efficiently for reasoning windows."}
  ,
  {
    "name": "cip.validate",
    "input": {"repo": "string", "path": "string"},
    "output": {"ok": "boolean", "errors": ["string"]},
    "desc": "Schema validation (meta.yaml/map.yaml) using CIP Core schemas."}
]
```

### 5.3 Audit Tools (SCBF‑aligned)

```json
[
  {"name": "audit.start_session", "input": {"repo": "string", "query": "string"}, "output": {"session_id": "string"}},
  {"name": "audit.log_event", "input": {"session_id": "string", "type": "string", "payload": "object"}},
  {"name": "audit.end_session", "input": {"session_id": "string", "status": "string"}}
]
```

Logs: tool invocations, files read, cross‑repo hops, validation calls. Later we extend payloads to include **model activation fingerprints**.

---

## 6) CIP File Schemas (v2.0, ingestion‑focused)

### 6.1 `.cip/meta.yaml` (root of each repo)

```yaml
schema_version: 2.0
instructions_file: .cip/instructions_v2.0.yaml
repo_id: field-theory
entry_points:
  - path: meta.yaml
  - path: map.yaml
links:
  see_also:
    - repo: cip-core
      path: spec/overview.md
```

### 6.2 Per‑directory `meta.yaml`

```yaml
schema_version: 2.0
directory_name: docs/infodynamics
description: >
  Core infodynamics references.
semantic_tags: [infodynamics, entropy, coherence]
files:
  - overview.md
  - glossary.md
child_directories:
  - proofs
links:
  concepts:
    - id: infodynamics.core
      href: repo://field-theory/docs/infodynamics/overview.md
```

### 6.3 Root `map.yaml`

```yaml
field-theory/
  meta.yaml
  map.yaml
  docs/
    meta.yaml
    infodynamics/
      meta.yaml
      overview.md
  models/
    meta.yaml
```

---

## 7) MCP Server Design Notes

* **Transport:** stdio by default (Claude Desktop). Optional SSE/HTTP for future desktop integrations.
* **Git access:** local working copies or on‑demand shallow clones to a cache dir per `repo-id` and `ref`.
* **Permissions:** allowlist `repo-id`, branches/refs; denylist sensitive paths.
* **Read‑only** initially; write tools gated by explicit confirmation and policy.
* **Validation:** embed CIP Core JSON Schemas; fail fast, return actionable errors.
* **Batching:** `cip.batch_fetch` chunks by token budget; returns logical bundles for agent reasoning.

---

## 8) SCBF: Audit Pathway (ingestion phase)

1. **Session start** → issue `audit.start_session(repo, query)`.
2. **Navigation events** → log `cip.get_meta`, `cip.get_map`, `cip.list_meta`, `cip.batch_fetch` with URIs.
3. **Cross‑repo hops** → record source/target repo and link provenance.
4. **Answer emission** → include file citations (URIs + git ref) and session id.
5. *(Later)* **Model activation layer** → attach activation fingerprints/hashes for the context that influenced the answer.

Artifacts feed an **SCBF ledger** for end‑to‑end traceability.

---

## 9) Roadmap

### Phase 0 – Bootstrapping (now)

* Implement MCP **CIP tools** (read‑only).
* Add registry repo & repo IDs.
* Enable multi‑repo reads via `links`.

### Phase 1 – Multi‑Repo Ingestion

* Expand `cip.resolve_links` for concept‑level traversal.
* Add `cip.validate` with schema bundles from CIP Core.
* Wire SCBF audit events (session lifecycle, provenance).

### Phase 2 – Compliance & Observability

* Central audit sink; dashboards for navigation graphs and provenance.
* License checks per repo; policy hints in responses.

### Phase 3 – Creation (staged)

* DevKit/SDK emit **CIP‑compliant repo scaffolds** from specs.
* “Explain‑to‑repo” flows: generate structured knowledge repos with embedded provenance.
* Validation loops via CIP Scorer.

---

## 10) Open Questions

* What minimal cross‑repo link vocabulary do we standardize first (`concept`, `see_also`, `implements`, `derives_from`)?
* How do we encode **immutable refs** (commit SHAs) in citations while still allowing humans to work on `main`?
* Where should the **registry** live (single repo vs. environment‑local config)?
* Which activation features are feasible in the first SCBF integration (hashes, attention maps, attribution scores)?

---

## 11) Acceptance Criteria (v0.1)

* MCP server exposes: `repo://` resource + tools: `cip.get_meta`, `cip.get_map`, `cip.list_meta`, `cip.search`, `cip.resolve_links`, `cip.batch_fetch`, `audit.*`.
* Can traverse from Field Theory → CIP Core via `links` and return files with proper URIs.
* All tool calls are logged with a session id; basic audit report generated.
* Read‑only; no write paths enabled.

---

## 12) Appendix: Example Agent Flow (Claude Desktop)

1. User: “Explain the core infodynamics concept referenced by Field Theory.”
2. Agent:

   * `audit.start_session(...)`
   * `cip.get_meta(repo=field-theory, path=.cip/meta.yaml)` → finds instructions file.
   * `cip.get_map(repo=field-theory)` → confirms paths.
   * `cip.list_meta(repo=field-theory, path=docs/infodynamics, recursive=true)`
   * `cip.batch_fetch([...overview.md, meta.yaml...])`
   * Follow `links` → `repo://cip-core/spec/overview.md` (fetch)
   * Synthesize answer with citations; `audit.end_session(status=ok)`.

> Result: Navigable, traceable, license‑respecting explanation grounded in repos, ready for SCBF auditing.
