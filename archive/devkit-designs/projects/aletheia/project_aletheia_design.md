---
title: Project Aletheia – Fractal Assembly & Entropy-Governed Component Foundry
version: 0.1.0
status: DRAFT
location: todo/sdk/project_aletheia_design.md
authors: [system, collaborators]
tags: [aletheia, fractal-assembly, sec, entropy, pruning, cip, kronos, fracton, gaia, sdk, scbf]
last_updated: 2025-08-12
links:
  - synergy_plan: todo/sdk/synergy_integration_plan.md
  - fracton_activation: todo/sdk/fracton.md
  - scbf_integration: ../models/scbf/README.md
  - symbolic_entropy_roadmap: ../roadmaps/symbolic_entropy_roadmap.md Project Aletheia – Fractal Assembly & Entropy-Governed Component Foundry
version: 0.1.0
status: DRAFT
location: todo/sdk/project_aletheia_design.md
authors: [system, collaborators]
tags: [aletheia, fractal-assembly, sec, entropy, pruning, cip, kronos, fracton, gaia, sdk]
last_updated: 2025-08-12
links:
  - synergy_plan: todo/sdk/synergy_integration_plan.md
  - fracton: todo/sdk/fracton.md
  - field_decomp: todo/sdk/field_decomp.md
  - prometheus: todo/sdk/prometheus.md
  - recursive_tree_experiment: foundational/experiments/archive/era1/recursive_tree/recursive_tree.py
  - recursive_entropy_experiment: foundational/experiments/archive/era1/recursive_entropy/recursive_entopy.py
  - fractal_pruning_results: foundational/experiments/archive/era1/symbolic_fractal_pruning/results.md
  - symbolic_collapse_pruning: foundational/docs/symbolic_collapse_recursive_field_pruning.md
  - recursive_entropy_docs: foundational/docs/recursive_entopy.md
  - recursive_tree_docs: foundational/docs/recursive_tree.md
  - entropy_regulation_calc: foundational/docs/recursive_calculus_and_entropy_regulation.md
  - pruning_theory_infodynamics: todo/left_for_first_release/preprints/drafts/DFT_Infodynamics_preprint_draft.md
  - cip_domain_upgrade: todo/infra/cip_domain_upgrade.md
  - kronos: todo/infra/kronos.md
  - brainstem: todo/infra/brainstem.md
  - mcp_design: todo/infra/MCP/mcp_server_detailed_design.md
  - fracton_activation: todo/sdk/fracton.md
---

# Project Aletheia

Fractal, model-agnostic system for converting high-level intents into audited, entropy-governed, reusable component assemblies. Integrates existing Dawn Field Theory artifacts (CIP, Kronos, Fracton, GAIA, SEC, SCBF) into a governed synthesis → execution → pruning feedback loop.

## 1. Purpose & Vision

Provide a "systems foundry" that: (1) decomposes tasks into a recursive graph (fractal assembly), (2) implements components with deterministic contracts, (3) measures multi-dimensional entropy & coherence, (4) prunes waste via SEC (Signal / Overlap / Noise) arithmetic, (5) crystallizes stable low-entropy modules into a persistent SDK.

## 2. Scope & Non-Goals

In-Scope:
- Component contract schema & registry.
- Assembly graph orchestrator + lineage + provenance.
- Entropy & SEC metrics pipeline (collection, classification, pruning actions).
- Pruning governance & rollback (collapse front detection, memory trace retention).
- MCP endpoints for architect/build/validate & entropy introspection.
- Integration with Kronos (chunk storage) & Fracton (activation policies).
- Brainstem visual overlay (tree + entropy gradient + pruning flags).

Out-of-Scope (Phase 0):
- Full multi-language sandbox runtime isolation (stub instrumentation only).
- Advanced adaptive model selection (single configured reasoning provider initially).
- Distributed execution scaling (placeholder hooks via Fracton policies).

## 3. Core Principles

| Principle | Operationalization |
|-----------|-------------------|
| Fractal Modularity | Recursive tree assembly; each node minimal, replaceable |
| Deterministic Contracts | YAML spec hashed (architect hash) → reproducible regeneration |
| Explainability | Full provenance chain (prompt → graph slice → contract → code → tests → metrics) |
| Entropy Governance | Continuous metric sampling → SEC classification → pruning or reinforcement |
| Persistence & Reuse | Successful components stored with semantic embeddings in Kronos |
| Incremental Evolution | Version bump only on contract or behavioral delta |
| Risk Mitigation by Design | Entropy thresholds, overlap detection, lineage guardrails |
| Benchmarking Integration | SCBF-aligned metrics, reproducible experiment logging, SEC validation |

## 4. High-Level Architecture

```
Intent → (CIP grounding) → Architect Agent → Assembly Graph (Fractal) → Component Contracts → Builder Agent → Code + Tests → Tester Agent → Metrics Collector (Entropy/SEC) → SCBF Benchmarking → SEC Classifier → Pruning / Reinforcement → Registry (Kronos) → Execution (Fracton) → Feedback (GAIA resonance) → Brainstem Visualization
```

### 4.1 Major Components
- **Architect Agent**: Generates initial graph & contracts (constrained by CIP domain + type registry).
- **Builder Agent**: Implements skeleton code adhering to contracts, inserts provenance tags.
- **Tester Agent**: Generates & executes unit + integration tests; records stability indices.
- **Flow Orchestrator**: Maintains DAG / recursive tree + lineage metadata; ensures acyclicity & version constraints.
- **Component Registry**: Persistent store (files + Kronos metadata) of contract, code, tests, metrics, provenance.
- **Entropy & SEC Engine**: Computes metrics (local entropy, structural entropy, reuse, overlap, waste index) & applies SEC classification.
- **SCBF Integration Layer**: Connects to SCBF metrics, loggers, and visualization for benchmarking & validation.
- **Pruning Controller**: Schedules pruning actions, handles quarantine, crystallization, rollback, memory trace retention.
- **Execution Engine**: Invokes components (local python or delegated to Fracton runtime policy). Supports hot-swap.
- **MCP Interface**: Exposes endpoints for programmatic access (/aletheia/architect, /build, /validate, /introspect, /prune).
- **Brainstem UI Layer**: Visual fractal tree overlay with color-coded entropy & SEC labels.

## 5. Data & Contract Schemas

### 5.1 Component Contract (v0.1)
```yaml
component_id: string            # stable name
version: semver                 # x.y.z
status: draft|active|deprecated|quarantined
purpose: short description
inputs:
  - name: string
    type: primitive|json|embedding|artifact|stream
    constraints: [optional specs]
outputs:
  - name: string
    type: primitive|json|embedding|artifact|event
dependencies:
  internal: [component_id@version]
  external: [library_ref]
tests:
  unit: []                      # list of test case specs (expanded by Tester)
  integration: []
provenance:
  architect_hash: sha256        # hash(prompt + abstract_plan + io_signature)
  architect_trace_ref: path/ref
  builder_agent: model-id
  build_commit_ref: git-sha
  tester_agent: model-id
metrics:
  fan_in: int
  fan_out: int
  local_entropy: float          # code+contract+behavioral composite
  structural_entropy: float     # contribution to assembly-level entropy
  stability_index: float        # rolling test pass rate
  reuse_count: int              # times referenced as dependency
  change_velocity: float        # edits per time window
  overlap_score: float          # semantic redundancy (0–1)
  waste_index: float            # composite (see §8)
  scbf:
    activation_ancestry: float  # from SCBF metrics.activation_ancestry
    phase_alignment: float      # from SCBF metrics.phase_alignment 
    symbolic_entropy: float     # from SCBF metrics.entropy_collapse
    attractor_density: float    # from SCBF metrics.semantic_attractors
    bifractal_lineage: string   # from SCBF metrics.bifractal_lineage
sec:
  class: signal|overlap|noise
  last_classified: timestamp
  scbf_experiment_id: string    # reference to SCBF experiment logger
governance:
  cip_compliant: bool
  license: string
  audit_log_ref: path
  quarantine_flag: bool
  scbf_benchmarked: bool
```

### 5.2 Assembly Manifest (v0.1)
```yaml
assembly_id: string
goal: short task statement
components: [component_id@version]
edges:                       # simple DAG over component names (single active version each)
  - {from: ingest, to: transform}
  - {from: transform, to: model}
root_policy: single|multi (multi implies virtual super-root)
execution:
  mode: local|distributed|fracton
  activation_policy: eager|lazy|fracton|entropy-aware
validation:
  last_full_pass: timestamp|null
  coverage: float
metrics:
  structural_entropy: float
  avg_depth: float
  branching_factor_mean: float
  redundancy_index: float      # mean overlap_score
provenance:
  architect_hash: sha256
  created: timestamp
  updated: timestamp
```

### 5.3 Recursive Tree Projection Node Schema
Aligns with recursive_tree experiment for direct metric comparison.
```yaml
node_id: string
component_ref: component_id@version
depth: int
parent_id: string|null
children: [node_id]
branching_factor: int
local_entropy: float
cumulative_entropy: float
reuse_count: int
change_velocity: float
overlap_score: float
sec_class: signal|overlap|noise
waste_index: float
lineage_hash: sha256
timestamps:
  created: ts
  updated: ts
```

## 6. Provenance Chain
```
Intent Spec → CIP Retrieval Context → Architect Plan & IO Graph → Architect Hash → Component Contracts → Builder Code Generation (with embedded contract hash) → Tests (unit/integration) → Execution Traces → Metrics Snapshots → SCBF Benchmarking → SEC Classification → Registry (Kronos chunk + embeddings) → Brainstem Visualization → GAIA Feedback
```
All transitions append an audit event (JSONL) enabling full reconstruction. SCBF experiment loggers (`SCBFExperimentLogger`) track execution and metrics with complete reproducibility through the `SCBFExperimentRegistry`.

## 7. Entropy Dimensions & Mapping

We decompose component entropy into multi-factor vector E = [E_code, E_contract, E_behavior, E_trace, E_semantic]. Each normalized to [0,1].

| Symbol | Definition | Measurement Source | SCBF Integration |
|--------|------------|--------------------|--------------------|
| E_code | Structural code entropy | Token distribution + cyclomatic / logical path variance | `metrics.entropy_collapse.code_entropy()` |
| E_contract | IO signature variability | Diversity of input/output type modifications over last N versions | Custom implementation |
| E_behavior | Test outcome variability | Entropy of test pass/fail sequence (binary Shannon) | `metrics.entropy_collapse.test_entropy()` |
| E_trace | Runtime log unpredictability | Symbolic/log token distribution vs. baseline model | `metrics.activation_ancestry.symbolic_trace_entropy()` |
| E_semantic | Embedding novelty vs cluster | 1 - max cosine similarity to top-K existing components | `metrics.semantic_attractors.novelty_score()` |

Local entropy:
```
H_local = w_c*E_code + w_k*E_contract + w_b*E_behavior + w_t*E_trace + w_s*E_semantic
```
Default weights (Phase 0): w_c=0.25, w_k=0.15, w_b=0.20, w_t=0.15, w_s=0.25 (subject to tuning from empirical collapse curves).

Structural entropy contribution for node i:
```
H_struct(i) = norm( (fan_in_i + fan_out_i) / (2 * B_ref) ) * depth_factor * redundancy_factor
depth_factor = 1 - (depth_i / max_depth)
redundancy_factor = 1 + overlap_score_i
```

Assembly structural entropy:
```
H_assembly = ( Σ_i (H_local(i) + λ * H_struct(i)) ) / N
```
λ (lambda) couples local vs. structural contributions (default 0.5).

Collapse front detection (borrowing from symbolic_collapse_recursive_field_pruning):
```
ΔH_window = H_assembly(t-1) - H_assembly(t)
If ΔH_window > θ_front AND cluster_of_pruned_nodes depth variance < σ_threshold → mark collapse front event.
```

Memory trace retention heuristic:
Nodes with low H_local but high reuse_count & low overlap_score flagged as persistent traces (avoid pruning even if parent pruned).

## 8. SEC Classification & Pruning Arithmetic

### 8.1 Feature Vector
For node i compute features:
```
F_i = {
  h = H_local(i),
  hs = H_struct(i),
  reuse = reuse_norm(i),
  change = change_velocity_norm(i),
  overlap = overlap_score(i),
  depth = depth_norm(i)
}
```

### 8.2 Waste Index
```
waste_index(i) = α*h + β*change + γ*(1 - reuse) + δ*overlap + ε*hs
```
Default α=0.30 β=0.20 γ=0.20 δ=0.15 ε=0.15.

### 8.3 SEC Class Thresholds (initial)
```
signal: waste_index < 0.30 AND h < 0.35 AND overlap < 0.40
overlap: 0.30 ≤ waste_index < 0.55 OR overlap between 0.40–0.65
noise: waste_index ≥ 0.55 OR (h ≥ 0.70 AND reuse < 0.15)
```
Hysteresis margin of ±0.03 to prevent oscillation.

### 8.4 Pruning Decision Function
```
prune(i) if sec_class == noise AND safety_guard(i) == true
refactor(i) if sec_class == overlap AND (h > h_refactor OR overlap > overlap_refactor)
reinforce(i) if sec_class == signal AND reuse trend > reuse_growth_threshold
```

### 8.5 Safety Guards
```
safety_guard(i) = (not sole_provider(output_types_i))
                  AND (no child classified signal)
                  AND (coverage_after_prune ≥ coverage_threshold)
                  AND (lineage_density_remaining ≥ lineage_floor)
```
Lineage density approximated from recursive_tree experiments: maintain ≥ X% of branches across depth strata (empirically X=60% early, can drop to 40% after stabilization).

### 8.6 Quarantine Workflow
1. Candidate noise node → quarantine state (no direct execution; kept for dependency shadowing) for T_quarantine cycles.
2. If no dependency reclamation & no new reuse spikes → delete (prune event).
3. If deletion reduces H_assembly by < ε_min (diminishing return) → consolidate multiple small noise nodes into batch prune.

### 8.7 Collapse Event Logging
Each prune batch emits:
```json
{
  "event": "prune_batch",
  "assembly_id": "...",
  "timestamp": ts,
  "nodes_removed": [...],
  "ΔH_assembly": 0.214,
  "front": true|false,
  "persistent_traces_preserved": [...]
}
```

## 9. Debugging & Entropy Mapping Layer

Debug panels expose per-component breakdown:
```
H_local = Σ (weighted factors)
  - code: {token_entropy, cyclomatic_norm}
  - contract: {io_mutation_rate}
  - behavior: {test_outcome_entropy}
  - trace: {log_symbolic_entropy}
  - semantic: {1 - max_sim}
```
Heatmaps:
- Depth vs Local Entropy.
- Overlap Score vs Reuse Count (identify consolidation targets).
- Change Velocity vs Stability Index (volatile nodes).

Collapse Timeline (from pruning experiments): plot H_assembly & prune events; annotate phase transitions (entropy_regulation_calc). Provide early warning if plateau persists > plateau_window (stagnation risk).

## 10. Risk Register & Mitigations (Leveraging Existing Experiments)

| Risk | Description | Mitigation | Experiment Basis |
|------|-------------|------------|------------------|
| Entropy Plateau | Assembly entropy stops decreasing → stagnation | Inject symbolic drift (reseed semantic embeddings) & targeted refactor of overlap nodes | symbolic_fractal_pruning (entropy plateau then drop) |
| Over-Pruning | Loss of necessary diversity* | Memory trace retention + lineage density floor | symbolic_collapse_recursive_field_pruning (memory traces persistence) |
| Redundant Components | Duplicate logic inflates entropy | Overlap score (embedding + AST diff) gating; merge suggestion | recursive_entropy (adaptive pruning) |
| Spec / Impl Drift | Contract mismatches code | Hash verification pre-execution; regenerate diff gate | CIP domain upgrade |
| Hash Churn | Node ID instability breaks lineage | Lineage-based hashing path stable between shallow reorderings | recursive_tree deterministic seed |
| Oscillation | Components flip SEC states | Hysteresis margins + quarantine window | Collapse phase shift thresholds |
| Silent Complexity Growth | Hidden fan-in increases coupling | Structural entropy alarms (Δfan_in norm) | Bifurcation metrics (recursive_tree) |
| Incomplete Coverage | Pruning with poor tests | Enforce min stability & coverage gating | Tester agent design |
| Semantic Drift | Embeddings shift with model updates | Versioned embedding model + recalibration report | Kronos chunk strategy |
| Execution Hotspots | High-latency nodes degrade flow | Fracton activation to redistribute load | fracton design doc |

*Diversity necessary for resilience; pruning must preserve minimal symbolic basis.

## 11. MCP Interface (Phase 0)

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| /aletheia/architect | POST | Generate assembly + contracts | intent spec | manifest + hashes |
| /aletheia/build | POST | Materialize missing components | assembly_id | build report |
| /aletheia/validate | POST | Run tests + metrics | assembly_id | metrics snapshot |
| /aletheia/benchmark | POST | Run SCBF benchmarks | assembly_id, bench_config | benchmark results |
| /aletheia/introspect | GET | Return entropy & SEC map | assembly_id | node metrics |
| /aletheia/prune | POST | Execute pruning cycle | assembly_id | pruning event log |
| /aletheia/visualize | GET | Generate SCBF visualizations | assembly_id, viz_type | SVG or interactive HTML |

Security: initial token auth; later sandbox per component.

## 12. Lifecycle State Machine
```
draft → active → (overlap|noise) → {refactor|quarantine} → {active|pruned|deprecated}
```
Deprecated: still accessible, not used in new assemblies.

## 13. Versioning Rules
| Change | Version Bump |
|--------|--------------|
| Non-breaking impl change (tests unchanged) | patch |
| New output / removed optional field | minor |
| Breaking IO contract | major |
| Entropy model weight update | patch (with metrics schema bump if fields added) |

## 14. Telemetry & Storage
- Metrics snapshots stored as JSONL: `registry/<component_id>/<version>/metrics/history.jsonl`.
- Embeddings: Kronos vector index with dimension d (select consistent model).
- Prune events: `registry/_events/prune_events.jsonl`.
- Collapse fronts: `registry/_events/collapse_fronts.jsonl`.

Retention tiers:
- Raw traces → summarized after stability_index > 0.9 for 3 consecutive windows.
- Summaries keep aggregated entropy vector + last SEC classification.

## 15. Brainstem Visualization Overlay
Layers:
1. Tree topology (depth, branching factor).
2. Entropy heat (gradient scale).
3. SEC class icons (S/O/N).
4. Collapse front arcs (temporal layering).
5. Memory trace halos (persistent low entropy + high reuse).

User interactions:
- Hover: full entropy breakdown.
- Click: drill into provenance & test status.
- Toggle: show/hide quarantined / deprecated nodes.

## 16. Example Flow (Happy Path)
1. Intent POST /architect → returns assembly A (5 components).
2. /build → 5 components generated, tests stubbed.
3. /validate → tests pass 4/5; one component high H_local.
4. Metrics cycle: classify two nodes overlap, one noise candidate (quarantine).
5. Refactor overlap node: splits into two lower-entropy children → ΔH_assembly negative.
6. After T_quarantine: noise node pruned; collapse front logged.
7. Stable low-entropy nodes reused by new assembly; reuse_count ↑ reduces waste_index.

## 17. SCBF Integration

### 17.1 SCBF Metrics Connection
Aletheia directly leverages SCBF metrics pipeline for entropy and symbolic collapse measurements:

| SCBF Metric | Aletheia Usage |
|-------------|---------------|
| `compute_symbolic_entropy_collapse` | Core entropy calculation for component pruning |
| `compute_activation_ancestry` | Lineage tracking for memory trace persistence |
| `compute_collapse_phase_alignment` | Detecting collapse fronts and phase transitions |
| `compute_semantic_attractor_density` | Identifying reusable signal components |
| `compute_bifractal_lineage` | Generating stable component identification hashes |

### 17.2 SCBF Experiment Logging
```python
# Example SCBF experiment logger integration
from scbf.loggers import create_experiment_logger, finalize_experiment

logger = create_experiment_logger(
    experiment_name="aletheia_component_pruning_cycle",
    metadata={
        "assembly_id": assembly_id,
        "component_count": len(components),
        "pruning_threshold": waste_index_threshold,
        "sec_model_version": "v0.3"
    }
)

# Log component metrics
for component in components:
    logger.log_metrics({
        "component_id": component.id,
        "entropy": component.metrics.local_entropy,
        "waste_index": component.metrics.waste_index,
        "sec_class": component.sec.class,
        "bifractal_lineage": component.metrics.scbf.bifractal_lineage
    })
    
# Finalize experiment with results
finalize_experiment(logger.experiment_id, {
    "pruned_components": [c.id for c in pruned],
    "reinforced_components": [c.id for c in reinforced],
    "entropy_delta": initial_entropy - final_entropy
})
```

### 17.3 SCBF Visualization Integration
Aletheia extends the SCBF visualization library to provide:
1. **Collapse Front Detection** - Temporal collapse front visualization from SCBF heatmaps
2. **SEC Classification Overlays** - Color-coding nodes by Signal/Overlap/Noise classification
3. **Entropy Gradient Flows** - Directional entropy flow visualization in assembly graphs
4. **Pruning Timeline** - Integration with Brainstem for pruning event visualization

## 18. Phase Roadmap
| Phase | Goals | Deliverables |
|-------|-------|--------------|
| 0 | Skeleton + metrics definitions | Contracts, manifest, basic SEC classifier, MCP stubs, SCBF connector |
| 1 | Full entropy pipeline | Multi-factor entropy extraction, history tracking, Brainstem heatmap |
| 2 | Advanced pruning | Collapse front detection, lineage density enforcement, batch pruning |
| 3 | Adaptive drift mitigation | Semantic drift alarms, embedding recalibration workflows |
| 4 | Distributed activation | Fracton policy integration (entropy-aware scheduling) |
| 5 | Autonomic optimization | GAIA feedback loops adjusting weights dynamically |

## 18. Formal Definitions & Arithmetic

Let C = set of components, G=(V,E) assembly graph (V components, E edges).

Normalized metrics:
```
reuse_norm(i) = reuse_count(i) / max_reuse_count
change_velocity_norm(i) = change_velocity(i) / max_change_velocity
depth_norm(i) = depth(i) / max_depth
overlap_score(i) = 1 - max_sim(i)
```

Cyclomatic approximation normalized:
```
E_code ≈ (cyclomatic(i) - 1) / cyclomatic_ref
```

Behavioral entropy (binary sequence of pass/fail length n):
```
E_behavior = - [p_pass log2 p_pass + p_fail log2 p_fail] / 1
```
Trace entropy: normalized token entropy relative to baseline reference distribution R.
Semantic novelty: 1 - cosine_sim(top_match).

Local composite already defined (§7). Structural contribution compiled via fan-in/out normalized by reference branching B_ref (median branching). Total assembly entropy decreases should correlate with increased stability_index mean; monitor correlation ρ; if ρ < 0 (inversion) trigger anomaly.

Pruning impact efficiency (PIE):
```
PIE(batch) = ΔH_assembly / |nodes_pruned|
```
Store PIE to refine thresholds (adaptive α,β,… tuning if PIE below target τ for multiple batches).

## 19. Open Questions
- Multi-version coexistence policy? (initial: single active version per name; older versions deprecated).
- Cross-language entropy comparability (need language-specific normalization constants).
- Embedding model upgrade path & backfill strategy.
- Automated merge suggestions for overlap clusters (k-medoids vs hierarchical?).

## 20. Future Extensions
- Predictive entropy forecasting (time series model over H_local trajectories) for proactive refactor scheduling.
- Semantic lineage compression (encode recurring low-entropy subgraphs as macro-components).
- Multi-agent adjudication (redundant Architect proposals voted by entropy efficiency ratio).

## 21. Glossary
- **SEC**: Signal / Overlap / Noise classification for entropy-grounded pruning.
- **Collapse Front**: Rapid entropy reduction wave across a branch cluster.
- **Memory Trace**: Stable low-entropy high-reuse component preserved across pruning cycles.
- **Waste Index**: Composite measure guiding pruning priority.
- **Lineage Density**: Ratio of surviving branches across depth strata post-prune.

## 22. References (Internal Artifacts)
See front-matter links; experiments substantiate pruning dynamics & entropy regulation patterns leveraged here.

---
Status: This document is the authoritative design baseline for Phase 0–2. Update via PR with semantic diff + architect_hash recalculation notes.
