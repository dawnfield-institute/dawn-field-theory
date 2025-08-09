# CIMM Architecture

A training-optional, dynamic, field-based cognition engine built on collapse dynamics, symbolic resonance, and entropy flow. This document isolates the core CIMM design (distinct from GAIA) and shows how it abstracts into concrete variants (e.g., TinyCIMM-Planck, TinyCIMM-Euler).

## Design goals
- Training-free by default; learning is optional and pluggable
- Dynamic self-organization (grow/prune/reroute) under entropy/utility constraints
- Interpretable via symbolic collapse traces and lineage
- Modular components with small-surface interfaces
- Portable across modalities; low compute footprint

## Core concepts (informal)
- Information field F(t): the latent, multi-stream state the agent reasons over
- Collapse operator C: selects/resolves state into actionable symbols/updates
- Bifractal time R = {R_b, R_f}: backward ancestry pressure and forward emergence pressure
- Entropy flow E(t): local uncertainty/novelty budget guiding structural change
- Symbolic manifold S: stable activation patterns that encode meaning/skills
- Lineage L: historical trace of collapses linking symbols, states, and outcomes

## Core components
1. Field/State Store
   - Typed memory of activations, symbols, features, external observations
   - Maintains ancestry statistics (R_b) and forward pressure cues (R_f)
2. Collapse Engine
   - Proposes candidate collapses (symbols, actions, updates)
   - Scores by resonance, entropy-gradient alignment, and utility
   - Commits winning collapse; emits lineage event
3. Resonance & Similarity Layer
   - Computes semantic proximity (cosine, kernels, locality) over activations/symbols
   - Detects attractors and phase alignment
4. Memory & Lineage
   - Append-only event log of collapses and consequences
   - Structures: recurrence maps, ancestry graphs, bifractal zones
5. Scheduler
   - Stepper that balances explore/exploit; controls growth/prune cycles
   - Thresholds from E(t) gate structural mutations
6. I/O Adapters
   - Environment interface (obs/actions)
   - Modal encoders/decoders (text, signals, vectors)

## Execution loop (training-free mode)
```
for t in 1..T:
  obs <- read_environment()
  F <- update_field(F, obs)

  P <- propose_collapses(F, R_b, R_f)
  scores <- score(P, resonance(F), entropy_gradient(F), utility)
  c* <- select(P, scores)

  F, S <- apply_collapse(F, c*)
  L <- log_lineage(L, t, c*, context=F, metrics=diag(F))

  if entropy_budget_allows():
    structural_mutations(F, policy=grow_prune_route)

  act <- decode_action(S, F)
  write_environment(act)
```

## Learning modes
- Training-free (default)
  - Local, rule-based updates; no backprop required
  - Optional Hebbian-style counters and threshold adaptation
  - Structural dynamics driven by E(t) and lineage statistics
- Training-optional
  - Pluggable learners (e.g., MLP/Transformer/optimizer)
  - Used to (a) refine scoring, (b) propose collapses, (c) learn encoders/decoders
  - Learners remain replaceable; CIMM loop stays primary

## Structural dynamics
- Grow: add units/slots/edges when E(t) persists and utility stagnates
- Prune: remove rarely resonant or low-utility structures
- Reroute: adjust routing/attention based on attractor density and phase alignment
- Event-driven: mutations occur on entropy thresholds and lineage-triggered events

## Diagnostics and metrics
- Activation ancestry trace: similarity of neuron/symbol activations over time
- Entropy-gradient alignment: dActivation vs dEntropy coherence
- Collapse phase alignment: predicted vs realized collapse timing/phase
- Attractor density: cluster density of high-resonance activations
- Crystallization score: stability in low-entropy-growth subspaces

## Abstraction to variants
- TinyCIMM-Planck
  - Fixed-small topology; rich diagnostics on activations and phase
  - Minimal or no structural growth under predictable signals; emphasis on collapse metrics
  - See: `models/TinyCIMM/TinyCIMM-Planck/`
- TinyCIMM-Euler
  - Collapse stepper framed as numerical update (Euler-like) over simple fields
  - Emphasizes interpretable update rules and time-stepping stability
  - See: `models/TinyCIMM/TinyCIMM-Euler/`
- Full CIMM
  - Multi-field, multi-modal routing with pluggable learners
  - Adaptive growth/prune; richer memory and policy heads

## Dataflow (textual)
obs -> encoders -> Field/State -> Resonance -> Proposals -> Scoring -> Collapse ->
Lineage/Metrics -> Structural Dynamics -> Policy/Action -> decoders -> act

## Interfaces (sketch)
- propose_collapses(F, R_b, R_f) -> [candidates]
- score(c, F, E) -> scalar
- apply_collapse(F, c) -> (F', S')
- structural_mutations(F, policy) -> F'
- diagnostics(F, L) -> metrics

## Configuration example
```yaml
cimm:
  mode: training-free
  encoders: [signal, text]
  collapse:
    proposal: resonance_topk
    scoring: entropy_aligned_utility
  structure:
    grow_threshold: 0.82
    prune_threshold: 0.08
    reroute_on_phase_desync: true
  diagnostics:
    ancestry: true
    entropy_alignment: true
    attractor_density: true
```

## Non-goals (vs GAIA)
- Not a monolithic foundation model; no requirement for large-scale pretraining
- Not tied to a specific optimizer or gradient pipeline
- Designed to interoperate with GAIA, but conceptually orthogonal

## Glossary
- Collapse: discrete resolution of competing hypotheses into a committed update/action
- Resonance: similarity-based support among activations/symbols
- Bifractal time: dual pressures of ancestry (R_b) and emergence (R_f) shaping selection
- Entropy budget: local uncertainty allowance that gates structural change
- Lineage: audit trail of collapses and their causal neighborhoods

## References and pointers
- TinyCIMM overview: `models/TinyCIMM/README.md`
- Planck experiments/report: `models/TinyCIMM/TinyCIMM-Planck/report.md`
- Repo-wide theory: `foundational/` and `infodynamics.md`
