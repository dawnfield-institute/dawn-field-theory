# Milestone 3: Energy Equivalence Integration & Methodology Validation

**Date**: 2026-02-17 08:50
**Commit**: (pending)
**Type**: engineering | research

## Summary

Created milestone3 experiment directory to formalize and validate findings from the energy_equivalence exploratory session (Feb 16-17), and to audit the methodology of milestones 1-2 before PACSeries paper consolidation. Updated the PREPRINT_UPDATE_PLAN.md master checklist with 20 new objectives.

## Changes

### Added
- `foundational/experiments/milestone3/` — Complete experiment directory with:
  - `README.md` — Milestone overview, 10-experiment plan in 4 blocks, success criteria, falsification conditions
  - `meta.yaml` — Schema 2.0 metadata with source/target cross-references
  - `SYNTHESIS.md` — Cross-connections to Papers 1-6 and milestones 1-2
  - `FALSIFICATION_REGISTRY.md` — 9 falsification tests (F1-F9), all pending
  - `core/` — Shared module (constants.py, xi_calculator.py, utils.py)
  - `scripts/` — 10 experiment scripts:
    - Block A: exp_01 (Fibonacci memory), exp_02 (ξ accumulation), exp_03 (prime reachability)
    - Block B: exp_04 (w2/w1 ratio), exp_05 (E-I-S oscillator), exp_06 (Θ recycling)
    - Block C: exp_07 (Wilson-Fisher null), exp_08 (Weinberg running), exp_09 (look-elsewhere)
    - Block D: exp_10 (independence audit)
  - `results/` — Empty, awaiting experiment outputs
  - `journals/` — Empty, awaiting research logs

### Changed
- `foundational/docs/preprints/PREPRINT_UPDATE_PLAN.md` — Added 20 new checklist items:
  - Paper 1: 3 items (cascade "why Fibonacci", ξ accumulation, Θ model dependence)
  - Paper 4: 4 items (search methodology, sin²θ_W energy scale, falsification count, milestone3 linkage)
  - Paper 6: 3 items (0.020 Hz mismatch, oscillator validation, GAIA perplexity)
  - Energy Equivalence Integration: 14 items (milestone3 experiments + results integration)

## Details

### Why Milestone 3

The audit of existing experiments revealed several gaps:
1. **Energy equivalence session** produced promising but unvalidated results (cascade dynamics, ξ accumulation, E-I-S resonance)
2. **0.020 Hz mismatch** — simulation gives 0.007 Hz, not the claimed 2/3 ratio
3. **Θ recycling** — 36%-94% efficiency depending on formula choice
4. **Search methodology** — mass ratios found by search-then-validate, not predicted
5. **Independence assumptions** — joint p-values assume cross-domain independence without testing

### Experiment Design Philosophy

Each experiment has:
- Predetermined falsification threshold (set before running)
- Connection to specific paper section
- Error bounds on all quantities
- Honest assessment even when results are unfavorable

### Critical Path

Experiments exp_05 (0.020 Hz) and exp_08 (sin²θ_W running) are blocking: their results determine whether Paper 6 can merge relativistic_mas and whether Paper 4 can claim the Weinberg angle at a specific energy scale.

## Related
- `internal/energy_equivilance/` — Source exploratory session
- `foundational/experiments/milestone1/` — PAC/SEC → SM chain
- `foundational/experiments/milestone2/` — Mass derivation
- `foundational/docs/preprints/PREPRINT_UPDATE_PLAN.md` — Master plan
