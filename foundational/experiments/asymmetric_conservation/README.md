# Asymmetric Conservation and PAC-Native Tensor Architecture

**Status**: ✅ Validated (11 experiments, 2026-01-23)  
**Created**: 2026-01-22  
**Updated**: 2026-01-23  
**Related**: milestone1, oscillation_attractor_dynamics, GAIA

---

## Executive Summary

**MODEL NOT FALSIFIED.** The asymmetric conservation hypothesis passes all 5 falsification tests:
- Δ remains bounded
- Sync ≡ Async final states
- P + A + Δ = C always holds
- Reconciliation clears Δ
- Survives extreme conditions

**Key Demonstration (exp_04):**
- Observer sees ΔA = 15.12 but initial P was only 6.29
- This is an "apparent violation" (ΔA > P)
- Hidden injection of 2.0 during window explains it
- Conservation is intact: C increased from 6.29 to 8.29

**Conclusion:** "Asymmetry is a frame effect, not a violation."

---

## Hypothesis

Conservation in PAC systems is **frame-dependent**: local observers measuring within a window [t₁, t₂] may see apparent asymmetry (ΔA > P(t₁)) without violating conservation, provided injection events I(τ) occurred during the window.

**Core claim**: Conservation is structural (enforced at reconciliation boundaries), not procedural (enforced at every timestep).

---

## Key Concepts

### 1. Asymmetric Conservation

Traditional: `P(t) + A(t) = C` at all times

Asymmetric: `P(t) + A(t) + Δ(t) = C` always, with Δ → 0 at reconciliation boundaries

The Δ buffer holds "unreconciled" actualization until parent nodes process child events.

### 2. PAC-Native Tensors

Node tensor:
```
T_n = [P_n, A_n, Δ_n, θ_n]
```
- P_n: remaining potential
- A_n: actualized value
- Δ_n: unresolved imbalance buffer
- θ_n: collapse threshold

Event tensor:
```
E_{n→p} = [δA, δP, σ]
```
- δA: actualization delta
- δP: potential delta
- σ: event tag (type, depth, symbol)

### 3. Event-Indexed vs Time-Indexed

Time-indexed: `T^{(t)}` - state at timestep t  
Event-indexed: `T^{(α)}` - state after actualization event α

Time becomes a derived statistic, not a primitive coordinate.

### 4. Confluence as Fundamental Operator

Parents don't "step forward"—they receive event tensors from children and reconcile:
```
T_parent ← ⊕_{i ∈ children} E_i
```

---

## Experiments

| Script | Purpose | Status | Key Result |
|--------|---------|--------|------------|
| exp_01_sync_baseline.py | Synchronous PAC (current model) | ✅ | Conservation at every step |
| exp_02_async_events.py | Asynchronous event-driven PAC | ✅ | Order-independent final states |
| exp_03_delta_buffer.py | Δ buffer dynamics and reconciliation | ⚠️ | Concept valid, needs refinement |
| exp_04_frame_asymmetry.py | Measure apparent asymmetry in windows | ✅ | **ΔA > P(t₁) demonstrated** |
| exp_05_xi_from_reconciliation.py | Test if Ξ emerges from delay distribution | ⚠️ | Inconclusive |
| exp_06_gaia_integration.py | Apply to GAIA PACTree | ✅ | V5 proposal validated |
| exp_07_falsification.py | Conditions that would disprove hypothesis | ✅ | **5/5 tests pass** |
| exp_08_poisson_async.py | True async with Poisson timing | ✅ | Conservation holds in continuous time |
| exp_09_cross_domain.py | PAC in Fibonacci, primes, DAGs, epidemics | ✅ | **PAC is domain-agnostic** |
| exp_10_xi_emergence.py | Eigenvalue analysis and Ξ search | ✅ | φ from self-similarity, not eigenvalues |
| exp_11_xi_cv_validation.py | 1 + θ·CV(P) ≈ Ξ hypothesis | ⚠️ | ~3% error, suggestive not exact |

---

## Key Finding: Constant Hierarchy (Jan 2026)

| Constant | Source | Role |
|----------|--------|------|
| φ, 1/φ | PAC alone | Self-similar collapse ratio |
| Ξ = 1 + π/55 | SEC + PAC | Reconciliation threshold at interface |
| λ* = 0.618432 | SEC alone | Prime density threshold |

**φ emerges from PAC's self-similarity constraint**: α/(1-α) = 1/α → α = 1/φ

**Ξ requires SEC+PAC coupling**: It encodes both π (continuous dynamics) and 55 = F₁₀ (Fibonacci structure)

See: `SYNTHESIS.md` for full analysis and `milestone1/SYNTHESIS.md` for integration.

---

## Success Criteria

1. **Δ buffer works**: Local asymmetry permitted, global conservation maintained
2. **Event-indexed = time-indexed**: Same final states, different execution model
3. **Ξ emergence**: Reconciliation delays produce oscillations near 1.0571
4. **GAIA compatibility**: Can retrofit to existing PACTree without breaking tests

---

## Falsification Conditions

1. Δ buffer grows unbounded (conservation actually violated)
2. Event-indexed gives different final state than time-indexed
3. Ξ has no relationship to reconciliation statistics
4. GAIA's synchronous model is mathematically required, not just convenient

---

## Theoretical Foundation

From the core document:

> Conservation is primary. Time is emergent. Forward evolution is an illusion
> created by reconciliation frequency.

This experiment tests whether that claim is computationally verifiable.
