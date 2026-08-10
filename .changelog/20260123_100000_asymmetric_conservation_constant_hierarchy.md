# Asymmetric Conservation Deep Dive: Constant Hierarchy Discovery

**Date**: 2026-01-23 10:00
**Commit**: (pending)
**Type**: research

## Summary
Extended asymmetric_conservation experiment with 4 new experiments (exp_08-11) investigating true async PAC, cross-domain patterns, and Ξ emergence. Key finding: φ emerges from PAC alone (self-similarity), while Ξ requires SEC+PAC coupling. Updated milestone1 to document the constant hierarchy.

## Changes

### Added
- `asymmetric_conservation/scripts/exp_08_poisson_async.py` - True async with Poisson timing
- `asymmetric_conservation/scripts/exp_09_cross_domain.py` - PAC in Fibonacci, primes, DAGs, epidemics
- `asymmetric_conservation/scripts/exp_10_xi_emergence.py` - Eigenvalue and statistical Ξ investigation
- `asymmetric_conservation/scripts/exp_11_xi_cv_validation.py` - 1 + θ·CV(P) ≈ Ξ validation
- `asymmetric_conservation/journals/2026-01-22_deep_dive_pac_dynamics.md` - Session journal
- `milestone1/SYNTHESIS.md` - New "Constant Hierarchy" section

### Changed
- `milestone1/README.md` - v1.2.0 with constant hierarchy, reference to asymmetric_conservation
- `asymmetric_conservation/SYNTHESIS.md` - Corrected eigenvalue interpretation, added Jan 2026 findings

## Details

### Key Discoveries

1. **PAC is domain-agnostic**: The pattern P + A + Δ = C appears in:
   - Fibonacci value flow
   - Prime number sequences (gaps as Δ buffer)
   - Random DAGs (multi-path value flow)
   - Network epidemics (SIS/SIR dynamics)

2. **φ from self-similarity, not eigenvalues**: The eigenvalue result (-1/φ for all tree sizes) is trivially true for any collapse ratio α. The real significance of φ is the **self-similarity constraint**: α/(1-α) = 1/α gives α = 1/φ.

3. **Ξ requires SEC+PAC coupling**: Ξ = 1 + π/55 encodes:
   - π (continuous dynamics from SEC)
   - 55 = F₁₀ (Fibonacci from PAC)
   This means Ξ marks the interface between information-entropy dynamics and value conservation.

4. **Constant hierarchy established**:
   | Constant | Source | Role |
   |----------|--------|------|
   | φ, 1/φ | PAC alone | Collapse ratio |
   | Ξ | SEC + PAC | Reconciliation threshold |
   | λ* | SEC alone | Prime density threshold |

### Experimental Results

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| exp_08 | ✅ | Poisson timing works, conservation holds |
| exp_09 | ✅ | PAC appears in 4 unrelated domains |
| exp_10 | ✅ | φ is self-similarity, not eigenvalue magic |
| exp_11 | ⚠️ | 1 + θ·CV(P) ≈ Ξ but ~3% error |

### Implications for Theory

The separation of constants suggests Dawn Field Theory has layered structure:
- Layer 1: PAC → φ, Fibonacci
- Layer 2: SEC → λ*, information thresholds  
- Layer 3: SEC+PAC coupling → Ξ, physical constants

## Related
- [milestone1](../foundational/experiments/milestone1/) - Parent derivation chain
- [oscillation_attractor_dynamics/exp_24](../foundational/experiments/oscillation_attractor_dynamics/) - Original Ξ derivation
- [pac_confluence_xi](../foundational/experiments/archive/era2/pac_confluence_xi/) - Ξ in Standard Model
