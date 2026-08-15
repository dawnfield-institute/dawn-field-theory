# Prime Growth Dynamics Experiment Created

**Date**: 2026-02-05 14:30
**Commit**: (pending)
**Type**: research

## Summary

Created comprehensive new experiment `prime_growth_dynamics` based on Discord conversation with Andy Farmer. Explores primes as base cases (not stuck recursions), number line growth direction, and crystallization model integrating milestone2 results with oscillation_attractor_dynamics findings.

## Changes

### Added
- `foundational/experiments/prime_growth_dynamics/` - New experiment folder
- `meta.yaml` - Schema v2.0 metadata linking to related experiments
- `README.md` - Overview with hypothesis, design, and success criteria
- `SYNTHESIS.md` - Cross-connections to sec_prime_manifold, oscillation_attractor_dynamics, prime_harmonic_manifold, milestone2
- `core/growth_engine.py` - Reusable library with:
  - PAC conservation functions (log, complexity, entropy)
  - SEC stress field computation
  - Growth model implementations (stack, accretion, slot-in)
  - Fibonacci/Mersenne utilities
- `scripts/exp_01_pac_conservation.py` - Test PAC conservation in factorization
- `scripts/exp_02_growth_direction.py` - Test local vs global, history dependence, slot predictability
- `scripts/exp_03_growth_models.py` - Test discrete vs continuous, residue class, depth cascade, Fibonacci timing
- `journals/2026-02-05_andy_conversation_origin.md` - Full conversation transcript and experimental motivations

### Added (supporting)
- `core/meta.yaml`, `scripts/meta.yaml`, `results/meta.yaml`, `journals/meta.yaml`

## Details

### Origin
Andy Farmer's reframe after seeing milestone2 results:
> "Primes are the integers; everything else is combination."

This sparked questions about how the number line "grows":
- Which end grows?  
- Stack growth vs frontier accretion vs slot-in?
- All at once, unit-by-unit, or type sequence?

### Key Hypotheses
1. **Primes as Base Cases**: Factorization = actualization trace
2. **Crystallization Model**: Primes inject, composites crystallize at intersections
3. **Mersenne Connection**: Same pattern governs both Mersenne primes AND Mersenne dimensions (1,3,7)

### Theoretical Connections
- **SEC Prime Manifold**: φ at critical λ* = 0.9816 is the injection/crystallization balance
- **Oscillation Attractor**: I(prime) > 0, I(composite) < 0 confirmed
- **Milestone2**: k = d × F_{d+1}, Mersenne dimensions host Fibonacci structure
- **Ackermann**: Primes as base cases, not stuck recursions

### Experimental Design
- exp_01: PAC conservation (log, complexity, entropy, Ω, depth)
- exp_02: Growth direction (local vs global, history, slots)
- exp_03: Growth models (discrete/continuous, residue, cascade, Fibonacci timing)
- Future: exp_04-12 for deeper validation

## Related
- [milestone2](../foundational/experiments/milestone2/README.md)
- [sec_prime_manifold](../foundational/experiments/sec_prime_manifold/SYNTHESIS.md)
- [oscillation_attractor_dynamics](../foundational/experiments/oscillation_attractor_dynamics/SYNTHESIS.md)
- [prime_harmonic_manifold](../foundational/experiments/archive/era2/prime_harmonic_manifold/SYNTHESIS.md)
