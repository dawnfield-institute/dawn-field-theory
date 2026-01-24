# Wealth Field Dynamics: PAC Collapse Integration and Ξ Correction

**Date**: 2026-01-23 10:35
**Commit**: pending
**Type**: research

## Summary

Major correction to wealth_field_dynamics experiments. Ξ was being tested as a "threshold" when it's actually EMERGENCE per PAC collapse level. Deprecated exp_09 (asked wrong question), created exp_14 and exp_15 to test the correct hypotheses, and updated all documentation.

## Changes

### Added
- `exp_13_pac_collapse_mechanism.py`: Connects exp_12 response type finding to PAC collapse mathematics
- `exp_14_phi_ratio_redistribution.py`: Tests whether wealth redistributions show φ-ratio splits (61.8/38.2)
- `exp_15_emergence_per_level.py`: Tests whether emergence per restructuring ≈ π/55 ≈ 5.71%

### Changed
- `exp_02_xi_inequality_threshold.py`: Updated docstring with proper Ξ derivation chain from exp_24
- `exp_09_threshold_specificity.py`: **DEPRECATED** - added warning that it asks the wrong question
- `README.md`: 
  - Deprecated exp_09, added exp_14-15
  - Updated Ξ understanding throughout
  - Fixed falsification criteria
- `SYNTHESIS.md`:
  - Added "Critical Correction: Ξ is NOT a Threshold" section
  - Updated validation criteria

## Details

### The Core Correction

**WRONG**: Testing if Gini/baseline crossing 1.057 predicts crises (threshold test)

**RIGHT**: Ξ - 1 = π/55 is the geometric emergence per PAC collapse level:
```
Within-level:   -0.0283 (φ-split reduces local coherence)
Cross-level:    +0.0854 (inter-branch adds coherence)
Net emergence:  +0.0571 = π/55 per level
```

At depth 55 (F₁₀): cumulative = π (one Möbius half-twist)

### Correct Tests

1. **Do wealth splits show φ-ratio?** - Need microdata (estates, business partitions)
2. **Is emergence per restructuring ~5.71%?** - Historical estimates vary; suggestive but not rigorous
3. **~55-year cycles?** - 1933-1988 is intriguingly close to 55 years

### What Would Validate
- φ-ratio splits in actual inheritance/divestiture microdata
- Productivity gains from restructuring clustering near 5.71%
- Cross-country universality of patterns

### What Would Falsify
- Splits uniformly distributed (no φ preference)
- Emergence random across restructuring events
- 50/50 or arbitrary ratios produce equally stable outcomes

## Related
- [oscillation_attractor_dynamics/scripts/exp_24_comprehensive_validation.py](../oscillation_attractor_dynamics/scripts/exp_24_comprehensive_validation.py) - The Ξ derivation
- [milestone1/SYNTHESIS.md](../milestone1/SYNTHESIS.md) - Constant hierarchy
