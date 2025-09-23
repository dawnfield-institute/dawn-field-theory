# PAC Physics Engine - Results Tether

## Summary

This document tethers the current state of PAC Physics Engine results to the broader context of information amplification and conservation law research. It provides a snapshot of recent findings, clarifies the meaning of key metrics, and links to the perspective-aware methodology now used throughout the codebase.

---

## Key Results (as of September 23, 2025)

- **Conservation Quality:** 1.0 (perfect)
- **Global Balance:** 1.0
- **Parent Perspective Amplification:** 1.0x (conservation enforced)
- **Child Perspective Amplification:** 2.7–3.1x (local concentration region)
- **Reference Ratio (dawn-field):** 15.56x (empirical, not a target)
- **Spatial Redistribution:** ~20% of field points show concentration
- **Entropy Change:** Negative (system self-organizes)
- **No Conservation Violations:** Zero at machine precision

---

## Interpretation

- **Amplification is Observational:** The 15.56x value is an empirical observation from a specific experiment, not a universal target or law.
- **Perspective Matters:** Local (child) measurements can show amplification due to spatial redistribution, while global (parent) measurements always show perfect conservation.
- **No Contradiction:** Conservation and amplification coexist because they are measured at different spatial/structural scales.
- **Focus:** The engine is now designed to enforce conservation and report observational measurements, not to chase arbitrary targets.

---

## Methodology Reference

See `docs/AMPLIFICATION_PERSPECTIVE.md` for a full explanation of the measurement perspective issue and why the codebase no longer treats 15.56x as a target.

---

## Next Steps

- Continue collecting results under varied conditions
- Compare with other experimental frameworks
- Refine spatial redistribution analysis
- Maintain strict conservation law enforcement

---

*This document is a living tether. Update as new results and insights emerge.*
