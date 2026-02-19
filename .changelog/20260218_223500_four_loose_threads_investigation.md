# Four Loose Threads Investigation — exp_22-25

**Date**: 2026-02-18 22:35
**Type**: research

## Summary
Investigated 4 loose threads identified in prior gap analysis, creating experiments exp_22–25 in milestone3. Each thread explored a specific area where the framework had open questions.

## Changes

### Added
- `scripts/exp_22_pac_depth_bound.py` — PAC derives MED depth ≤ 2 (3/4 PASS)
- `scripts/exp_23_f183_correction.py` — F₁₈₃ gravity correction term and uniqueness (3/4 PASS)
- `scripts/exp_24_pac_lazy_signal_anatomy.py` — Bootstrap/LOO/PCA decomposition of exp_21 (1/4 PASS)
- `scripts/exp_25_dark_matter_depth_map.py` — DM depth mapping, Ω_c prediction, φ-equilibrium (2/2 PASS)
- Result JSONs for all 4 experiments in `results/`

## Details

### Thread 1: MED depth from PAC (exp_22)
Analytical theorem: all k-step PAC recursions have max depth floor ≤ 2. Fibonacci (k=2) gives loosest bound at φ² = 2.618. K→∞ limit is exactly 2.0. Peak structural density at d = φ². Only k=2 gives the ln(φ) ratio.

### Thread 2: F₁₈₃ gravity hierarchy (exp_23)
Gap between F₁₈₃ and (M_Pl/m_p)² is factor 2.155 (0.333 log₁₀). Best correction: 1 + F₁₃/(πF₆²) = 2.159, residual 0.0008 in log₁₀. 183 is uniquely rank #1 among all cyclotomic depths; only 0.5% of random formulas match. Natural Fibonacci family φ⁴/π ≈ 2.182 gets within Δ=0.005.

### Thread 3: PAC-Lazy signal anatomy (exp_24)
Honest finding: exp_21's p=0.035 is fragile. Bootstrap 95% CI includes zero ([-0.044, 0.013]). p_e drives 60.8% of the signal. Effective DOF is 9 (above null space's 6). Running couplings domain shows d=-0.395 (medium effect) — signal is domain-specific, not universal.

### Thread 4: Dark matter depth mapping (exp_25)
Ω_c = F₃·Ξ/F₆ at 0.148% error (unique: only 0.34% of formulas match). Better: F₇·Ξ²/F₁₀ at 0.079%. φ-equilibrium: DE 68.5% vs 1/φ = 61.8%, crossed at z=0.10. WIMP-range depths: d=74–93. Proposed F₃₇–F₅₀ is GUT-scale, not WIMP. Cyclotomic F₆²+F₆+1=73 → 15 keV (sterile neutrino range).

## Related
- Prior session: f97de50 (104-file documentation update)
- exp_21 (F19): PAC-Lazy discrimination baseline
- exp_16 (F14): Null space prediction failure
- gravity_from_maxwell_pac experiments
