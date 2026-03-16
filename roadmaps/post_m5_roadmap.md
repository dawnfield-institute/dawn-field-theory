# Post-Milestone 5 Roadmap

**Created**: 2026-03-16
**Context**: M5 complete (13 experiments). SM derivation chain nearly closed. Simulator validated with de-actualization. Scorecard 8/13 (C).

---

## Immediate: Simulator Physics (Reality Engine)

### 1. Fix Scorecard Regressions
De-actualization improved coupling constants but regressed two Tier 2 metrics:
- **phi² mass spacing** (40.6% error, was 10.8%) — tune `quantum_pressure_coeff`, `mass_gen_coeff`, `field_scale`
- **Entropy reduction** (F) — de-actualization dissolves structure; may need entropy-aware gating

Approach: The scorecard tuning suggestions point to specific config knobs. Systematic sweep with de-actualization active.

### 2. Entropy-Coherence Gravity Modulation
Planned fix from M5 exp_10 analysis: gravity's G_local doesn't account for entropy vs coherence balance. When Xi_s < 1 (entropy dominates), gravity should weaken. This is already derived in DFT (exp_29, infodynamic gravity framework) but not yet in the engine.

### 3. Target: Scorecard >= 11/13 (B-)
Close remaining gaps to reach the M5 success criterion that wasn't met. Requires fixing phi² spacing + entropy reduction + improving at least one Tier 3 metric.

---

## Near-Term: Theory Publications

### 4. Paper 9 — Standard Model from Information: The Complete Derivation
Consolidates M1-M5 results into a single paper:
- All SM parameters from Fibonacci arithmetic (alpha, sin²θ_W, Koide, masses)
- Higgs mass at 83 ppm, lambda = phi/(4*pi)
- PMNS/CKM mixing angles as arctan(F_a/F_b)
- New identity: sin²θ_W = tan(θ_Cabibbo) = F4/F7 = 3/13
- Simulator validation of coupling attractors

### 5. Paper 10 — BSM Predictions from PAC Structure
- Z' at 395±20 GeV (from M1)
- Neutrino mass hierarchy (mixing angles derived in M5, masses still open)
- Dark matter candidates from PAC structure
- Testable predictions for LHC Run 4

---

## Medium-Term: Remaining Open Problems

### 6. Neutrino Mass Hierarchy
M5 derived mixing angles (PMNS < 0.3 deg error) but not absolute masses. The mass hierarchy should follow from the same Fibonacci arithmetic that gives charged lepton masses.

### 7. Tier 3 Constants
Fine structure (1/137), Koide Q (2/3), mu/e ratio (206.8) — these are derived analytically in DFT but the simulator can't reproduce them on a 128×64 2D grid. Open question: is this a resolution limit or a missing mechanism?

### 8. Causal Emergence Chains
The engine produces gravity wells, but the full causal chain (gravity → stars → fusion → heavy elements → chemistry) needs work. Each step should emerge from the existing operators, not be programmed.

---

## Long-Term: Validation & Community

### 9. 3D Möbius Manifold
Current engine is 2D (128×64 band). 3D Möbius would allow proper particle emergence with all three spatial dimensions. Major engineering effort.

### 10. External Validation
- arXiv submission of Papers 9-10
- Peer review engagement
- Community falsification programme
- Government/institution outreach

---

## Priority Order

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Fix phi² spacing + entropy | 1-2 sessions | Scorecard 10/13 |
| 2 | Entropy-coherence gravity | 1 session | Scorecard 11/13+ |
| 3 | Paper 9 draft | 2-3 sessions | Publication |
| 4 | Neutrino masses | 1-2 sessions | SM completion |
| 5 | Paper 10 draft | 1-2 sessions | BSM predictions |
| 6 | 3D manifold | Major | New physics |
