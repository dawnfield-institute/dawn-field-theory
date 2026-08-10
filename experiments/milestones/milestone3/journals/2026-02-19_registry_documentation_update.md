# 2026-02-19: Registry & Documentation Update

## Summary
Extended the falsification registry from F1-F10 to F1-F13, covering the stoichiometric experiments (exp_13-15) and promoting exp_12's complementarity finding from exploratory to scored. Updated PRELIMINARY_RESULTS.md and SYNTHESIS.md with honest assessments of current strengths and weaknesses across the full experiment suite.

## Timeline

### 09:00 - Analysis
Reviewed current state of FALSIFICATION_REGISTRY.md (F1-F10), PRELIMINARY_RESULTS.md (A1-A3, B1-B5), and SYNTHESIS.md ("9 tests, 8 pass, 1 weakened"). Identified five specific gaps flagged in honest assessment.

### 09:30 - Registry Extension
Added three new entries to FALSIFICATION_REGISTRY.md:
- **F11** (exp_12): Fibonacci–MED complementarity. Promoted from "exploratory, not scored" to a full scored test. The golden base paradox — Fibonacci coupling can't reach MED depth — is a genuine theoretical insight, not just a diagnostic.
- **F12** (exp_13, exp_14): Stoichiometric Fibonacci derivation. 8/10 PASS across both scripts. Notes the 0.86× selectivity weakness and its resolution by exp_17.
- **F13** (exp_15): PAC/SEC cost monotonicity. 4/4 PASS. SEC/PAC ratio crosses 1.0 at F₈.

Updated summary: 11/13 pass, 1 borderline (F8/α), 1 corrected (F9/independence).

### 10:00 - exp_12 Falsification Block
Added `results['falsification']` block to exp_12_coupling_base_residuals.py with F11 test ID, per-test scoring (4/5 PASS), hypothesis, and assessment string. This aligns it with the pattern used in exp_01–exp_11.

### 10:15 - PRELIMINARY_RESULTS.md Updates
- **A3** (λ*/β): Added explicit note that these closed forms were NOT directly tested in milestone3. No experiment validates the uniqueness of these candidate formulae. Changed status from "Guidance needed" to "Open" since the methodology exists (exp_09's formula search approach).
- **B5** (Θ recycling): Strengthened the honest range reporting. The 36%–94% spread means the self-funding mechanism is validated but the specific efficiency is model-dependent. Added recommendation that Paper 1 present the range, not a point estimate.

### 10:30 - SYNTHESIS.md Updates
- Updated milestone1 inheritance count from "(9 tests, 8 pass, 1 weakened)" to "(13 tests, 11 pass, 1 borderline, 1 corrected)"
- Added "Expanded Test Coverage" section with Block C (exp_12–17) paragraph summaries
- Added "Honest Assessment" table with per-finding strength ratings

### 10:45 - Key Framing Notes
Also updated FALSIFICATION_REGISTRY.md key findings:
- F8 (α): Added explicit note that Paper 4 should lead with joint constraint (exp_10), not α in isolation
- F9 (independence): Reframed 48 OOM correction as methodological strength, not weakness

## Key Findings
- Registry now covers 13 falsification tests across 15 experiment scripts
- exp_12 promotion was overdue — the complementarity finding constrains Papers 2 and 5
- A3 (λ*/β) is the most significant untested claim remaining in PRELIMINARY_RESULTS
- The "honest assessment" table in SYNTHESIS.md provides a quick reference for paper authors

## Next Steps
- [ ] Run exp_12 to verify the falsification block produces valid output
- [ ] Consider a dedicated exp_18 for A3 (λ*/β uniqueness testing)
- [ ] Paper 4 draft: restructure to lead with joint constraint, α as supporting evidence
- [ ] Paper 1 draft: present Θ recycling as range (36%–94%), not point estimate
