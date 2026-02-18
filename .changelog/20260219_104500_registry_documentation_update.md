# Registry & Documentation Update

**Date**: 2026-02-19 10:45
**Type**: documentation

## Summary
Extended the milestone3 falsification registry from F1-F10 to F1-F13, promoted exp_12 (coupling base complementarity) from exploratory to scored, and updated PRELIMINARY_RESULTS.md and SYNTHESIS.md with honest assessments reflecting the full experiment suite (exp_01 through exp_17).

## Changes

### Changed
- FALSIFICATION_REGISTRY.md: Added F11 (exp_12 complementarity, promoted), F12 (exp_13/14 stoichiometry, 8/10), F13 (exp_15 PAC/SEC, 4/4). Summary now 11/13 pass.
- FALSIFICATION_REGISTRY.md: Updated key findings #7 (α should lead with joint constraint), #8 (48 OOM as strength), added #11 (stoichiometric emergence), #12 (SEC cost scaling)
- PRELIMINARY_RESULTS.md: A3 (λ*/β) marked as "not directly tested in milestone3"; B5 (Θ recycling) updated with honest 36%–94% range
- SYNTHESIS.md: Updated milestone1 inheritance count to 13 tests. Added Block C (exp_12–17) summaries and honest assessment table.
- exp_12_coupling_base_residuals.py: Added `results['falsification']` block with F11 test ID and per-test scoring

## Details
Five specific improvements identified from honest assessment review:
1. **exp_09 (α) weakness**: Registry now explicitly notes Paper 4 should lead with joint constraint
2. **exp_10 (48 OOM) strength**: Registry reframes correction as methodological rigour
3. **exp_12 promotion**: Complementarity finding scored as F11 (4/5 PASS)
4. **Falsification coverage**: Registry now covers all scored experiments (F1-F13)
5. **A3/B5 honesty**: PRELIMINARY_RESULTS updated with untested acknowledgment and efficiency range

## Related
- .changelog/20260218_161500_tightening_experiments.md (exp_16/17 creation)
- milestone3/journals/2026-02-19_registry_documentation_update.md
