# Experiment Journal Schema v1.1

## Overview

Journals provide a **chronological trace** of research activity within an experiment folder. They serve as:

1. **Daily progress logs** - timestamped activity for audit/reproducibility
2. **Discovery documentation** - capturing insights as they happen
3. **Decision records** - why certain paths were taken or abandoned

---

## Filename Convention

```
YYYY-MM-DD_descriptive_slug.md
```

**Examples**:
- `2025-12-11_phi_eigenvalue_discovery.md`
- `2025-12-10_phase_transition_validation.md`
- `2025-12-09_initial_exploration.md`

**Rules**:
- One primary file per day (can have focused topic files if needed)
- Use lowercase with underscores
- Keep slugs short but descriptive (3-5 words max)

---

## Required Frontmatter

```markdown
# Title: Concise Description of Session

**Date**: Month Day, Year  
**Session**: Brief context or focus area  

---
```

---

## Document Structure

### 1. Summary (Required)
Brief overview of the session's key outcomes (2-5 sentences).

### 2. Timeline (Recommended for daily logs)
Chronological activities using H3 headers:

```markdown
## Timeline

### HH:MM - Activity Type

Description of what was done, what was found.

**Status**: ✅ Confirmed | ❌ Failed | 🔄 In Progress | 💡 Insight
```

**Activity Types**:
- `Setup` - Environment/data preparation
- `Experiment` - Running code/tests
- `Analysis` - Interpreting results
- `Discovery` - New findings
- `Bug Fix` - Resolving issues
- `Documentation` - Writing/updating docs
- `Planning` - Deciding next steps

### 3. Key Findings (Recommended)
Bulleted or tabled summary of important results.

### 4. Next Steps (Optional)
What to do next session.

### 5. Files Modified (Optional)
List of created/changed files.

---

## Example: Minimal Daily Log

```markdown
# PHM: Scale Invariance Validation

**Date**: December 12, 2025  
**Session**: Testing λ₁ stability across prime ranges  

---

## Summary

Confirmed λ₁ = 0.6183 ± 0.002 across 10k to 10M primes. Scale invariance holds.

## Timeline

### 09:00 - Setup

Prepared environment, loaded exp_02 script.

**Status**: ✅ Confirmed

### 09:30 - Experiment

Ran scale test across 9 prime limits.

**Status**: ✅ Confirmed

### 11:00 - Analysis

Computed mean/std of λ₁ values. Mean matches 1/φ to 4 decimals.

**Status**: 💡 Insight

## Key Findings

| Metric | Value |
|--------|-------|
| Mean λ₁ | 0.6183 |
| Std λ₁ | 0.0018 |
| 1/φ | 0.6180 |

## Next Steps

- Test with vocabulary scaling (exp_03)
- Investigate eigenvalue pair structure
```

---

## Example: Discovery-Focused Log

```markdown
# SEC: Phase Transition on Odd Manifold

**Date**: December 10, 2025  
**Session**: Why φ appears only for odd numbers  

---

## Summary

Major breakthrough: φ emergence occurs ONLY on the odd number manifold.
Even numbers show frac(E>0) ≈ 0.38, while odds show 0.6187 = 1/φ.

## The Discovery

While debugging why size=9 produces φ, discovered the code was computing
on all numbers (giving ~0.50) vs odd numbers only (giving φ).

[... detailed explanation ...]

## Validation

| Manifold | frac(E>0) | Error vs 1/φ |
|----------|-----------|--------------|
| ALL | 0.50 | 0.117 |
| ODD | 0.6187 | **0.0007** |
| EVEN | 0.38 | 0.235 |

## Interpretation

[... theoretical implications ...]

## Files Modified

- scripts/exp_03_factor_base_sweep.py (bug fix)
- results/exp_03_odd_manifold.json (new)
```

---

## Integration with meta.yaml

Each journals/ folder should have a meta.yaml listing files:

```yaml
schema_version: "2.0"
description: "Research journals for [experiment name]"
semantic_scope: "documentation"
proficiency_level: "research"

files:
  - "[2025-12-11_topic.md]": "Brief description"
  - "[2025-12-10_topic.md]": "Brief description"

chronological_order: "descending"  # newest first
```

---

## Best Practices

1. **Write as you go** - Don't wait until end of day
2. **Capture failures** - Failed experiments are valuable data
3. **Link to files** - Reference scripts and results by name
4. **Use tables** - Numerical results are clearer in tables
5. **Status markers** - Make scan-ability easy with ✅/❌/💡
6. **One day, one file** - Exceptions for major discoveries warranting separate docs
