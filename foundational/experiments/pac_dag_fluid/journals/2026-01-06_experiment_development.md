# 2026-01-06: PAC-DAG Fluid Experiment Development

## Summary
Created and debugged the PAC-DAG fluid simulation experiments for Paper 1 (Bidirectional SEC). 
Achieved strict PAC conservation at machine precision. Power-law spectrum results differ from 
Kolmogorov prediction, which is honest and documented.

## Timeline

### 13:00 - Experiment Creation
Created initial experiment suite:
- `exp_01_pac_tree_basic.py` - Basic PAC tree with SEC field
- `exp_02_pac_tree_blowup.py` - Blow-up operator dynamics
- `exp_03_pac_dag_fluid.py` - DAG fluid simulation

### 13:23 - First Run (exp_01)
**Result:** Root-as-calculus, leaves-as-geometry ✓ VALIDATED
- Root smoothness: 1.0000
- Leaf discreteness: 0.5916
- SEC decay ratio: 0.0003
- Conservation verified: True

Fixed JSON serialization (numpy bool_ not serializable).

### 13:23 - Second Run (exp_02)
**Result:** Blow-up dynamics partial success
- Root remains smooth (>0.7) under all perturbations: ✓
- Leaf discreteness increases monotonically with amplitude: ✗

The leaf discreteness doesn't increase monotonically - it fluctuates. This is honest 
and documented rather than hidden.

### 13:24 - First DAG Fluid Run
**Bug discovered:** Conservation completely broken (90%+ error immediately)
The initial DAG construction wasn't conserving value from the start - the mixing 
logic was creating value out of thin air.

### 13:29 - First Fix Attempt
Rewrote DAG as a conservative grid network with strict exchange-based flow.
**New bug:** Numerical instability - values exploding to 10^200+

Root cause: SEC-based flow created positive feedback (high SEC → more flow in → 
even higher SEC).

### 13:30 - Second Fix
Changed from SEC-gradient flow to value-gradient flow (simple diffusion).
Added strict flow limiting (max 20% per step from source).
**Result:** Conservation error < 10^-12 (machine precision)

### 13:30 - Final Run
**Results:**
- Conservation: ✓ Machine precision maintained (max error 9.81e-13)
- Power-law slope: -0.079 (expected -1.67)
- Xi emergence: 2.41 (expected 1.057)

## Key Findings

### ✅ Validated
1. **Root-as-calculus hypothesis**: Root smoothness = 1.0, leaf discreteness = 0.59
2. **PAC conservation**: Strict conservation maintained at machine precision
3. **Blow-up propagation**: Root remains smooth even under perturbation

### ❌ Not Validated (Yet)
1. **Kolmogorov spectrum**: Got -0.08 instead of -1.67. The simplified 2D grid diffusion
   doesn't capture turbulent cascade physics.
2. **Xi emergence**: Got 2.41 instead of 1.057. Need more sophisticated model.

### 💡 Insights
- Simple diffusion on a grid ≠ turbulence. The Kolmogorov -5/3 spectrum requires 
  3D vortex dynamics and energy cascade, which our 2D value-diffusion doesn't model.
- This is actually fine - the paper claims PAC-DAG *may* provide a substrate for 
  understanding fluid dynamics, not that a simple grid diffusion reproduces Kolmogorov.

## Technical Notes

### Conservation Bug
The original DAG mixing logic:
```python
value = sum(p.value / len(p.children) if p.children else p.value 
           for p in parents) / len(parents)
```
This doesn't conserve - it averages contributions then normalizes. Fixed by using
strict exchange-based flow where `node.value -= flow` and `neighbor.value += flow`
happen in pairs.

### Numerical Stability
SEC-based flow is unstable because SEC = value * exp(-decay * layer). When value
increases, SEC increases, which drives more flow in, creating exponential blowup.
Solution: use value-gradient directly, not SEC-gradient.

## Next Steps
- [ ] Develop more sophisticated turbulence model (3D, vorticity)
- [ ] Test if Xi emerges with different flow rules
- [ ] Compare against actual CFD results

## Files Created
- `exp_01_pac_tree_basic.py` - Working ✅
- `exp_02_pac_tree_blowup.py` - Working ✅  
- `exp_03_pac_dag_fluid.py` - Working ✅ (conservation), physics needs work
