# PACSeries Paper 5: Classical Physics from Information Geometry

## Overview

This paper demonstrates that the structure of classical electromagnetism — curl operations, three spatial dimensions, inverse-square forces, quantised charge — follows from three information-theoretic constraints: PAC conservation, MED bounds (depth ≤ 2, nodes ≤ 3), and SEC dynamics. A speculative extension to gravity through symmetric tensor projection is included.

## Key Results

| Result | Status | Section |
|--------|--------|---------|
| SEC wave equation → c | Exact (structural) | §2 |
| D = 3 from 5 independent paths | Convergent | §3 |
| Curl from depth-2 projection | Structural | §4 |
| Faraday's law to 10⁻¹⁶ | Numerical | §4.2 |
| Charge = winding number | Topological | §5 |
| Coulomb r⁻².⁰⁰⁰⁰ | Numerical | §5.2 |
| SEC–NS equivalence | Structural | §6 |
| k = d × F_{d+1} | Verified d=2,3 | §6.3 |
| Casimir 240 = F₃F₄F₅F₆ | Exact | §7 |
| Mersenne-Fibonacci correspondence | Observed d=1,3,7 | §7 |
| Ξ - 1 = π/55 derivation | Derived | §9 |
| Gravity at depth 183 | Speculative | §8 |

## Source Experiments

- `experiments/milestones/maxwell_from_pac_sec/` — 5 scripts (wave speed, charge, curl, α, D=3)
- `experiments/milestones/gravity_from_maxwell_pac/` — 12 scripts (projection duality, hierarchy, N-body)
- `papers/series/PACSeries/v0.2/feigenbaum_fibonacci_arithmetic/Data/` — Ξ derivation, Feigenbaum constants
- `archive/era2-prefield/navier-stokes/` — MED discovery, SEC-NS equivalence
- `experiments/milestones/milestone2/` — She-Lévêque, Casimir, Mersenne

## Dependencies

- Paper 1: Structure Cost of Erasure (SEC dynamics)
- Paper 2: Balance Constant (Ξ = 1 + π/55)
- Paper 3: Feigenbaum Constants (MED discovery, F₁₀ = 55)
- Paper 4: Standard Model Parameters (α formula, She-Lévêque, Casimir)

## Reproduction

```bash
cd experiments/milestones/maxwell_from_pac_sec/scripts/
python exp_01_sec_wave_speed.py
python exp_02_charge_quantization.py
python exp_03_curl_projection.py
python exp_04_fibonacci_alpha.py
python exp_05_3d_necessity.py
```

## Status

- [x] Draft complete
- [ ] Internal review
- [ ] Final voice pass
- [ ] Figures generated
- [ ] Code package assembled
