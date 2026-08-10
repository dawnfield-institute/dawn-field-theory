# Journal: exp_37 Correction Template Origin

**Date**: 2026-03-14
**Status**: complete (9/9 PASS)

---

## Origin

Open question #2 from the session: WHY does the correction template F_a/(nπF_b²) work for three forces (EM, gravity, dark energy)? The template was discovered empirically in milestone3 exp_23/26 and extended in exp_34/35, but the physical interpretation was missing.

## The Decoded Template

```
correction = 1 ± F_a / (n * pi * F_b²)

= 1 ± (cascade paths to depth a) / (n × cascade boundary area at depth b)

= 1 ± coupling / phase_space
```

## Component Interpretations

| Component | Symbol | Interpretation | Origin |
|-----------|--------|---------------|--------|
| Denominator area | πF_b² | Isotropic cascade boundary area at depth b | π from rotational symmetry, F_b² from cascade depth → area |
| Numerator paths | F_a | Cascade path count (coupling strength) | Fibonacci addition identity: F_a = F_b·F_{gap+1} + F_{b-1}·F_{gap} |
| Multiplicity | n | Independent field components (boundary sectors) | EM: 4 (A_μ), Gravity: 1 (scalar cascade density), Dark E: 4 (metric diagonal) |
| Sign | ± | Phase interference: screening (-) vs enhancement (+) | Odd spin → destructive, Even spin → constructive |
| Gap a−b | Fibonacci | Cascade coupling distance | Short (EM, 3=F_4) vs long (Gravity, 7=F_7) |

## Key Results

### πF_b² convergence
Ratio πF_b²/φ^(2b) → π/5 = 0.6283 (from F_b ~ φ^b/√5). The cascade boundary area is the isotropic (circular/spherical) wavefront of the PAC cascade at Fibonacci depth b.

### Fibonacci addition identity decomposition
F_a decomposes into two terms via the addition identity. The dominant term F_{gap+1}/F_b contributes ~71-73% for all three forces. This means most of the coupling comes from the gap structure, not the absolute depth.

### n = 1 for gravity is key
Gravity's n=1 means the cascade boundary is undivided — gravity is uniquely isotropic. This connects to Peter's physical insight: gravity creates spheres (n=1) while EM creates structured fields (n=4).

### Perturbative interpretation breaks for gravity
EM: x = 0.026 (small perturbation). Gravity: x = 1.16 (perturbation theory breaks). The template works for gravity not as a perturbative expansion but as an exact Fibonacci identity.

### Formula A vs B
Formula A (2Ξ = 2.117, 1.80%) = what the cascade DOES (round-trip × attractor).
Formula B (1 + F_13/(πF_6²) = 2.159, 0.18%) = what the cascade IS (Fibonacci geometry).
Ratio B/A = 1.020. Two descriptions of the same physics, 1.6% apart.

### Weinberg angle prediction
sin²(θ_W) = 0.2312 ≈ ln2/3 = 0.2310 at 0.07%. A PAC quantity match, though speculative.

## Honest Assessment

**INTERPRETIVE, NOT DERIVATIONAL.**

Can claim:
- Form F_a/(nπF_b²) = coupling/phase_space is natural for cascades
- Each component (π, F_b², F_a, n, sign) has clear physical meaning
- The Fibonacci structure enforces corrections through tree geometry

Cannot claim:
- Cannot predict which (a,b,n,sign) goes with which force from first principles
- Dark energy gap (4) is not Fibonacci, weakening that arm
- "Cascade boundary area" is geometric intuition, not rigorous PAC derivation
- Perturbative interpretation breaks for gravity (x > 1)

## Open Questions

- Can (a,b) be derived from force-specific properties (spin, gauge group)?
- Is the Weinberg angle connection real or coincidental?
- Why does the perturbative picture break for gravity but the template still works?
- Can the template predict the strong/weak force corrections?
