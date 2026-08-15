# Journal: exp_39 Strong Template Selection

**Date**: 2026-03-14
**Status**: complete (8/8 PASS)

---

## Origin

From exp_38: multiple α_s correction candidates, need structural selection not search. Which (a,b,n) is uniquely determined by PAC constraints?

## Key Results

### 1. The b Hierarchy (Part E)

b values in confirmed corrections decrease by 1: **b=7(EM) → 6(gravity) → 5(dark energy) → 4(strong)?**

Each force's cascade boundary is set by the NEXT LOWER gauge sector:
- EM (b=7): boundary = full SM (F₇=13 modes)
- Gravity (b=6): boundary = QCD sector (F₆=8 gluons)
- Dark energy (b=5): boundary = flavor sector (F₅=5 quarks)
- Strong (b=4?): boundary = weak sector (F₄=3 bosons)

### 2. n Selection Principle (Part C)

- Abelian (U(1), EM): n=4 from spacetime components (gauge index trivial)
- Non-abelian (SU(3), strong): n=adjoint from gluon self-coupling (gauge sectors dominate)
- Because gluons carry color charge and self-interact, the cascade boundary IS sectored by color

### 3. Two Leading Candidates (Part G)

| ID | b | n | a | Gap | α_s | Error | Score |
|----|---|---|---|-----|-----|-------|-------|
| C1 | 4 | 8 | 12 | 8=F₆ | 0.1264 | 7.24% | 4/5 |
| C2 | 2 | 3 | 5 | 3=F₄ | 0.1182 | 0.29% | 4/5 |
| C3 | 2 | 8 | 7 | 5=F₅ | 0.1172 | 0.58% | 3/5 |
| C4 | 4 | 6 | 11 | 7 | 0.1178 | 0.10% | 2/5 |

C2 wins precision + structural criteria but breaks b hierarchy.
C1 fits hierarchy but terrible error.

### 4. Scale Analysis (Part D)

Bare formula α_s = F₃/(2φ·F₆) = 0.0773 matches one-loop running at Q ≈ 3534 GeV (TeV range). The bare formula is a high-energy value, not infrared (opposite of EM).

### 5. The Unresolved Question

Does the strong cascade boundary sector by:
- Color charge (n=3, fundamental rep) → C2
- Force carrier (n=8, adjoint rep) → C3

This is a physics question about what the cascade "sees" — and it determines the unique selection.

## Honest Assessment

Narrowed from 100+ to 2-3 structurally motivated candidates. The b hierarchy is a genuine structural discovery. But unique selection requires resolving n=3 vs n=8 — which representation does the PAC cascade see?

## Formulas

```
C2: α_s = F₃/(2φ·F₆) × (1 + F₅/(3πF₂²))  = 0.1182  (0.29%)
C3: α_s = F₃/(2φ·F₆) × (1 + F₇/(8πF₂²))  = 0.1172  (0.58%)
```
