# Prime Growth Dynamics: Primes as Crystallization Points

**Version**: 0.3.0  
**Status**: ✅ Major Discovery (Even-Odd Oscillation)  
**Date**: 2026-02-05  
**Origin**: Discord conversation with Andy Farmer (2026-02-04)

---

## Core Model: Entropic Fizz → Crystallization

Inspired by [cosmo.py](../legacy/cosmo.py) - the CIMM cosmological simulation:

```
Pure Entropy (noise) → "Fizz" (SHA-seeded structure) → Crystallization (matter)
```

In arithmetic:

```
Pure Potential → Primes (first crystallization) → Composites (structure growth)
```

**Primes are where structure FIRST crystallizes from entropic potential.**
- They're not "stuck points" or anomalies
- They're the irreducible nucleation sites
- Composites grow FROM prime crystallization points
- 2 is the first bubble - its asymmetry propagates through all subsequent structure

---

## Executive Summary

This experiment investigates the ontological nature of prime numbers as **base cases** in a recursive structure, rather than as "stuck points" or anomalies. Key questions:

1. **Are primes the crystallization points of entropic potential?** (First structure from noise)
2. **How does structure propagate from primes?** (Composites as growth from crystallization sites)
3. **Why φ?** (Balance point of crystallization dynamics)

### MAJOR DISCOVERY: Even-Odd Oscillation

Factorization depth (Ω) oscillates by **parity of distance to nearest prime**:

| Distance | Mean Ω | Pattern |
|----------|--------|---------|
| Odd | 4.30 | HIGH |
| Even | 2.80 | LOW |

**Statistical significance**: t = 110.80, p ≈ 0

**Explanation** (exp_07): Distance parity = n parity (since all primes > 2 are odd). Even n has factor of 2, increasing Ω.

**φ in the oscillation**: Positive/negative integral ratio ≈ **1/φ** (0.0295 error)!

---

## Theoretical Background

### The Convergence

Milestone2 established remarkable cross-domain invariants:

| Discovery | Domain | Significance |
|-----------|--------|--------------|
| k = d × F_{d+1} | Turbulence | First-principles derivation of She-Leveque k=9 |
| 240 = F₃×F₄×F₅×F₆ | QFT/Casimir | Four consecutive Fibonacci in regularization |
| Fibonacci @ Mersenne d | String/M-theory | d = 2^k - 1 hosts fundamental theories |
| RG = PAC across scales | Field theory | φ is fixed-point attractor |

Andy's insight: **Primes are the integers; everything else is combination.**

### From Oscillation Attractor Dynamics (exp_03)

| Measure | Value | Interpretation |
|---------|-------|----------------|
| Mean I(prime) | +0.1595 | Primes inject POSITIVE impulse |
| Mean I(composite) | -0.0169 | Composites crystallize (NEGATIVE) |
| Primes with I > 0 | 100% | ALL primes are injection events |

**Primes inject structure into entropy. Composites crystallize around them.**

### Andy's Growth Questions

> "Does 12 grow from the end of 11, or does 1 grow and push all the other numbers up?  
> Or does 1 get moved up to 2 and another number is slotted into the space?  
> Is it all at once, whole unit by unit, or a piece of a unit at a time?"

These questions frame alternative models of how structure emerges from arithmetic foundations.

---

## Hypotheses

### H1: Primes as Base Cases (Adapted from Ackermann analogy)

In recursive function theory, base cases are the irreducible foundations. We hypothesize:

- **Primes are base cases**: The minimum set from which all structure derives
- **Factorization = actualization trace**: Breaking composites traces back to prime seeds
- **PAC conserved in factorization**: f(n) = Σf(primes) for appropriate f

### H2: Growth Direction Models

Three candidate models:

| Model | Description | Testable Prediction |
|-------|-------------|---------------------|
| **Stack Growth** | 1 grows to 2, pushing all up | Gap structure depends on local injection |
| **End Accretion** | n+1 forms at frontier | Gap structure depends on global density |
| **Slot-In** | Numbers slot into open positions | Gaps are pre-determined positions |

### H3: Primes Create Space for Composites

Rather than composites filling gaps between primes:
- **Primes seed structure** (injection points)
- **Composites crystallize** in the gradient field
- **New primes** appear where crystallization pressure is insufficient

---

## Experimental Design

### Part I: Base Case Verification (exp_01-03)

| Exp | Name | Tests |
|-----|------|-------|
| 01 | PAC Conservation | f(n) = Σf(prime factors) for entropy, complexity |
| 02 | Minimal Seed Set | Primes as optimal generators for N_{<n} |
| 03 | Irreducibility | Information-theoretic primality test |

### Part II: Growth Direction (exp_04-06)

| Exp | Name | Tests |
|-----|------|-------|
| 04 | Local vs Global | Does prime distribution depend on local or global properties? |
| 05 | Frontier Dynamics | Behavior at the "edge" of explored number line |
| 06 | Slot Analysis | Are there predictable "slots" for primes? |

### Part III: Sequential vs Parallel Growth (exp_07-09)

| Exp | Name | Tests |
|-----|------|-------|
| 07 | Prime-First | Do primes determine composite positions? |
| 08 | Simultaneous | Does structure emerge all at once? |
| 09 | Fibonacci Cascade | Does growth follow Fibonacci timing? |

### Part IV: Synthesis (exp_10-12)

| Exp | Name | Tests |
|-----|------|-------|
| 10 | Mersenne Connection | Why M_k = 2^k - 1 are special for both primes and theories |
| 11 | Ackermann Bridge | Connect recursion depth to prime density |
| 12 | Falsification | Test alternatives, negative controls |

---

## Success Criteria

### Core Predictions

- [ ] PAC conservation holds for factorization (exp_01)
- [ ] Primes form minimal generating set (exp_02)
- [ ] Growth model distinguishable from alternatives (exp_04-06)
- [ ] Mersenne pattern connects primes ↔ physics dimensions (exp_10)

### Falsification Conditions

If any of these hold, revise hypothesis:
- [ ] f(composite) ≠ Σf(prime factors) for natural f choices
- [ ] Composite set without primes has same structure
- [ ] Growth direction has no preferred model
- [ ] Mersenne connection is coincidence (random match rate)

---

## Key Questions for Andy

1. **Is 1 special?** (Not prime, not composite - the identity?)
2. **Does infinity exist?** Or just "deep recursion"?
3. **Can we formalize "growth rate"?** (Related to 22/7 ≈ 2L₅/L₄?)
4. **What about 55 = F₅ × L₅?** (Prime? Fibonacci? Lucas?)

---

## Dependencies

```
milestone2 (complete)
    ├── k = d × F_{d+1} derivation
    ├── Casimir 240 = F₃×F₄×F₅×F₆
    └── Mersenne dimensional pattern

sec_prime_manifold
    └── φ at criticality, phase transitions

oscillation_attractor_dynamics
    └── Primes as injection points

prime_harmonic_manifold
    └── λ₁ → 0.5, Cramér divergence

Ackermann recursion (Andy's observation)
    └── Primes as base cases
```

---

## Directory Structure

```
prime_growth_dynamics/
├── meta.yaml
├── README.md
├── SYNTHESIS.md
├── core/
│   ├── meta.yaml
│   └── growth_engine.py
├── scripts/
│   ├── meta.yaml
│   ├── exp_01_pac_conservation.py
│   ├── exp_02_minimal_seed.py
│   └── ...
├── results/
│   └── meta.yaml
└── journals/
    ├── meta.yaml
    └── 2026-02-05_andy_conversation_origin.md
```
