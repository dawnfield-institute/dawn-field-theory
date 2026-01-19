# Gravity from Maxwell via PAC/SEC

**Status**: 🔄 In Progress  
**Started**: January 19, 2026  
**Authors**: Peter Lorne Groom, Claude (Anthropic)

---

## Hypothesis

**Maxwell's equations and Einstein's gravity are both recursive projections of PAC conservation, differing only in Fibonacci depth.**

| Force | Fibonacci Depth | Key Structure | Coupling |
|-------|-----------------|---------------|----------|
| Electromagnetism | F₇ = 13 | Curl (∇×) | α ≈ 1/137 |
| Gravity | F₁₈₃ ≈ 10³⁸ | Divergence (∇·) | G ≈ 10⁻³⁸ |

The hierarchy 183 = F₇² + F₇ + 1 suggests gravity involves **gauge-squared** interaction.

---

## Core Claim

From `maxwell_from_pac_sec/`, we established:

```
PAC → SEC → MED(depth≤2) → Möbius projection → Maxwell (curl structure)
```

This experiment tests:

```
PAC → SEC → MED(depth≤183?) → different projection → Einstein (divergence structure)
```

If correct:
1. **G is not a free parameter** — it's determined by F₁₈₃
2. **Gravitational waves travel at c** — same SEC wave equation
3. **Inverse-square emerges** — same topological reason as Coulomb
4. **Dark matter may be intermediate depth** — F₃₇ to F₅₀ range

---

## Key Questions

### Q1: Why divergence instead of curl?

Maxwell: antisymmetric projection → curl (∇×)  
Gravity: symmetric projection → divergence (∇·)

The Möbius pre-field has both symmetric and antisymmetric components.
EM couples to the antisymmetric (phase). Gravity couples to the symmetric (amplitude/energy).

### Q2: Why depth 183?

183 = F₇² + F₇ + 1

This is:
- Centered hexagonal number (geometry)
- Cyclotomic polynomial structure (algebra)  
- F₇ squared + linear + constant (recursion of gauge depth)

Gravity involves *two-body mass interaction* → squared gauge coupling.

### Q3: How does mass emerge?

In EM: charge = topological winding number (integer quantized)  
In gravity: mass = resonance amplitude (continuous)

The difference: EM defects are phase singularities, mass is energy density.

---

## Experimental Plan

| Exp | Name | Tests |
|-----|------|-------|
| 01 | SEC wave unification | Same wave equation for EM and gravity |
| 02 | Projection duality | Antisymmetric→curl, symmetric→divergence |
| 03 | F183 hierarchy | G/α ≈ F₁₈₃ / F(EM indices) |
| 04 | Gravitational α | Define gravitational fine structure |
| 05 | Schwarzschild from SEC | Black hole as deep SEC collapse |
| 06 | Gravitational waves | Speed = c from SEC |
| 07 | Mass from resonance | Continuous vs quantized |
| 08 | Falsification tests | What would break this? |

---

## Success Criteria

1. **Derive G** from Fibonacci structure with <10% error on hierarchy
2. **Explain why gravity is attractive** (while EM has both signs)
3. **Predict gravitational wave speed = c** from SEC (confirmed by LIGO)
4. **Connect to recursive_gravity.py** informational tangle results

---

## Falsification Conditions

This hypothesis is FALSE if:

1. G/α ratio cannot be expressed as Fibonacci quotient
2. Projection symmetry doesn't map to curl/divergence split
3. Gravitational waves ≠ c (already falsified by GW170817!)
4. Black holes don't match SEC collapse predictions

---

## Dependencies

| Source | What We Use |
|--------|-------------|
| `maxwell_from_pac_sec/` | SEC wave equation, curl projection |
| `milestone1/` | F₁₈₃ hierarchy, falsification methodology |
| `recursive_gravity/` | Informational tangle simulations |
| `standard_model_connection/` | F₇ gauge closure |
| `pac_confluence_xi/` | Ξ balance operator |

---

## Connection to Existing Work

### From `recursive_gravity.py`:

```python
tangle_strength = np.exp(-dist)
```

This "informational tangle" that produces orbits without Newtonian gravity
may actually BE the F₁₈₃ recursion manifesting at macroscopic scales.

### From `maxwell_from_pac_sec/SYNTHESIS.md`:

> "The gauge hierarchy might literally BE the Fibonacci sequence, 
> with gravity at F₁₁ = 89 or deeper."

We now propose: **F₁₈₃**, not F₁₁, because 183 = F₇² + F₇ + 1.

---

*This is exploratory theoretical physics. All claims require validation.*
