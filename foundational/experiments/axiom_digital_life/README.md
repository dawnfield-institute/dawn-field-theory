# Axiom-Seeded Digital Life

**Status**: Exploratory  
**Version**: v0.1  
**Started**: 2026-04-15

---

## The Idea

Traditional artificial life simulates *physics* (gravity, friction, chemistry) and
hopes life emerges. This experiment takes a different approach:

> **Seed the world with DFT axioms. Life is a consequence — not of physics — but of
> information dynamics.**

The world has two fundamental rules (the DFT axioms), and nothing else:

| Axiom | Statement | Role |
|-------|-----------|------|
| **PAC** | `f(Parent) = Σ f(Children)` | Conservation: information is never created or destroyed |
| **SEC** | `∂S/∂t = α∇I − β∇H` | Dynamics: structure forms where information gradient dominates entropy |

From these two axioms — and the operator algebra `{⊕, ⊗, δ, Ξ}` — all life
behaviors emerge:

- **Metabolism** = absorption of I from the field, excretion of H (living costs entropy)
- **Movement** = SEC gradient ascent toward highest local ∇I (not physics — no mass/velocity)
- **Reproduction** = ⊗ entropic branching when Ξ_local > Ξ_threshold
- **Death** = ⊕ collapse merge when entropy dominates; PAC budget returns to field
- **Evolution** = genome mutation across generations within MED bounds

---

## DFT Constants (not tuned)

| Constant | Value | Source |
|----------|-------|--------|
| φ (golden ratio) | 1.618... | PAC unique stable solution |
| Ξ (balance operator) | 1.057 | Validated in `cellular_automata_pac_attractors` (p < 8.58×10⁻⁸) |
| ln(φ) | 0.481 | Information per recursion level |
| PAC split | φ⁻¹ : (1-φ⁻¹) = 0.618 : 0.382 | Budget ratio on reproduction |
| MED depth | ≤ 2 | Macro Emergence Dynamics bound |
| MED nodes | ≤ 3 | Macro Emergence Dynamics bound |

The reproduction threshold `θ_r = Ξ = 1.057` is **not** a free parameter —
it's the same constant that emerged from the Rule 110 cellular automaton.

---

## Creatures: Infobionts

Each Infobiont is a PAC tree node:

```
Infobiont {
  pos:    int      # Position on 1D field [0..63]
  pac:    float    # PAC budget (share of conserved information)
  genome: {
    alpha:   float  # Information absorption rate
    beta:    float  # Entropy excretion rate (metabolic cost)
    theta_r: float  # Reproduction threshold (seeded at Ξ=1.057)
    theta_d: float  # Death threshold (I/H below this → collapse)
    mu:      float  # Mutation magnitude
  }
  depth:  int      # Lineage depth (MED bound: ≤ 2)
  age:    int      # Steps lived
}
```

---

## World Architecture

```
1D field [64 cells]
  I[x]: information potential
  H[x]: entropy

PAC conservation: Σ I[x] + Σ H[x] + Σ creature.pac = C_total  (constant)

SEC dynamics each step:
  1. Creatures metabolize: absorb I[pos], excrete H[pos]
  2. Information diffuses: ∇²I term (spreading)
  3. Entropy grows at information edges: H += γ |∇I| dt
  4. PAC enforcement: renormalize to C_total
```

---

## Experiment Structure

```
axiom_digital_life/
├── scripts/
│   └── exp_01_axiom_digital_life.py   # Main prototype (self-contained)
├── results/
│   └── exp_01_axiom_digital_life.json  # JSON output (time-series + survivors)
├── journals/                            # Research logs
├── meta.yaml                            # Experiment metadata
└── README.md                            # This file
```

---

## Running

```bash
cd /workspace/dawn-field-theory
python foundational/experiments/axiom_digital_life/scripts/exp_01_axiom_digital_life.py
```

No external dependencies beyond `numpy`.

---

## Predictions (Falsifiable)

1. **PAC conservation holds**: max drift < 0.5% across all timesteps
2. **Reproduction occurs**: ⊗ branching events observed (births > 0)
3. **Collapse occurs**: ⊕ merge events observed (deaths > 0)
4. **Ξ clustering**: surviving population has mean local Ξ near 1.057
5. **Genome evolution**: offspring genomes diverge from seed values over generations

---

## Related Work

| Experiment | Connection |
|-----------|------------|
| `cellular_automata_pac_attractors` | Source of Ξ=1.057 (used as θ_r here) |
| `symbolic_emergence` | Agent interaction producing symbolic structure |
| `biology_experiments/evolution-symbolic-collapse` | Evolution via SEC collapse |
| `pre_field_recursion` | Pre-field recursion dynamics |

---

## Connections to Marletto (2015)

Constructor Theory of Life shows that life requires *digital information* to be
physically instantiable — not any specific physics. DFT's PAC axiom is exactly
such a digital information substrate: discrete, conserved, copyable. This
experiment tests whether the constructor-theoretic conditions for life
(self-reproduction, heritable variation) arise from PAC/SEC axioms alone.
