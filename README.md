# Dawn Field Theory

> When a hammer shatters glass, thermodynamics tells us where the energy goes — heat, sound,
> kinetic motion. But **new information was created**: each shard now has unique geometry,
> distinct edges, specific boundaries. Standard physics has no framework for where that
> structural information comes from.
>
> Dawn Field Theory is an attempt at the missing half — how information organizes,
> crystallizes, and drives the emergence of structure across scales.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17295102.svg)](https://doi.org/10.5281/zenodo.17295102)

An independent research programme by the Dawn Field Institute. Everything here is
computational: derivations, falsification tests, and an honest record of what failed.

---

## Start here

| If you want | Read |
|---|---|
| **The argument** — every claim, and where each is proved, tested and published | [`THEORY_MAP.md`](THEORY_MAP.md) |
| **What's being worked on now**, and what would falsify it | [`ROADMAP.md`](ROADMAP.md) |
| **The theory itself** | [`theory/dawn-field-theory.md`](theory/dawn-field-theory.md) |
| **What's proven** — as distinct from measured | [`formal/theorems/`](formal/theorems/README.md) |
| **Every experiment**, by status | [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) |
| **What a term means and where it came from** | [`theory/lexicon.yaml`](theory/lexicon.yaml) |
| **What we got wrong** | [`theory/corrections.md`](theory/corrections.md) |
| **How the framework got here** | [`timeline.md`](timeline.md) |

---

## Two axioms

| | Statement | Consequence |
|---|---|---|
| **PAC** — Potential-Actualization Conservation | f(parent) = Σ f(children) | Unique stable solution φ^(−k). The golden ratio is not found; it is forced. |
| **SEC** — Symbolic Entropy Collapse | ∂S/∂t = α∇I − β∇H | Structure forms where information gradients dominate entropy gradients. |

Two further pillars — **RBF** (geometry, far-from-equilibrium regulation) and **MED**
(optimization, bounded complexity) complete the framework. Milestone 10 showed that PAC,
SEC and MED are *not* independent axioms: all three fall out of one operation, self-applied
symmetry. RBF was not part of that derivation; its only milestone test is M7 exp_08, at
2/4 with two documented failures.

```
PAC → φ cascade → ln(φ) per level → Ξ = γ + ln(φ)
    → Fibonacci structure → Standard Model parameters
    → ADE geometry → complement → Lorentz → quantum mechanics
```

## What is established

Selected results. The full set, with scores and honest failures, is in
[`THEORY_MAP.md`](THEORY_MAP.md).

| Result | Precision | Where |
|---|---|---|
| α_EM from a Fibonacci formula — #1 of 10,440 candidates | 5.7 ppm | M6 |
| sin²θ_W = tan(θ_C) = 3/13 | 0.19% | M1, M5 |
| Higgs mass, λ = φ/4π | 83 ppm | M5 |
| μ/e and p/e mass ratios | 5 ppm · 0.0083% | M2 |
| Feigenbaum constants from closed form | 13 digits | `sec_threshold_detection` |
| S8 tension | 3.22σ → 0.09σ (blind) | M9 |
| Planck scale from cascade depth-183 | zero free parameters | M11 |

And three results that are **proven** rather than measured:

- **Ξ = γ + ln(φ) is fully determined** — φ uniquely selected by gravity-time duality,
  γ by harmonic counting. Zero free parameters.
- **PAC is spectral confinement** — eigenvector fixity exact to 2.4×10⁻¹⁵.
- **C₆ = −I**, and the M15 connection generator is the particle-in-a-box momentum operator.

## What is not established

Kept in the open, because a framework that only advertises its wins cannot be checked.

- **Holonomy may be dynamically inert.** M15 Phase 2 carries a standing kill-sentence: if
  it is, the milestone caps at a reclassification and is mathematics, not physics.
- **The coherence limit is not universal** — tested, 0/4, and recorded as falsified.
- **No isomorphism-invariant positive-definite metric on ADE exists.** A proven
  impossibility, and the lemma M15 is built on.
- **Open ends** carried honestly: CP violation at 3%, M9's 8.9% slope gap and the DESI w(z)
  tension, MED's finite-size correction.
- Roughly 60% of M11's tests are structural — internal consistency, not empirical
  validation. Hard observational tests await LISA, CTA and Euclid.

[`theory/corrections.md`](theory/corrections.md) is the standing record of claims made too
strongly and later withdrawn — including a "universal constant" that turned out to be a
local measurement.

## How the work is done

Claims are **pre-registered**: hypothesis, quantified thresholds and falsification
conditions committed *before* the run, outcomes committed separately citing the
registration hash. Thresholds are never relaxed after seeing results, and a test that
passes for a reason unrelated to what it guards is replaced rather than counted — which is
why hardening cycles sometimes *lower* a score.

Claims register **invariants, never absolute coordinates**. Registered relations survive;
registered coordinates die.

Full specification: [`STANDARDS.md`](STANDARDS.md).

## Structure

The tree follows the argument.

| | |
|---|---|
| [`theory/`](theory/) | what is claimed — framework, constants, lexicon, corrections |
| [`formal/`](formal/) | why it holds — theorems, derivations, conjectures |
| [`experiments/`](experiments/) | what was measured — milestones, sidecars, studies, spikes |
| [`papers/`](papers/) | what was published — PACSeries and standalone papers |
| [`archive/`](archive/) | lineage, by era — preserved, not deprecated |

[`INVENTORY.md`](INVENTORY.md) is the generated corpus view;
[`MIGRATION.md`](MIGRATION.md) maps every path from before the August 2026
reorganization.

## Ecosystem

| Repo | Role |
|---|---|
| `dawn-field-theory` | the physics — this repository |
| `fracton` | PAC mathematics library |
| `reality-engine` | simulator implementing DFT dynamics |
| `dawn-models` | GAIA and ML validation |

## Publications

PACSeries on Zenodo — concept DOI [10.5281/zenodo.17295102](https://doi.org/10.5281/zenodo.17295102),
latest v0.3. Papers, registries and DOIs are in [`papers/`](papers/).

## Contributing

See [`CONTRIBUTION.md`](CONTRIBUTION.md). Critique is the contribution most wanted: this
is a framework designed to be falsifiable, and the corrections registry is treated as a
first-class artifact.

Licensed under AGPL — see [`LICENSE`](LICENSE) and [`legal/`](legal/).

---

*© 2026 The Dawn Field Institute · [dawnfield.ca](https://dawnfield.ca) · info@dawnfield.ca*
