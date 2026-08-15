# Formal — why it holds

The layer between what the framework *claims* ([`theory/`](../theory/)) and what was
*measured* ([`experiments/`](../experiments/)). Three tiers, and the distinction between
them is the point:

| | |
|---|---|
| [`theorems/`](theorems/) | **Proven.** Each entry states the result and points to the derivation and the experiment that verified it. |
| [`derivations/`](derivations/) | **Chains.** Axiom → constant. How φ, ln φ and Ξ are reached from PAC/SEC rather than fitted. |
| [`conjectures/`](conjectures/) | **Attempted, unproven.** Kept because they are the record of what was tried. |

## Why this layer was rebuilt

It had inverted. Every document in what was `arithmetic/macro_emergence_dynamics/proofs/`
was written on **2025-08-20** and never touched again — and each describes itself, in its
own header, as a *"Conjecture"* or a *"Computational Investigation"*. The directory was
called `proofs/`; three filenames said `theorem` or `proof`. The contents said otherwise.

Meanwhile the results that *are* proven — the holonomy closed form, C₆ = −I, PAC as exact
eigenvector fixity, the origin of Ξ — were living in milestone **experiment journals**,
because that is where the work happened. The formal layer stopped being maintained when
the milestone stack took over, and nothing recorded the handoff.

So the tiers are named for what the documents actually are. The Era-1 MED material is
under `conjectures/` with its text untouched; it was never demoted, only labelled to match
what it already said about itself.

## Theorems index into their journals

`theorems/` does not restate proofs. Each entry is a pointer: statement, status, the
journal where the derivation lives, the experiment that verified it. One source of truth —
a second copy of an argument is a second thing that can drift, and this repository has
already paid for that twice (two `meta.yaml` specs, two journal specs, two conflicting
legends for the CIP filename scheme).

The journals stay where they are. They are dated evidence of when something was
established, and under the pre-registration protocol ([`STANDARDS.md`](../STANDARDS.md)
§2.7) that date is part of the claim.
