# Conjectures — attempted, unproven

Kept, not hidden. These are the record of what was tried in the framework's first formal
push, and several of the questions they opened are still live — they were reached from a
different direction later, by the milestone stack.

## Why these are filed as conjectures

They were previously in a directory called `proofs/`, and three of them carry `theorem` or
`proof` in the filename. **Their own headers disagree:**

| Document | Its own status line |
|---|---|
| `med-proofs/01_sec_navier_stokes_equivalence.md` | "**Conjecture 1.1**… We explore whether" |
| `med-proofs/02_bounded_complexity_regularity.md` | "**Research Conjecture**… We investigate whether" |
| `med-proofs/03_optimal_parameter_convergence_theorem_v0.1.md` | "Status: **Computational Investigation**" |
| `med-proofs/04_resolution_independence_theorem_v0.1.md` | "Status: **Computational Investigation**" |
| `med-proofs/proof_balance_operator_stability_v0.1.md` | "Status: **Mathematical Framework Development**" |
| `med-proofs/proof_universal_bounded_complexity_v0.1.md` | "Status: **Computational Investigation**" |

Nothing here was demoted. The directory name was corrected to match what the documents
had been saying about themselves since the day they were written — all eight on
**2025-08-20**, and untouched since.

The overclaim had in fact already been caught once internally:
`04_resolution_independence_theorem_v0.1.md` and
`04_resolution_independence_investigation_v0.2.md` carry the **same title**, and v0.2
renamed itself from "theorem" to "investigation". The v0.1 file was simply never retired.

## What became of these questions

Several were later reached from another direction, which is the reason to keep them
readable rather than archive them out of sight:

- **SEC ↔ Navier-Stokes** (`01`) — re-founded on milestone machinery in the `ade_cascade`
  work and the DNS instrument, and audited in the 2026-07-17 turbulence/cascade review.
- **Bounded complexity** (`02`, `proof_universal_bounded_complexity`) — MED's depth ≤ 2 /
  nodes ≤ 3 bound was derived as a **viability threshold at φ^(−1/N)** in Milestone 10
  exp_15, with a first-order 1.58 nat gap and 1.3% mean error.
- **Balance operator stability** — the RBF strand, now carried by
  `theory/` and the milestone stack.

That is the pattern this repository runs on: a question opened here, left unproven, and
answered later by a different method. The conjecture is the first half of that record.

## Contents

| | |
|---|---|
| `med-proofs/` | The eight 2025-08-20 documents, text unchanged |
| `med-formal-papers/` | MED mathematical foundations and framework integration |
| `med-notes/` | Working notes behind both |
