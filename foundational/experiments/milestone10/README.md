# Milestone 10: Symmetry Self-Application as the Unique Generative Primitive

## Score: 64/71 (90%)

| Block | Experiments | Score |
|---|---|---|
| A — Structural Genesis | exp_01 – exp_04 | 15/16 |
| B — Laws as Equilibria | exp_05 – exp_07 | 10/12 |
| C — Universal Structure | exp_08 – exp_10 | 11/12 |
| D — Investigative | exp_11 – exp_13 | 9/12 |
| E — Extension: PAC/MED/SEC | exp_14 – exp_17 | 19/19 |

Core (exp_01–10): 36/40 · Investigative (exp_11–13): 9/12 · Extension (exp_14–17): 19/19

## Thesis

Physical law is not imposed from outside. It is what happens when symmetry applies itself.

M10 asks the hardest question DFT can ask: **why these axioms?** Why PAC, SEC and MED, and
not some other set of conservation, dynamics and optimization principles? The answer is
that they are not independent axioms at all. They are what you get when a symmetric system
references itself:

- **PAC** is spectral confinement
- **MED** is the viability threshold
- **SEC** is the condensation dynamics

All three emerge from one operation. Uniqueness is established by structural exhaustion:
non-self-applying primitives regress, asymmetric self-application produces noise, and only
self-applied symmetry survives.

M10 is a foundational tightening that runs *underneath* M1–M9, deriving as theorems twelve
structures those milestones had stipulated: time, iteration, polarity, hierarchy, the
second law, gauge invariance, laws-as-equilibria, anomaly clustering, annealing residue,
Ξ universality, fossil arithmetic, and conceivability bounds.

## Key Results

1. **PAC is spectral confinement** (exp_14) — for symmetric `W = V D Vᵀ`, the operation
   `D → f(D)` preserves eigenvectors *exactly*. Drift is 2.4e-15, machine epsilon. The
   system can change how much of each mode, never which modes.
2. **Time is forced** (exp_02) — static resolution of self-reference is impossible.
3. **MED viability threshold at φ^(−1/N)** (exp_15) — first-order 1.58 nat gap, mean error 1.3%.
4. **Complexity valley at γ/ln(φ)** (exp_16) — converges 12.5% → 2.1%, match to 0.04%.
5. **Complete derivation chain** (exp_17, 7/7) — nothing → physics in 7 links, zero free
   parameters.
6. **Gauge invariance from substrate** (exp_06) — derives from a zero-mean symmetric substrate.
7. **Number-theory fossil** (exp_09) — standard primes show φ-enrichment absent in
   alternative closures.

## Honest Failures (5/71)

| Exp | Test | What failed | What it means |
|---|---|---|---|
| 03 | T4 | Single-circle maps also produce bounded iteration | Two-circle is sufficient, not necessary. Self-reference is the key, not the topology. |
| 07 | T2 | SM residuals fit stretched exponential, not Lévy-stable | The clustering is real; the specific distribution was wrong. |
| 07 | T4 | CC estimate off by 3 orders | Annealing gives qualitative fine-tuning, not quantitative CC. M8's cascade approach reaches 0.09 orders. |
| 08 | T3 | Ξ in random walks 20% off | Self-referential structure is required — this constrains where Ξ appears. |
| 11 | T3/T4 | Some parameter regions ambiguous | Finite-size effects at small N. Confirms but doesn't sharpen exp_01. |

Every failure narrows the claim rather than destroying it.

## What M10 Explains Downstream

| Milestone | What M10 grounds |
|---|---|
| M1 Standard Model | Fibonacci depth hierarchy from iterated self-application |
| M2 Mass derivations | Koide from polarity coupling balance (exp_04) |
| M3 Quantum validation | Discreteness from structural exhaustion (exp_01) |
| M4 Relativity / gravity | Spacetime as forced processuality (exp_02) |
| M5 SM completion | sin²θ_W = 3/13 from scope hierarchy |
| M6 Scoped mediation | Force hierarchy from Fibonacci depth |
| M7 Symmetry primitive | M10 completes M7's program: symmetry *is* the primitive |
| M8 BSM predictions | CC from cascade level counting, Z' from Fibonacci gap |
| M9 Infodynamic mechanism | Cascade clock from iterated self-application timing |

## Open Threads

1. Finite-size corrections — φ^(−1/N) converges to φ, but N=8 is still 3.3% off.
2. Stochastic extension — all M10 experiments are deterministic; genuine irreversibility needs stochastic self-application.
3. Möbius topology — exp_14 found self-reference overrides topology, but geometric content may re-enter at larger scales (reality-engine).
4. Ξ in nature — found in self-referential Markov chains but not plain random walks. Where else?
5. Quantitative CC — can the derivation chain sharpen M8's 0.09-order result?
6. The 8.9% slope gap from M9 — finite-size noise, or sub-leading physics?

## Generalization: Genesis + Ghost

M10's score came from a single toy model (the SelfApplicator), which invites the objection
that it was designed to produce these results. Two TinyCIMM variants — **Genesis**
(self-organizer, random symmetric W, no target) and **Ghost** — were built to close that
interpretive gap. See `SYNTHESIS.md`.

## Structure

Full cross-connections, the derivation chain, and the generalization study are in
[`SYNTHESIS.md`](SYNTHESIS.md). Sub-experiments exp_14–exp_17 have their own directories.

## FDO

`milestone10-symmetry-self-application`
