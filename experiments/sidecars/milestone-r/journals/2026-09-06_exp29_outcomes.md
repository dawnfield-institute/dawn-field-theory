# exp_29 outcomes — the ledger holds the web; whose web it is depends on the size

**Registration:** `journals/2026-09-06_exp29_registration.md`, sealed at dawn-field-theory
**`43e4ebc9`**. Instruments and runs: reality-engine `feat/v4-pac-ledger` (stacked on PR #10) —
ledger at `c5e961e`, scaffold and pre-seal artifacts at `78d59b3`, grids at `7b06fea` (PR reality-engine #11). Scored by
`scripts/exp_29_pac_ledger.py` against the sealed thresholds (`--selftest` OK); a test passes only
if it passes at every size present. Grid JSONs (per-run SHA256s) and the scored JSON
(`results/exp_29_pac_ledger_20260906_180850.json`) are in `results/`. Thresholds unchanged.

## Score: 3/4, and the kill fires by the letter

| Test | Proxy (n = 1000, 3 seeds) | n = 4000 (3 seeds) |
|---|---|---|
| T1 holding at κ = 0.5 above **both** controls, 3/3, > 2× pooled σ | **FAIL.** vs κ = ∞: 0.758 / 0.764 / 0.509 against 0.101 / 0.087 / 0.061 — 3/3, margin 0.594, σ 0.104 (5.7 σ). vs κ = 0: against 0.688 / 0.817 / 0.429 — **2/3**, margin 0.032, σ 0.174 (0.2 σ) | **FAIL.** vs κ = ∞: 0.622 / 0.441 / 0.778 against 0.035 / 0.038 / 0.028 — 3/3, margin 0.580, σ 0.120 (4.8 σ). vs κ = 0: against 0.395 / 0.315 / 0.608 — **3/3 seedwise**, margin 0.174, σ 0.160 (**1.1 σ**, short of 2) |
| T2 bound: KE/\|U_grav\| < 1 at every window mark, κ = 0.5 and 1 | **PASS.** maxima 0.51 / 0.53 / 0.50 and 0.32 / 0.35 / 0.32 | **PASS.** maxima 0.53 / 0.54 / 0.53 and 0.33 / 0.34 / 0.36 (today's substrate: 18.5 / 22.3 / 20.0) |
| T3 the ordering above the threshold, KE/\|U\| at κ = 1 < 2 < ∞ | **PASS** 3/3: 0.29–0.31 < 0.39–0.43 < 10.7–18.3 | **PASS** 3/3: 0.32–0.33 < 0.58–0.59 < 18.5–22.3 |
| T4 the ledger did the work | **PASS.** exact; work within allowance (−1.2 P₀ at κ = 0.5); the budget bound on every would-grow particle 3/3 | **PASS.** exact; work −4.2 P₀ at κ = 0.5, +0.28–0.30 at κ = 2; bound 3/3; 95–96% of the κ = 0.5 budget spent |

Gates at both sizes: transfer residual ≤ 10⁻⁶, total-ledger closure ≤ 10⁻², all runs finite,
`at_cap` ≤ 0.011. Vacuity: the budget binds; gravity alone exceeds the random floor (0.051 ± 0.014
proxy, 0.047 ± 0.013 full) by far — pre-declared, the frame was already the difference over G0.
**Kill, as registered:** *T1 fails at κ = 0.5 (2/3 on the proxy; 3/3 at n = 4000 but 1.1 σ against
a 2 σ bar), so PAC-on-particles as mapped here — a per-particle budget priced by exp_09's pair
energy — is retired as the object that holds structure beyond gravity alone at κ = 0.5.* The
theory is untouched; the ledger instruments are gated and stay. Scorecard: **62/116 → 65/120.**

## What the arms actually did

**The ledger works, at both sizes.** With the budget on, the total `KE + U_grav + E_SEC + ΣP` is
conserved to truncation, the transfer is exact, the budget binds on every particle that would grow,
and the substrate settles bound: KE/|U_grav| 0.49–0.54 at κ = 0.5 and 0.29–0.36 at κ = 1, against
10.7–22.3 unbounded. exp_28's relaxation oscillator is gone. The virial arithmetic stated before the
run — bound through κ = 1, the ordering monotone above it, the U-shape with its minimum at κ = 1 —
is what all six seeds show. The step never shrinks below `dt_ref` at κ ≤ 2.

**The web survives, at both sizes.** Whole-set percolation over t ∈ [10, 15] at κ = 0.5 is
0.51–0.76 on the proxy and 0.44–0.78 at n = 4000, ten to twenty times the unbounded engine and
ten times the random floor. Where exp_28's substrate held nothing, the ledgered substrate holds a
web.

**Whose web it is depends on the size — and that is the finding.** On the proxy the bounded engine
adds nothing to gravity's web at any κ and removes it monotonically: 0.51–0.76 (κ = 0.5) → 0.40–0.53
(1) → 0.15–0.28 (2) against gravity's 0.43–0.82. At n = 4000 it **adds**: κ = 0.5 sits above gravity
in every seed (0.62 / 0.44 / 0.78 vs 0.40 / 0.32 / 0.61), κ = 1 sits higher still — **0.76 / 0.70 / 0.80,
above gravity in every seed by a margin of 0.315 against a pooled σ of 0.113, 2.8 σ** — and κ = 2
collapses to 0.09–0.14. The full-size ordering is 1 > 0.5 > 0 > 2 > ∞. **This is exploring, not
predicting**: κ = 1 was never registered for T1, and the 2.8 σ is a post-hoc number on the three
registered seeds. It is written here so the next registration can be made on fresh seeds.

**Why the sizes disagree, plainly.** The pressure's range is 2r0 = 20 at both sizes. The proxy's box
is 37.8, so the repulsion reaches more than half the box and every particle feels most of the
others; at n = 4000 the box is 60 and the range is a third of it. The registration named the proxy
as the decider and did not declare the ratio box / 2r0 as a condition. On the proxy the bounded
pressure is a box-spanning restoring force (work −1.2 P₀); at n = 4000 it is a local one (−4.2 P₀
at κ = 0.5, −0.6 to −1.4 at κ = 1), and local repulsion between cores is what holds filaments open.
That reading is a hypothesis; it is the natural thing to register.

**Side predictions.** SP1 holds at both sizes (pressure work < 0 at κ = 0.5). SP2 holds (spent
fraction falls with κ). SP3 holds (no step shrink at κ ≤ 2). SP4 holds.

## The reading

exp_28 found the substrate could not hold because its engine was unbounded. exp_29 bounded the
engine and the web survived — at both sizes, by 4.8–5.7 σ over the unbounded substrate — and the
registered question of whether it survives *beyond gravity* got the answer "not at κ = 0.5 on the
proxy, and not by a 2 σ margin at n = 4000". By the seal that is a fail and the kill fires: the
mapping is retired *as the registered structure-holding object at κ = 0.5 with the proxy deciding*.

What the full grid then shows, and the seal does not score, is that the mapping was tested at the
wrong budget and in a finite-size regime. At n = 4000 and κ = 1 the ledgered engine holds more web
than gravity alone in every seed, at 2.8 σ, and less than gravity at κ = 2. A bounded, density-sourced
repulsion is therefore *not* structurally inert; on a box that is large compared with its range it
adds web up to about one binding energy of budget and destroys it beyond. The pressure's *form* —
the fracton SEC functional's gradient-penalising term — remains a candidate for a better pressure;
it is no longer the only lead, and the proxy that pointed at it was the wrong instrument for the
question.

## The bearing

- **R1b, registrable now:** at n = 4000, seeds {4, 5, 6}, κ ∈ {0.5, 1, 1.5}: the ledgered engine at
  κ = 1 holds more web than gravity alone, 3/3, > 2 σ (the post-hoc 2.8 σ is the expected direction,
  disclosed as such); the box-to-range ratio declared as a condition (box ≥ 3 · 2r0); the proxy
  retired as a decider for pressure-range questions. A one-day round.
- **Milestone R:** the ledger the thesis needs exists on particles, is exact, and the substrate it
  gives is bound. R2 (severance on a bound substrate) runs on κ = 1 at n = 4000.
- **The pressure's form** stays on the list, behind R1b.
- **M18's dynamics question** waits on a substrate that holds a *φ-coupled* web; a bounded engine
  that holds more than gravity is the first substrate on which that question could be asked.

## Process notes

Three registration clauses moved before the seal on calibration numbers, each with its reason in
the registration's §0. The seed-9 smoke test's structure numbers are in §0.4. Two instrument
corrections found by the gates (float64 for the gradient check; the exact `XI_ANALYTIC/PHI` for
the anchor, which then reproduces exp_28's baseline to 0.00). One test was committed failing
behind a pipe that masked its exit code and amended within the hour. The proxy-only draft of this
journal read "the web is gravity's, and the pressure can only take structure away"; the full grid
contradicted the second half within the hour and this journal was rewritten before anything was
pushed. The registration's omission — no declared box-to-range ratio — is the lesson carried
forward. No threshold moved after any result.
