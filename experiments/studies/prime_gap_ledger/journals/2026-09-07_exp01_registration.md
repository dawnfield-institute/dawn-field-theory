# exp_01 registration — the loop, the gap and the delta

**Date:** 2026-09-07 · **Layer:** mathematics / arithmetic (a study; no physics is claimed).
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_00_gates_20260907_114513.json`;
the first gate run, `_114418.json`, is kept — see §2). Run after; scored to this text. Kills have the scopes in §5.
**Target script:** `scripts/exp_01_cascade_truncation.py` · **Gate script (passed first):** `scripts/exp_00_gates.py`.

## §0 Postdiction disclosure

Seen before this seal:
- The reasoning of the conversation that opened the study (Peter: a gap is a self-terminating cascade; the global object
  is "limitless but bounded"; that is what a *unit* gap looks like), made precise as: local = the y-rough numbers of a
  window; global = the loop of units mod P(y); the delta as a function of the depth u = log x / log y.
- An independent review of the design (reasoning only, no numerics): the thinning model (a raw gap-histogram delta is
  Buchstab's density ratio in disguise — hence *shapes*); the Lemke Oliver–Soundararajan expansion (leading term
  ∝ log log x / log x — hence no "bias × log x" clause); Buchstab's least-prime-factor law (hence a matched null for
  the termination depth); the unsoundness of sampling the loop at an explicit huge offset (hence CRT-uniform residues).
- A 10⁸ sieve timing (5,761,455 primes; max gap 220; mean 17.36) — no registered quantity.
- The gates of §2, with their values.
- **A smoke run at decades m = 3 and m = 4** (`--windows 4 --L 10⁶`, out of the scored set; written to the scratchpad,
  not to `results/`). It showed the instrument end to end and returned INCONCLUSIVE on every relation, as floors at those
  sizes should. **Its m = 4 values were seen** — the diagonal deficit at u = 2 (0.198), ρ (0.968), the shape deltas — so
  **m = 4 is recorded and never scored**, and R2(b) starts at m = 5. Two instrument facts learned from it are declared:
  an enumerated loop is exact and carries floor 0; a loop with too few units has no diagonal expectation (NaN, excluded).

Not seen: any quantity at m ≥ 5 other than the gate densities and the LO–S *sign*. No shape delta, no deficit magnitude,
no termination depth at any scored decade.

## §1 Objects and instruments (closed at this seal; counting basis §6)

- **Primes** to 2·10⁹ by odd sieve. **Decades** D_m = [10^m, 2·10^m), m = 4..9. **Scored: m = 6, 7, 8, 9.**
  Recorded, not scored: m = 4 (seen) and m = 5 (its half-split floor ≈ 0.07 on 61 shape bins would decide nothing).
  *Declared fallback:* if m = 9 cannot be computed on this machine (memory), the scored set is m = 5..8 and every m = 5
  floor is reported beside its verdict — decided now, not after a result.
- **Depth grid** u ∈ {6, 5, 4.5, 4, 3.5, 3, 2.5, 2.25, 2}; y = round(x^{1/u}), and at u = 2 exactly y = ⌈√(2x)⌉ so that
  the window's y-rough numbers are the primes. A cell is **periodic** when P(y) ≤ L (window = loop, G8) and **live**
  otherwise. Only cells live in every scored decade are scored, and u = 2 is never scored in R1 (it is G5).
- **Local:** `segmented_rough` on [10^m, 2·10^m) with the residues of 10^m. **Loop:** the same routine with residues
  drawn uniformly and independently per prime p ≤ y (CRT-uniform on the loop); enumerated whole for k ≤ 9 primes;
  otherwise W = 25 windows of L = 2·10⁷ (seed 20260907). N mod 4 for a sampled loop is the drawn residue at 2 lifted
  uniformly to a class mod 4; N mod 10 is the CRT of the draws at 2 and 5.
- **Shape:** gaps in units of the sample's own mean, fixed edges 0..6 by 0.1 plus overflow (61 bins). **Δ(u; m)** =
  total-variation distance between the local and the loop shapes. **Floor** of a cell = max(TV between the two halves of
  the window's gap sequence, TV between the two halves of the sampled loop's windows); 0 for an enumerated loop.
- **Residue bias:** transition counts of consecutive survivors' residues mod q ∈ {3, 4, 10}; **δ** = 1 − (diagonal mass)
  / (diagonal mass expected under the product of the marginals). Scored at q = 3; q = 4, 10 recorded. Exact rational on
  enumerated loops where q | P(y). Floor = max of the two half-split |δ_a − δ_b| (window; loop windows; 0 if enumerated).
- **Termination depth:** d(p) = max least-prime-factor over the interior of the gap at p (lpf table of the decade over
  primes ≤ √(2·10^m)); **matched null:** the max lpf of the same number of independent composites drawn uniformly from
  the same decade (primes rejected). **ρ(m)** = median(log d/log p) / median(log d_null/log p). Floor = |ρ on the first
  half of the gaps − ρ on the second half|.
- **The drift law** (one rule for every relation): over the scored decades the successive differences
  D_m = |value(m) − value(m+1)| must each exceed max(floor(m), floor(m+1)) and **shrink strictly with m** (three
  differences for four decades). CONFIRM = all above floor and shrinking; KILL = all above floor and strictly growing;
  INCONCLUSIVE otherwise. A relation over the u-grid is CONFIRM when ≥ 3 live u are evaluable and every evaluable u
  confirms; KILL when ≥ 2 live u kill; INCONCLUSIVE otherwise.

## §2 Gates (all PASS before this seal; `results/exp_00_gates_20260907_114513.json`)

| gate | claim | value |
|---|---|---|
| G1 | loop count = φ(P_k), cyclic gaps sum to P_k, k ≤ 9 | exact, 9/9 — **first run failed at k = 1**: a one-unit loop got no wrap gap; `gaps_of` corrected (the single unit's gap is the period) |
| G2 | the enumerated loop's pair count at distance g = ∏_{p\|g}(p−1)·∏_{p∤g}(p−2), even g ≤ 30, k ≤ 8 | 120/120 exact |
| G3 | the champion gap is 6 in every decade, beating both 2 and 4 | 5/5 — **first draft demanded "6 > 4 > 2" and failed at every decade: a false known answer of mine.** Gaps 2 and 4 carry the same Hardy–Littlewood weight (no odd prime divides 4); the gate now demands only that 6 beats both, and records h₂, h₄ |
| G4 | local/loop density ratio = e^{γ}ω(u), 32 cells m = 5..8, within 3/log y | 0 failures; ω(2) = 0.5000, ω(8) = 0.5615 = e^{−γ} |
| G5 | at u = 2 the ratio → e^{γ}/2 = 0.8905 | 0.9188, 0.9114, 0.9088, 0.9058 at m = 5..8 (the known drift toward the limit) |
| G6 | the CRT-uniform sampler reproduces the enumerated loop, k = 9 | TV 9.8e-6 (half-split 2.6e-5) |
| G7 | LO–S sign: the diagonal is the least frequent transition at q = 3, 4, m = 4..8 (sign only printed) | every row, every decade |
| G8 | periodic cells: window = loop | 4 cells, max TV 6.8e-5 |

The gates fixed the words. Nothing in §3 is decided by them.

## §3 Registered relations (M = 3; relations, not coordinates)

**R1 — the loop reading is scale-free in shape.** Δ(u; m) obeys the drift law at every evaluable live u over m = 6..9.
The crossover u_c(m) where Δ first exceeds 0.05 is recorded, not scored. *Bearing (not scored):* the thinning model
predicts CONFIRM; a KILL would contradict the Hardy–Littlewood thinning picture of the primes and is said so.

**R2 — the residue bias is a truncation phenomenon.** (a) δ(u; m) at q = 3 obeys the drift law at every evaluable live
u; (b) at u = 2 (the primes) δ decreases strictly across m = 5..9; (c) at the largest live u in every scored decade
|δ(u; m) − δ_loop(y)| ≤ floor. CONFIRM = (a), (b), (c) all CONFIRM. KILL = (b) fails, or (c) fails in every scored
decade. INCONCLUSIVE otherwise. Recorded, not scored: the q-ratio δ₁₀/δ₃ at u = 2 (the LO–S leading term predicts 3; the
next term is unknown to us) and the loop's exact rational biases for q | P_k, k ≤ 9. *Bearing:* (b) is LO–S's
conjecture, numerically established; (a) and (c) are this study's content.

**R3 — each gap is a cascade that truncates itself, against the matched null.** ρ(m) obeys the drift law over
m = 6..9; its value is recorded, not scored. *Bearing, honest:* ρ = 1 would say a gap's interior is a random set of
composites of its decade; ρ ≠ 1 says the interior is sieve-conditioned beyond compositeness; the smoke saw ρ ≈ 0.97 at
m = 4 (disclosed), so the registered content is the scale-freedom, not the value.

## §4 What would count as vacuous

No live u in every scored decade (impossible on this grid: u ∈ {4.5, 4, 3.5, 3, 2.5, 2.25} are live from m = 6 up); a
depth that cannot be computed at a scored decade (then R3 is INCONCLUSIVE, said so); every difference below its floor
(INCONCLUSIVE, said so — floors are reported per cell so the reader sees why).

## §5 Kill scope

- R1: "the local gap shape is the global loop's shape truncated at depth y" as a scale-free statement. The theorems of
  §2 are untouched; the Hardy–Littlewood *thinning* picture would be wounded, and that is reported, not softened.
- R2: "the Lemke Oliver–Soundararajan bias is the loop's own bias, truncated at u = 2". LO–S itself is untouched.
- R3: "a gap's interior is a random set of composites" (KILL means it is not; CONFIRM means the deviation is scale-free).
- Nothing here touches any milestone, Ξ, φ, or any physics.

## §6 Counting basis and outputs

Survivors per cell, windows per sampled depth (25 × 2·10⁷), gaps per decade; every floor beside its value. Outputs:
`results/exp_01_cascade_truncation_<tag>_<ts>.json` + `_log.txt`, append-only, checkpointed after every decade;
outcomes in `journals/2026-09-07_exp01_outcomes.md` citing this seal's commit.

## §7 Registered threats to validity

- **Floors.** The window of a decade is what it is; at m = 6 the shape floor is ≈ 0.02. A relation can be INCONCLUSIVE
  by floors alone; that is a statement about the instrument's reach at 10⁹, reported as such.
- **The LO–S next term.** (b) is monotone decrease only; no rate is registered because the second coefficient is not
  known to us.
- **Memory at m = 9.** The fallback of §1 applies; decided now.
- **Rounding of y to a prime is not done** (y = round(x^{1/u}) and primes ≤ y are used) — declared; it moves k by at
  most one at small y and nothing at large y.

## Outcome commitment

CONFIRM, KILL or INCONCLUSIVE, in any mix, recorded in the outcomes journal citing this seal, pushed to the same PR,
folded into the study README, THEORY_MAP, the Lore node and memory regardless of direction. Thresholds and rules above
are final; any post-registration edit to them voids the affected relation.

---

**Forward note (2026-09-07, before any scored quantity was read).** Layer: arithmetic. If R2 holds, bundle 4 for Andy
Farmer gains one document: the loop's exact residue bias and the statement that his "keys are identifiers for the gaps"
has a precise form — the keys are residues in the primorial base, the values are the unit gaps, and the primes are
where the cascade of moduli stops. If it does not hold, the delta that fails to collapse is the document.
