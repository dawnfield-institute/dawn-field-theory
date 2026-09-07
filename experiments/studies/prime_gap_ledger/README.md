# The Prime Gap Ledger: the loop, the gap and the delta

**Status**: active · **Founded**: 2026-09-07 · **Score**: 0/3 (run 2026-09-07; three INCONCLUSIVE by the sealed rules — `journals/2026-09-07_exp01_outcomes.md`)

## The thesis

A prime gap is a cascade that truncates itself. The integers strictly between two consecutive primes are
each divisible by some prime up to √p, so the gap is the sieve by those primes running until it fails to
cover the next integer: a local instance with two ends, a cut-open loop rooted at p. It is not a Möbius
topology, and the only Möbius in the sieve is Möbius the function, a balanced signing of the divisor
lattice.

The global object is the **loop of units mod the primorial** P(y) = ∏_{p ≤ y} p: bounded, boundaryless,
naturally written in primorial (Chinese-remainder) coordinates — "the structure arithmetic forms
around". Its gaps are the *unit gaps*; their frequencies are the Hardy–Littlewood weights. The y-rough
integers of a window [x, 2x) are the loop read locally at depth u = log x / log y, and once y ≥ √(2x)
they are exactly the primes. **The delta between the local read and the loop, as a function of u, is the
object of this study** — not the gaps themselves. In the corpus's words: the loop is the global ledger,
the gap is the local cascade, and the delta is the third thing.

The known theorems that pin the words are gates, not results: Mertens (the loop's density
∏(1 − 1/p) ≈ e^{−γ}/log y), Buchstab (the local/global density ratio is e^{γ}ω(u), = e^{γ}/2 at u = 2 —
the classic 2e^{−γ} delta), Maier (the loop leaks into short intervals), Hardy–Littlewood (the loop's
pair weights), Lemke Oliver–Soundararajan (consecutive primes' residues are biased, the diagonal
repelled, the bias vanishing globally). What is registered is what none of them states.

## Registered relations (exp_01; `journals/2026-09-07_exp01_registration.md`)

| | relation | verdict |
|---|---|---|
| R1 | the mean-rescaled gap **shape** delta Δ(u; m) obeys a drift law across decades (successive-decade differences shrink at every evaluable live u) | **INCONCLUSIVE** — at u ≥ 3 the shapes agree within noise in every decade (converged, which the rule cannot confirm); at u ≤ 2.5 the metric is a bin-edge artifact for lattice gaps |
| R2 | the residue bias is a truncation phenomenon: the consecutive-survivor diagonal deficit δ(u; m) obeys the same drift law, decreases across decades at u = 2 (the primes), and meets the loop's own value at the deepest live u | **INCONCLUSIVE** — (a) CONFIRM at all six live u, (b) CONFIRM (0.176 → 0.103 from 10⁵ to 10⁹), (c) meets to 1e-4 in every decade but two floors sit below 1e-4 |
| R3 | the termination depth (max least-prime-factor over a gap's interior) against a matched null of random composites is scale-free | **INCONCLUSIVE** — ρ = 0.93, 0.95, 0.95, 0.96 (10⁶..10⁹), converged within floors from 10⁷ |

Live cells are those where the loop's period exceeds the window; where it does not, window = loop
exactly (a gate). Floors are half-splits, reported per cell.

## What the run recorded (not claimed — the seal did not score it)

**The window's residue bias depends on the depth y alone.** At every depth from y = 13 to y = 4,000 the
consecutive-survivor deficit mod 3 in the window equals the primorial loop's deficit at that depth to a
thousandth, whatever the decade (y = 100: 0.2217, 0.2243, 0.2225, 0.2232, 0.2231 across five decades against
the loop's 0.2231). **At the primes themselves the loop over-predicts by a constant:** at y = ⌈√(2x)⌉ the
loop's deficit exceeds the primes' by 0.0093, 0.0098, 0.0099, 0.0088 for x = 10⁶, 10⁷, 10⁸, 10⁹. So the
Lemke Oliver–Soundararajan bias is the loop's bias at the truncation depth minus a scale-free residual
ε(u) — zero for u ≥ 2.5, about 0.003 at u = 2.25, about 0.0095 at u = 2 — the local/global delta of the
transition structure, the same shape as Buchstab's delta for the density. The loop's exact deficits at
q = 3 are 5/12, 223/552, 2860783/8291520, … (k = 3, 4, 6, …). Gap interiors are smoother than random
composites of their decade (ρ ≈ 0.95). These are the objects of round 2, with a CONVERGED verdict class,
a lattice shape metric and a tolerance on "meets" — the three instrument lessons of this round.

## Lineage, and what is deliberately not reused

The corpus's earlier prime work is cited, not built on. `sec_prime_manifold`'s 1/φ threshold is a frame
artifact (STANDARDS §2.9's worked instance: the residue class chooses the constant).
`asymmetric_conservation`'s "PAC exact at every sieve step" defines composites as N − 1 − primes, an
identity (§2.8). Milestone 3's "prime cascade reachability" is a wave-coverage model whose Fibonacci-gap
clause has no matched null. `prime_growth_dynamics_v2` exp_06 found the gap-6 hub against a Cramér
reference at N = 5·10⁵. None of φ, Ξ or Fibonacci enters this study; nothing here leans on those
statistics.

## Scripts

| Script | Purpose |
|---|---|
| `core/rough.py` | the instrument: odd sieve; one segmented sieve for the window and for the loop (CRT-uniform residues); loop enumeration k ≤ 9; gap and shape histograms; Mertens product; Buchstab ω(u); transition matrices and the diagonal deficit (exact rationals on enumerated loops); the least-prime-factor table and termination depths with the matched null |
| `scripts/exp_00_gates.py` | G1–G8, run before the seal (loop counts; CRT pair formula; champion 6; Buchstab; Mertens at u = 2; sampler vs enumeration; LO–S sign; periodic cells) |
| `scripts/exp_01_cascade_truncation.py` | R1–R3 on decades m = 5..8 (m = 9 recorded only), CONFIRM / KILL / INCONCLUSIVE, append-only timestamped results |

## Discipline

Registered before running (§2.7): relations only, thresholds fixed, gates on the record before the seal,
outcomes cite the seal's commit, results append-only, floors reported, a script that diverges from the
seal is corrected toward the seal. Postdiction disclosed in the registration's §0.
