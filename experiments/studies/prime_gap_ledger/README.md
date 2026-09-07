# The Prime Gap Ledger: the loop, the gap and the delta

**Status**: active · **Founded**: 2026-09-07 · **Score**: 0/3 (registered, not run)

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
| R1 | the mean-rescaled gap **shape** delta Δ(u; m) obeys a drift law across decades (successive-decade differences shrink at every evaluable live u) | — |
| R2 | the residue bias is a truncation phenomenon: the consecutive-survivor diagonal deficit δ(u; m) obeys the same drift law, decreases across decades at u = 2 (the primes), and meets the loop's own value at the deepest live u | — |
| R3 | the termination depth (max least-prime-factor over a gap's interior) against a matched null of random composites is scale-free | — |

Live cells are those where the loop's period exceeds the window; where it does not, window = loop
exactly (a gate). The raw gap histogram is not compared — its delta is Buchstab's density ratio in
disguise — only shapes are. Floors are half-splits, reported per cell.

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
