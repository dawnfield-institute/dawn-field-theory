# 2026-09-07 (evening) — exploring: the delta is the position of the read

**Mode: EXPLORING, declared.** Nothing here is scored; it is disclosed as pre-seal for round 2. Script
`scripts/explore_r2_position_along_the_loop.py`, results `results/explore_r2_position_<ts>.json`.

## Peter's reading of exp_01's residual

The gap predictions miss because the gaps are tranches of information, and what the local read cannot see comes from
the global side — the identity of the integer once fully actualized, all its residues at once, exerting a pressure the
local model does not carry. So look completely locally and completely globally, and forget the delta as a thing.

Made precise: the uniform loop (a random residue per prime) is the loop read at a *random* position. The primes are
the loop read at its **origin** — the arc below y² where every residue is the integer's own name and all moduli agree
at once. The residual of exp_01 is not a third object; it is the inhomogeneity of one object along its own length.

## What sliding the read shows

Fix the depth y; slide a window from the origin outward (positions N = 10⁴ … 10⁵⁰ — only N mod p is needed); the
consecutive-survivor bias δ₃ against the position's own depth u_N = log N / log y:

| y | primes' arc (u_N ≈ 1.9) | u_N ≈ 2.2 | u_N ≈ 2.5–2.9 | u_N ≥ 3 | uniform loop |
|---|---|---|---|---|---|
| 448 | 0.1746 | 0.1750 | 0.1787, 0.1775 | 0.172–0.182 (noise ±0.005) | 0.1775 |
| 1,415 | 0.1453 | 0.1512 | 0.1529, 0.1535 | 0.152–0.157 | 0.1538 |
| 4,473 | 0.1291 | 0.1299 | 0.1374, 0.1400 | 0.135–0.138 | 0.1362 |

The bias climbs from the primes' value to the loop's within about half a unit of depth and then fluctuates around the
loop's value out to 10⁵⁰ — the same loop, read further from its origin. The survivor counts climb the same way
(y = 1,415: 140,531 → 154,600, Buchstab's ratio 0.909 → 1). exp_01's residual at u = 2 (≈ 0.0095) and u = 2.25
(≈ 0.003) is about nine percent of Buchstab's density deficit at both depths: the same origin effect, on the
transition structure instead of the count.

## What this says for round 2

- **Local and global are reads of one object at two positions**; the delta is the position. Register the bias as a
  function of the read's depth u_N alone (collapse across y), equal to the primes' value at u_N = 2 and the uniform
  loop's for u_N ≳ 3, with a CONVERGED verdict class for the plateau and a tolerance on "meets".
- The ratio ε / (1 − e^{γ}ω) ≈ 0.09 at two depths is a candidate relation between the transition residual and
  Buchstab's density deficit — registrable, scale-free, with the exact loop rationals as anchors.
- Noise at these window lengths is ±0.003–0.005 per window; round 2 needs longer windows or more of them per position.

Not claimed. Recorded so the registration can cite it as seen.
