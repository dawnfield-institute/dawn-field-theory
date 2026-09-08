# Note — the sieve recursion: what closes, what does not, and which Möbius is in the way

**Date:** 2026-09-08 · **Status:** EXPLORING, unregistered, unscored · **Mode:** a failed derivation, written down.
Peter's framing: see it as a recursive closing rather than a sum. The framing is correct and productive. The
derivation still does not close, and the failure localises the obstruction precisely, which is why it is filed.

## The recursion is real, and it is derived rather than guessed

Going from the loop of units mod P_k to mod P_{k+1} deletes the units divisible by p_{k+1}, and **each deletion
merges two adjacent gaps.** By CRT a unit's residue mod p_{k+1} is independent of its residue mod q, so the
deletion is independent of what δ_q measures. The next surviving unit is n steps ahead with probability
λ^{n−1}(1 − λ), λ = 1/p_{k+1}. Hence a **resolvent**:

```
transition matrix:   M' = (1-L) M (I - L M)^-1
gap distribution:    f' = sum_n L^(n-1) (1-L) f^(*n)        (merging = convolution)
```

Both are exact consequences of the merge rule *if* the process is renewal — independent gaps.

Buchstab's ω(u) is the continuum limit of this same recursion, satisfying (uω)′ = ω(u−1), and exp_08's gates
measured our densities tracking e^γω(u) to ≤ 0.4 %. So the recursion is demonstrably present in the object.

## What closes: the first moment, exactly

Iterating the gap-distribution resolvent from the **exact** k = 9 enumerated loop (y = 23) out to y = 997:

| y | mean gap predicted | mean gap true |
|---|---|---|
| 97 | 8.3114 | 8.3114 |
| 199 | 9.6251 | 9.6251 |
| 401 | 10.7705 | 10.7705 |
| 997 | 12.3510 | 12.3510 |

Exact at every depth, and not a coincidence: thinning by 1/p multiplies the mean gap by p/(p−1), which **is**
Mertens. The density recursion closes.

## What does not: everything above the first moment

Same iteration, same run, for the residue deficit:

| y | δ₃ predicted | δ₃ measured | error |
|---|---|---|---|
| 97 | 0.30455 | 0.22314 | +0.081 |
| 199 | 0.30841 | 0.19724 | +0.111 |
| 401 | 0.31101 | 0.18027 | +0.131 |
| 997 | 0.31383 | 0.16132 | +0.153 |

**The renewal model predicts δ stays put; reality decays.** Predicted 0.305 → 0.314 (flat); measured 0.223 →
0.161. The error grows monotonically. This is a structural statement, not noise: **under independent gaps,
residue memory persists — real gaps decorrelate faster than independent ones.** The correlations accelerate
the loss of structure rather than preserving it.

Single-step tests agree and localise it identically at both levels:

| level | improvement over "no change", k = 2…9 |
|---|---|
| transition matrix (resolvent) | 1.0, 1.1, 1.6, 3.0, 3.2, 3.3 × |
| gap distribution (convolution) | 1.5, 2.2, 2.6, 2.8, 3.1, 3.2, 3.3 × |

Both plateau at ≈ 3.3×. The operator captures ~70 % of each step and never more. Failing identically at both
levels means the fault is not the coarseness of the residue state space — it is the **renewal assumption
itself**.

## Which Möbius

The study's founding thesis says: *"It is not a Möbius topology, and the only Möbius in the sieve is Möbius the
function, a balanced signing of the divisor lattice."* That is right, and the obstruction refines it.

There are **two** Möbius functions here, on two posets, and they behave oppositely:

| poset | what it computes | status |
|---|---|---|
| **divisor lattice** — Σ_{d\|P} μ(d)⌊x/d⌋ | the density, the mean gap | **closes exactly** |
| **Boolean lattice of intervening positions** — alternating sum over which interior points are units | the *consecutive* condition | **does not close** |

"No unit strictly between" is an inclusion–exclusion with alternating signs over subsets of the interior. That
is Möbius inversion in Rota's sense, on the subset lattice — not the divisor lattice the README named. So the
thesis was correct that no topology is involved, and incomplete in assuming one poset. **The one that blocks
the derivation is not the one the study named.**

## What this is worth

- The merge operator is **derived** — the first derivation-flavoured object this study has. It is exactly
  right for the first moment and wrong above it, in a stated direction.
- The obstruction is now a single named assumption (renewal / independent gaps) rather than "inclusion–exclusion
  is hard".
- The sign of the failure is information: real gap correlations *speed up* decorrelation. Any correction term
  must therefore reduce residue memory relative to renewal, not add to it.

## What is not claimed

That F is derivable. That the resolvent is a useful approximation for δ (it is not — the error grows with
depth). Anything about the correlation structure beyond its sign and that it is the single obstruction. No
physics; every "φ" is Euler's totient.

## Forward

The renewal assumption is the whole gap. A round that measured the **two-gap joint distribution** on
enumerable loops — and asked whether a second-order (pair-correlated) resolvent closes where the first-order
one plateaus at 3.3× — would either fix it or show the correction is not low-order either. That is cheap: the
enumerated loops are already in hand.
