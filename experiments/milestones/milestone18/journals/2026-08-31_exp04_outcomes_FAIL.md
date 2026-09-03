# 2026-08-31 (night): exp_04 outcomes — P-B1 FAILS

**Registration**: 74bcd0df (sealed before any golden computation; calibration was
controls-only and blind). **Run**: grid, functional, and criteria exactly as sealed.
One instrument note: the panel run needed a singular-target guard (a per-s skip for
Dirichlet submatrices with vanishing determinant); the guard does not touch the golden
numbers, which computed cleanly on the unguarded path.

## Result

| diagram | s* | \|s*−1\| | sealed criterion | outcome |
|---|---|---|---|---|
| D6-tree | 0.79 | 0.21 | ≤ 0.01 | **FAIL** |
| E8-tree | 1.27 | 0.27 | ≤ 0.01 | **FAIL** |

Null (40 random 8-trees): median |s*−1| = 0.25. Symmetric panel range: 0.13–0.32.
**The golden diagrams behave like generic trees under this functional.** No confound
branch reached; the failure is clean.

## What this kills (kill-scope declared per the registration)

Plain-diffusion mean-first-passage time does **not** select the self-dual boundary
condition on the folding trees. The strong hypothesis — "balance-seeking dynamics
manufactures the golden channel" — is disfavored **in this functional**.

## What this does not kill

- Every exact result stands untouched: Block A (12/12), the Complement Identity, the
  Ledger Theorem landscape, the Bipartite Duality and its confirmed s=3 prediction, the
  knife-edge map. A dynamical null does not reach backward into theorems.
- The self-duality *question* survives with a sharpened bearing (below).
- Per the milestone kill sentence, Block B is wounded, not dead: exp_05 (the
  ledger-antisymmetry statistic Λ) remains registered and unrun, and Block C is unrun.

## The bearing (a null is a bearing, not a verdict)

1. **The E-series march has a limit, and it is not self-dual.** E6 → 1.47, E7 → 1.34,
   E8 → 1.27: monotone toward s = 1 but asymptoting in the 1.2s. Whatever that sequence
   converges to, it is a real, smooth, topology-driven quantity — and it is not the
   golden point. Worth computing on E-type extensions (E9-tree, E10-tree) exactly.
2. **Plain diffusion breaks the duality it was asked to find.** The λ ↔ 4−λ symmetry is
   the structure that pins s = 1; diffusion weights small eigenvalues and is maximally
   asymmetric under that flip. Asking it to find the self-dual point may have been a
   category error — the right dynamics for exp_05's design is **duality-symmetric**
   (functionals even under M ↦ 4I − M), for which the self-dual point is a symmetry
   point rather than a needle in a generic landscape. This is a design direction, not a
   registered claim; it goes through the full seal cycle before any run.
3. The failure itself instantiates the milestone's own thesis warning: I asked a
   *magnitude* question (where is the optimum?) of a *structural* symmetry. The
   structure-native question is what the dynamics does AT the self-dual point
   (degeneracy, susceptibility, mode-crossing), not whether it travels there.

Score: exp_04 registers 0/1 on P-B1. Recorded without threshold revision.
