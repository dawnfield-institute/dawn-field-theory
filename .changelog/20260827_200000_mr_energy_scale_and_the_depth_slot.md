# Milestone R energy-scale propagation — and depth was in the wrong slot

Started as the ROADMAP's "highest-value unfinished work". Recovered two test failures, narrowed
exp_24's root-cause claim, and found a structural problem underneath the fix itself.

## It was blocked, not deferred

`milestone-r/core/radiation_physics.py` resolved `MR_ROOT.parent / "milestone14"`. After the
August 2026 layer reorganization this sidecar sits at `experiments/sidecars/`, so that points at
`sidecars/milestone14`, which does not exist — and **every one of the 27 scripts imports through
that module.** None of them ran. `midnight/core/phase_rate.py` has the identical break.

Second blocker: **`np.trapz` was removed in NumPy 2.0**, and `requirements.txt` floors numpy at
1.24, so numpy 2.x is in-spec. Fixed in exp_03 via `scipy.integrate.trapezoid` — numerically
identical, exp_03 T2's reduced chi² is `0.3748079733738714` before and after. **Six other files
still carry it**: milestone11 ×3, milestone2, milestone4, `minimum_actualization_resolution` ×5.

## Propagated: 60/112 → 62/112

| exp | test | before | after |
|---|---|---|---|
| 03 | T3 beta endpoint | FAIL, best depth **19** — the top of its own search range, 0/3 within ×10 | **PASS**, depth 7, 2/3 |
| 05 | T1 Rydberg | FAIL, 24 orders off | **PASS**, 11.4 ppm |

Added `severance_energy_coupled()` beside the untouched `severance_energy`, following the pattern
exp_24 set with `coupling_boundary_count`.

**exp_24 fixes two of its six named test failures, not eight.** exp_03 T4, exp_04 T1 and exp_05
T2/T4 contain no scale term at all — exp_04 T1 fails on a negative *sign*, which no rescaling can
touch, and exp_05 T2's own note already says "Expected to fail."

**And the net score may go down when this is finished.** Several *passes* are tautological at the
Planck scale — exp_06 T1/T2/T4, exp_02 T1, exp_08 T1, exp_09 T4, all "everything rounds to zero."
A correct scale makes them real tests that can fail. That is the right outcome; the ROADMAP's
framing omits it.

## The finding: depth is read out of the wrong slot

1. **The (depth, mediator) pair is exactly degenerate.** `E ~ φ^(−2d)·m`, so scaling the mediator
   by r is identical to shifting depth by `ln(r)/(2 ln φ)`. Measured: m_p/m_e = 1836 predicts a
   shift of 7.809, and the beta-endpoint best fit moves electron d=0 → proton d=7 exactly.
   **A fitted depth measures the mediator choice.** exp_03's "depth 7" is not a physical result.
2. **`dft_energy_scale` is valid only for the nuclear case.** It uses `fdc(d) = φ^(−d)/√5`, which
   is **72× off** at EM depth; exp_24's T1 bypasses it for `ALPHA_EM_DFT`. The only test that
   uses it sits at 1.75× inside a factor-1000 window.
3. **Every DFT constant encodes scale in *which* Fibonacci indices appear, and carries φ exactly
   once.** α_EM's φ-power is **−1**, not −13. `fibonacci_depth_coupling` encodes scale as a
   *power of φ* and drops the indices — structurally incompatible representations.

Also killed, by one line of arithmetic: **M9's cascade depth cannot be the coupling depth.** The
clock spans N ∈ [0, 6.81] over all cosmic history; DEPTH_EM = 13 would need 271 Gyr of lookback.

## Class signature over 11 constants

φ⁻¹ appears **iff** the quantity is a gauge coupling. π appears **iff** it is a balance or
correction term. Everything else — mixing angles, mass ratios, Casimir, She-Leveque — is pure
Fibonacci ratio, no φ, no π. Fibonacci indices count, φ scales, π closes.

## Two leads, labelled and unscored

- **α_s is the only coupling with no π correction, and the worst** at 1.712% (2.24σ). One of
  α_EM's form is needed; α_s at ±0.76% cannot select which.
- **The correction index gaps are Lucas, not Fibonacci.** exp_37 asks "why are index gaps
  themselves Fibonacci?" — 4 and 7 are not Fibonacci. α_EM gap 3, Ω_Λ gap 4, G gap 7, and
  3+4=7: L₂, L₃, L₄. Fibonacci and Lucas are the two independent solutions of the same
  recursion. **Tested the same day and it does not survive.** Two further corrections exist in
  code and are missing from exp_37's list — Λ at (3,5), gap **−2** (M8 `exp_08` line 84) and the
  dark coupling at (8,6), gap +2 (M8 `bsm.py` line 208). **Negative gaps occur**, so the
  criterion must be |gap| — which readmits `F₃/(4π F₄²)` (gap −1, |−1| = L₁) as the *best* fit
  at 0.026%. Four candidates survive, not one. The unique selection was an artifact of excluding
  negative gaps and picking the {3,4,7} subset post hoc. The constraint is also weak: 1,2,3,4
  are all Lucas, so P(all five gaps Lucas by chance) is 0.03–0.10.

  **What survives**: exp_37's premise is still wrong — the gaps are *not* Fibonacci, since 4 and
  7 are not Fibonacci numbers. And |gap| ∈ {2,3,4,7} with **5 and 6 never occurring** across five
  corrections, which is the actual observation a larger inventory would test. Also surfaced a
  documentation discrepancy: exp_37 lists Ω_Λ as (9,5,n=4,+) while M8's code applies (3,5,n=4,−)
  to the cosmological constant.

Full detail in `experiments/sidecars/milestone-r/journals/2026-08-27_energy_scale_propagation_and_the_depth_slot.md`.
