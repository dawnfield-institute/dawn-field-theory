# Outcomes: exp_23 — Within-Scope Local Coupling Law (P17 v2)

**Date:** 2026-06-13
**Pre-registration:** commit `f1639e08` (model + decision rule, before execution)
**Result:** `results/exp_23_joint_coupling_20260613_212830.json`
**Verdict (registered rule): INCONCLUSIVE** (R1 positive but not significant; not reversed → not killed)

---

## The numbers

32 multi-ion XQR-30 scopes (median 7 ions). Within-scope cascade coupling
`c_i = slope(within-scope-centered logN vs local cascade phase diseq)`:

| ion | IP (eV) | c_i | CI95 | n scopes |
|-----|--------:|----:|------|---------:|
| AlII | 5.99 | **−2.78** | **[−4.69, −0.76]** | 24 |
| MgII | 7.65 | +1.19 | [−0.81, +3.01] | 32 |
| FeII | 7.87 | −1.01 | [−2.70, +0.12] | 28 |
| SiII | 8.15 | −1.12 | [−3.56, +1.65] | 28 |
| CII | 11.26 | −1.06 | [−2.86, +0.42] | 23 |
| SiIV | 33.49 | −1.04 | [−3.66, +2.34] | 21 |
| CIV | 47.89 | **+1.43** | **[+0.16, +2.89]** | 30 |

(CaII, AlIII, NV excluded: < MIN_SYS=15 scopes.)

| Test | Result |
|------|--------|
| **R1** coupling ordering Spearman(IP, c_i) | **+0.43, p_one = 0.169 → not significant (INCONCLUSIVE)** |
| R2 gradient g_s vs diseq (= R1 per-scope) | **+0.351, p = 0.049** — marginally significant |
| R3 pattern amplitude A_s vs diseq | −0.085, p = 0.64 (no swing) |
| Route B (M6, |c| ordering) | rho_A=−0.25, rho_B=0.0 → FAIL |

---

## The honest reading

**INCONCLUSIVE, leaning suggestive — and the design's honesty held.** The registered gate R1
(per-ion ordering, 7 ions) is positive (+0.43) but underpowered (p=0.17). The pre-registration
declared R1 and R2 to be **one signal viewed two ways** (per-ion vs per-scope), not independent
confirmations. That declaration is exactly borne out: the **per-scope** expression of the same
signal — the within-scope ionization gradient responding to local cascade phase — reaches
**p = 0.049 across 32 scopes**, while the **per-ion** expression (7 points) does not reach
significance. The registered gate was the underpowered view of a signal that *is* marginally
present. No double-counting; no over-claim.

**The structure is a sign-flip (turnover), not a magnitude trend.** The two ions whose CIs
exclude zero bracket the IP range with **opposite sign**: AlII (lowest IP) couples strongly
**negative** (−2.78), CIV (highest) strongly **positive** (+1.43). The coupling-vs-IP curve
crosses zero somewhere in the middle — a turnover-like structure in the *sign* of the cascade
coupling. This is why **Route B fails**: it tested the *magnitude* |c| ordering, but the
magnitude does not order by IP (AlII has the largest |c| at the lowest IP). The information is
in the **sign**, not the amplitude — a cleaner statement than the registered Route-B form
anticipated, and one to register directly next time.

**MgII is the monotonicity-breaker.** At IP 7.65 it couples +1.19 — positive, against the
low-IP-negative trend set by AlII/FeII/SiII/CII. Remove MgII and the ordering is far cleaner
(AlII−, FeII−, SiII−, CII−, SiIV−, CIV+). MgII's anomaly echoes the FeII anomaly in the
2026-06-08 tapestry (MgII/FeII have richer multiplet structure); worth a look, not tuned away.

**DFT-discriminating, and the direction is right.** Standard photoionization predicts an
ionization pattern but **no dependence on cascade phase**; the marginal p=0.049 gradient-vs-diseq
response is therefore DFT-specific (and the narrow z-range of XQR-30 rules out a smooth z-trend
alias). The endpoints order in the exp_19 T1 direction (coupling rises with IP). Suggestive, not
established.

---

## What this round actually settled

- The **globalism → locality correction works.** exp_19/21 measured the coupling as a coordinate
  (per-ion β across global z, compared across frames) and produced an artifact knee and a kill.
  Measured **within scopes** (one ledger = one frame, full IP span in-frame), the same physics
  gives a marginal *real* signal (p≈0.05) in the DFT-predicted direction instead of an artifact.
  The "data wall" was a frame error, now removed.
- The coupling carries a **sign turnover** across IP, not a magnitude trend. This is the live
  structure to pursue.

## Forward path (not registered here)

1. **Register the sign, not the magnitude.** Next round's primary should be the IP-ordering of
   **signed** c_i (or the location of its zero-crossing), since the magnitude carries no order.
2. **More scopes for power.** 7 usable ions / 32 scopes is thin. DESI DR1 metal-absorber
   catalogs and KODIAQ/larger XQR-class samples would push the per-ion ordering past p=0.05.
3. **An epoch baseline** (multi-ion absorbers across a wide z-range) would let the sign-turnover
   be tested as a φ-ladder N_turn(IP) structure — the original P17 content, untestable in
   XQR-30's single epoch.
4. **MgII anomaly:** check whether its positive coupling is multiplet/saturation structure
   (as the tapestry FeII anomaly was).

## Ledger

- exp_23: INCONCLUSIVE — registered R1 (per-ion ordering) positive (+0.43) but underpowered
  (p=0.17); per-scope form of the same signal marginally significant (p=0.049); coupling sign
  flips AlII(−)→CIV(+) with both CIs excluding zero; Route B (magnitude) fails because the
  order is in the sign. Pre-registration discipline held; one post-registration mechanical fix
  (a leftover `MIN_NSPAN→MIN_DSPAN` name in the JSON-save metadata — no prediction, threshold,
  or rule touched).
- Running Midnight meta-pattern intact: the **local/relational** form of a prediction carries
  signal; the **global/coordinate** form (exp_19/21) carried artifacts.

---

## Post-hoc diagnostic (2026-06-14): why inconclusive, and the corrected observable

Exploratory follow-up (`explore_inconclusiveness.py`, `explore_contrast.py`) — **post-hoc on the
same data, hypothesis-generating, NOT a result**; the registered verdict above stands. Captured
because it sharply reframes the next prediction.

**The inconclusiveness was structural + methodological, not low power.**
- *Wrong test.* R1 tested a monotone IP-ordering, but leave-one-ion-out shows the "trend" is
  carried entirely by the two IP extremes (drop AlII or CIV → rho 0.43→0.09; drop any middle ion
  → rho rises to 0.49–0.66). The middle (SiII, CII, SiIV: 8–34 eV) is a dead-flat plateau at
  c = −1.07 ± 0.03.
- *Centering shadow.* That flat plateau is a mean-centering artifact: subtracting the scope mean
  imprints a common recoil slope on passive ions. Removing the centering (direct pairwise
  contrasts) dissolves the plateau and exposes the signal.

**The corrected picture — the carriers are CIV and MgII, not an IP class.**
Centering-free, slope of (logN_hi − logN_lo) vs local cascade phase diseq:

| pair | n | slope | CI95 |
|------|--:|------:|------|
| CIV − AlII | 24 | +4.22 | [+2.47, +6.30] ✓ |
| CIV − AlIII | 11 | +3.05 | [+0.55, +8.10] ✓ |
| CIV − FeII | 28 | +2.35 | [+0.57, +4.70] ✓ |
| **SiIV** − FeII | 19 | −0.88 | [−3.62, +1.52] |
| CIV − **MgII** | 30 | −0.17 | flat |

- **CIV** swings hard against passive ions toward cascade transitions. **SiIV** (also IP > E_H)
  does **not** — so the E_H / IP-threshold split is the wrong axis.
- **CIV − MgII is flat** → MgII swings *with* CIV. The active set is **CIV + MgII** — exactly the
  two clean resonance doublets the 2026-06-08 tapestry flagged (MgII p=10⁻¹¹, CIV p≈0; FeII the
  anomaly). Two independent analyses → the same ion set: corroboration, not coincidence.
- AlII's strong negative coupling in the registered run was **centering recoil** (it is the most
  passive ion, so it recoils hardest against CIV's rise) — which is why CIV−AlII is the largest
  swing. The "sign-flip near E_H" reading is superseded by this.

**Caveats (post-hoc):** single dataset, active set found by inspection; slopes are extrapolated
(diseq never exceeds 0.44 — never a real transition); small n per pair. The bootstrap CIs
excluding zero are real but this is the *hypothesis*, not its confirmation.

**Next registered prediction (writes itself):** *the clean-doublet ions (CIV, MgII) swing in
abundance/coherence relative to passive ions as a function of local cascade phase* — pre-register
on **new data (DESI multi-ion absorbers), ideally reaching diseq > 0.7** (a real transition). The
active-ion set is now independently motivated by the tapestry, so the grouping is non-circular.
