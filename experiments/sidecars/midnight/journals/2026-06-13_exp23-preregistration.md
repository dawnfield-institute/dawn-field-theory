# Pre-registration: exp_23 — The Coupling Law, Within-Scope (Local) Form (P17 v2)

**Date:** 2026-06-13
**Status:** PRE-REGISTRATION (to be committed before any registered quantity is computed)
**Script:** `scripts/exp_23_joint_coupling.py` (same registration commit)
**Supersedes the model of:** exp_21 (KILLED 2026-06-11) and a first exp_23 draft (a global
cross-survey bridge, discarded — see §2)
**Derivation:** `journals/2026-06-11_p17-coupling-law-derivation.md` (P17), reframed locally.

---

## 1. Why v2 — the exp_21 kill

exp_21 fit `|β|(IP) = A/(1 + r·P(IP))` to **pooled** per-ion betas, treating magnitude and
sign as separable. Near a species' turnover β passes through zero because the **ladder flux
reverses**, not because monitoring is strong, so the pooled fit parked the knee at ~8 eV (a
flux artifact). The model was refuted as a structure.

## 2. The deeper correction — global → local (this is the point of v2)

The first instinct for v2 (and exp_19/21 before it) was to measure the coupling β per ion as a
slope of EW against the cascade clock across a **global redshift axis**, then compare betas
**across surveys**. This is the **globalist failure path**, and it is the reason this whole line
kept dying:

- **Raw β is a coordinate, not an invariant.** exp_19's "portability disaster" — XQR-30 betas
  1–2 orders larger than SDSS for the same ion; MgII +0.22 (SDSS) vs −0.43 (XQR-30) — is not
  weak/messy signal. It is the **same quantity read in two frames with no transformation law
  connecting their magnitudes.** Comparing or pooling them is the error.
- **Relativity / PAC / SEC all forbid the global frame.** There is no universal frame
  (relativity); conservation is **per-ledger** (PAC); collapse is relative to the **local
  gradient** (SEC, M13 definitional parallax). A prediction stated in a global frame is a
  coordinate claim, and the Midnight ledger has shown coordinate claims die while relational
  ones survive.
- **The signals that won were already local.** The tapestry (p=10⁻¹²), the doublet coherence
  (p=10⁻¹¹), exp_08 C (multi-absorber on the *same sightline*, "immune to z-trends") all win
  because each measurement is made **inside a single local ledger**. They did not win because
  they were widths; they won because they were frame-clean.

**The relativistically-correct experiment measures the coupling law within a single scope.** An
XQR-30 `SystemID` is one absorber = one local ledger = one frame, and within it many ions are
measured simultaneously at one redshift. The frame-clean observable is the multi-ion abundance
*pattern* inside the scope; the coupling is how that pattern responds to the **local** cascade
phase. This dissolves the imagined "data wall": a single system spans the full IP range
(5.99–47.89 eV) **at once**, so the IP leverage exp_19 lacked (it measured one ion at a time
across global z) is present inside every frame.

## 3. The data — XQR-30 as multi-ion scopes

`xqr30_merged_catalog.csv`, grouped by `SystemID`. Each system (scope) carries per-ion column
density `logN` on its primary-transition rows. **Census (non-registered, `--selftest`):**

```
32 multi-ion scopes (>=3 ions; median 7, max 9). Usable ions (>=15 scopes, >=4 cascade phases):
  ion    IP(eV) stage #scopes  diseq range
  AlII    5.99   2     25      0.03-0.33
  MgII    7.65   2     32      0.03-0.44
  FeII    7.87   2     28      0.03-0.44
  SiII    8.15   2     28      0.03-0.36
  CII    11.26   2     24      0.03-0.36
  SiIV   33.49   4     21      0.03-0.33
  CIV    47.89   4     30      0.03-0.44
```
Seven ions spanning IP 5.99→47.89 eV, each in 21–32 frames. **Constraint:** XQR-30 is a
near-single-epoch snapshot (all scopes N≈6.3–6.7), so the leverage is the **local cascade
phase** `diseq(N) = 1 − 2|N − round(N)|` (0→~0.44 here), **not** epoch N. The registered
observable is built on diseq accordingly (decision: "pattern response to cascade phase").

## 4. The registered model (within-scope coupling to local cascade phase)

For scope `s` with ions `{i}` present:
```
x_i^s = logN_i^s − mean_j(logN_j^s)              # within-scope-centered abundance (frame-invariant)
c_i   = slope of x_i^s vs diseq_s across scopes  # cascade coupling of ion i
g_s   = slope of x_i^s vs ln(IP_i) within scope  # within-scope ionization gradient
A_s   = std_i(x_i^s)                             # within-scope pattern amplitude
```
`x` is centered within each scope, so total column / metallicity / dust (the common
sightline factors) cancel — every `x_i^s` lives in its own frame. `diseq` is the SEC-local
gradient. The coupling law (P17), local form: an ion's within-scope relative abundance responds
to local cascade phase, and the response `c_i` is monotone in ionization energy IP. **This is
DFT-discriminating: standard photoionization predicts an ionization pattern but no dependence on
cascade phase.**

## 5. Registered predictions & decision rule (invariant-registration rule)

**PRIMARY (gates the verdict):**
- **R1 — cascade-coupling ordering.** `Spearman(IP, c_i) > 0`, one-sided p < 0.05, over the
  usable ions (≥15 scopes, ≥4 cascade phases). The within-scope cascade coupling orders by
  ionization energy. Frame-invariant; mirrors exp_19 T1 (`Spearman(IP, β) > 0`) but measured
  **locally**, with full in-frame IP leverage. **KILLED if `Spearman(IP, c_i) < 0` at p < 0.05.**

**SUPPORTING (reported, do not gate — and NOT independent of R1, see below):**
- **R2 — aggregate gradient response.** `Spearman(diseq, g_s)` across scopes — the within-scope
  ionization gradient tracks local cascade phase (the DFT-specific dependence). Direction
  reported, consistent with R1.
- **R3 — pattern amplitude.** `Spearman(diseq, A_s)` — how the within-scope multi-ion pattern
  amplitude moves with cascade phase (SEC divergence direction). Reported.

**R1 and R2 are ONE signal, not two confirmations (declared).** With within-scope mean-centering,
if the pattern is roughly linear in ln(IP) then `c_i ∝ (ln IP_i − ⟨ln IP⟩) × [slope of g_s vs
diseq]` — so the per-ion ordering R1 is *induced* by the gradient-vs-phase response R2 once the
pattern is near-linear. They are the same underlying claim — **the within-scope ionization
structure responds to local cascade phase** — viewed per-ion (R1) and per-scope (R2). They are
counted as ONE piece of evidence, not two. R1 carries the registered direction and the kill;
R2 is its per-scope expression.

**ROUTE B — M6 scoped mediation (independent ordering).**
`|c_i|` ordered by ionization **stage** (boundary count from the cosmic flow), `D = S_max −
stage`, `a = e^{−Ξ} = e^{−(γ+ln φ)}`. Stage is atomic structure, **not** φ-energy-ladder rungs
— the registered independence condition. PASS = `Spearman(IP, |c|) > 0` AND
`Spearman(stage, |c|) > 0`, both one-sided p < 0.05.

**VERDICT:**
- **SUPPORTED** = R1 holds.
- **KILLED** = R1 reversed (`Spearman(IP, c_i) < 0` at p < 0.05) — coupling anti-orders with IP.
- **INCONCLUSIVE** = too few multi-ion scopes / cascade-phase span to resolve R1.

## 6. What exp_23 must do (and must not)

- **Must:** group XQR-30 by SystemID into scopes; within-scope-center logN; compute `c_i` vs
  local diseq with bootstrap CIs; R1 ordering; R2/R3 aggregate; Route B; the registered verdict.
- **Must not:** compare or pool raw abundances/β **across scopes or surveys** (only within-scope-
  centered, frame-clean quantities aggregate); change any threshold, the ion set, MIN_IONS/
  MIN_SYS/MIN_DSPAN, or the decision rule after this commit; promote R2/R3 to gates.
- Pre-registration discipline: `--selftest` (loaders + multi-ion census) is the only thing run
  before the registration commit. The full run executes **after** Peter commits this journal +
  the script. A post-registration mechanical fix (e.g. a loader/column adaptation, as in exp_18)
  is allowed and logged, provided **no registered prediction, threshold, or rule changes.**

## 7. Honest accounting

- The "data wall" that nearly made me abandon the coupling law for kinematics was a globalist
  artifact: one-ion-at-a-time across global z, comparing coordinates across frames. Measured
  within scopes, the coupling law is well-powered (7 ions, 32 frames, full IP span).
- The single-epoch nature of XQR-30 means we test the coupling-to-**local-phase** (diseq), not
  the coupling-to-epoch (N). This is the appropriate local quantity and is DFT-discriminating;
  it does not test the φ-ladder turnover *spacing* (that needs an epoch baseline XQR-30 lacks —
  declared, deferred to a multi-epoch multi-ion sample).
- R1 is a genuine relational invariant with a real kill condition (sign reversal). Route B adds
  an independent (ionization-stage) ordering as a cross-check, enforced non-circular.
- This experiment is the local/relational descendant of the tapestry (p=10⁻¹²), generalized
  from doublet/pair contrasts to the full multi-ion ledger and resolved by ionization energy.
