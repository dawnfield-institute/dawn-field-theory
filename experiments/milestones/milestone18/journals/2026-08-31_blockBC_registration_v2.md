# REGISTERED: Block B and Block C amended registrations (v2)

**Status: SEALED by this commit (Peter's go, 2026-08-31). Supersedes the Block B portion
of cf886c00 as a forward correction; the original stands as lineage. Calibration was
completed on controls only; no golden diagram has been computed under Block B.**
Supersedes the Block B portion of cf886c00 on commit (forward correction; original stands
as lineage). Informed by `2026-08-31_night_exploration_sigma_ledger.md`.

## Block C — The σ-Ledger (exp_06–exp_07)

Frame: operators constructed from diagram data; expectations are exact algebra; same scope.

exp_06 (4 tests, mostly (S)-structural — labeled):
- T1 (S): σ(P) = I−P, P·σP = 0, (P−σP)² = I for ALL THREE foldings with exact symbolic
  projectors (D₆'s built by isotypic decomposition, not numeric scan).
- T2 (S): Ledger Theorem landscape: q·σ(q) on {A₄,D₆,E₈}; zero golden content on
  {A₅,D₄,D₅,E₆,E₇}; A-family sweep n=2..12.
- T3: R expressed in the simple-root basis is Aut(diagram)-equivariant. (Could fail.)
- T4: The ±1 eigenspaces of R, pulled to vertex space, REFINE the automorphism orbit
  quotient (M13.5 comparison). Either outcome informative; failure = the σ-ledger and the
  orbit metric are independent structures.

exp_07 (4 tests): the knife-edge under Bipartite Duality (see night journal R6):
- T1: spec M(2−s) = 4 − spec M(s) exact on every tree tested (parity conjugation).
- T2: the golden s-set is closed under s ↦ 2−s, with s=1 (Cartan) the unique self-dual point.
- T3: branched folding trees (D₆, E₈) are golden ONLY at s=1; paths carry golden dual
  pairs — full enumeration for n = 3..12 fixed BEFORE running.
- T4: the E8 s=¾ rational [4,4] (and its dual s=5/4) explained or flagged.

## Block B — Dynamical self-duality (exp_04–exp_05, replaces the φ-ratio/conjugacy forms)

The thesis's empirical content, sharpened by the night results: **does dynamics seek the
self-dual boundary condition?** Make s (the scope-boundary coupling) a degree of freedom;
golden structure exists only at the self-dual point s=1 (branched folding trees, night
journal R6). If SEC-type dynamics drives s → 1, golden structure is MANUFACTURED by
balance-seeking. If dynamics is indifferent to s, the σ-ledger is decoration and the kill
sentence takes the hit.

**TAUTOLOGY GUARD (declared at design time):** s=1 ⟺ uniform diagonal ⟺ equal vertex
self-coupling. Any objective that directly rewards uniformity (variance of the diagonal,
per-vertex self-energy equalization) reaches s=1 BY CONSTRUCTION and is inadmissible. The
registered functionals must be dynamical observables: (a) spectral gap / relaxation rate of
M(s), (b) stationary entropy-production rate under noise, (c) mean first-passage time,
(d) the corpus SEC-collapse efficiency. For each: the s-profile is computed on NON-GOLDEN
CONTROLS FIRST. If controls also optimize at s=1, that functional is generic and rejected
(a control's job is to be able to kill the design). Only folding-specific s=1 optimality
counts. Kill-scope note (Peter's rule, 2026-08-31): a functional failing this guard kills
THAT FUNCTIONAL's use, not the question.

Original framing (ledger-antisymmetric FPT statistic Λ) retained below as exp_05.

**CALIBRATION COMPLETE (controls only — golden diagrams untouched; results in
`results/exp_04_calibration_20260831.json`, `exp_04b_null_20260831.json`):**
- Functional selection: stationary-entropy proxy REJECTED (edge-seeking on all controls,
  generic); spectral gap WEAK (edge-dominated); **frame-free MFPT selected** — interior
  optima, topology-sensitive. Frame-free = averaged over all absorbing targets (an index
  is not a place; the vertex-0 version shifted D5's optimum from 1.8 to 0.89 — the frame
  fix was material and is documented).
- Null distribution (40 distinct random 8-trees, grid 0.5–1.5 step 0.01):
  |s*−1| median 0.25, 5th percentile 0.08; **P(|s*−1| ≤ 0.01) ≈ 0.025**.
- **Declared confound (found by the null itself): star6 lands at s* = 1.01.** High
  automorphism symmetry may pin optima near self-dual independent of goldenness. Sealed
  test therefore requires: (i) BOTH D6-tree and E8-tree at |s*−1| ≤ 0.01, (ii) a
  symmetric-control panel (stars, spiders, caterpillars with |Aut| ≥ |Aut(E8-tree)|)
  whose scatter the golden diagrams must beat, (iii) target-averaging convention declared
  (vertex-uniform).

**SEALED-PREDICTION TEXT (to be committed verbatim before any golden run):**
  P-B1: the frame-free MFPT optima of the D6-tree and E8-tree both satisfy |s*−1| ≤ 0.01
  (fine grid), jointly below the null (≈6×10⁻⁴ if independent), AND below the symmetric-
  control panel's best. Failure of either diagram fails P-B1. If symmetric controls also
  cluster at |s*−1| ≤ 0.01, the result is INCONCLUSIVE-BY-CONFOUND and says so — kill
  scope: that outcome kills this FUNCTIONAL's evidential use, not the self-duality
  question, not the night's exact results.
- Harness: M-R exp_15/16 stress-barrier FPT machinery, refactor-gated (exp_00-style bit-
  identical baseline reproduction before any registered run, per ade_cascade discipline).
- Channel: s=1 (Cartan/Dirichlet) weighting.
- Registered observable: FPT statistics decomposed along P vs σP mode subspaces; the
  ledger-antisymmetry statistic Λ = (stat_P − stat_σP) normalized.
- Predictions (thresholds to be sealed after harness calibration on NON-golden controls
  only — calibration never touches A₄/D₆/E₈): Λ significantly nonzero on golden diagrams,
  compatible with zero on matched controls (size, degree sequence); effect ordering stable
  across noise levels; z-detrending-style shuffle control (midnight discipline).
- Frame declaration: sampled = FPT ensembles per diagram; expectation = matched-control
  ensembles at identical size/degree; same scope.
- Inconclusive branch: if calibration shows Λ's null width exceeds any plausible effect,
  Block B returns INCONCLUSIVE-BY-POWER and says so (a null is a bearing).

Kill relevance: per the milestone kill sentence, Block B failing (with a green harness)
AND Block C's T3/T4 both failing kills the milestone's claim to physical content.

## Cross-repo proposal (NOT this milestone's lane — for Peter to carry if he wants it)

reality-engine v4 POC ablation ladder (active this week, Peter's branch): one additional
arm — scope-boundary coupling weighting swept along the Robin family toward the self-dual
point, measuring the existing web metrics (ξ floor, void fraction, density CV). Rationale:
M16's clumps-without-web may be structural if the engine's operators sit off the golden-
bearing boundary condition. This milestone only records the proposal; the engine branch
belongs to Peter.

---

**Layer (forward note, 2026-09-02, per the re-separation):** Block B measures a physical reach (`theory/`); Block C tests a conjecture — the σ-ledger and its knife-edge — whose proved parts are indexed in `formal/theorems/` and whose open parts are in `formal/conjectures/m18_open.md`.
