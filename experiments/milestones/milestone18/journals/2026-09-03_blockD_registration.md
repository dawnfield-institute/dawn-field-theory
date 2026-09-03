# Block D registration — physical reach of the fold certificate (SEALED by the commit carrying this file)

**Layer: physics (→ `theory/`, THEORY_MAP and ROADMAP).** Milestone 18 Block D, gated on Blocks B and C
(README, Kill sentence / Honest risks); the gate criterion was met by exp_18 T1 (47/47 against a null
computed before the run). Opening the gate is the decision recorded by this commit. Drafted 2026-09-02
(`2026-09-02_blockD_registration_DRAFT.md`, with its dated forward notes and the pre-seal section);
sealed 2026-09-03. Run after; scored to this text. Kills have the scopes in §7.

## §0 Gate accounting and postdiction disclosure
- exp_18 T1 PASS 47/47 meets the Honest-risks criterion ("a result that could have failed and didn't").
- **Seen before this seal (disclosed):** (i) the grades and sector classes of every object (O1 d = 3–8 by
  the sector route, O2, O3, O4) — grades are not certificates; (ii) the null carrier rates (T3); (iii)
  **the H₂-sector certificate values on the complete PAC trees at d = 3, 4, 5** — computed as the
  instrument gate for the sector route (draft §2.7 required that gate on n ≤ 63, which only these
  objects provide; the draft's "never before sealing" clause and its gate clause were inconsistent, and
  the gate was run): tr(R·D) = −3√5/5, −2√5/5, −4√5/5 and leak_oo = 1, 4/5, 8/5 — **not** the certificate.
  These three are therefore known answers and are scored nowhere below; T2b and T1 are stated only on
  what has not been computed.
- Not seen: any certificate value on O1 at d = 6, 7, 8, on O2, or on O3.

## §1 Objects (closed at this seal; counting basis §6)
| object | evaluable at this seal | route |
|---|---|---|
| **O1** complete PAC binary trees `pac_tree(d)` (adjacency only), **d = 6, 7, 8** (n = 127, 255, 511) | 3 | sector route (`core/pac_sectors.py`), gated exact vs Bézout at d = 3, 4, 5 |
| **O2** PAC growth trees replayed from `evolve_pac_tree` for every (max_depth, max_children, seed) of exp_07 / exp_08 / the exp_08 grid (`core/growth_harness.py`, bit-exact on 31/31 seeds): the **166** distinct trees with n ≤ 100 and golden content (core 9, partial 157) | 166 | Bézout route (`sector_projector_dm`, gated vs `sector_projector`) |
| **O3** M15 unicyclic controls (`random_unicyclic`, RandomState(152), m ∈ {7, 9, 11} × 20): the **4** with golden content | 4 | Bézout route |
| **O4** M13 density sweep (18 + 2 graphs) | **0 — no golden content; unreachable, declared** | — |
| **\|E\| live = 173** (3 + 166 + 4); O1 d = 3, 4, 5 are gate objects (§0), not scored. | | |
Enumerated unreachable, never scored: all ADE (input); regular graphs (blind by theorem); weighted / real
operators; mode-basis operators; ADE products; engine and percolation graphs; O2 trees with n > 100 (1,071
of 1,600, exact evaluation out of reach); O2 trees with no golden content (363).

## §2 Instruments and gates (all PASS before this seal; results timestamped in `results/`)
`core/certificate.py` (`explore_d0`): projector as a polynomial in C from the Bézout identity against
the cofactor, exact over ℚ(√5) — identical to `bezout_proj` on 13/13 strict exp_13 folds and to the sealed
exp_14 nullspace recipe on 80/80 core folds; KA1 A₄/E₈; KA2 D₆ (trace law holds; off-core laws fail as
exp_14's clean-regime rule predicts — the draft's KA2 was corrected); KA3 A₅/D₅/E₆/E₇ → none; KA4 the four
asymmetric n = 20 trees reproduce {34√5/125, 66√5/125, ±458√5/1235, ±6√5/65, −206√5/247}, leak ≠ 2/5;
KA5 det −80 carries H₃, det −620 H₂; KA6 C₅ blind; KA7 PAC d = 3, 4, 5 one pair, H₂ at scale² 2.
`core/growth_harness.py` (`explore_d0b`): replay bit-exact on integer metrics, 31/31 seeds; recorded
exp_07/exp_08 values reproduced to 10⁻¹². `core/pac_sectors.py` (`explore_d0c`): sector certificate =
Bézout certificate at d = 3, 4, 5 in all four quantities. `sector_projector_dm` (`explore_d0d`): PASS 38/38 halves (A₄, E₈, 13 exp_13 folds, the four asymmetric
n = 20 trees), 2.4× faster. Frame: Cartan channel C = 2I − A; D the graph's own degree matrix; B = 2I − D; all
quantities invariant under relabelling; **carrier** = some Galois half gives tr(R·D) = 2/√5 **and**
leak_oo = 2/5 (some-partner semantics, lesson 4); guards: regular ⇒ blind, odd n ⇒ matching form
impossible (parity theorem), degenerate ⇒ declared.

## §3 Pre-seal numbers (lessons 7–8; computed 2026-09-03, files in the draft's pre-seal section)
Grades: O1 partial at every depth, one H₂-type pair (t² − 4t + 1 ± √5, scale² 2) at d ≤ 7, **plus the first
H₃-type pair (t² − 4t − 1 ± √5, scale² 2) at d = 8** (multiplicity 1, from the k = 9 radial sector);
O2 1,600 distinct trees → 529 with n ≤ 100 → 166 with golden content; O3 4/60; O4 0/20.
**Null (T3):** certificate on every partial tree on 15 vertices (exhaustive): **0 carriers / 459**; random
degree-≤3 trees (seed 20260903): 0/5 at n = 31, 0/9 at n = 63. Null carrier rate ≤ 0.6 % (95 %).
Grade base rates: 459/7,741 at 15; 5/40 at 31; 9/20 at 63.

## §4 Tests (M = 4; T2a is a gate, not scored; T4 scored)
**T1 — reach.** At least one live object (O1 d = 6–8, O2, O3; 173 objects) is a carrier on some half.
*Null:* ≤ 0.6 % per object ⇒ P(≥ 1 of 173 by chance) ≲ 0.65 under the bound, and 0 observed in 473 null
trials. *Threshold:* ≥ 1 carrier passes; 0 carriers is a clean negative (fail, informative). *Expected
direction, stated:* 0 carriers.

**T2a — gate (known answer):** the H₃-type sector first appears in O1 at d = 8 and at no d ≤ 7 (sector
route, pre-seal). Recorded, not scored.

**T2b — class without representative.** The H₂-type sector of O1 at d = 6, 7, 8 and the H₃-type sector
at d = 8 carry no certificate: tr(R·D) ≠ 2/√5 or leak_oo ≠ 2/5 on every half. *Known at d ≤ 5 (§0):*
H₂ carries none. *Threshold:* 0 carriers among the 4 live (depth, class) cells passes; any carrier
fails (and is the surprise). *Expected direction:* pass — M15's class-pass / representative-fail shape.

**T3 — null calibration.** Recorded (§3), not scored.

**T4 — P1 forward-correction.** P1 as registered (README) conflated root-shell radii with eigenvalue
multisets and named golden-free data. Sealed here: (i) the eigenvalue-level content of "E₈ splits
as H₄ + φ·H₄" is the σ-pairing of the Ledger theorem (Block C), (ii) the φ-scaling content is the
root-shell ratio verified by exp_02 (Block A), (iii) the live successor prediction is T1/T2b above.
*Scored:* pass iff (i) the instrument grades E₈ strict with p = q·σ(q) and the H₄ half a carrier (KA1, rerun
in this run) and (ii) Block A's sealed `exp_02_projection_carries_phi.py`, rerun as a subprocess in this run,
passes its own gate (exit 0, PASS in its output) — a verification, scored because the correction is the claim.

## §5 What would count as vacuous
Zero evaluable objects in a class; a T2b cell whose sector projector is degenerate (declared).

## §6 Counting basis
Per object (tree); halves reported alongside; O2 per distinct tree (isomorphism classes reported).

## §7 Kill scope
**Block D can neither kill nor rescue the milestone** (README: clause B is decided in the Cartan channel
by theorem, clause C by proof). T1 = 0 and T2b pass retire "the fold is physically reached by the
corpus's generative objects at these sizes" and "the H₃ class has a fold representative in the PAC
family" — the physics reach at measured sizes, not the mathematics. A T1 carrier or a T2b failure
would be a discovery (a hidden double cover in a corpus object) and would open a Block D2.
Every theorem in `formal/` and every scored block stands regardless.

## §8 Outputs
`results/exp_09_block_d_reach_<ts>.json` (+ log), append-only; outcomes journal
`journals/2026-09-0X_exp09_outcomes.md` citing this seal's hash; README row D; THEORY_MAP Block D row;
`formal/conjectures/m18_open.md` last row; corrections.md if T4 adds anything.
