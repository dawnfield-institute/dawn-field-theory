# Milestone 18: The Non-Crystallographic Completion

**Status**: active (founding)
**Founded**: 2026-08-31
**Origin**: An independent derivation arriving from outside the corpus (see Provenance), plus
a pattern of failures inside it. The corpus's φ-hunt has produced one multi-seed survivor
and a long casualty list; every casualty sought φ as a *magnitude* inside crystallographic
objects. This milestone tests whether φ enters the framework as a *projection* instead.

---

## The Thesis

**The corpus's ADE structure completes non-crystallographically.** φ enters DFT not as a
magnitude but as the folding projections A₄ → H₂, D₆ → H₃, E₈ → H₄ — and the limits the
φ-hunt kept hitting are the crystallographic boundary, seen from inside.

The three foldings are exact, classical, and carry known answers:

| crystallographic | roots | h | → | non-crystallographic | roots | h |
|---|---|---|---|---|---|---|
| A₄ (Cartan spectrum {2−φ, 3−φ, 1+φ, 2+φ} — the unique golden A_n) | 20 | 5 | → | H₂ (pentagon) | 10 | 5 |
| D₆ | 60 | 10 | → | H₃ (icosahedron) | 30 | 10 |
| E₈ | 240 | 30 | → | H₄ | 120 | 30 |

Every folding is 2:1; the two image copies sit at scale ratio exactly φ. In this reading the
golden quadratic form a² + ab − b² (disc +5, signature (1,1), indefinite — *not* a lattice
norm) is what a crystallographic norm looks like **after** folding, and five-fold symmetry is
where the crystallographic restriction (2, 3, 4, 6 only) ends.

## Kill sentence

Bound to the thesis, not to any instrument: **if E₈-derived spectra do not split into two
φ-scaled families (Block B), and the indefinite golden form adds no separating power at the
orbit boundary (Block C), then φ is not structural in this corpus and this milestone dies.**
Either failure alone wounds it; both together kill it.

## Scorecard: 51/67 — Phases 3–7c and Block D run (A 12/12 · B 3/5 + 7: 1/2 + 7b: 1/2 + 7c: 1/2 · C 6/8 · C-ext 21/29 · D 2/3 · E 4/4)

**Kill-sentence status (honest accounting):** clause B (no golden split) is NOT met — the
split is a theorem in the Cartan channel; clause C (no separating power at the orbit
boundary) IS met — exp_06 T4 proved σ-ledger and orbit quotient irreconcilable at D₆'s
core. One clause wounds, both kill: **wounded, not dead.** Phase 2 registration: `journals/2026-09-01_phase2_registration.md` (sealed 06073227).

| Block | Question | Experiments | Score | State |
|---|---|---|---|---|
| **A — Instrumentation** | Construct the three foldings against textbook answers | exp_01 – exp_03 | **12/12** | complete — see `journals/2026-08-31_blockA_outcomes.md` |
| **B — Dynamical self-duality** | Does dynamics see the mirror — and now, the branch? | exp_04 – exp_05, exp_16 – exp_18 | **6/11** | exp_04 FAIL (74bcd0df); exp_05 3/4: rational dynamics is σ-blind (T1 theorem). **Phase 7 / exp_16 1/2** (`journals/2026-09-02_exp16_outcomes.md`): the defect edge of a strict fold is σ-FIXED (√5·R = S + 2Π is an integer matrix), so the σ-blindness theorem no longer forbids the question; T1 (defect distinguished in degree-matched classes under the heat kernel) FAIL — underpowered by design (15/21 folds uninformative; lesson 7), residual direction all-maxima; T2 null calibration PASS. **Phase 7b / exp_17 1/2** (`journals/2026-09-02_exp17_outcomes.md`): Π-asymmetry decays from the defect on 18/18 folds and peaks adjacent to it on 21/21, but at n ≤ 16 the permutation null cannot separate branch from geometry (T1 fail) and the d ≤ 1 null is saturated (T2 pass by one; lesson 8). **Phase 7c / exp_18 1/2 at n = 20** (`journals/2026-09-02_exp18_outcomes.md`): **T1 PASS 47/47** — the sheet-asymmetry peak sits AT the branch on every fold (null expectation 15, quantile 22; β-robust; copy sheet > conjugate 47/47): the first powered dynamical fingerprint of the reinjection port. T2 (profile organized uniquely by the branch vs its neighbours) FAIL as sealed — a null-resolution limit |
| **C — The σ-Ledger** | Conjugation = complementation; the knife-edge under duality | exp_06 – exp_07 | **6/8** | exp_06 2/4 (T3/T4 fail AT the mod-5 ramification locus, T4 by impossibility proof); exp_07 **4/4** (5\|n and 5\|(n+1) enumeration predicted and confirmed) |
| **C-ext — The census and the fold laws** | Which trees are completely σ-paired, and what laws do the folds obey? | exp_08, exp_11–exp_15 | **21/29** | exp_08 1/2; exp_11 **4/4** at n = 14 (3939176d); exp_12 **3/6** (4ab2ff35; T3/T5/T6 fail as sealed, each replaced — see journal); exp_13 **5/7** at n = 16 (`journals/2026-09-01_exp13_outcomes.md`): the invariant held on **13/13 fresh strict folds**; T2 fail = the first sector-strict tree (det −775, strict but not a one-5 fold — the r9 mixed mechanism producing strictness); T4 fail = cospectral 5-bond placements (the fold's vertex structure selects the placement; halves law holds under some-partner semantics). Core anatomy (r11–r14): fold laws are off-core laws; modulated vertex law √5·R_off,vv = ±(1−Qc_vv). exp_14 **3/4** (`journals/2026-09-01_exp14_outcomes.md`): off-core instrument reaches all 80 core folds at 16 (0 declared vs 76); Lemma-1 identity proved and checked 80/80; selector sufficiency + layer order confirmed out of sample; the exact-iff clause failed — a third kernel class ((t−2)⁴, masses {2/5, 3/5}) is clean, characterization open. r15 (exploration, 20/20 all strict folds ever found): √5·R = S + 2Π — a signed perfect matching; the Π-quotient IS the one-5 diagram with the realized 5-bond as the unique multiplicity-3 edge; the entire strict-law package reduces to this one conjecture. **Construction theorem** (r17): parents are branched double covers (γ² + γ − 1 = 0 decoupling) — one-5 existence PROVED, every strict law a theorem on constructions. **Phase 6 / exp_15 5/6 at n = 20** (`journals/2026-09-02_exp15_outcomes.md`): 823,065 trees, 66 strict on 64 polynomials; T1 zero orphans at k = 10; **rigidity holds 47/47** (T3–T6 all pass; 6 degenerate partners declared); T2 FAIL = a third strict species — four ASYMMETRIC strict trees (trivial automorphism group, no one-5 partner) |
| **D — Physical reach** | Is the H₃ class reachable through the folding (M15 test)? | exp_09 | 2/3 | **RUN 2026-09-03** — sealed `0130ebe0` (`journals/2026-09-03_blockD_registration.md`): T1 reach **FAIL** (0 carriers / 173 live objects: PAC trees d ≤ 8, 166 growth trees, 4 unicyclics), T2b class-without-representative **PASS** (4/4 cells), T4 P1 forward-correction **PASS**; gate T2a held (H₃-type sector first at d = 8). The reach at measured sizes is retired; the mathematics stands. Outcomes `journals/2026-09-03_exp09_outcomes.md` |
| **E — Re-derivations** | φ⁻⁴ (exp_37) and z_c = φ⁵ as units, not magnitudes | exp_10 | **4/4** | exact (974380c6) |

Block A is front-loaded per M17's rule: no DFT system is measured until the instruments
verify against answers known in advance. Block D is gated on B **and** C so the milestone
cannot drift into compendium mode — the failure shape it is most exposed to (see Honest
risks). **Gate status (2026-09-02):** exp_18 T1 passed 47/47 against a null computed before the
run — the Honest-risks criterion ("a result that could have failed and didn't") is met by
Block B; opening Block D is a decision recorded in the commit that seals its registration, not
a consequence of this line.

**Theorems filed: 8 · open conjectures: 8 · killed: 1** — in [`formal/theorems/`](../../../formal/theorems/README.md)
and [`formal/conjectures/m18_open.md`](../../../formal/conjectures/m18_open.md). The scorecard
above counts sealed tests; this line counts proofs. Neither is the other. The kill (2026-09-02,
r21): the *polynomial* form of integrality, 5·b(t) ∈ ℤ[t], fails on three construction parents at
n = 20 (den 3); the matrix form stays a theorem. The eighth theorem, proved the same evening, is
the denominator bound den(5·b) | 2^{deg q₁}·Res(q₀, q₁) — the ramified prime is controlled
outright, the unramified ones by the diagram resultant; the exact denominator is the new open row.

## Layers

The milestone's work is kept in three layers, each feeding a different part of the repository
(STANDARDS §2.7 item 7, on `docs/frame-control-standard`). Every sealed registration carries a
forward note naming its layer.

- **Mathematics → `formal/`.** Phases 3–6 (exp_11–exp_15): the one-5 conjecture, the fold
  laws, the off-core instrument, the matching structure. Closed to one conjecture (rigidity);
  the theorems are indexed, the conjectures filed.
- **Physics → `theory/`** (via THEORY_MAP and ROADMAP). Blocks B and D; Phases 7–7c
  (exp_16–exp_18): dynamics at the branch. Block D sealed and run 2026-09-03 (0 carriers / 173;
  `journals/2026-09-03_blockD_registration.md`, outcomes `journals/2026-09-03_exp09_outcomes.md`;
  the draft and inventory journals kept as lineage).
- **Instruments → `core/`.** Block A (the foldings), the census and its known-answer gates
  (exp_08, exp_11b), the fold-law machinery, the matching and sector extraction — as importable
  modules (`core/ledger.py`, `folding.py`, `foldlaws.py`, `census.py`, `matching.py`,
  `sectors.py`), never `exec`'d.

## Predictions registry

Registered as invariants, never coordinates (STANDARDS §2.7.2):

1. **P1 (Block B).** Any E₈-derived eigenvalue multiset in the corpus (Milestone-R spectral
   line) splits into two sub-multisets S, S′ with |S| = |S′| and scale ratio φ between
   matched elements — a ratio between *sets*, representation-free. Frame declared per §2.7.6
   before any measurement.
2. **P2 (Block C).** Either the M13.5 Lemma's conclusion extends to indefinite invariant
   forms (PSD premise not load-bearing), or the golden form separates same-orbit vertices —
   exactly one of these holds, and which one is decided by construction, not tuning.
3. **P3 (Block E).** exp_37's surviving 1/φ⁴ is re-derivable as a unit norm statement
   (N(φ⁻⁴) = +1) rather than a fitted magnitude.

**Forward note on P1 (2026-09-02; the registered text above is untouched).** P1 conflates two
scopes: the φ ratio that exp_02 verified is between the *root-shell radii* of the two 600-cells,
not between elements of an eigenvalue multiset; and the Milestone-R spectral line it names
stores Laplacian spectra, which over ℚ(√5) are golden-free for A₄, D₆ and E₈ (the golden content
lives in the Cartan channel — the exp_07 knife-edge). On eigenvalue multisets the true statement
is the Galois pairing S′ = σ(S) — the Ledger theorem, Block C. P1's ratio form is therefore a
frame artifact inside a registration (STANDARDS §2.9) and is retired as such; its successor
prediction was Block D's fold certificate on the corpus's own operators
(sealed `journals/2026-09-03_blockD_registration.md`; run 2026-09-03: 0 carriers of 173 — the successor
prediction failed cleanly, `journals/2026-09-03_exp09_outcomes.md`). Recorded in `theory/corrections.md` §6.

## What is postdictive (declared)

The founding observation — that the foldings retro-explain the φ casualty list (M-R exp_02,
exp_05, exp_06; confluent_identity exp_38; sec_prime_manifold's 1/φ, whose headline reading is
unsupported once the frames are matched — a frame artifact, STANDARDS §2.9) — is **postdictive** and is used as a bearing, not as
evidence. The falsifiable content of this milestone is P1–P3.

## Honest risks

- The A₄/H₂ observation was found quickly, is elegant, and explains a lot — the exact
  profile of three results killed by their own controls on the day this milestone was
  founded. Hence the front-loaded Block A and the gate on D.
- The mirror failure: a survey that finds its own geometry everywhere has zero
  discriminating power (§2.8's tautology test applied to corpora). Block D is the block
  most exposed; it does not open until B and C have produced at least one result that
  could have failed and didn't.

## Provenance & attribution

The non-crystallographic reading arrived by **independent convergence** on 2026-08-31:
**Andy Farmer** (University of Southampton; WOLF collaboration), working from quadratic
forms, Eisenstein integers, and exactly-solved lattice models — sent two documents (`GOLDEN_RATIO_ALGEBRA.md`,
`triangular_lattice_analysis_for_peter.md`) that independently identified: the golden
algebra ℤ[φ] with its canonical form and unit structure; the A₂/Eisenstein lattice as the
fundamental triangular geometry; the three-way Gaussian/Eisenstein/golden norm
classification; and Baxter's hard-hexagon critical activity as a golden-integer object
(z_c = 3 + 5φ, which his own power-reduction rule closes to φ⁵ — a unit, N = −1).
His pentagonal geometry is H₂ and his icosahedral coordinates are H₃ — the images of this
corpus's A₄ and D₆ under the foldings above. **Independence, stated accurately (corrected 2026-08-31):** the convergence is *partially*
independent — Andy had prior exposure to this corpus's φ/ADE claims through the March–April
2026 WOLF collaboration (the ADE framework and the SEC 1/φ partition were shared with him),
and his analyses are LLM-assisted, so public material may also inform them. What is
genuinely independent — and is the founding evidence — is the *route*: A₂/Eisenstein
lattices, the three-norm classification, and the Baxter connection arrived through classical
number theory and exactly-solved models, none of which this corpus had used. Convergence of
routes, not of vocabularies. Recorded in
`journals/2026-08-31_founding.md`. Block A's deliverable doubles as the validation package
returned to him.

## Dependencies

- M13/M13.5 (identity-IS-complement; the PSD Lemma) — Block C attacks the Lemma's premise.
- M15 (DFT-Hodge conjecture) — Block D is a direct test: H₃ classes reachable through
  folding ⇒ conjecture extended; unreachable ⇒ a hole exactly where φ lives.
- Milestone-R spectral line (exp_20–exp_27) — Block B's data source.
- STANDARDS §2.9 (frame artifacts) — every block declares its frame at pre-registration.

## FDO Links

- Lore typed node `milestone18-noncrystallographic-completion` (`meta.yaml: fdo`).
- Related: `milestone15-representative-problem` (Block D is its direct test); the Milestone-R node.

## Forward path

A (instruments) → B ∥ C (prediction + Lemma boundary) → gate → D (physics) + E
(re-derivations) → synthesis. Two-perspective review at every phase gate; the independent
reviewer for this milestone is by preference the person whose framework is *actually*
independent (see Provenance).
