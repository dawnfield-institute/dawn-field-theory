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

## Scorecard: 33/41 — Phase 3 run (A 12/12 · B 3/5 · C 6/8 + census 1/2 · E 4/4 · C-ext exp_11 4/4 · exp_12 3/6)

**Kill-sentence status (honest accounting):** clause B (no golden split) is NOT met — the
split is a theorem in the Cartan channel; clause C (no separating power at the orbit
boundary) IS met — exp_06 T4 proved σ-ledger and orbit quotient irreconcilable at D₆'s
core. One clause wounds, both kill: **wounded, not dead.** Phase 2 registration (draft,
unsealed): `journals/2026-09-01_phase2_registration_DRAFT.md`.

| Block | Question | Experiments | Score | State |
|---|---|---|---|---|
| **A — Instrumentation** | Construct the three foldings against textbook answers | exp_01 – exp_03 | **12/12** | complete — see `journals/2026-08-31_blockA_outcomes.md` |
| **B — Dynamical self-duality** | Does dynamics see the mirror? | exp_04 – exp_05 | **3/5** | exp_04 FAIL (74bcd0df); exp_05 **3/4** (06073227): rational dynamics is σ-blind (T1 theorem); golden-probe leakage first-order and nonzero, but the sealed "golden coefficient" clause fails — provably, the coefficient is σ-fixed hence rational |
| **C — The σ-Ledger** | Conjugation = complementation; the knife-edge under duality | exp_06 – exp_07 | **6/8** | exp_06 2/4 (T3/T4 fail AT the mod-5 ramification locus, T4 by impossibility proof); exp_07 **4/4** (5\|n and 5\|(n+1) enumeration predicted and confirmed) |
| **C-ext — The census and the fold laws** | Which trees are completely σ-paired, and what laws do the folds obey? | exp_08, exp_11, exp_12 | **8/12** | exp_08 1/2 (165 classes, n ≤ 10; Andy's "iff for any graph" falsified; forward-corrected by the exhaustive n ≤ 13 census); exp_11 **4/4** (3939176d): every 7-node one-5 diagram has a 14-vertex tree parent, zero orphans, all core-grade, no strict tree at 14; exp_12 **3/6** (`journals/2026-09-01_exp12_outcomes.md`): T1 resonance law, T2 det = N(q(0)) 68/68, T4 strict invariant 7/7 pass; T3 fails as sealed (labels reversed; replaced by two signature theorems and a det-sign rule — three det > 0 folds at 14 have both copies indefinite); T5 fails as sealed (tr(RD) = 2/√5 on 12/12 of its domain, leakage 2/5 on 10/12); T6 fails as sealed (quotient value 0 on 8/11; two n = 14 golden trees fold by both mechanisms on separate automorphism sectors — explore_r9: quotient in the trivial sector, A₄→H₂ / D₆→H₃ Galois folds of swapped subtrees in the sign sector; exclusive per sector, exhaustive overall, exploration-grade). Standing: 2/√5 certifies a Coxeter-diagram fold, 19/19 |
| **D — Physical reach** | Is the H₃ class reachable through the folding (M15 test)? | exp_09 | 0/4 | **gated on B and C** (B closed 3/5 with its scored prediction failing on a self-forbidden clause — gate not yet satisfied) |
| **E — Re-derivations** | φ⁻⁴ (exp_37) and z_c = φ⁵ as units, not magnitudes | exp_10 | **4/4** | exact (974380c6) |

Block A is front-loaded per M17's rule: no DFT system is measured until the instruments
verify against answers known in advance. Block D is gated on B **and** C so the milestone
cannot drift into compendium mode — the failure shape it is most exposed to (see Honest
risks).

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

## What is postdictive (declared)

The founding observation — that the foldings retro-explain the φ casualty list (M-R exp_02,
exp_05, exp_06; confluent_identity exp_38; sec_prime_manifold's 1/φ, killed 2026-08-31 as a
frame artifact, STANDARDS §2.9) — is **postdictive** and is used as a bearing, not as
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

## Forward path

A (instruments) → B ∥ C (prediction + Lemma boundary) → gate → D (physics) + E
(re-derivations) → synthesis. Two-perspective review at every phase gate; the independent
reviewer for this milestone is by preference the person whose framework is *actually*
independent (see Provenance).
