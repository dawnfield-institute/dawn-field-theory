# 2026-08-31: Pre-registration — Block A (instrumentation) and Block B (the E₈ split)

Registered before any script is written or run. Invariants only; no absolute coordinates.

## Block A — Instrumentation (exp_01–exp_03)

Every answer below is known in advance from the classical theory; Block A verifies that OUR
constructions reproduce them. A failure here is an instrument fault, never physics.

**Frame declaration (§2.7.6).** Sampled: root systems and Cartan spectra of A₄, D₆, E₈,
constructed from their diagrams. Expectation: textbook values. Same scope by construction —
no sublattice, no windowed expectation.

### exp_01 — The three foldings exist and are exact (4 tests)
- T1: Constructed root counts equal (rank × Coxeter number): 20, 60, 240. Exact integers.
- T2: Constructed H₂/H₃/H₄ root counts equal 10, 30, 120; every folding is exactly 2:1.
- T3: Coxeter numbers match across each folding pair: (5,5), (10,10), (30,30). Exact.
- T4: A₄ is the UNIQUE A_n (2 ≤ n ≤ 12) whose full Cartan spectrum lies in ℚ(φ);
      the spectrum equals {2−φ, 3−φ, 1+φ, 2+φ} as exact algebraic numbers (no floats).

### exp_02 — The projection carries φ (4 tests)
- T1: An explicit E₈ → 4D projection maps the 240 roots onto two H₄ root systems of 120.
- T2: The scale ratio between the two copies is φ, exact to the working algebra
      (symbolic where possible; else better than 1e-12 relative).
- T3: The projection matrix's entries lie in ℚ(√5) — φ is IN the map, not in the data.
- T4: Same construction for D₆ → H₃ ⊕ φH₃ (60 → 30+30) succeeds with the same properties.

### exp_03 — Negative controls (4 tests)
The instrument must be able to say NO.
- T1: A₅ (spectral field ℚ(√3)) admits no φ-scaled 2:1 self-folding under the same procedure.
- T2: D₅ and E₆ (Coxeter numbers 8, 12 — no H partner) fail the folding construction.
- T3: A random orthogonal 4D projection of E₈'s roots does NOT produce two φ-scaled copies
      (n = 100 draws; count of accidental φ-splits reported, expected 0).
- T4: Shuffled root labels destroy the split (the construction reads geometry, not labels).

## Block B — The live prediction (exp_04–exp_05)

**P1 (from the registry).** Any E₈-derived eigenvalue multiset in the Milestone-R spectral
line (exp_20–exp_27 data, as committed) splits into two sub-multisets S, S′ with |S| = |S′|
and matched-element scale ratio φ.

**Frame declaration.** Sampled: eigenvalue multisets ALREADY COMMITTED in milestone-r results
(no new runs, no selection). Expectation: the φ-split predicted by exp_02's verified
instrument. Both sides are whole-spectrum objects — same scope. The comparison statistic is a
ratio between SETS, invariant under relabeling and rescaling of either set.

### exp_04 — The split on corpus E₈ data (4 tests, thresholds fixed now)
- T1: Partition into equal halves with median matched-pair ratio within 1% of φ.
- T2: The φ-partition beats the best of 1000 random equal partitions on the same statistic
      (empirical p < 0.01).
- T3: The split is stable across every E₈-derived dataset present (no cherry-picking; all
      datasets enumerated in the outcomes journal before analysis).
- T4: A-family and D-family spectra of the same sizes do NOT pass T1–T2 (specificity).

### exp_05 — Sensitivity (4 tests)
- T1: The statistic degrades monotonically under controlled noise injection (instrument
      responds to signal loss).
- T2: Detection survives 10% eigenvalue deletion (robustness to incomplete spectra).
- T3: Detection fails on φ-free synthetic spectra with matched density (false-positive floor
      < 5% over 1000 draws).
- T4: Result unchanged under the two legitimate eigenvalue conventions in the corpus
      (Laplacian vs Cartan, λ vs 1/λ), or the dependence is derived and declared.

## Falsification

Block A failing = instrument fault; fix and re-run (scores may lower; that is correct
behaviour). Block B failing WITH Block A green = P1 is false = the thesis is wounded per the
kill sentence. If Block C subsequently also fails, Milestone 18 dies and says so.

---

**Layer (forward note, 2026-09-02, per the re-separation):** Block A validates instruments (feeds the milestone's `core/`); Block B measures a physical reach (feeds `theory/` via THEORY_MAP and ROADMAP). Mixed by design at founding; later phases are single-layer.
