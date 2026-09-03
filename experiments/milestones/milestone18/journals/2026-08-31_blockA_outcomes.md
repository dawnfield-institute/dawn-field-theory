# 2026-08-31: Block A outcomes — 12/12, after two instrument faults did their job

Registration: cf886c00 (`2026-08-31_blockA_blockB_preregistration.md`). All thresholds as
registered; none relaxed.

## Scores

| Exp | Result | Notes |
|---|---|---|
| exp_01 | **4/4** | Root counts 20/60/240 exact; H2/H3/H4 constructed explicitly (10/30/120, every folding 2:1); Coxeter element orders 5/10/30 computed, matching both sides; A4 the unique golden A_n with spectrum {2−φ, 3−φ, 1+φ, 2+φ} exact (sympy, no floats) |
| exp_02 | **4/4** (first run 2/4 — instrument) | E8 roots project to two shells of 120; radius ratio = φ at **2.2e-16** (machine epsilon); the orthogonal projector onto the H4 subspace has exactly six distinct entries, **all golden rationals**: 0, ±√5/10, 1/2, (5±2√5)/10 — φ is in the map, not the data; both shells verified as genuine H4 root systems (600-cell inner-product spectrum + reflection closure); D6 → H3 ⊕ φH3 succeeds (30/30, ratio φ) |
| exp_03 | **4/4** | A5: zero φ-splits over all eigenplane subspaces; D5 zero; E6 (constructed as the 72-root subsystem of E8, count verified) zero; 100 random 4D projections of E8: zero accidental φ-splits; 20 isometric scrambles against the fixed H4 projector: split destroyed 20/20 |

**Block A: 12/12.** The instrument can produce the split where theory says it exists,
recover φ to machine precision, exhibit φ in the projector, and — the part that makes the
rest mean something — say NO everywhere it should.

## Honest notes: the two instrument faults (first exp_02 run scored 2/4)

Both first-run failures were instrument faults caught by registered thresholds, per this
milestone's front-loaded-instrumentation rule (M17's lesson):

1. **T2 failed its 1e-12 threshold at 1.9e-10** — the ratio was being computed from
   `shell_split`'s 9-decimal *bin labels* instead of raw norms. A bin label is not a
   measurement. Fixed: ratio from raw-norm means → deviation 2.2e-16. (The tell: the error
   was bit-identical across two different eigensolvers — deterministic, therefore not
   solver noise.)
2. **T3's golden-rational recognizer was algorithmically wrong** (rounded p before solving
   for q — a 2D recognition done as two 1D roundings). Rewritten as a proper per-denominator
   2D search; all six projector entries then recognized exactly.
   Additionally `eigenplane_basis` was moved from `eig` on the nonsymmetric Coxeter matrix
   to `eigh` on (W+Wᵀ)/2, whose eigenspaces at cos(2πm/h) are the invariant planes —
   ~1e-15 where the nonsymmetric solve loses five digits.

Thresholds were not touched; the instrument was fixed until it met them. That is the
intended direction of repair.

## Note on T4's construction (declared)

The D6 → H3 direction inside the 2D (−1)-eigenspace (exponent 5 has multiplicity two) was
located by a 1D scan and then verified against the registered criteria (30/30 shells, ratio
φ). The scan constructs the isotypic direction whose existence is the classical claim; the
test content is that the verification passes, and exp_03's controls show the same scan
finds nothing on A5/D5/E6.

## Next
Block B awaits re-conception (the registered form has the channel problem — conversation
with Peter first). Block C awaits its re-pose. Nothing further runs until those are settled.
