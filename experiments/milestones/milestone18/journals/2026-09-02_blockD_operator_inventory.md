# Block D — inventory of corpus operators (2026-09-02)

**Mode: EXPLORING (read-only survey; no certificate value computed on any admissible object).**
Layer: prepares a *physical reach* registration (feeds `theory/` via THEORY_MAP and ROADMAP).

The fold certificate — tr(R·D) = 2/√5 on the Bezout reflection, with its companion laws — needs an
**operator** (a rational symmetric matrix with a vertex basis and the graph's own degree matrix), not
a spectrum. This journal lists what the corpus actually contains, what is admissible for Block D
(not ADE by construction, exact, rational), what is unreachable and why, and where full-precision
artifacts exist or must be regenerated. Grades are the Cartan-channel σ-pairing grade over ℚ(√5)
(strict / core / partial / none, as in `core/foldlaws.py`).

## Three findings that shape the block

1. **The Milestone-R "spectral line" cannot carry P1 in either form.** Its only eigenvalue arrays are
   4-dp Laplacian spectra of ADE Dynkin graphs (`sidecars/milestone-r/results/exp_18_…_102622.json →
   test_results.T3.pair_results[].eigs*`; produced by `scripts/exp_18_…py` L295–300, `exp_20 …py`
   L118–127). Over ℚ(√5) the Laplacian characteristic polynomials of A₄, D₆ and E₈ have grade
   **none** — golden content lives in the Cartan channel only (the exp_07 knife-edge seen from the
   data side). Even in the Cartan channel the E₈ σ-split is a Galois *pairing*, not a *scaling*
   (matched-element ratios {46.9, 1.335, 0.859, 0.874}); the φ ratio exp_02 verified is between
   **root-shell radii** (`scripts/exp_02_projection_carries_phi.py` L22–31), a different object.
2. **Regular graphs are certificate-blind, by theorem.** tr(P_off) ∈ ℚ ⇒ tr(σP_off) = tr(P_off) ⇒
   tr(R_off) = 0 ⇒ tr(R_off·D) = d·0 = 0; B = (2−d)I commutes with P ⇒ leak ≡ 0; vertex-transitive
   ⇒ R_vv ≡ 0. M15's affine cycles (C₅, C₁₀, C₁₅ are golden-graded), Petersen and K₆ are outside the
   instrument's domain — enumerated as unreachable, never scored.
3. **The theory's own object is golden, √2-scaled, and odd.** Complete PAC binary trees of depth 3, 4,
   5 (n = 15, 31, 63) grade **partial** with exactly one conjugate pair, (t² − 4t + 1 ± √5)^m — the
   h = 5 (H₂-type) content of the tree's radial quotient, a √2-weighted path of length 4. The same
   √2-path table predicts the first h = 10 (H₃-type, exponents {1, 3, 7, 9}) sector at radial length
   9, i.e. **depth 8, n = 511**, as (t² − 4t − 1 ± √5). n = 2^(d+1) − 1 is odd, so strict/core grade
   and the matching form are impossible by the parity theorem; only the off-core certificate is
   testable on PAC trees.

## Inventory

| # | Operator | Produced by | Type / size | ADE baked in? | Artifact | Grade (pre-seal) | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | ADE Dynkin adjacency / Cartan | `milestone12/core/connection_geometry.py` L81–143 (`DynkinDiagram`, `cartan_matrix`), L191 (`all_ade_diagrams`); `milestone-r/core/radiation_physics.py` L191 | trees, n ≤ 8 | **yes — the input** | M-R exp_18 JSON (4-dp Laplacian) | A₄ strict, D₆ core, E₈ strict; A₅/D₅/E₆/E₇ none | **Known-answer controls only.** Certifying them is tautological (§2.8). |
| 2 | Laplacian spectra of ADE graphs (P1's named data) | M-R `exp_18` L295–300, `exp_20` L118–127 | 4-dp arrays | yes | as above | A₄/D₆/E₈ Laplacian: **none** | Cannot carry P1 or the certificate; declared golden-free. |
| 3 | Weighted A→D/E deformations `weighted_dn/en(n, w)` | M-R `exp_21` ~L61–75, `exp_22` ~L56–90 | weighted trees, w real | yes | derived scalars | not rational unless w ∈ ℚ | Excluded (non-rational; ADE-interpolated). |
| 4 | Affine cycles C_m (Ã_{m−1}) | `milestone15/core/representative.py` L68; M15 exp_01 L139–150 (m = 4..13), exp_04–06 (m to 2000) | regular, integer | affine ADE | regenerable | C₅, C₁₀, C₁₅ partial; others none | **Declared blind** (regular ⇒ certificate ≡ 0). |
| 5 | Random unicyclic holonomy controls | `representative.py` L90; M15 exp_01 L185–195: `RandomState(152)`, m ∈ {7, 9, 11}, 20 each = 60 | unicyclic, non-regular | no | regenerable bit-exactly | to grade pre-seal | **Admissible (O3)**, physically motivated (M15 exp_01 T4 controls). |
| 6 | Tadpoles | `representative.py` L76 | unicyclic | no | never used by a script | C₅+tail partial | Construction control only, declared. |
| 7 | Complement-frame connection, edge transports, holonomy H; momentum generator G_k | `representative.py` L139–194; M15 `exp_06` L31–37 | orthogonal / rational skew on a **mode** basis | no | results exp_04–06 | n/a | **Unreachable** — no vertex basis, not symmetric (G₃'s field is ℚ(√106)). |
| 8 | **Complete PAC binary tree** (adjacency) | `connection_geometry.py` L263 (`pac_tree`); `radiation_physics.py` L851; `midnight/core/phase_rate.py` L249 (φ-potentials as vertex data); `studies/exp_33_black_hole_cascade/scripts/exp_33d_…py` | trees, n = 2^(d+1) − 1 | **no** (φ enters only as vertex potentials) | midnight exp_02 (depth 6); regenerable | d = 1, 2 none; **d = 3, 4, 5 partial**, single pair; h = 10 predicted at d = 8 | **Primary admissible object (O1).** The φ-weighted variant is excluded (φ baked in as magnitude). |
| 9 | **PAC growth trees** (stochastic branching, collapse pruning) | `studies/prime_growth_dynamics_v2/core/phase_engine.py` L350–395 (`evolve_pac_tree`); exp_07 L40/L71, exp_08 L38/L101, exp_09 L80 (seeds recorded) | random rooted trees, tens–hundreds of vertices | no | **topology not stored**; regenerable only by an RNG-replay harness gated on the recorded `total_nodes/depth_reached/collapse_events` | to grade pre-seal | **Admissible (O2)** — the only physically generated tree ensemble in the corpus; declared *new* if the harness cannot replay bit-exactly. |
| 10 | M13 density-lump graphs | `milestone13/core/identity_complement.py` L465; exp_11 L162/L175 (n = 12), sweep L455–460 (18 graphs) | chain + extra edges, integer | no | regenerable | n = 12/e = 4: none | Admissible (O4); expected informative count ≈ 0, declared. |
| 11 | Petersen, K₆ | `identity_complement.py` L798, L818 | regular | no | — | none | Blind and golden-free; declared. |
| 12 | ER random connected graphs | `identity_complement.py` L848; M13/M15 seeded | general symmetric | no | regenerable | — | Null source, not an object. |
| 13 | Product graphs (adj □ adj, adj = ADE) | `milestone14/core/quantum_complement.py` L489, L511; exp_09 L40 | grids | **inherited** from the ADE factor | regenerable | P₂□P₄ strict, P₃□P₄ strict, P₄□P₄ core, P₂□P₅ none | Excluded from objects (golden content inherited); usable as declared inheritance controls. |
| 14 | M-R severance/ledger objects | `radiation_physics.py` L201–259 (ADE minus a vertex), L675 (D⁻¹A, non-symmetric), L461–492 (real-noise adjacency) | sub-ADE / non-symmetric / real | yes / n/a | results exp_01–16 | — | Excluded. |
| 15 | Cascade coupling kernels | `studies/ade_cascade/core/coupling.py` L51–63; `milestone4/core/utils.py` L141 | real kernels on ADE | yes | results | not in ℚ(√5) | Excluded. |
| 16 | Lattice-fluid Laplacians; scope transfer matrices | `studies/confluent_identity/scripts/_shared.py` L52–75; `milestone6/core/scope.py` L64, L176 | real-weighted; torus skeleton | no | results | not rational; skeleton regular | Excluded (doubly). |
| 17 | Percolation clusters; reality-engine neighbour graphs | `milestone17/core/criticality.py` L470, L355; `reality-engine/proof_of_concepts/v4/particles.py` L149–160, L427–436; `structure.py` L375 | integer, state-dependent, n ~ 10³–10⁴ | no | engine state | — | **Deferred**: needs a declared binning/kNN frame to define a vertex basis; exact certificate infeasible at that size. |
| 18 | Midnight number-field objects; M13 Lie generators / Killing forms | `phase_rate.py` L115–147; `identity_complement.py` L717, L332, L588 | scalars / Lie-algebra matrices | — | — | n/a | Not vertex-basis operators. |

**Positive controls for an H₃-sector detector that are not D₆:** the n = 14 mixed trees
(`results/explore_r9_mixed_trees_n14.json`) — det −80 carries exactly H₃'s off-core pair
(t² − 4t + 3/2 ∓ √5/2, from `core/ledger.py` L26/L35) on two swapped D₆ subtrees; det −620 carries
the A₄/H₂ pair the same way. **Specificity controls:** the four asymmetric strict trees at n = 20
(exp_15 T2; `results/exp_15_n20.json`, `partnered: false`) — certificate values already computed
∉ {2/√5, 2/5}, no matching form, |R_vv| not constant: the certificate's one-sidedness survives the
first non-partnered strict objects.

## What P1 asserts, and on what

As registered (README Predictions registry, item 1; `journals/2026-08-31_blockA_blockB_preregistration.md`
L38–45): a split of *eigenvalue multisets* from the M-R line into S, S′ with |S| = |S′| and
matched-element scale ratio φ, with the frame "the φ-split predicted by exp_02's verified
instrument". exp_02 verifies the ratio of **root-shell radii** (two shells of 120 projected roots),
not eigenvalues; and the named data are Laplacian spectra, golden-free for A₄/D₆/E₈. Root shells
and spectra are different scopes: **P1 is a frame artifact inside a registration** (STANDARDS §2.9,
`docs/frame-control-standard`). On eigenvalue multisets the true statement is Galois pairing —
the Ledger theorem, Block C. Block D therefore tests the **certificate**, and P1 is
forward-corrected in the registry rather than scored on data it cannot apply to.
