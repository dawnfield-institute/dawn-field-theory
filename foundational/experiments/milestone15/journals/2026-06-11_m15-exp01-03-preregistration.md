# Pre-Registration: M15 exp_01–exp_03 (Class-Level Re-Posing of the Continuum Failures)

**Date:** 2026-06-11
**Status:** REGISTERED BEFORE EXECUTION — registered quantities not computed as of this
commit (smoke tests of builders/transport only). Basis: `2026-06-11_m15-founding.md`
(same commit). All claims relational per the invariant-registration rule
(midnight outcome journal, 2026-06-11).

---

## exp_01: Rapidity as Connection — the Arc, the Chord, and the First Holonomy

Re-poses M13 exp_08 (rapidity composition, 99–292% errors). Diagnosis registered up
front: exp_08 compared *chord* (pairwise complement-spectral distance) to *arc* (path
sum) — a category error, since the pairwise distance is not a path metric. Further:
any scalar or vector quantity built from per-vertex spectra is a *potential*, hence
exact, hence trivially additive along paths and zero around cycles. The genuinely
non-exact object is the **complement-eigenvector connection**: orthogonal (Procrustes)
transport of each vertex's complement eigenframe to its neighbor's over shared support
(`core/representative.py: edge_transport`, top-k eigenvectors, k = 2 primary, k = 3
robustness; per-vertex frames computed once; degeneracy guard via eigengap, reported).
This is M13's definitional parallax made into a connection.

**Honest uncertainty registered:** affine-A cycles are vertex-transitive; the holonomy
could be exactly trivial by symmetry, or a genuine rotation. We do not know. T3's kill
condition is live.

### Registered tests

- **T1 [harness validity]:** the scalar potential g(v) = Σ complement spectrum
  telescopes exactly along every path (machine precision) on A_5..A_8 — confirming the
  exact sector is exact and locating exp_08's failure in the chord/arc confusion, not
  in path composition.
- **T2 [relation]:** the chord–arc deficit (shortest-path arc length minus chord, in the
  unsigned deformation metric) is a CLASS quantity on A_8 and D_6: (a) its multiset is
  invariant under 3 random relabelings (tol 1e-9); (b) it is constant across
  automorphism-equivalent vertex pairs (within machine tolerance); (c) chord = 0 ⟺
  same orbit (the Lemma, verified in this construction).
- **T3 [relation]:** affine holonomy on Â_n (cycles C_{n+1}, n = 3..12, k = 2):
  (a) deficit ||H − I||_F > 1e-6 for all n (non-trivial connection);
  (b) rotation-angle spectrum invariant under 3 random relabelings (tol 1e-8);
  (c) leading angle rank-stable: CV < 0.1 over Â_8..Â_12.
  The VALUE of the angle is reported [D], not scored.
- **T4 [relation]:** at matched |V| (m = 7, 9, 11), the Â holonomy deficit is extremal
  (max OR min — registered as either-end extremality; direction reported [D]) against
  20 random unicyclic controls per size, in ≥ 80% of sizes tested. Maximal symmetry ↔
  extremal holonomy.

**KILL:** deficit multiset varies under relabeling (not a class quantity); or T3(a)
fails for all n with k = 2 AND k = 3 (the connection is flat — no curvature content
in the complement frame bundle); or Â is unexceptional among controls (T4 fails AND
angles unstable).

## exp_02: Coherence Limits Per-Scope

Re-poses M13.5 exp_15 (0/4). Same rate (max single-step complement deformation,
`max_deformation_rate`), extended range (ranks 3..28), claims restated per-scope.

- **T1 [relation]:** the A-family even and odd parity classes EACH converge: last-5
  CV < 0.05 per class over the extended range (exp_15 had only 3 points per class;
  even CV was 0.086 — this is a genuine retest with more data, not a re-run).
- **T2 [relation]:** class-limit RATIOS are scope invariants: r₁ = lim(A_even)/lim(A_odd)
  and r₂ = lim(D)/lim(A_even), each computed in two disjoint rank windows
  (14–20 and 22–28), agree within 5%. Ratio values reported [D]; any constant match
  reported as derived consequence only, not scored.
- **T3 [relation]:** per-scope constraint restated: each ADE class's last-5 CV is
  smaller than the CV of 20 random connected graphs at matched size (the exp_15 T3
  "paradox" retested with class scoping; an honest second failure is reportable).

**KILL:** either parity class fails to converge over the extended range — the
oscillation is then not class structure and the per-scope re-pose dies.

## exp_03: The Representative Gauge (stretch — registered now, run when session allows)

Re-poses M14 exp_06 (vertex interference ≡ 0). Claim: a position representative of the
(passing) orbit-interference class requires Aut-breaking frame data; visibility scales
with the breaking and vanishes in the symmetric limit.

Setup: D_4; orbit superposition with relative SEC phase (M14 exp_06 T1 construction,
reimplemented minimally); Hamiltonian = graph Laplacian; perturbation ε added to ONE
leaf edge weight (3 choices, orbit-equivalent); vertex-space interference visibility
V(ε) over ε ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}.

- **T1 [relation]:** V(0) ≤ 1e-10 (reproduces M14's failure as the symmetric limit)
  and V is consistent with → 0 as ε → 0 (two smallest nonzero ε give the two smallest
  nonzero V).
- **T2 [relation]:** V(ε) monotone increasing (Spearman ρ ≥ 0.9 over the grid).
- **T3 [relation]:** the small-ε scaling exponent p (fit V ∝ ε^p over the four smallest
  nonzero ε) is equal across the 3 orbit-equivalent edge choices: CV(p) < 0.1.
  p's value reported [D].

**KILL:** V(0) > 1e-6 (contradicts M14's own result — harness bug or conjecture wrong);
or exponents differ across orbit-equivalent directions (gauge response is not
class-determined).

**DEFERRAL (pre-registration smoke test, before this commit):** the minimal harness does
NOT reproduce M14's null — evolved vertex visibility V(0) = 0.32 in the symmetric graph,
because M14 exp_06's null concerns *static* superposition cross-terms (orbit indicator
states have disjoint vertex supports), while Laplacian evolution mixes orbit amplitudes
onto common vertices and produces real vertex-space interference. Two consequences,
recorded honestly: (1) exp_03 is DEFERRED until it can be posed on a faithful
reimplementation of M14 exp_06's measurement construction — registering it on this
harness would test a claim the harness cannot anchor; (2) the wrinkle itself is
substantive — M14's "no positional interference" is a statement about static
cross-terms, and dynamical vertex interference on the symmetric graph deserves its own
look (possible scope-note for Paper 11 later). exp_01 and exp_02 are unaffected and
register as written.

## Outcome commitment

All outcomes journaled and integrated whichever way they land. If any KILL fires,
falsifier 3 of the M15 founding document is engaged and scored against the conjecture.
