# Block F registration — Orientation: the signed cycle and the first non-tree class (exp_20)

**Layer: mathematics (→ `formal/`), with an M15 cross-link** (the holonomy instrument is M15's, imported;
ROADMAP M15 item (a), "ℤ₂ twist classification across cycle structures", starts here). **Sealed by the
commit that carries this file and the gate results; run after; scored to this text.** Kills have the
scopes stated in §7. The reading that opened the block is `2026-09-07_local_global_delta.md`
(exploring, unscored).

## §0 Postdiction disclosure

Seen before this seal:
- Hand computations of the twisted-cycle spectra at n = 4, 5, 6, 8, 12 (the conversation of
  2026-09-07): the twist selects the odd exponents; the twisted hexagon has spectrum ±√3, 0; the
  twisted square ±√2; twisted C₁₂ is strict over √2 and core over √3, √6.
- An independent review of the design (reasoning only, no numerics): Theorem 1 (iii)–(v), Theorem 2
  and the cyclomatic count of Theorem 3 below; the observation that every pure-cycle statement is a
  theorem and would be tautological as a test (STANDARDS §2.8); the `certificate.grade` odd-degree
  screen; the recommendation to register the first class where nothing forces the answer.
- M15 exp_05 K3's recorded rows (balanced cycles, m ≤ 30, k ∈ {2, 3, 4}) and M15's theorem C₆ = −I.
- Gates KA-1..8 (§2), run before this seal; every value is on the record in
  `results/explore_f0_gates_20260907_*.json`.

Not seen: any signed unicyclic graph's grade, strict count or field set; any T1–T3 quantity. Nothing
in §4 has been computed on any unicyclic graph other than the KA-7 counts and the KA-2/KA-8
instrument checks (one twisted unicyclic graph at n = 9 for switching invariance; the n = 8 class for
the charpoly cross-check — no grades were read).

## §1 Objects (closed at this seal; counting basis §6)

| | object | size |
|---|---|---|
| O1 | signed connected unicyclic graphs on n vertices, n ∈ [3, 14], **both switching classes**: *balanced* (the ordinary Cartan matrix) and *twisted* (one cycle edge negated — any; all choices are switching-equivalent, Theorem 1a) | 61,131 graphs (OEIS A001429, n = 3..14), 122,262 signed objects |
| O2 | the sub-range n ∈ [3, 10] for the seven-field test | 1,040 graphs, 2,080 signed objects |

**Enumeration.** `core/signed.py:unicyclic_graphs(n)`: every tree on n vertices plus one non-edge,
deduplicated by Weisfeiler–Lehman hash then exact isomorphism; the count is **asserted** against
A001429 at every n (KA-7 on the record for n ≤ 11; n = 12, 13, 14 assert at run time — an enumerator
that miscounts stops the run, it does not score).

**Frame.** The Cartan channel C = 2I − A_signed, exact over ℤ (`charpoly_exact`, DomainMatrix; KA-8).
Fields d ∈ {2, 3, 5, 6, 7, 13, 15}. Grade = exp_12 part-1's grade over ℚ(√d) — `'-'` (no factor
splits), `strict` (every ℚ-irreducible factor splits), `core` (non-splitting factors all to even
multiplicity), `partial` — computed per ℚ-irreducible factor (`grade_by_factor`; identical by unique
factorization, KA-8: 1,400 cells, 0 mismatches). "Pairs over ℚ(√d)" = grade ≠ `'-'`.
**Prefilter.** A rational eigenvalue of a monic integer polynomial is an integer, so "no rational
Cartan eigenvalue" is decided numerically (`eigvalsh`, 1e-9) before any exact work. **Norm screen.**
M18's proven necessary condition for strictness over √5 (`census.is_norm` at `SCREEN_POINTS`).

## §2 Instruments and gates (all PASS before this seal; `results/explore_f0_gates_20260907_*.json`)

| gate | what | value |
|---|---|---|
| KA-1 | balanced C_m reproduces M15 exp_05 K3 (`milestone15/results/exp_05_general_k_limit_20260717_150325.json`) | 38 rows: det H = +1, angles to 1e-8, reflection parity even — all reproduced. **Forward note to M15 K3:** the *raw* reflection count per row is not an invariant — eigh's column signs act on each transport's det and telescope only round the loop (Theorem 2a) — and differs from the 2026-07-17 record in 21 rows across numpy builds; K3's claim ("reflections telescope to an even count; det H = +1") is the parity statement and holds |
| KA-2 | switching invariance: exact charpoly and holonomy invariants under 20 random switchings (twisted C₁₂ k = 3, twisted C₉ k = 2, a twisted unicyclic graph at n = 9) | charpoly deviations 0; holonomy max deviation 4.8e-15 |
| KA-3 | charpoly(C₂ₙ^bal) = charpoly(Cₙ^bal)·charpoly(Cₙ^tw) exactly (n ≤ 15); det C_tw = 4 (n ≤ 30) | 0 failures |
| KA-4 | ones ∈ ker C_bal; λ_min(C_tw) = 4 sin²(π/2n), n ≤ 30 | 0.0; 1.8e-15 |
| KA-5 | the Möbius holonomy relation (Theorem 2): angles → π − θ, det → (−1)^k det, deficit² sum 4k; m ∈ [6, 30], k ∈ {2, 3, 4} minus (6, 4) | 74 cells, 0 failures, 0 degenerate; at (6, 2): deficit_bal = 2.828427 = 2√2 (H = −I, M15's theorem), **deficit_tw = 1.9e-15 (H = I)** |
| KA-6 | the grade table of the signed cycle, n ∈ [3, 30], both classes, seven fields, against Theorem 1 | 392 cells, **40 positive**, 0 off-theorem. Strict/core: n = 4, 8, 12, 16, 20, 24, 28 twisted strict over √2; twisted core over √3 (6, 12, 18, 24, 30), √5 (10, 20, 30), √6 (12, 24), √7 (14, 28), √13 (26), √15 (30). Partial: 15 balanced cells (cond(d) \| n) and 4 twisted odd-n cells (5, 15, 25 over √5; 13 over √13). Every other cell `'-'` |
| KA-7 | `unicyclic_graphs(n)` vs A001429, n ≤ 11 | 1, 2, 5, 13, 33, 89, 240, 657, 1806 — exact |
| KA-8 | `grade_by_factor` = exp_12 part-1's `grade` on all 200 trees n ≤ 10 × 7 fields; `charpoly_exact` = `Matrix.charpoly` on those trees and on both classes of all 89 unicyclic graphs at n = 8 | 1,400 cells, 0 mismatches; 0 charpoly mismatches |

Excluded from every gate and test: `certificate.grade`'s odd-degree screen (it returns `none` for a
whole polynomial when any ℚ-factor has odd degree > 1 — correct for a tree's golden pairing, wrong
as a general grade; balanced C₂₈ over √7 is a false negative); non-adjacent transport pairs; the
`theta_T` row-flip convention of M15 exp_05 (not equivariant under M ↦ −M).

## §3 Pre-seal numbers (all on the pure cycle; nothing below touches a unicyclic graph)

- The twist selects the odd exponents and doubles the conductor (KA-3, KA-6): balanced pairs over
  ℚ(√d) iff cond(d) | n; twisted iff cond(d) | 2n. For n odd the two classes coincide up to
  A ↦ −A (Theorem 1v): declared duplicate rows. d ∈ {5, 13} are untouched by the twist (odd conductor).
- The twisted cycle is strict over exactly one field, √2, exactly when 4 | n; the balanced cycle is
  never strict and never core (a simple zero mode).
- On M15's instrument, H_tw = ε·S·H_bal·S⁻¹ (KA-5): the balanced class reconciles every loop
  (det +1, K3); the twisted class reconciles in even k and fails to in odd k.

These are the bearings for §4; they are theorems, not evidence.

## §4 Tests (M = 3; scored per size and combined)

**T1 — the parity law on the first non-tree class.** A strict (over √5) signed unicyclic graph
exists only at n ≡ 0 (mod 4), in both classes. *Threshold:* 0 exceptions over n ∈ [3, 14].
*Informative if* ≥ 1 strict signed unicyclic graph exists at any n ≤ 14; otherwise recorded
vacuous, not passed. *Recorded, not scored:* strict counts per n per class — whether the twist
creates or destroys strictness.

**T2 — the twist never loses a field.** For every unicyclic graph at n ≤ 10, the set of d over
which the twisted class pairs contains the set over which the balanced class pairs. *Threshold:*
0 exceptions over 1,040 graphs × 7 fields. *Informative if* ≥ 1 graph pairs over ≥ 1 field in
either class (the cycles themselves do; this clause cannot fail to be met).

**T3 — the obstruction is a root.** At n ≡ 2 (mod 4), every signed unicyclic graph with golden
content over √5 has a rational (integer) Cartan eigenvalue — equivalently, no graph without an
integer eigenvalue pairs over √5 at n ∈ {6, 10, 14}. This is Panel G's odd-diagram mechanism
("the fixed point λ = 2 of the bipartite duality is a rational root") stated as a prediction for the
parity law. *Threshold:* 0 exceptions. *Informative if* ≥ 1 graph without an integer eigenvalue
exists at those n; otherwise vacuous.

**Predicted direction, stated before the run:** T1 pass; T2 pass (the cycle theorem as a bearing);
T3 an honest 50/50, predicted pass by the mechanism reading — an exception would say the parity
obstruction can be an irreducible rational quadratic rather than a root, which is a bearing for the
tree proof, not a loss.

## §5 What would count as vacuous

No strict signed unicyclic graph at any n ≤ 14 in either class (T1 uninformative: the parity law
would be untested here, not confirmed). No prefilter survivor at n ∈ {6, 10, 14} (T3 uninformative).
T2 cannot be vacuous. A gate that fails at run time (the A001429 assertion at n = 12–14) stops the
run; a partial run is scored per completed size and said to be partial.

## §6 Counting basis

Graphs up to isomorphism (A001429); a *signed object* is (graph, class); strict counts are per class
per n; T2 counts graphs, not fields; T3 counts signed objects without an integer eigenvalue.
Sizes may be run in separate invocations (n ≤ 12 in the foreground; n = 13–14 in the background with
`--nmin 13 --nmax-fields 0`); each invocation writes its own timestamped JSON; the outcomes journal
combines them per size and overall, naming every file.

## §7 Kill scope

- T1 exception: the parity law dies **as a graph-class law**; it stays a tree law (exhaustive to
  n = 24) and a theorem for construction parents. The milestone is untouched.
- T2 exception: "the twist doubles the conductor" dies beyond the pure cycle; Theorem 1 stands.
- T3 exception: the rational-root mechanism dies as *the* explanation of the parity law; Panel G's
  odd-diagram theorem stands (it is about diagrams, proved).
- No outcome here touches the kill sentence of M18, M15's standing kill-sentence, or any physics.

## §8 Outputs (append-only, timestamped)

`results/exp_20_signed_unicyclic_<ts>.json` and `_log.txt` per invocation; `results/explore_f0_gates_
<ts>.json` (this seal); `journals/2026-09-07_exp20_outcomes.md` citing the seal hash; README Block F
row; `formal/theorems/README.md` (Theorems 1–3); `formal/conjectures/m18_open.md` (the parity-law and
field-resonance rows).

---

## Theorems (proved here; filed the same day in `formal/theorems/README.md`)

**Theorem 1 (the signed cycle).** Let A_ε be an adjacency of the cycle C_n with edge signs of product
ε ∈ {±1}, and C = 2I − A_ε.
(a) *Switching.* D A D with D = diag(s), s ∈ {±1}^n, multiplies the sign of edge (i, i+1) by s_i s_{i+1};
the product round the cycle is invariant, and any two signings with the same product are
switching-equivalent (choose s along the path 0, 1, …, n−1 to clear every sign but the last, which
the product then fixes). So the spectrum depends only on ε, and there are exactly two classes.
(b) *Spectra.* Balanced: A ~ the unsigned cycle, eigenvalues 2cos(2πj/n). Twisted, with the sign on
edge (n−1, 0): v_l = ω^l with ω^n = −1, i.e. ω = e^{iπ(2j+1)/n}, satisfies (Av)_l = ω^{l−1} + ω^{l+1}
at interior l; at l = 0, −ω^{n−1} + ω = ω^{−1} + ω since ω^n = −1; at l = n−1, ω^{n−2} − 1 =
ω^{n−1}(ω^{−1} + ω) since ω^{1−n} = −ω. So spec(A_tw) = {2cos((2j+1)π/n) : j = 0..n−1}.
(c) *The double cover.* spec(C_{2n}^bal) = {2cos(πj/n) : j = 0..2n−1}; even j give the balanced C_n,
odd j the twisted C_n. Hence charpoly(C_{2n}^bal) = charpoly(C_n^bal)·charpoly(C_n^tw) (KA-3).
(d) *det C_tw = 4.* det C_tw = Π_j (2 − ω_j − ω_j^{−1}) = Π_j |1 − ω_j|² = |Π_j (1 − ω_j)|² = |1^n + 1|² = 4,
the ω_j being the roots of x^n + 1.
(e) *Fields.* 2cos((2j+1)π/n) = ζ_{2n}^{2j+1} + ζ_{2n}^{−(2j+1)} has minimal polynomial Ψ_m for
m = 2n/gcd(2j+1, 2n); with n = 2^v·n_odd these m are 2^{v+1}m′, m′ | n_odd, and m = 2n occurs (j = 0).
A real quadratic ℚ(√d) lies in ℚ(ζ_m) iff cond(d) | m (Kronecker–Weber; the quadratic subfields of
ℚ(ζ_m) are the ℚ(√d*) with fundamental discriminant d* | m). So the twisted cycle pairs over ℚ(√d)
iff cond(d) | 2n; the balanced cycle (m | n, m = n occurring) iff cond(d) | n.
(f) *Strictness.* Strict needs every factor to split, so cond(d) | 2^{v+1}m′ for every m′ | n_odd,
in particular (m′ = 1) cond(d) | 2^{v+1}: cond(d) is a power of 2, so d = 2 (cond 8) and v ≥ 2, i.e.
4 | n; conversely for d = 2 and 4 | n every m is divisible by 8 and every factor splits. No rational
eigenvalue occurs when 4 | n: 2cos((2j+1)π/n) ∈ {0, ±1, ±2} needs (2j+1)/n ∈ {½, ⅓, ⅔, 1}, which
forces n ≡ 2 (mod 4) or n odd. For even n every eigenvalue is doubled (j ↔ n−1−j), so non-splitting
factors are squares: grade *core* whenever there is golden content and d ≠ 2. For odd n the eigenvalue
−2 (j = (n−1)/2) is simple: *partial*. The balanced cycle has the simple eigenvalue 2 (j = 0; Cartan
0, the constant null vector): never strict, never core.
(g) *Odd n.* −A_bal has sign product (−1)^n = −1, so it lies in the twisted class: spec(A_tw) =
−spec(A_bal) and charpoly_tw(t) = charpoly_bal(4 − t); the two classes have identical fields and
grades cell by cell. ∎ (KA-6: 392 cells at the predicted grade.)

**Theorem 2 (the Möbius holonomy relation).** On C_m with M15's transport — k-frames from the top-k
eigenvectors of the vertex-deleted complement, T_{uv} = polar(V_vᵀV_u) over V∖{u, v} for adjacent u, v —
let H_ε be the holonomy of the class ε. Then **H_tw = ε·S·H_bal·S⁻¹** for a diagonal S ∈ {±1}^k.
*Proof.* (a) Switching by D: the complement of u switches by D|_{V∖u}, so V_u ↦ D_u V_u S_u where S_u
is eigh's column-sign gauge; on the shared support D_u and D_v restrict to the same diagonal, so
V_vᵀV_u ↦ S_v (V_vᵀ D D V_u) S_u = S_v M S_u; the polar factor is equivariant under orthogonal
conjugation, T ↦ S_v T S_u; round the loop the S's telescope, H ↦ S H S⁻¹ (KA-2). (b) The
complement of a cycle vertex is a signed *path*, which a switching D_u trivialises, unique up to a
global sign; normalise D_u(u+1) = +1, so D_u(w) is the product of the signs along the path from u+1
to w. On the shared support V∖{u, u+1} (one path from u+2 round to u−1) both D_u and D_{u+1}
trivialise the same signed path, hence D_u = c_{u,u+1}·D_{u+1} there, and evaluating at u+2 gives
c_{u,u+1} = σ(u+1, u+2). So M^{tw} = c_{u,u+1}·S_{u+1} M^{bal} S_u; polar(cM) = c·polar(M) for c = ±1
(the polar factor M(MᵀM)^{−½} is unique for nonsingular M, and (−M)ᵀ(−M) = MᵀM); therefore
T^{tw}_{u,u+1} = σ(u+1, u+2)·S_{u+1} T^{bal}_{u,u+1} S_u and, round the loop, H_tw = [Π_u σ(u+1, u+2)]·
S H_bal S⁻¹ = ε·S H_bal S⁻¹. Requires nonsingular M and simple top-k complement eigenvalues (a path's
spectrum is simple; M15's eigengap guard applies). ∎ *Corollaries.* eig(H_tw) = −eig(H_bal), so the
rotation angles go θ ↦ π − θ, det H_tw = (−1)^k det H_bal, and ‖H_tw − I‖² + ‖H_bal − I‖² = 4k (H
orthogonal). At (m, k) = (6, 2), H_bal = −I (M15's theorem C₆ = −I), so **H_tw = I** (KA-5:
deficit 1.9e-15). Per-edge transport dets are gauge-dependent; their parity round the loop is not.

**Theorem 3 (the cover doubles the cyclomatic number).** The branched double cover of a diagram
with V nodes and E edges (r17: two lifts of every ordinary edge, three of the bond) has 2V vertices
and 2(E − 1) + 3 = 2E + 1 edges, so its cyclomatic number is (2E + 1) − 2V + 1 = 2(E − V + 1): twice
the diagram's. A construction parent is a tree iff its diagram is; no non-tree diagram, twisted or
not, has a tree parent. The third species is not reached by this route. ∎

---

**Forward note (2026-09-07, before any result was read).** Layer: mathematics. T1–T3 feed
`formal/conjectures/m18_open.md` (the parity-law row and the field-resonance row) and, if T1 or T3
holds at every size, the tree proof gets a proved sibling on the first non-tree class. Theorems 1–3
go to `formal/theorems/README.md` today. Nothing here is physics: the standing kill-sentence —
*if holonomy is dynamically inert, it is mathematics, not physics* — is untouched, and the engine's
Möbius is reported on, not changed.
