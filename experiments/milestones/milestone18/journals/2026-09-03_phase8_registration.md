# Phase 8 registration — the census at n = 22 and n = 24: rigidity, the denominator, the certificate, the species

**Layer: mathematics (→ `formal/`).** Every test below tests a row of
[`formal/conjectures/m18_open.md`](../../../../formal/conjectures/m18_open.md) or the boundary of an
indexed theorem. Nothing physical is claimed. **Sealed by the commit that carries this file; run
after; scored to this text.** Kills have the scopes stated in §6.

## §0 Postdiction disclosure
Rigidity and the third species were sealed in Phase 6 (n = 20) and are here tested at their first
new sizes. The 5-exclusion (T2), the exact-denominator relation (T3) and the certificate reading
(T4) were **observed at n ≤ 20 in r21** (2026-09-02) and are here tested out of sample for the
first time; they are predictions from this seal onward and postdictions before it.

## §1 Objects and instruments
- **Objects.** All trees on n = 22 (2,144,505) and n = 24 (14,828,074) vertices, Cartan channel
  C = 2I − A. Strict trees = those with p = q·σ(q), q ∈ ℤ[φ][t] with no rational factor.
- **Census instrument.** `core/census.py: strict_hunt_parallel` — the proven-necessary norm screen at
  the six sealed points (0, ±1, 2, 3, −2) followed by exact factorization over ℚ(√5). Validated through
  the sealed known-answer gate at n = 16 before this seal: **PASS (15 trees, 14 polynomials), the same
  15 edge sets as the committed census, 9.2 s on 8 workers** (`results/explore_r23_parallel_census_n16_*.json`).
- **Partner instrument.** `core/census.py: one5_partners_fast` — one-5 tree diagrams on k = 11 and
  k = 12 nodes by the integer edge expansion q = (q_del − q_cut) − φ·q_cut, every placement kept.
  Validated before this seal against the symbolic map at k ≤ 8 (**identical 156 keys, 156/156 q up to
  conjugation**) and against the sealed count of 610 targets at k = 10 (**610**); it yields 1,576 targets at
  k = 11 and 4,074 at k = 12 (`results/explore_r23b_partner_map_gate_*.json`, gate PASS).
- **Battery.** `scripts/exp_19_phase8_census.py`: Phase 6's T3–T6 battery verbatim in content, with the
  r22 repairs (every cospectral placement, every isomorphism, no `nsimplify`), plus the three new
  tests. Its own known-answer run at n = 16 (before the seal) must reproduce r21/r22: 15 strict
  trees, 14 polynomials, 14 parents all passing the battery, 1 sector-strict non-parent, all fold-half
  denominators 1, 0 integral halves on the non-parent. **Done before the seal: 15/14, T1 14/14 with 0
  failures, T2 0/14 exceptions, T3 uninformative (all Res = 1, as known), T4 0 false positives with
  sensitivity 14/14, T5 0 asymmetric (as known at 16), T6 0 orphans at k = 8; 0 errors**
  (`results/exp_19_phase8_n16_*.json`).
- **Counting basis.** Per tree; per-polynomial tallies reported alongside (66 trees / 64 polynomials
  at 20). Degenerate partners (q not coprime to σq) are declared, never scored.
- **Frame.** All quantities are invariant under vertex relabelling. b is the unique minimal-degree
  Bézout representative; q₀, q₁ are the ℤ[φ]-coordinates of q (q = q₀ − φ·q₁); Res(q₀, q₁) is an
  integer. A fold half is the half equal (up to conjugation) to a partner diagram's q.

## §2 Pre-seal numbers (in hand from n ≤ 20; nothing below was computed on 22 or 24)
| quantity | value at n ≤ 20 |
|---|---|
| strict trees / polynomials at 20 | 66 / 64 |
| construction parents · degenerate partners · unpartnered (sector-strict + asymmetric) | 47 · 6 · 13 (9 + 4) |
| fold halves of parents, n ≤ 20 | 68: Res(q₀, q₁) = ±1 → den 1 (63); 4 → 1 (2); 9 → 3 (3) |
| fold halves with 5 \| Res(q₀, q₁) | 0 / 68 |
| non-fold scorable halves with 5 \| Res(q₀, q₁) | 50 / 140 (36 %) — the null for T2 |
| parents with an integral Galois half · non-parents with one | 44 / 47 · 0 / 19 — the null for T4 |
| asymmetric strict trees (third species) | 0 at 12, 0 at 16, 4 at 20 (3 polynomials) |
| expected runtime | census ≈ 8 min at 22, ≈ 1 h at 24 (8 workers, 1.7 ms/tree); battery minutes |

## §3 Tests (M = 6; each scored at n = 22 and at n = 24 separately, reported as N/M per size and combined)

**T1 — Rigidity.** Every partnered strict tree with a non-degenerate fold half passes the full
matching battery: P² = P, P + σ(P) = I, rank n/2; √5·R = S + 2Π with SΠ = −ΠS; the Π-quotient is
isomorphic to a partner diagram with the realized 5-bond the unique multiplicity-3 edge; exactly
one copy-internal defect edge, over that edge; tr(R·D) = 2/√5, ‖(I−P)BP‖² = 2/5, |R_vv| = 1/√5.
*Threshold:* 100 % of evaluable trees. *Informative if* ≥ 1 evaluable tree. *Kill:* one failure is a
partnered strict tree that is not a construction parent — the rigidity row dies.

**T2 — The ramified boundary.** On every fold half of a construction parent, 5 ∤ Res(q₀, q₁).
*Null:* 36 % of non-fold halves at n ≤ 20 have 5 | Res(q₀, q₁). *Threshold:* 0 exceptions.
*Informative if* ≥ 7 fold halves (0.64⁷ < 0.05); otherwise recorded uninformative, not passed.

**T3 — The exact denominator (relation).** On every fold half: den(5·b) equals the odd part of
rad(Res(q₀, q₁)) — 2 never divides den(5·b), every odd prime dividing Res(q₀, q₁) divides den(5·b),
and no other prime does. *Threshold:* 0 exceptions. *Informative if* ≥ 3 fold halves across 22 ∪ 24
have |Res(q₀, q₁)| ≠ 1 (5 of 47 at 20); otherwise recorded uninformative. The multiset of
Res(q₀, q₁) values is reported regardless.

**T4 — The certificate has no false positives.** No strict non-parent (unpartnered or
degenerate-partner) has any Galois half with 5·b ∈ ℤ[t]. *Null:* 0/19 at 20 against 44/47 among
parents. *Threshold:* 0 false positives. *Informative if* ≥ 5 non-parent trees. Sensitivity
(parents with an integral half) is reported, not scored.

**T5 — The third species persists.** At n = 22 and at n = 24 there exists at least one strict
tree with trivial automorphism group and no one-5 partner. *Threshold:* ≥ 1 at each size. *Fail
is informative:* it would make the species an n = 20 accident and retire the classification's
third term.

**T6 — Zero orphans at k = 12.** Every one-5 diagram on 12 nodes whose q is coprime to σ(q) has a
strict parent at n = 24 in the census (the construction theorem's existence half against the
exhaustive census; k = 11 is odd and its parents are core-grade, outside the screen — declared out
of scope). *Threshold:* 0 orphans. *On failure:* suspect the instrument first (census or partner
map), then the theorem; no theorem claim moves until the instrument is re-checked.

## §4 What would count as vacuous
Zero strict trees at a size; T2/T3 below their informative counts (recorded uninformative, not
passed); T4 with fewer than 5 non-parents.

## §5 Outputs (append-only, timestamped)
`results/exp_19_phase8_n22_<ts>.json`, `results/exp_19_phase8_n24_<ts>.json` (census, strict lists,
per-tree battery, denominators, halves, species flags, orphan list, tests block); logs alongside.
Outcomes journal after both sizes: `journals/2026-09-0X_exp19_outcomes.md`, citing this seal's hash.

## §6 Kill scope
T1 kills rigidity (the row), not the construction theorem. T2/T3 kill the exact-denominator row
as stated; the denominator bound (theorem) stands. T4 kills the certificate reading. T5's failure
retires "third species" as a persistent class. T6's failure is an instrument alarm. None of the
six touches the milestone thesis or its kill-sentence; this is the mathematics layer.

---

**Forward note (2026-09-03, during the n = 22 run, before any result was read).** The tree counts in
§1 are wrong by one index of OEIS A000055: 2,144,505 is the count at n = 21 and 14,828,074 the count
at n = 23. Verified by direct enumeration on this machine: n = 20 → 823,065 (the sealed Phase 6
count), n = 21 → 2,144,505, n = 22 → **5,623,756**; hence n = 24 → **39,299,897**. No test, threshold
or null in §3 depends on the count — the objects are all trees on n vertices and the census
enumerates them all, reconciling the exact count from the enumeration itself — but the §2 runtime
estimates were 2.6× too small (≈ 80 min at 22, ≈ 13 h at 24 on eight cores). The sealed text stands
unedited; this note corrects it. Registration lesson 10: a count quoted in a seal is verified by
enumeration on the day, not recalled.
