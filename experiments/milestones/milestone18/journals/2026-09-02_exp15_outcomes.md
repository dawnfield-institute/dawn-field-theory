# exp_15 outcomes — Phase 6: the matching structure at n = 20 (seal 16b98837)

**Verdict: 5/6** — T1, T3, T4, T5, T6 pass; T2 fails as sealed and names a third strict species.
Census: 823,065 trees → 14,111 norm-screen survivors → **66 strict trees on 64 polynomials**
(two cospectral pairs). Instrument history: the first run stalled in a quadratic symbolic loop
(fixed, checkpointed); the second crashed on a partner whose q shares roots with σ(q) (declared
path added); the third run scored. Known-answer gate (14 polynomials / 15 trees at n = 16)
passed before the seal.

| Test | Result | Count |
|---|---|---|
| T1 one-5 conjecture at k = 10 | **PASS** | 610 diagram targets, zero orphans (53 strict-parented, 557 core-parented) |
| T2 strict ⇒ construction fold or sector-strict | **FAIL** | 62/66; four strict trees are neither |
| T3 matching form √5·R = S + 2Π, SΠ = −ΠS | **PASS** | **47/47** evaluable; 6 degenerate partners declared |
| T4 Π-quotient ≅ a one-5 partner, multiplicities [2,…,2,3], mult-3 = 5-bond | **PASS** | 47/47 |
| T5 single defect, copy-internal, over the mult-3 edge | **PASS** | 47/47 |
| T6 trace 2/√5, leak 2/5, vertex law | **PASS** | 47/47 (consistency, as sealed) |

## Rigidity holds — the last conjecture survives its test

Every evaluable partnered strict fold at n = 20 has the matching form with all its consequences:
47 objects that did not exist in any record when the form was found on 7. With the construction
theorem (r17: construction ⇒ form ⇒ laws) and r15 Addendum 5 (form ⇒ construction's lift
pattern), the strict-fold theory is closed on the evaluable set at n ≤ 20. T1's zero orphans is
now also what the theorem requires; the search confirms the proof.

## Declared: six degenerate partners

Six strict trees are partnered by one-5 diagrams whose q shares roots with σ(q) — repeated golden
factors — so the sealed Bezout P is undefined and T3–T6 do not apply; three of the six are
sector-strict, three are not. Their anatomy (orbit counts, whether they are constructions of a
different placement) is being probed (r18) and will be recorded as an addendum.

## T2 — the third species: asymmetric strict trees

Four strict trees (dets −31, −155, and a cospectral pair at 169) have **trivial automorphism
group and no one-5 partner**. They are strict — p = q·σ(q) with no rational factor — but q is
not the Gram polynomial of any one-5 tree diagram, and with no symmetry there are no sectors to
fold by. Neither construction parents nor sector-strict: a species the classification did not
contain. Scored FAIL to the sealed text. Identification in progress (r18: are they parents of
diagrams with **two** golden bonds, or of non-tree diagrams?); addendum to follow. Registration
lesson 9: a classification sealed as exhaustive must name what it would call a residue *before*
the run — T2 did ("neither"), which is why this is a discovery and not an instrument fault.

## Standing

Strict trees at n ≤ 20 = construction parents (rigid, all laws theorems) ∪ sector-strict trees ∪
asymmetric strict trees (new), plus the degenerate-partner overlap awaiting anatomy. Kill scope
honoured: nothing here touches the milestone kill sentence; T2's failure retires "exhaustive at
20" for the two-species classification and replaces it with a three-species one whose third
member is not yet understood.
