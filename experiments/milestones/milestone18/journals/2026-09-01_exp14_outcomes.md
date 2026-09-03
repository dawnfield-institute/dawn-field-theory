# exp_14 outcomes — Phase 5: the off-core instrument at n = 16 (registration be3f7044, SEALED)

**Verdict: 3/4 scored tests** (T2, T3, T5 pass; T1 fails as sealed; T4 was recorded-only).
Objects: the 80 partnered core-grade Galois folds at n = 16. The off-core instrument reached
every one of them — zero declarations, against 76 of 80 declared under Phase 4's conic recipe.

| Test | Result | Count |
|---|---|---|
| Lemma-1 identity (theorem; consistency) | holds | **80/80** |
| T1 Clean ⟺ (B-invariant ∨ uniform mass) | **FAIL** | sufficiency holds 24/24; necessity fails — 28 clean folds, 4 outside the selector |
| T2 selector ⇒ trace law | **PASS** | 24/24 |
| T3 selector ⇒ structure (some-partner halves) | **PASS** | 24/24 |
| T5 layer order (Clean ⇒ trace ∧ structure) | **PASS** | 28/28 |

## T1 — the third road

The selector was exact at n ≤ 14 (18 = 16 ∪ 2 of 61, both directions). At n = 16 the
sufficiency direction survives untouched, but four folds are Clean without being B-invariant or
uniform-mass — and they share a single signature: **core (t−2)⁴ with kernel masses {2/5, 3/5}**
(complementary values, m and 1 − m) on all four (dets 16, 16, 464, −176). So the clean regime
has at least three entrances, and the two-condition characterization was an n ≤ 14 accident.
Scored FAIL to the sealed text. Open, stated precisely: characterize the kernel geometries for
which the off-core laws hold; the data now spans B-invariant (any masses), uniform mass (any
B-behaviour), and complementary-pair masses {m, 1−m} with m = 2/5 — whether the third class is
{m, 1−m} in general or something wider is undetermined at these sizes.

## What passed, and what it means

Every fold the selector does cover obeys the trace law and the full sign-split structure with
some-partner halves (T2, T3), and no fold anywhere is vertex/leak-clean while breaking trace or
structure (T5) — the peeling order is now confirmed out of sample. Lemma 1 (proved in the proof
notes the same day, before this run) checked 80/80 as a consistency gate. The off-core
formulation is established as the right instrument: the conic recipe's 95% declaration rate at
n = 16 drops to zero.

## Registration lesson

6. An exact biconditional found on ≤ 61 objects earned only its sufficiency direction; sealing
   necessity too was a bet, and it lost to the first larger sample. Sufficiency and necessity
   deserve separate clauses with separate evidence bars.
