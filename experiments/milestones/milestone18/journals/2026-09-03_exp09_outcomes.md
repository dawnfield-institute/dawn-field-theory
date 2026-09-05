# exp_09 outcomes — Block D, physical reach of the fold certificate (2026-09-03)

**Registration:** [`2026-09-03_blockD_registration.md`](2026-09-03_blockD_registration.md), sealed by commit
`0130ebe0` (2026-09-03). **Run:** `scripts/exp_09_block_d_reach.py` on the devbox (8 cores), 369 s;
`results/exp_09_block_d_reach_20260903_235832.json` (+ `_log.txt`), append-only. Mode: **predicting** — every
threshold below is the sealed one; no value in §1–§3 was seen before the seal except the H₂ cells at
d ≤ 5 disclosed in the registration's §0.

## Scorecard: 2/3 scored (T1 fail · T2b pass · T4 pass) · T2a gate held · T3 recorded

**Declared defect in the sealed text.** §4's header says *M = 4* while its body carries thresholds
for three tests only (T1, T2b, T4; T2a and T3 are explicitly "not scored"). The header count is the
error — scoring against a fourth test that has no threshold would be inventing one — so the block
scores **2/3**, and the milestone scorecard moves 49/64 → **51/67**. Recorded, not repaired
(sealed text is never edited).

## §1 Objects as run

|E| = 173 live objects, exactly the registration's count: O1 = 3 complete PAC trees (d = 6, 7, 8;
n = 127, 255, 511; sector route), O2 = 166 replayed growth trees (n = 6–100, median 54; 141
isomorphism classes under the WL-6 hash), O3 = 4 unicyclic controls (n = 7, 9, 9, 9). O4 unreachable
(declared at the seal). Grades on O2 ∪ O3: 161 partial, 9 core (the cores are all n ≤ 12), 0 none, 0 blind,
0 degenerate (declared) halves; 179 Galois halves evaluated in all. No worker errors.

## §2 Tests

**T1 — reach: FAIL (0 carriers / 173).** No live object is a carrier on any half: no O1 sector cell,
none of the 179 O2/O3 halves has (tr(R·D), leak_oo) = (2/√5, 2/5). The sealed expected direction was
0; the null gave 0/473. This is the clean negative the registration named: *the fold is not physically
reached by the corpus's own generative operators at these sizes* (PAC trees to n = 511, growth trees
to n = 100, unicyclics to n = 9).

**T2a — gate: HELD.** The H₃-type sector (t² − 4t − 1 ± √5, class H3 up to bond scaling) first
appears in O1 at d = 8 and at no d ≤ 7; d = 6 and 7 carry the H₂-type sector only.

**T2b — class without representative: PASS (0 carriers / 4 cells).** Sector-route certificate values,
(tr(R·D), leak_oo), on the pair's two members (signs flip with the member; leaks agree):

| cell | tr(R·D) | leak_oo | leak_total |
|---|---|---|---|
| d = 6, H₂ | ∓8√5/5 | 16/5 | 16/5 |
| d = 7, H₂ | ∓16√5/5 | 32/5 | 32/5 |
| d = 8, H₂ | ∓67√5/10 | 53/4 | 107/8 ∓ √5/40 |
| d = 8, H₃ | ∓3√5/10 | 9/100 | 87/200 ∓ 23√5/200 |
| d = 8, combined halves (4) | ∓7√5, ∓32√5/5 | 1339/100, 336/25 | — |

With the disclosed d = 3, 4, 5 cells (∓3√5/5, 1; ∓2√5/5, 4/5; ∓4√5/5, 8/5) the H₂ leak doubles
with each depth from d = 4 to d = 7 (4/5, 8/5, 16/5, 32/5) and the trace with it — the sector's mass
grows with the tree, it never settles at the fold's 2/5. The H₃ cell at d = 8 is small (leak 9/100) and
equally not the certificate. M15's shape holds: the class is present in the PAC family, its fold
representative is not.

**T3 — null: recorded** (registration §3: 0/473).

**T4 — P1 forward-correction: PASS.** (i) The instrument grades E₈ strict with p = q·σ(q) and the H₄
half a carrier (KA1 rerun in this run: true). (ii) Block A's `exp_02_projection_carries_phi.py`, rerun
as a subprocess in this run: exit 0, `SCORE 4 /4`, shell ratio 1.6180339887498951 (φ to 2.2×10⁻¹⁶),
both shells H₄; its full output is stored in the results file under `T4.exp02_tail`.
*Instrument defect, declared:* the harness's flag `T4.ii_exp02_rerun` is **false** in the results file
because it searched the output for the literal token `PASS`, which exp_02 never prints (it prints
`"pass": true` per test and `SCORE 4 /4`). The sealed criterion is "passes its own gate", which it does
4/4; the parenthetical operationalization was mis-specified by me and never exercised before the seal.
The flag stays as written (append-only); this paragraph is the reading. **Lesson 12:** a subprocess
gate's verdict string is checked against the target's actual output *before* the seal.
*Side effect, repaired:* exp_02 writes a fixed-name result (`exp_02_projection_20260831.json`, pre
append-only), so the rerun overwrote Block A's stored result on the devbox checkout; that copy was
reset from git and never left the devbox. The committed file is untouched.

## §3 Readings (exploring, not registered — a bearing, not a verdict)

- **Half-certificates.** 12 growth trees (n = 31–62; 11 from `exp_07` d = 5, mc = 2, one from the
  exp_08 grid) and one unicyclic (n = 7) carry the H₂-type pair with
  (tr(R·D), leak_oo) = (−2√5/5, 4/5) **exactly** — the trace half of the certificate with the leak
  doubled, and the same value the complete PAC tree shows at d = 4. Two more unicyclics split the other
  way: one with leak_oo = 2/5 but tr = 0 (n = 9), one with tr = 2/√5 but leak 3/5 (n = 9). No object has
  both. Whether (2/√5, 4/5) is a value class of the H₂-type sector in its own right (two sheets, no
  branch) is a question for a Block D2, not a finding here.
- **The H₂-type sector recurs in the growth ensemble.** 13 of the O2 halves (13 trees of 166) are
  exactly t² − 4t + 1 ± √5; the growth process keeps producing the A₄-fold's sector without ever
  producing its fold. Degrees of the halves: {1: 67, 2: 87, 3: 8, 4: 13, 5: 2, 6: 2}.
- **Off→off is the whole leak on most objects:** leak_oo = leak_total on 57 of 179 halves.

## §4 What dies, what survives (kill scope §7)

- **Retired:** "the fold is physically reached by the corpus's generative objects at these sizes" and
  "the H₃ class has a fold representative in the PAC family (d ≤ 8)". P1's ratio form retires with
  them (T4: its content lives in the Ledger theorem and exp_02).
- **Stands:** every theorem in `formal/`, every scored block, the milestone's kill sentence unchanged
  (Block D can neither kill nor rescue it). No Block D2 is opened — no carrier appeared.
- **Sizes not reached:** PAC trees d ≥ 9, growth trees n > 100 (topology beyond the replayed seeds),
  engine/percolation graphs (frame undeclared, excluded at the seal).
