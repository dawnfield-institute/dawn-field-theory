# explore_r11 — the core is inert (exploration, not registered)

**Written 2026-09-01, after exp_12 was scored (commit 4ab2ff35) and while exp_13's census stage
was still running — before any n = 16 core fold had been evaluated.** Script
`explore_r11_core_anatomy.py`, results `explore_r11_core_anatomy.json`. Objects: the 12
registered-domain core-grade folds of exp_12 T5 (10 ≤ n ≤ 14). P_off is the off-core Bezout
projector (rational core removed and projected out), R_off = P_off − σ(P_off) — gauge-free by
construction, no conic, no choice.

## Findings (12/12 unless stated)

1. **The structure law is an off-core law.** Under the sign split of R_off's diagonal: copy side
   connected, cut exactly 2 edges, conjugate side exactly two components whose sizes equal the
   diagram's halves at its 5-bond — **12/12, including det −464**, the one exception under the
   full-R split (exp_12 journal, r7b). With the seven strict folds (where R_off = R): 19/19.
2. **The trace law is an off-core law.** tr(R_off·D) = 2/√5 on 12/12. The golden core line
   contributes nothing to the trace — that is *why* the trace was gauge-independent on this domain.
3. **The leakage law is an off-core law.** (I−P)BP splits into four mutually orthogonal blocks by
   left/right factor. The off→off block ‖σ(P_off)·B·P_off‖² = **2/5 on 12/12** — also on dets −44
   and −284, whose total leak is 28/45. Their excess is exactly 1/9 + 1/9, sitting symmetrically
   in the two mixed blocks (off→core and core-conjugate→off, equal by σ-symmetry); the
   core→core block is 0 on 12/12.
4. **Half-Binet weights.** |R_off,vv| takes exactly two values per fold: 1/√5 off the core support
   and a reduced value on it — 1/(2√5) on 10/12, and 2/(3√5) precisely on the two folds (−44,
   −284) whose mixed core blocks leak. The reduced weight and the core leak are the same
   phenomenon seen twice; what selects 1/2 vs 2/3 is open.

## Reading (labelled as such)

The core is inert: it carries no trace, no structure, and no off-sector leakage. All three fold
laws — trace, leakage, two-component-with-halves — are statements about the off-core reflection,
and the exp_12 T5 leakage "failure" was a bookkeeping artifact of including the core in P. The
core's only degrees of freedom are the gauge direction (exp_06's conic) and the two symmetric
mixed leak blocks; when those blocks are nonzero the core also drags the off-core diagonal from
1/(2√5) to 2/(3√5) on its support.

## Prediction on record before exp_13's stage 2 (not sealed; stated for honesty)

exp_13's sealed T5b tests the component relations under the sign split of the **full** R (the
seal predates r11). If any n = 16 registered-domain core fold has core-scrambled vertex signs (as
det −464 does at 14), sealed T5b can fail there while the R_off form of the law holds. exp_13 is
scored to its seal regardless; this note exists so that outcome, if it occurs, is explained by a
dated prior analysis and not by a post-hoc one.

## explore_r12 — the modulated vertex law (12/12)

The half-Binet observation resolves into an identity. Since P_off·σ(P_off) = 0 and
P_off + σ(P_off) = I − Qc, the off-core reflection satisfies R_off² = I − Qc; and at every vertex
of all 12 registered-domain core folds:

**√5 · R_off,vv = ±(1 − Qc_vv)**

— the Binet weight scaled by the off-core mass at that vertex (`explore_r12_modulated_vertex_law.py`).
The strict vertex law is the Qc = 0 case, so one proof will cover both. The core mass is uniform
per fold: **1/2 on ten folds** — the core kernel is a leaf-difference pair, precisely the seal's
"core sits on leaf-difference vectors" — and **1/3 on exactly the two leaking folds** (dets −44,
−284), whose kernel vectors spread over three vertices. Reading (labelled): the leak selector is
the core's kernel geometry — a leaf-pair core is fully inert; a wider core stays trace-inert and
structure-inert but opens the two symmetric mixed leak blocks.
