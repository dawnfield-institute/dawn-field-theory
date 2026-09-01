# 2026-09-01: exp_11 outcomes — 4/4 at n = 14

Registration: 5bc8faff (sealed before any computation at n = 14). Exhaustive over all 3,159
non-isomorphic trees on 14 vertices; the instrument passed its known-answer self-tests
(E₈ ↔ [3,3,5], cat8 ↔ [3,5,3], D₆ core) before enumerating.

| Test | Prediction | Result |
|---|---|---|
| T1 | every 7-node tree-shaped Coxeter diagram of 3s with exactly one 5-bond has a 14-vertex tree parent | **PASS** — all diagrams parented, zero orphans |
| T2 | no 7-node linear diagram with two 5s has a tree parent | **PASS** — 0 of 5 |
| T3 | every parent is core-grade (odd k ⇒ λ = 2 a rational root of q) | **PASS** |
| T4 | no strict √5-golden tree on 14 vertices exists | **PASS** — 0 strict among 3,159 (50 core, 153 partial) |

The one-5 rule now holds at k = 2..7 (parents on 4..14 vertices) with no orphan at any size;
the odd-k parity theorem holds at its third odd size; and "strict ⇒ Galois fold, with even k"
survives an exhaustive test three times larger than any before. Block C-ext: exp_11 4/4.
