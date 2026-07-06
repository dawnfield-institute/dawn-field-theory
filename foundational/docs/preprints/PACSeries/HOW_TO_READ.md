# How to Read the PACSeries

The PACSeries is an evolving research program, not a finished theory. This note states plainly how the work is built and how to weigh its claims — a frame we would rather make explicit than have you infer.

## The stance

"Is it perfect?" is the wrong question to ask of any physical framework. Every theory in the history of physics has been a successively-less-wrong model; Newton was not wrong, he was less close. The PACSeries is built to *move* — to be sharpened by scrutiny, to log its own corrections, and to converge rather than to arrive. We do not claim to be right. We try to be less wrong than the current picture, and to make every step checkable so you can judge for yourself.

## What that means in practice

1. **Results are tiered by derivation type.** Every quantitative claim is labeled:
   - **Type A (structural)** — follows from the axioms through theorems, with at most one empirical step. It would produce the same result without knowing the target value.
   - **Type B (identified)** — a clean expression matches a known value after a few empirical identifications.
   - **Type C (pattern-matched)** — found by searching against known values, and acknowledged as such.

   The tier matters more than any single number. The companion `derivation_classification.md` gives the full chain for each result.

2. **Failures are reported, not hidden.** Every paper has a "what fails" section. One pre-registered prediction has already been falsified against data (Paper 12, against ~443,000 quasar absorption systems). A framework that can die, and says so, is behaving like physics.

3. **Corrections are logged.** When a result turns out to be a curve-fitting artifact or a tautological test, it is recorded in the Epistemic Corrections Registry (`EPISTEMIC_CORRECTIONS_REGISTRY.md`, repository root) as a *collapse event* — the mechanism by which the framework improves, not something to bury.

4. **Everything is reproducible.** Each paper ships the code, data, and figure-generation that produced its numbers, traced to source (`Code/`, `Data/`, and `Code/trace.yaml`). You do not have to trust the prose; you can run it.

## How to weigh a given claim

Check its tier first. A Type-A result is a theorem plus a stated postulate — to reject it you must reject the axioms or find the error in the proof. A Type-C result is a pattern whose weight lies in the *joint* structure across independent domains, not in any individual match; read it with the look-elsewhere effect in mind. Read the precision tables and the failure sections together — neither is complete without the other.

---

*The series is released under this frame deliberately. To declare it finished would stop the motion; to keep it open, tiered, and falsifiable is to let it improve. See also `state_of_the_pac_series.md` for the series-wide classification of every result.*
