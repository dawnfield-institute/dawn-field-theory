# Frame artifacts: the fourth thing that looks like a finding

## Summary

Adds STANDARDS.md Section 2.9, naming a failure mode the existing Section 2.8 triad
(recursive / tautological / circular) does not catch: a **frame artifact**, where the probe
and the observable sit at different scopes and the difference between frames is read as the
thing being studied.

The distinguishing property is that it defeats replication. Frame artifacts are directional
and perfectly reproducible, so bootstraps, multi-seed runs and independent reruns confirm
them. Only re-running with matched scopes kills one. Every check we currently apply for
tautology and circularity passes a frame artifact cleanly.

## Changes

### Added
- `STANDARDS.md` Section 2.9 — definition, the discriminating test alongside the existing
  three, two worked instances from this corpus, and practical consequences.
- `STANDARDS.md` Section 2.7 item 6 — the frame must be declared at pre-registration.
- `STANDARDS.md` Section 2.7 item 7 (added 2026-09-02) — a registration names the layer it
  feeds (conjecture test → `formal/`, physical reach → `theory/`, instrument validation →
  `core/`), and a proved result is indexed in `formal/theorems/` in the commit that proves it.
  Motivated by Milestone 18, which produced seven theorems inside experiment journals with no
  index entry, and by `formal/README.md`'s own account of how the formal layer lapsed.

## Details

Two instances, both from 2026-08, both of which passed the Section 2.8 tests:

- **`sec_prime_manifold`'s 1/phi threshold.** `frac_E_positive` is measured on the odd
  sublattice while its expectation `S_hat` is a moving window over all integers, with 2 in
  the factor base. Odds are never divisible by 2, so `I = S_hat - S` is positive on odds by
  construction. At factor-base size 9 the fraction is 0.6187, 0.04% from 1/phi. Controls run
  against the study's own `core/sec_core.py`: removing 2 from the basis gives 0.505;
  including the evens gives 0.501; sampling `n = 2 mod 3` gives 0.661 ~ 2/3. The residue
  class chooses the constant. `SYNTHESIS.md` already records the mechanism ("2 must be in
  factor base -- creates the bias that enables asymmetry") without drawing the consequence.
- **ADE Laplacian redistribution.** Potential started on vertex 0, entropy measured over the
  whole graph; "E-series redistributes faster" held across three ranks. Vertex 0 is an
  endpoint in A_n and a branch node in E_n. Orbit-averaged, the ordering did not survive.

Note the asymmetry with Section 2.8's cases: those were caught by inspecting the procedure.
Neither of these could be. Both needed a control that changed the measurement geometry while
holding the physics fixed.

Section 2.9 does not declare any new required `meta.yaml` field, per Section 5.3's rule that
nothing is called required unless a validator enforces it.

## Related
- `STANDARDS.md` Sections 2.7, 2.8, 2.8.1, 5.3
- `.github/instructions/experiment-schema.instructions.md` (workspace root, not in this repo)
- Follow-ups not in this change: annotating `sec_prime_manifold`'s README/SYNTHESIS with the
  control result, and updating the `golden-ratio-primes` Lore node.
