#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 7: The Symmetry Primitive and Scoped Mediation
====================================================================================

Runs the Milestone 7 (symmetry primitive) + Milestone 6 (scoped mediation)
experiments that back the paper.

Usage:
    python reproduce.py               # run all experiments
    python reproduce.py m7_01         # run one experiment
    python reproduce.py m6_09 m7_04   # run several
    python reproduce.py --list        # list experiments and their paper sections

Reproducibility model (repo-relative)
-------------------------------------
This package ships inside the dawn-field-theory repository. The experiment
scripts are verbatim copies of the milestone6 + milestone7 experiments and
import their shared DFT core in place:

  * Milestone 7 scripts import ``core.symmetry`` (constants, save_results),
    which lives at experiments/milestones/milestone9/core.
  * Milestone 6 scripts import ``scope`` / ``core.scope`` (transfer-matrix
    infrastructure) from papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/core, and
    ``_shared`` (confluent-identity baseline loaders) from
    experiments/milestones/confluent_identity/scripts.

Because M6 and M7 both expose a package literally named ``core`` (core.scope
vs core.symmetry), the two cannot share one PYTHONPATH without colliding.
reproduce.py therefore sets a *per-experiment* PYTHONPATH from the experiment's
home milestone, so each verbatim script runs unmodified. reproduce.py locates
experiments/milestones/ automatically.

A fresh run regenerates timestamped JSON into the experiments' native results
directories. The frozen snapshot used to write this paper is in
../Data/results/. Compare the two to verify.

Requires: numpy, scipy, matplotlib  (see requirements.txt)
Provenance for every script: see trace.yaml
"""

import sys
import os
import subprocess
from pathlib import Path

# Home milestones whose core dirs the scripts import from.
SOURCE_MILESTONES = ["milestone6", "milestone7"]

# exp id -> (script filename, home milestone, paper section(s))
EXPERIMENTS = {
    # ---- Milestone 7: The Symmetry Primitive (Part I) ----
    "m7_01": ("exp_01_self_reference_generates_recursion.py", "milestone7", "Section 2   (phi from cross-scale relational self-reference)"),
    "m7_02": ("exp_02_nothing_instability.py",                "milestone7", "Section 3   ('nothing' unstable under multi-scale drive)"),
    "m7_03": ("exp_03_xi_from_symmetric_restoration.py",      "milestone7", "Section 4   (Xi = gamma + ln(phi) per boundary)"),
    "m7_04": ("exp_04_inv_phi_attenuation.py",                "milestone7", "Section 5   (1/phi attenuation emergent, R^2=0.995)"),
    "m7_05": ("exp_05_global_local_asymmetry.py",             "milestone7", "Section 6   (global symmetry requires local asymmetry)"),
    "m7_06": ("exp_06_symmetry_breaking_as_seeking.py",       "milestone7", "Section 6   (symmetry breaking = seeking; 3/4)"),
    "m7_07": ("exp_07_ade_closure_termination.py",            "milestone7", "Section 7   (ADE closure at D=3; L4 diverges)"),
    "m7_08": ("exp_08_rbf_from_symmetry.py",                  "milestone7", "Section 14.1 (RBF memory damping fails; 2/4)"),
    "m7_09": ("exp_09_compatibility_scorecard.py",            "milestone7", "Section 8   (100% compatibility with M1-M6)"),
    "m7_10": ("exp_10_predictions_from_primitive.py",         "milestone7", "Section 15  (predictions from the primitive)"),
    # ---- Milestone 6: Scoped Mediation (Part II) ----
    "m6_01": ("exp_01_scope_boundary_transfer_matrix.py",     "milestone6", "Section 9   (transfer matrices; rank-1 67/67; 3/4)"),
    "m6_02": ("exp_02_ade_scope_identification.py",           "milestone6", "Section 9   (ADE scope; KAN transition rho=1.0)"),
    "m6_03": ("exp_03_tetration_penalty_derivation.py",       "milestone6", "Section 9.3 (tetration penalty; R^2=0.67 FAIL; 2/4)"),
    "m6_04": ("exp_04_coupling_from_scope_depth.py",          "milestone6", "Section 10  (coupling from depth; EM 5.7 ppm)"),
    "m6_05": ("exp_05_dark_sector_depth_73.py",               "milestone6", "Section 13  (dark sector depth 73; 3/4)"),
    "m6_06": ("exp_06_neutrino_masses_from_scope.py",         "milestone6", "Section 12.3 (neutrino masses; splitting 44%; 3/4)"),
    "m6_07": ("exp_07_xi_as_scope_fixed_point.py",            "milestone6", "Section 12.2 (Xi fixed point; Euler gap 0.09%; 3/4)"),
    "m6_08": ("exp_08_pac_conservation_across_scopes.py",     "milestone6", "Section 11  (PAC conservation 3.47e-18)"),
    "m6_09": ("exp_09_alpha_em_as_survival_ratio.py",         "milestone6", "Section 10.2 (alpha_EM #1 of 10,440; 5.7 ppm)"),
    "m6_10": ("exp_10_scoped_mediation_master_test.py",       "milestone6", "Section 12  (master test; 0 contradictions)"),
    # ---- Milestone 6: post-hoc analyses (M6 failures explained via M7) ----
    "m6_al": ("analysis_local_asymmetry_as_mechanism.py",     "milestone6", "Section 6   (local asymmetry as mechanism)"),
    "m6_xi": ("analysis_xi_multiplicative_aggregation.py",    "milestone6", "Section 11  (Xi multiplicative aggregation)"),
}


def find_experiments_root(start: Path):
    for anc in start.resolve().parents:
        cand = anc / "theory" / "experiments"
        if (cand / "milestone7" / "core").is_dir() and (cand / "milestone6" / "core").is_dir():
            return cand
    return None


def pythonpath_for(milestone: str, exproot: Path):
    """The core dir(s) a script from `milestone` needs on PYTHONPATH.

    M6 and M7 both ship a package called `core`, so the paths are built
    per-milestone (never merged) to avoid a `core` namespace collision.
    """
    if milestone == "milestone7":
        # M7 scripts do `from core.symmetry import ...`  -> milestone7 root
        return [exproot / "milestone7"]
    if milestone == "milestone6":
        # exp_01 does `from scope import ...`      -> milestone6/core
        # others do `from core.scope import ...`   -> milestone6 root
        # several do `from _shared import ...`      -> confluent_identity/scripts
        return [
            exproot / "milestone6",
            exproot / "milestone6" / "core",
            exproot / "confluent_identity" / "scripts",
        ]
    raise ValueError(f"unknown milestone {milestone}")


def main():
    here = Path(__file__).parent
    exp_dir = here / "experiments"

    if "--list" in sys.argv:
        print("PACSeries Paper 7 experiments:")
        for key, (script, milestone, desc) in EXPERIMENTS.items():
            print(f"  {key:>6}  [{milestone}]  {script:<48} {desc}")
        return

    exproot = find_experiments_root(here)
    if exproot is None:
        print("ERROR: could not locate experiments/milestones/"
              "{milestone6,milestone7}/core.\n"
              "This package must be run from inside the dawn-field-theory repo.")
        sys.exit(1)

    if len(sys.argv) > 1:
        wanted = []
        for raw in sys.argv[1:]:
            if raw in EXPERIMENTS:
                wanted.append(raw)
            else:
                print(f"Unknown experiment: {raw}")
        if not wanted:
            sys.exit(1)
    else:
        wanted = list(EXPERIMENTS.keys())

    failures = []
    for key in wanted:
        script, milestone, desc = EXPERIMENTS[key]
        path = exp_dir / script
        env = os.environ.copy()
        core_paths = [str(p) for p in pythonpath_for(milestone, exproot)]
        env["PYTHONPATH"] = os.pathsep.join(core_paths + [env.get("PYTHONPATH", "")])
        print(f"\n{'='*70}\nRunning {key}: {desc}\n{'='*70}")
        r = subprocess.run([sys.executable, str(path)], env=env)
        if r.returncode != 0:
            print(f"[WARNING] {script} exited with code {r.returncode}")
            failures.append(key)

    print(f"\n{'='*70}")
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"All {len(wanted)} experiment(s) ran successfully.")


if __name__ == "__main__":
    main()
