#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 10: Connection, Identity, and Spacetime
=============================================================================

Runs the Milestone 12 (13 experiments) + Milestone 13 / 13.5 (17 experiments)
scripts that back the paper.

Usage:
    python reproduce.py               # run all experiments
    python reproduce.py 12_03         # run milestone12 exp_03 only
    python reproduce.py 13_09         # run milestone13 exp_09 only
    python reproduce.py --list        # list experiments and their paper sections

Reproducibility model (repo-relative)
-------------------------------------
This package ships inside the dawn-field-theory repository. The experiment
scripts import the shared DFT core chain
    identity_complement -> connection_geometry -> quantum_gravity -> foundations
(milestones 13 -> 12 -> 11 -> 10). reproduce.py locates
experiments/milestones automatically and places milestone12/core and
milestone13/core on PYTHONPATH; the core modules resolve the rest of the chain
themselves (each core adds its parent milestone's core), so the verbatim
scripts run unmodified.

Because milestone12 and milestone13 share some experiment filenames
(exp_12_cross_milestone_compatibility.py) and both number 01..13, the packaged
scripts are kept in per-milestone subdirectories:
    experiments/milestone12/...   experiments/milestone13/...

A fresh run regenerates timestamped JSON into the *source* milestone results
directories (the experiments' native output location). The frozen snapshot used
to write this paper is in ../Data/results/. Compare the two to verify.

Requires: numpy, scipy, matplotlib  (see requirements.txt)
Provenance for every script: see trace.yaml
"""

import sys
import os
import subprocess
from pathlib import Path

# Milestone core dir(s) the scripts import from (added to PYTHONPATH).
# The rest of the core chain is resolved by the core modules themselves.
SOURCE_MILESTONES = ["milestone12", "milestone13"]

# exp id -> (script path relative to experiments/, paper section(s))
EXPERIMENTS = {
    # ---- Part I: Connection as Primitive (Milestone 12) ----
    "12_01": ("milestone12/exp_01_pac_is_ade.py",                    "Section 2   (connection = addition; PAC recursion IS ADE, spectral radius = phi)"),
    "12_02": ("milestone12/exp_02_self_loop_minimal_connection.py",  "Section 2,8 (self-loop = minimal connection; phi from self-application)"),
    "12_03": ("milestone12/exp_03_gauge_groups_from_ade.py",         "Section 2.2 (SU(2),SU(3) uniquely Fibonacci; F_7 = 13 = 1+3+8+1 gauge closure)"),
    "12_04": ("milestone12/exp_04_branch_foreclosure.py",            "Section 3   (branch foreclosure; PAC conserved; rate-density fail T4)"),
    "12_05": ("milestone12/exp_05_entropy_redistribution_rate.py",   "Section 3.2 (entropy = redistribution rate; Shannon = SEC; info-Fiedler fail T3)"),
    "12_06": ("milestone12/exp_06_basin_dynamics.py",                "Section 3   (laws as basin attractors; basin-depth coupling fail T2)"),
    "12_07": ("milestone12/exp_07_relaxation_time_taxonomy.py",      "Section 3.1 (force hierarchy from basin geometry; phi^(d2-d1) ratios)"),
    "12_08": ("milestone12/exp_08_crystallizing_law_signatures.py",  "Section 3.2 (crystallizing-law signatures; variance narrowing vs drift)"),
    "12_09": ("milestone12/exp_09_alpha_indices_as_ade_positions.py","Section 3   (alpha indices as ADE cascade positions)"),
    "12_10": ("milestone12/exp_10_sec_complexification.py",          "Section 4   (SEC complexification: A_1 su(2) -> sl(2,C), compactness broken)"),
    "12_11": ("milestone12/exp_11_lorentz_from_ade.py",              "Section 4.1 (Lorentz from ADE; 15 commutators close to 1.1e-16; Killing (3,3))"),
    "12_12": ("milestone12/exp_12_cross_milestone_compatibility.py", "Section 8   (zero contradictions with M1-M11)"),
    "12_13": ("milestone12/exp_13_m12_synthesis.py",                 "Section 8,10 (M12 synthesis; 8 predictions registered)"),
    # ---- Part II: Identity as Complement (Milestone 13 / 13.5) ----
    "13_01": ("milestone13/exp_01_complement_determines_identity.py","Section 5.1 (complement spectrum uniquely identifies vertex)"),
    "13_02": ("milestone13/exp_02_definitional_parallax.py",         "Section 5.2 (definitional parallax scales with observer distance)"),
    "13_03": ("milestone13/exp_03_complement_ade_structure.py",      "Section 5   (complement preserves ADE structure; edge conservation)"),
    "13_04": ("milestone13/exp_04_complement_transformation_group.py","Section 6.1 (complement-transformations form the Weyl group)"),
    "13_05": ("milestone13/exp_05_complement_algebra_embeds_lie.py", "Section 9.2 (complement -> Lie embedding; 3 honest failures, PSD not PD; 1/4)"),
    "13_06": ("milestone13/exp_06_a1_weyl_to_lorentz.py",            "Section 6.2 (A_1 Weyl Z_2 -> SL(2,C) via SEC; adversarial control)"),
    "13_07": ("milestone13/exp_07_complement_lorentz_structure.py",  "Section 6.2 (so(3,1) commutation; ONLY A_1 gives Lorentz; Thomas rotation)"),
    "13_08": ("milestone13/exp_08_speed_of_light_coherence.py",      "Section 7.1 (speed of light as coherence limit; rates in [0.59,2.00]; T3 fail)"),
    "13_09": ("milestone13/exp_09_invariant_interval.py",            "Section 6.3 (invariant interval; Killing (3,3)->(1,3); Minkowski unique)"),
    "13_10": ("milestone13/exp_10_proper_time_deformation.py",       "Section 7.2 (proper time d(tau)=dt/cosh(eta); graph twin paradox)"),
    "13_11": ("milestone13/exp_11_curvature_from_density.py",        "Section 10  (curvature from connection-density gradients; geodesic bending)"),
    "13_12": ("milestone13/exp_12_cross_milestone_compatibility.py", "Section 8   (zero contradictions with M1-M12)"),
    "13_13": ("milestone13/exp_13_m13_synthesis.py",                 "Section 8,10 (M13 synthesis; 12-link chain; 8 predictions)"),
    "13_14": ("milestone13/exp_14_complement_lie_projection.py",     "Section 9.2 (M13.5 investigation; Gram PSD not PD is fundamental; T3 fail)"),
    "13_15": ("milestone13/exp_15_coherence_limit_universality.py",  "Section 9.2 (M13.5 investigation; coherence limit NOT universal; 0/4)"),
    "13_16": ("milestone13/exp_16_alternative_complement_metrics.py","Section 9.2 (M13.5 investigation; all metrics PSD not PD; 0/4)"),
    "13_17": ("milestone13/exp_17_random_graph_paradox.py",          "Section 9.2 (M13.5 investigation; random graphs vs ADE; 2/4)"),
}


def find_experiments_root(start: Path):
    marker = SOURCE_MILESTONES[0]
    for anc in start.resolve().parents:
        cand = anc / "theory" / "experiments"
        if (cand / marker / "core").is_dir():
            return cand
    return None


def main():
    here = Path(__file__).parent
    exp_dir = here / "experiments"

    if "--list" in sys.argv:
        print("PACSeries Paper 10 experiments:")
        for key, (script, desc) in EXPERIMENTS.items():
            print(f"  {key:>6}  {script:<52} {desc}")
        return

    exproot = find_experiments_root(here)
    if exproot is None:
        print("ERROR: could not locate experiments/milestones/"
              f"{SOURCE_MILESTONES[0]}/core.\n"
              "This package must be run from inside the dawn-field-theory repo.")
        sys.exit(1)

    env = os.environ.copy()
    core_paths = [str(exproot / m / "core") for m in SOURCE_MILESTONES]
    env["PYTHONPATH"] = os.pathsep.join(core_paths + [env.get("PYTHONPATH", "")])

    if len(sys.argv) > 1:
        wanted = []
        for raw in sys.argv[1:]:
            if raw in EXPERIMENTS:
                wanted.append(raw)
            else:
                print(f"Unknown experiment: {raw}")
    else:
        wanted = list(EXPERIMENTS.keys())

    failures = []
    for key in wanted:
        script, desc = EXPERIMENTS[key]
        path = exp_dir / script
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
