#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 11: Quantum Mechanics from Graph Structure
================================================================================

Runs the Milestone 14 + P13-P16 experiments that back the paper.

Usage:
    python reproduce.py            # run all experiments
    python reproduce.py 07         # run exp_07 only  (also: p13, p14, ...)
    python reproduce.py --list     # list experiments and their paper sections

Reproducibility model (repo-relative)
-------------------------------------
This package ships inside the dawn-field-theory repository. The experiment
scripts import the shared DFT core chain
    quantum_complement -> identity_complement -> connection_geometry -> quantum_gravity -> foundations
which lives at experiments/milestones/milestone*/core. reproduce.py locates
that directory automatically and places the home-milestone core on PYTHONPATH,
so the verbatim scripts run unmodified.

A fresh run regenerates timestamped JSON into the *source* milestone results
directory (the experiments' native output location). The frozen snapshot used
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
SOURCE_MILESTONES = ["milestone14"]

# exp id -> (script filename, paper section(s))
EXPERIMENTS = {
    "01":  ("exp_01_orbit_hilbert_space.py",              "Section 2   (orbit Hilbert space, Gram = I)"),
    "02":  ("exp_02_permutation_rep_decomposition.py",    "Section 2   (orbit count = trivial-irrep multiplicity)"),
    "03":  ("exp_03_born_rule_from_orbit_measure.py",     "Section 3   (Born rule from orbit measure; T3 pre-registered fail)"),
    "04":  ("exp_04_measurement_as_gauge_fixing.py",      "Section 4   (measurement as gauge fixing)"),
    "05":  ("exp_05_sec_complexification_interference.py","Section 5,7 (SEC complexification enables interference)"),
    "06":  ("exp_06_graph_double_slit.py",                "Section 12.2 (graph double-slit; structural failure 1/4)"),
    "07":  ("exp_07_noncommuting_observables_d4.py",      "Section 5   (D_4 is the only non-abelian ADE; NC=1.2247)"),
    "08":  ("exp_08_robertson_uncertainty.py",            "Section 5   (Robertson uncertainty bound > 0 only for D_4)"),
    "09":  ("exp_09_entanglement_product_graphs.py",      "Section 6   (entanglement from Cartesian product graphs)"),
    "10":  ("exp_10_cross_milestone_compatibility.py",    "Section 12  (zero contradictions with M1-M13)"),
    "11":  ("exp_11_m14_synthesis.py",                    "Section 14,15 (M14 synthesis)"),
    "p13": ("exp_p13_bell_violation_topology.py",         "Section 7   (CHSH = 2sqrt2 requires nontrivial Aut)"),
    "p14": ("exp_p14_orbit_dynamics.py",                  "Section 8   (Schrodinger from orbit Laplacian)"),
    "p15": ("exp_p15_path_integral_zeno.py",              "Section 9,10 (path integral, Zeno / anti-Zeno)"),
    "p16": ("exp_p16_decoherence_orbit_env.py",           "Section 11  (decoherence, einselection)"),
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
        print("PACSeries Paper 11 experiments:")
        for key, (script, desc) in EXPERIMENTS.items():
            print(f"  {key:>3}  {script:<42} {desc}")
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
        keys = [k.lstrip("0") or "0" for k in sys.argv[1:]]
        # normalize: allow "7" -> "07", "p13" -> "p13"
        wanted = []
        for raw in sys.argv[1:]:
            k = raw if raw in EXPERIMENTS else raw.zfill(2)
            if k in EXPERIMENTS:
                wanted.append(k)
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
