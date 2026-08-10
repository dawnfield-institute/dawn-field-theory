#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 8: Quantum Gravity from Information Conservation
=====================================================================================

Runs the Milestone 11 experiments that back the paper.

Usage:
    python reproduce.py            # run all experiments
    python reproduce.py 05         # run exp_05 only
    python reproduce.py --list     # list experiments and their paper sections

Reproducibility model (repo-relative)
-------------------------------------
This package ships inside the dawn-field-theory repository. The experiment
scripts import the shared DFT core chain
    quantum_gravity -> foundations -> infodynamics -> bsm
which lives at experiments/milestones/milestone{11,10,9,8}/core. reproduce.py
locates milestone11/core automatically and places it on PYTHONPATH; the rest of
the chain is resolved by the core modules themselves, so the verbatim scripts
run unmodified.

A NumPy-2.0 compatibility shim (Code/_compat/sitecustomize.py) is also placed on
PYTHONPATH: it restores the ``np.trapz`` alias that NumPy 2.0 renamed to
``np.trapezoid`` (identical numerics), which exp_04, exp_07 and the shared core
still call. It changes no result.

Output location quirk
---------------------
The Milestone 11 scripts call save_results(results, RESULTS_DIR, name) with the
directory and name arguments transposed relative to the helper's signature
(save_results(results_dict, experiment_name, results_dir)). Because
experiment_name receives an absolute Path, the join collapses to it and a fresh
run writes its JSON as
    <scripts' parent>/results_<TIMESTAMP>.json
named results_*.json (not labelled by experiment, not under a results/ subdir),
and also leaves two empty stub dirs (results/ and a name-shaped dir). When the
verbatim copies here are run via reproduce.py, "<scripts' parent>" is this Code/
directory; when the originals are run in place it is the milestone11 root. Either
way the frozen, experiment-labelled snapshots used to write this paper are in
../Data/results/. Compare the two to verify (the stubs can be deleted).

Requires: numpy, scipy, matplotlib  (see requirements.txt)
Provenance for every script: see trace.yaml
"""

import sys
import os
import subprocess
from pathlib import Path

# Milestone core dir(s) the scripts import from (added to PYTHONPATH).
# The rest of the core chain (milestone10/9/8) is resolved by the core modules.
SOURCE_MILESTONES = ["milestone11"]

# exp id -> (script filename, paper section(s))
EXPERIMENTS = {
    "01": ("exp_01_response_time_hierarchy.py",     "Section 3.3 (response-time hierarchy; phi^-183 vs alpha_grav 0.04%)"),
    "02": ("exp_02_planck_from_negotiation.py",     "Section 4   (four Planck-scale routes; inner convergence 1.13x, span 4x)"),
    "03": ("exp_03_discrete_cascade_time.py",       "Section 4   (discrete cascade tick T_MVAE=0.7213 t_P; echo error 1e19 at n=100)"),
    "04": ("exp_04_singularity_saturation.py",      "Section 5   (cascade saturation; Kretschmann finite; area law slope 2.000)"),
    "05": ("exp_05_hawking_from_pac.py",            "Section 6   (Hawking T*M = 1/(8pi); micro-BH cascade correction)"),
    "06": ("exp_06_page_curve_unitarity.py",        "Section 7   (Page curve peak at k/N=0.5; epsilon-PAC breaks return-to-zero)"),
    "07": ("exp_07_cascade_density_quantization.py","Section 8   (Fibonacci spectrum, 1/k^2 propagator, finite loops)"),
    "08": ("exp_08_graviton_from_cascade.py",       "Section 8   (graviton spin-2 99.5%, massless, 2 polarizations)"),
    "09": ("exp_09_stochastic_irreversibility.py",  "Section 9   (arrow of time; phi-selection + gamma; multi-ratio Landauer)"),
    "10": ("exp_10_desi_subleading.py",             "Section 11.2 (QG correction ~1e-60 at observable z; S8 stable)"),
    "11": ("exp_11_planck_star_bounce.py",          "Section 10  (Planck-star bounce ~1 t_P; burst E ~ (M/M_P)^-1/3)"),
    "12": ("exp_12_observational_contact.py",       "Section 11  (GW dispersion 67 orders below GW170817; M_min = phi^2 M_P)"),
    "13": ("exp_13_m11_synthesis.py",               "Section 12-16 (M11 synthesis; 0 contradictions, 12 predictions)"),
}

COMPAT_DIR = Path(__file__).resolve().parent / "_compat"


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
        print("PACSeries Paper 8 experiments:")
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
    # _compat first so the sitecustomize shim loads at interpreter startup.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(COMPAT_DIR)] + core_paths + [env.get("PYTHONPATH", "")]
    )

    if len(sys.argv) > 1:
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
