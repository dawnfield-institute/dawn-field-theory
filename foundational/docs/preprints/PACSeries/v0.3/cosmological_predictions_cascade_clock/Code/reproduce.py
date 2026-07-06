#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 9: Cosmological Predictions and the Cascade Clock
======================================================================================

Runs the Milestone 8 (BSM predictions) + Milestone 9 (infodynamic mechanism /
cascade clock) experiments that back the paper.

Usage:
    python reproduce.py                # run all experiments
    python reproduce.py m8_07          # run one experiment by id
    python reproduce.py --list         # list experiments and their paper sections

Reproducibility model (repo-relative)
-------------------------------------
This package ships inside the dawn-field-theory repository. The experiment
scripts import their milestone's shared DFT core as a package:
    milestone8 scripts:  from core.bsm import ...
    milestone9 scripts:  from core.infodynamics import ...   (which itself pulls
                         milestone8/core/bsm in place)
Both milestones ship a `core` package, so the two cannot share a single
PYTHONPATH entry (the `core` name would collide). reproduce.py therefore places
the *owning* milestone directory (the parent of its core/) on PYTHONPATH per
script:
    foundational/experiments/milestone8   -> resolves core.bsm
    foundational/experiments/milestone9   -> resolves core.infodynamics
so the verbatim scripts run unmodified.

A fresh run regenerates timestamped JSON into this package's Code/results/
directory (the copied scripts' native output location). The frozen snapshot used
to write this paper is in ../Data/results/. Compare the two to verify.

Requires: numpy, scipy, matplotlib  (see requirements.txt)
Provenance for every script: see trace.yaml
"""

import sys
import os
import subprocess
from pathlib import Path

# Milestone core dir(s) the scripts import from. Each script is mapped to its
# owning milestone below; that milestone dir goes on PYTHONPATH for that run.
SOURCE_MILESTONES = ["milestone8", "milestone9"]

# exp id -> (milestone, script filename, paper section(s))
EXPERIMENTS = {
    # ---- Milestone 8: BSM predictions & observational contact ----
    "m8_01": ("milestone8", "exp_01_depth73_coupling_derivation.py",   "Section 6   (depth-73 dark matter coupling)"),
    "m8_02": ("milestone8", "exp_02_dark_matter_mass_spectrum.py",     "Section 6   (dark matter mass 6.44 keV, two routes)"),
    "m8_03": ("milestone8", "exp_03_relic_abundance_production.py",    "Section 6.3 (relic abundance, Dodelson-Widrow)"),
    "m8_04": ("milestone8", "exp_04_zprime_395_quantification.py",     "Section 7   (Z' at 395 GeV, not excluded)"),
    "m8_05": ("milestone8", "exp_05_neutrino_absolute_masses.py",      "Section 11  (neutrino masses, hierarchy, CP)"),
    "m8_06": ("milestone8", "exp_06_fibonacci_depth_sweep.py",         "Section 6.1, 11.2 (Fibonacci/cyclotomic depths)"),
    "m8_07": ("milestone8", "exp_07_hubble_tension_quantification.py", "Section 4   (Hubble ratio phi^{1/6})"),
    "m8_08": ("milestone8", "exp_08_cosmological_constant_precision.py","Section 3   (cosmological constant -122.09)"),
    "m8_09": ("milestone8", "exp_09_jwst_structure_prediction.py",     "Section 12  (JWST high-z structure floor)"),
    "m8_10": ("milestone8", "exp_10_bsm_master_test.py",               "Section 12.2, 13 (BSM master test / registry)"),
    "m8_11": ("milestone8", "exp_11_cross_consistency.py",             "Section 12.3 (cross-consistency; N-range fail)"),
    "m8_12": ("milestone8", "exp_12_look_elsewhere.py",                "Section 4.3 (look-elsewhere; phi^{1/6} rank 2)"),
    # ---- Milestone 9: the infodynamic mechanism / cascade clock ----
    "m9_01": ("milestone9", "exp_01_phi_timing_from_pac.py",           "Section 8.4, 8.5 (phi timing, self-similarity)"),
    "m9_02": ("milestone9", "exp_02_xi_transition_cost.py",            "Section 8.3 (Xi = gamma + ln phi, unique cost)"),
    "m9_03": ("milestone9", "exp_03_slope_correction.py",             "Section 10.1 (slope gap 8.9%, Monte Carlo)"),
    "m9_04": ("milestone9", "exp_04_sec_temporal_flow.py",            "Section 8   (SEC temporal flow, arrow)"),
    "m9_05": ("milestone9", "exp_05_gravity_time_from_cascade.py",    "Section 8   (gravity-time from cascade)"),
    "m9_06": ("milestone9", "exp_06_arrow_of_time.py",                "Section 8   (arrow of time, entropy production)"),
    "m9_07": ("milestone9", "exp_07_s8_redshift_evolution.py",        "Section 5   (S8 tension 3.22->0.07 sigma)"),
    "m9_08": ("milestone9", "exp_08_h0_scale_dependence.py",          "Section 4.4, 8.3 (H0 scale dependence)"),
    "m9_09": ("milestone9", "exp_09_dark_energy_evolution.py",        "Section 3.3, 10.2 (dark energy w(z); DESI wa)"),
    "m9_10": ("milestone9", "exp_10_m9_synthesis.py",                "Section 9   (M9 synthesis; 2->1 parameters)"),
}


def find_experiments_root(start: Path):
    for anc in start.resolve().parents:
        cand = anc / "foundational" / "experiments"
        if all((cand / m / "core").is_dir() for m in SOURCE_MILESTONES):
            return cand
    return None


def main():
    here = Path(__file__).parent
    exp_dir = here / "experiments"

    if "--list" in sys.argv:
        print("PACSeries Paper 9 experiments:")
        for key, (mile, script, desc) in EXPERIMENTS.items():
            print(f"  {key:>6}  {script:<42} {desc}")
        return

    exproot = find_experiments_root(here)
    if exproot is None:
        print("ERROR: could not locate foundational/experiments/"
              f"{{{','.join(SOURCE_MILESTONES)}}}/core.\n"
              "This package must be run from inside the dawn-field-theory repo.")
        sys.exit(1)

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
        mile, script, desc = EXPERIMENTS[key]
        path = exp_dir / script
        env = os.environ.copy()
        # Per-script PYTHONPATH: only the owning milestone dir, so `core`
        # resolves to that milestone's core package (they collide otherwise).
        env["PYTHONPATH"] = os.pathsep.join(
            [str(exproot / mile), env.get("PYTHONPATH", "")]
        )
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
