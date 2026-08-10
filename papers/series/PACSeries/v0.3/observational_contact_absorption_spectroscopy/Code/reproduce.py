"""
reproduce.py — Reproduce all results in PACSeries Paper 12

Observational Contact: Cascade Clock Signatures in Quasar Absorption Spectroscopy

Prerequisites:
  pip install astropy scipy numpy

Data (download before running):
  1. SDSS DR16 MgII: https://wwwmpa.mpa-garching.mpg.de/SDSS/MgII/
     -> fits_files/SDSS_DR16_QSO_based_MgII_Absorber_Catalog.fits
  2. SDSS DR16 FeII: same URL
     -> fits_files/FeII_based_SDSS_DR16_MgII_Absorber_Catalog.fits
  3. SDSS DR12 CIV: https://zenodo.org/records/7872725
     -> Table95All_Pciv_positive_all.dat
  4. XQR-30: https://github.com/XQR-30/Metal-catalogue
     -> Merge all AbsorberCatalogs/*/..._absorber_catalog.csv

Place data in ../Data/catalogs/

Usage:
  python reproduce.py              # Run all experiments
  python reproduce.py exp_12       # Run specific experiment
  python reproduce.py --list       # List available experiments
"""

import sys
import os
from pathlib import Path

EXPERIMENTS = {
    'exp_03': ('exp_03_photon_archaeology.py', 'Section 3,5: Alpha invariance, SEC line widths'),
    'exp_05': ('exp_05_bifractal_mesh_signal.py', 'Section 7.1: SDSS MgII cascade signal'),
    'exp_07': ('exp_07_deep_cascade_probe.py', 'Section 7.1: Z-detrending test'),
    'exp_08': ('exp_08_cascade_panel.py', 'Section 7.1: Z-trend-immune tests'),
    'exp_09': ('exp_09_discovery_panel.py', 'Section 3.4: Discovery panel'),
    'exp_11': ('exp_11_deep_targets.py', 'Section 7.1-7.2: CIV detrend, spatial dipole'),
    'exp_12': ('exp_12_smooth_cascade.py', 'Section 3.2-3.3: Smooth cascade reframing (KEY)'),
    'exp_13': ('exp_13_faster_time.py', 'Section 4-6: Faster time signatures (KEY)'),
    'exp_16': ('exp_16_structural_regularity.py', 'Section 8.1: Topology as regulator (KEY)'),
}

def main():
    if '--list' in sys.argv:
        print("Available experiments:")
        for key, (script, desc) in EXPERIMENTS.items():
            print(f"  {key}: {desc}")
        return

    to_run = sys.argv[1:] if len(sys.argv) > 1 else EXPERIMENTS.keys()

    for key in to_run:
        if key not in EXPERIMENTS:
            print(f"Unknown experiment: {key}")
            continue
        script, desc = EXPERIMENTS[key]
        print(f"\n{'='*60}")
        print(f"Running {key}: {desc}")
        print(f"{'='*60}")
        script_path = Path(__file__).parent / "experiments" / script
        os.system(f'{sys.executable} "{script_path}"')


if __name__ == '__main__':
    main()
