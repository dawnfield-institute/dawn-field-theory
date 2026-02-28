#!/usr/bin/env python3
"""
Experiment 03: Deep Dive Analysis
=================================

Purpose:
    Deep investigation into:
    1. Optimal w/R for E/B = φ
    2. Charge localization and phase singularities
    3. Power law refinement
    4. Golden ratio geometry relationships

Author: Peter Lorne Groom, Claude (Anthropic)
Date: February 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import MobiusField, SECOperator, EMProjector
from core.constants import PHI, PHI_INV, XI


def investigate_optimal_wR():
    """Fine-grained search for optimal w/R."""
    
    print("\n" + "=" * 60)
    print("Investigation 1: Fine-Grained w/R Search")
    print("=" * 60)
    
    R = 2.0
    # Fine sweep around expected optimal
    w_values = np.linspace(0.4, 0.8, 21)
    
    proj = EMProjector(n=20, L=3.0)
    results = []
    
    print(f"\n{'w/R':<10} {'E/B':<12} {'φ-dev%':<10}")
    print("-" * 35)
    
    for w in w_values:
        field = MobiusField(n_u=48, n_v=24, R=R, w=w)
        sec = SECOperator(damping=0.98, pi_coupling=0.05)
        
        for _ in range(150):
            sec.step(field)
        
        em = proj.project(field)
        phi_dev = abs(em['EB_ratio'] - PHI) / PHI * 100
        
        results.append({
            'w_R': w/R,
            'EB': em['EB_ratio'],
            'phi_dev': phi_dev
        })
        
        print(f"{w/R:<10.4f} {em['EB_ratio']:<12.4f} {phi_dev:<10.2f}")
    
    best = min(results, key=lambda x: x['phi_dev'])
    print(f"\nBest: w/R = {best['w_R']:.4f}, E/B = {best['EB']:.4f}, dev = {best['phi_dev']:.2f}%")
    
    return results, best


def investigate_charge():
    """Analyze charge distribution and singularities."""
    
    print("\n" + "=" * 60)
    print("Investigation 2: Charge Localization")
    print("=" * 60)
    
    # Use geometry near optimal
    field = MobiusField(n_u=64, n_v=32, R=2.0, w=0.55)
    sec = SECOperator(damping=0.98, pi_coupling=0.05)
    proj = EMProjector(n=28, L=3.5)
    
    # Evolve
    for _ in range(200):
        sec.step(field)
    
    # Project
    em = proj.project(field)
    
    # Analyze charge
    charge = em['charge_density']
    mask = proj.mask
    
    charge_abs = np.abs(charge[mask])
    high_threshold = charge_abs.mean() + 2 * charge_abs.std()
    
    print(f"\nE/B ratio: {em['EB_ratio']:.4f}")
    print(f"Charge total (positive): {em['charge_total_pos']:.2f}")
    print(f"Charge total (negative): {em['charge_total_neg']:.2f}")
    print(f"Net charge: {em['charge_net']:.2f}")
    
    # Find high-charge locations
    high_charge = np.abs(charge) > high_threshold
    if high_charge.any():
        x_hc = proj.X[high_charge & mask]
        y_hc = proj.Y[high_charge & mask]
        z_hc = proj.Z[high_charge & mask]
        r_hc = np.sqrt(x_hc**2 + y_hc**2 + z_hc**2)
        
        print(f"\nHigh-charge region:")
        print(f"  Points: {len(r_hc)}")
        print(f"  Mean radius: {r_hc.mean():.2f}")
        print(f"  Radius std: {r_hc.std():.2f}")
        structure = 'shell' if r_hc.std() < 0.5 else 'distributed'
        print(f"  Structure: {structure}")
    
    # Check singularities on Möbius
    n_sing = field.singularity_count()
    print(f"\nMöbius singularities: {n_sing}")
    
    return em


def investigate_power_law():
    """Refine power law fit."""
    
    print("\n" + "=" * 60)
    print("Investigation 3: Power Law Refinement")
    print("=" * 60)
    
    R = 2.0
    w_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    proj = EMProjector(n=18, L=3.0)
    eb_values = []
    
    for w in w_values:
        field = MobiusField(n_u=48, n_v=24, R=R, w=w)
        sec = SECOperator(damping=0.98, pi_coupling=0.05)
        
        for _ in range(150):
            sec.step(field)
        
        em = proj.project(field)
        eb_values.append(em['EB_ratio'])
    
    eb_values = np.array(eb_values)
    wR_values = w_values / R
    powers = np.log(eb_values) / np.log(PHI)
    
    # Linear fit
    A = np.vstack([wR_values, np.ones(len(wR_values))]).T
    slope, intercept = np.linalg.lstsq(A, powers, rcond=None)[0]
    
    # R²
    powers_fit = slope * wR_values + intercept
    ss_res = ((powers - powers_fit) ** 2).sum()
    ss_tot = ((powers - powers.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    
    print(f"\nPower Law: φ-power = {slope:.3f} × (w/R) + {intercept:.3f}")
    print(f"R² = {r_squared:.4f}")
    print(f"\nE/B = φ when w/R = {(1 - intercept) / slope:.4f}")
    
    return {'slope': slope, 'intercept': intercept, 'r_squared': r_squared}


def investigate_golden_geometry():
    """Test golden ratio geometries."""
    
    print("\n" + "=" * 60)
    print("Investigation 4: Golden Ratio Geometries")
    print("=" * 60)
    
    R = 2.0
    special = {
        '1/φ²': PHI_INV**2,
        '1/φ': PHI_INV,
        'φ-1': PHI - 1,
        '1/2': 0.5,
    }
    
    proj = EMProjector(n=20, L=3.0)
    results = []
    
    print(f"\n{'Name':<12} {'w/R':<10} {'E/B':<10} {'φ-dev%':<10}")
    print("-" * 45)
    
    for name, ratio in special.items():
        w = R * ratio
        field = MobiusField(n_u=48, n_v=24, R=R, w=w)
        sec = SECOperator(damping=0.98, pi_coupling=0.05)
        
        for _ in range(150):
            sec.step(field)
        
        em = proj.project(field)
        phi_dev = abs(em['EB_ratio'] - PHI) / PHI * 100
        
        results.append({'name': name, 'ratio': ratio, 'EB': em['EB_ratio'], 'phi_dev': phi_dev})
        print(f"{name:<12} {ratio:<10.4f} {em['EB_ratio']:<10.4f} {phi_dev:<10.1f}")
    
    return results


def run_experiment():
    """Run all deep dive investigations."""
    
    print("=" * 70)
    print("EXPERIMENT 03: Deep Dive Analysis")
    print("=" * 70)
    
    results = {}
    
    # Run investigations
    wR_results, best_wR = investigate_optimal_wR()
    results['optimal_wR'] = {'sweep': wR_results, 'best': best_wR}
    
    charge_em = investigate_charge()
    results['charge'] = {
        'EB_ratio': charge_em['EB_ratio'],
        'charge_net': charge_em['charge_net']
    }
    
    power_law = investigate_power_law()
    results['power_law'] = power_law
    
    golden = investigate_golden_geometry()
    results['golden_geometry'] = golden
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOptimal w/R for E/B = φ: {best_wR['w_R']:.4f}")
    print(f"Power law: E/B = φ^({power_law['slope']:.2f} × w/R + {power_law['intercept']:.2f})")
    print(f"R² = {power_law['r_squared']:.4f}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_03_deep_dive',
        'results': results
    }
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"exp_03_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_experiment()
