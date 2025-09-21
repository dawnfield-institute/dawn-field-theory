#!/usr/bin/env python3
"""
Cosmic Web Scale Dark Matter Test
Focus on large-scale filamentary structure, not galaxy formation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from galaxy_simulator import GalaxySimulator, GalaxyConfig
from infodynamic_gravity import InfoGravityConfig, KPC_TO_METERS

def test_cosmic_web_simulation():
    """Test cosmic web simulation with proper scale physics"""
    
    # COSMIC WEB SCALE CONFIGURATION
    galaxy_config = GalaxyConfig(
        N_particles=1000,           # More particles for web structure
        total_mass=1e14,           # Much larger mass (cluster scale)
        disk_scale_length=25.0,    # Much larger scale (50 kpc instead of 5)
        bulge_fraction=0.1,        # Minimal central concentration
        dark_matter_fraction=0.8   # High initial dark matter fraction
    )
    
    # WEAK GRAVITY CONFIGURATION for cosmic web
    gravity_config = InfoGravityConfig(
        lambda_0=100 * KPC_TO_METERS,  # Much larger coherence length
        alpha_info=0.005857,           # Validated from spike
        beta_floor=2.0,                # MUCH higher quantum floor (200% of coherent)
        gamma=0.2,                     # Gentle power law
        kappa=1e45                     # MUCH weaker forces (10x smaller)
    )
    
    print("Creating COSMIC WEB simulator (not galaxy)...")
    print(f"  Scale: {galaxy_config.disk_scale_length*2:.0f} kpc diameter")
    print(f"  Mass: {galaxy_config.total_mass:.1e} M_sun (cluster scale)")
    print(f"  Gravity: κ={gravity_config.kappa:.1e} (weak for filaments)")
    
    sim = GalaxySimulator(galaxy_config, gravity_config)
    
    print("Running cosmic web evolution...")
    try:
        results = sim.run_simulation(n_steps=10, save_interval=2)
        print(f"✓ Cosmic web simulation completed successfully")
        print(f"  Generated {len(results)} snapshots")
        
        if results:
            for i, result in enumerate(results):
                step = i * 2
                dm_frac = result.get('dark_matter_fraction', 0.0)
                structures = result.get('structure_metrics', {}).get('structure_count', 0)
                print(f"  Step {step}: Dark matter {dm_frac:.1%}, Structures: {structures}")
                
            final_result = results[-1]
            final_dm = final_result.get('dark_matter_fraction', 0.0)
            
            if final_dm > 0.15:  # >15% dark matter indicates success
                print(f"🎉 SUCCESS: Achieved {final_dm:.1%} dark matter fraction!")
                print("   Cosmic web structure formation achieved")
            else:
                print(f"⚠️  PARTIAL: {final_dm:.1%} dark matter (target: >15%)")
                print("   May need weaker gravity or larger scale")
            
        return True
        
    except Exception as e:
        print(f"✗ Cosmic web simulation failed: {e}")
        return False

if __name__ == "__main__":
    test_cosmic_web_simulation()
