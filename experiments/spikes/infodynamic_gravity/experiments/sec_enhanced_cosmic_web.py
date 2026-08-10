#!/usr/bin/env python3
"""
SEC-Enhanced Cosmic Web Simulation
Applying insights from test.py to improve infodynamic gravity cosmic web formation

Key insights from test.py:
- ALPHA = 0.0005 (very weak gravity)
- Periodic boundaries essential
- Large box, low density → filaments
- 300+ timesteps for structure development
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from galaxy_simulator import GalaxySimulator, GalaxyConfig
from infodynamic_gravity import InfoGravityConfig, KPC_TO_METERS
from scale_dependent_arithmetic import get_scale_dependent_parameters

class SECEnhancedCosmicWeb:
    """
    Enhanced cosmic web simulation incorporating successful SEC techniques
    """
    
    def __init__(self):
        # SEC-inspired parameters (matching test.py success factors)
        self.setup_sec_cosmic_parameters()
        
    def setup_sec_cosmic_parameters(self):
        """Setup parameters inspired by successful test.py SEC simulation"""
        
        # Universe-scale configuration (like test.py)
        self.galaxy_config = GalaxyConfig(
            N_particles=800,              # Same as test.py success
            total_mass=1e13,              # Large cosmic web mass
            disk_scale_length=50.0,       # 100 kpc diameter (like test.py BOX_SIZE=100)
            bulge_fraction=0.05,          # Minimal central concentration
            dark_matter_fraction=0.8      # High initial dark matter
        )
        
        # Calculate system scale to verify cosmic web regime
        L_system = self.galaxy_config.disk_scale_length * 2  # Total system size
        scale_params = get_scale_dependent_parameters(L_system)
        
        print(f"System scale: {L_system:.0f} kpc")
        print(f"Scale regime: {scale_params['scale_regime']}")
        print(f"Expected dark matter: {scale_params['β_floor']:.1%}")
        
        # SEC-inspired weak gravity (matching test.py ALPHA=0.0005)
        # Convert test.py ALPHA to our κ parameter scale
        sec_alpha = 0.0005
        estimated_kappa = sec_alpha * 1e50  # Rough conversion to our force scale
        
        self.gravity_config = InfoGravityConfig(
            scale_dependent=True,         # Use scale-dependent parameters
            lambda_0=200 * KPC_TO_METERS, # Large coherence (like test.py)
            alpha_info=0.005857,          # Validated from darkmatter_SEC_WIP
            beta_floor=3.0,               # Very high quantum floor (300%)
            gamma=0.15,                   # Gentle power law for filaments
            kappa=estimated_kappa         # SEC-inspired weak coupling
        )
        
        print(f"SEC-inspired κ: {estimated_kappa:.1e}")
        print(f"Quantum floor: {self.gravity_config.beta_floor:.1f} (300%)")
        
    def run_sec_enhanced_simulation(self, n_steps: int = 300) -> List[Dict[str, Any]]:
        """
        Run cosmic web simulation with SEC-enhanced parameters
        
        Args:
            n_steps: Number of evolution steps (like test.py TIMESTEPS=300)
            
        Returns:
            List of simulation snapshots
        """
        print("=" * 60)
        print("SEC-ENHANCED COSMIC WEB SIMULATION")
        print("=" * 60)
        print(f"Particles: {self.galaxy_config.N_particles}")
        print(f"Box size: {self.galaxy_config.disk_scale_length*2:.0f} kpc")
        print(f"Timesteps: {n_steps}")
        print(f"Force coupling: {self.gravity_config.kappa:.1e}")
        print()
        
        # Create simulator
        sim = GalaxySimulator(self.galaxy_config, self.gravity_config)
        
        # Run simulation with frequent snapshots (like test.py)
        save_interval = max(1, n_steps // 20)  # 20 snapshots total
        
        print("Running SEC-enhanced cosmic web evolution...")
        results = sim.run_simulation(n_steps=n_steps, save_interval=save_interval)
        
        if results:
            print(f"✓ Simulation completed: {len(results)} snapshots")
            self.analyze_sec_enhanced_results(results)
        else:
            print("✗ Simulation failed")
            
        return results
        
    def analyze_sec_enhanced_results(self, results: List[Dict[str, Any]]):
        """Analyze results focusing on cosmic web structure formation"""
        
        print("\nSEC-ENHANCED COSMIC WEB ANALYSIS:")
        print("-" * 40)
        
        for i, result in enumerate(results):
            step = i * (300 // len(results)) if len(results) > 1 else i
            
            # Extract key metrics
            dm_frac = result.get('dark_matter_fraction', 0.0)
            total_info = result.get('total_information', 0.0)
            
            # Structure metrics
            structure_metrics = result.get('structure_metrics', {})
            structure_count = structure_metrics.get('structure_count', 0)
            
            print(f"Step {step:3d}: DM={dm_frac:.1%}, Structures={structure_count:2d}, Info={total_info:.2e}")
            
        # Final analysis
        if results:
            final_result = results[-1]
            final_dm = final_result.get('dark_matter_fraction', 0.0)
            
            print(f"\nFINAL COSMIC WEB STATE:")
            print(f"Dark matter fraction: {final_dm:.1%}")
            
            # Compare with test.py success
            if final_dm > 0.40:  # Cosmic web should show high dark matter
                print("🎉 SUCCESS: SEC-enhanced cosmic web formation achieved!")
                print("   High dark matter fraction indicates filamentary structure")
            elif final_dm > 0.20:
                print("🔶 PARTIAL: Some cosmic web features, may need parameter tuning")
            else:
                print("⚠️  MINIMAL: Low dark matter suggests insufficient cosmic web formation")
                
            # Recommendations based on test.py insights
            print(f"\nSEC INSIGHTS APPLICATION:")
            print(f"✓ Used test.py particle count: {self.galaxy_config.N_particles}")
            print(f"✓ Applied weak force coupling (inspired by ALPHA=0.0005)")
            print(f"✓ Large-scale box size for filament formation")
            print(f"✓ Extended evolution time for structure development")
            
    def plot_sec_comparison(self, results: List[Dict[str, Any]], save_path: str = None):
        """Plot comparison with test.py SEC approach"""
        
        if not results:
            return
            
        steps = [i * (300 // len(results)) for i in range(len(results))]
        dm_fractions = [r.get('dark_matter_fraction', 0.0) for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, dm_fractions, 'b-o', label='SEC-Enhanced Infodynamic Gravity')
        plt.axhline(y=0.6, color='r', linestyle='--', label='Cosmic Web Target (60%)')
        plt.axhline(y=0.4, color='orange', linestyle=':', label='Filament Threshold (40%)')
        
        plt.xlabel('Simulation Step')
        plt.ylabel('Dark Matter Fraction')
        plt.title('SEC-Enhanced Cosmic Web Formation\n(Inspired by test.py)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

def run_sec_enhanced_test():
    """Run the SEC-enhanced cosmic web test"""
    
    # Create SEC-enhanced simulator
    sec_sim = SECEnhancedCosmicWeb()
    
    # Run simulation with test.py-inspired parameters
    results = sec_sim.run_sec_enhanced_simulation(n_steps=300)
    
    # Plot results
    if results:
        sec_sim.plot_sec_comparison(results, 'sec_enhanced_cosmic_web.png')
        
    return sec_sim, results

if __name__ == "__main__":
    run_sec_enhanced_test()
