#!/usr/bin/env python3
"""
Cross-Scale Information Exchange Test
Testing hierarchical information fields (I_local + I_global) from scale-dependent specification

Based on successful SEC insights from test.py and scale-dependent implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from infodynamic_gravity import InfoGravityField, InfoGravityConfig, KPC_TO_METERS
from scale_dependent_arithmetic import get_scale_dependent_parameters, calculate_characteristic_length

class CrossScaleInfoTest:
    """
    Test hierarchical information field implementation
    
    Implements:
    I_total(r) = I_local(r) + I_global(r)
    
    Where:
    I_local(r) = Σ_nearby I₀_local × exp(-r/λ_galaxy)
    I_global(r) = Σ_all I₀_global × exp(-r/λ_cosmic) × (1 + r/λ_cosmic)^(-γ)
    """
    
    def __init__(self):
        self.setup_hierarchical_test()
        
    def setup_hierarchical_test(self):
        """Setup test configuration for hierarchical information fields"""
        
        # Multi-scale test configuration
        # Galaxy-scale particles (local information)
        self.n_local = 200
        self.local_scale = 20.0  # kpc - galaxy scale
        
        # Cosmic-scale particles (global information) 
        self.n_global = 100
        self.global_scale = 200.0  # kpc - cosmic web scale
        
        # Information exchange parameters (from specification)
        self.nu_exchange = 0.01  # Exchange rate
        self.lambda_galaxy = 30.0  # kpc
        self.lambda_cosmic = 2000.0  # kpc (2 Mpc)
        
        print("Cross-Scale Information Exchange Test Setup:")
        print(f"Local particles: {self.n_local} (galaxy scale: {self.local_scale} kpc)")
        print(f"Global particles: {self.n_global} (cosmic scale: {self.global_scale} kpc)")
        print(f"λ_galaxy: {self.lambda_galaxy} kpc")
        print(f"λ_cosmic: {self.lambda_cosmic} kpc")
        print(f"Exchange rate ν: {self.nu_exchange}")
        
    def create_hierarchical_system(self) -> Dict[str, Any]:
        """Create multi-scale particle system"""
        
        # Galaxy-scale particles (clustered)
        galaxy_center = np.array([50.0, 50.0, 50.0])  # kpc
        local_positions = galaxy_center + np.random.normal(0, self.local_scale/3, (self.n_local, 3))
        local_masses = np.ones(self.n_local) * 1e8  # Solar masses
        
        # Cosmic-scale particles (distributed)
        global_positions = np.random.uniform(0, self.global_scale, (self.n_global, 3))
        global_masses = np.ones(self.n_global) * 1e10  # Larger cosmic structures
        
        # Combine systems
        all_positions = np.vstack([local_positions, global_positions])
        all_masses = np.concatenate([local_masses, global_masses])
        all_velocities = np.zeros_like(all_positions)
        
        # Mark which particles are local vs global
        local_mask = np.concatenate([np.ones(self.n_local, dtype=bool), 
                                    np.zeros(self.n_global, dtype=bool)])
        
        return {
            'positions': all_positions * KPC_TO_METERS,  # Convert to meters
            'masses': all_masses * 1.989e30,  # Convert to kg
            'velocities': all_velocities,
            'local_mask': local_mask,
            'n_local': self.n_local,
            'n_global': self.n_global
        }
        
    def calculate_hierarchical_information(self, positions: np.ndarray, masses: np.ndarray, 
                                         local_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate hierarchical information field
        
        Returns:
            I_local, I_global, I_total arrays
        """
        N = len(masses)
        positions_kpc = positions / KPC_TO_METERS
        
        I_local = np.zeros(N)
        I_global = np.zeros(N)
        
        # Calculate local information (galaxy-scale)
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = np.linalg.norm(positions_kpc[i] - positions_kpc[j])
                    
                    # Local contribution (nearby particles only)
                    if r_ij < 3 * self.lambda_galaxy:  # Local neighborhood
                        I_0_local = masses[j] / 1.989e30  # Normalize to solar masses
                        I_local[i] += I_0_local * np.exp(-r_ij / self.lambda_galaxy)
                    
                    # Global contribution (all particles)
                    I_0_global = masses[j] / 1.989e30  # Normalize to solar masses
                    power_law = (1 + r_ij / self.lambda_cosmic)**(-0.2)  # γ = 0.2
                    I_global[i] += I_0_global * np.exp(-r_ij / self.lambda_cosmic) * power_law
        
        I_total = I_local + I_global
        
        return I_local, I_global, I_total
        
    def calculate_information_exchange(self, I_local: np.ndarray, I_global: np.ndarray,
                                     positions: np.ndarray) -> np.ndarray:
        """
        Calculate cross-scale information exchange
        
        dI_exchange/dt = -ν × (I_local - I_global) × ∇²L
        """
        positions_kpc = positions / KPC_TO_METERS
        N = len(positions)
        
        # Calculate Laplacian of scale (simplified as local density gradient)
        laplacian_L = np.zeros(N)
        
        for i in range(N):
            # Estimate local scale gradient
            local_densities = []
            for j in range(N):
                if i != j:
                    r_ij = np.linalg.norm(positions_kpc[i] - positions_kpc[j])
                    if r_ij < 100:  # Local neighborhood for gradient calculation
                        density = 1.0 / (r_ij**2 + 1e-10)
                        local_densities.append(density)
            
            if local_densities:
                # Simplified Laplacian estimate
                laplacian_L[i] = np.var(local_densities) / (np.mean(local_densities) + 1e-10)
        
        # Information exchange rate
        dI_exchange = -self.nu_exchange * (I_local - I_global) * laplacian_L
        
        return dI_exchange
        
    def run_hierarchical_test(self, n_steps: int = 50) -> Dict[str, Any]:
        """Run hierarchical information field test"""
        
        print("\n" + "="*60)
        print("HIERARCHICAL INFORMATION FIELD TEST")
        print("="*60)
        
        # Create multi-scale system
        system = self.create_hierarchical_system()
        
        # Analyze system scales
        L_system = calculate_characteristic_length(
            system['positions'] / KPC_TO_METERS, 
            system['masses'] / 1.989e30
        )
        
        scale_params = get_scale_dependent_parameters(L_system)
        print(f"\nSystem Analysis:")
        print(f"Characteristic length: {L_system:.1f} kpc")
        print(f"Scale regime: {scale_params['scale_regime']}")
        print(f"Expected dark matter: {scale_params['β_floor']:.1%}")
        
        # Track evolution
        evolution_data = {
            'I_local_total': [],
            'I_global_total': [],
            'I_total': [],
            'information_exchange': [],
            'local_fraction': [],
            'steps': []
        }
        
        positions = system['positions'].copy()
        masses = system['masses']
        local_mask = system['local_mask']
        
        print(f"\nEvolution over {n_steps} steps:")
        print("Step | I_local | I_global | Exchange | Local% | Total_Info")
        print("-" * 60)
        
        for step in range(n_steps):
            # Calculate hierarchical information
            I_local, I_global, I_total = self.calculate_hierarchical_information(
                positions, masses, local_mask
            )
            
            # Calculate information exchange
            dI_exchange = self.calculate_information_exchange(I_local, I_global, positions)
            
            # Apply information exchange (update positions slightly)
            if step > 0:
                exchange_force = dI_exchange.reshape(-1, 1) * 1e-15  # Tiny adjustment
                positions += exchange_force * np.random.normal(0, 1, positions.shape)
            
            # Record metrics
            I_local_total = np.sum(I_local)
            I_global_total = np.sum(I_global)
            I_total_sum = np.sum(I_total)
            exchange_magnitude = np.mean(np.abs(dI_exchange))
            local_fraction = I_local_total / (I_local_total + I_global_total + 1e-30)
            
            evolution_data['I_local_total'].append(I_local_total)
            evolution_data['I_global_total'].append(I_global_total)
            evolution_data['I_total'].append(I_total_sum)
            evolution_data['information_exchange'].append(exchange_magnitude)
            evolution_data['local_fraction'].append(local_fraction)
            evolution_data['steps'].append(step)
            
            # Print every 10 steps
            if step % 10 == 0:
                print(f"{step:4d} | {I_local_total:7.1e} | {I_global_total:8.1e} | "
                      f"{exchange_magnitude:8.1e} | {local_fraction:5.1%} | {I_total_sum:9.1e}")
        
        return evolution_data
        
    def analyze_hierarchical_results(self, evolution_data: Dict[str, Any]):
        """Analyze hierarchical information field results"""
        
        print(f"\nHIERARCHICAL INFORMATION ANALYSIS:")
        print("-" * 40)
        
        final_local = evolution_data['I_local_total'][-1]
        final_global = evolution_data['I_global_total'][-1]
        final_exchange = evolution_data['information_exchange'][-1]
        final_local_frac = evolution_data['local_fraction'][-1]
        
        print(f"Final local information: {final_local:.2e}")
        print(f"Final global information: {final_global:.2e}")
        print(f"Final exchange rate: {final_exchange:.2e}")
        print(f"Local information fraction: {final_local_frac:.1%}")
        
        # Test hierarchical structure
        local_to_global_ratio = final_local / (final_global + 1e-30)
        
        if 0.1 < local_to_global_ratio < 10:
            print("✅ HIERARCHICAL BALANCE: Local and global info both significant")
        elif local_to_global_ratio > 10:
            print("⚠️  LOCAL DOMINANCE: Galaxy-scale information dominates")
        else:
            print("⚠️  GLOBAL DOMINANCE: Cosmic-scale information dominates")
        
        # Test information exchange
        exchange_variation = np.std(evolution_data['information_exchange'])
        if exchange_variation > 0:
            print("✅ ACTIVE EXCHANGE: Cross-scale information flow detected")
        else:
            print("⚠️  STATIC SYSTEM: No cross-scale information exchange")
        
        # Test scale-dependent behavior
        if final_local_frac > 0.2 and final_local_frac < 0.8:
            print("✅ SCALE MIXING: Both local and global scales contribute")
        else:
            print("⚠️  SCALE SEPARATION: One scale dominates completely")
            
    def plot_hierarchical_evolution(self, evolution_data: Dict[str, Any], save_path: str = None):
        """Plot hierarchical information field evolution"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        steps = evolution_data['steps']
        
        # Information components
        ax1.plot(steps, evolution_data['I_local_total'], 'b-', label='I_local', linewidth=2)
        ax1.plot(steps, evolution_data['I_global_total'], 'r-', label='I_global', linewidth=2)
        ax1.plot(steps, evolution_data['I_total'], 'k--', label='I_total', linewidth=1)
        ax1.set_xlabel('Evolution Step')
        ax1.set_ylabel('Total Information')
        ax1.set_title('Hierarchical Information Components')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Information exchange
        ax2.plot(steps, evolution_data['information_exchange'], 'g-', linewidth=2)
        ax2.set_xlabel('Evolution Step')
        ax2.set_ylabel('Exchange Rate |dI/dt|')
        ax2.set_title('Cross-Scale Information Exchange')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Local fraction
        ax3.plot(steps, evolution_data['local_fraction'], 'm-', linewidth=2)
        ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Balance')
        ax3.set_xlabel('Evolution Step')
        ax3.set_ylabel('Local Information Fraction')
        ax3.set_title('Local vs Global Information Balance')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Information ratios
        local_to_global = np.array(evolution_data['I_local_total']) / (np.array(evolution_data['I_global_total']) + 1e-30)
        ax4.plot(steps, local_to_global, 'c-', linewidth=2)
        ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal')
        ax4.set_xlabel('Evolution Step')
        ax4.set_ylabel('I_local / I_global')
        ax4.set_title('Local-to-Global Information Ratio')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Hierarchical analysis plot saved: {save_path}")
        else:
            plt.show()

def run_hierarchical_test():
    """Run the hierarchical information field test"""
    
    # Create hierarchical test
    hier_test = CrossScaleInfoTest()
    
    # Run evolution
    evolution_data = hier_test.run_hierarchical_test(n_steps=50)
    
    # Analyze results
    hier_test.analyze_hierarchical_results(evolution_data)
    
    # Plot evolution
    hier_test.plot_hierarchical_evolution(evolution_data, 'hierarchical_information.png')
    
    return hier_test, evolution_data

if __name__ == "__main__":
    run_hierarchical_test()
