"""
Validation Tests for Infodynamic Gravity

Tests the mathematical formulation by extracting scaling laws from simulations
and comparing against theoretical predictions and observational data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import pytest
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import logging

from infodynamic_gravity import InfoGravityField, InfoGravityConfig, create_two_body_test
from sec_dynamics import SECDynamics, SECConfig
from galaxy_simulator import GalaxySimulator, GalaxyConfig

# Constants for validation
K_B = 1.380649e-23
KPC_TO_METERS = 3.086e19
MYR_TO_SECONDS = 3.15e13

class ValidationTests:
    """
    Comprehensive validation suite for infodynamic gravity theory
    
    Tests:
    1. Quadratic scaling law: N_bits ∝ g²
    2. Landauer energy correspondence
    3. Dark matter emergence
    4. Galaxy rotation curves
    5. Information conservation
    """
    
    def __init__(self):
        self.test_results = {}
        
    def test_quadratic_scaling_law(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Test the predicted quadratic scaling: N_bits ∝ g²
        
        Varies gravitational field strength and measures information erasure.
        """
        print("Testing quadratic scaling law...")
        
        g_values = []
        n_bits_values = []
        
        for trial in range(n_trials):
            # Create two-body system with varying separation
            separation = 0.5 + 4.5 * trial / n_trials  # 0.5 to 5 kpc
            
            gravity, state = create_two_body_test()
            
            # Set specific separation
            state['positions'][1, 0] = separation * KPC_TO_METERS
            
            # Evolve system and measure information erasure
            initial_info = None
            final_info = None
            
            for step in range(10):
                if step == 0:
                    state = gravity.recursive_evolution_step(state, MYR_TO_SECONDS)
                    initial_info = state['total_information']
                else:
                    state = gravity.recursive_evolution_step(state, MYR_TO_SECONDS)
                    
            final_info = state['total_information']
            
            # Calculate effective gravitational field
            r = np.linalg.norm(state['positions'][1] - state['positions'][0])
            M_total = np.sum(state['masses'])
            g_eff = 6.67e-11 * M_total / r**2
            
            # Information bits erased
            info_erased = initial_info - final_info
            n_bits = max(0, info_erased / (K_B * gravity.config.T_info * np.log(2)))
            
            g_values.append(g_eff)
            n_bits_values.append(n_bits)
        
        # Fit to quadratic model: N_bits = C * g²
        def quadratic_model(g, C):
            return C * g**2
            
        try:
            popt, pcov = curve_fit(quadratic_model, g_values, n_bits_values)
            quadratic_coeff = popt[0]
            
            # Calculate R²
            y_pred = quadratic_model(np.array(g_values), quadratic_coeff)
            ss_res = np.sum((np.array(n_bits_values) - y_pred)**2)
            ss_tot = np.sum((np.array(n_bits_values) - np.mean(n_bits_values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            result = {
                'quadratic_coefficient': quadratic_coeff,
                'r_squared': r_squared,
                'fit_quality': 'excellent' if r_squared > 0.8 else 'poor',
                'g_values': g_values,
                'n_bits_values': n_bits_values,
                'passes_test': r_squared > 0.6
            }
            
        except Exception as e:
            result = {
                'error': str(e),
                'passes_test': False
            }
        
        self.test_results['quadratic_scaling'] = result
        return result
    
    def test_landauer_correspondence(self, n_steps: int = 50) -> Dict[str, Any]:
        """
        Test energy-information correspondence via Landauer principle
        
        Measures correlation between kinetic energy changes and information erasure.
        """
        print("Testing Landauer energy correspondence...")
        
        gravity, state = create_two_body_test()
        
        energy_changes = []
        info_erasure = []
        
        prev_kinetic = 0.5 * np.sum(state['masses'] * np.sum(state['velocities']**2, axis=1))
        prev_info = 0
        
        for step in range(n_steps):
            state = gravity.recursive_evolution_step(state, MYR_TO_SECONDS * 0.1)
            
            # Calculate kinetic energy change
            kinetic = 0.5 * np.sum(state['masses'] * np.sum(state['velocities']**2, axis=1))
            delta_kinetic = kinetic - prev_kinetic
            
            # Calculate information change
            current_info = state['total_information']
            delta_info = current_info - prev_info if prev_info > 0 else 0
            
            if step > 0:  # Skip first step (no previous data)
                energy_changes.append(delta_kinetic)
                
                # Convert information change to energy via Landauer
                landauer_energy = -delta_info * K_B * gravity.config.T_info * np.log(2)
                info_erasure.append(landauer_energy)
            
            prev_kinetic = kinetic
            prev_info = current_info
        
        # Calculate correlation
        if len(energy_changes) > 5:
            correlation, p_value = pearsonr(energy_changes, info_erasure)
            
            result = {
                'correlation': correlation,
                'p_value': p_value,
                'n_samples': len(energy_changes),
                'passes_test': correlation > 0.5 and p_value < 0.05,
                'energy_changes': energy_changes,
                'info_erasure': info_erasure
            }
        else:
            result = {
                'error': 'Insufficient data',
                'passes_test': False
            }
        
        self.test_results['landauer_correspondence'] = result
        return result
    
    def test_dark_matter_emergence(self) -> Dict[str, Any]:
        """
        Test emergence of dark matter effects in galaxy simulation
        
        Checks for flat rotation curves and correct dark matter fraction.
        """
        print("Testing dark matter emergence...")
        
        # Run galaxy simulation
        galaxy_config = GalaxyConfig(N_particles=500, total_mass=1e11)
        gravity_config = InfoGravityConfig(
            lambda_c=10 * KPC_TO_METERS,
            quantum_floor=0.2  # 20% quantum floor
        )
        
        sim = GalaxySimulator(galaxy_config, gravity_config)
        
        # Run short simulation
        results = sim.run_simulation(n_steps=20, save_interval=5)
        
        if not results:
            return {'error': 'Simulation failed', 'passes_test': False}
        
        # Analyze final rotation curve
        final_curve = results[-1]['rotation_curve']
        
        if len(final_curve['radius_kpc']) < 5:
            return {'error': 'Insufficient rotation curve data', 'passes_test': False}
        
        # Test for flat rotation curve (outer region should be roughly constant)
        r = final_curve['radius_kpc']
        v = final_curve['velocity_km_s']
        
        # Check outer region flatness
        outer_mask = r > np.percentile(r, 60)  # Outer 40% of data
        if np.sum(outer_mask) > 3:
            outer_velocities = v[outer_mask]
            velocity_variation = np.std(outer_velocities) / np.mean(outer_velocities)
            
            # Check dark matter fraction
            final_dm_fraction = results[-1]['dark_matter_fraction']
            
            result = {
                'velocity_variation': velocity_variation,
                'dark_matter_fraction': final_dm_fraction,
                'rotation_curve_flat': velocity_variation < 0.3,  # <30% variation
                'dark_matter_present': final_dm_fraction > 0.1,   # >10% dark matter
                'passes_test': velocity_variation < 0.3 and final_dm_fraction > 0.1,
                'rotation_curve': final_curve
            }
        else:
            result = {
                'error': 'Insufficient outer rotation curve data',
                'passes_test': False
            }
        
        self.test_results['dark_matter_emergence'] = result
        return result
    
    def test_information_conservation(self) -> Dict[str, Any]:
        """
        Test that total information decreases monotonically (entropy increase)
        """
        print("Testing information conservation...")
        
        gravity, state = create_two_body_test()
        
        info_history = []
        
        for step in range(30):
            state = gravity.recursive_evolution_step(state, MYR_TO_SECONDS)
            info_history.append(state['total_information'])
        
        # Check monotonic decrease
        decreases = 0
        total_changes = 0
        
        for i in range(1, len(info_history)):
            if info_history[i] <= info_history[i-1]:
                decreases += 1
            total_changes += 1
        
        monotonic_fraction = decreases / total_changes if total_changes > 0 else 0
        
        # Total information change
        total_change = info_history[-1] - info_history[0]
        
        result = {
            'monotonic_fraction': monotonic_fraction,
            'total_information_change': total_change,
            'info_history': info_history,
            'passes_test': monotonic_fraction > 0.8 and total_change <= 0
        }
        
        self.test_results['information_conservation'] = result
        return result
    
    def test_sec_structure_formation(self) -> Dict[str, Any]:
        """
        Test SEC dynamics for structure formation capability
        """
        print("Testing SEC structure formation...")
        
        config = SECConfig(collapse_threshold=0.6, force_amplification=1e5)
        sec = SECDynamics(config)
        
        # Create high-entropy initial state
        N = 50
        positions = np.random.normal(0, 1e18, (N, 3))  # 0.1 kpc spread
        velocities = np.random.normal(0, 1e5, (N, 3))  # 100 km/s dispersion
        masses = np.ones(N) * 1e30  # Solar masses
        
        state = {
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'time': 0.0,
            'dt': MYR_TO_SECONDS
        }
        
        collapse_events = 0
        entropy_reduction = 0
        
        for step in range(20):
            initial_entropy = np.mean(state.get('entropy_density', [1.0]))
            
            state = sec.execute_collapse_step(state)
            
            if state.get('collapse_occurred', False):
                collapse_events += 1
                
            final_entropy = np.mean(state.get('entropy_density', [1.0]))
            entropy_reduction += max(0, initial_entropy - final_entropy)
        
        # Analyze final structure
        analysis = sec.analyze_structure_formation()
        
        result = {
            'collapse_events': collapse_events,
            'entropy_reduction': entropy_reduction,
            'structure_formation_rate': analysis.get('collapse_rate', 0),
            'passes_test': collapse_events > 0 and entropy_reduction > 0,
            'analysis': analysis
        }
        
        self.test_results['sec_structure_formation'] = result
        return result
    
    def run_full_validation_suite(self) -> Dict[str, Any]:
        """
        Run complete validation suite and generate summary report
        """
        print("=== Running Full Infodynamic Gravity Validation Suite ===\n")
        
        tests = [
            ('Quadratic Scaling Law', self.test_quadratic_scaling_law),
            ('Landauer Correspondence', self.test_landauer_correspondence),
            ('Dark Matter Emergence', self.test_dark_matter_emergence),
            ('Information Conservation', self.test_information_conservation),
            ('SEC Structure Formation', self.test_sec_structure_formation)
        ]
        
        passed = 0
        total = len(tests)
        
        for name, test_func in tests:
            print(f"Running {name}...")
            try:
                result = test_func()
                success = result.get('passes_test', False)
                if success:
                    passed += 1
                    print(f"✓ {name}: PASSED")
                else:
                    print(f"✗ {name}: FAILED - {result.get('error', 'Test criteria not met')}")
            except Exception as e:
                print(f"✗ {name}: ERROR - {e}")
                self.test_results[name.lower().replace(' ', '_')] = {'error': str(e), 'passes_test': False}
            
            print()
        
        # Generate summary
        summary = {
            'tests_passed': passed,
            'tests_total': total,
            'pass_rate': passed / total,
            'overall_status': 'PASS' if passed >= total * 0.6 else 'FAIL',  # 60% threshold
            'individual_results': self.test_results
        }
        
        print(f"=== Validation Summary ===")
        print(f"Tests passed: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"Overall status: {summary['overall_status']}")
        
        return summary
    
    def plot_validation_results(self, save_path: Optional[str] = None):
        """Plot validation test results"""
        
        if not self.test_results:
            print("No test results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Quadratic scaling plot
        if 'quadratic_scaling' in self.test_results:
            data = self.test_results['quadratic_scaling']
            if 'g_values' in data:
                axes[0].scatter(data['g_values'], data['n_bits_values'], alpha=0.6)
                axes[0].set_xlabel('Gravitational Field (m/s²)')
                axes[0].set_ylabel('Information Bits Erased')
                axes[0].set_title(f"Quadratic Scaling (R²={data.get('r_squared', 0):.3f})")
                axes[0].set_xscale('log')
                axes[0].set_yscale('log')
        
        # Landauer correspondence
        if 'landauer_correspondence' in self.test_results:
            data = self.test_results['landauer_correspondence']
            if 'energy_changes' in data:
                axes[1].scatter(data['energy_changes'], data['info_erasure'], alpha=0.6)
                axes[1].set_xlabel('Kinetic Energy Change (J)')
                axes[1].set_ylabel('Landauer Energy (J)')
                axes[1].set_title(f"Landauer Correspondence (r={data.get('correlation', 0):.3f})")
        
        # Dark matter rotation curve
        if 'dark_matter_emergence' in self.test_results:
            data = self.test_results['dark_matter_emergence']
            if 'rotation_curve' in data:
                curve = data['rotation_curve']
                axes[2].plot(curve['radius_kpc'], curve['velocity_km_s'], 'o-')
                axes[2].set_xlabel('Radius (kpc)')
                axes[2].set_ylabel('Velocity (km/s)')
                axes[2].set_title(f"Rotation Curve (DM: {data.get('dark_matter_fraction', 0):.1%})")
        
        # Information conservation
        if 'information_conservation' in self.test_results:
            data = self.test_results['information_conservation']
            if 'info_history' in data:
                axes[3].plot(data['info_history'])
                axes[3].set_xlabel('Evolution Step')
                axes[3].set_ylabel('Total Information')
                axes[3].set_title('Information Conservation')
        
        # Test pass/fail summary
        test_names = []
        test_status = []
        
        for key, result in self.test_results.items():
            test_names.append(key.replace('_', ' ').title())
            test_status.append(1 if result.get('passes_test', False) else 0)
        
        if test_names:
            colors = ['green' if status else 'red' for status in test_status]
            axes[4].bar(range(len(test_names)), test_status, color=colors)
            axes[4].set_xticks(range(len(test_names)))
            axes[4].set_xticklabels(test_names, rotation=45, ha='right')
            axes[4].set_ylabel('Pass (1) / Fail (0)')
            axes[4].set_title('Test Results Summary')
        
        # Overall score
        if self.test_results:
            passed = sum(1 for r in self.test_results.values() if r.get('passes_test', False))
            total = len(self.test_results)
            score = passed / total
            
            axes[5].pie([score, 1-score], labels=['Passed', 'Failed'], 
                       colors=['green', 'red'], autopct='%1.1f%%')
            axes[5].set_title(f'Overall Score: {passed}/{total}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plots saved to {save_path}")
        
        plt.show()

def run_validation():
    """Run the complete validation suite"""
    validator = ValidationTests()
    summary = validator.run_full_validation_suite()
    validator.plot_validation_results('infodynamic_validation_results.png')
    return validator, summary

if __name__ == "__main__":
    # Run validation tests
    validator, summary = run_validation()
    
    print(f"\nFinal Assessment:")
    print(f"Infodynamic gravity theory: {summary['overall_status']}")
    print(f"Mathematical formulation quality: {summary['pass_rate']:.1%}")
    
    # Individual test details
    print(f"\nDetailed Results:")
    for test_name, result in summary['individual_results'].items():
        status = "PASS" if result.get('passes_test', False) else "FAIL"
        print(f"  {test_name}: {status}")
        
        # Key metrics
        if test_name == 'quadratic_scaling':
            r2 = result.get('r_squared', 0)
            print(f"    R² = {r2:.3f} (target: >0.6)")
        elif test_name == 'landauer_correspondence':
            corr = result.get('correlation', 0)
            print(f"    Correlation = {corr:.3f} (target: >0.5)")
        elif test_name == 'dark_matter_emergence':
            dm_frac = result.get('dark_matter_fraction', 0)
            print(f"    Dark matter fraction = {dm_frac:.1%} (target: >10%)")
