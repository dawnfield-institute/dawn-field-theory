"""
Run Infodynamic Gravity Experiments

Main entry point for running infodynamic gravity experiments and validations.
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# Import experiment modules
from infodynamic_gravity import InfoGravityField, InfoGravityConfig, create_two_body_test
from sec_dynamics import SECDynamics, SECConfig
from galaxy_simulator import GalaxySimulator, GalaxyConfig, run_dark_matter_test
from validation_tests import ValidationTests, run_validation

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('infodynamic_experiments.log')
        ]
    )

def run_basic_test():
    """Run basic two-body infodynamic gravity test"""
    print("=== Basic Infodynamic Gravity Test ===")
    
    gravity, state = create_two_body_test()
    
    print(f"Initial configuration:")
    print(f"  Particles: 2 solar masses")
    print(f"  Separation: {state['positions'][1, 0]/3.086e19:.2f} kpc")
    print(f"  Coherence length: {gravity.config.lambda_c/3.086e19:.2f} kpc")
    print(f"  Quantum floor: {gravity.config.quantum_floor:.1%}")
    
    print(f"\nEvolution:")
    
    for step in range(10):
        state = gravity.recursive_evolution_step(state, 3.15e13)  # 1 Myr steps
        
        separation = np.linalg.norm(state['positions'][1] - state['positions'][0])
        
        print(f"  Step {step+1}: sep={separation/3.086e19:.3f} kpc, "
              f"info={state['total_information']:.2e}, "
              f"DM={state['dark_matter_fraction']:.1%}")
    
    # Conservation check
    conservation = gravity.validate_conservation_laws(state)
    print(f"\nConservation laws:")
    for key, value in conservation.items():
        print(f"  {key}: {value:.2e}")

def run_sec_test():
    """Run SEC dynamics test"""
    print("=== SEC Dynamics Test ===")
    
    from sec_dynamics import test_sec_dynamics
    sec, final_state = test_sec_dynamics()
    
    analysis = sec.analyze_structure_formation()
    print(f"\nSEC Analysis:")
    print(f"  Collapse events: {analysis.get('total_events', 0)}")
    print(f"  Structure formation rate: {analysis.get('collapse_rate', 0):.2e}")
    print(f"  Mean entropy reduction: {analysis.get('mean_entropy_reduction', 0):.3f}")

def run_galaxy_test():
    """Run galaxy simulation test"""
    print("=== Galaxy Simulation Test ===")
    
    sim, results = run_dark_matter_test()
    
    if results:
        final_result = results[-1]
        print(f"\nGalaxy simulation completed:")
        print(f"  Evolution time: {final_result['time_myr']:.1f} Myr")
        print(f"  Dark matter fraction: {final_result['dark_matter_fraction']:.1%}")
        print(f"  Structure events: {final_result['structure_metrics'].get('total_structure_events', 0)}")
        
        # Rotation curve analysis
        curve = final_result['rotation_curve']
        if len(curve['radius_kpc']) > 0:
            print(f"  Rotation curve: {len(curve['radius_kpc'])} points")
            print(f"  Max radius: {max(curve['radius_kpc']):.1f} kpc")
            print(f"  Velocity range: {min(curve['velocity_km_s']):.0f}-{max(curve['velocity_km_s']):.0f} km/s")
    else:
        print("Galaxy simulation failed to produce results")

def run_full_validation():
    """Run complete validation suite"""
    print("=== Full Validation Suite ===")
    
    validator, summary = run_validation()
    
    print(f"\nValidation completed:")
    print(f"  Overall status: {summary['overall_status']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")
    
    return summary

def run_parameter_study():
    """Run parameter sensitivity study"""
    print("=== Parameter Sensitivity Study ===")
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Test different coherence lengths
    lambda_c_values = np.logspace(18, 21, 10)  # 0.1 to 100 kpc
    quantum_floors = [0.05, 0.1, 0.2, 0.3]
    
    results = {}
    
    for i, lambda_c in enumerate(lambda_c_values):
        for j, q_floor in enumerate(quantum_floors):
            print(f"Testing λ_c={lambda_c/3.086e19:.1f} kpc, q_floor={q_floor:.1%}")
            
            # Quick two-body test
            config = InfoGravityConfig(
                lambda_c=lambda_c,
                quantum_floor=q_floor
            )
            
            gravity = InfoGravityField(config)
            _, state = create_two_body_test()
            
            # Run short evolution
            initial_info = 0
            final_info = 0
            
            for step in range(5):
                state = gravity.recursive_evolution_step(state, 3.15e13)
                if step == 0:
                    initial_info = state['total_information']
                final_info = state['total_information']
            
            # Store results
            key = f"lc_{i}_qf_{j}"
            results[key] = {
                'lambda_c_kpc': lambda_c / 3.086e19,
                'quantum_floor': q_floor,
                'dark_matter_fraction': state['dark_matter_fraction'],
                'info_change': final_info - initial_info,
                'final_separation': np.linalg.norm(state['positions'][1] - state['positions'][0]) / 3.086e19
            }
    
    # Plot parameter space
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data for plotting
    lc_vals = [results[k]['lambda_c_kpc'] for k in results.keys()]
    qf_vals = [results[k]['quantum_floor'] for k in results.keys()]
    dm_fracs = [results[k]['dark_matter_fraction'] for k in results.keys()]
    
    # Dark matter fraction vs parameters
    scatter = axes[0, 0].scatter(lc_vals, qf_vals, c=dm_fracs, cmap='viridis')
    axes[0, 0].set_xlabel('Coherence Length (kpc)')
    axes[0, 0].set_ylabel('Quantum Floor')
    axes[0, 0].set_title('Dark Matter Fraction')
    axes[0, 0].set_xscale('log')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Information change
    info_changes = [results[k]['info_change'] for k in results.keys()]
    scatter2 = axes[0, 1].scatter(lc_vals, qf_vals, c=info_changes, cmap='plasma')
    axes[0, 1].set_xlabel('Coherence Length (kpc)')
    axes[0, 1].set_ylabel('Quantum Floor')
    axes[0, 1].set_title('Information Change')
    axes[0, 1].set_xscale('log')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Separation change
    seps = [results[k]['final_separation'] for k in results.keys()]
    axes[1, 0].scatter(lc_vals, seps, c=qf_vals, cmap='cool')
    axes[1, 0].set_xlabel('Coherence Length (kpc)')
    axes[1, 0].set_ylabel('Final Separation (kpc)')
    axes[1, 0].set_title('Gravitational Effect')
    axes[1, 0].set_xscale('log')
    
    # Summary statistics
    optimal_dm = max(dm_fracs)
    optimal_idx = dm_fracs.index(optimal_dm)
    optimal_params = list(results.values())[optimal_idx]
    
    axes[1, 1].text(0.1, 0.8, f"Parameter Study Results:", fontsize=12, weight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Max dark matter: {optimal_dm:.1%}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"Optimal λ_c: {optimal_params['lambda_c_kpc']:.1f} kpc", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"Optimal q_floor: {optimal_params['quantum_floor']:.1%}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f"Parameter ranges tested:", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f"  λ_c: {min(lc_vals):.1f}-{max(lc_vals):.1f} kpc", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, f"  q_floor: {min(qf_vals):.1%}-{max(qf_vals):.1%}", transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_study_results.png', dpi=300)
    plt.show()
    
    print(f"\nParameter study completed:")
    print(f"  Optimal dark matter fraction: {optimal_dm:.1%}")
    print(f"  Best parameters: λ_c={optimal_params['lambda_c_kpc']:.1f} kpc, q_floor={optimal_params['quantum_floor']:.1%}")

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run Infodynamic Gravity Experiments')
    parser.add_argument('--test', choices=['basic', 'sec', 'galaxy', 'validation', 'parameters', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--output-dir', type=str, default='results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=== Infodynamic Gravity Experiment Suite ===")
    print(f"Output directory: {output_dir}")
    print(f"Log level: {args.log_level}")
    print()
    
    # Import numpy here to avoid issues with argument parsing
    import numpy as np
    
    # Run selected tests
    if args.test in ['basic', 'all']:
        run_basic_test()
        print()
    
    if args.test in ['sec', 'all']:
        run_sec_test()
        print()
    
    if args.test in ['galaxy', 'all']:
        run_galaxy_test()
        print()
    
    if args.test in ['validation', 'all']:
        summary = run_full_validation()
        
        # Save validation results
        import json
        with open(output_dir / 'validation_summary.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_summary = {}
            for key, value in summary.items():
                if isinstance(value, dict):
                    json_summary[key] = {}
                    for k2, v2 in value.items():
                        if isinstance(v2, np.ndarray):
                            json_summary[key][k2] = v2.tolist()
                        elif isinstance(v2, dict):
                            json_summary[key][k2] = {k3: v3.tolist() if isinstance(v3, np.ndarray) else v3 
                                                    for k3, v3 in v2.items()}
                        else:
                            json_summary[key][k2] = v2
                else:
                    json_summary[key] = value
            
            json.dump(json_summary, f, indent=2)
        
        print(f"Validation summary saved to {output_dir / 'validation_summary.json'}")
        print()
    
    if args.test in ['parameters', 'all']:
        run_parameter_study()
        print()
    
    print("=== Experiments Complete ===")

if __name__ == "__main__":
    main()
